"""Parameter-count and end-to-end shape checks for the assembled pi0 model and policy.

References: openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 (src/openpi/models/pi0.py, src/openpi/policies/policy.py),
pi0 paper arXiv:2410.24164v1 Sec. III ("3.3 billion parameters"). Run on CPU: `uv run pytest pi/pi0/infer -q`.
"""

import numpy as np
import torch

from pi.pi0.action_expert import model as A
from pi.pi0.data.data import ByteEncoder, NormStats, PromptTokenizer, build_batch, make_bool_mask
from pi.pi0.flow_matching import model as FM
from pi.pi0.infer import model as M
from pi.pi0.vlm import model as V

B = 2


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- parameter table (README Sec. 3.2)
def test_paper_pi0_parameter_table():
    with torch.device("meta"):
        m = M.Pi0()
    parts = {
        "siglip": n_params(m.img),
        "embedding": n_params(m.embedder),
        "expert0": sum(n_params(l.experts[0]) for l in m.llm.layers) + n_params(m.llm.final_norms[0]),
        "expert1": sum(n_params(l.experts[1]) for l in m.llm.layers) + n_params(m.llm.final_norms[1]),
        "proj": n_params(m.proj),
    }
    assert parts == {"siglip": 414_803_696, "embedding": 526_647_296, "expert0": 1_981_884_416, "expert1": 311_464_960, "proj": 3_248_160}
    assert n_params(m) == sum(parts.values()) == 3_238_048_528  # paper: "a total of 3.3 billion parameters"


# ---------------------------------------------------------------- assembly changes nothing
def _obs(vocab):
    images = {k: torch.rand(B, 224, 224, 3) * 2 - 1 for k in V.IMAGE_KEYS}
    image_masks = {k: torch.ones(B, dtype=torch.bool) for k in V.IMAGE_KEYS}
    image_masks["right_wrist_0_rgb"] = torch.zeros(B, dtype=torch.bool)
    tokens = torch.randint(3, vocab, (B, 48))
    token_mask = torch.zeros(B, 48, dtype=torch.bool)
    token_mask[:, :7] = True
    return M.Observation(images, image_masks, torch.randn(B, 32), tokens, token_mask)


def test_pi0_sample_actions_equals_manual_chain():
    torch.manual_seed(0)
    m = M.tiny_pi0().eval()
    obs = _obs(m.embedder.input_embedding.shape[0])
    noise = torch.randn(B, 50, 32)
    with torch.no_grad():
        x_pi0 = m.sample_actions(obs, noise)
        # the same thing spelled out with the pieces from ../vlm, ../action_expert, ../flow_matching
        emb, mask, ar = m.embed_prefix(obs)
        (_, none), kv = m.llm([emb, None], mask.long().cumsum(1) - 1, V.make_attn_mask(mask, ar))
        assert none is None
        v = FM.make_velocity_fn(m.llm, m.proj, kv, mask, obs.state)
        x_manual = FM.sample_actions(v, noise, 10)
    torch.testing.assert_close(x_pi0, x_manual)
    assert x_pi0.shape == (B, 50, 32)


def test_embed_prefix_matches_vlm_paligemma():
    """Pi0.embed_prefix must be ../vlm's PaliGemma.embed_prefix (same weights -> same tensor)."""
    torch.manual_seed(0)
    m = M.tiny_pi0().eval()
    pg = V.PaliGemma(V.tiny_vit(), A.tiny_gemma()).eval()
    pg.img.load_state_dict(m.img.state_dict())
    pg.llm.embedder.load_state_dict(m.embedder.state_dict())
    obs = _obs(m.embedder.input_embedding.shape[0])
    with torch.no_grad():
        e1, m1, a1 = m.embed_prefix(obs)
        e2, m2, a2 = pg.embed_prefix(obs.images, obs.image_masks, obs.tokenized_prompt, obs.tokenized_prompt_mask)
    torch.testing.assert_close(e1, e2)
    assert torch.equal(m1, m2) and torch.equal(a1, a2)


# ---------------------------------------------------------------- policy end to end
def _robot(d, *dims):
    return M.RobotSpec(
        norm_stats={"state": NormStats(mean=np.zeros(d, np.float32), std=np.ones(d, np.float32)),
                    "actions": NormStats(mean=np.zeros(d, np.float32), std=np.ones(d, np.float32))},
        delta_mask=make_bool_mask(*dims),
        native_dim=d,
    )


def _raw(rng, d, cams):
    return {"images": {k: rng.integers(0, 256, (B, 200, 300, 3), dtype=np.uint8) for k in cams},
            "state": rng.standard_normal((B, d)).astype(np.float32),
            "prompt": ["a", "b"]}


def test_policy_infer_two_robots():
    """A 7-dim robot with one wrist camera and a 14-dim bimanual robot with two, same weights, different RobotSpec."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = M.Pi0Policy(M.tiny_pi0(), PromptTokenizer(ByteEncoder()), _robot(7, 6, -1))
    out = policy.infer(_raw(rng, 7, ["base_0_rgb", "left_wrist_0_rgb"]))
    assert out["actions"].shape == (B, 50, 7) and np.isfinite(out["actions"]).all()
    assert set(out["timing"]) == {"data preprocessing", "image encoders", "observation forward pass", "x10 action forward pass (flow)", "inverse transforms", "total"}
    policy.robot = _robot(14, 6, -1, 6, -1)
    out = policy.infer(_raw(rng, 14, ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]))
    assert out["actions"].shape == (B, 50, 14)


def test_policy_is_deterministic_given_noise():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    policy = M.Pi0Policy(M.tiny_pi0(), PromptTokenizer(ByteEncoder()), _robot(7, 6, -1))
    raw = _raw(rng, 7, ["base_0_rgb"])
    noise = torch.randn(B, 50, 32)
    a1 = policy.infer(raw, noise=noise)["actions"]
    a2 = policy.infer(raw, noise=noise)["actions"]
    np.testing.assert_allclose(a1, a2, atol=1e-6)
    a3 = policy.infer(raw, noise=torch.randn(B, 50, 32))["actions"]
    assert not np.allclose(a1, a3)


def test_policy_gripper_dim_is_absolute_and_joints_are_delta():
    """With identity norm stats, the gripper column must not have the state added; joint columns must."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    m = M.tiny_pi0()
    policy = M.Pi0Policy(m, PromptTokenizer(ByteEncoder()), _robot(7, 6, -1))
    raw = _raw(rng, 7, ["base_0_rgb"])
    noise = torch.randn(B, 50, 32)
    out = policy.infer(raw, noise=noise)["actions"]
    obs, _ = build_batch(raw, policy.robot.norm_stats, policy.tokenizer, delta_mask=policy.robot.delta_mask, train=False)
    with torch.no_grad():
        x_0 = m.sample_actions(obs, noise).numpy()
    np.testing.assert_allclose(out[..., 6], x_0[..., 6], atol=1e-5)  # gripper: absolute, untouched
    np.testing.assert_allclose(out[..., :6], x_0[..., :6] + raw["state"][:, None, :6], atol=1e-5)  # joints: + q_t
