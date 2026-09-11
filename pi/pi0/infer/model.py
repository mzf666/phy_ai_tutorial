"""pi0 end-to-end inference: the assembled model and the policy wrapper from raw robot inputs to executable actions.

Minimal PyTorch re-implementation of openpi's JAX code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/models/pi0.py (Pi0.__init__ L66-L103, embed_prefix L106-L137, sample_actions L216-L279)
          src/openpi/policies/policy.py (Policy.infer L67-L106), src/openpi/policies/policy_config.py (L18-L95)
  paper   pi0 arXiv:2410.24164v1 Sec. III, Appendix D, Table I
Upstream license: Apache-2.0. This file re-implements, it does not copy.

Nothing new is computed here. Pi0 owns the weights of ../vlm (SigLIP, Embedder), ../action_expert (MoEGemma,
ActionProjections) and calls ../flow_matching's sampler; Pi0Policy adds ../data's preprocessing in front and the
inverse transforms behind, and times every stage. No training code (repo rule): ../flow_matching/train.py and
../train use the same Pi0 weights.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch
import torch.nn as nn

from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, PI0_EXPERTS, ActionProjections, MoEGemma, tiny_experts
from pi.pi0.data.data import IMAGE_KEYS, NormStats, Observation, PromptTokenizer, build_batch
from pi.pi0.flow_matching.model import make_velocity_fn, sample_actions, to_executable_actions
from pi.pi0.vlm.model import SIGLIP_SO400M_14, Embedder, GemmaConfig, SigLIP, ViTConfig, make_attn_mask, tiny_vit


# ======================================================================================
# 1. The model. openpi@215abfb pi0.py L66-L103: PaliGemma.img, PaliGemma.llm (embedder + two-expert layers),
#    and the five projections hanging off Pi0 itself.
# ======================================================================================
class Pi0(nn.Module):
    def __init__(self, vit_cfg: ViTConfig = SIGLIP_SO400M_14, experts: tuple[GemmaConfig, GemmaConfig] = PI0_EXPERTS,
                 action_dim: int = ACTION_DIM, action_horizon: int = ACTION_HORIZON):
        super().__init__()
        assert vit_cfg.out_dim == experts[0].width
        self.img = SigLIP(vit_cfg)  # PaliGemma.img (L81-L90)
        self.embedder = Embedder(experts[0])  # PaliGemma.llm.embedder, expert 0's vocabulary (gemma.py L355-L359)
        self.llm = MoEGemma(experts)  # PaliGemma.llm layers + final_norms, both experts (L73-L80)
        self.proj = ActionProjections(experts[1], action_dim, action_horizon)  # L92-L100
        self.action_horizon = action_horizon

    def embed_prefix(self, obs: Observation):
        """images + prompt -> (emb f32[B, 816, 2048], input_mask bool[B, 816], ar_mask bool[816]). pi0.py L106-L137.
        Same as ../vlm PaliGemma.embed_prefix, using this object's SigLIP and embedding table."""
        embs, masks, ar = [], [], []
        for k in IMAGE_KEYS:
            t = self.img(obs.images[k])
            embs.append(t)
            masks.append(obs.image_masks[k][:, None].expand(-1, t.shape[1]))
            ar.append(torch.zeros(t.shape[1], dtype=torch.bool, device=t.device))
        e = self.embedder.encode(obs.tokenized_prompt)
        embs.append(e)
        masks.append(obs.tokenized_prompt_mask)
        ar.append(torch.zeros(e.shape[1], dtype=torch.bool, device=e.device))
        return torch.cat(embs, 1), torch.cat(masks, 1), torch.cat(ar, 0)

    def prefix_cache(self, obs: Observation):
        """Run expert 0 once over the prefix; returns (kv_cache: 18 x (k, v) f32[B, 816, 1, 256], prefix_mask).
        pi0.py L233-L237. This is everything the VLM does per action chunk."""
        emb, mask, ar = self.embed_prefix(obs)
        (_, none), kv = self.llm([emb, None], mask.long().cumsum(1) - 1, make_attn_mask(mask, ar))
        assert none is None
        return kv, mask

    def sample_actions(self, obs: Observation, noise: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Observation + noise f32[B, 50, 32] -> normalized action chunk f32[B, 50, 32]. pi0.py L216-L279:
        prefix once, then num_steps Euler steps of the action expert against the cache (../flow_matching)."""
        kv, prefix_mask = self.prefix_cache(obs)
        v = make_velocity_fn(self.llm, self.proj, kv, prefix_mask, obs.state)
        return sample_actions(v, noise, num_steps)


def tiny_pi0() -> Pi0:
    return Pi0(tiny_vit(), tiny_experts())


# ======================================================================================
# 2. The policy: raw robot inputs -> executable actions, with per-stage timing.
#    openpi@215abfb policy.py L67-L106 and the transform chains of policy_config.py L77-L88.
# ======================================================================================
@dataclasses.dataclass
class RobotSpec:
    """What changes from robot to robot while the weights stay the same (../data README Sec. 9)."""

    norm_stats: dict[str, NormStats]  # per-dim mean/std of this robot's training data (checkpoint's norm_stats.json)
    delta_mask: tuple[bool, ...] | None  # which dims are delta (joints) vs absolute (grippers)
    native_dim: int  # rows of the output to keep: 7 for LIBERO, 14 for ALOHA


class Pi0Policy:
    def __init__(self, model: Pi0, tokenizer: PromptTokenizer, robot: RobotSpec, num_steps: int = 10):
        self.model, self.tokenizer, self.robot, self.num_steps = model.eval(), tokenizer, robot, num_steps

    @torch.no_grad()
    def infer(self, raw: dict, noise: torch.Tensor | None = None) -> dict:
        """raw = {"images": {slot: uint8[B, h, w, 3]}, "state": f32[B, d], "prompt": [str] * B}
        -> {"actions": f32[B, 50, d] absolute targets in physical units, "timing": {stage: ms}}.
        Stage names follow paper Table I where they exist; the two data stages are not in the paper's timing."""
        timing, t0 = {}, time.perf_counter()

        def lap(name):
            nonlocal t0
            t1 = time.perf_counter()
            timing[name] = (t1 - t0) * 1e3
            t0 = t1

        obs, _ = build_batch(raw, self.robot.norm_stats, self.tokenizer, delta_mask=self.robot.delta_mask, train=False)
        lap("data preprocessing")
        m = self.model
        # image encoders (Table I row 1) and observation forward pass (row 2), split for the timing table
        img_tokens = {k: m.img(obs.images[k]) for k in IMAGE_KEYS}
        lap("image encoders")
        embs = [img_tokens[k] for k in IMAGE_KEYS] + [m.embedder.encode(obs.tokenized_prompt)]
        masks = [obs.image_masks[k][:, None].expand(-1, 256) for k in IMAGE_KEYS] + [obs.tokenized_prompt_mask]
        emb, prefix_mask = torch.cat(embs, 1), torch.cat(masks, 1)
        ar = torch.zeros(emb.shape[1], dtype=torch.bool, device=emb.device)
        (_, _), kv = m.llm([emb, None], prefix_mask.long().cumsum(1) - 1, make_attn_mask(prefix_mask, ar))
        lap("observation forward pass")
        b = obs.state.shape[0]
        if noise is None:
            noise = torch.randn(b, m.action_horizon, m.proj.action_dim, device=obs.state.device)
        x_0 = sample_actions(make_velocity_fn(m.llm, m.proj, kv, prefix_mask, obs.state), noise, self.num_steps)
        lap(f"x{self.num_steps} action forward pass (flow)")
        actions = to_executable_actions(x_0, obs.state, self.robot.norm_stats, self.robot.delta_mask, self.robot.native_dim)
        lap("inverse transforms")
        timing["total"] = sum(timing.values())
        return {"actions": actions, "timing": timing}


# ======================================================================================
# 3. One end-to-end call with the tiny config: shapes at every stage, then the timing table.
#    uv run python -m pi.pi0.infer.model
# ======================================================================================
def main():
    from pi.pi0.data.data import ByteEncoder, make_bool_mask

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B, d = 2, 7  # a pretend 20 Hz robot: 6 joints + 1 gripper, one base camera + one wrist camera (like LIBERO)
    model = tiny_pi0()
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    print(f"tiny Pi0 params: img {n(model.img):,}  embedder {n(model.embedder):,}  llm {n(model.llm):,}  proj {n(model.proj):,}  total {n(model):,}")

    robot = RobotSpec(
        norm_stats={"state": NormStats(mean=np.zeros(d, np.float32), std=np.full(d, 0.5, np.float32)),
                    "actions": NormStats(mean=np.zeros(d, np.float32), std=np.full(d, 0.1, np.float32))},
        delta_mask=make_bool_mask(6, -1),
        native_dim=d,
    )
    policy = Pi0Policy(model, PromptTokenizer(ByteEncoder()), robot)
    raw = {
        "images": {"base_0_rgb": rng.integers(0, 256, (B, 480, 640, 3), dtype=np.uint8),
                   "left_wrist_0_rgb": rng.integers(0, 256, (B, 240, 320, 3), dtype=np.uint8)},
        "state": rng.standard_normal((B, d)).astype(np.float32),
        "prompt": ["put the cup in the sink", "close the drawer"],
    }
    print("[raw]     images", {k: tuple(v.shape) for k, v in raw["images"].items()}, " state", raw["state"].shape, " prompt", len(raw["prompt"]))

    # --- the same stages as Pi0Policy.infer, unrolled to show shapes ---
    with torch.no_grad():
        obs, _ = build_batch(raw, robot.norm_stats, policy.tokenizer, delta_mask=robot.delta_mask, train=False)
        print("[data]    Observation: images 3 x", tuple(obs.images["base_0_rgb"].shape), " masks",
              {k: bool(v[0]) for k, v in obs.image_masks.items()}, " state", tuple(obs.state.shape), " prompt", tuple(obs.tokenized_prompt.shape))
        kv, prefix_mask = model.prefix_cache(obs)
        print("[prefix]  kv_cache", len(kv), "x", tuple(kv[0][0].shape), f" valid prefix tokens {int(prefix_mask[0].sum())}/816 (2 cameras x 256 + prompt)")
        noise = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
        x_0 = model.sample_actions(obs, noise)
        print("[flow]    x_0", tuple(x_0.shape), " normalized / delta / padded")
        actions = to_executable_actions(x_0, obs.state, robot.norm_stats, robot.delta_mask, robot.native_dim)
        print("[post]    actions", tuple(actions.shape), " = [B, 50, 7] absolute joint targets + gripper; execute rows 0..15 at 20 Hz")

        # --- the public entry point gives the same tensor, and a timing table ---
        out = policy.infer(raw, noise=noise)
        assert np.allclose(out["actions"], actions, atol=1e-5)
    print("\n[timing]  tiny config on this CPU, float32 (paper Table I: RTX 4090, bf16, 3 cameras; not comparable):")
    for k, v in out["timing"].items():
        print(f"          {k:32s} {v:8.1f} ms")


if __name__ == "__main__":
    main()
