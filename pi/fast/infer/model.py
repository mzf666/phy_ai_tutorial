"""pi0-FAST end-to-end inference: raw robot inputs -> prefix sequence -> prefill + token-by-token decoding -> FAST ids
-> action chunk -> executable absolute actions, with per-stage timing; the DROID input adapter.

Re-implementation (PyTorch / NumPy). Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/policies/policy.py (Policy.infer L67-L106), src/openpi/policies/droid_policy.py (L10-L18, L30-L81),
           src/openpi/transforms.py (ExtractFASTActions L291-L306, quantile Unnormalize L175-L181, AbsoluteActions L226-L245),
           src/openpi/training/config.py (output transform order L150-L159; pi0_fast_droid L615-L625)
  paper    FAST, arXiv:2501.09747v1, Sec. VI-E (latency), Appendix C (decoding), Appendix D (DROID setup)
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Nothing new is computed here: ../data builds the sequence, ../model decodes, ../data cuts the ids out, ../tokenizer
decodes the chunk, pi.pi0.data undoes the delta. This file wires them and times them. No training code.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from pi.fast.data.data import ACTION_DIM, FASTSequenceTokenizer, build_fast_batch
from pi.fast.model.model import MAX_DECODING_STEPS, Pi0FAST
from pi.fast.tokenizer.tokenizer import QuantileStats, unnormalize_quantile
from pi.pi0.data.data import to_absolute_actions

# DROID (droid_policy.py L10-L18, L36-L40, L81; paper Appendix D)
DROID_STATE_DIM = 8  # 7 joint positions + 1 gripper position
DROID_ACTION_DIM = 8  # 7 joint velocities + 1 absolute gripper (paper); DroidOutputs keeps the first 8 dims (L81)
DROID_ACTION_HORIZON_PAPER = 15  # Appendix D "15-step action chunks"
DROID_EXECUTE_STEPS = (8, 15)  # Appendix D "we execute 8 or 15-step chunks open-loop"
DROID_ACTION_HORIZON_OPENPI = 10  # config.py L617 pi0_fast_droid (16 for pi0_fast_full_droid_finetune, L838)
DROID_IMAGE_KEYS = {"observation/exterior_image_1_left": "base_0_rgb", "observation/wrist_image_left": "wrist_0_rgb"}  # L53-L55
IMAGE_SLOT_ALIASES = {"left_wrist_0_rgb": "wrist_0_rgb"}  # pi0-named raw (e.g. pi.pi0.infer.eval.libero_obs_to_raw) -> FAST slots


@dataclasses.dataclass
class FASTRobotSpec:
    """What changes from robot to robot while the weights stay the same (../data README Sec. 9)."""

    norm_stats: dict[str, QuantileStats]  # per-dim q01 / q99 of this robot's training data (config.py L187: quantile for FAST)
    delta_mask: tuple[bool, ...] | None  # which dims are delta (joints) vs absolute (grippers)
    native_dim: int  # rows of the output to keep: 8 for DROID, 7 for LIBERO
    action_horizon: int  # H the tokens were fit with: FAST decode needs it (transforms.py L293-L295)


def fast_image_slots(raw: dict) -> dict:
    """Rename pi0 image slots to FAST ones (left_wrist_0_rgb -> wrist_0_rgb); other keys untouched."""
    images = {IMAGE_SLOT_ALIASES.get(k, k): v for k, v in raw["images"].items()}
    return {**raw, "images": images}


def to_executable_actions(tokens: np.ndarray, state_raw: np.ndarray, robot: FASTRobotSpec, seq: FASTSequenceTokenizer) -> np.ndarray:
    """tokens i64[B, T] (generated ids), state_raw f32[B, d] in physical units -> f32[B, H, native_dim].
    Order = the reverse of config.py L150-L159: ExtractFASTActions (transforms.py L291-L306, zeros if no marker)
    -> quantile Unnormalize (L175-L181) -> AbsoluteActions (L226-L245, delta dims += raw state) -> keep native dims."""
    chunks = np.stack([seq.extract_actions(np.asarray(t), robot.action_horizon, robot.native_dim) for t in tokens])
    chunks = unnormalize_quantile(chunks, robot.norm_stats["actions"])
    chunks = to_absolute_actions(state_raw.astype(np.float32), chunks, robot.delta_mask)
    return chunks[..., : robot.native_dim].astype(np.float32)


class Pi0FASTPolicy:
    def __init__(self, model: Pi0FAST, seq: FASTSequenceTokenizer, robot: FASTRobotSpec, *,
                 max_decoding_steps: int = MAX_DECODING_STEPS, temperature: float = 0.0):
        # temperature: 0 = greedy (paper default); 0.7 for the bimanual tasks (paper Appendix C)
        self.model, self.seq, self.robot = model.eval(), seq, robot
        self.max_decoding_steps, self.temperature = max_decoding_steps, temperature

    @torch.no_grad()
    def infer(self, raw: dict, generator: torch.Generator | None = None) -> dict:
        """raw = {"images": {slot: uint8[B, h, w, 3]}, "state": f32[B, d], "prompt": [str] * B}
        -> {"actions": f32[B, H, native_dim], "tokens": i64[B, max_decoding_steps], "n_steps": int, "timing": {stage: ms}}.
        Upstream times only sample_actions (policy.py L91-L96, "infer_ms"); the two data stages are added here."""
        raw = fast_image_slots(raw)
        timing, t0 = {}, time.perf_counter()

        def lap(name):
            nonlocal t0
            t1 = time.perf_counter()
            timing[name] = (t1 - t0) * 1e3
            t0 = t1

        obs, _ = build_fast_batch(raw, self.robot.norm_stats, self.seq, action_horizon=self.robot.action_horizon, action_dim=ACTION_DIM,
                                  delta_mask=self.robot.delta_mask, train=False)  # inference mode: prefix only
        lap("data preprocessing")
        tokens, n_steps = self.model.sample_actions(obs, max_decoding_steps=self.max_decoding_steps, temperature=self.temperature, generator=generator)
        lap("sample_actions (prefill + decode steps)")
        actions = to_executable_actions(tokens.cpu().numpy(), np.asarray(raw["state"], np.float32), self.robot, self.seq)
        lap("extract + inverse transforms")
        timing["total"] = sum(timing.values())
        timing["ms per decode step"] = timing["sample_actions (prefill + decode steps)"] / max(n_steps, 1)
        return {"actions": actions, "tokens": tokens, "n_steps": n_steps, "timing": timing}


def droid_obs_to_raw(obs: dict, prompt: str) -> dict:
    """One DROID observation (droid_policy.py L35-L74) -> raw with batch size 1. base_1_rgb is left out: build_fast_batch
    fills a black image with mask True (L53-L56). Scalar gripper position is promoted to [1] (L36-L40)."""
    gripper = np.asarray(obs["observation/gripper_position"], np.float32).reshape(-1)
    state = np.concatenate([np.asarray(obs["observation/joint_position"], np.float32), gripper])
    assert state.shape == (DROID_STATE_DIM,), state.shape
    images = {slot: np.asarray(obs[key], np.uint8)[None] for key, slot in DROID_IMAGE_KEYS.items()}
    return {"images": images, "state": state[None], "prompt": [prompt]}


def tiny_fast_policy(horizon: int = DROID_ACTION_HORIZON_PAPER, native_dim: int = DROID_ACTION_DIM, seed: int = 0, **policy_kwargs):
    """Tiny model + a FAST tokenizer fit on synthetic chunks + DROID-shaped robot spec; for main() and tests."""
    from pi.fast.data.data import ByteTextCodec, tiny_fast_tokenizer
    from pi.fast.model.model import tiny
    from pi.pi0.data.data import make_bool_mask

    torch.manual_seed(seed)
    model = Pi0FAST(*tiny())
    seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(horizon, native_dim, seed=seed), max_len=180)
    stats = {"state": QuantileStats(np.full(native_dim, -1.0), np.full(native_dim, 1.0)), "actions": QuantileStats(np.full(native_dim, -1.0), np.full(native_dim, 1.0))}
    robot = FASTRobotSpec(stats, make_bool_mask(native_dim - 1, -1), native_dim, horizon)
    return Pi0FASTPolicy(model, seq, robot, **policy_kwargs), seq


# ======================================================================================
#   uv run python -m pi.fast.infer.model
# ======================================================================================
def main() -> None:
    from pi.fast.data.data import discretize_state, paligemma_to_fast
    from pi.fast.tokenizer.tokenizer import normalize_quantile

    rng = np.random.default_rng(0)
    policy, seq = tiny_fast_policy(max_decoding_steps=48)
    robot = policy.robot
    obs = {"observation/exterior_image_1_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
           "observation/wrist_image_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
           "observation/joint_position": rng.uniform(-0.5, 0.5, 7).astype(np.float32),
           "observation/gripper_position": np.float32(0.3)}
    raw = droid_obs_to_raw(obs, "put the marker in the cup")
    print(f"DROID obs keys: {list(obs)}")
    print(f"raw: images {[(k, tuple(v.shape)) for k, v in raw['images'].items()]} (base_1_rgb: black, mask True)  state {tuple(raw['state'].shape)}  prompt {raw['prompt']}")
    print(f"robot: native_dim {robot.native_dim}, action_horizon {robot.action_horizon} (paper App. D; openpi pi0_fast_droid {DROID_ACTION_HORIZON_OPENPI}), delta_mask {robot.delta_mask}")
    st = normalize_quantile(raw["state"][0], robot.norm_stats["state"])
    print(f"state -> quantile norm {np.round(st, 3).tolist()} -> bins {discretize_state(st).tolist()} (text in the prefix)")

    out = policy.infer(raw)
    tokens = out["tokens"][0].numpy()
    print(f"\n[infer]  tokens {tuple(out['tokens'].shape)} i64, ran {out['n_steps']} decode steps (cap {policy.max_decoding_steps}); first ids {tokens[:6].tolist()}")
    marker = seq._action_marker
    found = any(tokens[i : i + len(marker)].tolist() == marker for i in range(len(tokens) - len(marker)))
    print(f"[extract] 'Action: ' marker found in the output? {found} -> {'FAST ids -> decode' if found else 'zeros fallback (random weights)'}")
    print(f"[actions] {tuple(out['actions'].shape)} = [B, H, native_dim]; row 0: {np.round(out['actions'][0, 0], 3).tolist()}")
    print("[timing] " + "  ".join(f"{k}: {v:.1f} ms" for k, v in out["timing"].items()))

    # the same path on a sequence that DOES contain action tokens: what a trained model would emit
    chunk = np.clip(0.3 * rng.standard_normal((robot.action_horizon, robot.native_dim)), -0.9, 0.9).astype(np.float32)
    toks, m, ar, _ = seq.tokenize("put the marker in the cup", st, chunk)
    gen = toks[m & (ar == 1)]  # the postfix a trained model would generate
    acts = to_executable_actions(gen[None], raw["state"], robot, seq)
    ref = to_absolute_actions(raw["state"], unnormalize_quantile(chunk[None], robot.norm_stats["actions"]), robot.delta_mask)
    print(f"\n[oracle] postfix of {len(gen)} ids ({len(paligemma_to_fast(gen[len(marker):-2]))} FAST ids) -> actions {tuple(acts.shape)}, "
          f"max |err| vs analytic inverse {np.abs(acts - ref).max():.4f} (<= 0.5/gamma * sqrt(H) * (q99-q01)/2 = {0.05 * np.sqrt(robot.action_horizon):.3f})")
    print(f"\nreal model: ~750 ms per chunk on an RTX 4090 vs pi0 ~100 ms (paper Sec. VI-E); execute {DROID_EXECUTE_STEPS} of {DROID_ACTION_HORIZON_PAPER} steps open-loop (App. D)")


if __name__ == "__main__":
    main()
