"""pi0.5 end-to-end inference: raw robot inputs -> normalized, discrete-state prompt -> flow matching (low level only,
the upstream openpi path) or high level + low level (../hier) -> quantile inverse -> absolute targets for the
18 / 19-dim mobile manipulator; per-stage timing; the mobile-manipulator observation adapter.

Re-implementation (PyTorch / NumPy). Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/policies/policy.py (Policy.infer L67-L106), src/openpi/training/config.py (L126-L138 PI05 chain,
           L187 quantile, L630-L642 pi05_droid, L865-L894 pi05_full_droid_finetune), src/openpi/policies/droid_policy.py
           L47-L52, src/openpi/transforms.py (Unnormalize L175-L181, AbsoluteActions L226-L245)
  paper    pi0.5 arXiv:2504.16054v1 Sec. IV-E (robot system: 4 cameras, 18-19 DoF, 50 Hz PD tracking), Appendix E (H = 50);
           Hi Robot arXiv:2502.19417v2 Appendix B.3 (latency)
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Nothing new is computed here: ../data builds the sequence, ../hier decodes and samples, pi.fast.tokenizer undoes the
quantile map, pi.pi0.data undoes the delta. This file wires, converts and times. No training code.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch

from pi.fast.tokenizer.tokenizer import QuantileStats, normalize_quantile, unnormalize_quantile
from pi.pi0.data.data import to_absolute_actions
from pi.pi0.flow_matching.model import sample_actions as euler_sample
from pi.pi0.vlm.model import make_attn_mask
from pi.pi05.data.data import ACTION_DIM, ACTION_HORIZON, HL_IMAGE_KEYS, LL_IMAGE_KEYS, Pi05SequenceTokenizer, build_pi05_batch
from pi.pi05.hier.model import HL_PERIOD_S, MAX_NEW_TOKENS, HierarchicalPolicy, Pi05

CONTROL_HZ = 50  # paper Sec. IV-E: targets "at 50 Hz (with action chunking)"
# The mobile manipulators of Sec. IV-E: two 6-DoF arms with parallel-jaw grippers, a holonomic base (2D linear + 1D
# angular velocity), a torso lift (1D or 2D) -> 18 or 19 dims. The ORDER is this repo's reading of the text (README Sec. 8).
MOBILE_DIMS_18 = ("left_joint",) * 6 + ("left_gripper",) + ("right_joint",) * 6 + ("right_gripper",) + ("base_vx", "base_vy", "base_w") + ("lift_z",)
MOBILE_DIMS_19 = MOBILE_DIMS_18 + ("lift_x",)
MOBILE_CAMERAS = {"front": "base_0_rgb", "rear": "base_1_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}


def mobile_delta_mask(dims: tuple[str, ...]) -> tuple[bool, ...]:
    """Joints and lift positions are delta (target relative to the current state), grippers absolute (pi0 convention),
    base VELOCITIES absolute (a velocity is not a position increment). Base / lift treatment is inferred (README Sec. 8)."""
    return tuple(("joint" in d) or d.startswith("lift") for d in dims)


@dataclasses.dataclass
class Pi05RobotSpec:
    """What changes from robot to robot while the weights stay the same."""

    norm_stats: dict[str, QuantileStats]  # q01 / q99 of this robot's training data (config.py L187)
    delta_mask: tuple[bool, ...] | None
    native_dim: int  # 18 / 19 mobile, 8 DROID, 7 LIBERO
    action_horizon: int = ACTION_HORIZON  # 50 (pi0_config.py L26); pi05_droid 15 (config.py L631), full DROID finetune 16 (L869)
    control_hz: float = CONTROL_HZ
    discrete_state: bool = True  # pi05_libero: False (config.py L745)

    @property
    def chunk_seconds(self) -> float:
        return self.action_horizon / self.control_hz


def mobile_spec(n_dims: int = 19, norm_stats: dict[str, QuantileStats] | None = None) -> Pi05RobotSpec:
    dims = MOBILE_DIMS_19 if n_dims == 19 else MOBILE_DIMS_18
    if norm_stats is None:  # identity map, for mains and tests only
        norm_stats = {"state": QuantileStats(np.full(n_dims, -1.0), np.full(n_dims, 1.0)), "actions": QuantileStats(np.full(n_dims, -1.0), np.full(n_dims, 1.0))}
    return Pi05RobotSpec(norm_stats, mobile_delta_mask(dims), n_dims)


def to_executable_actions(x_0: torch.Tensor, state_raw: np.ndarray, robot: Pi05RobotSpec) -> np.ndarray:
    """x_0 f32[B, H, 32] normalized / delta / padded, state_raw f32[B, d] physical units -> f32[B, H, native_dim].
    Order = the DataConfig output chain (config.py, transforms.py): quantile Unnormalize (L175-L181) -> AbsoluteActions
    (L226-L245, delta dims += the state the model was given) -> keep native dims."""
    a = unnormalize_quantile(x_0.detach().cpu().numpy(), robot.norm_stats["actions"])
    a = to_absolute_actions(state_raw.astype(np.float32), a, robot.delta_mask)
    return a[..., : robot.native_dim].astype(np.float32)


def mobile_obs_to_raw(obs: dict, prompt: str, *, hl: bool) -> dict:
    """{"front", "rear", "left_wrist", "right_wrist": uint8[h, w, 3], "state": f32[18|19]} -> raw with batch 1.
    hl=True keeps all four cameras (HL_IMAGE_KEYS), hl=False drops the rear camera (LL_IMAGE_KEYS). Paper Sec. IV-E.
    This robot interface is the repo's stand-in: upstream ships no mobile-manipulator policy file (README Sec. 8)."""
    keys = HL_IMAGE_KEYS if hl else LL_IMAGE_KEYS
    images = {slot: np.asarray(obs[name], np.uint8)[None] for name, slot in MOBILE_CAMERAS.items() if slot in keys and name in obs}
    return {"images": images, "state": np.asarray(obs["state"], np.float32)[None], "prompt": [prompt]}


# ======================================================================================
# 1. Low level only: the upstream openpi pi0.5 path (config.py L126-L138). policy.py L67-L106 for the stages.
# ======================================================================================
class Pi05Policy:
    def __init__(self, model: Pi05, seq: Pi05SequenceTokenizer, robot: Pi05RobotSpec, *, num_steps: int = 10, image_keys=LL_IMAGE_KEYS):
        self.model, self.seq, self.robot, self.num_steps, self.image_keys = model.eval(), seq, robot, num_steps, tuple(image_keys)

    @torch.no_grad()
    def infer(self, raw: dict, noise: torch.Tensor | None = None) -> dict:
        """raw = {"images": {slot: uint8[B, h, w, 3]}, "state": f32[B, d], "prompt": [str] * B}
        -> {"actions": f32[B, H, native_dim] absolute targets, "x_0": f32[B, H, 32], "timing": {stage: ms}}."""
        timing, t0 = {}, time.perf_counter()

        def lap(name):
            nonlocal t0
            t1 = time.perf_counter()
            timing[name] = (t1 - t0) * 1e3
            t0 = t1

        obs, _ = build_pi05_batch(raw, self.robot.norm_stats, self.seq, layout="flow", image_keys=self.image_keys, action_horizon=self.robot.action_horizon,
                                  delta_mask=self.robot.delta_mask, train=False, discrete_state=self.robot.discrete_state)
        lap("data preprocessing")
        m = self.model
        img_tokens = {k: m.img(obs.images[k]) for k in self.image_keys}
        lap("image encoders")
        embs = [img_tokens[k] for k in self.image_keys] + [m.embedder.encode(obs.tokenized_prompt)]
        masks = [obs.image_masks[k][:, None].expand(-1, 256) for k in self.image_keys] + [obs.tokenized_prompt_mask]
        ars = [torch.zeros(b.shape[:2], dtype=torch.long) for b in embs[:-1]] + [obs.token_ar_mask.long()]
        emb, prefix_mask, ar = torch.cat(embs, 1), torch.cat(masks, 1), torch.cat(ars, 1)
        (_, _), kv = m.llm([emb, None], prefix_mask.long().cumsum(1) - 1, make_attn_mask(prefix_mask, ar))
        lap("observation forward pass")
        b = obs.state.shape[0]
        if noise is None:
            noise = torch.randn(b, self.robot.action_horizon, m.proj.action_dim)
        x_0 = euler_sample(m.make_velocity_fn(kv, prefix_mask), noise, self.num_steps)
        lap(f"x{self.num_steps} action forward pass (flow)")
        actions = to_executable_actions(x_0, np.asarray(raw["state"], np.float32), self.robot)
        lap("inverse transforms")
        timing["total"] = sum(timing.values())
        return {"actions": actions, "x_0": x_0, "timing": timing}


# ======================================================================================
# 2. Two levels: ../hier's policy with the robot's normalization in front and the inverse transforms behind.
# ======================================================================================
class Pi05HierPolicy(HierarchicalPolicy):
    def __init__(self, model: Pi05, seq: Pi05SequenceTokenizer, robot: Pi05RobotSpec, *, hl_period_s: float = HL_PERIOD_S,
                 num_steps: int = 10, max_new_tokens: int = MAX_NEW_TOKENS):
        super().__init__(model, seq, hl_period_s=hl_period_s, num_steps=num_steps, max_new_tokens=max_new_tokens, action_horizon=robot.action_horizon)
        self.robot = robot

    def _normalized(self, raw: dict) -> dict:
        return {**raw, "state": normalize_quantile(np.asarray(raw["state"], np.float32), self.robot.norm_stats["state"])}

    def step(self, raw_hl: dict, raw_ll: dict, t_now: float, user_message: str | None = None, noise=None, generator=None) -> dict:
        """Same as ../hier step, with raw states in physical units; adds "actions_exec" f32[1, H, native_dim]."""
        out = super().step(self._normalized(raw_hl), self._normalized(raw_ll), t_now, user_message, noise, generator)
        out["actions_exec"] = to_executable_actions(out["actions"], np.asarray(raw_ll["state"], np.float32), self.robot)
        return out


def tiny_pi05_policy(n_dims: int = 19, seed: int = 0, hier: bool = False, **kw):
    """Tiny model + byte codec + identity quantile stats + the mobile spec; for mains and tests."""
    from pi.pi05.data.data import tiny_pi05_tokenizer
    from pi.pi05.hier.model import tiny_pi05

    torch.manual_seed(seed)
    model, seq, robot = tiny_pi05(), tiny_pi05_tokenizer(10, 7), mobile_spec(n_dims)
    return (Pi05HierPolicy(model, seq, robot, **kw) if hier else Pi05Policy(model, seq, robot, **kw)), robot


# ======================================================================================
# 3. One end-to-end call with the tiny config.  uv run python -m pi.pi05.infer.model
# ======================================================================================
def main() -> None:
    rng = np.random.default_rng(0)
    policy, robot = tiny_pi05_policy(19)
    obs = {name: rng.integers(0, 256, (1, 96, 128, 3), dtype=np.uint8)[0] for name in MOBILE_CAMERAS}
    obs["state"] = rng.uniform(-0.5, 0.5, 19).astype(np.float32)
    raw_ll = mobile_obs_to_raw(obs, "pick up the plate", hl=False)
    raw_hl = mobile_obs_to_raw(obs, "clean the kitchen", hl=True)
    print(f"robot: {robot.native_dim} dims = {len(MOBILE_DIMS_19)} names, delta_mask {''.join('1' if d else '0' for d in robot.delta_mask)} (joints / lift delta; grippers, base velocities absolute)")
    print(f"       H {robot.action_horizon} @ {robot.control_hz:.0f} Hz = {robot.chunk_seconds:.1f} s per chunk; quantile norm_stats q01/q99")
    print(f"[raw LL] cameras {list(raw_ll['images'])} state {raw_ll['state'].shape} prompt {raw_ll['prompt']}")
    print(f"[raw HL] cameras {list(raw_hl['images'])} state {raw_hl['state'].shape} prompt {raw_hl['prompt']}")

    noise = torch.randn(1, robot.action_horizon, ACTION_DIM)
    out = policy.infer(raw_ll, noise=noise)
    print(f"\n[low level] x_0 {tuple(out['x_0'].shape)} normalized  ->  actions {tuple(out['actions'].shape)} = [1, 50, 19] absolute targets")
    print(f"            row 0: joints {out['actions'][0, 0, :3]} ... base v {out['actions'][0, 0, 14:17]} lift {out['actions'][0, 0, 17:]}")
    print("[timing]    tiny config on this CPU (Hi Robot App. B.3, RTX 4090, pi0: 14 + 32 + 27 = 73 ms):")
    for k, v in out["timing"].items():
        print(f"            {k:32s} {v:8.1f} ms")

    hier, _ = tiny_pi05_policy(19, hier=True, max_new_tokens=8)
    o = hier.step(raw_hl, {"images": raw_ll["images"], "state": raw_ll["state"]}, 0.0)
    print(f"\n[two-level] hl_ran {o['hl_ran']} ({o['hl_tokens']} tokens) subtask {o['subtask']!r} -> actions_exec {tuple(o['actions_exec'].shape)}; timing "
          + ", ".join(f"{k}: {v:.0f} ms" for k, v in o["timing"].items()))
    print("            execute the first k rows at 50 Hz (k undisclosed, README Sec. 8), observe, repeat; high level again after 1 s")


if __name__ == "__main__":
    main()
