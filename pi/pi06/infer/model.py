"""pi0.6* end-to-end inference: raw static-bimanual inputs -> (at a lower rate) subtask decoding -> the flow sequence
with the Advantage token -> optional classifier-free guidance over the conditional / unconditional prefixes ->
5 Euler steps -> quantile inverse and delta -> 14-dim joint targets at 50 Hz; per-stage timing.

Re-implementation (PyTorch / NumPy). Sources of truth:
  paper    pi0.6* arXiv:2511.14759v2 Sec. IV-B (Eq. 2: beta = 1 <=> sample with I = True), Sec. V-A (subtask predicted first,
           at a lower frequency than actions; joints + grippers at 50 Hz), Sec. V-D (deployment: I_t = True), Appendix E
           (Eq. 12-13: CFG with beta > 1 from the conditional and unconditional models, beta in [1.5, 2.5] "where useful",
           high beta pushes actions to the support boundary), Fig. 5 (two 6-DoF arms with parallel-jaw grippers, joint
           position control at 50 Hz, base camera between the arms + one wrist camera per arm)
  card     pi0.6 model card Sec. 2: 5 denoising steps, 3 cameras, 63 ms per action chunk on one H100
  openpi   https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 policy.py L67-L106
           (the infer stages and timing), transforms.py L175-L181 / L226-L245 (unnormalize, absolute actions) via pi.pi05.infer
  Hi Robot arXiv:2502.19417v2 Sec. 4.1 (the 1-second high-level rate this repo reuses; pi0.6*'s own rate is undisclosed)
License: Apache-2.0 for the openpi pieces. Re-implements, does not copy. No training code here.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from pi.fast.data.data import EOS_ID
from pi.fast.tokenizer.tokenizer import QuantileStats
from pi.pi0.flow_matching.model import VelocityFn, sample_actions as euler_sample
from pi.pi05.hier.model import HL_PERIOD_S
from pi.pi05.infer.model import Pi05RobotSpec, to_executable_actions  # noqa: F401  (re-exported: unchanged inverse chain)
from pi.pi06.backbone.model import MAX_NEW_TOKENS, NUM_DENOISING_STEPS, AdaRMSNorm, Pi06
from pi.pi06.data.data import ACTION_DIM, STATIC_IMAGE_KEYS, Pi06SequenceTokenizer, build_pi06_batch

CONTROL_HZ = 50  # paper Sec. V-A "joint angles and gripper commands at 50 Hz"; Fig. 5 "controlled at 50 Hz with joint positions"
# Fig. 5: two 6-DoF arms with parallel-jaw grippers -> 2 x (6 joints + 1 gripper) = 14 dims. The order is this repo's (README Sec. 8).
STATIC_DIMS_14 = ("left_joint",) * 6 + ("left_gripper",) + ("right_joint",) * 6 + ("right_gripper",)
STATIC_CAMERAS = {"base": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}  # Fig. 5 -> ../data slots
LATENCY_H100_MS = 63.0  # card Sec. 2, 5 steps, 3 cameras, one H100 (stage breakdown undisclosed, README Sec. 8)
CFG_BETA_RANGE = (1.5, 2.5)  # App. E "moderate settings (e.g. beta in [1.5, 2.5]) where useful"; default deployment beta = 1 (Sec. V-D)


def static_delta_mask(dims: tuple[str, ...] = STATIC_DIMS_14) -> tuple[bool, ...]:
    """Joints delta, grippers absolute (pi0 convention, pi.pi0.data)."""
    return tuple("joint" in d for d in dims)


def static_spec(norm_stats: dict[str, QuantileStats] | None = None) -> Pi05RobotSpec:
    n = len(STATIC_DIMS_14)
    if norm_stats is None:  # identity map, mains / tests only
        norm_stats = {"state": QuantileStats(np.full(n, -1.0), np.full(n, 1.0)), "actions": QuantileStats(np.full(n, -1.0), np.full(n, 1.0))}
    return Pi05RobotSpec(norm_stats, static_delta_mask(), n, control_hz=CONTROL_HZ)


def static_obs_to_raw(obs: dict, prompt: str) -> dict:
    """{"base", "left_wrist", "right_wrist": uint8[h, w, 3], "state": f32[14]} -> raw with batch 1 (STATIC_IMAGE_KEYS slots)."""
    images = {slot: np.asarray(obs[name], np.uint8)[None] for name, slot in STATIC_CAMERAS.items() if name in obs}
    return {"images": images, "state": np.asarray(obs["state"], np.float32)[None], "prompt": [prompt]}


# ======================================================================================
# 1. Classifier-free guidance on the velocity field. App. E Eq. 13: follow grad log pi(a|o) + beta (grad log pi(a|I,o)
#    - grad log pi(a|o)); the flow model "effectively learns the gradient of the likelihoods", so the same combination
#    is applied to the two velocity fields (CFGRL [4]); beta = 1 is the plain conditional model.
# ======================================================================================
def guided_velocity_fn(v_cond: VelocityFn, v_uncond: VelocityFn, beta: float) -> VelocityFn:
    if beta == 1.0:
        return v_cond

    def v(x_t, t):
        vu = v_uncond(x_t, t)
        return vu + beta * (v_cond(x_t, t) - vu)

    return v


# ======================================================================================
# 2. The policy. openpi policy.py L67-L106 stages, plus the subtask schedule and the advantage / CFG branches.
# ======================================================================================
class Pi06Policy:
    def __init__(self, model: Pi06, seq: Pi06SequenceTokenizer, robot: Pi05RobotSpec, *, num_steps: int = NUM_DENOISING_STEPS, beta: float = 1.0,
                 hl_period_s: float = HL_PERIOD_S, image_keys=STATIC_IMAGE_KEYS, max_new_tokens: int = MAX_NEW_TOKENS, metadata: str | None = None):
        self.model, self.seq, self.robot = model.eval(), seq, robot
        self.num_steps, self.beta, self.hl_period_s, self.image_keys, self.max_new_tokens, self.metadata = num_steps, beta, hl_period_s, tuple(image_keys), max_new_tokens, metadata
        self.subtask, self.last_hl_time = None, None

    def reset(self):
        self.subtask, self.last_hl_time = None, None

    def hl_due(self, t_now: float) -> bool:
        return self.last_hl_time is None or (t_now - self.last_hl_time) >= self.hl_period_s

    @torch.no_grad()
    def predict_subtask(self, raw: dict) -> tuple[str, int]:
        """layout "hl_prompt" -> greedy decode until '\\n' / EOS -> subtask text (Sec. V-A: ell_hat is predicted first)."""
        obs, _ = build_pi06_batch(raw, self.robot.norm_stats, self.seq, layout="hl_prompt", image_keys=self.image_keys, action_horizon=self.robot.action_horizon,
                                  delta_mask=None, train=False, metadata=[self.metadata] * raw["state"].shape[0])
        toks, n = self.model.sample_text(obs, max_new_tokens=self.max_new_tokens, stop_ids=(EOS_ID, self.seq.newline_id))
        return self.seq.extract_subtask(toks[0].numpy()), n

    @torch.no_grad()
    def infer(self, raw: dict, t_now: float = 0.0, noise: torch.Tensor | None = None) -> dict:
        """raw = {"images": {slot: uint8[1, h, w, 3]}, "state": f32[1, 14], "prompt": [str]} ->
        {"actions": f32[1, H, 14] absolute joint / gripper targets, "x_0": f32[1, H, 32], "subtask": str, "hl_ran": bool, "timing": {stage: ms}}."""
        timing, t0 = {}, time.perf_counter()

        def lap(name):
            nonlocal t0
            t1 = time.perf_counter()
            timing[name] = (t1 - t0) * 1e3
            t0 = t1

        b = raw["state"].shape[0]
        hl_ran = False
        if self.hl_due(t_now):  # high level at a lower rate (Sec. V-A); period from Hi Robot (README Sec. 8)
            self.subtask, n_tok = self.predict_subtask(raw)
            self.last_hl_time, hl_ran = t_now, True
            lap(f"subtask decode ({n_tok} tokens)")
        # deployment: I = True (Sec. V-D) = beta = 1 in Eq. 2; the CFG branch additionally needs the unconditional prefix (App. E)
        common = dict(norm_stats=self.robot.norm_stats, seq=self.seq, layout="flow", image_keys=self.image_keys, action_horizon=self.robot.action_horizon,
                      delta_mask=self.robot.delta_mask, train=False, subtasks=[self.subtask] * b, metadata=[self.metadata] * b)
        obs_c, _ = build_pi06_batch(raw, advantages=[True] * b, **common)
        obs_u = None if self.beta == 1.0 else build_pi06_batch(raw, advantages=[None] * b, **common)[0]
        lap("data preprocessing")
        cache_c, valid_c = self.model.prefix_cache(obs_c)
        v = self.model.make_velocity_fn(cache_c, valid_c)
        if obs_u is not None:
            cache_u, valid_u = self.model.prefix_cache(obs_u)
            v = guided_velocity_fn(v, self.model.make_velocity_fn(cache_u, valid_u), self.beta)
        lap("observation forward pass" + ("" if obs_u is None else " x2 (CFG)"))
        if noise is None:
            noise = torch.randn(b, self.robot.action_horizon, ACTION_DIM)
        x_0 = euler_sample(v, noise, self.num_steps)
        lap(f"x{self.num_steps} action forward pass (flow{'' if obs_u is None else ', x2 per step'})")
        actions = to_executable_actions(x_0, np.asarray(raw["state"], np.float32), self.robot)
        lap("inverse transforms")
        timing["total"] = sum(timing.values())
        return {"actions": actions, "x_0": x_0, "subtask": self.subtask, "hl_ran": hl_ran, "timing": timing}


def tiny_pi06_policy(seed: int = 0, **kw):
    from pi.pi06.backbone.model import tiny_pi06
    from pi.pi06.data.data import tiny_pi06_tokenizer

    torch.manual_seed(seed)
    robot = static_spec()
    return Pi06Policy(tiny_pi06(), tiny_pi06_tokenizer(10, 7), robot, **kw), robot


# ======================================================================================
# 3. One end-to-end call with the tiny config.  uv run python -m pi.pi06.infer.model
# ======================================================================================
def main() -> None:
    rng = np.random.default_rng(0)
    policy, robot = tiny_pi06_policy(max_new_tokens=6)
    obs = {name: rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for name in STATIC_CAMERAS}
    obs["state"] = rng.uniform(-0.5, 0.5, len(STATIC_DIMS_14)).astype(np.float32)
    raw = static_obs_to_raw(obs, "make a double espresso")
    print(f"robot: {robot.native_dim} dims = {len(STATIC_DIMS_14)} names, delta_mask {''.join('1' if d else '0' for d in robot.delta_mask)} (joints delta, grippers absolute)")
    print(f"       H {robot.action_horizon} @ {robot.control_hz:.0f} Hz = {robot.chunk_seconds:.1f} s per chunk; 3 cameras {list(STATIC_CAMERAS)} -> {list(raw['images'])}")
    noise = torch.randn(1, robot.action_horizon, ACTION_DIM)
    out = policy.infer(raw, t_now=0.0, noise=noise)
    print(f"\n[t = 0.0 s] high level ran: {out['hl_ran']}, subtask {out['subtask']!r} (untrained: noise); x_0 {tuple(out['x_0'].shape)} -> actions {tuple(out['actions'].shape)} absolute 14-dim targets")
    print(f"            row 0: left joints {np.round(out['actions'][0, 0, :3], 3)} ... grippers {np.round(out['actions'][0, 0, [6, 13]], 3)}")
    print(f"[timing]    tiny CPU (card: {LATENCY_H100_MS:.0f} ms per chunk on one H100, 5 steps, 3 cameras):")
    for k, v in out["timing"].items():
        print(f"            {k:44s} {v:8.1f} ms")
    out2 = policy.infer(raw, t_now=0.5, noise=noise)
    print(f"[t = 0.5 s] high level ran: {out2['hl_ran']} (period {policy.hl_period_s} s, Hi Robot); same subtask reused; actions identical to t=0? {np.allclose(out['actions'], out2['actions'])}")
    # at init the zero-initialised adaRMSNorm gates make the expert ignore the prefix (v_c == v_u); perturb them so CFG shows
    cfg_policy, _ = tiny_pi06_policy(beta=2.0, max_new_tokens=6)
    for m in cfg_policy.model.modules():
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    cfg_policy.beta = 1.0
    ref = cfg_policy.infer(raw, t_now=0.0, noise=noise)
    cfg_policy.beta = 2.0
    out3 = cfg_policy.infer(raw, t_now=0.0, noise=noise)
    out = ref
    print(f"[CFG beta 2] (expert gates perturbed) x_0 differs from beta 1 by rms {float((out3['x_0'] - out['x_0']).pow(2).mean().sqrt()):.4f} (v = v_u + 2 (v_c - v_u), App. E Eq. 13); "
          f"timing: {', '.join(f'{k}: {v:.0f} ms' for k, v in out3['timing'].items() if 'forward' in k)}")
    print("            deployment default is beta = 1 (Sec. V-D: I_t fixed True); beta in [1.5, 2.5] 'where useful' (App. E)")


if __name__ == "__main__":
    main()
