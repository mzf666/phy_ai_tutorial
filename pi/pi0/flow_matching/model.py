"""pi0 flow-matching sampler: 10 Euler steps from noise to an action chunk, reusing the prefix KV cache.

Minimal PyTorch re-implementation of openpi's JAX code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/models/pi0.py sample_actions (L216-L279)
  paper   pi0 arXiv:2410.24164v1 Sec. III (Euler rule, 10 steps), Appendix D (inference)
Upstream license: Apache-2.0. This file re-implements, it does not copy.

Inference only (repo rule). Training-side pieces (timestep sampling, interpolation, loss, joint forward) are in
train.py. The network is ../action_expert; this file only integrates its velocity field.

Sign convention (pi0.py L226-L227): t = 1 is pure noise, t = 0 is the clean chunk, dt = -1/num_steps. This is the
opposite of the paper's tau (tau = 1 - t); README Sec. 1.3 maps the two.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import numpy as np

from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, ActionProjections, MoEGemma, suffix_forward
from pi.pi0.data.data import NormStats, to_absolute_actions, unnormalize

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x_t [B,50,32], t [B]) -> v [B,50,32]


def make_velocity_fn(llm: MoEGemma, proj: ActionProjections, kv_cache, prefix_mask, state) -> VelocityFn:
    """Package ../action_expert's embed_suffix -> suffix_forward -> decode as v(x_t, t). The prefix KV cache and
    the state are fixed for the whole denoising loop; only x_t and t change. pi0.py L239-L269 (`step`)."""

    def v(x_t, t):
        suffix_emb, suffix_mask, suffix_ar = proj.embed_suffix(state, x_t, t)
        out = suffix_forward(llm, kv_cache, prefix_mask, suffix_emb, suffix_mask, suffix_ar)
        return proj.decode(out)

    return v


def sample_actions(velocity_fn: VelocityFn, noise: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """Forward Euler from x_1 = noise to x_0. noise f32[B, 50, 32] -> actions f32[B, 50, 32] (normalized space).
    dt = -1/num_steps; t runs 1.0, 1-1/n, ..., 1/n; the loop condition `t >= -dt/2` is the float-robust form of
    "num_steps iterations" (pi0.py L228, L271-L278). The paper's rule A^{tau+delta} = A^tau + delta v with delta = 0.1
    is the same walk in the other direction."""
    dt = -1.0 / num_steps
    b = noise.shape[0]
    x_t, t = noise, 1.0
    while t >= -dt / 2:
        v_t = velocity_fn(x_t, torch.full((b,), t, dtype=noise.dtype, device=noise.device))
        x_t = x_t + dt * v_t
        t = t + dt
    return x_t


def to_executable_actions(x_0: torch.Tensor, state: torch.Tensor, norm_stats: dict[str, NormStats],
                          delta_mask, native_dim: int) -> np.ndarray:
    """From the sampler output to what the robot controller consumes. Inverse of ../data, in openpi's order
    (policy.py L92-L102 builds {"state": normalized padded state, "actions": x_0}; policy_config.py L84-L88 then
    applies Unnormalize -> AbsoluteActions -> the robot's Outputs transform):

      x_0    f32[B, 50, 32]  normalized, delta on joint dims, zero-padded          (model space)
      1. unnormalize actions AND state with the robot's norm_stats: x * (std + 1e-6) + mean, padded dims pass through
                            -> physical units (rad / m / gripper), still delta       (transforms.py L168-L171)
      2. to_absolute_actions: joint dims += the current state q_t (the same q_t the model was given), gripper dims
         (delta_mask False) untouched                                                (transforms.py L226-L245)
      3. truncate to the robot's native dim, e.g. [:7] for LIBERO, [:14] for ALOHA   (libero_policy.py L94-L100)
      -> f32[B, 50, native_dim]: row t' is the absolute target joint position + gripper command for control step t'.

    What happens next is not code in this repo: a platform-specific gripper conversion (aloha_policy.py maps pi's
    gripper angle back to ALOHA's linear position), then the controller sends one row per control period to the arm's
    low-level position controller (the PD loop lives in the motor drivers; the model never outputs torques). Only the
    first 25 rows (50 Hz) or 16 rows (20 Hz) are executed, open-loop, before the next observation is taken
    (paper Appendix D).

    `state` is the normalized 32-dim state that went into embed_suffix; it is unnormalized here (openpi does the same,
    the Unnormalize transform covers both keys of norm_stats)."""
    a = unnormalize(x_0.detach().cpu().numpy(), norm_stats["actions"])
    q = unnormalize(state.detach().cpu().numpy(), norm_stats["state"])
    a = to_absolute_actions(q, a, delta_mask)
    return a[..., :native_dim]


# ======================================================================================
# Walk one full sampling call with the tiny config and print every step.
#   uv run python -m pi.pi0.flow_matching.model
# ======================================================================================
def main():
    from pi.pi0.action_expert.model import tiny_experts
    from pi.pi0.vlm.model import IMAGE_KEYS, PaliGemma, make_attn_mask, tiny_vit

    torch.manual_seed(0)
    B = 2
    vlm_cfg, exp_cfg = tiny_experts()
    pg = PaliGemma(tiny_vit(), vlm_cfg).eval()
    llm = MoEGemma((vlm_cfg, exp_cfg)).eval()
    proj = ActionProjections(exp_cfg).eval()

    images = {k: torch.rand(B, 224, 224, 3) * 2 - 1 for k in IMAGE_KEYS}
    image_masks = {k: torch.ones(B, dtype=torch.bool) for k in IMAGE_KEYS}
    tokens = torch.randint(3, vlm_cfg.vocab_size, (B, 48))
    token_mask = torch.zeros(B, 48, dtype=torch.bool)
    token_mask[:, :9] = True
    state = torch.randn(B, ACTION_DIM)

    with torch.no_grad():
        # --- prefix once (../vlm + expert 0 of ../action_expert) ---
        prefix_emb, prefix_mask, prefix_ar = pg.embed_prefix(images, image_masks, tokens, token_mask)
        (_, none), kv_cache = llm([prefix_emb, None], prefix_mask.long().cumsum(1) - 1, make_attn_mask(prefix_mask, prefix_ar))
        print("[prefix] kv_cache:", len(kv_cache), "layers x", tuple(kv_cache[0][0].shape), " computed once per action chunk")

        # --- the sampler, unrolled so every step is visible ---
        v = make_velocity_fn(llm, proj, kv_cache, prefix_mask, state)
        noise = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
        print("[start]  x_1 = noise f32", tuple(noise.shape), f" |x| rms {noise.pow(2).mean().sqrt():.3f}")
        num_steps, dt = 10, -0.1
        x_t, t, step = noise, 1.0, 0
        while t >= -dt / 2:
            v_t = v(x_t, torch.full((B,), t))
            x_next = x_t + dt * v_t
            print(f"[step {step:2d}] t={t:.1f}  v_t {tuple(v_t.shape)} rms {v_t.pow(2).mean().sqrt():.3f}"
                  f"  |x_t+dt - x_t| rms {(x_next - x_t).pow(2).mean().sqrt():.4f}")
            x_t, t, step = x_next, t + dt, step + 1
        print(f"[end]    {step} steps, final t={t:.2f}, x_0 f32", tuple(x_t.shape))

        # --- same thing through the public entry point ---
        x_0 = sample_actions(v, noise, num_steps)
        assert torch.allclose(x_0, x_t)
    print("[output] x_0 (normalized, delta, 32-dim padded)", tuple(x_0.shape))

    # --- from x_0 to a control signal, for a pretend 7-dim robot (6 joints + 1 gripper, 20 Hz, like LIBERO) ---
    from pi.pi0.data.data import make_bool_mask

    native_dim = 7
    norm_stats = {  # in practice loaded from the checkpoint's norm_stats.json, computed over that robot's training set
        "state": NormStats(mean=np.zeros(native_dim, np.float32), std=np.full(native_dim, 0.5, np.float32)),
        "actions": NormStats(mean=np.zeros(native_dim, np.float32), std=np.full(native_dim, 0.1, np.float32)),
    }
    delta_mask = make_bool_mask(6, -1)  # joints are delta, the gripper is absolute
    a = unnormalize(x_0.numpy(), norm_stats["actions"])
    q = unnormalize(state.numpy(), norm_stats["state"])
    print("[post 1] unnormalize: actions", tuple(a.shape), "state", tuple(q.shape), " physical units, still delta on joints")
    a = to_absolute_actions(q, a, delta_mask)
    print("[post 2] to_absolute_actions: joints += q_t, gripper untouched", tuple(a.shape))
    a = a[..., :native_dim]
    print("[post 3] truncate to native dim", tuple(a.shape), " = [B, 50, 7]: absolute joint targets + gripper per control step")
    assert np.allclose(a, to_executable_actions(x_0, state, norm_stats, delta_mask, native_dim))
    print("[exec]   send rows 0..15 (20 Hz robot: 16 steps = 0.8 s) one per control period to the position controller,"
          " then observe again and re-run")


if __name__ == "__main__":
    main()
