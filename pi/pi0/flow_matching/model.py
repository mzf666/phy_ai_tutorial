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

from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, ActionProjections, MoEGemma, suffix_forward

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
    print("[output] actions (normalized, delta, 32-dim padded)", tuple(x_0.shape),
          " -> ../data unnormalize / to-absolute / truncate -> execute the first 25 (or 16) steps")


if __name__ == "__main__":
    main()
