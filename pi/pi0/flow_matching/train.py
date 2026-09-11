"""pi0 flow-matching training objective: Beta timestep, linear interpolation, target velocity, MSE, joint forward.

Minimal PyTorch re-implementation of openpi's JAX code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/models/pi0.py compute_loss (L188-L214)
  paper   pi0 arXiv:2410.24164v1 Sec. III (loss), Appendix B "Sampling the flow matching timestep", Fig. 14
Upstream license: Apache-2.0. This file re-implements, it does not copy.

Training side only (repo rule); the sampler is in model.py. The optimizer, schedule, freezing and the batch-level
aggregation of the per-sample loss belong to ../train.

Sign convention: t = 1 is noise (openpi). Paper's tau = 1 - t; README Sec. 1.3 shows every formula in both forms.
"""

from __future__ import annotations

import torch

from pi.pi0.action_expert.model import ActionProjections, MoEGemma
from pi.pi0.flow_matching.model import VelocityFn
from pi.pi0.vlm.model import make_attn_mask


def sample_timestep(batch_size: int, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    """t ~ Beta(1.5, 1) * 0.999 + 0.001, f32[B]. pi0.py L197.
    Beta(1.5, 1) has density 1.5 * sqrt(b) on [0, 1], rising to its max at b = 1 (mean 0.6): most samples are
    noisy. The affine map keeps t in [0.001, 1.0]: t < 0.001 is never trained, which is fine as long as the Euler
    step (0.1) is larger than 0.001 (paper Appendix B, s = 0.999). Same law as the paper's
    p(tau) = Beta((s - tau)/s; 1.5, 1) under tau = 1 - t (README Sec. 1.3)."""
    # inverse-CDF sampling of Beta(1.5, 1): CDF(b) = b^1.5, so b = u^(1/1.5). Keeps the draw on a torch.Generator.
    u = torch.rand(batch_size, generator=generator, device=device)
    b = u.pow(1.0 / 1.5)
    return b * 0.999 + 0.001


def interpolate(actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """actions, noise f32[B, 50, 32], t f32[B] -> (x_t, u_t). pi0.py L198-L200.
    Linear (optimal-transport) path x_t = t * noise + (1 - t) * actions; its time derivative u_t = noise - actions
    is constant along the path, so the network learns a t-independent target and Euler integrates it exactly.
    One t per sample is broadcast over all 50 steps of the chunk."""
    t = t[:, None, None]
    x_t = t * noise + (1 - t) * actions
    u_t = noise - actions
    return x_t, u_t


def joint_forward(llm: MoEGemma, prefix_emb, prefix_mask, prefix_ar, suffix_emb, suffix_mask, suffix_ar):
    """Training-time pass: prefix and suffix together, xs = [prefix, suffix], 867 tokens, no cache. pi0.py L202-L211.
    Returns (prefix_out, suffix_out). Its inference counterpart is ../action_expert suffix_forward."""
    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)  # [B, 867]
    ar_mask = torch.cat([prefix_ar, suffix_ar], dim=0)  # [867] = 816 x 0, 1, 1, 49 x 0: the three blocks of Appendix B
    mask = make_attn_mask(input_mask, ar_mask)  # [B, 867, 867]
    positions = input_mask.long().cumsum(1) - 1  # L208
    (prefix_out, suffix_out), _ = llm([prefix_emb, suffix_emb], positions, mask)
    return prefix_out, suffix_out


def make_train_velocity_fn(llm: MoEGemma, proj: ActionProjections, prefix_emb, prefix_mask, prefix_ar, state) -> VelocityFn:
    """v(x_t, t) through the joint forward. Same signature as model.make_velocity_fn; same numbers (test_parity)."""

    def v(x_t, t):
        suffix_emb, suffix_mask, suffix_ar = proj.embed_suffix(state, x_t, t)
        _, suffix_out = joint_forward(llm, prefix_emb, prefix_mask, prefix_ar, suffix_emb, suffix_mask, suffix_ar)
        return proj.decode(suffix_out)

    return v


def compute_loss(velocity_fn: VelocityFn, actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Per-sample, per-step flow-matching loss f32[B, 50] = mean over the 32 action dims of (v(x_t, t) - u_t)^2.
    pi0.py L212-L214. No mask: padded action dims and past-episode-end steps are included (README Sec. 4.1).
    The mean over batch and horizon is taken by the training loop (../train)."""
    x_t, u_t = interpolate(actions, noise, t)
    v_t = velocity_fn(x_t, t)
    return (v_t - u_t).pow(2).mean(dim=-1)


# ======================================================================================
# Walk one training forward with the tiny config.
#   uv run python -m pi.pi0.flow_matching.train
# ======================================================================================
def main():
    from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, tiny_experts
    from pi.pi0.vlm.model import IMAGE_KEYS, PaliGemma, tiny_vit

    torch.manual_seed(0)
    B = 4
    vlm_cfg, exp_cfg = tiny_experts()
    pg = PaliGemma(tiny_vit(), vlm_cfg)
    llm = MoEGemma((vlm_cfg, exp_cfg))
    proj = ActionProjections(exp_cfg)

    images = {k: torch.rand(B, 224, 224, 3) * 2 - 1 for k in IMAGE_KEYS}
    image_masks = {k: torch.ones(B, dtype=torch.bool) for k in IMAGE_KEYS}
    tokens = torch.randint(3, vlm_cfg.vocab_size, (B, 48))
    token_mask = torch.zeros(B, 48, dtype=torch.bool)
    token_mask[:, :9] = True
    state = torch.randn(B, ACTION_DIM)
    actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM)  # ../data output: normalized, delta, zero-padded
    actions[:, :, 14:] = 0  # pretend a 14-dim robot: dims 14..31 are padding and still get a loss
    print("[input]  actions f32", tuple(actions.shape), " state f32", tuple(state.shape))

    t = sample_timestep(B)
    noise = torch.randn_like(actions)
    print("[t]      sample_timestep ->", [f"{x:.3f}" for x in t.tolist()], " (Beta(1.5,1)*0.999+0.001; 1 = noise)")
    x_t, u_t = interpolate(actions, noise, t)
    print("[interp] x_t = t*noise + (1-t)*actions", tuple(x_t.shape), f" rms per sample {[f'{r:.2f}' for r in x_t.pow(2).mean((1, 2)).sqrt().tolist()]}")
    print("[interp] u_t = noise - actions        ", tuple(u_t.shape), " (independent of t)")

    prefix_emb, prefix_mask, prefix_ar = pg.embed_prefix(images, image_masks, tokens, token_mask)
    suffix_emb, suffix_mask, suffix_ar = proj.embed_suffix(state, x_t, t)
    print("[joint]  prefix", tuple(prefix_emb.shape), " suffix", tuple(suffix_emb.shape), " -> one pass over", prefix_emb.shape[1] + suffix_emb.shape[1], "tokens, both experts, no cache")
    p_out, s_out = joint_forward(llm, prefix_emb, prefix_mask, prefix_ar, suffix_emb, suffix_mask, suffix_ar)
    print("[joint]  prefix_out", tuple(p_out.shape), " (unused)  suffix_out", tuple(s_out.shape))
    v_t = proj.decode(s_out)
    print("[decode] v_t", tuple(v_t.shape))

    loss = compute_loss(make_train_velocity_fn(llm, proj, prefix_emb, prefix_mask, prefix_ar, state), actions, noise, t)
    assert torch.allclose(loss, (v_t - u_t).pow(2).mean(-1))
    print("[loss]   per (sample, step) f32", tuple(loss.shape), f" mean {loss.mean():.4f}; padded dims 14..31 contribute too")
    loss.mean().backward()
    grads = [n for n, p in list(llm.named_parameters()) + list(proj.named_parameters()) + list(pg.named_parameters()) if p.grad is not None]
    print(f"[grad]   parameters with gradient: {len(grads)} (both experts, the projections, SigLIP and the embedding table)")
    print("\n../train adds the optimizer, schedule, freezing / LoRA, and the data mixture on top of this")


if __name__ == "__main__":
    main()
