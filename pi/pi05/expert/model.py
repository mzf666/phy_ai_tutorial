"""pi0.5 action expert: the same Gemma 300M expert as pi0, but the flow-matching timestep enters every layer through
adaptive RMSNorm (zero-initialised scale / shift / gate, gated residuals) instead of once at the entry, and there is
no state token (the state is text in the prompt, ../data).

Minimal PyTorch re-implementation of openpi's JAX code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/models/gemma.py (RMSNorm with cond L113-L131, Block L293-L331, _gated_residual L453-L459,
          Module.__call__ adarms_cond L392-L411, Module.init L413-L421), src/openpi/models/pi0.py (__init__ L92-L100,
          embed_suffix L140-L186, sample_actions.step L239-L269), src/openpi/models/pi0_config.py L28-L31
  paper   pi0.5 arXiv:2504.16054v1 Appendix E ("Model technical details")
Upstream license: Apache-2.0. This file re-implements, it does not copy.

Inference only (repo rule): the suffix embedding, the two-expert stack with adaptive norms on expert 1, and the cached
suffix forward. The joint (prefix + suffix) training forward and the loss are in ../train. Everything unchanged from
pi0 (MoEBlock layout, attention, RoPE, posemb_sincos, the mask rule) is imported from pi.pi0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, PI0_EXPERTS, MoEBlock, MoEGemma, posemb_sincos, tiny_experts
from pi.pi0.vlm.model import GemmaConfig, apply_rope, attend, make_attn_mask

PI05_EXPERTS = PI0_EXPERTS  # pi0_config.py L21-L22: gemma_2b + gemma_300m, unchanged by the pi05 flag


# ======================================================================================
# 1. Adaptive RMSNorm. gemma.py L113-L131. With cond=None upstream builds a plain RMSNorm with a learned `scale`
#    (L119-L125); with cond it builds ONLY a zero-initialised Dense(cond -> 3 * dim) and no `scale` (L127-L131).
#    The two are different modules with different parameters; this class is the second one.
# ======================================================================================
class AdaRMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + 1e-6) * (1 + scale) + shift, returns (y, gate); (scale, shift, gate) = split(W cond + b).
    W and b are zero at init (L128 kernel_init=zeros; flax bias default zeros), so y starts as the bare normalised x
    and gate starts at 0."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.modulation = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)  # L117-L118, statistics in float32
        scale, shift, gate = self.modulation(cond.float())[:, None, :].chunk(3, dim=-1)  # L128-L129: [B, 1, dim] each
        return (normed * (1 + scale) + shift).to(x.dtype), gate.to(x.dtype)  # L130-L131


# ======================================================================================
# 2. Timestep -> condition vector, and the suffix. pi0.py L92-L95 (time_mlp_in/out), L159-L186 (embed_suffix, pi05
#    branch). Paper Appendix E: swish(W2 swish(W1 phi(tau))), phi = sinusoidal encoding of tau.
# ======================================================================================
class Pi05ActionProjections(nn.Module):
    """action_in_proj, time_mlp_in, time_mlp_out (entry) and action_out_proj (exit). 2,165,792 params at paper size.
    pi0's state_proj / action_time_mlp_in / action_time_mlp_out do not exist here (pi0.py L96-L99 `else` branch)."""

    def __init__(self, cfg: GemmaConfig, action_dim: int = ACTION_DIM, action_horizon: int = ACTION_HORIZON):
        super().__init__()
        w = cfg.width
        self.cfg, self.action_dim, self.action_horizon = cfg, action_dim, action_horizon
        self.action_in_proj = nn.Linear(action_dim, w)  # L92
        self.time_mlp_in = nn.Linear(w, w)  # L94, W1 (nnx.Linear has a bias; the paper writes a bare matrix, README Sec. 8)
        self.time_mlp_out = nn.Linear(w, w)  # L95, W2
        self.action_out_proj = nn.Linear(w, action_dim)  # L100

    def time_cond(self, timestep: torch.Tensor) -> torch.Tensor:
        """f32[B] in [0, 1] -> f32[B, w]: swish(time_mlp_out(swish(time_mlp_in(posemb_sincos(tau))))). L161-L167."""
        emb = posemb_sincos(timestep, self.cfg.width, min_period=4e-3, max_period=4.0)  # L161, same phi as pi0
        return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(emb))))  # L164-L167 (nnx.swish == silu)

    def embed_suffix(self, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        """noisy_actions f32[B, 50, 32], timestep f32[B] -> (tokens f32[B, 50, w], input_mask bool[B, 50] all True,
        ar_mask bool[50] = [1, 0 x 49], cond f32[B, w]). No state token (L151 `if not self.pi05`); the action tokens
        are the plain projection (L159, L168), the timestep goes to `cond` only (L169)."""
        b = noisy_actions.shape[0]
        tokens = self.action_in_proj(noisy_actions)
        cond = self.time_cond(timestep)
        input_mask = torch.ones(b, self.action_horizon, dtype=torch.bool, device=noisy_actions.device)  # L180
        ar_mask = torch.tensor([True] + [False] * (self.action_horizon - 1), device=noisy_actions.device)  # L182
        return tokens, input_mask, ar_mask, cond

    def decode(self, suffix_out: torch.Tensor) -> torch.Tensor:
        """f32[B, 50, w] -> velocity f32[B, 50, 32]. L212 / L269 take the last H tokens; with no state token that is all of them."""
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])


# ======================================================================================
# 3. Two-expert Gemma with adaptive norms on expert 1. gemma.py L293-L331 (Block), L340-L411 (Module).
#    Expert 0 is untouched (its RMSNorms keep the learned scale, gate None -> plain residual, L457-L458), so the
#    PaliGemma weights load as in pi0. Expert 1's two per-layer norms and its final norm become AdaRMSNorm and its
#    residuals are gated (L303-L311, L318-L331, L410). cond=None makes expert 1 behave like pi0's (used for parity).
# ======================================================================================
class AdaMoEBlock(MoEBlock):
    def __init__(self, cfgs: tuple[GemmaConfig, ...], cond_dim: int):
        super().__init__(cfgs)
        e1 = self.experts[1]
        e1.pre_attention_norm = AdaRMSNorm(cfgs[1].width, cond_dim)  # L303 with adarms_cond[1]
        e1.pre_ffw_norm = AdaRMSNorm(cfgs[1].width, cond_dim)  # L318
        self.plain_norm = nn.ModuleList([_PlainRMSNorm(cfgs[1].width), _PlainRMSNorm(cfgs[1].width)])  # for cond=None only (no params)

    def _norm(self, i, which, x, cond):
        e = self.experts[i]
        norm = e.pre_attention_norm if which == 0 else e.pre_ffw_norm
        if i == 1:
            if cond is None:
                return self.plain_norm[which](x), None
            return norm(x, cond)
        return norm(x), None

    def forward(self, xs, positions, mask, kv_cache=None, cond=None):
        """xs [x0 | None, x1 | None], cond f32[B, cond_dim] | None (expert 1 only). Returns (xs, (k, v))."""
        present = [(i, x) for i, x in enumerate(xs) if x is not None]
        qs, ks, vs, gates = [], [], [], {}
        for i, x in present:  # L299-L305: per-expert norm (expert 1 returns a gate) + projection
            h, gate = self._norm(i, 0, x, cond)
            gates[i] = gate
            q, k, v = self.experts[i].attn.qkv(h)
            qs.append(q), ks.append(k), vs.append(v)
        q, k, v = torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1)
        q = apply_rope(q, positions) * self.cfgs[0].head_dim**-0.5
        k = apply_rope(k, positions)
        enc, kv = attend(q, k, v, mask, kv_cache)
        out = list(xs)
        start = 0
        for i, x in present:  # L311: x + gate * attn_out; L316-L331: x + gate * mlp(norm(x))
            e = self.experts[i]
            end = start + x.shape[1]
            x = _gated_residual(x, e.attn.out(enc[:, start:end]), gates[i])
            h, gate = self._norm(i, 1, x, cond)
            out[i] = _gated_residual(x, e.mlp(h), gate)
            start = end
        return out, kv


class _PlainRMSNorm(nn.Module):
    """Parameter-free RMSNorm (what AdaRMSNorm computes at zero init); only used when cond is None."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def _gated_residual(x, y, gate):
    """gemma.py L453-L459: x + y, or x + y * gate when the norm returned one."""
    return x + y if gate is None else x + y * gate


class AdaMoEGemma(MoEGemma):
    """MoEGemma whose expert 1 has adaptive norms. forward(xs, positions, mask, kv_cache=None, cond=None)
    -> ([h0, h1], kv_cache); h1 goes through an adaptive final norm too (L410, gate discarded)."""

    def __init__(self, cfgs: tuple[GemmaConfig, ...], cond_dim: int | None = None, with_embedder: bool = False):
        super().__init__(cfgs, with_embedder)
        cond_dim = cfgs[1].width if cond_dim is None else cond_dim  # pi0.py L94: the time MLP lives in the expert width
        self.layers = nn.ModuleList(AdaMoEBlock(cfgs, cond_dim) for _ in range(cfgs[0].depth))
        self.final_norms[1] = AdaRMSNorm(cfgs[1].width, cond_dim)  # L382 + L410
        self._plain_final = _PlainRMSNorm(cfgs[1].width)

    def forward(self, xs, positions, mask, kv_cache=None, cond=None):
        xs = list(xs)
        new_cache = []
        for i, layer in enumerate(self.layers):
            xs, kv = layer(xs, positions, mask, None if kv_cache is None else kv_cache[i], cond)
            new_cache.append(kv)
        h0 = None if xs[0] is None else self.final_norms[0](xs[0])
        if xs[1] is None:
            h1 = None
        elif cond is None:
            h1 = self._plain_final(xs[1])
        else:
            h1 = self.final_norms[1](xs[1], cond)[0]
        return [h0, h1], new_cache


# ======================================================================================
# 4. Cached suffix forward at inference. pi0.py L239-L269: same mask / positions as pi0, plus cond.
# ======================================================================================
def suffix_forward(llm: AdaMoEGemma, kv_cache, prefix_mask, suffix_emb, suffix_mask, suffix_ar, cond):
    """Suffix (50 action tokens) attending into the cached prefix; expert 0 absent. Returns f32[B, 50, w1]."""
    b, s = suffix_emb.shape[:2]
    prefix_cols = prefix_mask[:, None, :].expand(-1, s, -1)  # L249
    suffix_cols = make_attn_mask(suffix_mask, suffix_ar)  # L246
    mask = torch.cat([prefix_cols, suffix_cols], dim=-1)  # L252
    positions = prefix_mask.long().sum(-1, keepdim=True) + suffix_mask.long().cumsum(-1) - 1  # L259
    (prefix_out, suffix_out), _ = llm([None, suffix_emb], positions, mask, kv_cache=kv_cache, cond=cond)  # L261-L267
    assert prefix_out is None
    return suffix_out


def expert_param_count(cfg: GemmaConfig, depth: int | None = None) -> int:
    """Closed form for expert 1 of AdaMoEGemma: per layer q + kv + out + GeGLU + 2 AdaRMSNorm, plus the final AdaRMSNorm."""
    w, n, k, h, f = cfg.width, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.mlp_dim
    ada = w * 3 * w + 3 * w
    per_layer = w * n * h + w * 2 * k * h + n * h * w + (2 * w * f + f * w) + 2 * ada
    return (cfg.depth if depth is None else depth) * per_layer + ada


# ======================================================================================
# 5. Walk one denoising step with the tiny config and print every shape, next to pi0's expert for contrast.
#    uv run python -m pi.pi05.expert.model
# ======================================================================================
def main():
    from pi.pi0.action_expert.model import ActionProjections
    from pi.pi0.vlm.model import IMAGE_KEYS, PaliGemma, tiny_vit

    torch.manual_seed(0)
    B = 2
    vlm_cfg, exp_cfg = tiny_experts()
    pg = PaliGemma(tiny_vit(), vlm_cfg).eval()
    llm = AdaMoEGemma((vlm_cfg, exp_cfg)).eval()
    proj = Pi05ActionProjections(exp_cfg).eval()
    n = lambda m: sum(p.numel() for p in m.parameters())
    e1 = sum(n(l.experts[1]) for l in llm.layers) + n(llm.final_norms[1])
    print(f"tiny configs: vlm expert={vlm_cfg}\n              action expert={exp_cfg}")
    print(f"params: expert1 {e1:,} (closed form {expert_param_count(exp_cfg):,}; pi0's tiny expert {sum(n(l.experts[1]) for l in MoEGemma((vlm_cfg, exp_cfg)).layers) + exp_cfg.width:,})"
          f"  projections {n(proj):,} (pi0: {n(ActionProjections(exp_cfg)):,})")

    images = {k: torch.rand(B, 224, 224, 3) * 2 - 1 for k in IMAGE_KEYS}
    image_masks = {k: torch.ones(B, dtype=torch.bool) for k in IMAGE_KEYS}
    tokens = torch.randint(3, vlm_cfg.vocab_size, (B, 200))
    token_mask = torch.zeros(B, 200, dtype=torch.bool)
    token_mask[:, :60] = True  # a pi0.5 prompt with state bins is ~60-120 tokens (../data)
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    timestep = torch.ones(B)
    print("\n[input]  x_t f32", tuple(x_t.shape), " timestep f32", tuple(timestep.shape), " (no state: it is in the prompt tokens)")

    with torch.no_grad():
        prefix_emb, prefix_mask, prefix_ar = pg.embed_prefix(images, image_masks, tokens, token_mask)
        (p_out, none_out), kv_cache = llm([prefix_emb, None], prefix_mask.long().cumsum(1) - 1, make_attn_mask(prefix_mask, prefix_ar))
        print("[prefix] emb", tuple(prefix_emb.shape), " -> expert 0 once, kv_cache", len(kv_cache), "x", tuple(kv_cache[0][0].shape), " (unchanged from pi0)")

        emb = posemb_sincos(timestep, exp_cfg.width, 4e-3, 4.0)
        cond = proj.time_cond(timestep)
        print("[time]   posemb_sincos(tau)", tuple(emb.shape), " -> swish(W2 swish(W1 .))", tuple(cond.shape), f" cond rms {cond.pow(2).mean().sqrt():.3f}  (pi0: concat with actions instead)")
        suffix_emb, suffix_mask, suffix_ar, cond2 = proj.embed_suffix(x_t, timestep)
        assert torch.equal(cond, cond2)
        print("[suffix] action_in_proj(x_t)", tuple(suffix_emb.shape), " = [B, 50, w]: no state token (pi0: 51)  ar_mask", suffix_ar[:3].int().tolist(), "... (pi0: [1, 1, 0, ...])")

        # one layer by hand
        layer0 = llm.layers[0]
        h, gate = layer0.experts[1].pre_attention_norm(suffix_emb, cond)
        print(f"[layer0] AdaRMSNorm: normed {tuple(h.shape)}  gate {tuple(gate.shape)}  |scale|,|shift|,|gate| max = "
              f"{[f'{v:.3f}' for v in layer0.experts[1].pre_attention_norm.modulation(cond)[0].abs().view(3, -1).amax(1).tolist()]}  (all 0 at init -> identity layer)")
        s_out = suffix_forward(llm, kv_cache, prefix_mask, suffix_emb, suffix_mask, suffix_ar, cond)
        ident = llm._plain_final(suffix_emb)
        print(f"[cache]  suffix_forward -> {tuple(s_out.shape)};  == plain_norm(action_in_proj(x_t))? {torch.allclose(s_out, ident, atol=1e-5)}  (zero-init: the expert is the identity)")
        v_t = proj.decode(s_out)
        print("[decode] v_t", tuple(v_t.shape), " = [B, 50, 32], all 50 tokens decoded (pi0 drops the state token)")

        # perturb the modulation weights: now every layer acts and tau matters
        for m in llm.modules():
            if isinstance(m, AdaRMSNorm):
                nn.init.normal_(m.modulation.weight, std=0.05)
        v1 = proj.decode(suffix_forward(llm, kv_cache, prefix_mask, *proj.embed_suffix(x_t, torch.full((B,), 1.0))[:3], proj.time_cond(torch.full((B,), 1.0))))
        v2 = proj.decode(suffix_forward(llm, kv_cache, prefix_mask, *proj.embed_suffix(x_t, torch.full((B,), 0.5))[:3], proj.time_cond(torch.full((B,), 0.5))))
        print(f"[perturb] modulation ~ N(0, 0.05^2): |v(tau=1) - v(tau=0.5)| rms {(v1 - v2).pow(2).mean().sqrt():.4f}  (was 0 at init)")
    print("\n../hier assembles SigLIP + embedder + AdaMoEGemma + these projections; ../train adds the joint forward and the loss")


if __name__ == "__main__":
    main()
