"""pi0 action expert: Gemma 300M weights that share attention with the VLM, plus the state / action / timestep embedding.

Minimal PyTorch re-implementation of openpi's JAX code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/models/pi0.py (embed_suffix, forward paths), src/openpi/models/gemma.py (two-expert Attention / Block / Module)
  paper   pi0 arXiv:2410.24164v1 Sec. III, Appendix B, Appendix D
Upstream license: Apache-2.0 (openpi, big_vision). This file re-implements, it does not copy.

What this module covers: (state, noisy_actions, timestep) -> 51 suffix tokens in the expert width; a Gemma stack where
every layer holds one set of weights per expert and the experts meet only inside attention; the training-style joint
forward over prefix + suffix and the inference-style suffix forward against a prefix KV cache; decoding the last 50
tokens to the velocity field. Timestep sampling, the loss and the Euler loop live in ../flow_matching.

Building blocks (RMSNorm, ExpertAttnProj, attend, apply_rope, GeGLU, GemmaBlock, make_attn_mask) are imported from
../vlm/model.py; the single-expert GemmaBlock there is reused as "one expert's slot" of a two-expert block.
Attribute names follow the openpi parameter tree; expert i > 0 gets the "_i" suffix in openpi (gemma.py L443-L451),
here it is experts[i].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi.pi0.vlm.model import GEMMA_2B, GEMMA_300M, GemmaConfig, GemmaBlock, RMSNorm, apply_rope, attend, make_attn_mask, tiny_gemma

ACTION_DIM = 32  # pi0_config.py L25
ACTION_HORIZON = 50  # pi0_config.py L26


def tiny_expert() -> GemmaConfig:
    """CPU-sized action expert. Narrower than tiny_gemma (32 vs 64) on purpose: the paper expert is also narrower than
    the VLM (1024 vs 2048). Heads / kv heads / head_dim must match tiny_gemma (gemma.py L165-L168)."""
    return GemmaConfig(width=32, depth=4, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=16, vocab_size=1)


# ======================================================================================
# 1. Timestep embedding. openpi@215abfb pi0.py L47-L63.
# ======================================================================================
def posemb_sincos(pos: torch.Tensor, embedding_dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """pos float32[B] -> float32[B, embedding_dim] = concat(sin(2*pi*pos/period), cos(2*pi*pos/period)).
    period is log-uniform from min_period to max_period over embedding_dim/2 channels. pi0 calls this with
    (timestep, 1024, 4e-3, 4.0): the slowest channel completes 1/4 of a cycle over tau in [0, 1], the fastest 250."""
    assert embedding_dim % 2 == 0
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    x = pos.float()[:, None] * (2 * math.pi / period)[None, :]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# ======================================================================================
# 2. Two-expert Gemma. openpi@215abfb gemma.py L158-L249 (Attention), L284-L333 (Block), L340-L411 (Module).
#    Every expert owns its own norms, q/kv/out projections and MLP; q, k, v of all present experts are concatenated
#    along the sequence axis and go through ONE attention. Nothing else is shared.
# ======================================================================================
class MoEBlock(nn.Module):
    def __init__(self, cfgs: tuple[GemmaConfig, ...]):
        super().__init__()
        c0 = cfgs[0]
        assert all((c.num_heads, c.num_kv_heads, c.head_dim) == (c0.num_heads, c0.num_kv_heads, c0.head_dim) for c in cfgs)
        self.cfgs = cfgs
        # experts[i] holds pre_attention_norm / attn / pre_ffw_norm / mlp for expert i; its own forward is not used.
        self.experts = nn.ModuleList(GemmaBlock(c) for c in cfgs)

    def forward(self, xs, positions, mask, kv_cache=None):
        """xs: list over experts of x_i[B, T_i, w_i] or None. positions int64[B, sum T_i], mask bool[B, sum T_i, S].
        Returns (list of updated x_i or None, (k, v) over all present tokens incl. cache)."""
        present = [(i, x) for i, x in enumerate(xs) if x is not None]
        qs, ks, vs = [], [], []
        for i, x in present:  # step 1: per-expert norm + projection into the shared head space (L172-L199)
            e = self.experts[i]
            q, k, v = e.attn.qkv(e.pre_attention_norm(x))
            qs.append(q), ks.append(k), vs.append(v)
        q, k, v = torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1)  # step 2: concat along sequence (L201)
        q = apply_rope(q, positions) * self.cfgs[0].head_dim**-0.5
        k = apply_rope(k, positions)
        enc, kv = attend(q, k, v, mask, kv_cache)  # step 3: one attention for everybody (L216-L231)
        out = list(xs)
        start = 0
        for i, x in present:  # step 4-5: split back, per-expert out projection, residual, per-expert MLP (L233-L247, L314-L330)
            e = self.experts[i]
            end = start + x.shape[1]
            x = x + e.attn.out(enc[:, start:end])
            out[i] = x + e.mlp(e.pre_ffw_norm(x))
            start = end
        return out, kv


class MoEGemma(nn.Module):
    """Decoder stack with one set of weights per expert. gemma.py L340-L411.
    forward(xs, positions, mask, kv_cache=None) -> (list of final-normed hidden states or None, kv_cache list over layers).
    The VLM's vocabulary embedding belongs to expert 0 (gemma.py L355-L359); ../vlm's PaliGemma already owns one, so
    `with_embedder` is off by default here and ../infer decides where it lives."""

    def __init__(self, cfgs: tuple[GemmaConfig, ...], with_embedder: bool = False):
        super().__init__()
        assert all(c.depth == cfgs[0].depth for c in cfgs)  # L352
        self.cfgs = cfgs
        self.embedder = None
        if with_embedder:
            from pi.pi0.vlm.model import Embedder

            self.embedder = Embedder(cfgs[0])
        self.layers = nn.ModuleList(MoEBlock(cfgs) for _ in range(cfgs[0].depth))
        self.final_norms = nn.ModuleList(RMSNorm(c.width) for c in cfgs)  # L382

    def forward(self, xs, positions, mask, kv_cache=None):
        xs = list(xs)
        new_cache = []
        for i, layer in enumerate(self.layers):
            xs, kv = layer(xs, positions, mask, None if kv_cache is None else kv_cache[i])
            new_cache.append(kv)
        return [None if x is None else n(x) for n, x in zip(self.final_norms, xs)], new_cache


# ======================================================================================
# 3. The robotics-specific projections and the suffix. openpi@215abfb pi0.py L92-L100, L140-L186, L212.
# ======================================================================================
class ActionExpert(nn.Module):
    def __init__(self, cfg: GemmaConfig = GEMMA_300M, action_dim: int = ACTION_DIM, action_horizon: int = ACTION_HORIZON):
        super().__init__()
        w = cfg.width
        self.cfg, self.action_dim, self.action_horizon = cfg, action_dim, action_horizon
        self.state_proj = nn.Linear(action_dim, w)  # L97; state and action share the 32-dim padded layout (../data)
        self.action_in_proj = nn.Linear(action_dim, w)  # L92, W1 in Appendix B
        self.action_time_mlp_in = nn.Linear(2 * w, w)  # L98, W2
        self.action_time_mlp_out = nn.Linear(w, w)  # L99, W3
        self.action_out_proj = nn.Linear(w, action_dim)  # L100

    def embed_suffix(self, state, noisy_actions, timestep):
        """state f32[B, 32], noisy_actions f32[B, 50, 32], timestep f32[B] in [0, 1]
        -> (tokens f32[B, 51, w], input_mask bool[B, 51] all True, ar_mask bool[51] = [1, 1, 0 x 49]). pi0.py L140-L186."""
        b = state.shape[0]
        state_token = self.state_proj(state)[:, None, :]  # [B, 1, w], L153
        action_tokens = self.action_in_proj(noisy_actions)  # [B, 50, w], L159
        time_emb = posemb_sincos(timestep, self.cfg.width, min_period=4e-3, max_period=4.0)  # [B, w], L161
        time_tokens = time_emb[:, None, :].expand(-1, self.action_horizon, -1)  # L172
        x = torch.cat([action_tokens, time_tokens], dim=-1)  # [B, 50, 2w], L173
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))  # W3 swish(W2 concat(...)), L174-L176
        tokens = torch.cat([state_token, x], dim=1)  # [B, 51, w]
        input_mask = torch.ones(b, 1 + self.action_horizon, dtype=torch.bool, device=state.device)  # L155, L180
        # state opens a block (prefix must not see it), the first action opens the action block, the rest join it. L157, L182
        ar_mask = torch.tensor([True, True] + [False] * (self.action_horizon - 1), device=state.device)
        return tokens, input_mask, ar_mask

    def decode(self, suffix_out):
        """f32[B, 51, w] -> velocity field f32[B, 50, 32]; the state token's output is dropped. pi0.py L212."""
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])


# ======================================================================================
# 4. The two ways prefix and suffix meet. openpi@215abfb pi0.py L202-L211 (training) and L239-L269 (inference).
# ======================================================================================
def joint_forward(llm: MoEGemma, prefix_emb, prefix_mask, prefix_ar, suffix_emb, suffix_mask, suffix_ar):
    """One pass over [prefix | suffix]: expert 0 gets the prefix, expert 1 the suffix. Returns (prefix_out, suffix_out)."""
    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)  # [B, 867]
    ar_mask = torch.cat([prefix_ar, suffix_ar], dim=0)  # [867]: 816 x 0, 1, 1, 49 x 0 -> the three blocks of Appendix B
    mask = make_attn_mask(input_mask, ar_mask)  # [B, 867, 867]
    positions = input_mask.long().cumsum(1) - 1  # L208, padding consumes no positions
    (prefix_out, suffix_out), _ = llm([prefix_emb, suffix_emb], positions, mask)
    return prefix_out, suffix_out


def suffix_forward(llm: MoEGemma, kv_cache, prefix_mask, suffix_emb, suffix_mask, suffix_ar):
    """Suffix only, attending into the cached prefix keys / values: expert 0 is absent. Returns suffix_out [B, 51, w1].
    Mask rows are the 51 suffix queries; columns are 816 cached prefix keys (visible iff valid) then 51 suffix keys."""
    b, s = suffix_emb.shape[:2]
    prefix_cols = prefix_mask[:, None, :].expand(-1, s, -1)  # [B, 51, 816], L249
    suffix_cols = make_attn_mask(suffix_mask, suffix_ar)  # [B, 51, 51], L246
    mask = torch.cat([prefix_cols, suffix_cols], dim=-1)  # [B, 51, 867], L252
    positions = prefix_mask.long().sum(-1, keepdim=True) + suffix_mask.long().cumsum(-1) - 1  # L259
    (prefix_out, suffix_out), _ = llm([None, suffix_emb], positions, mask, kv_cache=kv_cache)
    assert prefix_out is None
    return suffix_out


PI0_EXPERTS = (GEMMA_2B, GEMMA_300M)  # pi0_config.py L21-L22: paligemma_variant gemma_2b, action_expert_variant gemma_300m


def tiny_experts() -> tuple[GemmaConfig, GemmaConfig]:
    return tiny_gemma(), tiny_expert()
