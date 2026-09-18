"""pi0.6 backbone: a minimal Gemma 3 (GQA 8 q / 4 kv heads, QK-norm, 5 local : 1 global layers with a 1024-token
sliding window, RoPE base 10k local / 1M global with 1/8 position scaling, post-norms on attention and FFN, 262,144-word
vocabulary), the 448x448 SigLIP path (1024 patches -> 2x2 average pool -> 256 tokens -> RMSNorm -> linear), the
"images bidirectional, text causal" prefix mask, the 34-layer adaRMSNorm action expert that shares attention with the
backbone, and the Pi06 model: prefix KV cache, 5-step flow sampling, subtask decoding.

Minimal PyTorch re-implementation. Sources of truth:
  gemma    https://github.com/google-deepmind/gemma @ 0513283af5afffa27390b6ede2facc35d0f16e08
           gemma/gm/nn/_gemma.py: Gemma3_4B L221-L246 (embed 2560, hidden 2560*8//2, 8 heads, 4 kv heads, head_dim 256,
           post norms, qk-norm, sliding 1024, local 10k / global 1M, global_scale_factor 8), GEMMA3_ATTENTION_PATTERN
           L39-L46 (L L L L L G), _NUM_LAYERS_GEMMA3_4B = 34 L34, Gemma3_1B L169-L193;
           gemma/gm/nn/_modules.py: create_sliding_mask L36-L52, Attention qk-norm L160-L161 / L192-L194, RoPE + query
           scaling L196-L204, sliding mask applied L258-L267, Block L400-L490 (pre / post norms);
           gemma/gm/nn/_layers.py RMSNorm (x * rsqrt(mean(x^2) + 1e-6) * (1 + scale));
           gemma/gm/math/_positional_embeddings.py apply_rope L23-L75 (sinusoid /= scale_factor);
           gemma/gm/nn/vision/_vision.py VisionExit L202-L231 (avg-pool to 256), SigLiPFromPatches L234-L284;
           gemma/gm/nn/_modules.py Embedder.encode_vision (mm_soft_embedding_norm + mm_input_projection)
  report   Gemma 3 arXiv:2503.19786v1 Sec. 2 (architecture), Table 1 (4B: vision 417M, embedding 675M, non-embedding 3,209M)
  card     pi0.6 model card (2025-11-17) Sec. 2: Gemma 3 4B backbone, action expert with the same number of layers and
           about 860M parameters, up to four 448x448 images, bidirectional images / causal text / bidirectional actions,
           5 denoising steps, 63 ms per chunk on one H100 with 3 cameras
  paper    pi0.6* arXiv:2511.14759v2 Sec. V-A (860M expert, ell_hat before actions, expert does not read FAST tokens)
  openpi   https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 has no pi0.6 code;
           the two-expert layout, adaRMSNorm expert and the decoding loop are the pi0 / pi0.5 ones (pi.pi0, pi.pi05)
Licenses: Apache-2.0 (gemma, openpi). This file re-implements, it does not copy.

Inference only (repo rule): structure, prefix forward with KV cache, the cached expert forward, sampling and decoding.
The KI training forward (stop-gradient inside attention) and the losses are in ../train. The `insulate` flag of
Gemma3MoEBlock.forward is the one training-time switch that must live here, because KI defines it inside the attention
operation (KI Eq. 5-6); it defaults to False and nothing in this file sets it.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi.fast.data.data import EOS_ID
from pi.fast.model.model import left_to_right_align
from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON
from pi.pi0.flow_matching.model import VelocityFn, sample_actions as euler_sample
from pi.pi0.vlm.model import BIG_NEG, Embedder, ExpertAttnProj, GeGLU, GemmaConfig, RMSNorm, SigLIP, ViTConfig, apply_rope
from pi.pi05.expert.model import AdaRMSNorm, Pi05ActionProjections, _gated_residual
from pi.pi06.data.data import IMAGE_RESOLUTION, Pi06Observation

NUM_DENOISING_STEPS = 5  # card Sec. 2: "With 5 denoising steps and 3 camera inputs, pi0.6 takes 63ms" (pi0.5: 10)
IMAGE_TOKENS = 256  # Gemma 3 report Sec. 2.1 / _vision.py L41: every image becomes 256 soft tokens; pi0.6 undisclosed (README Sec. 8)
MAX_NEW_TOKENS = 64  # subtask decoding cap, undisclosed (README Sec. 8)


# ======================================================================================
# 1. Configuration. _gemma.py L221-L246 (4B), L169-L193 (1B); report Table 1.
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class Gemma3Config(GemmaConfig):
    sliding_window: int = 1024  # _gemma.py L240 (4B); 512 for 1B (L188)
    local_per_global: int = 5  # GEMMA3_ATTENTION_PATTERN L39-L46: five LOCAL_SLIDING then one GLOBAL, repeated
    local_rope_base: float = 10_000.0  # L242
    global_rope_base: float = 1_000_000.0  # L243
    global_rope_scale: float = 8.0  # L244 global_scale_factor (positional interpolation for the 128k context, report Sec. 2)
    use_qk_norm: bool = True  # L234
    post_norms: bool = True  # L232-L233 use_post_attn_norm / use_post_ffw_norm

    def is_global(self, layer: int) -> bool:
        """_config.make_attention_layers_types: pattern * (depth // 6) + pattern[: depth % 6]; layer 5, 11, ... are global."""
        return layer % (self.local_per_global + 1) == self.local_per_global


GEMMA3_4B = Gemma3Config(width=2560, depth=34, mlp_dim=2560 * 8 // 2, num_heads=8, num_kv_heads=4, head_dim=256, vocab_size=262_144)  # L221-L246, L34
GEMMA3_1B = Gemma3Config(width=1152, depth=26, mlp_dim=6 * 1152, num_heads=4, num_kv_heads=1, head_dim=256, vocab_size=262_144, sliding_window=512)  # L169-L193, L33
# SigLIP So400m/14 at 448 (card Sec. 2). Gemma 3 runs the same encoder at 896 (_vision.py L243-L244); the projection into the
# LM width is NOT the ViT head here (Gemma: mm_input_projection after the pool), so out_dim = width and the head is dropped.
SIGLIP_400M_448 = ViTConfig(width=1152, depth=27, mlp_dim=4304, num_heads=16, patch_size=14, image_size=IMAGE_RESOLUTION[0], out_dim=1152)


def expert_config(width: int, mlp_dim: int, backbone: Gemma3Config = GEMMA3_4B) -> Gemma3Config:
    """The action expert: same depth and attention head layout as the backbone (it shares attention), own width.
    Width and mlp_dim are undisclosed (card: "about 860M parameters"; README Sec. 8 lists the candidates)."""
    return dataclasses.replace(backbone, width=width, mlp_dim=mlp_dim, vocab_size=1)


PI06_EXPERT = None  # undisclosed width, see README Sec. 8 and expert_param_count(); ../train and tests use tiny_experts()


def tiny_gemma3() -> Gemma3Config:
    """CPU-sized: 6 layers = one full L L L L L G pattern, window 8 so the sliding mask is visible in a 200-token prompt."""
    return Gemma3Config(width=64, depth=6, mlp_dim=128, num_heads=4, num_kv_heads=2, head_dim=16, vocab_size=262_144, sliding_window=8)


def tiny_expert3() -> Gemma3Config:
    return expert_config(32, 64, tiny_gemma3())


def tiny_experts() -> tuple[Gemma3Config, Gemma3Config]:
    return tiny_gemma3(), tiny_expert3()


def tiny_vit448() -> ViTConfig:
    """Same 448 / 14 -> 32x32 = 1024 patch grid as the paper so every pooled sequence length matches."""
    return ViTConfig(width=32, depth=2, mlp_dim=64, num_heads=2, patch_size=14, image_size=IMAGE_RESOLUTION[0], out_dim=32)


# ======================================================================================
# 2. Attention with QK-norm, per-layer RoPE and the sliding window. _modules.py L36-L52, L150-L270.
# ======================================================================================
class Gemma3AttnProj(ExpertAttnProj):
    """pi0's q / kv / out projections plus the two RMSNorms over head_dim (_modules.py L160-L161, applied L192-L194)."""

    def __init__(self, cfg: Gemma3Config):
        super().__init__(cfg)
        self.q_norm = RMSNorm(cfg.head_dim) if cfg.use_qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim) if cfg.use_qk_norm else None

    def qkv(self, x):
        q, k, v = super().qkv(x)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v


def rope_positions(positions: torch.Tensor, cfg: Gemma3Config, layer: int) -> tuple[torch.Tensor, float]:
    """(positions to feed apply_rope, base). Global layers divide the position by global_rope_scale (positional
    interpolation, _positional_embeddings.py L64-L66: sinusoid /= scale_factor) and use base 1M; local layers base 10k."""
    if cfg.is_global(layer):
        return positions.float() / cfg.global_rope_scale, cfg.global_rope_base
    return positions.float(), cfg.local_rope_base


def sliding_mask(pos_q: torch.Tensor, pos_k: torch.Tensor, window: int) -> torch.Tensor:
    """create_sliding_mask L36-L52: key visible iff pos_q - window < pos_k < pos_q + window (both directions: image
    tokens are bidirectional). pos_q i64[B, T], pos_k i64[B, S] -> bool[B, T, S]."""
    d = pos_k[:, None, :] - pos_q[:, :, None]
    return (d > -window) & (d < window)


def attend(q, k, v, pos_q, pos_k, mask, window: int | None, kv_cache=None):
    """GQA core. q[B,T,N,H], k,v[B,S,K,H] of the NEW tokens (already RoPE'd), pos_q[B,T], pos_k[B,S]; mask bool[B,T,S_total]
    over cached + new keys. kv_cache = (k, v, pos) of earlier keys. window=None: global layer. -> (enc[B,T,N,H], (k, v, pos)).
    Compared with pi.pi0.vlm.attend the cache also stores key positions, which the sliding mask needs (L258-L267)."""
    if kv_cache is not None:
        k, v, pos_k = torch.cat([kv_cache[0], k], 1), torch.cat([kv_cache[1], v], 1), torch.cat([kv_cache[2], pos_k], 1)
    if window is not None:
        mask = mask & sliding_mask(pos_q, pos_k, window)
    b, t, n, h = q.shape
    kh = k.shape[2]
    qg = q.view(b, t, kh, n // kh, h)
    logits = torch.einsum("btkgh,bskh->bkgts", qg, k).float()
    logits = torch.where(mask[:, None, None, :, :], logits, torch.full_like(logits, BIG_NEG))
    probs = torch.softmax(logits, -1).to(v.dtype)
    return torch.einsum("bkgts,bskh->btkgh", probs, v).reshape(b, t, n, h), (k, v, pos_k)


# ======================================================================================
# 3. One layer, one or two experts. _modules.py Block L400-L490 (pre_attn_norm -> attn -> post_attn_norm -> +x;
#    pre_ffw_norm -> mlp -> post_ffw_norm -> +x). Expert 1 (the action expert) replaces its two pre-norms and the
#    final norm by adaRMSNorm with gated residuals, as in pi0.5 (openpi gemma.py L293-L331); its post-norms stay plain.
#    Whether pi0.6's expert keeps adaRMSNorm / post-norms / qk-norm is undisclosed (README Sec. 8).
# ======================================================================================
class Gemma3Block(nn.Module):
    def __init__(self, cfg: Gemma3Config, cond_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        ada = cond_dim is not None
        self.pre_attention_norm = AdaRMSNorm(cfg.width, cond_dim) if ada else RMSNorm(cfg.width)
        self.attn = Gemma3AttnProj(cfg)
        self.post_attention_norm = RMSNorm(cfg.width) if cfg.post_norms else None
        self.pre_ffw_norm = AdaRMSNorm(cfg.width, cond_dim) if ada else RMSNorm(cfg.width)
        self.mlp = GeGLU(cfg)
        self.post_ffw_norm = RMSNorm(cfg.width) if cfg.post_norms else None
        self.ada = ada

    def norm(self, which: int, x, cond):
        n = self.pre_attention_norm if which == 0 else self.pre_ffw_norm
        return n(x, cond) if self.ada else (n(x), None)

    def post(self, which: int, y):
        n = self.post_attention_norm if which == 0 else self.post_ffw_norm
        return y if n is None else n(y)


class Gemma3MoEBlock(nn.Module):
    """Experts meet only inside attention (pi0 gemma.py L158-L249 layout). experts[0] = backbone, experts[1] = action expert."""

    def __init__(self, cfgs: tuple[Gemma3Config, ...], layer: int, cond_dim: int | None = None):
        super().__init__()
        c0 = cfgs[0]
        assert all((c.num_heads, c.num_kv_heads, c.head_dim, c.is_global(layer)) == (c0.num_heads, c0.num_kv_heads, c0.head_dim, c0.is_global(layer)) for c in cfgs)
        self.cfgs, self.layer = cfgs, layer
        self.window = None if c0.is_global(layer) else c0.sliding_window
        self.experts = nn.ModuleList(Gemma3Block(c, cond_dim if i == 1 else None) for i, c in enumerate(cfgs))

    def forward(self, xs, positions, mask, kv_cache=None, cond=None, insulate: bool = False):
        """xs: [x0 | None, x1 | None] with x_i[B, T_i, w_i]; positions i64[B, sum T_i]; mask bool[B, sum T_i, S_total].
        cond f32[B, cond_dim] for expert 1's adaRMSNorm. insulate=True (training only, ../train): expert-1 queries see
        stop-gradient copies of the backbone keys / values (KI Eq. 5-6). Returns (xs, (k, v, pos) over all present tokens)."""
        present = [(i, x) for i, x in enumerate(xs) if x is not None]
        qs, ks, vs, gates = [], [], [], {}
        for i, x in present:
            h, gate = self.experts[i].norm(0, x, cond)
            gates[i] = gate
            q, k, v = self.experts[i].attn.qkv(h)
            qs.append(q), ks.append(k), vs.append(v)
        q, k, v = torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1)
        rp, base = rope_positions(positions, self.cfgs[0], self.layer)
        q = apply_rope(q, rp, base) * self.cfgs[0].head_dim**-0.5  # query_pre_attn_scalar = 1/sqrt(head_dim), _gemma.py L238
        k = apply_rope(k, rp, base)
        if insulate and len(present) == 2:
            # KI Eq. 5-6: rows of the backbone tokens attend as usual; rows of the expert tokens attend sg(K_b), sg(V_b) and their own K_a, V_a.
            t0 = present[0][1].shape[1]
            k_sg, v_sg = torch.cat([k[:, :t0].detach(), k[:, t0:]], 1), torch.cat([v[:, :t0].detach(), v[:, t0:]], 1)
            enc0, cache = attend(q[:, :t0], k, v, positions[:, :t0], positions, mask[:, :t0], self.window, kv_cache)
            enc1, _ = attend(q[:, t0:], k_sg, v_sg, positions[:, t0:], positions, mask[:, t0:], self.window, kv_cache)
            enc = torch.cat([enc0, enc1], 1)
        else:
            enc, cache = attend(q, k, v, positions, positions, mask, self.window, kv_cache)
        out = list(xs)
        start = 0
        for i, x in present:
            e = self.experts[i]
            end = start + x.shape[1]
            x = _gated_residual(x, e.post(0, e.attn.out(enc[:, start:end])), gates[i])
            h, gate = e.norm(1, x, cond)
            out[i] = _gated_residual(x, e.post(1, e.mlp(h)), gate)
            start = end
        return out, cache


class Gemma3Stack(nn.Module):
    """The decoder stack, one or two experts. forward(xs, positions, mask, kv_cache=None, cond=None, insulate=False)
    -> ([h0, h1] after the final norms, cache list over layers). Layer i is global iff i % 6 == 5 (4B: 5 of 34 are global)."""

    def __init__(self, cfgs: tuple[Gemma3Config, ...], cond_dim: int | None = None):
        super().__init__()
        assert all(c.depth == cfgs[0].depth for c in cfgs)
        self.cfgs = cfgs
        if len(cfgs) > 1 and cond_dim is None:
            cond_dim = cfgs[1].width  # pi0.5 convention: the time MLP lives in the expert width
        self.cond_dim = cond_dim
        self.layers = nn.ModuleList(Gemma3MoEBlock(cfgs, i, cond_dim) for i in range(cfgs[0].depth))
        self.final_norms = nn.ModuleList([RMSNorm(cfgs[0].width)] + [AdaRMSNorm(c.width, cond_dim) for c in cfgs[1:]])

    def forward(self, xs, positions, mask, kv_cache=None, cond=None, insulate: bool = False):
        xs = list(xs)
        new_cache = []
        for i, layer in enumerate(self.layers):
            xs, kv = layer(xs, positions, mask, None if kv_cache is None else kv_cache[i], cond, insulate)
            new_cache.append(kv)
        outs = [None if xs[0] is None else self.final_norms[0](xs[0])]
        for x, n in zip(xs[1:], self.final_norms[1:]):
            outs.append(None if x is None else n(x, cond)[0])
        return outs, new_cache


# ======================================================================================
# 4. Vision: SigLIP at 448 -> 1024 patches -> 2x2 avg-pool -> 256 -> RMSNorm -> linear into the LM width.
#    _vision.py VisionExit L202-L231 + SigLiPFromPatches L282-L284; Embedder.encode_vision (norm + projection).
#    pi0's SigLIP carries a zero-initialised head into the LM width; Gemma 3's projection sits after the pool instead.
# ======================================================================================
class SigLIPNoHead(SigLIP):
    def __init__(self, cfg: ViTConfig):
        assert cfg.out_dim == cfg.width, "the projection is VisionEmbed.proj, not the ViT head"
        super().__init__(cfg)
        self.head = nn.Identity()


class VisionEmbed(nn.Module):
    def __init__(self, vit_cfg: ViTConfig = SIGLIP_400M_448, lm_width: int = GEMMA3_4B.width, num_tokens: int = IMAGE_TOKENS):
        super().__init__()
        self.img = SigLIPNoHead(vit_cfg)
        self.num_tokens = num_tokens
        self.mm_soft_embedding_norm = RMSNorm(vit_cfg.width)
        self.mm_input_projection = nn.Linear(vit_cfg.width, lm_width, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """f32[B, 448, 448, 3] in [-1, 1] -> f32[B, 256, lm_width]."""
        x = self.img(images)  # [B, 1024, w_v]
        b, n, w = x.shape
        g, go = int(math.sqrt(n)), int(math.sqrt(self.num_tokens))
        assert g * g == n and go * go == self.num_tokens and g % go == 0, (n, self.num_tokens)
        x = x.view(b, g, g, w).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, g // go).permute(0, 2, 3, 1).reshape(b, self.num_tokens, w)  # VisionExit L226-L231
        return self.mm_input_projection(self.mm_soft_embedding_norm(x))


# ======================================================================================
# 5. The prefix mask. Card Sec. 2: "bidirectional attention among all of the image tokens ... causal attention among the
#    text tokens. Action tokens fed into the action expert use bidirectional attention." Plus the data-level rule of who
#    may read which text segment (../data expert_visible) and pi0's "the prefix never sees the expert".
# ======================================================================================
def make_pi06_mask(n_img: int, token_mask: torch.Tensor, n_expert: int = 0, expert_visible: torch.Tensor | None = None) -> torch.Tensor:
    """token axis = [n_img image tokens | L text tokens | E expert tokens]. token_mask bool[B, L] (valid text),
    expert_visible bool[B, L] (which text columns the expert may read; default: all valid). -> bool[B, S+E, S+E].
    Image rows: every valid image column (bidirectional; image tokens of a masked camera are invalid, ../data).
    Text row i: all images + valid text columns j <= i (causal). Expert rows: images + expert_visible + all expert columns.
    Image / text rows never see expert columns. Whether image rows may read text is undisclosed (README Sec. 8): no here."""
    b, l = token_mask.shape
    dev = token_mask.device
    s = n_img + l
    n = s + n_expert
    img_valid = torch.ones(b, n_img, dtype=torch.bool, device=dev)
    valid = torch.cat([img_valid, token_mask], 1)  # [B, S]
    mask = torch.zeros(b, n, n, dtype=torch.bool, device=dev)
    mask[:, :n_img, :n_img] = True
    idx = torch.arange(l, device=dev)
    causal = (idx[None, :] <= idx[:, None])[None] & token_mask[:, None, :]  # [B, L, L]
    mask[:, n_img:s, :n_img] = True
    mask[:, n_img:s, n_img:s] = causal
    if n_expert:
        vis = token_mask if expert_visible is None else (expert_visible & token_mask)
        mask[:, s:, :n_img] = True
        mask[:, s:, n_img:s] = vis[:, None, :].expand(-1, n_expert, -1)
        mask[:, s:, s:] = True
    mask &= torch.cat([valid, torch.ones(b, n_expert, dtype=torch.bool, device=dev)], 1)[:, :, None]  # invalid rows attend nothing
    return mask


def image_columns_valid(obs: Pi06Observation, n_tokens: int = IMAGE_TOKENS) -> torch.Tensor:
    """bool[B, n_img * n_tokens]: the columns of a masked camera are invalid (pi0 rule)."""
    return torch.cat([obs.image_masks[k][:, None].expand(-1, n_tokens) for k in obs.images], 1)


# ======================================================================================
# 6. The models. Gemma3VLM = vision + embedder + single-expert stack (the value function's body, ../value).
#    Pi06 = Gemma3VLM + the action expert + pi0.5's projections; prefix cache, 5-step flow, subtask decoding.
# ======================================================================================
class Gemma3VLM(nn.Module):
    def __init__(self, vit_cfg: ViTConfig = SIGLIP_400M_448, experts: tuple[Gemma3Config, ...] = (GEMMA3_4B,), cond_dim: int | None = None):
        super().__init__()
        self.vision = VisionEmbed(vit_cfg, experts[0].width)
        self.embedder = Embedder(experts[0])  # 262,144 x width; tied logits head
        self.llm = Gemma3Stack(experts, cond_dim)
        self.cfg = experts[0]

    def logits_head(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.embedder.input_embedding.t()

    def embed_prefix(self, obs: Pi06Observation):
        """-> (emb f32[B, n_img*256 + L, W], valid bool[B, S], n_img_tokens). Images in slot order, then the text."""
        embs = [self.vision(obs.images[k]) for k in obs.images]
        n_img = sum(e.shape[1] for e in embs)
        emb = torch.cat(embs + [self.embedder.encode(obs.tokens)], 1)
        valid = torch.cat([image_columns_valid(obs, embs[0].shape[1]), obs.token_mask], 1)
        return emb, valid, n_img

    def prefix_mask(self, obs: Pi06Observation, n_img: int, n_expert: int = 0, expert_visible=None) -> torch.Tensor:
        m = make_pi06_mask(n_img, obs.token_mask, n_expert, expert_visible)
        img_valid = image_columns_valid(obs, n_img // len(obs.images))
        m[:, :, :n_img] &= img_valid[:, None, :]  # masked camera: its columns unreadable
        m[:, :n_img, :] &= img_valid[:, :, None]  # ... and its rows attend nothing
        return m

    def forward_prefix(self, obs: Pi06Observation):
        """Expert 0 once over the prefix -> (h f32[B, S, W], cache, valid, n_img). Positions = cumsum of valid tokens."""
        emb, valid, n_img = self.embed_prefix(obs)
        pos = valid.long().cumsum(1) - 1
        xs = [emb] + [None] * (len(self.llm.cfgs) - 1)
        outs, cache = self.llm(xs, pos, self.prefix_mask(obs, n_img), None)
        return outs[0], cache, valid, n_img


class Pi06(Gemma3VLM):
    def __init__(self, vit_cfg: ViTConfig, experts: tuple[Gemma3Config, Gemma3Config], action_dim: int = ACTION_DIM, action_horizon: int = ACTION_HORIZON):
        super().__init__(vit_cfg, experts)
        self.proj = Pi05ActionProjections(experts[1], action_dim, action_horizon)  # action_in_proj, time MLP, action_out_proj
        self.action_horizon = action_horizon

    def prefix_cache(self, obs: Pi06Observation):
        _, cache, valid, n_img = self.forward_prefix(obs)
        return cache, valid

    def make_velocity_fn(self, kv_cache, prefix_valid) -> VelocityFn:
        """Expert 1 only, attending into the cached prefix (every valid prefix column: at inference no FAST tokens exist).
        Positions continue after the valid prefix tokens (pi0.py L259)."""

        def v(x_t, t):
            tokens, smask, _sar, cond = self.proj.embed_suffix(x_t, t)
            b, e = tokens.shape[:2]
            mask = torch.cat([prefix_valid[:, None, :].expand(-1, e, -1), torch.ones(b, e, e, dtype=torch.bool, device=x_t.device)], -1)
            pos = prefix_valid.long().sum(1, keepdim=True) + torch.arange(e, device=x_t.device)[None]
            outs, _ = self.llm([None, tokens], pos, mask, kv_cache, cond)
            return self.proj.decode(outs[1])

        return v

    def sample_actions(self, obs: Pi06Observation, noise: torch.Tensor, num_steps: int = NUM_DENOISING_STEPS) -> torch.Tensor:
        """Observation (layout "flow") + noise f32[B, H, 32] -> normalised chunk f32[B, H, 32]; 5 Euler steps (card Sec. 2)."""
        cache, valid = self.prefix_cache(obs)
        return euler_sample(self.make_velocity_fn(cache, valid), noise, num_steps)

    @torch.no_grad()
    def sample_text(self, obs: Pi06Observation, *, max_new_tokens: int = MAX_NEW_TOKENS, stop_ids: tuple[int, ...] = (EOS_ID,),
                    temperature: float = 0.0, generator=None) -> tuple[torch.Tensor, int]:
        """Observation (layout "hl_prompt") -> (tokens i64[B, max_new_tokens], n_steps). Greedy by default; a sample stops
        at any id in stop_ids (../infer passes EOS and the newline that ends the Subtask segment). Same right-aligned
        prefill + cache window as pi0.5 (pi.pi05.hier); the cache carries key positions for the sliding layers."""
        emb, valid, n_img = self.embed_prefix(obs)
        emb, valid, attn = left_to_right_align(emb, valid, self.prefix_mask(obs, n_img))
        b, size = valid.shape
        n_valid = valid.long().sum(1)
        start = size - n_valid
        pos = valid.long().cumsum(1) - 1
        outs, cache = self.llm([emb, None], pos, attn, None)
        last = self.logits_head(outs[0][:, -1:])
        tokens = torch.zeros(b, max_new_tokens, dtype=torch.long, device=emb.device)
        stopped = torch.zeros(b, dtype=torch.bool, device=emb.device)
        col = torch.arange(size + max_new_tokens, device=emb.device)
        stop = torch.tensor(stop_ids, device=emb.device)
        step = 0
        while step < max_new_tokens:
            if temperature > 0.0:
                tok = torch.multinomial(torch.softmax(last[:, 0] / temperature, -1), 1, generator=generator)
            else:
                tok = last[:, 0].argmax(-1, keepdim=True)
            tokens[:, step] = tok[:, 0]
            stopped |= torch.isin(tok[:, 0], stop)
            step += 1
            if bool(stopped.all()):
                break
            position = (n_valid + step)[:, None]  # upstream's +1 offset kept (pi.fast.model README Sec. 8)
            n_cols = size + step
            cmask = (col[None, None, :n_cols] >= start[:, None, None]) & (col[None, None, :n_cols] < n_cols)
            outs, cache = self.llm([self.embedder.encode(tok), None], position, cmask, cache)
            last = self.logits_head(outs[0])
        return tokens, step


def tiny_pi06() -> Pi06:
    return Pi06(tiny_vit448(), tiny_experts())


# ======================================================================================
# 7. Parameter counts. Report Table 1 (4B): vision 417M, embedding 675M, non-embedding 3,209M.
# ======================================================================================
def block_param_count(cfg: Gemma3Config, ada: bool = False) -> int:
    """One Gemma3Block: q + kv + out projections, qk-norms, GeGLU, 2 pre-norms (adaRMSNorm: Dense(w -> 3w) + bias), 2 post-norms."""
    w, n, k, h, f = cfg.width, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.mlp_dim
    attn = w * n * h + w * 2 * k * h + n * h * w + (2 * h if cfg.use_qk_norm else 0)
    mlp = 2 * w * f + f * w
    pre = 2 * (3 * w * w + 3 * w) if ada else 2 * w
    post = 2 * w if cfg.post_norms else 0
    return attn + mlp + pre + post


def backbone_param_count(cfg: Gemma3Config = GEMMA3_4B) -> dict[str, int]:
    return {"embedding": cfg.vocab_size * cfg.width, "non_embedding": cfg.depth * block_param_count(cfg) + cfg.width}


def expert_param_count(cfg: Gemma3Config) -> int:
    """adaRMSNorm expert: depth x block + the final adaRMSNorm (no vocabulary)."""
    return cfg.depth * block_param_count(cfg, ada=True) + 3 * cfg.width * cfg.width + 3 * cfg.width


def expert_width_candidates(target: float = 860e6, tol: float = 0.03, mlp_ratio: int = 4, backbone: Gemma3Config = GEMMA3_4B) -> list[tuple[int, int, int]]:
    """Widths (multiples of 64, mlp = ratio x width) whose adaRMSNorm expert lands within tol of the card's 'about 860M'."""
    out = []
    for w in range(256, 2561, 64):
        n = expert_param_count(expert_config(w, mlp_ratio * w, backbone))
        if abs(n - target) / target <= tol:
            out.append((w, mlp_ratio * w, n))
    return out


# ======================================================================================
# 8. Walk one prefix forward, one denoising step and a few decoding steps with the tiny config.
#    uv run python -m pi.pi06.backbone.model
# ======================================================================================
def main():
    import numpy as np

    from pi.pi06.data.data import STATIC_IMAGE_KEYS, build_pi06_batch, tiny_pi06_tokenizer, unit_stats

    torch.manual_seed(0)
    B, H, d = 2, 10, 7
    cfg, ecfg = tiny_experts()
    model = tiny_pi06().eval()
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"tiny configs: backbone {cfg}\n              expert   {ecfg}")
    print(f"layer types: {''.join('G' if cfg.is_global(i) else 'L' for i in range(cfg.depth))} (4B: {''.join('G' if GEMMA3_4B.is_global(i) else 'L' for i in range(GEMMA3_4B.depth))})")
    print(f"params: vision {n(model.vision):,}  embedder {n(model.embedder):,}  backbone layers {sum(n(l.experts[0]) for l in model.llm.layers):,}  "
          f"expert {sum(n(l.experts[1]) for l in model.llm.layers) + n(model.llm.final_norms[1]):,} (closed form {expert_param_count(ecfg):,})  proj {n(model.proj):,}")
    pc = backbone_param_count()
    print(f"Gemma 3 4B closed form: non-embedding {pc['non_embedding']:,} (report 3,209M), embedding {pc['embedding']:,} (+ mm projection {1152 * 2560:,} -> report 675M)")
    print(f"860M expert candidates (mlp = 4 x width): {[(w, f'{c/1e6:.0f}M') for w, _, c in expert_width_candidates()]}  <- width undisclosed, README Sec. 8")

    rng = np.random.default_rng(0)
    raw = {"images": {k: rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
           "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32), "prompt": ["make a double espresso", "fold the shirt"]}
    seq = tiny_pi06_tokenizer(H, d)
    obs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="flow", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False,
                              subtasks=["grab the portafilter", "grab the collar"], advantages=[True, True])
    with torch.no_grad():
        x = model.vision.img(obs.images["base_0_rgb"])
        print(f"\n[vision] SigLIP 448/14 -> {tuple(x.shape)} patches -> avg-pool 2x2 -> {tuple(model.vision(obs.images['base_0_rgb']).shape)} (256 tokens in LM width)")
        emb, valid, n_img = model.embed_prefix(obs)
        L = obs.tokens.shape[1]
        print(f"[prefix] emb {tuple(emb.shape)} = 3 x 256 image + {L} text; valid {int(valid[0].sum())} (right_wrist masked -> 256 invalid columns; {int(obs.token_mask[0].sum())} real text tokens)")
        m = model.prefix_mask(obs, n_img)
        ti = n_img + 5
        print(f"[mask]   {tuple(m.shape)}; image row sees images {bool(m[0, 0, :n_img][valid[0, :n_img]].all())}, text {bool(m[0, 0, n_img:].any())}; "
              f"text row {ti - n_img} sees text <= itself {bool(m[0, ti, n_img:ti + 1].all())}, later text {bool(m[0, ti, ti + 1:].any())}; masked camera columns {bool(m[0, ti, 512:768].any())}")
        pos = valid.long().cumsum(1) - 1
        sm = sliding_mask(pos, pos, cfg.sliding_window)
        print(f"[window] layer 0 (local, window {cfg.sliding_window}): text row {ti - n_img} additionally keeps {int((m[0, ti] & sm[0, ti]).sum())} of {int(m[0, ti].sum())} visible keys; layer 5 (global) keeps all")
        h, cache, valid, n_img = model.forward_prefix(obs)
        print(f"[prefix] forward -> h {tuple(h.shape)}; cache {len(cache)} layers x (k {tuple(cache[0][0].shape)}, v, pos {tuple(cache[0][2].shape)})")
        noise = torch.randn(B, H, ACTION_DIM)
        v = model.make_velocity_fn(cache, valid)
        vt = v(noise, torch.ones(B))
        print(f"[expert] x_t {tuple(noise.shape)}, tau 1.0 -> v_t {tuple(vt.shape)}; expert reads {int(valid[0].sum())} prefix columns + its {H} tokens")
        x0 = model.sample_actions(obs, noise, NUM_DENOISING_STEPS)
        print(f"[flow]   {NUM_DENOISING_STEPS} Euler steps -> x_0 {tuple(x0.shape)}  (card: 63 ms per chunk on one H100 with 3 cameras)")
        hobs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="hl_prompt", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False)
        toks, steps = model.sample_text(hobs, max_new_tokens=6, stop_ids=(EOS_ID, seq.newline_id))
        print(f"[decode] hl_prompt -> {steps} greedy steps, ids {toks[0].tolist()} (untrained: noise; stops at '\\n' or EOS)")
    print("\n../value builds the value function on Gemma3VLM; ../infer wires CFG, the robot spec and the timing; ../train adds the KI forward with insulate=True")


if __name__ == "__main__":
    main()
