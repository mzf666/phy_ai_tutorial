"""pi0 VLM backbone: SigLIP So400m/14 image encoder + Gemma 2B decoder, i.e. the PaliGemma structure.

Minimal PyTorch re-implementation of openpi's JAX code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/models/siglip.py (ViT, from big_vision), src/openpi/models/gemma.py (Gemma), src/openpi/models/pi0.py
  papers  pi0 arXiv:2410.24164v1 Appendix B; PaliGemma arXiv:2407.07726v2 Sec. 3; Gemma arXiv:2403.08295v4 Sec. 2
Upstream license: Apache-2.0 (openpi, big_vision). This file re-implements, it does not copy.

What this module covers: images -> 256 tokens per image in Gemma width; prompt tokens -> embeddings;
one Gemma forward over the concatenated "prefix" with an attention mask and RoPE positions, returning
hidden states and a KV cache. The checkpoint (PaliGemma pt_224) is treated as given; loading is not here.
The two-expert mixture (action expert) builds on the pieces below and lives in ../action_expert.

Attribute names mirror the openpi parameter tree (embedding, pos_embedding, encoderblock, encoder_norm, head;
embedder.input_embedding, pre_attention_norm, attn.q_einsum/kv_einsum/attn_vec_einsum, pre_ffw_norm,
mlp.gating_einsum/linear, final_norm) so a checkpoint mapping is a rename, not a puzzle.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # same slots as ../data


# ======================================================================================
# Configs. Values: openpi@215abfb siglip.py L308-L370 (decode_variant "So400m/14"), gemma.py L69-L87.
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class ViTConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    patch_size: int
    image_size: int
    out_dim: int  # the ViT "head": projection into the language model width (pi0.py L81-L87: num_classes=paligemma width)


@dataclasses.dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int = 257_152  # PALIGEMMA_VOCAB_SIZE, gemma.py L41


SIGLIP_SO400M_14 = ViTConfig(width=1152, depth=27, mlp_dim=4304, num_heads=16, patch_size=14, image_size=224, out_dim=2048)
GEMMA_2B = GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
GEMMA_300M = GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)  # action expert


def tiny_vit() -> ViTConfig:
    """CPU-sized ViT. Same 224/14 -> 256 token grid as the paper so every sequence length matches."""
    return ViTConfig(width=32, depth=2, mlp_dim=64, num_heads=2, patch_size=14, image_size=224, out_dim=64)


def tiny_gemma() -> GemmaConfig:
    """openpi's "dummy" variant (gemma.py L60-L68) with a small vocab."""
    return GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16, vocab_size=1024)


# ======================================================================================
# 1. SigLIP image encoder (a plain pre-LN ViT). openpi@215abfb siglip.py L188-L290.
#    image float32[B, 224, 224, 3] in [-1, 1]  ->  float32[B, 256, out_dim]
# ======================================================================================
class ViTBlock(nn.Module):
    """LN -> MHA (with biases, xavier init) -> residual; LN -> MLP(GELU) -> residual. siglip.py L75-L108."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.width)
        self.attn = nn.MultiheadAttention(cfg.width, cfg.num_heads, batch_first=True)  # bias=True like flax default
        self.ln2 = nn.LayerNorm(cfg.width)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, cfg.mlp_dim), nn.GELU(approximate="tanh"), nn.Linear(cfg.mlp_dim, cfg.width))

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class SigLIP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        g = cfg.image_size // cfg.patch_size  # 16
        self.embedding = nn.Conv2d(3, cfg.width, cfg.patch_size, stride=cfg.patch_size)  # "stem", siglip.py L216-L223
        self.pos_embedding = nn.Parameter(torch.randn(1, g * g, cfg.width) / math.sqrt(cfg.width))  # learned, L40-L47
        self.encoderblock = nn.ModuleList(ViTBlock(cfg) for _ in range(cfg.depth))
        self.encoder_norm = nn.LayerNorm(cfg.width)  # L161
        self.head = nn.Linear(cfg.width, cfg.out_dim)  # L284-L288; pool_type="none" so every token goes through it
        nn.init.zeros_(self.head.weight)  # head_zeroinit=True (PaliGemma: "zero initialized linear projection")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """float32[B, H, W, 3] in [-1,1] -> float32[B, (H/14)*(W/14), out_dim]. No pooling, no CLS token."""
        x = self.embedding(images.permute(0, 3, 1, 2))  # [B, width, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 256, width], row-major over the patch grid (L225-L226)
        x = x + self.pos_embedding
        for blk in self.encoderblock:
            x = blk(x)
        return self.head(self.encoder_norm(x))


# ======================================================================================
# 2. Gemma building blocks. openpi@215abfb gemma.py.
# ======================================================================================
class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + 1e-6) * (1 + scale), scale zero-init, statistics in float32. gemma.py L113-L125."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + 1e-6) * (1 + self.scale.float())).to(x.dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0) -> torch.Tensor:
    """RoPE on x[B, L, N, H] with integer positions[B, L]; pairs dim i with i + H/2 (split, not interleave).
    gemma.py L424-L440."""
    h = x.shape[-1]
    exps = (2.0 / h) * torch.arange(h // 2, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**exps
    radians = positions.float()[..., None] / timescale  # [B, L, H/2]
    radians = radians[..., None, :]  # [B, L, 1, H/2]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.float().split(h // 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class ExpertAttnProj(nn.Module):
    """One expert's q / kv / out projections (no biases). Kept separate from the attention core so that
    several experts can project their own tokens and then attend jointly (gemma.py L172-L201, L233-L247)."""

    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.q_einsum = nn.Linear(cfg.width, cfg.num_heads * cfg.head_dim, bias=False)
        self.kv_einsum = nn.Linear(cfg.width, 2 * cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.attn_vec_einsum = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.width, bias=False)

    def qkv(self, x):
        b, t, _ = x.shape
        q = self.q_einsum(x).view(b, t, self.cfg.num_heads, self.cfg.head_dim)
        k, v = self.kv_einsum(x).view(b, t, 2, self.cfg.num_kv_heads, self.cfg.head_dim).unbind(2)
        return q, k, v

    def out(self, encoded):  # encoded [B, T, N, H] -> [B, T, width]
        return self.attn_vec_einsum(encoded.flatten(2))


BIG_NEG = -2.3819763e38  # gemma.py L225, "see gemma/modules.py"


def attend(q, k, v, mask, kv_cache=None):
    """Grouped-query attention core. q[B,T,N,H], k,v[B,S,K,H], mask bool[B,T,S_total]; N = K*G.
    RoPE is applied to q and k of the *new* tokens before concatenating the cache (gemma.py L203-L231)."""
    if kv_cache is not None:
        k = torch.cat([kv_cache[0], k], dim=1)
        v = torch.cat([kv_cache[1], v], dim=1)
    b, t, n, h = q.shape
    kh = k.shape[2]
    qg = q.view(b, t, kh, n // kh, h)
    logits = torch.einsum("btkgh,bskh->bkgts", qg, k).float()
    logits = torch.where(mask[:, None, None, :, :], logits, torch.full_like(logits, BIG_NEG))
    probs = torch.softmax(logits, dim=-1).to(v.dtype)
    enc = torch.einsum("bkgts,bskh->btkgh", probs, v).reshape(b, t, n, h)
    return enc, (k, v)


class GeGLU(nn.Module):
    """gelu(x W_gate) * (x W_up) W_down, no biases. gemma.py L253-L280."""

    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.gating_einsum = nn.Parameter(torch.empty(2, cfg.width, cfg.mlp_dim))
        self.linear = nn.Parameter(torch.empty(cfg.mlp_dim, cfg.width))
        nn.init.normal_(self.gating_einsum, std=cfg.width**-0.5)  # lecun_normal
        nn.init.normal_(self.linear, std=cfg.mlp_dim**-0.5)

    def forward(self, x):
        gate = F.gelu(x @ self.gating_einsum[0], approximate="tanh")  # flax nn.gelu default is the tanh approximation
        return (gate * (x @ self.gating_einsum[1])) @ self.linear


class GemmaBlock(nn.Module):
    """Single-expert pre-norm block: x += attn(norm(x)); x += mlp(norm(x)). gemma.py L284-L333 with one expert."""

    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_attention_norm = RMSNorm(cfg.width)
        self.attn = ExpertAttnProj(cfg)
        self.pre_ffw_norm = RMSNorm(cfg.width)
        self.mlp = GeGLU(cfg)

    def forward(self, x, positions, mask, kv_cache=None):
        q, k, v = self.attn.qkv(self.pre_attention_norm(x))
        q = apply_rope(q, positions) * self.cfg.head_dim**-0.5
        k = apply_rope(k, positions)
        enc, kv = attend(q, k, v, mask, kv_cache)
        x = x + self.attn.out(enc)
        x = x + self.mlp(self.pre_ffw_norm(x))
        return x, kv


class Embedder(nn.Module):
    """Token embedding table; encode scales by sqrt(width) (gemma.py L135-L154). decode (tied logits) is unused by pi0."""

    def __init__(self, cfg: GemmaConfig):
        super().__init__()
        self.input_embedding = nn.Parameter(torch.randn(cfg.vocab_size, cfg.width))
        self.width = cfg.width

    def encode(self, tokens):
        return self.input_embedding[tokens] * math.sqrt(self.width)


class Gemma(nn.Module):
    """Decoder stack. embed: int64[B, L] -> float32[B, L, width].
    forward: x[B, T, width], positions int64[B, T], mask bool[B, T, S] (S = cached + T), optional kv_cache
    -> (float32[B, T, width] after final RMSNorm, kv_cache: list over layers of (k, v) each [B, S, K, H]).
    gemma.py L340-L411."""

    def __init__(self, cfg: GemmaConfig, with_embedder: bool = True):
        super().__init__()
        self.cfg = cfg
        self.embedder = Embedder(cfg) if with_embedder else None
        self.layers = nn.ModuleList(GemmaBlock(cfg) for _ in range(cfg.depth))
        self.final_norm = RMSNorm(cfg.width)

    def embed(self, tokens):
        return self.embedder.encode(tokens)

    def forward(self, x, positions, mask, kv_cache=None):
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, positions, mask, None if kv_cache is None else kv_cache[i])
            new_cache.append(kv)
        return self.final_norm(x), new_cache


# ======================================================================================
# 3. Attention mask from a per-token "starts a new block" flag. openpi@215abfb pi0.py L19-L45 (from big_vision).
# ======================================================================================
def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """input_mask bool[B, N] (False = padding), mask_ar bool[N] or [B, N] (True = this token cannot be seen by
    earlier tokens, i.e. it opens a new block) -> bool[B, N, N], True where query i may attend key j.
    Token i attends j iff cumsum(mask_ar)[j] <= cumsum(mask_ar)[i] and both are valid.
    [0 0 0 1 1 1] = prefix-LM; [1 1 1 1] = causal; [0 0 1 0 1 0] = three bidirectional blocks in causal order.

    Logic: cumsum(mask_ar) assigns every token a block id; blocks are causal w.r.t. each other, tokens inside a
    block are bidirectional. Worked example with 6 tokens [img0 img1 txt | state | act0 act1]:
        mask_ar = [0 0 0 1 1 0]  ->  block id = [0 0 0 1 2 2]
        query\key   img0 img1 txt state act0 act1
        img0  (b0)    1    1    1    0    0    0
        img1  (b0)    1    1    1    0    0    0
        txt   (b0)    1    1    1    0    0    0
        state (b1)    1    1    1    1    0    0
        act0  (b2)    1    1    1    1    1    1
        act1  (b2)    1    1    1    1    1    1
    The prefix sees only itself, state sees prefix + itself, actions see everything and each other: the three
    blocks of pi0 Appendix B. If txt were padding (input_mask False) its row and column would be all 0.
    Real pi0: mask_ar has 768 image + 48 text zeros, then 1 (state), 1 (first action), 49 zeros.

    Why this encoding (from big_vision, used for PaliGemma's prefix-LM): the prefix and suffix each return their
    own 1-D ar_mask, a concat + one call gives the full 2-D mask; and because block ids depend only on cumsum,
    at inference the suffix rows can be built against the cached prefix keys (pi0.py sample_actions,
    `full_attn_mask`), so the KV cache and the mask agree by construction."""
    mask_ar = mask_ar.to(torch.long).expand(input_mask.shape)
    c = mask_ar.cumsum(dim=1)
    attn = c[:, None, :] <= c[:, :, None]
    valid = input_mask[:, None, :] & input_mask[:, :, None]
    return attn & valid


# ======================================================================================
# 4. PaliGemma = SigLIP + Gemma over the pi0 "prefix" (images + prompt). pi0.py L69-L138.
# ======================================================================================
class PaliGemma(nn.Module):
    def __init__(self, vit_cfg: ViTConfig = SIGLIP_SO400M_14, gemma_cfg: GemmaConfig = GEMMA_2B):
        super().__init__()
        assert vit_cfg.out_dim == gemma_cfg.width
        self.img = SigLIP(vit_cfg)
        self.llm = Gemma(gemma_cfg)

    def embed_prefix(self, images, image_masks, tokens, token_mask):
        """images {key: f32[B,224,224,3]}, image_masks {key: bool[B]}, tokens i64[B,48], token_mask bool[B,48]
        -> (emb f32[B, 3*256+48, width], input_mask bool[B, S], ar_mask bool[S]).
        Order: images in IMAGE_KEYS order, then prompt. Every prefix token has ar_mask False: images and text
        attend to each other bidirectionally (PaliGemma prefix-LM). pi0.py L106-L137."""
        embs, masks, ar = [], [], []
        for k in IMAGE_KEYS:
            t = self.img(images[k])  # [B, 256, width]
            embs.append(t)
            masks.append(image_masks[k][:, None].expand(-1, t.shape[1]))
            ar.append(torch.zeros(t.shape[1], dtype=torch.bool, device=t.device))
        e = self.llm.embed(tokens)
        embs.append(e)
        masks.append(token_mask)
        ar.append(torch.zeros(e.shape[1], dtype=torch.bool, device=e.device))
        return torch.cat(embs, 1), torch.cat(masks, 1), torch.cat(ar, 0)

    def forward(self, images, image_masks, tokens, token_mask):
        """Prefix forward as done once per action chunk at inference (pi0.py L233-L237): returns hidden states
        f32[B, S, width] and the per-layer KV cache the action expert will attend into."""
        emb, input_mask, ar_mask = self.embed_prefix(images, image_masks, tokens, token_mask)
        mask = make_attn_mask(input_mask, ar_mask)
        positions = input_mask.long().cumsum(1) - 1  # padded tokens consume no positions (pi0.py L208)
        return self.llm(emb, positions, mask)


# ======================================================================================
# 5. Walk one prefix forward with the tiny config and print every intermediate shape.
#    uv run python -m pi.pi0.vlm.model
# ======================================================================================
def main():
    torch.manual_seed(0)
    B = 2
    vit_cfg, gemma_cfg = tiny_vit(), tiny_gemma()
    pg = PaliGemma(vit_cfg, gemma_cfg).eval()
    print(f"tiny configs: vit={vit_cfg}\n              gemma={gemma_cfg}")
    print(f"params: SigLIP {sum(p.numel() for p in pg.img.parameters()):,}  Gemma {sum(p.numel() for p in pg.llm.parameters()):,}")

    # --- inputs: what ../data produces (no state here; state goes to the action expert) ---
    images = {k: torch.rand(B, 224, 224, 3) * 2 - 1 for k in IMAGE_KEYS}
    image_masks = {k: torch.ones(B, dtype=torch.bool) for k in IMAGE_KEYS}
    image_masks["right_wrist_0_rgb"][:] = False  # pretend this robot has no right wrist camera
    tokens = torch.randint(3, gemma_cfg.vocab_size, (B, 48))
    token_mask = torch.zeros(B, 48, dtype=torch.bool)
    token_mask[:, :9] = True  # a 9-token prompt, rest is padding
    print("\n[input]  images x3 f32", tuple(images["base_0_rgb"].shape), " image_masks bool", tuple(image_masks["base_0_rgb"].shape),
          " tokens i64", tuple(tokens.shape), " token_mask", tuple(token_mask.shape))

    # Steps 1-3 below are PaliGemma.forward unrolled by hand, so every intermediate tensor can be printed:
    #   1 = SigLIP.forward (one camera), 2 = the first three lines of PaliGemma.forward, 3 = Gemma.forward.
    # Step 4 calls the real PaliGemma.forward and asserts it gives the same tensor as the unrolled walk.
    # torch.no_grad() only stops autograd from recording a graph (no backward here); it changes no math.
    with torch.no_grad():
        # --- 1. SigLIP, one camera at a time (shared weights) ---
        x = pg.img.embedding(images["base_0_rgb"].permute(0, 3, 1, 2))
        print("[siglip] patch conv           ", tuple(x.shape), " = [B, width, 16, 16]")
        x = x.flatten(2).transpose(1, 2) + pg.img.pos_embedding
        print("[siglip] tokens + pos_embedding", tuple(x.shape), " = [B, 256, width]")
        for blk in pg.img.encoderblock:
            x = blk(x)
        x = pg.img.head(pg.img.encoder_norm(x))
        print("[siglip] after blocks/norm/head", tuple(x.shape), " = [B, 256, gemma width]")

        # --- 2. prefix assembly ---
        emb, input_mask, ar_mask = pg.embed_prefix(images, image_masks, tokens, token_mask)
        print("[prefix] emb", tuple(emb.shape), " input_mask", tuple(input_mask.shape), " ar_mask", tuple(ar_mask.shape))
        print(f"[prefix] valid tokens per sample: {int(input_mask[0].sum())} of {input_mask.shape[1]} "
              f"(2 cameras x 256 + 9 prompt); ar_mask any? {bool(ar_mask.any())}")
        mask = make_attn_mask(input_mask, ar_mask)
        positions = input_mask.long().cumsum(1) - 1
        print("[prefix] attn mask", tuple(mask.shape), f" fraction attendable {mask[0].float().mean():.3f}")
        print("[prefix] positions of first prompt token / last valid token:",
              int(positions[0, 768]), "/", int(positions[0, input_mask[0].nonzero().max()]))

        # --- 3. Gemma layer by layer ---
        h = emb
        kv_cache = []
        for i, layer in enumerate(pg.llm.layers):
            h, kv = layer(h, positions, mask)
            kv_cache.append(kv)
            if i == 0:
                print("[gemma]  layer 0 out", tuple(h.shape), " k/v cache", tuple(kv[0].shape), " = [B, S, kv_heads, head_dim]")
        h = pg.llm.final_norm(h)
        print(f"[gemma]  after {len(pg.llm.layers)} layers + final_norm", tuple(h.shape))

        # --- 4. same thing through the public entry point ---
        hidden, cache = pg(images, image_masks, tokens, token_mask)
        assert torch.allclose(hidden, h) and len(cache) == gemma_cfg.depth
    print("[output] hidden", tuple(hidden.shape), " kv_cache: list of", len(cache), "x (k, v)", tuple(cache[0][0].shape))
    print("\nthe action expert will attend into this kv_cache; see ../action_expert")


if __name__ == "__main__":
    main()
