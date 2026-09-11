"""Parameter-count and shape parity checks for the pi0 VLM backbone (SigLIP So400m/14 + Gemma 2B).

References: openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 (src/openpi/models/siglip.py, gemma.py, pi0.py),
Gemma technical report arXiv:2403.08295v4 Table 1-2. Run on CPU: `uv run pytest pi/pi0/vlm -q`.
Paper-size models are built on the `meta` device, so no 3B-parameter tensors are allocated.
"""

import torch

from pi.pi0.vlm import model as M

B = 2


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- constants (openpi gemma.py L79-L87, siglip.py L318-L363)
def test_paper_configs_match_openpi():
    assert (M.GEMMA_2B.width, M.GEMMA_2B.depth, M.GEMMA_2B.mlp_dim) == (2048, 18, 16384)
    assert (M.GEMMA_2B.num_heads, M.GEMMA_2B.num_kv_heads, M.GEMMA_2B.head_dim) == (8, 1, 256)
    assert M.GEMMA_2B.vocab_size == 257_152
    v = M.SIGLIP_SO400M_14
    assert (v.width, v.depth, v.mlp_dim, v.num_heads, v.patch_size, v.image_size) == (1152, 27, 4304, 16, 14, 224)
    assert v.out_dim == M.GEMMA_2B.width  # SigLIP head projects straight into Gemma's width (pi0.py L82)


# ---------------------------------------------------------------- parameter counts
def test_gemma_2b_param_count_matches_gemma_report():
    with torch.device("meta"):
        g = M.Gemma(M.GEMMA_2B)
    embed = g.embedder.input_embedding.numel()
    non_embed = n_params(g) - embed
    # Gemma report Table 2: 2B non-embedding parameters = 1,981,884,416. Exact match.
    assert non_embed == 1_981_884_416
    # PaliGemma extends Gemma's 256,128 vocab to 257,152 (1024 <loc> + 128 <seg> tokens, PaliGemma paper Sec. 4.x)
    assert embed == 257_152 * 2048


def test_siglip_so400m_param_count():
    with torch.device("meta"):
        s = M.SigLIP(M.SIGLIP_SO400M_14)
    # Analytic count from the openpi variant table (siglip.py): 27 x 15,239,504 per block + stem + posemb + norm + head.
    assert n_params(s) == 414_803_696


def test_gemma_300m_param_count_is_311m():
    with torch.device("meta"):
        g = M.Gemma(M.GEMMA_300M, with_embedder=False)  # the action expert has no vocabulary embedding
    # openpi gemma.py L70 comments "311M params" for gemma_300m.
    assert round(n_params(g) / 1e6) == 311


# ---------------------------------------------------------------- shapes, tiny configs
def test_siglip_output_shape_and_token_grid():
    s = M.SigLIP(M.tiny_vit())
    imgs = torch.rand(B, 224, 224, 3) * 2 - 1
    out = s(imgs)
    assert out.shape == (B, 256, M.tiny_gemma().width)  # 224/14 = 16 -> 16*16 = 256 tokens
    assert s.pos_embedding.shape == (1, 256, M.tiny_vit().width)


def test_gemma_embed_is_scaled_by_sqrt_width():
    g = M.Gemma(M.tiny_gemma())
    tok = torch.tensor([[5, 7]])
    e = g.embed(tok)
    torch.testing.assert_close(e[0, 0], g.embedder.input_embedding[5] * g.cfg.width**0.5)


def test_rmsnorm_zero_init_is_identity_scale():
    n = M.RMSNorm(8)
    x = torch.randn(3, 8) * 5
    y = n(x)
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(3), atol=1e-4, rtol=1e-4)


def test_rope_position_zero_is_identity():
    x = torch.randn(B, 4, 2, 16)
    torch.testing.assert_close(M.apply_rope(x, torch.zeros(B, 4, dtype=torch.long)), x)
    assert not torch.allclose(M.apply_rope(x, torch.ones(B, 4, dtype=torch.long)), x)


def test_make_attn_mask_blocks():
    input_mask = torch.ones(1, 6, dtype=torch.bool)
    ar = torch.tensor([False, False, False, True, False, True])  # blocks: [0,1,2] [3,4] [5]
    m = M.make_attn_mask(input_mask, ar)[0]
    assert m[0, 2] and m[2, 0]  # bidirectional inside block 0
    assert m[3, 0] and not m[0, 3]  # later block sees earlier, not vice versa
    assert m[4, 3] and m[3, 4]  # bidirectional inside block 1
    assert m[5, 4] and not m[4, 5]


def test_gemma_forward_shapes_and_kv_cache():
    cfg = M.tiny_gemma()
    g = M.Gemma(cfg)
    x = torch.randn(B, 10, cfg.width)
    mask = torch.ones(B, 10, 10, dtype=torch.bool)
    pos = torch.arange(10).expand(B, 10)
    y, kv = g(x, pos, mask)
    assert y.shape == (B, 10, cfg.width)
    assert len(kv) == cfg.depth
    k, v = kv[0]
    assert k.shape == (B, 10, cfg.num_kv_heads, cfg.head_dim) and v.shape == k.shape
    # continue with cache: 3 new tokens attend to 10 cached + themselves
    x2 = torch.randn(B, 3, cfg.width)
    mask2 = torch.ones(B, 3, 13, dtype=torch.bool)
    pos2 = torch.arange(10, 13).expand(B, 3)
    y2, kv2 = g(x2, pos2, mask2, kv_cache=kv)
    assert y2.shape == (B, 3, cfg.width) and kv2[0][0].shape == (B, 13, cfg.num_kv_heads, cfg.head_dim)


def test_padded_tokens_do_not_change_valid_outputs():
    """Masked-out tokens (image_mask False or prompt padding) must be invisible: same output as if absent."""
    cfg = M.tiny_gemma()
    g = M.Gemma(cfg).eval()
    x = torch.randn(1, 6, cfg.width)
    full_mask = torch.ones(1, 6, dtype=torch.bool)
    ar = torch.zeros(6, dtype=torch.bool)
    y_full, _ = g(x, torch.arange(6)[None], M.make_attn_mask(full_mask, ar))
    # drop tokens 2 and 3 via input_mask; positions follow cumsum so token 4 gets position 2
    im = torch.tensor([[True, True, False, False, True, True]])
    pos = im.long().cumsum(1) - 1
    y_masked, _ = g(x, pos, M.make_attn_mask(im, ar))
    y_ref, _ = g(x[:, [0, 1, 4, 5]], torch.arange(4)[None], M.make_attn_mask(torch.ones(1, 4, dtype=torch.bool), ar[:4]))
    torch.testing.assert_close(y_masked[:, [0, 1, 4, 5]], y_ref, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(y_masked[:, [0, 1, 4, 5]], y_full[:, [0, 1, 4, 5]])


def test_paligemma_prefix_contract():
    pg = M.PaliGemma(M.tiny_vit(), M.tiny_gemma())
    images = {k: torch.rand(B, 224, 224, 3) * 2 - 1 for k in M.IMAGE_KEYS}
    image_masks = {k: torch.ones(B, dtype=torch.bool) for k in M.IMAGE_KEYS}
    image_masks["right_wrist_0_rgb"] = torch.zeros(B, dtype=torch.bool)
    tokens = torch.randint(0, 100, (B, 48))
    token_mask = torch.zeros(B, 48, dtype=torch.bool)
    token_mask[:, :7] = True
    emb, input_mask, ar_mask = pg.embed_prefix(images, image_masks, tokens, token_mask)
    S = 3 * 256 + 48
    assert emb.shape == (B, S, pg.llm.cfg.width)
    assert input_mask.shape == (B, S) and ar_mask.shape == (S,)
    assert input_mask[:, 512:768].sum() == 0  # masked camera -> its 256 tokens are padding
    assert input_mask[:, 768:].sum() == B * 7
    assert not ar_mask.any()  # the whole prefix is one bidirectional block (pi0.py L121, L131)
    hidden, kv = pg(images, image_masks, tokens, token_mask)
    assert hidden.shape == (B, S, pg.llm.cfg.width) and len(kv) == pg.llm.cfg.depth
