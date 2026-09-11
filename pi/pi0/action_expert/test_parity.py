"""Parameter-count and shape parity checks for the pi0 action expert and the two-expert Gemma.

References: openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 (src/openpi/models/pi0.py, gemma.py, pi0_config.py),
pi0 paper arXiv:2410.24164v1 Appendix B. Run on CPU: `uv run pytest pi/pi0/action_expert -q`.
Paper-size models are built on the `meta` device, so no 300M-parameter tensors are allocated.
"""

import torch

from pi.pi0.action_expert import model as M
from pi.pi0.vlm import model as V

B = 2
P = 3 * 256 + 48  # prefix length
S = 1 + M.ACTION_HORIZON  # suffix length


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- constants
def test_paper_configs_match_openpi():
    assert M.PI0_EXPERTS == (V.GEMMA_2B, V.GEMMA_300M)  # pi0_config.py L21-L22
    e = V.GEMMA_300M
    assert (e.width, e.depth, e.mlp_dim, e.num_heads, e.num_kv_heads, e.head_dim) == (1024, 18, 4096, 8, 1, 256)
    assert (M.ACTION_DIM, M.ACTION_HORIZON) == (32, 50)
    # experts may differ in width and mlp_dim but must share the head layout (gemma.py L165-L168)
    assert (V.GEMMA_2B.num_heads, V.GEMMA_2B.num_kv_heads, V.GEMMA_2B.head_dim) == (e.num_heads, e.num_kv_heads, e.head_dim)
    assert V.GEMMA_2B.width != e.width and V.GEMMA_2B.mlp_dim != e.mlp_dim


# ---------------------------------------------------------------- parameter counts
def test_action_expert_gemma_300m_param_count():
    with torch.device("meta"):
        llm = M.MoEGemma(M.PI0_EXPERTS)
    expert1 = sum(n_params(l.experts[1]) for l in llm.layers) + n_params(llm.final_norms[1])
    per_layer = 1024 * 2048 + 1024 * 2 * 256 + 2048 * 1024 + 2 * 1024 + (2 * 1024 * 4096 + 4096 * 1024)
    assert per_layer == 17_303_552
    assert expert1 == 18 * per_layer + 1024 == 311_464_960  # gemma.py L70 comments "311M params"
    assert round(expert1 / 1e6) == 311
    # expert 0 is Gemma 2B minus the vocabulary embedding, exactly as in ../vlm (Gemma report Table 2)
    expert0 = sum(n_params(l.experts[0]) for l in llm.layers) + n_params(llm.final_norms[0])
    assert expert0 == 1_981_884_416


def test_projection_param_count_and_total():
    with torch.device("meta"):
        ae = M.ActionProjections()
    w, d = 1024, 32
    assert n_params(ae.state_proj) == d * w + w == 33_792
    assert n_params(ae.action_in_proj) == 33_792
    assert n_params(ae.action_time_mlp_in) == 2 * w * w + w == 2_098_176
    assert n_params(ae.action_time_mlp_out) == w * w + w == 1_049_600
    assert n_params(ae.action_out_proj) == w * d + d == 32_800
    assert n_params(ae) == 3_248_160
    # whole pi0: VLM (SigLIP + Gemma 2B incl. embedding, ../vlm) + expert + projections; paper says "3.3 billion"
    assert 2_923_335_408 + 311_464_960 + 3_248_160 == 3_238_048_528


# ---------------------------------------------------------------- suffix contract
def test_embed_suffix_shapes_and_ar_mask():
    cfg = M.tiny_expert()
    ae = M.ActionProjections(cfg)
    tokens, input_mask, ar = ae.embed_suffix(torch.randn(B, 32), torch.randn(B, 50, 32), torch.rand(B))
    assert tokens.shape == (B, S, cfg.width)
    assert input_mask.shape == (B, S) and input_mask.all()
    assert ar.tolist() == [True, True] + [False] * 49  # pi0.py L157, L182
    assert ae.decode(tokens).shape == (B, 50, 32)


def test_posemb_sincos():
    e = M.posemb_sincos(torch.zeros(3), 8, 4e-3, 4.0)
    assert e.shape == (3, 8)
    torch.testing.assert_close(e[:, :4], torch.zeros(3, 4))  # sin(0)
    torch.testing.assert_close(e[:, 4:], torch.ones(3, 4))  # cos(0)
    assert not torch.allclose(M.posemb_sincos(torch.tensor([0.3]), 8, 4e-3, 4.0), M.posemb_sincos(torch.tensor([0.7]), 8, 4e-3, 4.0))


def test_timestep_changes_action_tokens_not_state_token():
    ae = M.ActionProjections(M.tiny_expert())
    st, act = torch.randn(B, 32), torch.randn(B, 50, 32)
    t1, _, _ = ae.embed_suffix(st, act, torch.full((B,), 0.2))
    t2, _, _ = ae.embed_suffix(st, act, torch.full((B,), 0.8))
    torch.testing.assert_close(t1[:, 0], t2[:, 0])  # state token has no timestep dependence
    assert not torch.allclose(t1[:, 1:], t2[:, 1:])


def test_full_mask_has_three_blocks():
    """Appendix B: [images, prompt] | [state] | [actions]; blocks causal, bidirectional inside."""
    prefix_ar = torch.zeros(P, dtype=torch.bool)
    _, _, suffix_ar = M.ActionProjections(M.tiny_expert()).embed_suffix(torch.zeros(1, 32), torch.zeros(1, 50, 32), torch.zeros(1))
    ar = torch.cat([prefix_ar, suffix_ar])
    m = M.make_attn_mask(torch.ones(1, P + S, dtype=torch.bool), ar)[0]
    assert m[:P, :P].all() and not m[:P, P:].any()  # prefix sees only prefix
    assert m[P, : P + 1].all() and not m[P, P + 1 :].any()  # state sees prefix + itself, not actions
    assert m[P + 1 :, :].all()  # actions see everything, bidirectional among themselves


# ---------------------------------------------------------------- semantics
def test_moe_with_only_expert0_equals_single_expert_gemma():
    """Two-expert stack restricted to expert 0 must reproduce ../vlm's Gemma exactly (same weights)."""
    cfg = M.tiny_gemma()
    g = V.Gemma(cfg, with_embedder=False).eval()
    llm = M.MoEGemma((cfg, M.tiny_expert())).eval()
    for lm, lg in zip(llm.layers, g.layers):
        lm.experts[0].load_state_dict(lg.state_dict())
    llm.final_norms[0].load_state_dict(g.final_norm.state_dict())
    x = torch.randn(B, 10, cfg.width)
    pos = torch.arange(10).expand(B, 10)
    mask = M.make_attn_mask(torch.ones(B, 10, dtype=torch.bool), torch.zeros(10, dtype=torch.bool))
    y_ref, kv_ref = g(x, pos, mask)
    (y, none), kv = llm([x, None], pos, mask)
    assert none is None
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(kv[0][0], kv_ref[0][0])


def test_cached_suffix_forward_equals_joint_forward():
    """Inference path (prefix cache + suffix only) must give the same suffix output as the training path,
    including with padding in the prefix (a masked camera and a short prompt)."""
    torch.manual_seed(0)
    vlm_cfg, exp_cfg = M.tiny_experts()
    llm = M.MoEGemma((vlm_cfg, exp_cfg)).eval()
    ae = M.ActionProjections(exp_cfg).eval()
    prefix_emb = torch.randn(B, P, vlm_cfg.width)
    prefix_mask = torch.ones(B, P, dtype=torch.bool)
    prefix_mask[:, 512:768] = False  # right wrist camera missing
    prefix_mask[:, 768 + 9 :] = False  # 9-token prompt
    prefix_ar = torch.zeros(P, dtype=torch.bool)
    suffix_emb, suffix_mask, suffix_ar = ae.embed_suffix(torch.randn(B, 32), torch.randn(B, 50, 32), torch.rand(B))
    with torch.no_grad():
        p_out, s_joint = M.joint_forward(llm, prefix_emb, prefix_mask, prefix_ar, suffix_emb, suffix_mask, suffix_ar)
        assert p_out.shape == (B, P, vlm_cfg.width) and s_joint.shape == (B, S, exp_cfg.width)
        (_, none), kv = llm([prefix_emb, None], prefix_mask.long().cumsum(1) - 1, M.make_attn_mask(prefix_mask, prefix_ar))
        assert none is None and kv[0][0].shape == (B, P, exp_cfg.num_kv_heads, exp_cfg.head_dim)
        s_cached = M.suffix_forward(llm, kv, prefix_mask, suffix_emb, suffix_mask, suffix_ar)
    torch.testing.assert_close(s_cached, s_joint, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ae.decode(s_cached), ae.decode(s_joint), atol=1e-5, rtol=1e-5)


def test_experts_do_not_mix_outside_attention():
    """With a mask that forbids cross-expert attention, expert 1's output must not depend on expert 0's tokens."""
    vlm_cfg, exp_cfg = M.tiny_experts()
    llm = M.MoEGemma((vlm_cfg, exp_cfg)).eval()
    x0a, x0b = torch.randn(B, 5, vlm_cfg.width), torch.randn(B, 5, vlm_cfg.width)
    x1 = torch.randn(B, 3, exp_cfg.width)
    pos = torch.arange(8).expand(B, 8)
    mask = torch.zeros(B, 8, 8, dtype=torch.bool)
    mask[:, :5, :5] = True
    mask[:, 5:, 5:] = True  # two isolated blocks
    with torch.no_grad():
        (_, ya), _ = llm([x0a, x1], pos, mask)
        (_, yb), _ = llm([x0b, x1], pos, mask)
    torch.testing.assert_close(ya, yb)
