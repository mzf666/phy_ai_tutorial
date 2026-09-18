"""Alignment checks for the pi0.6 backbone: Gemma 3 4B parameter counts against the report, the L L L L L G layer pattern,
the sliding-window and global-RoPE rules, QK-norm at init, the images-bidirectional / text-causal / expert-visibility mask,
KV-cache consistency through a sliding layer, the expert's identity at init, the insulate flag, the 5-step sampler and
the decoder's stop rule."""

import numpy as np
import pytest
import torch

import pi.pi06.backbone.model as M
from pi.fast.data.data import EOS_ID
from pi.pi06.data.data import SEG_ACTION, STATIC_IMAGE_KEYS, build_pi06_batch, tiny_pi06_tokenizer, unit_stats

B, H, D = 2, 10, 7


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = M.tiny_pi06().eval()
    seq = tiny_pi06_tokenizer(H, D)
    rng = np.random.default_rng(0)
    raw = {"images": {k: rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
           "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32), "prompt": ["make a double espresso", "fold the shirt"]}
    obs, _ = build_pi06_batch(raw, unit_stats(D), seq, layout="flow", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False,
                              subtasks=["grab the portafilter", "grab the collar"], advantages=[True, None])
    return model, seq, raw, obs


# ---------------------------------------------------------------- parameter counts (report Table 1)
def test_gemma3_4b_param_counts_match_the_report():
    with torch.device("meta"):
        stack = M.Gemma3Stack((M.GEMMA3_4B,))
        emb = M.Embedder(M.GEMMA3_4B)
    pc = M.backbone_param_count()
    assert n_params(stack) == pc["non_embedding"] == 3_209_010_688  # report: 3,209M non-embedding
    assert n_params(emb) == pc["embedding"] == 262_144 * 2560  # 671M; + 1152 x 2560 mm projection = 674M ~ report 675M
    assert M.GEMMA3_4B.depth == 34 and M.GEMMA3_4B.mlp_dim == 10_240 and (M.GEMMA3_4B.num_heads, M.GEMMA3_4B.num_kv_heads, M.GEMMA3_4B.head_dim) == (8, 4, 256)


def test_siglip_400m_at_448():
    with torch.device("meta"):
        v = M.VisionEmbed(M.SIGLIP_400M_448, 2560)
    n = n_params(v.img)
    assert abs(n - 417e6) / 417e6 < 0.01  # report Table 1: 417M vision encoder (pos-embedding grid differs: 1024 vs 4096 at 896)
    assert n_params(v.mm_input_projection) == 1152 * 2560


def test_expert_size_candidates():
    cands = M.expert_width_candidates()
    assert cands == [(1024, 4096, M.expert_param_count(M.expert_config(1024, 4096)))]  # the only width (mult. of 64, mlp 4x) within 3% of 860M
    assert abs(cands[0][2] - 860e6) / 860e6 < 0.01
    with torch.device("meta"):
        st = M.Gemma3Stack((M.GEMMA3_4B, M.expert_config(1024, 4096)))
    e1 = sum(n_params(l.experts[1]) for l in st.layers) + n_params(st.final_norms[1])
    assert e1 == cands[0][2]


# ---------------------------------------------------------------- layer pattern, rope, window, qk-norm
def test_layer_pattern_and_rope():
    g = [i for i in range(M.GEMMA3_4B.depth) if M.GEMMA3_4B.is_global(i)]
    assert g == [5, 11, 17, 23, 29]  # pattern L L L L L G repeated, 34 % 6 = 4 trailing locals
    pos = torch.arange(16)[None]
    p, base = M.rope_positions(pos, M.GEMMA3_4B, 5)
    assert base == 1e6 and torch.allclose(p, pos.float() / 8)
    p, base = M.rope_positions(pos, M.GEMMA3_4B, 0)
    assert base == 1e4 and torch.allclose(p, pos.float())


def test_sliding_mask_matches_brute_force():
    pq = torch.tensor([[0, 1, 2, 5, 9, 20]])
    pk = torch.tensor([[0, 1, 2, 3, 4, 5, 9, 12, 20]])
    m = M.sliding_mask(pq, pk, 4)
    ref = torch.tensor([[[abs(int(a) - int(b)) < 4 for b in pk[0]] for a in pq[0]]])
    assert torch.equal(m, ref)


def test_qk_norm_is_active_at_init():
    proj = M.Gemma3AttnProj(M.tiny_gemma3())
    q, k, _ = proj.qkv(torch.randn(1, 5, 64) * 7)
    assert torch.allclose(q.pow(2).mean(-1), torch.ones(1, 5, 4), atol=1e-4) and torch.allclose(k.pow(2).mean(-1), torch.ones(1, 5, 2), atol=1e-4)


# ---------------------------------------------------------------- the mask
def test_prefix_mask_rules():
    tm = torch.tensor([[True, True, True, False], [True, True, False, False]])
    vis = torch.tensor([[True, False, True, False], [True, True, False, False]])
    m = M.make_pi06_mask(3, tm, n_expert=2, expert_visible=vis)
    assert m.shape == (2, 9, 9)
    assert m[0, :3, :3].all() and not m[0, :3, 3:].any()  # images: bidirectional among images, blind to text and expert
    assert m[0, 3:6, :3].all()  # valid text sees all images
    assert m[0, 4, 3:7].tolist() == [True, True, False, False] and m[0, 5, 3:7].tolist() == [True, True, True, False]  # causal, padding out
    assert not m[0, 6].any()  # a padding row attends nothing
    assert m[0, 7, 3:7].tolist() == [True, False, True, False] and m[0, 7:, :3].all() and m[0, 7:, 7:].all()  # expert: visible text + images + itself
    assert not m[0, :7, 7:].any()  # nobody sees the expert
    assert m[1, 8, 3:7].tolist() == [True, True, False, False]


# ---------------------------------------------------------------- forward, cache, expert, sampler, decoder
def test_kv_cache_matches_full_forward_through_sliding_layers(setup):
    model, _, _, obs = setup
    emb, valid, n_img = model.embed_prefix(obs)
    mask = model.prefix_mask(obs, n_img)
    pos = valid.long().cumsum(1) - 1
    with torch.no_grad():
        (h_full, _), _ = model.llm([emb, None], pos, mask)
        cut = n_img + 40  # split inside the text, after the window has filled (window 8)
        (h_a, _), cache = model.llm([emb[:, :cut], None], pos[:, :cut], mask[:, :cut, :cut])
        (h_b, _), _ = model.llm([emb[:, cut:], None], pos[:, cut:], mask[:, cut:, :], cache)
    both = torch.cat([h_a, h_b], 1)
    assert torch.allclose(both[valid], h_full[valid], atol=1e-4)  # invalid rows (masked camera, padding) attend nothing and are garbage in both


def test_expert_is_identity_at_init_and_insulate_flag_changes_nothing_in_the_values(setup):
    model, _, _, obs = setup
    cache, valid = model.prefix_cache(obs)
    x_t = torch.randn(B, H, 32)
    with torch.no_grad():
        v = model.make_velocity_fn(cache, valid)(x_t, torch.ones(B))
        plain = model.proj.decode(model.llm.final_norms[1](model.proj.action_in_proj(x_t), model.proj.time_cond(torch.ones(B)))[0])
    assert v.shape == (B, H, 32) and torch.allclose(v, plain, atol=1e-5)  # zero-init adaRMSNorm gates: every expert layer is the identity
    emb, valid, n_img = model.embed_prefix(obs)
    tokens, _, _, cond = model.proj.embed_suffix(x_t, torch.ones(B))
    mask = model.prefix_mask(obs, n_img, n_expert=H, expert_visible=obs.expert_visible)
    pos = torch.cat([valid.long().cumsum(1) - 1, valid.long().sum(1, keepdim=True) + torch.arange(H)[None]], 1)
    for m in model.modules():  # make the expert act, so the flag is exercised on non-trivial values
        if isinstance(m, M.AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    with torch.no_grad():
        a, _ = model.llm([emb, tokens], pos, mask, None, cond, insulate=False)
        b, _ = model.llm([emb, tokens], pos, mask, None, cond, insulate=True)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)


def test_sampler_uses_five_steps_and_decoder_stops(setup):
    model, seq, raw, obs = setup
    calls = []
    cache, valid = model.prefix_cache(obs)
    v = model.make_velocity_fn(cache, valid)

    def counted(x, t):
        calls.append(float(t[0]))
        return v(x, t)

    from pi.pi0.flow_matching.model import sample_actions

    x0 = sample_actions(counted, torch.randn(B, H, 32), M.NUM_DENOISING_STEPS)
    assert x0.shape == (B, H, 32) and len(calls) == 5 and np.allclose(calls, [1.0, 0.8, 0.6, 0.4, 0.2])
    hobs, _ = build_pi06_batch(raw, unit_stats(D), seq, layout="hl_prompt", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False)
    toks, steps = model.sample_text(hobs, max_new_tokens=5, stop_ids=(EOS_ID, seq.newline_id))
    assert toks.shape == (B, 5) and 1 <= steps <= 5
    for i in range(B):
        row = toks[i, :steps].tolist()
        assert (seq.newline_id in row or EOS_ID in row) or steps == 5
    assert SEG_ACTION not in hobs.segment[hobs.token_mask].tolist()  # the prompt carries no FAST tokens
