"""Alignment checks for the pi0.5 action expert: paper-size parameter counts, AdaRMSNorm at zero init, the expert as an
identity map at init, expert 0 unchanged from pi0's MoEGemma, the suffix contract, and the first-step gradients."""

import pytest
import torch

from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, MoEGemma, tiny_experts
from pi.pi0.vlm.model import RMSNorm, make_attn_mask
from pi.pi05.expert.model import PI05_EXPERTS, AdaMoEGemma, AdaRMSNorm, Pi05ActionProjections, expert_param_count, suffix_forward

B = 2


@pytest.fixture(scope="module")
def parts():
    torch.manual_seed(0)
    vlm_cfg, exp_cfg = tiny_experts()
    llm = AdaMoEGemma((vlm_cfg, exp_cfg)).eval()
    proj = Pi05ActionProjections(exp_cfg).eval()
    prefix_emb = torch.randn(B, 40, vlm_cfg.width)
    prefix_mask = torch.ones(B, 40, dtype=torch.bool)
    prefix_mask[1, 30:] = False
    prefix_ar = torch.zeros(40, dtype=torch.bool)
    with torch.no_grad():
        (_, _), kv = llm([prefix_emb, None], prefix_mask.long().cumsum(1) - 1, make_attn_mask(prefix_mask, prefix_ar))
    return llm, proj, kv, prefix_mask, prefix_emb, prefix_ar


# ---------------------------------------------------------------- parameter counts (README Sec. 1.5)
def test_paper_parameter_counts():
    vlm_cfg, exp_cfg = PI05_EXPERTS
    assert expert_param_count(exp_cfg) == 427_932_672
    with torch.device("meta"):
        llm = AdaMoEGemma(PI05_EXPERTS)
        proj = Pi05ActionProjections(exp_cfg)
        pi0_llm = MoEGemma(PI05_EXPERTS)
    n = lambda m: sum(p.numel() for p in m.parameters())
    e1 = sum(n(l.experts[1]) for l in llm.layers) + n(llm.final_norms[1])
    assert e1 == 427_932_672
    assert n(llm.layers[0].experts[1].pre_attention_norm) == 3_148_800
    assert n(proj) == 2_165_792
    e1_pi0 = sum(n(l.experts[1]) for l in pi0_llm.layers) + n(pi0_llm.final_norms[1])
    assert e1_pi0 == 311_464_960
    assert e1 - e1_pi0 == 37 * (3_148_800 - 1_024)  # 18 layers x 2 norms + final norm, each swapping a 1024-scale for the modulation
    e0 = sum(n(l.experts[0]) for l in llm.layers) + n(llm.final_norms[0])
    assert e0 == sum(n(l.experts[0]) for l in pi0_llm.layers) + n(pi0_llm.final_norms[0])  # expert 0 untouched
    assert 2_923_335_408 + e1 + n(proj) == 3_353_433_872


# ---------------------------------------------------------------- AdaRMSNorm
def test_ada_rmsnorm_zero_init_is_bare_rmsnorm_with_zero_gate():
    norm = AdaRMSNorm(16, 8)
    x, cond = torch.randn(B, 5, 16), torch.randn(B, 8)
    y, gate = norm(x, cond)
    plain = RMSNorm(16)  # scale zero-init -> (1 + 0) * normed
    assert torch.allclose(y, plain(x), atol=1e-6) and bool((gate == 0).all()) and gate.shape == (B, 1, 16)
    torch.nn.init.normal_(norm.modulation.weight, std=0.1)
    y2, gate2 = norm(x, cond)
    scale, shift, g = norm.modulation(cond)[:, None, :].chunk(3, -1)
    assert torch.allclose(y2, plain(x) * (1 + scale) + shift, atol=1e-5) and torch.equal(gate2, g)


# ---------------------------------------------------------------- the expert at init is the identity
def test_zero_init_expert_is_identity(parts):
    llm, proj, kv, prefix_mask, _, _ = parts
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    with torch.no_grad():
        out_a = suffix_forward(llm, kv, prefix_mask, *proj.embed_suffix(x_t, torch.ones(B)))
        out_b = suffix_forward(llm, kv, prefix_mask, *proj.embed_suffix(x_t, torch.full((B,), 0.3)))
        ident = llm._plain_final(proj.action_in_proj(x_t))
    assert torch.allclose(out_a, ident, atol=1e-5) and torch.allclose(out_b, ident, atol=1e-5)  # no tau, no prefix dependence
    for m in llm.modules():
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    with torch.no_grad():
        out_c = suffix_forward(llm, kv, prefix_mask, *proj.embed_suffix(x_t, torch.ones(B)))
        out_d = suffix_forward(llm, kv, prefix_mask, *proj.embed_suffix(x_t, torch.full((B,), 0.3)))
    assert not torch.allclose(out_c, ident, atol=1e-3) and not torch.allclose(out_c, out_d, atol=1e-4)
    for m in llm.modules():
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.zeros_(m.modulation.weight)


# ---------------------------------------------------------------- expert 0 == pi0
def test_expert0_matches_pi0_moegemma_bitwise(parts):
    llm, _, _, prefix_mask, prefix_emb, prefix_ar = parts
    pi0 = MoEGemma(llm.cfgs).eval()
    missing = pi0.load_state_dict({k: v for k, v in llm.state_dict().items() if ".experts.1." not in k and "final_norms.1" not in k and "plain" not in k}, strict=False)
    assert all(".experts.1." in k or "final_norms.1" in k for k in missing.missing_keys)
    pos, mask = prefix_mask.long().cumsum(1) - 1, make_attn_mask(prefix_mask, prefix_ar)
    with torch.no_grad():
        (a, _), kv_a = llm([prefix_emb, None], pos, mask)
        (b, _), kv_b = pi0([prefix_emb, None], pos, mask)
    assert torch.equal(a, b) and all(torch.equal(x[0], y[0]) for x, y in zip(kv_a, kv_b))


# ---------------------------------------------------------------- suffix contract
def test_suffix_contract(parts):
    _, proj, _, _, _, _ = parts
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    tokens, mask, ar, cond = proj.embed_suffix(x_t, torch.rand(B))
    assert tokens.shape == (B, ACTION_HORIZON, proj.cfg.width) and mask.shape == (B, ACTION_HORIZON) and bool(mask.all())
    assert ar.tolist() == [True] + [False] * (ACTION_HORIZON - 1)
    assert cond.shape == (B, proj.cfg.width)
    assert not torch.allclose(proj.time_cond(torch.zeros(B)), proj.time_cond(torch.ones(B)))
    assert not hasattr(proj, "state_proj") and not hasattr(proj, "action_time_mlp_in")
    assert proj.decode(torch.randn(B, ACTION_HORIZON, proj.cfg.width)).shape == (B, ACTION_HORIZON, ACTION_DIM)


# ---------------------------------------------------------------- first-step gradients at zero init
def test_first_step_gradients_reach_the_modulation(parts):
    llm, proj, kv, prefix_mask, _, _ = parts
    llm.train()
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    out = suffix_forward(llm, kv, prefix_mask, *proj.embed_suffix(x_t, torch.rand(B)))
    proj.decode(out).pow(2).mean().backward()
    mod = llm.layers[0].experts[1].pre_attention_norm.modulation
    g = mod.weight.grad.view(3, -1, mod.weight.shape[1])
    assert g is not None and g[2].abs().sum() > 0  # the gate component gets gradient although gate == 0
    assert llm.layers[0].experts[1].attn.q_einsum.weight.grad.abs().sum() == 0  # gated off: no gradient through the branch yet
    assert proj.action_in_proj.weight.grad.abs().sum() > 0
    llm.zero_grad(), proj.zero_grad()
    llm.eval()
