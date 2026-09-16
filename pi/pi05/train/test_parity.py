"""Alignment checks for pi0.5 training: the Fig. 18 mask relations, CE == pi.fast.train's target logits path, the
expert output == the inference path when no FAST tokens are present and its independence from the FAST tokens, the
oracle / alpha = 0 identities, expert re-initialisation, and the disclosed schedules."""

import numpy as np
import pytest
import torch

from pi.fast.tokenizer.tokenizer import make_smooth_chunks
from pi.fast.train.train import cross_entropy
from pi.pi0.data.data import make_bool_mask
from pi.pi0.flow_matching.train import interpolate
from pi.pi0.train.train import lr_at
from pi.pi0.vlm.model import make_attn_mask
from pi.pi05.data.data import LL_IMAGE_KEYS, build_pi05_batch, tiny_pi05_tokenizer, unit_stats
from pi.pi05.expert.model import AdaRMSNorm, suffix_forward
from pi.pi05.hier.model import tiny_pi05
from pi.pi05.train.train import (
    ALPHA_POSTTRAIN,
    ALPHA_PRETRAIN,
    POSTTRAIN_STEPS,
    PRETRAIN_STEPS,
    expert_parameter_names,
    hirobot_hl_config,
    init_expert_for_posttraining,
    joint_forward,
    joint_loss,
    make_joint_mask,
    openpi_libero_config,
    posttrain_config,
    pretrain_config,
    split_prefix_fast,
)

B, H, D = 2, 10, 7


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = tiny_pi05().eval()
    m.proj.action_horizon = H
    return m


@pytest.fixture(scope="module")
def seq():
    return tiny_pi05_tokenizer(H, D)


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(4)
    return {"images": {"base_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8), "left_wrist_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8)},
            "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32),
            "actions": (0.3 * make_smooth_chunks(B, H + 2, D, rng)).astype(np.float32),
            "prompt": ["put the plate in the sink", "pick up the cup"]}


def batch(raw, seq, layout):
    return build_pi05_batch(raw, unit_stats(D), seq, layout=layout, image_keys=LL_IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=False)


# ---------------------------------------------------------------- the mask
def test_joint_mask_relations():
    pre = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)  # 7 token columns
    post = torch.tensor([[0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    m = make_joint_mask(pre, post, 2)
    assert m.shape == (2, 9, 9)
    a = m[0]
    assert a[:3, :3].all() and not a[:3, 3:].any()  # prefix <-> prefix only
    assert a[3:6, :3].all() and torch.equal(a[3:6, 3:6], torch.tril(torch.ones(3, 3, dtype=torch.bool))) and not a[3:6, 7:].any()  # FAST: prefix + causal FAST, no expert
    assert a[7:, :3].all() and not a[7:, 3:7].any() and a[7:, 7:].all()  # expert: prefix + expert, never FAST
    assert not a[6, :].any() and not a[:, 6].any()  # pad row / column
    b = m[1]
    assert b[7:, :2].all() and not b[7:, 2:7].any()
    # no FAST tokens: identical to pi0's make_attn_mask over [prefix | expert] with ar = [0.., 1, 0..]
    pre2 = torch.ones(1, 5, dtype=torch.bool)
    post2 = torch.zeros(1, 5, dtype=torch.bool)
    ref = make_attn_mask(torch.ones(1, 8, dtype=torch.bool), torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.bool))
    ours = make_joint_mask(pre2, post2, 3)
    assert torch.equal(ours, ref)


# ---------------------------------------------------------------- CE part == FAST's training forward
def test_ce_matches_fast_target_logits(model, seq, raw):
    obs, _ = batch(raw, seq, "fast")
    with torch.no_grad():
        logits, targets, lm, v = joint_forward(model, obs, None, None)
        # reference: pi.fast.train.target_logits' recipe on this model (prefix-LM + causal postfix via the cumsum mask)
        emb, input_mask, ar = model.embed_prefix(obs)
        attn = make_attn_mask(input_mask, ar)
        (h, _), _ = model.llm([emb, None], input_mask.long().cumsum(1) - 1, attn)
        L = obs.tokenized_prompt.shape[1]
        n_img = emb.shape[1] - L
        ref_logits = model.logits_head(h[:, n_img : n_img + L - 1])
    assert v is None and logits.shape == (B, L - 1, model.embedder.input_embedding.shape[0])
    valid = lm | obs.tokenized_prompt_mask[:, :-1]
    assert torch.allclose(logits[valid], ref_logits[valid], atol=2e-3)
    out = joint_loss(model, obs, None, alpha=ALPHA_PRETRAIN)
    assert torch.allclose(out["loss"], cross_entropy(ref_logits, targets, lm).mean(), atol=2e-3) and bool((out["mse"] == 0).all())


# ---------------------------------------------------------------- MSE part == inference path; independent of FAST tokens
def test_expert_output_matches_inference_and_ignores_fast_tokens(model, seq, raw):
    torch.manual_seed(1)
    for m in model.modules():  # a non-identity expert so the check is not vacuous
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    obs_flow, actions = batch(raw, seq, "flow")
    x_t = torch.randn(B, H, 32)
    t = torch.rand(B)
    with torch.no_grad():
        _, _, _, v_train = joint_forward(model, obs_flow, x_t, t)
        kv, pm = model.prefix_cache(obs_flow)
        tokens, smask, sar, cond = model.proj.embed_suffix(x_t, t)
        v_infer = model.proj.decode(suffix_forward(model.llm, kv, pm, tokens, smask, sar, cond))
    assert torch.allclose(v_train, v_infer, atol=1e-4)
    # with FAST tokens in the sequence the expert output must not change when those tokens change
    obs_fast, _ = batch(raw, seq, "fast")
    with torch.no_grad():
        _, _, _, v_a = joint_forward(model, obs_fast, x_t, t)
        pre, post, _ = split_prefix_fast(obs_fast)
        n_img = 256 * len(obs_fast.images)
        ids = obs_fast.tokenized_prompt.clone()
        pm_tok = post[:, n_img:]
        ids[pm_tok] = (ids[pm_tok] + 7) % 200 + 3  # scramble the postfix ids only
        obs_b = type(obs_fast)(obs_fast.images, obs_fast.image_masks, obs_fast.state, ids, obs_fast.tokenized_prompt_mask, obs_fast.token_ar_mask, obs_fast.token_loss_mask)
        _, _, _, v_b = joint_forward(model, obs_b, x_t, t)
    assert torch.allclose(v_a, v_b, atol=1e-5)
    for m in model.modules():
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.zeros_(m.modulation.weight)


# ---------------------------------------------------------------- Eq. 1 identities
def test_oracle_and_alpha_zero(model, seq, raw, monkeypatch):
    obs, actions = batch(raw, seq, "fast")
    t, noise = torch.rand(B), torch.randn(B, H, 32)
    x_t, u_t = interpolate(actions, noise, t)
    orig = joint_forward

    def oracle(model_, obs_, x, tt):
        logits, targets, lm, _ = orig(model_, obs_, None, None)
        return logits, targets, lm, u_t  # a network that returns the true velocity

    monkeypatch.setattr("pi.pi05.train.train.joint_forward", oracle)
    out = joint_loss(model, obs, actions, alpha=ALPHA_POSTTRAIN, t=t, noise=noise)
    assert bool((out["mse"].abs() < 1e-7).all()) and torch.allclose(out["loss"], out["ce"].mean())
    monkeypatch.undo()
    out0 = joint_loss(model, obs, actions, alpha=0.0, t=t, noise=noise)
    assert torch.allclose(out0["loss"], out0["ce"].mean()) and bool((out0["mse"] == 0).all())
    # has_actions masks the MSE of text-only samples
    out1 = joint_loss(model, obs, actions, alpha=ALPHA_POSTTRAIN, t=t, noise=noise, has_actions=torch.tensor([True, False]))
    expected = out1["ce"].mean() + ALPHA_POSTTRAIN * out1["mse"][0]
    assert torch.allclose(out1["loss"], expected)


# ---------------------------------------------------------------- post-training starts from a fresh, identity expert
def test_init_expert_for_posttraining(model):
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    names = init_expert_for_posttraining(model, seed=3)
    expert = set(expert_parameter_names(model))
    for n, p in model.named_parameters():
        if n in expert and "modulation" not in n:
            assert not torch.equal(p, before[n]), n
        elif n not in expert:
            assert torch.equal(p, before[n]), n
    assert set(names) == expert
    for m in model.modules():
        if isinstance(m, AdaRMSNorm):
            assert bool((m.modulation.weight == 0).all()) and bool((m.modulation.bias == 0).all())


# ---------------------------------------------------------------- configurations
def test_disclosed_configs():
    assert pretrain_config().num_train_steps == PRETRAIN_STEPS == 280_000 and pretrain_config().alpha == 0.0 and pretrain_config().optimizer is None
    assert posttrain_config().num_train_steps == POSTTRAIN_STEPS == 80_000 and posttrain_config().alpha == 10.0
    lib = openpi_libero_config()
    assert lr_at(10_000, lib) == pytest.approx(5e-5) and lr_at(25_000, lib) == pytest.approx(5e-5) and lr_at(0, lib) < 1e-8  # warmup then constant
    hl = hirobot_hl_config()
    assert hl.weight_decay == 0.0 and hl.ema_decay == 0.999 and hl.batch_size == 512 and lr_at(5_000, hl) == pytest.approx(1e-5)
