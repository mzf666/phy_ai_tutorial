"""Alignment checks for pi0-FAST training: the loss formula (oracle 0, uniform ln V, postfix-only, per-sample
normalization, shift by one), teacher forcing == the model's full forward, the paper schedule (constant after warmup),
LoRA parameter count at paper size, the freeze rule, and that LoRA with B = 0 is the base model."""

import dataclasses

import numpy as np
import pytest
import torch

from pi.fast.data.data import ACTION_DIM, PALIGEMMA_VOCAB_SIZE, ByteTextCodec, FASTSequenceTokenizer, build_fast_batch, tiny_fast_tokenizer
from pi.fast.model.model import Pi0FAST, paper, tiny
from pi.fast.tokenizer.tokenizer import QuantileStats, make_smooth_chunks
from pi.fast.train.train import (
    LORA_ALPHA,
    LORA_RANK,
    apply_lora,
    cross_entropy,
    libero_config,
    lora_config,
    lora_param_count,
    paper_config,
    select_trainable,
    target_logits,
    train_step,
)
from pi.pi0.data.data import make_bool_mask
from pi.pi0.train.train import EMA, lr_at, make_optimizer

H, D, B = 10, 7, 2


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Pi0FAST(*tiny())


@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(3)
    seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, D), max_len=180)
    raw = {"images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
           "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32),
           "actions": (0.3 * make_smooth_chunks(B, H + 2, D, rng)).astype(np.float32),
           "prompt": ["pick up the red block", "close the drawer"]}
    stats = {"state": QuantileStats(np.full(D, -1.0), np.full(D, 1.0)), "actions": QuantileStats(np.full(D, -1.0), np.full(D, 1.0))}
    o, _ = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=False)
    return o


# ---------------------------------------------------------------- loss: analytic properties
def test_oracle_logits_give_zero_and_uniform_logits_give_log_vocab():
    V, T = 50, 6
    targets = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    oracle = torch.full((B, T, V), -1e4).scatter(-1, targets[..., None], 1e4)
    assert torch.allclose(cross_entropy(oracle, targets, mask), torch.zeros(B), atol=1e-6)
    uniform = torch.zeros(B, T, V)
    assert torch.allclose(cross_entropy(uniform, targets, mask), torch.full((B,), float(np.log(V))), atol=1e-6)


def test_loss_is_postfix_only_and_per_sample_normalized():
    V, T = 30, 8
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 5:] = True  # 3 loss tokens
    mask[1, 2:] = True  # 6 loss tokens
    loss = cross_entropy(logits, targets, mask)
    logp = torch.log_softmax(logits, -1).gather(-1, targets[..., None])[..., 0]
    assert torch.allclose(loss[0], -logp[0, 5:].mean()) and torch.allclose(loss[1], -logp[1, 2:].mean())
    # masked-out positions do not matter: change their targets / logits
    logits2, targets2 = logits.clone(), targets.clone()
    logits2[:, :2] += 100.0
    targets2[:, :2] = (targets2[:, :2] + 1) % V
    assert torch.allclose(cross_entropy(logits2, targets2, mask), loss)
    # empty mask -> 0, not nan (clip to 1)
    assert torch.equal(cross_entropy(logits, targets, torch.zeros_like(mask)), torch.zeros(B))


def test_target_logits_shapes_and_shift_by_one(model, obs):
    with torch.no_grad():
        logits, targets, loss_mask = target_logits(model, obs)
    L = obs.tokenized_prompt.shape[1]
    assert logits.shape == (B, L - 1, PALIGEMMA_VOCAB_SIZE) and targets.shape == (B, L - 1) and loss_mask.shape == (B, L - 1)
    assert torch.equal(targets, obs.tokenized_prompt[:, 1:]) and torch.equal(loss_mask, obs.token_loss_mask[:, 1:])
    # the loss positions are exactly the postfix tokens minus the shift; the first loss target is the 'A' of "Action: "
    n_post = obs.token_loss_mask.sum(1)
    assert torch.equal(loss_mask.sum(1), n_post)
    first = [int(torch.nonzero(loss_mask[b])[0]) for b in range(B)]
    for b in range(B):
        assert int(targets[b, first[b]]) == int(obs.tokenized_prompt[b][obs.token_loss_mask[b]][0])


def test_teacher_forcing_matches_full_forward(model, obs):
    """Training logits (inputs [:, :-1], mask [:, :-1, :-1]) equal the model's full forward at the same positions:
    dropping the last (padding) column changes nothing for the other queries."""
    with torch.no_grad():
        logits, _, _ = target_logits(model, obs)
        pre_logits, _, _ = model(obs)
        full = model.logits_head(pre_logits[:, :-1][:, -logits.shape[1] :])
    valid = obs.tokenized_prompt_mask[:, :-1]  # padding rows are fully masked softmax rows: garbage, never in the loss
    assert torch.allclose(logits[valid], full[valid], atol=1e-4, rtol=1e-4)
    assert not valid.all()


# ---------------------------------------------------------------- configs
def test_paper_schedule_is_constant_after_warmup():
    cfg = paper_config()
    assert lr_at(0, cfg) == pytest.approx(5e-5 / 1001) and lr_at(cfg.warmup_steps, cfg) == pytest.approx(5e-5)
    assert lr_at(50_000, cfg) == pytest.approx(5e-5) and lr_at(cfg.num_train_steps, cfg) == pytest.approx(5e-5)
    assert (cfg.b1, cfg.b2, cfg.clip_gradient_norm, cfg.batch_size) == (0.9, 0.95, 1.0, 256)
    lib = libero_config()
    assert lr_at(lib.warmup_steps, lib) == pytest.approx(2.5e-5) and lr_at(lib.decay_steps, lib) == pytest.approx(2.5e-6)
    assert lora_config().ema_decay is None and lib.ema_decay == 0.99


# ---------------------------------------------------------------- LoRA: parameter count, freeze rule, identity
def test_lora_parameter_count_at_paper_size():
    with torch.device("meta"):
        m = apply_lora(Pi0FAST(*paper()))
    n_lora = sum(p.numel() for n, p in m.named_parameters() if "lora" in n)
    assert n_lora == lora_param_count(m.cfg) == 27_869_184
    assert sum(p.numel() for p in m.parameters()) == 2_923_335_408 + 27_869_184
    trainable = select_trainable(m, "lora")
    assert sum(p.numel() for p in trainable) == 27_869_184 + 414_803_696  # lora + SigLIP (regex only matches llm)
    assert all(p.requires_grad for p in m.img.parameters())
    assert not m.llm.embedder.input_embedding.requires_grad
    assert sum(p.numel() for p in select_trainable(m, "full")) == 2_923_335_408 + 27_869_184


def test_lora_with_zero_b_is_the_base_model(obs):
    torch.manual_seed(1)
    base = Pi0FAST(*tiny()).eval()
    lora = apply_lora(Pi0FAST(*tiny()))
    lora.load_state_dict(base.state_dict(), strict=False)
    lora.eval()
    with torch.no_grad():
        for n, p in lora.named_parameters():
            if "lora_b" in n:
                p.zero_()
        a, _, _ = target_logits(base, obs)
        b, _, _ = target_logits(lora, obs)
        assert torch.allclose(a, b, atol=1e-5)
        for n, p in lora.named_parameters():  # upstream init: B ~ N(0, 0.01^2), so the output moves at init
            if "lora_b" in n:
                p.normal_(0, 0.01)
        c, _, _ = target_logits(lora, obs)
    assert not torch.allclose(a, c, atol=1e-5)
    assert LORA_ALPHA / LORA_RANK == 1.0


def test_train_step_updates_only_selected_params(obs):
    torch.manual_seed(2)
    model = apply_lora(Pi0FAST(*tiny()))
    params = select_trainable(model, "lora")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = dataclasses.replace(lora_config(), warmup_steps=1)
    info = train_step(model, obs, params, make_optimizer(params, cfg), None, cfg, 1)
    assert info["loss"] > 0 and info["loss_tokens"] == float(obs.token_loss_mask.sum())
    for n, p in model.named_parameters():
        moved = not torch.equal(before[n], p.detach())
        if n.startswith("llm.") and "lora" not in n:
            assert not moved, n  # frozen
        elif "lora" in n or n == "img.head.weight":
            assert moved, n
        # other SigLIP params are trainable but get a zero gradient at step 0: the SigLIP head is zero-initialised
        # (pi0/vlm head_zeroinit), so nothing before it receives a gradient until the head moves
