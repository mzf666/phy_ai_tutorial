"""Checks for the pi0 training loop: schedule anchors, optimizer values, clipping, EMA, parameter provenance.

References: openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 (src/openpi/training/optimizer.py, config.py,
weight_loaders.py, scripts/train.py), pi0 paper arXiv:2410.24164v1 Sec. V-A. Run on CPU: `uv run pytest pi/pi0/train -q`.
"""

import math

import numpy as np
import torch

from pi.pi0.data.data import ByteEncoder, NormStats, PromptTokenizer, build_batch, make_bool_mask
from pi.pi0.infer import model as I
from pi.pi0.train import train as T


# ---------------------------------------------------------------- hyperparameters (config.py, optimizer.py)
def test_defaults_match_openpi():
    c = T.TrainConfig()
    assert (c.warmup_steps, c.peak_lr, c.decay_steps, c.decay_lr) == (1_000, 2.5e-5, 30_000, 2.5e-6)
    assert (c.b1, c.b2, c.eps, c.weight_decay, c.clip_gradient_norm) == (0.9, 0.95, 1e-8, 1e-10, 1.0)
    assert (c.ema_decay, c.batch_size, c.num_train_steps) == (0.99, 32, 30_000)


def test_lr_schedule_anchors_match_optax():
    c = T.TrainConfig()
    assert math.isclose(T.lr_at(0, c), c.peak_lr / (c.warmup_steps + 1))  # init_value
    assert math.isclose(T.lr_at(c.warmup_steps, c), c.peak_lr)  # end of warmup
    assert math.isclose(T.lr_at(c.decay_steps, c), c.decay_lr)  # end of cosine
    assert math.isclose(T.lr_at(c.decay_steps + 5_000, c), c.decay_lr)  # constant afterwards
    mid = c.warmup_steps + (c.decay_steps - c.warmup_steps) // 2
    assert math.isclose(T.lr_at(mid, c), 0.5 * (c.peak_lr + c.decay_lr), rel_tol=1e-6)  # cosine midpoint
    assert T.lr_at(500, c) < c.peak_lr and T.lr_at(500, c) > T.lr_at(0, c)  # warmup is increasing
    # monotone: rising then falling
    lrs = [T.lr_at(s, c) for s in range(0, c.decay_steps + 1, 100)]
    peak = lrs.index(max(lrs))
    assert all(a <= b for a, b in zip(lrs[:peak], lrs[1:peak + 1])) and all(a >= b for a, b in zip(lrs[peak:], lrs[peak + 1:]))


def test_optimizer_is_adamw_with_openpi_values():
    c = T.TrainConfig()
    p = [torch.nn.Parameter(torch.zeros(3))]
    opt = T.make_optimizer(p, c)
    g = opt.param_groups[0]
    assert isinstance(opt, torch.optim.AdamW)
    assert (g["betas"], g["eps"], g["weight_decay"]) == ((0.9, 0.95), 1e-8, 1e-10)


def test_clip_and_step_clips_global_norm_and_sets_lr():
    c = T.TrainConfig()
    p = [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))]
    opt = T.make_optimizer(p, c)
    for q in p:
        q.grad = torch.full((4,), 3.0)  # global norm = sqrt(8 * 9) = 8.49 > 1
    pre = T.clip_and_step(p, opt, c, step=c.warmup_steps)
    assert math.isclose(pre, math.sqrt(8 * 9), rel_tol=1e-5)
    assert opt.param_groups[0]["lr"] == c.peak_lr
    assert all(q.grad is None for q in p)


def test_ema_update_formula():
    m = torch.nn.Linear(2, 2)
    ema = T.EMA(m, 0.9)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    for n, p in m.named_parameters():
        torch.testing.assert_close(ema.shadow[n], 0.9 * before[n] + 0.1 * p.detach())


# ---------------------------------------------------------------- parameter provenance (weight_loaders.py)
def test_split_params_counts_at_paper_size():
    with torch.device("meta"):
        m = I.Pi0()
    groups = T.split_params(m)
    sizes = {n: p.numel() for n, p in m.named_parameters()}
    loaded = sum(sizes[n] for n in groups["from_paligemma"])
    fresh = sum(sizes[n] for n in groups["from_scratch"])
    assert loaded == 2_923_335_408  # SigLIP + embedding + Gemma 2B (../vlm README)
    assert fresh == 311_464_960 + 3_248_160  # Gemma 300M expert + projections (../action_expert README)
    assert loaded + fresh == 3_238_048_528 and len(groups["from_paligemma"]) + len(groups["from_scratch"]) == len(sizes)
    assert all(".experts.1." not in n for n in groups["from_paligemma"])
    assert all(n.startswith(("llm.layers", "llm.final_norms.1", "proj.")) for n in groups["from_scratch"])


def test_mixture_weights_power_law():
    w = T.mixture_weights({"a": 10**8, "b": 10**6})
    assert math.isclose(w["a"] / w["b"], 100**0.43, rel_tol=1e-9)
    assert math.isclose(sum(w.values()), 1.0)


# ---------------------------------------------------------------- one real step on the tiny model
def _batch(B=2, d=7, seed=0):
    r = np.random.default_rng(seed)
    raw = {"images": {"base_0_rgb": r.integers(0, 256, (B, 64, 64, 3), dtype=np.uint8)},
           "state": r.standard_normal((B, d)).astype(np.float32),
           "actions": r.standard_normal((B, 55, d)).astype(np.float32), "prompt": ["x"] * B}
    stats = {"state": NormStats(np.zeros(d, np.float32), np.ones(d, np.float32)), "actions": NormStats(np.zeros(d, np.float32), np.ones(d, np.float32))}
    return build_batch(raw, stats, PromptTokenizer(ByteEncoder()), delta_mask=make_bool_mask(6, -1), train=True,
                       generator=torch.Generator().manual_seed(seed))


def test_train_step_updates_trainable_freezes_frozen_and_moves_ema():
    torch.manual_seed(0)
    m = I.tiny_pi0()
    cfg = T.TrainConfig(warmup_steps=1, decay_steps=4)
    params = T.select_trainable(m, "llm")  # freeze both experts; SigLIP, embedder, projections train
    assert all(not p.requires_grad for n, p in m.named_parameters() if n.startswith("llm."))
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt, ema = T.make_optimizer(params, cfg), T.EMA(m, cfg.ema_decay)
    info = T.train_step(m, _batch(), params, opt, ema, cfg, step=1, generator=torch.Generator().manual_seed(0))
    assert set(info) == {"loss", "grad_norm", "param_norm", "lr"} and info["lr"] == cfg.peak_lr and info["loss"] > 0
    for n, p in m.named_parameters():
        if n.startswith("llm."):
            torch.testing.assert_close(p, before[n])  # frozen
            torch.testing.assert_close(ema.shadow[n], before[n])  # EMA of a frozen param stays put
        elif n.startswith("proj."):
            assert not torch.equal(p, before[n])  # trained
            torch.testing.assert_close(ema.shadow[n], 0.99 * before[n] + 0.01 * p.detach())


def test_evaluate_loss_is_deterministic():
    torch.manual_seed(0)
    m = I.tiny_pi0()
    b = [_batch(seed=1), _batch(seed=2)]
    assert T.evaluate_loss(m, b, seed=3) == T.evaluate_loss(m, b, seed=3)
    assert T.evaluate_loss(m, b, seed=3) != T.evaluate_loss(m, b, seed=4)
