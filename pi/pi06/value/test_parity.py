"""Alignment checks for the pi0.6* value function: head shape and readout position, V in [-1, 0] and the expectation
identity, the Gemma 3 1B stand-in counts, Eq. 1 oracle, the co-training mask, both advantage estimators (including
N >= T reducing to the whole-episode form and the zero-advantage oracle), the quantile thresholds, the indicator with
corrections / SFT, and one training step."""

import numpy as np
import pytest
import torch

import pi.pi06.value.train as T
from pi.pi06.backbone.model import GEMMA3_1B, backbone_param_count
from pi.pi06.data.data import C_FAIL_TINY, NUM_BINS, STATIC_IMAGE_KEYS, EpisodeLabels, bin_values, build_pi06_batch, episode_rewards, tiny_pi06_tokenizer, unit_stats
from pi.pi06.value.model import ValueFunction, tiny_value_function, value_param_count

B, H, D = 2, 10, 7


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    vf = tiny_value_function().eval()
    seq = tiny_pi06_tokenizer(H, D)
    rng = np.random.default_rng(0)
    raw = {"images": {k: rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
           "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32), "prompt": ["make a double espresso", "fold the shirt"]}
    obs, _ = build_pi06_batch(raw, unit_stats(D), seq, layout="value", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, value_bins=[200, 37])
    return vf, seq, raw, obs


def test_head_shape_readout_and_value_range(setup):
    vf, _, _, obs = setup
    logits, h = vf(obs)
    assert logits.shape == (B, NUM_BINS) and h.shape[0] == B
    v = vf.value(obs)
    assert v.shape == (B,) and bool((v >= -1).all()) and bool((v <= 0).all())
    p = vf.distribution(obs)
    assert torch.allclose(p @ vf.bin_values, v, atol=1e-6) and torch.allclose(vf.bin_values, torch.from_numpy(bin_values()))
    # readout = last valid column: nothing valid follows it, and changing that token changes the logits
    _, _, valid, _ = vf.vlm.forward_prefix(obs)
    idx = vf.readout_index(valid)
    assert bool((valid[torch.arange(B), idx]).all()) and not bool(valid[0, idx[0] + 1 :].any())
    obs.tokens[:, idx[0] - 256 * 3] = 7  # change the last real text token of sample 0
    assert not torch.allclose(vf(obs)[0][0], logits[0])


def test_gemma3_1b_stand_in_counts():
    pc = backbone_param_count(GEMMA3_1B)
    assert pc["non_embedding"] == 697_896_064 and pc["embedding"] == 301_989_888  # report Table 1: 698M / 302M
    vc = value_param_count()
    assert vc["value_head"] == 1152 * 201 + 201
    with torch.device("meta"):
        vf = ValueFunction(cfg=GEMMA3_1B)
    assert sum(p.numel() for p in vf.value_head.parameters()) == vc["value_head"]


def test_eq1_oracle_and_cotrain_mask(setup):
    vf, seq, raw, obs = setup
    ce = T.value_loss(vf, obs)
    assert ce.shape == (B,) and bool((ce > 0).all())
    # oracle: a head that puts all mass on the target bin -> CE ~ 0
    logits = torch.full((B, NUM_BINS), -50.0)
    logits[torch.arange(B), obs.value_bin] = 50.0
    assert float(torch.nn.functional.cross_entropy(logits, obs.value_bin)) < 1e-6
    tb, _ = build_pi06_batch(raw, unit_stats(D), seq, layout="text", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, target_text=["a dog", "a cup"])
    tce = T.cotrain_text_loss(vf, tb)
    assert tce.shape == (B,) and bool((tce > 0).all())
    assert bool(T.cotrain_text_loss(vf, obs).eq(0).all())  # a "value" batch has no SEG_TEXT: the text term is exactly 0


def test_advantage_estimators():
    T1, Tmax = 6, 10
    lab = EpisodeLabels("t", True, Tmax, T1)
    R, _ = lab.value_targets(C_FAIL_TINY)
    r = episode_rewards(T1, True, C_FAIL_TINY) / Tmax
    assert np.allclose(T.advantage_whole_episode(R, R), 0.0)  # a perfect critic gives zero advantage
    assert np.allclose(T.advantage_nstep(r, R, n=50), 0.0, atol=1e-6) and np.allclose(T.advantage_nstep(r, R, n=2), 0.0, atol=1e-6)  # Bellman-consistent targets too
    V = np.linspace(-0.9, -0.1, T1).astype(np.float32)
    assert np.allclose(T.advantage_nstep(r, V, n=T1), T.advantage_whole_episode(R, V))  # N >= T reduces to the whole-episode form
    a2 = T.advantage_nstep(r, V, n=2)
    assert np.isclose(a2[0], r[0] + r[1] + V[2] - V[0]) and np.isclose(a2[T1 - 1], r[T1 - 1] - V[T1 - 1])  # bootstrap inside, none past the end
    assert T.LOOKAHEAD_N == 50


def test_threshold_and_indicator():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50_000)
    for frac in (T.POSITIVE_FRACTION_PRETRAIN, T.POSITIVE_FRACTION_FINETUNE, T.POSITIVE_FRACTION_TSHIRT):
        eps = T.improvement_threshold(a, frac, rng)
        assert abs((a > eps).mean() - frac) < 0.02  # 10k-sample estimate of the quantile
    assert (T.POSITIVE_FRACTION_PRETRAIN, T.POSITIVE_FRACTION_FINETUNE, T.POSITIVE_FRACTION_TSHIRT, T.THRESHOLD_SAMPLE_SIZE) == (0.3, 0.4, 0.1, 10_000)
    adv = np.array([-1.0, 0.5, -0.2, 0.9])
    corr = np.array([False, False, True, False])
    assert T.improvement_indicator(adv, 0.0).tolist() == [False, True, False, True]
    assert T.improvement_indicator(adv, 0.0, corr).tolist() == [False, True, True, True]  # corrections forced positive
    assert T.improvement_indicator(adv, 0.0, corr, sft=True).all()  # SFT stage: all True
    assert T.pretrain_config().lookahead is None and T.finetune_config().lookahead == 50 and T.pretrain_config().optimizer is None


def test_train_step_runs(setup):
    vf, seq, raw, obs = setup
    tb, _ = build_pi06_batch(raw, unit_stats(D), seq, layout="text", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, target_text=["a dog", "a cup"])
    cfg = T.tiny_config()
    params = [p for p in vf.parameters() if p.requires_grad]
    opt = T.make_optimizer(params, cfg)
    m = T.train_step(vf, obs, params, opt, None, cfg, 0, text_batch=tb)
    assert m["loss"] > 0 and m["value_ce"] > 0 and m["text_ce"] > 0 and np.isfinite(m["grad_norm"])
    vf.eval()
