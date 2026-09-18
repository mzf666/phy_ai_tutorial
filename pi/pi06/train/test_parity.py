"""Alignment checks for pi0.6* training: the KI stop-gradient (MSE gradients reach the expert only when insulate=True,
and the backbone too when False), the forward values being independent of the flag, CE independent of the expert's
presence, expert output independent of the FAST tokens, alpha = 1 and the stage configurations, indicator modes
(SFT all True, value mode with corrections and 30% dropout), one training step on joint and text batches, and the
RECAP loop refitting from the pre-trained checkpoints while the dataset grows."""

import copy

import numpy as np
import pytest
import torch

import pi.pi06.train.train as T
from pi.pi0.data.data import make_bool_mask
from pi.pi0.flow_matching.train import interpolate
from pi.pi06.backbone.model import AdaRMSNorm, tiny_pi06
from pi.pi06.data.data import ADVANTAGE_DROPOUT, C_FAIL_TINY, SEG_ACTION, STATIC_IMAGE_KEYS, EpisodeLabels, build_pi06_batch, episode_rewards, tiny_pi06_tokenizer, unit_stats
from pi.pi06.value.model import tiny_value_function

H, D, B = 10, 7, 2


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = tiny_pi06()
    for m in model.modules():  # zero-init gates would make every stop-gradient test vacuous
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    seq = tiny_pi06_tokenizer(H, D)
    raw = {"images": {k: rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
           "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32), "actions": rng.uniform(-0.5, 0.5, (B, H + 2, D)).astype(np.float32),
           "prompt": ["make me an espresso", "fold the shirt"]}
    obs, actions = build_pi06_batch(raw, unit_stats(D), seq, layout="joint", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=False,
                                    subtasks=["grab the portafilter", None], advantages=[True, None])
    t = torch.tensor([0.7, 0.3])
    noise = torch.randn(actions.shape)
    return model, seq, raw, obs, actions, t, noise


def test_stop_gradient_isolates_the_backbone(setup):
    model, _, _, obs, actions, t, noise = setup
    for ins, expect_backbone in ((True, 0.0), (False, None)):
        model.zero_grad(set_to_none=True)
        T.ki_loss(model, obs, actions, t=t, noise=noise, insulate=ins)["mse"].mean().backward()
        g = T.grad_norms_by_part(model)
        assert g["expert"] > 0
        if expect_backbone == 0.0:
            assert g["backbone"] == 0.0  # KI Eq. 5-6: nothing from the flow loss reaches vision / embedder / experts[0]
        else:
            assert g["backbone"] > 0  # the joint-training baseline (no stop-gradient) does leak
    model.zero_grad(set_to_none=True)
    T.ki_loss(model, obs, actions, t=t, noise=noise, insulate=True)["ce"].mean().backward()
    g = T.grad_norms_by_part(model)
    assert g["backbone"] > 0 and g["expert"] == 0.0  # the CE never touches the expert (nobody reads it)
    model.zero_grad(set_to_none=True)


def test_forward_values_independent_of_the_flag_and_masks(setup):
    model, _, _, obs, actions, t, noise = setup
    with torch.no_grad():
        a = T.ki_loss(model, obs, actions, t=t, noise=noise, insulate=True)
        b = T.ki_loss(model, obs, actions, t=t, noise=noise, insulate=False)
        assert torch.allclose(a["ce"], b["ce"], atol=1e-5) and torch.allclose(a["mse"], b["mse"], atol=1e-5)
        # CE does not depend on the expert being present (text rows never see expert columns)
        logits_j, tg_j, idx_j, v_t = T.ki_forward(model, obs, *interpolate(actions, noise, t)[:1], t)
        logits_t, tg_t, idx_t, none = T.ki_forward(model, obs, None, None)
        assert none is None and torch.equal(tg_j, tg_t) and torch.equal(idx_j, idx_t) and torch.allclose(logits_j, logits_t, atol=1e-4)
        # the expert's velocity does not depend on the FAST tokens (it cannot read them)
        obs2 = copy.deepcopy(obs)
        fast = obs2.segment == SEG_ACTION
        obs2.tokens[fast] = (obs2.tokens[fast] + 7) % 300 + 3
        _, _, _, v2 = T.ki_forward(model, obs2, *interpolate(actions, noise, t)[:1], t)
        assert torch.allclose(v_t, v2, atol=1e-5)
        # ... but it does depend on the Advantage token (sample 0 has one)
        obs3 = copy.deepcopy(obs)
        adv = (obs3.segment == 2) & obs3.token_mask
        obs3.tokens[adv] = (obs3.tokens[adv] + 5) % 300 + 3
        _, _, _, v3 = T.ki_forward(model, obs3, *interpolate(actions, noise, t)[:1], t)
        assert not torch.allclose(v_t[0], v3[0], atol=1e-5) and torch.allclose(v_t[1], v3[1], atol=1e-5)


def test_alpha_stages_and_text_batch(setup):
    model, seq, raw, obs, actions, t, noise = setup
    assert T.ALPHA == 1.0 and T.KI_EXTRA_COMPUTE == 0.2 and T.KI_STEPS_VS_PI0 == 7.5
    pre, sft, rec = T.pretrain_config(), T.sft_config(), T.recap_config()
    assert (pre.indicator, pre.positive_fraction, pre.advantage_dropout) == ("value", 0.3, ADVANTAGE_DROPOUT)
    assert (sft.indicator, sft.positive_fraction, sft.init_from) == ("true", None, "pi_pre")
    assert (rec.indicator, rec.positive_fraction, rec.init_from, rec.insulate) == ("value", 0.4, "pi_pre", True)
    assert all(c.optimizer is None for c in (pre, sft, rec))
    tobs, _ = build_pi06_batch(raw, unit_stats(D), seq, layout="text", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, target_text=["a dog", "a cup"])
    with torch.no_grad():
        o = T.ki_loss(model, tobs, None)
    assert bool((o["mse"] == 0).all()) and bool((o["ce"] > 0).all())  # M^act = 0: no expert term
    # alpha scales only the MSE term
    with torch.no_grad():
        a1 = T.ki_loss(model, obs, actions, alpha=1.0, t=t, noise=noise)
        a2 = T.ki_loss(model, obs, actions, alpha=2.0, t=t, noise=noise)
    assert torch.allclose(a2["loss"] - a1["loss"], a1["mse"].mean(), atol=1e-5)


def test_indicator_modes():
    rng = np.random.default_rng(0)
    lab = EpisodeLabels("t", True, 40, 8, is_correction=np.array([False] * 6 + [True] * 2))
    R, _ = lab.value_targets(C_FAIL_TINY)
    ep = T.Episode(lab, [], episode_rewards(8, True, C_FAIL_TINY) / 40, R, [])
    out = T.stage_indicators(None, [ep] * 50, T.sft_config(), None, rng)
    flat = [x for e in out for x in e]
    assert False not in flat and abs(flat.count(None) / len(flat) - ADVANTAGE_DROPOUT) < 0.05  # SFT: True or dropped, never False

    class FakeVF:  # values that make every advantage equal, so the quantile puts ~40% above with corrections forced True
        def eval(self):
            return self

        def value(self, obs):
            return torch.zeros(1)

    steps = [None] * 8
    ep2 = T.Episode(lab, steps, ep.norm_rewards, ep.norm_returns, [])
    T.label_episode.__globals__["episode_values"]  # exists
    import pi.pi06.value.train as VT

    orig = VT.episode_values
    VT.episode_values = lambda vf, b: np.full(len(b), -0.5, np.float32)
    try:
        out = T.stage_indicators(FakeVF(), [ep2], T.recap_config(), VT.finetune_config(), np.random.default_rng(1))[0]
    finally:
        VT.episode_values = orig
    assert out[6] in (True, None) and out[7] in (True, None)  # corrections forced True (before dropout)
    assert len(out) == 8


def test_train_step_and_recap_loop(setup):
    model, seq, raw, obs, actions, _, _ = setup
    cfg = T.tiny_config()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = T.make_optimizer(params, cfg)
    m = T.train_step(model, (obs, actions), params, opt, None, cfg, 0)
    assert m["loss"] > 0 and m["mse"] > 0 and np.isfinite(m["grad_norm"])
    vf = tiny_value_function()
    pi_state, v_state = copy.deepcopy(model.state_dict()), copy.deepcopy(vf.state_dict())
    starts = []

    def collect(policy):
        return [T.Episode(EpisodeLabels("t", True, 40, 5), [], np.zeros(5), np.zeros(5), [])] * 2

    def fit_value(v_pre_state, dataset):
        assert all(torch.equal(v_pre_state[k], v_state[k]) for k in v_state)  # always from V_pre
        return vf

    def fit_policy(pi_pre_state, dataset, vf_k, stage_cfg):
        assert all(torch.equal(pi_pre_state[k], pi_state[k]) for k in pi_state)  # always from pi_pre
        starts.append((stage_cfg.stage, len(dataset)))
        return model

    res = T.recap(model, vf, [collect(None)[0]], collect=collect, fit_value=fit_value, fit_policy=fit_policy, iterations=2, log=lambda *_: None)
    assert starts == [("sft", 1), ("recap", 3), ("recap", 5)] and len(res["dataset"]) == 5 and [h["episodes"] for h in res["history"]] == [1, 3, 5]
    model.eval()
