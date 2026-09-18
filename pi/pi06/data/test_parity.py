"""Alignment checks for the pi0.6* data pipeline: segment layout and loss masks per layout, the advantage token wording
and placement, the expert-visibility rule, the 30% dropout, the FAST round trip, the Eq. 5 reward / return / bin chain,
the 448 image contract, and the token budget of the paper robot."""

import numpy as np
import pytest

from pi.fast.data.data import BOS_ID, EOS_ID
from pi.fast.tokenizer.tokenizer import make_smooth_chunks
from pi.pi0.data.data import make_bool_mask
from pi.pi06.data.data import (
    ACTION_DIM,
    ADVANTAGE_DROPOUT,
    ADVANTAGE_TEXT,
    IMAGE_KEYS,
    IMAGE_RESOLUTION,
    MAX_TOKEN_LEN,
    NUM_BINS,
    SEG_ACTION,
    SEG_ADVANTAGE,
    SEG_MARKER,
    SEG_PREFIX,
    SEG_SUBTASK,
    SEG_TEXT,
    STATIC_IMAGE_KEYS,
    EpisodeLabels,
    advantage_text,
    bin_to_value,
    bin_values,
    build_pi06_batch,
    drop_indicator,
    episode_rewards,
    expert_visible,
    normalize_return,
    returns,
    subtask_text,
    tiny_pi06_tokenizer,
    unit_stats,
    value_to_bin,
    with_metadata,
)

H, D, B = 10, 7, 2


@pytest.fixture(scope="module")
def seq():
    return tiny_pi06_tokenizer(H, D)


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(1)
    state = rng.uniform(-0.5, 0.5, D).astype(np.float32)
    actions = (0.3 * make_smooth_chunks(1, H, D, rng)[0]).astype(np.float32)
    return "make a double espresso", state, actions


# ---------------------------------------------------------------- text pieces
def test_advantage_and_subtask_wording():
    assert ADVANTAGE_TEXT == {True: "Advantage: positive", False: "Advantage: negative"}  # paper Sec. V-B verbatim
    assert advantage_text(True) == "Advantage: positive\n" and advantage_text(False) == "Advantage: negative\n" and advantage_text(None) == ""
    assert subtask_text("pick_up the cup\n") == "Subtask: pick up the cup\n"
    assert with_metadata("fold the shirt", None) == "fold the shirt" and with_metadata("fold the shirt", "speed: fast") == "fold the shirt speed: fast"


def test_dropout_rate_is_30_percent():
    rng = np.random.default_rng(0)
    out = [drop_indicator(True, rng) for _ in range(20_000)]
    assert abs(out.count(None) / len(out) - ADVANTAGE_DROPOUT) < 0.01 and False not in out  # never flips the sign


# ---------------------------------------------------------------- the sequence
def test_joint_layout_segments_order_and_losses(seq, sample):
    prompt, state, actions = sample
    toks, mask, seg, loss = seq.tokenize(prompt, state, layout="joint", actions=actions, subtask="grab the portafilter", advantage=True)
    n = int(mask.sum())
    real = seg[:n]
    assert toks[0] == BOS_ID and toks[n - 1] == EOS_ID
    # order PREFIX < SUBTASK < ADVANTAGE < ACTION, each contiguous
    order = [SEG_PREFIX, SEG_SUBTASK, SEG_ADVANTAGE, SEG_ACTION]
    boundaries = [int((real == s).sum()) for s in order]
    assert all(b > 0 for b in boundaries)
    assert real.tolist() == sum(([s] * b for s, b in zip(order, boundaries)), [])
    # CE on subtask + FAST only; the advantage token is an input (Sec. V-B "only the action log-likelihoods are affected")
    assert loss[:n].tolist() == [False] * boundaries[0] + [True] * boundaries[1] + [False] * boundaries[2] + [True] * boundaries[3]
    assert not mask[n:].any() and not loss[n:].any()
    # the advantage segment is exactly the wording
    adv_ids = toks[(seg == SEG_ADVANTAGE) & mask].tolist()
    assert seq.text.decode(adv_ids) == "Advantage: positive\n"
    # expert reads prefix + subtask + advantage, never the FAST tokens
    vis = expert_visible(seg, mask)
    assert vis.sum() == sum(boundaries[:3]) and not vis[(seg == SEG_ACTION)].any()


def test_optional_segments_and_flow_marker(seq, sample):
    prompt, state, actions = sample
    base, m0, s0, _ = seq.tokenize(prompt, state, layout="value")
    n0 = int(m0.sum())
    assert (s0[:n0] == SEG_PREFIX).all()
    # dropped advantage and no subtask: joint = prefix + FAST only
    t, m, s, _ = seq.tokenize(prompt, state, layout="joint", actions=actions, advantage=None)
    n = int(m.sum())
    assert t[:n0].tolist() == base[:n0].tolist() and set(s[:n].tolist()) == {SEG_PREFIX, SEG_ACTION}
    # flow: prefix (+ subtask + advantage) + 'Action: ' marker, no loss anywhere
    t, m, s, loss = seq.tokenize(prompt, state, layout="flow", subtask="grab the portafilter", advantage=False)
    n = int(m.sum())
    assert s[n - 1] == SEG_MARKER and t[n - len(seq._action_marker) : n].tolist() == seq._action_marker and not loss.any()  # inference: no loss
    assert expert_visible(s, m).sum() == n  # everything in a flow sequence is visible to the expert
    # value / hl_prompt: identical prefix-only sequences
    t2, m2, _, _ = seq.tokenize(prompt, state, layout="hl_prompt")
    assert t2.tolist() == base.tolist() and m2.tolist() == m0.tolist()


def test_text_layout_and_inverses(seq, sample):
    prompt, state, actions = sample
    t, m, s, loss = seq.tokenize(prompt, state, layout="text", target_text="a dog catches a frisbee")
    n = int(m.sum())
    assert (s[:n][loss[:n]] == SEG_TEXT).all() and t[n - 1] == EOS_ID
    assert seq.extract_text(t[(s == SEG_TEXT) & m]) == "a dog catches a frisbee"
    t, m, s, _ = seq.tokenize(prompt, state, layout="joint", actions=actions, subtask="grab the portafilter", advantage=True)
    rec = seq.extract_actions(t, H, D)
    assert rec.shape == (H, D) and np.abs(rec - actions).max() < 0.5 / seq.fast.scale * np.sqrt(H) * 2
    gen = t[(s == SEG_SUBTASK) & m]  # what a decoder would emit for the subtask segment
    assert seq.extract_subtask(gen) == "grab the portafilter"
    assert seq.extract_subtask(seq.text.encode("grab the portafilter", add_eos=True)) == "grab the portafilter"  # marker skipped


def test_truncation_keeps_max_len(seq, sample):
    _, state, _ = sample
    with pytest.warns(UserWarning):
        t, m, _, _ = seq.tokenize("x" * 300, state, layout="value")
    assert t.shape == (MAX_TOKEN_LEN,) and m.all()


# ---------------------------------------------------------------- RL labels
def test_reward_return_and_bins():
    r = episode_rewards(5, True, c_fail=100.0)
    assert r.tolist() == [-1, -1, -1, -1, 0] and returns(r).tolist() == [-4, -3, -2, -1, 0]  # Eq. 5, R_t = -(T - t)
    r = episode_rewards(5, False, c_fail=100.0)
    assert r[-1] == -100.0 and returns(r)[0] == -104.0
    v = normalize_return(returns(episode_rewards(5, True, 100.0)), max_episode_len=8)
    assert np.allclose(v, [-0.5, -0.375, -0.25, -0.125, 0.0])  # per-task normalisation by T_max
    assert normalize_return(np.array([-104.0]), 8).tolist() == [-1.0]  # failure clipped to the bottom
    vals = bin_values()
    assert len(vals) == NUM_BINS == 201 and vals[0] == -1.0 and vals[-1] == 0.0 and np.allclose(np.diff(vals), 0.005)
    b = value_to_bin(np.array([-1.0, -0.5, 0.0, -0.0026, -0.0024]))
    assert b.tolist() == [0, 100, 200, 199, 200]  # nearest bin
    x = np.linspace(-1, 0, 1001)
    assert np.abs(bin_to_value(value_to_bin(x)) - x).max() <= 0.5 / (NUM_BINS - 1) + 1e-6
    lab = EpisodeLabels("t", True, 40, 12, is_correction=np.zeros(12, bool))
    v, bins = lab.value_targets(c_fail=40.0)
    assert v[-1] == 0.0 and bins[-1] == 200 and bins[0] == value_to_bin(np.array([-11 / 40]))[0]


# ---------------------------------------------------------------- the batch
def test_batch_contract(seq):
    rng = np.random.default_rng(2)
    raw = {"images": {"base_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8), "right_wrist_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8)},
           "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32),
           "actions": (0.3 * make_smooth_chunks(B, H + 2, D, rng)).astype(np.float32),
           "prompt": ["make a double espresso", "fold the shirt"]}
    obs, actions = build_pi06_batch(raw, unit_stats(D), seq, layout="joint", image_keys=IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=False,
                                    subtasks=["grab the portafilter", None], advantages=[True, None], metadata=[None, "speed: fast"])
    assert list(obs.images) == list(IMAGE_KEYS) and len(IMAGE_KEYS) == 4 and set(STATIC_IMAGE_KEYS) < set(IMAGE_KEYS)
    assert tuple(obs.images["base_0_rgb"].shape) == (B, *IMAGE_RESOLUTION, 3) and IMAGE_RESOLUTION == (448, 448)
    assert [bool(obs.image_masks[k][0]) for k in IMAGE_KEYS] == [True, False, False, True] and bool((obs.images["base_1_rgb"] == -1).all())
    assert obs.state.shape == (B, ACTION_DIM) and actions.shape == (B, H, ACTION_DIM) and bool((actions[:, :, D:] == 0).all())
    assert obs.tokens.shape == (B, MAX_TOKEN_LEN) and obs.has_actions.tolist() == [True, True] and obs.advantage.tolist() == [1, -1]
    assert bool((obs.segment[1] != SEG_SUBTASK).all()) and bool((obs.segment[1] != SEG_ADVANTAGE).all())  # sample 1: no subtask, dropped advantage
    assert obs.expert_visible.shape == (B, MAX_TOKEN_LEN) and not bool(obs.expert_visible[0][obs.segment[0] == SEG_ACTION].any())
    rec = seq.extract_actions(obs.tokens[0].numpy(), H, D)
    assert np.abs(rec - actions[0, :, :D].numpy()).max() < 0.5 / seq.fast.scale * np.sqrt(H) * 2
    assert "speed: fast" in seq.text.decode(obs.tokens[1].tolist())
    vobs, _ = build_pi06_batch(raw, unit_stats(D), seq, layout="value", action_horizon=H, delta_mask=None, train=False, value_bins=[200, 37])
    assert vobs.value_bin.tolist() == [200, 37] and not vobs.has_actions.any() and not vobs.loss_mask.any()


def test_paper_robot_fits_the_budget(seq):
    state = np.zeros(14, np.float32)  # two 6-DoF arms + two grippers (paper Fig. 5)
    _, m, _, _ = seq.tokenize("assemble the box", state, layout="flow", subtask="fold the left flap inwards", advantage=True, metadata="speed: fast")
    assert 48 < int(m.sum()) < MAX_TOKEN_LEN
