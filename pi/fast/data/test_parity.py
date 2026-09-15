"""Alignment checks for the pi0-FAST data path: sequence layout, the three masks, vocabulary-tail mapping, state binning,
round trip through extract_actions, padding / truncation, FAST camera-slot rules, and agreement with the shared pi0 steps."""

import numpy as np
import pytest
import torch

from pi.fast.data.data import (
    ACTION_DIM,
    BOS_ID,
    EOS_ID,
    FAST_SKIP_TOKENS,
    IMAGE_KEYS,
    MAX_TOKEN_LEN,
    PALIGEMMA_VOCAB_SIZE,
    ByteTextCodec,
    FASTSequenceTokenizer,
    build_fast_batch,
    discretize_state,
    fast_to_paligemma,
    paligemma_to_fast,
    state_to_text,
    tiny_fast_tokenizer,
)
from pi.fast.tokenizer.tokenizer import QuantileStats, make_smooth_chunks, normalize_quantile
from pi.pi0.data.data import make_bool_mask, normalize, to_delta_actions

H, D = 10, 7


@pytest.fixture(scope="module")
def seq():
    return FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, D), max_len=180)


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    return make_smooth_chunks(1, 1, D, rng)[0, 0], make_smooth_chunks(1, H, D, rng)[0]


def test_constants_match_upstream():
    assert PALIGEMMA_VOCAB_SIZE == 257_152 and FAST_SKIP_TOKENS == 128 and MAX_TOKEN_LEN == 250
    assert BOS_ID == 2 and EOS_ID == 1 and ACTION_DIM == 32
    assert IMAGE_KEYS == ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")


def test_state_binning_edges_and_range():
    x = np.array([-1.0, -1.0 + 1e-6, -1.0 + 2 / 256, 0.0, 1.0 - 2 / 256, 1.0 - 1e-6, 1.0, 1.7, -3.0])
    b = discretize_state(x)
    assert b.tolist() == [0, 0, 1, 128, 255, 255, 255, 255, -1]  # last bin is open; below -1 gives -1 (upstream quirk)
    assert state_to_text(np.array([-1.0, 0.0, 1.0])) == "0 128 255"


def test_vocabulary_tail_mapping_is_an_involution_in_the_right_range():
    t = np.arange(2048)
    pg = fast_to_paligemma(t)
    assert pg.min() == 257_152 - 1 - 128 - 2047 == 254_976 and pg.max() == 257_023
    assert np.array_equal(paligemma_to_fast(pg), t)
    assert pg.max() < PALIGEMMA_VOCAB_SIZE - FAST_SKIP_TOKENS  # never touches the 128 special tokens


def test_sequence_layout_and_masks(seq, sample):
    state, actions = sample
    tok, m, ar, lm = seq.tokenize("Pick_up the Block", state, actions)
    assert tok.shape == m.shape == ar.shape == lm.shape == (180,)
    n = int(m.sum())
    assert tok[0] == BOS_ID and tok[n - 1] == EOS_ID
    codec = ByteTextCodec()
    prefix_len = int(((ar == 0) & m).sum())
    assert codec.decode(tok[:prefix_len].tolist()) == f"Task: pick up the block, State: {state_to_text(state)};\n"
    marker = codec.encode("Action: ")
    assert tok[prefix_len : prefix_len + len(marker)].tolist() == marker
    assert tok[n - 2] == codec.encode("|")[0]
    # masks: prefix (bidirectional, no loss) then postfix (causal, loss), nothing after the real tokens
    assert np.all(ar[:prefix_len] == 0) and np.all(ar[prefix_len:n] == 1) and np.all(ar[n:] == 0)
    assert not lm[:prefix_len].any() and lm[prefix_len:n].all() and not lm[n:].any()
    assert np.array_equal(lm, (ar == 1))  # loss exactly on the causal part
    # the action ids sit in the PaliGemma tail
    act = tok[prefix_len + len(marker) : n - 2]
    assert act.min() >= 254_976 and act.max() <= 257_023 and len(act) > 0


def test_inference_mode_has_empty_postfix(seq, sample):
    state, _ = sample
    tok, m, ar, lm = seq.tokenize("close the drawer", state, None)
    n = int(m.sum())
    assert not ar.any() and not lm.any()
    assert tok[n - 1] != EOS_ID  # no EOS: the model must produce the postfix and its own EOS


def test_round_trip_through_extract_actions(seq, sample):
    state, actions = sample
    tok, m, ar, lm = seq.tokenize("x", state, actions)
    rec = seq.extract_actions(tok, H, D)
    assert rec.shape == (H, D) and rec.dtype == np.float32
    assert np.abs(rec - actions).max() <= 0.5 / seq.fast.scale * np.sqrt(H) + 1e-6  # only FAST rounding
    # only the postfix is needed, pad and EOS are tolerated
    postfix = tok[m & (ar == 1)]
    assert np.allclose(seq.extract_actions(postfix, H, D), rec)
    assert np.allclose(seq.extract_actions(np.concatenate([postfix, np.zeros(20, dtype=np.int64)]), H, D), rec)


def test_extract_actions_falls_back_to_zeros(seq, sample):
    state, actions = sample
    tok, m, ar, lm = seq.tokenize("x", state, actions)
    prefix_only = tok[(ar == 0) & m]
    assert np.all(seq.extract_actions(prefix_only, H, D) == 0)  # no 'Action: ' marker
    bad = tok.copy()
    bad[int(((ar == 0) & m).sum()) + len(seq._action_marker) + 1] = ByteTextCodec().encode("a")[0]  # text inside the span
    assert np.all(seq.extract_actions(bad, H, D) == 0)


def test_padding_and_truncation(seq, sample):
    state, actions = sample
    short = FASTSequenceTokenizer(ByteTextCodec(), seq.fast, max_len=40)
    with pytest.warns(UserWarning, match="exceeds max length"):
        tok, m, ar, lm = short.tokenize("a very long prompt " * 3, state, actions)
    assert tok.shape == (40,) and m.all()
    tok, m, ar, lm = seq.tokenize("a", state, None)
    assert tok.shape == (180,) and tok[~m].sum() == 0 and not ar[~m].any() and not lm[~m].any()


def test_build_fast_batch_slots_masks_and_shared_steps(seq):
    rng = np.random.default_rng(1)
    B = 2
    raw = {
        "images": {"base_0_rgb": rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8), "wrist_0_rgb": rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8)},
        "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32),
        "actions": rng.uniform(-0.5, 0.5, (B, H + 2, D)).astype(np.float32),
        "prompt": ["a", "b"],
    }
    stats = {"state": QuantileStats(np.full(D, -1.0), np.full(D, 1.0)), "actions": QuantileStats(np.full(D, -1.0), np.full(D, 1.0))}
    mask = make_bool_mask(6, -1)
    obs, actions = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=mask, train=False)
    assert set(obs.images) == set(IMAGE_KEYS)
    for k in IMAGE_KEYS:
        assert obs.images[k].shape == (B, 224, 224, 3) and obs.image_masks[k].all()  # FAST: never masked
    assert torch.all(obs.images["base_1_rgb"] == -1.0)  # missing camera = black image
    assert obs.state.shape == (B, ACTION_DIM) and actions.shape == (B, H, ACTION_DIM)
    for name in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
        assert getattr(obs, name).shape == (B, 180)
    # the tokens encode exactly the delta + quantile-normalized native-dim actions (rounding aside)
    expect = normalize_quantile(to_delta_actions(raw["state"], raw["actions"][:, :H], mask), stats["actions"])
    assert np.allclose(actions[:, :, :D].numpy(), expect, atol=1e-6)
    rec = seq.extract_actions(obs.tokenized_prompt[0].numpy(), H, D)
    assert np.abs(rec - expect[0]).max() <= 0.5 / seq.fast.scale * np.sqrt(H) + 1e-6
    # quantile stats with q01=-1, q99=1 are the identity up to 1e-6, so pi0's z-score path must NOT be what we used
    z = normalize(raw["state"], type("S", (), {"mean": np.zeros(D), "std": np.ones(D)})())
    assert np.allclose(obs.state[:, :D].numpy(), z, atol=1e-5)  # identity stats coincide; documents the switch point
