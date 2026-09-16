"""Alignment checks for the pi0.5 data pipeline: the four layouts share one prefix, the flow prefix is the FAST prefix
plus the marker, both inverses round-trip, the no-state fallback, the camera mask rule, HL text parsing, the 19-dim
prompt budget, and the batch contract."""

import numpy as np
import pytest

from pi.fast.data.data import BOS_ID, EOS_ID, discretize_state
from pi.fast.tokenizer.tokenizer import make_smooth_chunks
from pi.pi0.data.data import make_bool_mask
from pi.pi05.data.data import (
    ACTION_DIM,
    HL_IMAGE_KEYS,
    LL_IMAGE_KEYS,
    MAX_TOKEN_LEN,
    MOBILE_IMAGE_KEYS,
    build_pi05_batch,
    hl_target_text,
    parse_hl_text,
    state_prefix_text,
    tiny_pi05_tokenizer,
    unit_stats,
    with_control_mode,
)

H, D, B = 10, 7, 2


@pytest.fixture(scope="module")
def seq():
    return tiny_pi05_tokenizer(H, D)


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(1)
    state = rng.uniform(-0.5, 0.5, D).astype(np.float32)
    actions = (0.3 * make_smooth_chunks(1, H, D, rng)[0]).astype(np.float32)
    return "Pick_up the red\nblock", state, actions


# ---------------------------------------------------------------- prefix text
def test_prefix_text_matches_upstream_format(sample):
    prompt, state, _ = sample
    text = state_prefix_text(prompt, state)
    bins = " ".join(map(str, discretize_state(state)))
    assert text == f"Task: Pick up the red block, State: {bins};\n"  # tokenizer.py L23-L28: strip, '_' and '\n' -> ' ', no lower()
    assert state_prefix_text(prompt, None) == "Pick up the red block\n"  # L30-L33 pi0 branch (pi05_libero)


def test_control_mode_tag_wording():
    assert with_control_mode("clean the kitchen", "joint") == "clean the kitchen <control mode> joint <control mode>"
    with pytest.raises(AssertionError):
        with_control_mode("x", "velocity")


# ---------------------------------------------------------------- the four layouts
def test_layouts_share_the_prefix_and_flow_adds_only_the_marker(seq, sample):
    prompt, state, actions = sample
    out = {layout: seq.tokenize(prompt, state, layout=layout, actions=actions, target_text="Subtask: x") for layout in ("flow", "fast", "text", "hl_prompt")}
    n_prefix = int((out["fast"][2] == 0).sum() - (~out["fast"][1]).sum())  # ar 0 and real
    base = out["hl_prompt"][0][: int(out["hl_prompt"][1].sum())].tolist()
    assert base[0] == BOS_ID
    for layout in ("fast", "text"):
        toks, mask, ar, loss = out[layout]
        assert toks[:n_prefix].tolist() == base and not ar[:n_prefix].any() and not loss[:n_prefix].any()
        n_real = int(mask.sum())
        assert (ar[n_prefix:n_real] == 1).all() and loss[n_prefix:n_real].all() and toks[n_real - 1] == EOS_ID
        assert not mask[n_real:].any() and not loss[n_real:].any()
    flow_toks, flow_mask, flow_ar, flow_loss = out["flow"]
    n_flow = int(flow_mask.sum())
    assert flow_toks[:n_flow].tolist() == base + seq._action_marker  # tokenizer.py L28
    assert not flow_ar.any() and not flow_loss.any()  # everything is prefix, nothing gets a loss
    # "fast" postfix starts with the same marker ids, but as postfix (ar 1)
    fast_toks, _, fast_ar, _ = out["fast"]
    assert fast_toks[n_prefix : n_prefix + len(seq._action_marker)].tolist() == seq._action_marker and fast_ar[n_prefix] == 1


def test_both_inverses_round_trip(seq, sample):
    prompt, state, actions = sample
    toks, _, _, _ = seq.tokenize(prompt, state, layout="fast", actions=actions)
    rec = seq.extract_actions(toks, H, D)
    assert rec.shape == (H, D) and np.abs(rec - actions).max() < 0.5 / seq.fast.scale * np.sqrt(H) * 2  # FAST rounding bound
    target = hl_target_text("pick up the plate", [("plate", (0.4, 0.1, 0.9, 0.2)), ("sink", (0.0, 0.5, 0.3, 0.99))])
    toks, mask, ar, _ = seq.tokenize(prompt, state, layout="text", target_text=target)
    n_prefix = int(((ar == 0) & mask).sum())
    assert seq.extract_text(toks[n_prefix:]) == target
    assert seq.extract_text(toks) != target  # the prefix is not part of the answer


def test_hl_text_parse_is_inverse_and_loc_tokens_round_to_1024_bins():
    boxes = [("plate", (0.4, 0.1, 0.9, 0.2)), ("cutting board", (0.0, 0.5, 0.3, 0.99))]
    text = hl_target_text("pick up the plate", boxes)
    assert text.startswith("Bounding boxes: <loc0409><loc0102><loc0921><loc0204>plate <loc0000>") and text.endswith("\nSubtask: pick up the plate")
    sub, parsed = parse_hl_text(text)
    assert sub == "pick up the plate" and [b[0] for b in parsed] == ["plate", "cutting board"]
    assert all(abs(p - t) < 1 / 1024 for (_, pc), (_, tc) in zip(parsed, boxes) for p, t in zip(pc, tc))
    assert parse_hl_text("pick up the cup") == ("pick up the cup", [])  # no marker: whole text is the subtask
    assert hl_target_text("close the drawer") == "Subtask: close the drawer"


def test_truncation_warns_and_keeps_max_len(seq, sample):
    prompt, state, actions = sample
    with pytest.warns(UserWarning):
        toks, mask, _, _ = seq.tokenize("x" * 300, state, layout="flow")
    assert toks.shape == (MAX_TOKEN_LEN,) and mask.all()


# ---------------------------------------------------------------- the batch
def test_batch_contract_and_camera_mask_rule(seq):
    rng = np.random.default_rng(2)
    raw = {"images": {"base_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8), "left_wrist_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8)},
           "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32),
           "actions": (0.3 * make_smooth_chunks(B, H + 2, D, rng)).astype(np.float32),
           "prompt": ["clean the kitchen", "make the bed"]}
    obs, actions = build_pi05_batch(raw, unit_stats(D), seq, layout="fast", image_keys=HL_IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=False)
    assert list(obs.images) == list(HL_IMAGE_KEYS) and set(LL_IMAGE_KEYS) < set(HL_IMAGE_KEYS) and HL_IMAGE_KEYS == MOBILE_IMAGE_KEYS
    assert [bool(obs.image_masks[k][0]) for k in HL_IMAGE_KEYS] == [True, False, True, False]  # missing slots are masked (pi0 rule, not FAST's)
    assert bool((obs.images["base_1_rgb"] == -1).all())  # black image in model range
    assert obs.state.shape == (B, ACTION_DIM) and actions.shape == (B, H, ACTION_DIM) and bool((actions[:, :, D:] == 0).all())
    assert obs.tokenized_prompt.shape == (B, MAX_TOKEN_LEN) and obs.token_loss_mask.any()
    # the FAST postfix encodes exactly the returned (normalized, delta) actions
    rec = seq.extract_actions(obs.tokenized_prompt[0].numpy(), H, D)
    assert np.abs(rec - actions[0, :, :D].numpy()).max() < 0.5 / seq.fast.scale * np.sqrt(H) * 2
    # discrete_state=False: no 'Task:' in the prompt (pi05_libero)
    obs2, _ = build_pi05_batch(raw, unit_stats(D), seq, layout="flow", action_horizon=H, delta_mask=None, train=False, discrete_state=False)
    assert seq.text.decode(obs2.tokenized_prompt[0].tolist()).startswith("clean the kitchen\n")


def test_19_dim_state_fits_the_200_budget(seq):
    state = np.zeros(19, np.float32)  # the paper's largest robot (Sec. IV-E)
    toks, mask, _, _ = seq.tokenize(with_control_mode("put the dishes in the sink", "joint"), state, layout="flow")
    n = int(mask.sum())
    assert n < MAX_TOKEN_LEN and n > 48  # would not fit pi0's 48 (byte codec; SentencePiece is shorter still)
