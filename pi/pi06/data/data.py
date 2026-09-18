"""pi0.6* data pipeline: one token sequence with five segments (prefix / subtask / advantage / FAST actions / text
target) and causal text, the binarised advantage token and its 30% dropout, the metadata field, four 448x448 camera
slots, the three KI sample kinds and their loss masks, and the RL labels: Eq. 5 rewards -> per-task normalised
returns -> 201-bin value targets.

Re-implementation (NumPy / PyTorch). Sources of truth:
  paper    pi0.6* arXiv:2511.14759v2 Sec. IV-A (B = 201 bins, Eq. 1), IV-B (Eq. 3, I_t, corrections forced True),
           V-A (ell = ell_t + s metadata; factorisation ell_hat -> a^ell -> a; expert does not read FAST tokens),
           V-B ("Advantage: positive / negative" after ell_hat, before the actions), V-C (Eq. 5 reward, values
           normalised to (-1, 0) per task by the maximum episode length), V-D (I_t = True during SFT),
           Appendix F (30% advantage dropout; thresholds)
  card     pi0.6 model card (2025-11-17) Sec. 2: up to four 448x448 images (base, up to two wrists, optional backward
           camera), bidirectional image tokens, CAUSAL text tokens, bidirectional action tokens, metadata in the prompt
  KI       Knowledge Insulation arXiv:2505.23705v1 Sec. 5.1 (Eq. 4: M^ell / M^act loss masks; three sample kinds:
           VLM data, action-only data, action + language data), Appendix B (FAST tokens attend prefix and earlier
           FAST tokens; expert attends prefix and itself, never the FAST tokens)
  openpi   https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 ships NO pi0.6
           code; the prefix text, quantile normalisation, FAST postfix and camera-mask rule are the pi0.5 / FAST ones
           (pi.pi05.data, pi.fast.data), unchanged
License: Apache-2.0 for the openpi pieces this file builds on. Re-implements, does not copy.

Everything unchanged from pi0.5 is imported: `state_prefix_text` (Task / State prefix), `hl_target_text` (the
'Subtask: ...' wording), the quantile / delta / padding / augmentation chain. This file only holds the pi0.6* increment.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Sequence

import numpy as np
import torch

from pi.fast.data.data import BOS_ID, EOS_ID, ByteTextCodec, FASTSequenceTokenizer, fast_to_paligemma
from pi.fast.tokenizer.tokenizer import FASTTokenizer, QuantileStats, normalize_quantile
from pi.pi0.data.data import augment, pad_to_dim, resize_with_pad, to_delta_actions, uint8_to_model_range
from pi.pi05.data.data import MOBILE_IMAGE_KEYS, clean_prompt, state_prefix_text

# --------------------------------------------------------------------------------------
# Constants.
# --------------------------------------------------------------------------------------
IMAGE_RESOLUTION = (448, 448)  # model card Sec. 2: "each having resolution 448x448" (pi0.5: 224)
IMAGE_KEYS = MOBILE_IMAGE_KEYS  # card Sec. 2: base + up to two wrists + optional backward camera = the 4 pi0.5 slots
STATIC_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # paper Fig. 5: base between the arms + two wrists
ACTION_DIM = 32  # pi0.5 convention kept; pi0.6 undisclosed (README Sec. 8)
ACTION_HORIZON = 50  # paper Sec. V-A: "action chunks a_{t:t+H} ... at 50 Hz"; H itself undisclosed, pi0.5's 50 kept (README Sec. 8)
MAX_TOKEN_LEN = 200  # undisclosed for pi0.6; pi0.5's 200 kept (README Sec. 8)
ADVANTAGE_TEXT = {True: "Advantage: positive", False: "Advantage: negative"}  # paper Sec. V-B, verbatim
ADVANTAGE_DROPOUT = 0.3  # paper Appendix F: "randomly drop out the conditioning on the advantage indicator 30% of the time"
NUM_BINS = 201  # paper Sec. IV-A: "B = 201 bins"
VALUE_RANGE = (-1.0, 0.0)  # paper Sec. V-C: "we normalize the values predicted to be between (-1, 0)"

# Segment ids over the token axis. Text is causal everywhere (card Sec. 2); the ids say who may READ a segment
# (train.py / backbone build the masks from them) and which segments carry a cross-entropy loss.
SEG_PREFIX = 0  # [BOS] Task: <prompt + metadata>, State: <bins>;\n         input; the value function reads exactly this
SEG_SUBTASK = 1  # Subtask: <ell_hat>\n                                     CE target (log pi(ell_hat | o, ell)), expert reads it
SEG_ADVANTAGE = 2  # Advantage: positive|negative\n                          input only (no loss), expert reads it
SEG_ACTION = 3  # Action: <FAST ids> | [EOS]                                  CE target (log pi(a^ell | ...)), expert must NOT read it
SEG_TEXT = 4  # <free text target> [EOS]                                      CE target of a VLM co-training sample
SEG_MARKER = 5  # 'Action: ' at the end of the "flow" layout                  input; expert reads it (pi0.5 tokenizer.py L28)
EXPERT_READS = (SEG_PREFIX, SEG_SUBTASK, SEG_ADVANTAGE, SEG_MARKER)  # paper Sec. V-A, KI App. B
CE_SEGMENTS = (SEG_SUBTASK, SEG_ACTION, SEG_TEXT)
LAYOUTS = ("joint", "flow", "text", "value", "hl_prompt")


# --------------------------------------------------------------------------------------
# 1. Prompt text. Paper Sec. V-A: ell = ell_t + s, "additional language inputs s providing metadata that further
#    modulates how the task is performed"; card Sec. 2 "conditioning metadata in the prompt". Format undisclosed
#    (README Sec. 8): this repo appends it after a space, like pi0.5's control-mode tag.
# --------------------------------------------------------------------------------------
def with_metadata(prompt: str, metadata: str | None) -> str:
    return clean_prompt(prompt) if not metadata else f"{clean_prompt(prompt)} {clean_prompt(metadata)}"


def subtask_text(subtask: str) -> str:
    """'Subtask: <ell_hat>\\n'. Wording from pi0.5 Fig. 4 (pi.pi05.data.hl_target_text); the trailing newline is this
    repo's separator before the advantage / action segments (README Sec. 8)."""
    return f"Subtask: {clean_prompt(subtask)}\n"


def advantage_text(indicator: bool | None) -> str:
    """'Advantage: positive\\n' / 'Advantage: negative\\n' (Sec. V-B); None = the indicator is dropped (App. F)."""
    return "" if indicator is None else ADVANTAGE_TEXT[bool(indicator)] + "\n"


def drop_indicator(indicator: bool, rng: np.random.Generator, p: float = ADVANTAGE_DROPOUT) -> bool | None:
    """Appendix F: with probability p the token is omitted, so the same model represents pi(a | o, ell) (uncond) and
    pi(a | I, o, ell) (cond). This replaces the loss multiplier alpha of Eq. 3 and enables CFG at inference."""
    return None if rng.random() < p else bool(indicator)


# --------------------------------------------------------------------------------------
# 2. The sequence. Five segments, all causal. pi0.5 had prefix (bidirectional) + one postfix; pi0.6 (Sec. V-A)
#    factorises log pi(ell_hat | o, ell) + log pi(a^ell | o, ell, ell_hat) + log pi(a | o, ell, ell_hat) and puts the
#    advantage token between ell_hat and the two action heads (Sec. V-B), so a training sample reads
#
#    [BOS] Task: p s, State: b;\n | Subtask: ell_hat\n | Advantage: positive\n | Action: <FAST> | [EOS]      "joint"
#      SEG_PREFIX                   SEG_SUBTASK (CE)     SEG_ADVANTAGE            SEG_ACTION (CE)
#    and the 50 continuous action tokens of the expert (train.py) read PREFIX + SUBTASK + ADVANTAGE only.
#
#    layout      after the prefix                                    use
#    "joint"     [Subtask]? [Advantage]? Action: FAST | EOS           KI training sample with actions (MSE on the expert too)
#    "flow"      [Subtask]? [Advantage]? 'Action: ' (SEG_MARKER)      inference / flow-only finetune; no loss
#    "text"      <target> EOS  (SEG_TEXT)                             VLM co-training (caption / VQA / bbox / a bare subtask)
#    "value"     (nothing)                                            value-function input: the same ell, no subtask, no advantage
#    "hl_prompt" (nothing)                                            the model writes 'Subtask: ...\n' itself (../infer)
# --------------------------------------------------------------------------------------
class Pi06SequenceTokenizer:
    def __init__(self, text: ByteTextCodec, fast: FASTTokenizer | None = None, max_len: int = MAX_TOKEN_LEN):
        self.text, self.fast, self.max_len = text, fast, max_len
        self._action_marker = text.encode("Action: ")  # pi0.5 tokenizer.py L28 / FAST L84
        self._fast_seq = None if fast is None else FASTSequenceTokenizer(text, fast, max_len)
        self.newline_id = text.encode("\n")[0]

    def tokenize(self, prompt: str, state: np.ndarray | None, *, layout: str, actions: np.ndarray | None = None,
                 subtask: str | None = None, advantage: bool | None = None, target_text: str | None = None,
                 metadata: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One sample -> (tokens i64[max_len], token_mask bool[max_len], segment i64[max_len], loss_mask bool[max_len]).
        state float[d] normalised or None (then the pi0 '<prompt>\\n' prefix, as in pi0.5); actions float[H, d]
        normalised for "joint"; advantage True / False / None (None = dropped, or the unconditional branch)."""
        assert layout in LAYOUTS, layout
        ids = self.text.encode(state_prefix_text(with_metadata(prompt, metadata), state), add_bos=True)
        seg = [SEG_PREFIX] * len(ids)
        loss = [False] * len(ids)

        def add(piece: list[int], s: int, l: bool):
            ids.extend(piece), seg.extend([s] * len(piece)), loss.extend([l] * len(piece))

        if layout in ("joint", "flow"):
            if subtask is not None:
                add(self.text.encode(subtask_text(subtask)), SEG_SUBTASK, layout == "joint")  # CE only when training
            if advantage is not None:
                add(self.text.encode(advantage_text(advantage)), SEG_ADVANTAGE, False)
            if layout == "joint":
                assert actions is not None and self.fast is not None
                fast_ids = self.fast(np.asarray(actions)[None])[0]
                add(self._action_marker + fast_to_paligemma(fast_ids).tolist() + self.text.encode("|", add_eos=True), SEG_ACTION, True)
            else:
                add(list(self._action_marker), SEG_MARKER, False)
        elif layout == "text":
            assert target_text is not None
            add(self.text.encode(target_text, add_eos=True), SEG_TEXT, True)
        # "value" / "hl_prompt": prefix only
        mask = [True] * len(ids)
        n = len(ids)
        if n < self.max_len:
            pad = self.max_len - n
            ids, mask, seg, loss = ids + [0] * pad, mask + [False] * pad, seg + [SEG_PREFIX] * pad, loss + [False] * pad
        elif n > self.max_len:  # pi0.5 tokenizer.py L40-L48 behaviour kept
            warnings.warn(f"Token length ({n}) exceeds max length ({self.max_len}), truncating.", stacklevel=2)
            ids, mask, seg, loss = (x[: self.max_len] for x in (ids, mask, seg, loss))
        return np.asarray(ids, np.int64), np.asarray(mask, bool), np.asarray(seg, np.int64), np.asarray(loss, bool)

    def extract_subtask(self, tokens: np.ndarray) -> str:
        """Generated ids -> the subtask text: cut at the first newline (the segment separator) or EOS, strip the
        'Subtask: ' marker. A model that skipped the marker still gives a usable command."""
        ids = [int(t) for t in np.asarray(tokens).reshape(-1)]
        for stop in (self.newline_id, EOS_ID):
            if stop in ids:
                ids = ids[: ids.index(stop)]
        text = self.text.decode([t for t in ids if t not in (0, BOS_ID)]).strip()
        return text[len("Subtask:"):].strip() if text.startswith("Subtask:") else text

    def extract_text(self, tokens: np.ndarray) -> str:
        ids = [int(t) for t in np.asarray(tokens).reshape(-1)]
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID)]
        return self.text.decode([t for t in ids if t not in (0, BOS_ID)]).strip()

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        assert self._fast_seq is not None
        return self._fast_seq.extract_actions(tokens, action_horizon, action_dim)


def expert_visible(segment: np.ndarray | torch.Tensor, token_mask) -> np.ndarray | torch.Tensor:
    """Which token columns the 50 continuous action tokens may attend: valid PREFIX / SUBTASK / ADVANTAGE / MARKER,
    never ACTION (FAST) or TEXT. Paper Sec. V-A "The action expert does not receive these as input"; KI App. B."""
    if isinstance(segment, torch.Tensor):
        vis = torch.zeros_like(token_mask)
        for s in EXPERT_READS:
            vis |= segment == s
        return vis & token_mask
    vis = np.isin(segment, EXPERT_READS)
    return vis & np.asarray(token_mask)


# --------------------------------------------------------------------------------------
# 3. RL labels. Paper Sec. V-C, Eq. 5; Sec. IV-A (bins); Appendix F (which return the advantage uses is ../value).
#    An episode has steps t = 0..T (T = the last step). Reward: -1 per step, 0 at a successful end, -C_fail at a
#    failed end. Return R_t = sum_{t' >= t} r_t' = -(T - t) for a success, -(T - t) - C_fail for a failure. The value
#    function predicts R_t / T_max(ell), clipped to [-1, 0]: 0 = done, -1 = "as far from success as this task gets".
# --------------------------------------------------------------------------------------
def episode_rewards(num_steps: int, success: bool, c_fail: float) -> np.ndarray:
    """Eq. 5 over t = 0..T with T = num_steps - 1 -> f32[num_steps]. c_fail is undisclosed (README Sec. 8): pass it."""
    assert num_steps >= 1 and c_fail >= 0
    r = -np.ones(num_steps, np.float32)
    r[-1] = 0.0 if success else -float(c_fail)
    return r


def returns(rewards: np.ndarray) -> np.ndarray:
    """R_t = sum_{t' >= t} r_t' (no discount, Sec. III) -> f32[T + 1]."""
    return np.cumsum(np.asarray(rewards, np.float32)[::-1])[::-1].copy()


def normalize_return(R: np.ndarray, max_episode_len: int) -> np.ndarray:
    """Sec. V-C: per-task normalisation by the task's maximum episode length, into VALUE_RANGE. Clipping is this
    repo's reading of "between (-1, 0)" for failures whose C_fail pushes R below -T_max (README Sec. 8)."""
    lo, hi = VALUE_RANGE
    return np.clip(np.asarray(R, np.float32) / float(max_episode_len), lo, hi)


def bin_values(num_bins: int = NUM_BINS) -> np.ndarray:
    """v(b) for b = 0..B-1: uniform over VALUE_RANGE (v(0) = -1, v(200) = 0). Uniform spacing is undisclosed (README Sec. 8)."""
    return np.linspace(VALUE_RANGE[0], VALUE_RANGE[1], num_bins, dtype=np.float32)


def value_to_bin(v: np.ndarray, num_bins: int = NUM_BINS) -> np.ndarray:
    """Nearest bin index i64 for normalised values in VALUE_RANGE (Sec. IV-A 'discretizing the empirical return')."""
    lo, hi = VALUE_RANGE
    frac = (np.clip(np.asarray(v, np.float32), lo, hi) - lo) / (hi - lo)
    return np.rint(frac * (num_bins - 1)).astype(np.int64)


def bin_to_value(b: np.ndarray, num_bins: int = NUM_BINS) -> np.ndarray:
    return bin_values(num_bins)[np.asarray(b, np.int64)]


@dataclasses.dataclass
class EpisodeLabels:
    """What the data collection of Sec. IV (step 1) attaches to an episode: an outcome label (human), the task's
    max length (task table), and per-step 'this action came from a human correction' flags (teleop log)."""

    task: str
    success: bool
    max_episode_len: int
    num_steps: int
    is_correction: np.ndarray | None = None  # bool[num_steps]; None = fully autonomous or a demonstration

    def value_targets(self, c_fail: float) -> tuple[np.ndarray, np.ndarray]:
        """-> (normalised return f32[num_steps], bin i64[num_steps]) = the training target of Eq. 1 at every step."""
        v = normalize_return(returns(episode_rewards(self.num_steps, self.success, c_fail)), self.max_episode_len)
        return v, value_to_bin(v)


# --------------------------------------------------------------------------------------
# 4. The batch. Same chain as pi0.5 (camera slots -> delta -> quantile -> resize -> tokenize -> pad dims -> [-1, 1]
#    + augmentation), with 448x448 images, the new sequence, and the RL fields.
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Pi06Observation:
    """images:      {slot: f32[B, 448, 448, 3]} in [-1, 1]      image_masks: {slot: bool[B]} (missing slot -> False)
    state:         f32[B, 32] normalised, zero-padded (not read by the model; kept for the inverse transforms)
    tokens:        i64[B, max_len]   token_mask: bool[B, max_len]   segment: i64[B, max_len] (SEG_*)   loss_mask: bool[B, max_len]
    has_actions:   bool[B]  M^act of KI Eq. 4: the expert MSE applies to this sample
    advantage:     i8[B]    1 positive, 0 negative, -1 dropped / absent (what the sequence already encodes; for bookkeeping)
    value_bin:     i64[B] or None  Eq. 1 target for a "value" layout batch"""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokens: torch.Tensor
    token_mask: torch.Tensor
    segment: torch.Tensor
    loss_mask: torch.Tensor
    has_actions: torch.Tensor
    advantage: torch.Tensor
    value_bin: torch.Tensor | None = None

    @property
    def expert_visible(self) -> torch.Tensor:
        return expert_visible(self.segment, self.token_mask)


def build_pi06_batch(
    raw: dict,
    norm_stats: dict[str, QuantileStats] | None,
    seq: Pi06SequenceTokenizer,
    *,
    layout: str,
    image_keys: Sequence[str] = STATIC_IMAGE_KEYS,
    action_horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
    delta_mask: Sequence[bool] | None,
    train: bool,
    subtasks: Sequence[str | None] | None = None,
    advantages: Sequence[bool | None] | None = None,
    target_text: Sequence[str] | None = None,
    value_bins: Sequence[int] | None = None,
    metadata: Sequence[str | None] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Pi06Observation, torch.Tensor | None]:
    """raw = {"images": {slot: uint8[B, h, w, 3]} (subset of image_keys), "state": f32[B, d], "actions": f32[B, >=H, d]
    (absolute; "joint" only), "prompt": [str] * B}. Returns (Pi06Observation, actions f32[B, H, 32] normalised /
    delta / padded or None). advantages[i] None = token omitted (dropout or the unconditional CFG branch)."""
    b = raw["state"].shape[0]
    state = np.asarray(raw["state"], np.float32)
    actions = None if "actions" not in raw else np.asarray(raw["actions"], np.float32)[:, :action_horizon]
    # (a) camera slots: missing -> black + mask False (pi0.5 rule)
    ref = np.asarray(next(iter(raw["images"].values())))
    images = {k: np.asarray(raw["images"][k]) if k in raw["images"] else np.zeros_like(ref) for k in image_keys}
    image_masks = {k: np.full((b,), k in raw["images"], bool) for k in image_keys}
    # (b) delta, (c) quantile: unchanged
    if actions is not None:
        actions = to_delta_actions(state, actions, delta_mask)
    if norm_stats is not None:
        state = normalize_quantile(state, norm_stats["state"])
        if actions is not None:
            actions = normalize_quantile(actions, norm_stats["actions"])
    # (d) resize with pad to 448 (card Sec. 2)
    images_t = {k: resize_with_pad(torch.from_numpy(v), *IMAGE_RESOLUTION) for k, v in images.items()}
    # (e) the sequence
    per = [seq.tokenize(raw["prompt"][i], state[i], layout=layout, actions=None if actions is None else actions[i],
                        subtask=None if subtasks is None else subtasks[i], advantage=None if advantages is None else advantages[i],
                        target_text=None if target_text is None else target_text[i], metadata=None if metadata is None else metadata[i])
           for i in range(b)]
    tokens, token_mask, segment, loss_mask = (np.stack(x) for x in zip(*per))
    # (f) pad dims to 32
    state = pad_to_dim(state, action_dim).astype(np.float32)
    if actions is not None:
        actions = pad_to_dim(actions, action_dim).astype(np.float32)
    # (g) [-1, 1] + train-time augmentation (pi0 Appendix E parameters; pi0.6's undisclosed, README Sec. 8)
    for k in image_keys:
        img = uint8_to_model_range(images_t[k])
        images_t[k] = augment(img, k, generator) if train else img
    adv = np.full((b,), -1, np.int8)
    if advantages is not None:
        adv = np.asarray([-1 if a is None else int(bool(a)) for a in advantages], np.int8)
    obs = Pi06Observation(
        images=images_t,
        image_masks={k: torch.from_numpy(v) for k, v in image_masks.items()},
        state=torch.from_numpy(state),
        tokens=torch.from_numpy(tokens),
        token_mask=torch.from_numpy(token_mask),
        segment=torch.from_numpy(segment),
        loss_mask=torch.from_numpy(loss_mask),
        has_actions=torch.full((b,), layout == "joint", dtype=torch.bool),
        advantage=torch.from_numpy(adv),
        value_bin=None if value_bins is None else torch.as_tensor(np.asarray(value_bins, np.int64)),
    )
    return obs, None if actions is None else torch.from_numpy(actions)


# --------------------------------------------------------------------------------------
# 5. Tiny helpers and a walk-through.  uv run python -m pi.pi06.data.data
# --------------------------------------------------------------------------------------
def tiny_pi06_tokenizer(horizon: int, dim: int, seed: int = 0, max_len: int = MAX_TOKEN_LEN) -> Pi06SequenceTokenizer:
    from pi.fast.data.data import tiny_fast_tokenizer

    return Pi06SequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(horizon, dim, seed=seed), max_len=max_len)


def unit_stats(dim: int) -> dict[str, QuantileStats]:
    return {"state": QuantileStats(np.full(dim, -1.0), np.full(dim, 1.0)), "actions": QuantileStats(np.full(dim, -1.0), np.full(dim, 1.0))}


C_FAIL_TINY = 40.0  # tiny only, not a paper value: C_fail is undisclosed (README Sec. 8); >= T_max makes every failed step clip to -1


def main() -> None:
    from pi.pi0.data.data import make_bool_mask

    rng = np.random.default_rng(0)
    B, H, d = 2, 10, 7  # tiny 7-dim arm, H = 10 (the paper robot: 14 dims, H = 50)
    raw = {
        "images": {k: rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
        "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
        "actions": rng.uniform(-0.5, 0.5, (B, H + 3, d)).astype(np.float32),
        "prompt": ["make a double espresso", "fold the shirt"],
    }
    seq = tiny_pi06_tokenizer(H, d)
    print(f"raw: images {list(raw['images'])} {raw['images']['base_0_rgb'].shape}, state {raw['state'].shape}, actions {raw['actions'].shape}, prompts {raw['prompt']}")
    print(f"prefix text (sample 0, metadata 'speed: fast'): {state_prefix_text(with_metadata(raw['prompt'][0], 'speed: fast'), raw['state'][0])!r}")
    print(f"segments: {subtask_text('pick up the portafilter')!r} + {advantage_text(True)!r} / {advantage_text(False)!r} / dropped -> {advantage_text(None)!r}")
    drops = [drop_indicator(True, rng) for _ in range(1000)]
    print(f"drop_indicator(True) x 1000: {drops.count(None)} None (~{ADVANTAGE_DROPOUT:.0%}), {drops.count(True)} True, {drops.count(False)} False")

    cases = [
        ("joint", dict(subtasks=["pick up the portafilter", None], advantages=[True, None])),
        ("flow", dict(subtasks=["pick up the portafilter"] * B, advantages=[True, False])),
        ("text", dict(target_text=["a dog catches a frisbee", "Subtask: pick up the cup"])),
        ("value", {}),
        ("hl_prompt", {}),
    ]
    for layout, kw in cases:
        obs, actions = build_pi06_batch(raw, unit_stats(d), seq, layout=layout, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=False, metadata=["speed: fast", None], **kw)
        t, m, s, l = obs.tokens[0].numpy(), obs.token_mask[0].numpy(), obs.segment[0].numpy(), obs.loss_mask[0].numpy()
        n = int(m.sum())
        counts = {name: int(((s == sid) & m).sum()) for name, sid in [("prefix", SEG_PREFIX), ("subtask", SEG_SUBTASK), ("adv", SEG_ADVANTAGE), ("fast", SEG_ACTION), ("text", SEG_TEXT), ("marker", SEG_MARKER)]}
        print(f"\n[{layout:9s}] images {tuple(obs.images['base_0_rgb'].shape)} masks {[int(v[0]) for v in obs.image_masks.values()]} tokens {tuple(obs.tokens.shape)} "
              f"actions {None if actions is None else tuple(actions.shape)} has_actions {obs.has_actions.tolist()} advantage {obs.advantage.tolist()}")
        print(f"            sample 0: {n} real tokens = " + " + ".join(f"{k} {v}" for k, v in counts.items() if v) + f"; CE on {int(l.sum())} tokens; expert reads {int(obs.expert_visible[0].sum())}")
        print(f"            text: {seq.text.decode(t[:n].tolist())!r}"[:230])
        if layout == "joint":
            rec = seq.extract_actions(t, H, d)
            print(f"            extract_actions -> {rec.shape}, max|err| {np.abs(rec - actions[0, :, :d].numpy()).max():.4f}")
    # RL labels for a tiny episode
    lab = EpisodeLabels("fold the shirt", success=True, max_episode_len=40, num_steps=12, is_correction=np.array([False] * 8 + [True] * 4))
    v, bins = lab.value_targets(C_FAIL_TINY)
    print(f"\n[labels] success episode, 12 steps, T_max 40: rewards {episode_rewards(12, True, C_FAIL_TINY).tolist()}")
    print(f"         returns {returns(episode_rewards(12, True, C_FAIL_TINY)).tolist()}")
    print(f"         normalised {[f'{x:.3f}' for x in v]} -> bins {bins.tolist()} (201 bins over [-1, 0], bin 200 = 0)")
    v2, b2 = EpisodeLabels("fold the shirt", False, 40, 12).value_targets(C_FAIL_TINY)
    print(f"         failure episode: normalised {[f'{x:.3f}' for x in v2]} -> bins {b2.tolist()} (C_fail {C_FAIL_TINY:.0f} clips to -1)")
    print(f"         bin_to_value(bins) - v max|err| {np.abs(bin_to_value(bins) - v).max():.4f} (<= half a bin {0.5 / (NUM_BINS - 1):.4f})")
    big = np.zeros(14, np.float32)
    n14 = int(seq.tokenize("fold the shirt", big, layout="flow", subtask="grab the collar", advantage=True)[1].sum())
    print(f"\n14-dim state (paper robot) flow layout with subtask + advantage: {n14} of {seq.max_len} tokens (byte codec)")


if __name__ == "__main__":
    main()
