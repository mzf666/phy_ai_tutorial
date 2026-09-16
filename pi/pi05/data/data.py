"""pi0.5 data pipeline: the discrete-state prompt, one token sequence with three possible postfixes (FAST action
tokens / a text target / nothing), high-level vs low-level camera slots, and the batch that carries all of it.

Re-implementation (NumPy / PyTorch). Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/models/tokenizer.py L22-L48 (PaligemmaTokenizer.tokenize, pi05 branch L24-L29, pi0 branch L30-L33),
           src/openpi/models/pi0_config.py L28-L41 (pi05 flag, max_token_len 200, discrete_state_input),
           src/openpi/training/config.py L126-L138 (PI05 transform chain), L187 (quantile normalization), L745, L866-L870,
           src/openpi/transforms.py L247-L266 (TokenizePrompt), src/openpi/policies/droid_policy.py L47-L52 (camera masks)
  paper    pi0.5 arXiv:2504.16054v1 Sec. IV-A (state as text), IV-C (control mode tag, quantile norm, zero-pad, HL / WD),
           IV-D (post-training data), IV-E (cameras, 18 / 19 dims), Fig. 4 (HL target text), Appendix E (augmentation);
           Hi Robot arXiv:2502.19417v2 Sec. 4.3, Appendix A (synthetic HL data, background only)
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Everything shared with pi0 (image resize / augmentation, delta actions, padding) comes from pi.pi0.data; everything
shared with pi0-FAST (256-bin state text, FAST ids -> PaliGemma tail, the FAST postfix and its inverse) from
pi.fast.data. This file only holds the pi0.5 increment.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from typing import Sequence

import numpy as np
import torch

from pi.fast.data.data import (
    BOS_ID,
    EOS_ID,
    ByteTextCodec,
    FASTSequenceTokenizer,
    fast_to_paligemma,
    state_to_text,
)
from pi.fast.tokenizer.tokenizer import FASTTokenizer, QuantileStats, normalize_quantile
from pi.pi0.data.data import (
    IMAGE_RESOLUTION,
    augment,
    extract_action_chunk,  # noqa: F401  (re-exported: chunking is unchanged)
    pad_to_dim,
    resize_with_pad,
    to_delta_actions,
    uint8_to_model_range,
)

# --------------------------------------------------------------------------------------
# Constants.
# --------------------------------------------------------------------------------------
MAX_TOKEN_LEN = 200  # pi0_config.py L39: 200 if pi05 else 48
ACTION_DIM = 32  # pi0_config.py L25; config.py L902 "pi05 is trained with 32-dim actions"
ACTION_HORIZON = 50  # pi0_config.py L26; paper Appendix E ("action horizon of 50", see README Sec. 8 for "H = 49")
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # model.py L39-L43: the upstream 3 slots
# The paper's mobile manipulators have four cameras: forward, backward, two wrists (Sec. IV-E). Upstream has no
# 4-slot layout; the slot names below are this repo's (README Sec. 8). HL uses all four, LL wrist + forward.
MOBILE_IMAGE_KEYS = ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
HL_IMAGE_KEYS = MOBILE_IMAGE_KEYS  # "We use all four cameras for high-level inference" (Sec. IV-E)
LL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # "the wrist and forward cameras for the low-level"
CONTROL_MODES = ("joint", "end effector")  # paper Sec. IV-C
LAYOUTS = ("flow", "fast", "text", "hl_prompt")
LOC_BINS = 1024  # PaliGemma location tokens <loc0000> .. <loc1023>


# --------------------------------------------------------------------------------------
# 1. Prompt text. tokenizer.py L23-L28 (pi05: state in the prompt), L30-L33 (pi0: no state); paper Sec. IV-C (tag).
# --------------------------------------------------------------------------------------
def clean_prompt(prompt: str) -> str:
    """tokenizer.py L23. Unlike FAST's tokenizer (L67) there is no lower()."""
    return prompt.strip().replace("_", " ").replace("\n", " ")


def state_prefix_text(prompt: str, state: np.ndarray | None) -> str:
    """-> 'Task: <prompt>, State: <256-bin ints>;\\n' (L26-L28). state=None -> the pi0 format '<prompt>\\n' (L33),
    which is what pi05_libero uses (config.py L745 discrete_state_input=False): then the model has NO state input."""
    cleaned = clean_prompt(prompt)
    if state is None:
        return cleaned + "\n"
    return f"Task: {cleaned}, State: {state_to_text(np.asarray(state))};\n"


def with_control_mode(prompt: str, mode: str) -> str:
    """Paper Sec. IV-C: "we add '<control mode> joint/end effector <control mode>' to the text prompt". Where in the
    prompt is undisclosed (README Sec. 8); this repo appends it."""
    assert mode in CONTROL_MODES, mode
    return f"{prompt} <control mode> {mode} <control mode>"


# --------------------------------------------------------------------------------------
# 2. High-level target text. Paper Sec. IV-C "HL": bounding boxes are predicted before the subtask; Fig. 4 shows
#    'Bounding boxes: <loc0405><loc0011><loc0911><loc0197>closet' and 'Subtask: move to closet'. Line break and
#    punctuation between them are read off the figure (README Sec. 8). <locXXXX> is PaliGemma's location vocabulary:
#    XXXX = int(coord * 1024), four per box in the order y_min x_min y_max x_max.
# --------------------------------------------------------------------------------------
Box = tuple[str, tuple[float, float, float, float]]  # (label, (y0, x0, y1, x1)) with coordinates in [0, 1)


def loc_token(coord: float) -> str:
    return f"<loc{min(int(coord * LOC_BINS), LOC_BINS - 1):04d}>"


def hl_target_text(subtask: str, boxes: Sequence[Box] | None = None) -> str:
    lines = []
    if boxes:
        lines.append("Bounding boxes: " + " ".join("".join(loc_token(c) for c in coords) + label for label, coords in boxes))
    lines.append(f"Subtask: {subtask}")
    return "\n".join(lines)


_BOX_RE = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>([^<]*?)(?=\s<loc|\s*$)")


def parse_hl_text(text: str) -> tuple[str, list[Box]]:
    """Inverse of hl_target_text on generated text: -> (subtask, boxes). Without a 'Subtask:' line the whole text is
    the subtask (a model that skipped the marker still gives a usable command)."""
    boxes: list[Box] = []
    subtask = text.strip()
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Bounding boxes:"):
            for m in _BOX_RE.finditer(line[len("Bounding boxes:"):].strip()):
                coords = tuple(int(m.group(i)) / LOC_BINS for i in range(1, 5))
                boxes.append((m.group(5).strip(), coords))  # type: ignore[arg-type]
        elif line.startswith("Subtask:"):
            subtask = line[len("Subtask:"):].strip()
    return subtask, boxes


# --------------------------------------------------------------------------------------
# 3. The sequence: one prefix, three postfixes. tokenizer.py L22-L48 for the prefix and the flow-only marker;
#    pi.fast.data for the FAST postfix; the text postfix follows the same [postfix][EOS] shape as FAST's.
#
#    layout      prefix (ar 0, loss False)              postfix (ar 1, loss True)
#    "flow"      [BOS] Task: p, State: s;\nAction:      (empty)                          L28-L29: post-training inference / finetune
#    "fast"      [BOS] Task: p, State: s;\n             Action: <FAST ids> | [EOS]       pre-training, joint objective (paper Sec. IV-C, Eq. 1)
#    "text"      [BOS] Task: p, State: s;\n             <target text> [EOS]              HL subtask / bbox, WD (Sec. IV-C)
#    "hl_prompt" [BOS] Task: p, State: s;\n             (empty)                          HL inference: the model writes the postfix
# --------------------------------------------------------------------------------------
class Pi05SequenceTokenizer:
    def __init__(self, text: ByteTextCodec, fast: FASTTokenizer | None = None, max_len: int = MAX_TOKEN_LEN):
        self.text, self.fast, self.max_len = text, fast, max_len
        self._action_marker = text.encode("Action: ")  # L28 (flow prefix), FAST L84 (postfix start)
        self._fast_seq = None if fast is None else FASTSequenceTokenizer(text, fast, max_len)  # extract_actions lives there

    def _prefix_ids(self, prompt: str, state: np.ndarray | None) -> list[int]:
        return self.text.encode(state_prefix_text(prompt, state), add_bos=True)  # L29 add_bos=True

    def tokenize(self, prompt: str, state: np.ndarray | None, *, layout: str, actions: np.ndarray | None = None,
                 target_text: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One sample. state float[d] normalized or None; actions float[H, d] normalized (native dim) for "fast";
        target_text for "text". Returns int64[max_len] tokens, bool[max_len] token_mask, int64[max_len] ar_mask,
        bool[max_len] loss_mask (0-padded on the right; truncated with a warning if longer, L38-L48)."""
        assert layout in LAYOUTS, layout
        prefix = self._prefix_ids(prompt, state)
        if layout == "flow":
            prefix = prefix + self._action_marker  # L28: 'Action: ' belongs to the (bidirectional) prefix
            postfix: list[int] = []
        elif layout == "fast":
            assert actions is not None and self.fast is not None
            fast_ids = self.fast(np.asarray(actions)[None])[0]
            postfix = self._action_marker + fast_to_paligemma(fast_ids).tolist() + self.text.encode("|", add_eos=True)  # FAST L83-L87
        elif layout == "text":
            assert target_text is not None
            postfix = self.text.encode(target_text, add_eos=True)
        else:  # "hl_prompt"
            postfix = []
        tokens = prefix + postfix
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix) + [1] * len(postfix)
        loss_mask = [False] * len(prefix) + [True] * len(postfix)
        n = len(tokens)
        if n < self.max_len:  # L35-L38
            pad = self.max_len - n
            tokens, token_mask, ar_mask, loss_mask = tokens + [0] * pad, token_mask + [False] * pad, ar_mask + [0] * pad, loss_mask + [False] * pad
        elif n > self.max_len:  # L40-L48: truncate (for "fast" that cuts action tokens; for "flow" the marker!)
            warnings.warn(f"Token length ({n}) exceeds max length ({self.max_len}), truncating. Consider increasing max_token_len.", stacklevel=2)
            tokens, token_mask, ar_mask, loss_mask = (x[: self.max_len] for x in (tokens, token_mask, ar_mask, loss_mask))
        return np.asarray(tokens, dtype=np.int64), np.asarray(token_mask, dtype=bool), np.asarray(ar_mask, dtype=np.int64), np.asarray(loss_mask, dtype=bool)

    def extract_text(self, tokens: np.ndarray) -> str:
        """Generated ids (any length, may include pad / EOS) -> text up to the first EOS. Counterpart of
        pi.fast.data.FASTSequenceTokenizer.extract_actions for the "text" / "hl_prompt" layouts."""
        ids = [int(t) for t in np.asarray(tokens).reshape(-1)]
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID)]
        return self.text.decode([t for t in ids if t not in (0, BOS_ID)]).strip()

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """"fast" layout inverse: pi.fast.data's search for 'Action: ' .. '|' and the FAST decode (zeros if absent)."""
        assert self._fast_seq is not None
        return self._fast_seq.extract_actions(tokens, action_horizon, action_dim)


# --------------------------------------------------------------------------------------
# 4. The batch. config.py L126-L138 (PI05 chain = pi0's with discrete_state_input) + data_loader.py L183-L190.
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Pi05Observation:
    """model.py L83-L107 plus the two FAST masks. Keys of images / image_masks are the `image_keys` given to
    build_pi05_batch (upstream 3 slots, or this repo's 4 mobile slots).

    images:                {slot: float32[B, 224, 224, 3]} in [-1, 1]
    image_masks:           {slot: bool[B]} False for a missing camera (droid_policy.py L48-L51: PI0 and PI05 alike)
    state:                 float32[B, 32] normalized, zero-padded. NOT read by the pi0.5 model (no state_proj,
                           pi0.py L151-L157); kept for the inverse transforms and parity checks
    tokenized_prompt:      int64[B, max_len]   the sequence (prefix + postfix + pad)
    tokenized_prompt_mask: bool[B, max_len]    True on real tokens
    token_ar_mask:         int64[B, max_len]   0 prefix, 1 postfix
    token_loss_mask:       bool[B, max_len]    True on the postfix
    """

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor
    token_loss_mask: torch.Tensor


def build_pi05_batch(
    raw: dict,
    norm_stats: dict[str, QuantileStats] | None,
    seq: Pi05SequenceTokenizer,
    *,
    layout: str,
    image_keys: Sequence[str] = IMAGE_KEYS,
    action_horizon: int = ACTION_HORIZON,
    action_dim: int = ACTION_DIM,
    delta_mask: Sequence[bool] | None,
    train: bool,
    target_text: Sequence[str] | None = None,
    discrete_state: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Pi05Observation, torch.Tensor | None]:
    """raw = {"images": {slot: uint8[B, h, w, 3]} (subset of image_keys), "state": f32[B, d], "actions": f32[B, >=H, d]
    (absolute, training only), "prompt": [str] * B}. norm_stats = {"state": QuantileStats, "actions": QuantileStats} or None.
    discrete_state=False reproduces pi05_libero (config.py L745): the prompt carries no state at all.
    Returns (Pi05Observation, actions f32[B, H, action_dim] normalized / delta / zero-padded, or None)."""
    b = raw["state"].shape[0]
    state = np.asarray(raw["state"], dtype=np.float32)
    actions = None if "actions" not in raw else np.asarray(raw["actions"], dtype=np.float32)[:, :action_horizon]

    # (a) camera slots: a missing slot is a black image with mask False (droid_policy.py L48-L51; same as pi0).
    ref = np.asarray(next(iter(raw["images"].values())))
    images = {k: np.asarray(raw["images"][k]) if k in raw["images"] else np.zeros_like(ref) for k in image_keys}
    image_masks = {k: np.full((b,), k in raw["images"], dtype=bool) for k in image_keys}

    # (b) DeltaActions (transforms.py L204-L223), unchanged.
    if actions is not None:
        actions = to_delta_actions(state, actions, delta_mask)

    # (c) quantile normalization (config.py L187: every non-pi0 model; paper Sec. IV-C).
    if norm_stats is not None:
        state = normalize_quantile(state, norm_stats["state"])
        if actions is not None:
            actions = normalize_quantile(actions, norm_stats["actions"])

    # (d) ResizeImages, unchanged.
    images_t = {k: resize_with_pad(torch.from_numpy(v), *IMAGE_RESOLUTION) for k, v in images.items()}

    # (e) TokenizePrompt with discrete_state_input (transforms.py L256-L265): the sequence, per sample, native dim.
    per_sample = [
        seq.tokenize(raw["prompt"][i], state[i] if discrete_state else None, layout=layout,
                     actions=None if actions is None else actions[i], target_text=None if target_text is None else target_text[i])
        for i in range(b)
    ]
    tokens, token_mask, ar_mask, loss_mask = (np.stack(x) for x in zip(*per_sample))

    # (f) PadStatesAndActions to 32 (transforms.py L328-L340; config.py L868).
    state = pad_to_dim(state, action_dim).astype(np.float32)
    if actions is not None:
        actions = pad_to_dim(actions, action_dim).astype(np.float32)

    # (g) inside the model: uint8 -> [-1, 1]; train-time augmentation (model.py L176-L181 == paper Appendix E).
    for k in image_keys:
        img = uint8_to_model_range(images_t[k])
        images_t[k] = augment(img, k, generator) if train else img

    obs = Pi05Observation(
        images=images_t,
        image_masks={k: torch.from_numpy(v) for k, v in image_masks.items()},
        state=torch.from_numpy(state),
        tokenized_prompt=torch.from_numpy(tokens),
        tokenized_prompt_mask=torch.from_numpy(token_mask),
        token_ar_mask=torch.from_numpy(ar_mask),
        token_loss_mask=torch.from_numpy(loss_mask),
    )
    return obs, None if actions is None else torch.from_numpy(actions)


# --------------------------------------------------------------------------------------
# 5. Tiny helpers and a walk-through.  uv run python -m pi.pi05.data.data
# --------------------------------------------------------------------------------------
def tiny_pi05_tokenizer(horizon: int, dim: int, seed: int = 0, max_len: int = MAX_TOKEN_LEN) -> Pi05SequenceTokenizer:
    from pi.fast.data.data import tiny_fast_tokenizer

    return Pi05SequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(horizon, dim, seed=seed), max_len=max_len)


def unit_stats(dim: int) -> dict[str, QuantileStats]:
    """q01 = -1, q99 = +1 on every dim: normalization is the identity. For mains and tests."""
    return {"state": QuantileStats(np.full(dim, -1.0), np.full(dim, 1.0)), "actions": QuantileStats(np.full(dim, -1.0), np.full(dim, 1.0))}


def main() -> None:
    from pi.pi0.data.data import make_bool_mask

    rng = np.random.default_rng(0)
    B, H, d = 2, 10, 7  # a LIBERO-like arm (7-dim, H 10) keeps the tiny FAST tokenizer small; the paper robot is 18 / 19-dim, H 50
    raw = {
        "images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8), "left_wrist_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
        "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
        "actions": rng.uniform(-0.5, 0.5, (B, H + 3, d)).astype(np.float32),
        "prompt": [with_control_mode("clean the kitchen", "joint"), "pick up the pillow"],
    }
    seq = tiny_pi05_tokenizer(H, d)
    codec = seq.text
    print(f"raw: images {list(raw['images'])} {raw['images']['base_0_rgb'].shape}, state {raw['state'].shape}, actions {raw['actions'].shape}")
    print(f"     prompts {raw['prompt']}")
    print(f"prefix text (sample 1): {state_prefix_text(raw['prompt'][1], raw['state'][1])!r}")
    print(f"prefix text, no state : {state_prefix_text(raw['prompt'][1], None)!r}   (pi05_libero, config.py L745)")

    for layout, kw in [("flow", {}), ("fast", {}), ("text", {"target_text": [hl_target_text("pick up the plate", [("plate", (0.40, 0.11, 0.91, 0.19))]), hl_target_text("close the drawer")]}), ("hl_prompt", {})]:
        keys = HL_IMAGE_KEYS if layout in ("text", "hl_prompt") else LL_IMAGE_KEYS
        obs, actions = build_pi05_batch(raw, unit_stats(d), seq, layout=layout, image_keys=keys, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=False, **kw)
        m, ar, lm = obs.tokenized_prompt_mask[0].numpy(), obs.token_ar_mask[0].numpy(), obs.token_loss_mask[0].numpy()
        n_pre, n_post = int(((ar == 0) & m).sum()), int((ar == 1).sum())
        print(f"\n[{layout:9s}] cameras {list(obs.images)} masks {[int(v[0]) for v in obs.image_masks.values()]}  images {tuple(obs.images['base_0_rgb'].shape)}"
              f"  state {tuple(obs.state.shape)}  tokens {tuple(obs.tokenized_prompt.shape)}  actions {None if actions is None else tuple(actions.shape)}")
        print(f"            sample 0: {int(m.sum())} real = prefix {n_pre} + postfix {n_post}; ar {''.join(map(str, ar[:int(m.sum())]))[:60]}...")
        t = obs.tokenized_prompt[0].numpy()
        print(f"            prefix text: {codec.decode(t[:n_pre].tolist())!r}")
        if layout == "fast":
            rec = seq.extract_actions(t, H, d)
            print(f"            postfix: {n_post} ids = 'Action: '({len(seq._action_marker)}) + FAST + '|' + EOS; extract_actions -> {rec.shape}, max|err| {np.abs(rec - actions[0, :, :d].numpy()).max():.4f}")
        if layout == "text":
            print(f"            postfix text: {seq.extract_text(t[n_pre:])!r}  -> parse_hl_text {parse_hl_text(seq.extract_text(t[n_pre:]))}")
    big = np.zeros(19, np.float32)
    n19 = int(seq.tokenize("clean the kitchen", big, layout="flow")[1].sum())
    print(f"\n19-dim state (paper robot), byte codec: flow prefix = {n19} tokens of {seq.max_len} (pi0's limit was 48)")


if __name__ == "__main__":
    main()
