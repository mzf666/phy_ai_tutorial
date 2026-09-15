"""pi0-FAST data pipeline: raw robot sample -> one token sequence (prefix text + state bins + FAST action tokens) with
its three masks, plus the inverse that cuts the actions back out of a generated sequence.

Re-implementation (PyTorch / NumPy). Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/models/tokenizer.py L51-L139 (FASTTokenizer: tokenize, extract_actions, vocab mapping),
           src/openpi/transforms.py L270-L306 (TokenizeFASTInputs, ExtractFASTActions),
           src/openpi/training/config.py L139-L160 (data path for PI0_FAST) and L187 (quantile normalization),
           src/openpi/policies/droid_policy.py L56-L60 and libero_policy.py L64-L69 (camera slots, no image masking),
           src/openpi/models/model.py L60-L107 (Observation fields), src/openpi/models/pi0_fast.py L84 (max_token_len)
  paper    FAST, arXiv:2501.09747v1, Sec. VI-A (vocabulary overwrite), Appendix C (state binning)
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Everything that is the same as pi0 (image resize / pad / augmentation, delta actions, action chunking, padding to the
model action dim) is imported from pi.pi0.data. This file only holds the increment.

Pipeline order (openpi@215abfb training/data_loader.py L183-L190, with the FAST transforms from config.py L150-L159):
  robot adapter -> DeltaActions -> Normalize (quantile) -> ResizeImages -> TokenizeFASTInputs -> PadStatesAndActions
Note the order: actions are FAST-tokenized in their native dim, BEFORE zero-padding to the model action dim.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Sequence

import numpy as np
import torch

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
PALIGEMMA_VOCAB_SIZE = 257_152  # openpi@215abfb gemma.py L41; sentencepiece vocab_size() in tokenizer.py L139
FAST_SKIP_TOKENS = 128  # tokenizer.py L62: the last 128 PaliGemma ids are special tokens, never overwritten
MAX_TOKEN_LEN = 250  # pi0_fast.py L84 (base model). Fine-tuning configs use 180 (config.py L711, L839)
STATE_BINS = 256  # tokenizer.py L70; paper Appendix C
BOS_ID, EOS_ID = 2, 1  # Gemma special ids; pi0_fast.py L20 PALIGEMMA_EOS_TOKEN = 1
IMAGE_KEYS = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")  # pi0_fast.py L107-L111; differs from pi0's slot names
ACTION_DIM = 32  # Pi0FASTConfig default (pi0_fast.py L82); LIBERO uses 7, DROID 8 (config.py L711, L617)


# --------------------------------------------------------------------------------------
# 1. Text codec stand-in. The real model uses the PaliGemma SentencePiece vocabulary (257,152 pieces, BOS 2, EOS 1;
#    tokenizer.py L56-L58, gs://big_vision/paligemma_tokenizer.model). Like pi.pi0.data.ByteEncoder we use one id per
#    UTF-8 byte (+3) so the tiny config needs no download, but we keep the *declared* vocab size at 257,152 so the
#    action-id mapping below lands in exactly the same ids as upstream. NOT the PaliGemma vocabulary.
# --------------------------------------------------------------------------------------
class ByteTextCodec:
    vocab_size = PALIGEMMA_VOCAB_SIZE
    bos_id, eos_id = BOS_ID, EOS_ID

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [b + 3 for b in text.encode("utf-8")]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids: Sequence[int]) -> str:
        """Drops special ids (< 3) and any id outside the byte range (e.g. action ids), like a SentencePiece decode of
        an unknown piece would not produce readable text."""
        return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------------------
# 2. State -> 256 bins -> text. tokenizer.py L69-L74; paper Appendix C ("discretizing into 256 bins ... tokenize the
#    integers as part of the text input"). Input is the quantile-normalized state, "assumed range [-1, 1]" (L69).
#    Quantile normalization does not clip, so ~2% of values fall outside: x >= 1 lands in bin 255 (the last bin is
#    open), but x < -1 gives np.digitize == 0, minus 1 == -1, i.e. the text contains "-1". Upstream does not guard
#    against this; we reproduce it (see README gap ledger).
# --------------------------------------------------------------------------------------
_STATE_BIN_EDGES = np.linspace(-1, 1, STATE_BINS + 1)[:-1]  # 256 edges: -1, -1 + 2/256, ..., 1 - 2/256


def discretize_state(state: np.ndarray) -> np.ndarray:
    """float[d] in ~[-1, 1] -> int64[d], normally in [0, 255]. Bin i covers [-1 + 2i/256, -1 + 2(i+1)/256); the last bin
    is open (x >= 1 -> 255); x < -1 -> -1 (upstream behaviour, not clamped)."""
    return np.digitize(state, bins=_STATE_BIN_EDGES) - 1


def state_to_text(state: np.ndarray) -> str:
    """int bins joined by single spaces, e.g. '127 200 0 255 12 12 12'."""
    return " ".join(map(str, discretize_state(state)))


# --------------------------------------------------------------------------------------
# 3. FAST action ids <-> PaliGemma ids. tokenizer.py L136-L139: pg = vocab_size - 1 - 128 - t. The map is an
#    involution (applying it twice gives t back), so one function serves both directions. FAST id 0 lands on
#    257,023 and FAST id 2047 on 254,976: the 2048 ids just below the 128 special tokens, "the least used tokens
#    in the VLM vocabulary" (paper Sec. VI-A). Their embedding rows keep the PaliGemma weights and are fine-tuned.
# --------------------------------------------------------------------------------------
def fast_to_paligemma(ids: np.ndarray | Sequence[int], vocab_size: int = PALIGEMMA_VOCAB_SIZE) -> np.ndarray:
    return vocab_size - 1 - FAST_SKIP_TOKENS - np.asarray(ids, dtype=np.int64)


paligemma_to_fast = fast_to_paligemma  # same formula; kept as a second name for readability at call sites


# --------------------------------------------------------------------------------------
# 4. The sequence. tokenizer.py L64-L117 (tokenize) and L119-L134 (extract_actions).
#
#    [BOS] Task: <prompt>, State: <b_1 b_2 ... b_d>;\n   Action: <FAST ids in PaliGemma space> | [EOS]   [pad ...]
#    |<------------------- prefix: ar_mask 0, loss False ------------->|<---- postfix: ar 1, loss True ---->|
#
#    ar_mask feeds make_attn_mask (pi0_fast.py L23-L48): prefix tokens attend to each other bidirectionally (like the
#    images), every postfix token attends causally. loss_mask: cross-entropy only on the postfix (train.py). At
#    inference `actions is None`, the postfix is empty and the model generates it.
# --------------------------------------------------------------------------------------
class FASTSequenceTokenizer:
    """prompt + normalized state (+ normalized actions) -> (tokens, token_mask, ar_mask, loss_mask), each length max_len."""

    def __init__(self, text: ByteTextCodec, fast: FASTTokenizer, max_len: int = MAX_TOKEN_LEN):
        self.text, self.fast, self.max_len = text, fast, max_len
        self._action_marker = text.encode("Action: ")  # tokenizer.py L84, L124, L129
        self._end_marker = text.encode("|")  # tokenizer.py L86, L129

    def tokenize(self, prompt: str, state: np.ndarray, actions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One sample. state float[d] normalized; actions float[H, d] normalized (native dim, not yet padded) or None.
        Returns int64[max_len] tokens (0-padded), bool[max_len] token_mask, int64[max_len] ar_mask, bool[max_len] loss_mask."""
        cleaned = prompt.lower().strip().replace("_", " ")  # tokenizer.py L67
        prefix = f"Task: {cleaned}, State: {state_to_text(state)};\n"  # L74
        prefix_ids = self.text.encode(prefix, add_bos=True)  # L75
        if actions is not None:
            fast_ids = self.fast(np.asarray(actions)[None])[0]  # L79
            postfix_ids = self._action_marker + fast_to_paligemma(fast_ids).tolist() + self.text.encode("|", add_eos=True)  # L83-L87
        else:
            postfix_ids = []  # L89: inference, the model writes the postfix
        tokens = prefix_ids + postfix_ids
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_ids) + [1] * len(postfix_ids)  # L95
        loss_mask = [False] * len(prefix_ids) + [True] * len(postfix_ids)  # L96
        n = len(tokens)
        if n < self.max_len:  # L100-L105: pad with 0 / False
            pad = self.max_len - n
            tokens, token_mask, ar_mask, loss_mask = tokens + [0] * pad, token_mask + [False] * pad, ar_mask + [0] * pad, loss_mask + [False] * pad
        elif n > self.max_len:  # L107-L115: truncate the tail (which is the action tokens!) and warn
            warnings.warn(f"Token length ({n}) exceeds max length ({self.max_len}), truncating. Consider increasing max_token_len.", stacklevel=2)
            tokens, token_mask, ar_mask, loss_mask = (x[: self.max_len] for x in (tokens, token_mask, ar_mask, loss_mask))
        return np.asarray(tokens, dtype=np.int64), np.asarray(token_mask, dtype=bool), np.asarray(ar_mask, dtype=np.int64), np.asarray(loss_mask, dtype=bool)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """Generated ids (any length, may include pad / EOS) -> float[H, D] normalized actions; zeros if no 'Action: '.

        Upstream (L121-L134) decodes the whole sequence to text, splits on 'Action: ' and '|', re-encodes the middle
        to ids and maps them back. That round trip relies on SentencePiece decoding the 2048 overwritten pieces and
        re-encoding them to the same ids. Our byte codec cannot decode those ids, so we do the equivalent search on
        ids: find the 'Action: ' marker ids, take everything up to the '|' marker (or EOS / end), map back, FAST-decode.
        Same result whenever the upstream round trip is exact (本仓库推断, see README gap ledger)."""
        ids = [int(t) for t in np.asarray(tokens).reshape(-1)]
        start = _find(ids, self._action_marker)
        if start < 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)  # L124-L125
        body = ids[start + len(self._action_marker) :]
        end = _find(body, self._end_marker)
        if end >= 0:
            body = body[:end]
        if EOS_ID in body:
            body = body[: body.index(EOS_ID)]
        body = [t for t in body if t != 0]  # strip pad (id 0)
        fast_ids = paligemma_to_fast(body)
        if len(fast_ids) and (fast_ids.min() < 0 or fast_ids.max() >= self.fast.vocab_size):
            return np.zeros((action_horizon, action_dim), dtype=np.float32)  # text ids inside the action span: garbage
        return self.fast.decode([fast_ids.tolist()], time_horizon=action_horizon, action_dim=action_dim)[0].astype(np.float32)


def _find(seq: list[int], sub: list[int]) -> int:
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return -1


# --------------------------------------------------------------------------------------
# 5. The batch. Same skeleton as pi.pi0.data.build_batch; the differences are marked (FAST).
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class FASTObservation:
    """model.py L83-L107 with the two pi0-FAST fields.

    images:                {key: float32[B, 224, 224, 3]} in [-1, 1], keys == IMAGE_KEYS (base_0, base_1, wrist_0).
    image_masks:           {key: bool[B]} all True for FAST (missing slots are black images that are NOT masked).
    state:                 float32[B, 32] normalized, zero-padded. Carried in the Observation but the FAST model does not
                           read it: pi0_fast.py embed_inputs (L159-L195) uses only images and tokenized_prompt; the
                           state enters as text bins inside the prompt.
    tokenized_prompt:      int64[B, max_len] the whole sequence (prefix + postfix + pad).
    tokenized_prompt_mask: bool[B, max_len] True on real tokens.
    token_ar_mask:         int64[B, max_len] 0 on prefix, 1 on postfix.
    token_loss_mask:       bool[B, max_len] True on postfix only.
    """

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor
    token_loss_mask: torch.Tensor


def build_fast_batch(
    raw: dict,
    norm_stats: dict[str, QuantileStats] | None,
    seq_tokenizer: FASTSequenceTokenizer,
    *,
    action_horizon: int,
    action_dim: int = ACTION_DIM,
    delta_mask: Sequence[bool] | None,
    train: bool,
    generator: torch.Generator | None = None,
) -> tuple[FASTObservation, torch.Tensor | None]:
    """Raw robot sample -> (FASTObservation, actions).

    raw = {"images": {name: uint8[B, h, w, 3]} any subset of IMAGE_KEYS, "state": float32[B, d], "actions": float32[B, >=H, d]
           (training only, absolute), "prompt": list[str]}.
    norm_stats = {"state": QuantileStats, "actions": QuantileStats} or None. Returns actions float32[B, H, action_dim]
    (normalized, delta, zero-padded: the same tensor the FAST tokens encode, kept for parity checks) or None.
    """
    b = raw["state"].shape[0]
    state = np.asarray(raw["state"], dtype=np.float32)
    actions = None if "actions" not in raw else np.asarray(raw["actions"], dtype=np.float32)[:, :action_horizon]

    # (a) robot adapter (FAST): three slots base_0 / base_1 / wrist_0; a missing camera is a black image with mask True.
    #     droid_policy.py L56-L60 "We don't mask out padding images for FAST models"; libero_policy.py L67-L68.
    base = np.asarray(raw["images"]["base_0_rgb"])
    images = {k: np.asarray(raw["images"][k]) if k in raw["images"] else np.zeros_like(base) for k in IMAGE_KEYS}
    image_masks = {k: np.ones((b,), dtype=bool) for k in IMAGE_KEYS}

    # (b) DeltaActions, unchanged from pi0 (transforms.py L204-L223).
    if actions is not None:
        actions = to_delta_actions(state, actions, delta_mask)

    # (c) Normalize (FAST): quantile, q01 -> -1, q99 -> +1 (config.py L187, transforms.py L141-L145).
    if norm_stats is not None:
        state = normalize_quantile(state, norm_stats["state"])
        if actions is not None:
            actions = normalize_quantile(actions, norm_stats["actions"])

    # (d) ResizeImages, unchanged.
    images_t = {k: resize_with_pad(torch.from_numpy(v), *IMAGE_RESOLUTION) for k, v in images.items()}

    # (e) TokenizeFASTInputs (FAST): prompt + state bins + FAST(actions) -> one sequence + 3 masks, per sample.
    #     transforms.py L270-L288. Runs on the native-dim state / actions, before padding.
    per_sample = [seq_tokenizer.tokenize(raw["prompt"][i], state[i], None if actions is None else actions[i]) for i in range(b)]
    tokens, token_mask, ar_mask, loss_mask = (np.stack(x) for x in zip(*per_sample))

    # (f) PadStatesAndActions to action_dim, unchanged (transforms.py L328-L340).
    state = pad_to_dim(state, action_dim).astype(np.float32)
    if actions is not None:
        actions = pad_to_dim(actions, action_dim).astype(np.float32)

    # (g) inside the model: uint8 -> [-1, 1]; train-time augmentation (same as pi0, preprocess_observation).
    for k in IMAGE_KEYS:
        img = uint8_to_model_range(images_t[k])
        images_t[k] = augment(img, k, generator) if train else img

    obs = FASTObservation(
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
# 6. Tiny helpers and a walk-through.
# --------------------------------------------------------------------------------------
def tiny_fast_tokenizer(horizon: int, dim: int, seed: int = 0, vocab_size: int = 320) -> FASTTokenizer:
    """A FAST tokenizer fit on synthetic smooth chunks (stand-in for FAST+). Real runs load FASTTokenizer.from_hf_dir()."""
    from pi.fast.tokenizer.tokenizer import make_smooth_chunks

    return FASTTokenizer.fit(list(make_smooth_chunks(64, horizon, dim, np.random.default_rng(seed))), scale=10, vocab_size=vocab_size)


def main() -> None:
    from pi.pi0.data.data import make_bool_mask

    rng = np.random.default_rng(0)
    B, H, d = 2, 10, 7  # a LIBERO-like arm: 7-dim, action_horizon 10 (config.py L711)
    raw = {
        "images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8), "wrist_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
        "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
        "actions": rng.uniform(-0.5, 0.5, (B, H + 3, d)).astype(np.float32),
        "prompt": ["Pick_up the red block", "close the drawer"],
    }
    stats = {"state": QuantileStats(np.full(d, -1.0), np.full(d, 1.0)), "actions": QuantileStats(np.full(d, -1.0), np.full(d, 1.0))}
    seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, d), max_len=180)
    print(f"raw: images {list(raw['images'])} {raw['images']['base_0_rgb'].shape}, state {raw['state'].shape}, actions {raw['actions'].shape}")
    print(f"     prompts {raw['prompt']}")
    obs, actions = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=False)
    print(f"\nFASTObservation:")
    for k, v in obs.images.items():
        print(f"  images[{k}] {tuple(v.shape)} {v.dtype}  mask {obs.image_masks[k].tolist()}  ({'real camera' if k in raw['images'] else 'black, still mask True'})")
    print(f"  state {tuple(obs.state.shape)} (float, unused by the FAST model; the model reads the text bins)")
    print(f"  tokenized_prompt {tuple(obs.tokenized_prompt.shape)}  ar_mask {tuple(obs.token_ar_mask.shape)}  loss_mask {tuple(obs.token_loss_mask.shape)}")
    print(f"actions (what the tokens encode) {tuple(actions.shape)}")

    i = 0
    t, m, ar, lm = obs.tokenized_prompt[i].numpy(), obs.tokenized_prompt_mask[i].numpy(), obs.token_ar_mask[i].numpy(), obs.token_loss_mask[i].numpy()
    n_pre, n_post, n_real = int(((ar == 0) & m).sum()), int((ar == 1).sum()), int(m.sum())
    st = normalize_quantile(raw["state"][i], stats["state"])
    print(f"\nsample 0 sequence: {n_real} real tokens = prefix {n_pre} + postfix {n_post}, padded to {len(t)}")
    print(f"  state bins: {discretize_state(st).tolist()}")
    print(f"  prefix text: {ByteTextCodec().decode(t[:n_pre].tolist())!r}")
    pg = t[m & (ar == 1)]
    fast_ids = paligemma_to_fast(pg[len(seq._action_marker) : -2])
    print(f"  postfix ids: 'Action: ' {pg[:len(seq._action_marker)].tolist()} + {len(fast_ids)} FAST ids in [{pg[len(seq._action_marker):-2].min()}, {pg[len(seq._action_marker):-2].max()}] (PaliGemma tail) + '|' {pg[-2:-1].tolist()} + EOS {pg[-1:].tolist()}")
    print(f"  FAST ids after mapping back: {fast_ids.tolist()}")
    print(f"  ar_mask   : {''.join(map(str, ar[:n_real].tolist()))}...")
    print(f"  loss_mask : {''.join('1' if x else '0' for x in lm[:n_real].tolist())}...")
    rec = seq.extract_actions(t, H, d)
    print(f"\nextract_actions(sequence) -> {rec.shape}; max |rec - actions[:, :d]| = {np.abs(rec - actions[i, :, :d].numpy()).max():.4f} (FAST rounding only)")
    garbage = seq.extract_actions(t[:n_pre], H, d)
    print(f"extract_actions(prefix only, no 'Action: ') -> zeros: {bool((garbage == 0).all())}")


if __name__ == "__main__":
    main()
