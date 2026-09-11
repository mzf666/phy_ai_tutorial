"""pi0 data pipeline: raw robot sample -> (Observation, actions) the model consumes.

Re-implementation (PyTorch / NumPy) of the openpi data path. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi
          commit 215abfb217dbac7d5f1273282331b9b1866c0479 (referenced as `openpi@215abfb`)
  paper   pi0: A Vision-Language-Action Flow Model for General Robot Control, arXiv:2410.24164v1
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Pipeline order (openpi@215abfb src/openpi/training/data_loader.py L183-L190):
  robot adapter -> DeltaActions -> Normalize -> ResizeImages -> TokenizePrompt
  -> PadStatesAndActions -> [inside the model] uint8->[-1,1], train-time augmentation.
`build_batch` at the bottom of this file runs exactly that order.

Read top to bottom. Every function states its I/O contract in the docstring.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Constants. Values are the pi0 defaults in openpi@215abfb src/openpi/models/pi0_config.py
# L25 (action_dim), L26 (action_horizon), L39 (max_token_len for pi0), and
# src/openpi/models/model.py L39-L47 (IMAGE_KEYS, IMAGE_RESOLUTION).
# Note: the paper (Sec. V-A) pads to 18 dims, the largest robot in the paper's mixture.
# The released checkpoints and openpi use 32. We follow the release.
# --------------------------------------------------------------------------------------
ACTION_DIM = 32  # every robot's state/action is zero-padded to this width
ACTION_HORIZON = 50  # H in the paper: one chunk = 50 future actions
MAX_TOKEN_LEN = 48  # prompt tokens (incl. BOS and trailing "\n"), padded/truncated
IMAGE_RESOLUTION = (224, 224)  # SigLIP So400m/14 input size
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # fixed 3 camera slots


# --------------------------------------------------------------------------------------
# 1. Output protocol of this module == input protocol of the model.
#    Mirrors openpi@215abfb src/openpi/models/model.py L83-L100 (class Observation).
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class Observation:
    """What the model sees at one control step (batched).

    images:               {key: float32[B, 224, 224, 3]} in [-1, 1], channels-last, keys == IMAGE_KEYS.
    image_masks:          {key: bool[B]} False if that camera slot is padding (robot has no such camera).
    state:                float32[B, 32] normalized proprioception, zero-padded to 32.
    tokenized_prompt:     int64[B, 48] token ids, right-padded with 0.
    tokenized_prompt_mask: bool[B, 48] True on real tokens.
    """

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor


# Training target: float32[B, 50, 32] normalized (and, for joint dims, delta) actions, zero-padded.


# --------------------------------------------------------------------------------------
# 2. Images: resize without distortion, pad with black.
#    openpi@215abfb src/openpi/shared/image_tools.py L13-L54 (resize_with_pad, JAX) and
#    L57-L126 (resize_with_pad_torch). Semantics of tf.image.resize_with_pad.
#
#    Why pad instead of stretch or crop. The only stated upstream motive is "without distortion"
#    (image_tools.py L20-L21); the paper does not discuss it. The reasoning below is ours.
#    SigLIP So400m/14 has position embeddings learned on a 16x16 patch grid of a 224x224 square,
#    so every camera frame (4:3 or 16:9 on real robots) must become square. Three options:
#      * stretch: geometry is distorted (circles -> ellipses), and each camera distorts differently,
#        so the model would have to learn a per-camera warp on top of the task;
#      * centre crop: geometry kept, but ~1/4 of the field of view is thrown away, and in manipulation
#        the gripper, the object, or the second arm is often exactly at the frame edge;
#      * resize + pad (chosen): geometry and field of view kept; the price is resolution, a 4:3 frame
#        uses only 224x168 of the 224x224 pixels, i.e. ~1/4 of the image tokens look at black bars.
#    Padding also makes all 7 robot configurations look geometrically identical to the model, which
#    matters for cross-embodiment training. Black (0 / -1.0) is a constant, easy-to-ignore signal;
#    reflection or mean padding would hallucinate content. Inference MUST apply the same transform,
#    since norm stats, augmentation and the learned "this is a border" cue all assume it; openpi
#    re-checks the shape inside the model and pads again if needed (model.py L164-L166).
# --------------------------------------------------------------------------------------
def resize_with_pad(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """uint8[*B, h, w, c] in [0,255] or float32 in [-1,1]  ->  same dtype [*B, height, width, c].

    Scales so the longer side fits, bilinear; pads the rest symmetrically with black
    (0 for uint8, -1.0 for float). Aspect ratio is preserved: a 640x480 image becomes
    224x168 content inside a 224x224 frame.
    """
    squeeze = images.ndim == 3
    if squeeze:
        images = images[None]
    x = images.permute(0, 3, 1, 2)  # -> [B, c, h, w] for F.interpolate
    b, c, h, w = x.shape
    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)
    y = F.interpolate(x.float(), size=(rh, rw), mode="bilinear", align_corners=False)
    if images.dtype == torch.uint8:
        y = torch.round(y).clamp(0, 255).to(torch.uint8)
        fill = 0
    elif images.dtype == torch.float32:
        y = y.clamp(-1.0, 1.0)
        fill = -1.0
    else:
        raise ValueError(f"unsupported image dtype {images.dtype}")
    pad_h0, rem_h = divmod(height - rh, 2)
    pad_w0, rem_w = divmod(width - rw, 2)
    y = F.pad(y, (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h), value=fill)
    y = y.permute(0, 2, 3, 1)
    return y[0] if squeeze else y


def uint8_to_model_range(images: torch.Tensor) -> torch.Tensor:
    """uint8[..., 3] in [0,255] -> float32 in [-1, 1].  openpi@215abfb model.py L118."""
    return images.to(torch.float32) / 255.0 * 2.0 - 1.0


# --------------------------------------------------------------------------------------
# 3. Train-time image augmentation (only when train=True).
#    openpi@215abfb src/openpi/models/model.py L169-L188. Runs in [0,1] space, then back to [-1,1].
#    Non-wrist cameras: RandomCrop(95%) -> Resize back -> Rotate(-5..5 deg).  All cameras: ColorJitter
#    (brightness=0.3, contrast=0.4, saturation=0.5). The ops are augmax 0.4.1 (openpi uv.lock L203-L204;
#    augmax@7095ead src/augmax/geometric.py, colorspace.py, functional/colorspace.py), whose semantics
#    differ from torchvision and are reproduced exactly here:
#      * RandomCrop: crop centre uniform inside the image; Resize: bilinear; Rotate: uniform angle in
#        [-5, 5] deg, always applied (p=1), bilinear, black (constant) fill.
#      * ColorJitter: per pixel in HSV. Applied with probability p=0.5 per image (augmax default).
#        Order brightness -> contrast -> hue -> saturation, each amount ~ U(-strength, strength).
#        hue strength defaults to 0.1 and openpi does not override it, so hue IS jittered.
#        The saturation branch in augmax 0.4.1 discards its result (colorspace.py, `F.adjust_brightness(
#        saturation, ...)` unassigned), so saturation=0.5 is effectively a no-op. We keep that no-op.
# --------------------------------------------------------------------------------------
def augment(images: torch.Tensor, key: str, generator: torch.Generator | None = None) -> torch.Tensor:
    """float32[B, H, W, 3] in [-1,1] -> same shape/range, randomly augmented per sample."""
    x = (images / 2.0 + 0.5).permute(0, 3, 1, 2)  # [B, 3, H, W] in [0, 1]
    if "wrist" not in key:
        x = _random_crop_resize(x, 0.95, generator)
        x = _random_rotate(x, 5.0, generator)
    x = _color_jitter(x, brightness=0.3, contrast=0.4, hue=0.1, p=0.5, g=generator)
    return (x.clamp(0.0, 1.0) * 2.0 - 1.0).permute(0, 2, 3, 1)


def _rand(shape, lo, hi, g):
    return torch.rand(shape, generator=g) * (hi - lo) + lo


def _random_crop_resize(x, frac, g):
    b, c, h, w = x.shape
    ch, cw = int(h * frac), int(w * frac)
    out = torch.empty_like(x)
    top = torch.randint(0, h - ch + 1, (b,), generator=g)
    left = torch.randint(0, w - cw + 1, (b,), generator=g)
    for i in range(b):  # per-sample crop offsets, like jax.vmap over rngs
        crop = x[i : i + 1, :, top[i] : top[i] + ch, left[i] : left[i] + cw]
        out[i] = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)[0]
    return out


def _random_rotate(x, max_deg, g):
    b = x.shape[0]
    theta = _rand((b,), -max_deg, max_deg, g) * math.pi / 180.0
    cos, sin = torch.cos(theta), torch.sin(theta)
    mat = torch.stack([torch.stack([cos, -sin, torch.zeros(b)], 1), torch.stack([sin, cos, torch.zeros(b)], 1)], 1)
    grid = F.affine_grid(mat, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _color_jitter(x, *, brightness, contrast, hue, p, g):
    """augmax ColorJitter (augmax@7095ead colorspace.py L220-L282, functional/colorspace.py L11-L37)."""
    b = x.shape[0]
    shape = (b, 1, 1, 1)
    h, s, v = _rgb_to_hsv(x)
    # brightness: V*(1+a) if a<0 else V*(1-a)+a   (pulls V toward 0 or toward 1)
    a = _rand(shape, -brightness, brightness, g)
    v = torch.where(a < 0, v * (1 + a), v * (1 - a) + a)
    # contrast: piecewise-linear S-curve on V with slope tan((a+1)*pi/4) in the middle
    a = _rand(shape, -contrast, contrast, g)
    slant = torch.tan((a + 1.0) * (math.pi / 4))
    p1 = (slant - slant**2) / (2 * (1 - slant**2))
    p2 = 1 - p1
    v = torch.where(v < p1, v / slant, torch.where(v > p2, v / slant + 1 - 1 / slant, slant * (v - 0.5) + 0.5))
    # hue: additive shift, wraps around
    h = (h + _rand(shape, -hue, hue, g)) % 1.0
    # saturation: no-op in augmax 0.4.1 (see comment above)
    out = _hsv_to_rgb(h, s, v)
    apply = (torch.rand((b, 1, 1, 1), generator=g) < p).to(x.dtype)
    return apply * out + (1 - apply) * x


def _rgb_to_hsv(x):
    r, g_, b_ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    v, _ = x.max(dim=1, keepdim=True)
    mn, _ = x.min(dim=1, keepdim=True)
    d = v - mn
    s = torch.where(v > 0, d / v.clamp_min(1e-12), torch.zeros_like(v))
    dd = d.clamp_min(1e-12)
    h = torch.where(v == r, (g_ - b_) / dd, torch.where(v == g_, 2.0 + (b_ - r) / dd, 4.0 + (r - g_) / dd))
    h = torch.where(d > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    return h, s, v


def _hsv_to_rgb(h, s, v):
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p_, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p_, torch.where(i == 3, p_, torch.where(i == 4, t, v)))))
    g_ = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p_, p_)))))
    b_ = torch.where(i == 0, p_, torch.where(i == 1, p_, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.cat([r, g_, b_], dim=1)


# --------------------------------------------------------------------------------------
# 4. State / action vectors: padding, delta actions, normalization.
# --------------------------------------------------------------------------------------
def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """[..., d] -> [..., target_dim] by appending `value`; no-op if d >= target_dim.
    openpi@215abfb transforms.py L423-L430."""
    cur = x.shape[axis]
    if cur >= target_dim:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, target_dim - cur)
    return np.pad(x, pad, constant_values=value)


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """make_bool_mask(6, -1, 6, -1) -> 6 True, 1 False, 6 True, 1 False.
    Positive = 'convert to delta' (joints), negative = 'keep absolute' (gripper).
    openpi@215abfb transforms.py L433-L450."""
    out: list[bool] = []
    for d in dims:
        out.extend([d > 0] * abs(d))
    return tuple(out)


def to_delta_actions(state: np.ndarray, actions: np.ndarray, mask: Sequence[bool] | None) -> np.ndarray:
    """state[B, d], actions[B, H, d] absolute -> actions where masked dims become (a_t - q_t).
    The subtraction uses the *current* state for every step of the chunk, not the previous action.
    openpi@215abfb transforms.py L204-L223 (DeltaActions)."""
    if mask is None:
        return actions
    m = np.asarray(mask)
    n = m.shape[-1]
    out = actions.copy()
    out[..., :n] -= np.where(m, state[..., None, :n], 0.0)
    return out


def to_absolute_actions(state: np.ndarray, actions: np.ndarray, mask: Sequence[bool] | None) -> np.ndarray:
    """Inverse of to_delta_actions; applied to model outputs at inference.
    openpi@215abfb transforms.py L226-L245 (AbsoluteActions)."""
    if mask is None:
        return actions
    m = np.asarray(mask)
    n = m.shape[-1]
    out = actions.copy()
    out[..., :n] += np.where(m, state[..., None, :n], 0.0)
    return out


@dataclasses.dataclass
class NormStats:
    """Per-dimension statistics computed over the training set, one set per robot/dataset.
    mean, std: float[d] with d = that robot's native dim (<= 32). openpi ships them as
    norm_stats.json next to each checkpoint. openpi@215abfb src/openpi/shared/normalize.py L10-L15."""

    mean: np.ndarray
    std: np.ndarray


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """z-score: (x - mean) / (std + 1e-6). pi0 uses z-score; pi0.5/pi0-FAST use quantiles.
    openpi@215abfb transforms.py L137-L139 and training/config.py L187 (use_quantile_norm = model != PI0)."""
    d = x.shape[-1]
    return (x - stats.mean[..., :d]) / (stats.std[..., :d] + 1e-6)


def unnormalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse; accepts the padded 32-dim model output: padded dims get mean 0, std 1 (pass-through).
    openpi@215abfb transforms.py L168-L171."""
    mean = pad_to_dim(stats.mean, x.shape[-1], value=0.0)
    std = pad_to_dim(stats.std, x.shape[-1], value=1.0)
    return x * (std + 1e-6) + mean


def extract_action_chunk(episode_actions: np.ndarray, t: int, horizon: int = ACTION_HORIZON) -> np.ndarray:
    """episode_actions[T, d], step t -> [horizon, d] = a_t, a_{t+1}, ..., clamped to the last step.

    openpi reads chunks through LeRobot's delta_timestamps = [k / fps for k in range(H)]
    (openpi@215abfb training/data_loader.py L143-L145); LeRobot clamps indices past the episode end,
    so the tail of a chunk repeats the final action. openpi does not mask those repeated steps.
    """
    idx = np.clip(np.arange(t, t + horizon), 0, len(episode_actions) - 1)
    return episode_actions[idx]


# --------------------------------------------------------------------------------------
# 5. Prompt tokenization. openpi@215abfb src/openpi/models/tokenizer.py L14-L48 (PaligemmaTokenizer).
#    Real model: SentencePiece `paligemma_tokenizer.model` (gs://big_vision/paligemma_tokenizer.model),
#    vocab 257,152, BOS id 2. pi0 format: encode(prompt, add_bos=True) + encode("\n"); the "\n"
#    marks "start of answer". We take any `encode: str -> list[int]` so the tiny config needs no download.
# --------------------------------------------------------------------------------------
class ByteEncoder:
    """Stand-in encoder for the tiny config: one token per UTF-8 byte, ids offset by 3 so 0 is pad
    and 2 is BOS, matching Gemma's special-token ids. NOT the PaliGemma vocabulary."""

    bos_id = 2

    def __call__(self, text: str, add_bos: bool = False) -> list[int]:
        ids = [b + 3 for b in text.encode("utf-8")]
        return ([self.bos_id] + ids) if add_bos else ids


class PromptTokenizer:
    def __init__(self, encode: Callable[..., list[int]], max_len: int = MAX_TOKEN_LEN):
        self.encode, self.max_len = encode, max_len

    def __call__(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """str -> (int64[max_len] ids right-padded with 0, bool[max_len] mask). Truncates if longer."""
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self.encode(cleaned, add_bos=True) + self.encode("\n")
        tokens = tokens[: self.max_len]
        n = len(tokens)
        ids = np.zeros(self.max_len, dtype=np.int64)
        ids[:n] = tokens
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:n] = True
        return ids, mask


# --------------------------------------------------------------------------------------
# 6. The whole pipeline, in openpi's order.
# --------------------------------------------------------------------------------------
def build_batch(
    raw: dict,
    norm_stats: dict[str, NormStats] | None,
    tokenizer: PromptTokenizer,
    *,
    delta_mask: Sequence[bool] | None,
    train: bool,
    generator: torch.Generator | None = None,
) -> tuple[Observation, torch.Tensor | None]:
    """Raw robot sample -> (Observation, actions).

    raw = {
      "images":  {name: uint8[B, h, w, 3]}   any subset of IMAGE_KEYS, any h, w (robot adapter has
                                              already renamed cameras to these slot names),
      "state":   float32[B, d]                native dim d <= 32, un-normalized,
      "actions": float32[B, >=50, d]          absolute actions, training only,
      "prompt":  list[str] of length B,
    }
    norm_stats = {"state": NormStats, "actions": NormStats} or None (skip normalization).
    delta_mask = make_bool_mask(...) for this robot, or None (actions already delta / absolute wanted).
    Returns Observation (contract above) and float32[B, 50, 32] actions, or None at inference.
    """
    b = raw["state"].shape[0]
    state = np.asarray(raw["state"], dtype=np.float32)
    actions = None if "actions" not in raw else np.asarray(raw["actions"], dtype=np.float32)[:, :ACTION_HORIZON]

    # (a) robot adapter: missing camera slots -> black image + mask False.
    #     openpi@215abfb src/openpi/policies/aloha_policy.py L50-L70.
    base = np.asarray(raw["images"]["base_0_rgb"])
    images = {k: np.asarray(raw["images"][k]) if k in raw["images"] else np.zeros_like(base) for k in IMAGE_KEYS}
    image_masks = {k: np.full((b,), k in raw["images"], dtype=bool) for k in IMAGE_KEYS}

    # (b) DeltaActions: joints relative to current state, grippers absolute.
    if actions is not None:
        actions = to_delta_actions(state, actions, delta_mask)

    # (c) Normalize state and actions with per-robot stats (z-score for pi0).
    if norm_stats is not None:
        state = normalize(state, norm_stats["state"])
        if actions is not None:
            actions = normalize(actions, norm_stats["actions"])

    # (d) ResizeImages to 224x224 (uint8, aspect-preserving, black pad).
    images_t = {k: resize_with_pad(torch.from_numpy(v), *IMAGE_RESOLUTION) for k, v in images.items()}

    # (e) TokenizePrompt.
    ids, masks = zip(*(tokenizer(p) for p in raw["prompt"]))

    # (f) PadStatesAndActions to ACTION_DIM. (stats may be float64 from json; the model wants float32.)
    state = pad_to_dim(state, ACTION_DIM).astype(np.float32)
    if actions is not None:
        actions = pad_to_dim(actions, ACTION_DIM).astype(np.float32)

    # (g) inside the model: uint8 -> [-1, 1]; train-time augmentation.
    for k in IMAGE_KEYS:
        img = uint8_to_model_range(images_t[k])
        images_t[k] = augment(img, k, generator) if train else img

    obs = Observation(
        images=images_t,
        image_masks={k: torch.from_numpy(v) for k, v in image_masks.items()},
        state=torch.from_numpy(state),
        tokenized_prompt=torch.from_numpy(np.stack(ids)),
        tokenized_prompt_mask=torch.from_numpy(np.stack(masks)),
    )
    return obs, None if actions is None else torch.from_numpy(actions)
