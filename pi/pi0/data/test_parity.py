"""Shape / constant parity checks for the pi0 data pipeline.

Reference values come from openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
(src/openpi/models/pi0_config.py L25-L39, src/openpi/models/model.py L39-L47).
Run on CPU: `uv run pytest pi/pi0/data -q`.
"""

import numpy as np
import pytest
import torch

from pi.pi0.data import data as D

B = 2


@pytest.fixture
def raw_sample():
    """A raw sample the way a robot-specific adapter hands it over (see README, sec. 4.1)."""
    rng = np.random.default_rng(0)
    return {
        "images": {
            "base_0_rgb": rng.integers(0, 256, size=(B, 480, 640, 3), dtype=np.uint8),
            "left_wrist_0_rgb": rng.integers(0, 256, size=(B, 224, 224, 3), dtype=np.uint8),
        },
        "state": rng.standard_normal((B, 14)).astype(np.float32),
        "actions": rng.standard_normal((B, 60, 14)).astype(np.float32),
        "prompt": ["fold the towel"] * B,
    }


def test_constants_match_openpi():
    assert D.ACTION_DIM == 32
    assert D.ACTION_HORIZON == 50
    assert D.MAX_TOKEN_LEN == 48
    assert D.IMAGE_RESOLUTION == (224, 224)
    assert D.IMAGE_KEYS == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def test_resize_with_pad_keeps_aspect_and_pads_black():
    img = torch.full((B, 480, 640, 3), 255, dtype=torch.uint8)
    out = D.resize_with_pad(img, 224, 224)
    assert out.shape == (B, 224, 224, 3) and out.dtype == torch.uint8
    # 640x480 -> 224x168, padded 28 rows top and bottom with 0 (black)
    assert out[:, :28].max() == 0 and out[:, -28:].max() == 0
    assert out[:, 28:-28].min() == 255
    f = D.resize_with_pad(torch.ones((B, 480, 640, 3)), 224, 224)
    assert f.dtype == torch.float32 and f[0, 0, 0, 0].item() == -1.0  # float pad value is -1


def test_pad_to_dim_and_bool_mask():
    x = np.ones((B, 14))
    assert D.pad_to_dim(x, 32).shape == (B, 32)
    assert D.pad_to_dim(x, 32)[:, 14:].sum() == 0
    assert D.pad_to_dim(x, 8).shape == (B, 14)  # never truncates
    assert D.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert D.make_bool_mask(6, -1, 6, -1) == (True,) * 6 + (False,) + (True,) * 6 + (False,)


def test_delta_actions_roundtrip():
    rng = np.random.default_rng(1)
    state = rng.standard_normal((B, 14)).astype(np.float32)
    actions = rng.standard_normal((B, 50, 14)).astype(np.float32)
    mask = D.make_bool_mask(6, -1, 6, -1)
    delta = D.to_delta_actions(state, actions, mask)
    # gripper dims (6 and 13) untouched, joint dims shifted by state
    np.testing.assert_allclose(delta[..., 6], actions[..., 6])
    np.testing.assert_allclose(delta[..., 0], actions[..., 0] - state[:, None, 0], rtol=1e-6)
    np.testing.assert_allclose(D.to_absolute_actions(state, delta, mask), actions, rtol=1e-5, atol=1e-6)


def test_normalize_roundtrip_and_padding():
    rng = np.random.default_rng(2)
    stats = D.NormStats(mean=rng.standard_normal(14), std=rng.random(14) + 0.5)
    x = rng.standard_normal((B, 50, 14)).astype(np.float32)
    z = D.normalize(x, stats)
    np.testing.assert_allclose(D.unnormalize(z, stats), x, rtol=1e-4, atol=1e-5)
    # unnormalize accepts padded model outputs (32 dims); padded dims pass through
    z32 = D.pad_to_dim(z, 32)
    out = D.unnormalize(z32, stats)
    assert out.shape == (B, 50, 32)
    np.testing.assert_allclose(out[..., 14:], 0.0)


def test_action_chunk_clamps_at_episode_end():
    ep = np.arange(20, dtype=np.float32)[:, None]  # T=20, d=1
    chunk = D.extract_action_chunk(ep, t=18, horizon=5)
    assert chunk.shape == (5, 1)
    np.testing.assert_array_equal(chunk[:, 0], [18, 19, 19, 19, 19])


def test_tokenizer_shapes():
    tok = D.PromptTokenizer(D.ByteEncoder(), max_len=D.MAX_TOKEN_LEN)
    ids, mask = tok("fold the towel")
    assert ids.shape == (48,) and ids.dtype == np.int64
    assert mask.shape == (48,) and mask.dtype == bool
    assert mask.sum() == len("fold the towel") + 2  # + bos + "\n"
    ids_long, mask_long = tok("x" * 100)
    assert ids_long.shape == (48,) and mask_long.all()


def test_augment_preserves_shape_and_range():
    g = torch.Generator().manual_seed(0)
    img = torch.rand((B, 224, 224, 3)) * 2 - 1
    for key in D.IMAGE_KEYS:
        out = D.augment(img, key, g)
        assert out.shape == img.shape
        assert out.min() >= -1.0 and out.max() <= 1.0


def test_full_pipeline_io_contract(raw_sample):
    stats = {
        "state": D.NormStats(mean=np.zeros(14), std=np.ones(14)),
        "actions": D.NormStats(mean=np.zeros(14), std=np.ones(14)),
    }
    tok = D.PromptTokenizer(D.ByteEncoder())
    obs, actions = D.build_batch(
        raw_sample, stats, tok, delta_mask=D.make_bool_mask(6, -1, 6, -1), train=True,
        generator=torch.Generator().manual_seed(0),
    )
    # --- Observation contract (README sec. 1) ---
    assert set(obs.images) == set(D.IMAGE_KEYS)
    for k in D.IMAGE_KEYS:
        assert obs.images[k].shape == (B, 224, 224, 3) and obs.images[k].dtype == torch.float32
        assert obs.images[k].min() >= -1.0 and obs.images[k].max() <= 1.0
        assert obs.image_masks[k].shape == (B,) and obs.image_masks[k].dtype == torch.bool
    assert obs.image_masks["right_wrist_0_rgb"].tolist() == [False] * B  # missing camera
    assert obs.state.shape == (B, 32) and obs.state.dtype == torch.float32
    assert obs.tokenized_prompt.shape == (B, 48) and obs.tokenized_prompt.dtype == torch.int64
    assert obs.tokenized_prompt_mask.shape == (B, 48) and obs.tokenized_prompt_mask.dtype == torch.bool
    # --- Actions contract ---
    assert actions.shape == (B, 50, 32) and actions.dtype == torch.float32
    assert torch.all(actions[..., 14:] == 0)


def test_full_pipeline_inference_mode_has_no_actions(raw_sample):
    raw_sample.pop("actions")
    tok = D.PromptTokenizer(D.ByteEncoder())
    obs, actions = D.build_batch(raw_sample, None, tok, delta_mask=None, train=False)
    assert actions is None
    assert obs.state.shape == (B, 32)


def test_hsv_roundtrip():
    x = torch.rand((B, 3, 8, 8))
    h, s, v = D._rgb_to_hsv(x)
    torch.testing.assert_close(D._hsv_to_rgb(h, s, v), x, rtol=1e-5, atol=1e-5)
