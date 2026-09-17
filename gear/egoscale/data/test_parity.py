"""对齐检查: shape / 维度, 解析性质 (往返、退化维、占位状态), 采样分布.

论文: EgoScale arXiv:2602.16710v1 Sec. 2.2, Sec. 2.3, Sec. 2.5.
上游: Isaac-GR00T@4af2b62 gr00t/model/transforms.py L240-L299, state_action.py L98-L213.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear.egoscale.data.data import (
    CAMERA_SLOTS,
    INVERTIBLE_MODES,
    NORM_MODES,
    SOURCES,
    Mixture,
    Normalizer,
    build_sample,
    build_video,
    collate,
    padding_fraction,
    paper,
    prepare_action,
    prepare_state,
    stats_from,
    tiny,
    unpad_action,
)

ALL_EMB = ("human_wild", "human_aligned", "r1pro_sharpa", "g1_trifinger")


@pytest.fixture
def cfg():
    return tiny()


def _pool(dim: int, n: int = 64, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, 8, dim, generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------
# shape / 维度
# --------------------------------------------------------------------------
def test_camera_slots_are_the_three_of_the_paper():
    """论文 Sec. 2.5: 一个头部相机 + 两个腕部相机. 顺序是模型学到的约定."""
    assert CAMERA_SLOTS == ("head", "left_wrist", "right_wrist")


@pytest.mark.parametrize("name", ALL_EMB)
def test_sample_shapes(cfg, name):
    emb = cfg.embodiments[name]
    rng = np.random.default_rng(0)
    h, w = cfg.image_hw
    frames = {s: rng.integers(0, 256, (cfg.state_horizon, h, w, 3), dtype=np.uint8)
              for s in emb.cameras}
    state = None if not emb.has_proprio else torch.randn(cfg.state_horizon, emb.state_dim)
    action = torch.randn(cfg.action_horizon, emb.action_dim)

    s = build_sample(emb, cfg, frames, "pick up the card", state, action)
    assert s["video"].shape == (cfg.state_horizon, len(CAMERA_SLOTS), h, w, 3)
    assert s["view_mask"].shape == (len(CAMERA_SLOTS),)
    assert s["state"].shape == (cfg.state_horizon, cfg.max_state_dim)
    assert s["state_mask"].shape == s["state"].shape
    assert s["action"].shape == (cfg.action_horizon, cfg.max_action_dim)
    assert s["action_mask"].shape == s["action"].shape
    assert s["embodiment_id"] == emb.embodiment_id


def test_collate_stacks_on_a_new_batch_dim(cfg):
    rng = np.random.default_rng(1)
    h, w = cfg.image_hw
    samples = []
    for name in ALL_EMB:
        emb = cfg.embodiments[name]
        frames = {s: rng.integers(0, 256, (cfg.state_horizon, h, w, 3), dtype=np.uint8)
                  for s in emb.cameras}
        state = None if not emb.has_proprio else torch.randn(cfg.state_horizon, emb.state_dim)
        samples.append(build_sample(emb, cfg, frames, "fold the t-shirt", state,
                                    torch.randn(cfg.action_horizon, emb.action_dim)))
    b = collate(samples)
    assert b["video"].shape[0] == len(ALL_EMB)
    assert b["action"].shape == (len(ALL_EMB), cfg.action_horizon, cfg.max_action_dim)
    assert b["embodiment_id"].tolist() == [cfg.embodiments[n].embodiment_id for n in ALL_EMB]
    assert isinstance(b["language"], list) and len(b["language"]) == len(ALL_EMB)


def test_state_is_truncated_but_action_asserts(cfg):
    """上游有意的不对称: state 超宽截断 (L256-L259), action 超宽断言失败 (L288-L290)."""
    wide_state = torch.randn(cfg.state_horizon, cfg.max_state_dim + 7)
    s, mask, _ = prepare_state(wide_state, cfg.max_state_dim, cfg.state_horizon)
    assert s.shape[-1] == cfg.max_state_dim and mask.all()

    with pytest.raises(AssertionError):
        prepare_action(torch.randn(cfg.action_horizon, cfg.max_action_dim + 1),
                       cfg.max_action_dim, cfg.action_horizon)


# --------------------------------------------------------------------------
# 解析性质
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["min_max", "mean_std"])
def test_normalizer_roundtrip(mode):
    x = _pool(11)
    n = Normalizer(mode, stats_from(x))
    assert torch.allclose(n.inverse(n.forward(x)), x, atol=1e-4)


def test_q99_is_lossy_outside_the_quantile_band():
    """q99 在 forward 里 clamp 到 [-1, 1] (上游 L136), 所以尾部 1% 的值不可还原.

    这不是实现瑕疵而是上游的实际语义: q99 的目的就是压掉离群值. min_max 与 mean_std 没有
    这一步, 因而是严格可逆的. 三种模式的这个差别在换归一化模式时会改变数据语义.
    """
    x = _pool(11)
    n = Normalizer("q99", stats_from(x))
    back = n.inverse(n.forward(x))
    stats = stats_from(x)
    inside = (x >= torch.as_tensor(stats["q01"])) & (x <= torch.as_tensor(stats["q99"]))
    assert torch.allclose(back[inside], x[inside], atol=1e-4)
    assert not torch.allclose(back[~inside], x[~inside], atol=1e-4)
    assert inside.float().mean() > 0.97  # 只有尾部被压掉


def test_scale_has_no_inverse_upstream():
    """上游 Normalizer.inverse (L193-L213) 没有 scale 分支, 落到 ValueError."""
    x = _pool(5)
    with pytest.raises(ValueError):
        Normalizer("scale", stats_from(x)).inverse(x)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Normalizer("zscore", stats_from(_pool(3)))


def test_min_max_output_range():
    x = _pool(9)
    y = Normalizer("min_max", stats_from(x)).forward(x)
    assert y.min() >= -1 - 1e-5 and y.max() <= 1 + 1e-5


def test_degenerate_dims():
    """退化维 (min == max / std == 0) 的处理三种模式不一致, 这是上游的实际行为, 不是笔误.

    min_max 置 0 (L170-L173); q99 与 mean_std 保留原值 (L133-L135 / L149-L151).
    """
    x = _pool(4)
    x[..., 2] = 7.5  # 这一维常数 -> min == max, std == 0
    stats = stats_from(x)

    y_mm = Normalizer("min_max", stats).forward(x)
    assert torch.allclose(y_mm[..., 2], torch.zeros_like(y_mm[..., 2]))

    y_q = Normalizer("q99", stats).forward(x)
    y_ms = Normalizer("mean_std", stats).forward(x)
    # q99 保留原值后还要过 clamp(-1, 1), 所以 7.5 被截到 1.0
    assert torch.allclose(y_q[..., 2], torch.ones_like(y_q[..., 2]))
    assert torch.allclose(y_ms[..., 2], x[..., 2])


def test_padding_roundtrip(cfg):
    """原生 -> 归一化 -> padding -> 逆 padding -> 逆归一化 必须是恒等."""
    for name in ALL_EMB:
        emb = cfg.embodiments[name]
        pool = _pool(emb.action_dim, seed=hash(name) % 1000)
        norm = Normalizer(cfg.norm_mode, stats_from(pool))
        native = torch.randn(cfg.action_horizon, emb.action_dim)
        padded, mask, _ = prepare_action(norm.forward(native), cfg.max_action_dim,
                                         cfg.action_horizon)
        back = norm.inverse(unpad_action(padded, mask))
        assert back.shape == native.shape
        assert torch.allclose(back, native, atol=1e-4), name


def test_padded_dims_are_zero_and_masked_out(cfg):
    emb = cfg.embodiments["g1_trifinger"]
    a, mask, _ = prepare_action(torch.randn(cfg.action_horizon, emb.action_dim),
                                cfg.max_action_dim, cfg.action_horizon)
    assert (a[:, emb.action_dim:] == 0).all()
    assert mask[:, : emb.action_dim].all() and not mask[:, emb.action_dim:].any()
    assert int(mask.sum()) == cfg.action_horizon * emb.action_dim


@pytest.mark.parametrize("name", ["human_wild", "human_aligned"])
def test_human_samples_carry_the_placeholder_state(cfg, name):
    """论文 Sec. 2.3: 人类演示没有 q_t. 数据侧是全 0 + 全 False mask (上游 L245-L250)."""
    emb = cfg.embodiments[name]
    rng = np.random.default_rng(2)
    h, w = cfg.image_hw
    frames = {s: rng.integers(0, 256, (cfg.state_horizon, h, w, 3), dtype=np.uint8)
              for s in emb.cameras}
    # 即使调用方硬塞一个 state, has_proprio=False 也必须让它走占位分支
    s = build_sample(emb, cfg, frames, "iron the t-shirt",
                     torch.randn(cfg.state_horizon, 58),
                     torch.randn(cfg.action_horizon, emb.action_dim))
    assert s["has_proprio"] is False
    assert (s["state"] == 0).all()
    assert not s["state_mask"].any()


def test_missing_camera_slot_is_black_filled(cfg):
    """读者契约: 缺相机填黑 + mask, 不是跳过. 见 README Sec. 8."""
    h, w = cfg.image_hw
    rng = np.random.default_rng(3)
    frames = {"head": rng.integers(1, 256, (cfg.state_horizon, h, w, 3), dtype=np.uint8)}
    video, mask = build_video(frames, cfg.image_hw, cfg.state_horizon)
    assert video.shape == (cfg.state_horizon, 3, h, w, 3)
    assert mask.tolist() == [True, False, False]
    assert (video[:, 1] == 0).all() and (video[:, 2] == 0).all()
    assert video[:, 0].any()


def test_inference_sample_has_no_action(cfg):
    """上游 apply_single 只在 training=True 时准备 action / action_mask (L316-L323)."""
    emb = cfg.embodiments["r1pro_sharpa"]
    rng = np.random.default_rng(4)
    h, w = cfg.image_hw
    frames = {s: rng.integers(0, 256, (cfg.state_horizon, h, w, 3), dtype=np.uint8)
              for s in emb.cameras}
    s = build_sample(emb, cfg, frames, "unscrew the cap", torch.randn(cfg.state_horizon, 58),
                     None, training=False)
    assert "action" not in s and "action_mask" not in s


def test_paper_config_refuses_to_run():
    """未披露的值是 None 占位, 不能被悄悄当成能跑的默认值. 见 README Sec. 8."""
    cfg = paper()
    assert cfg.max_action_dim is None and cfg.norm_mode is None and cfg.action_horizon is None
    assert all(e.state_dim is None for e in cfg.embodiments.values())
    with pytest.raises(ValueError):
        Mixture(SOURCES, cfg.mixture_weights)


# --------------------------------------------------------------------------
# 分布检查
# --------------------------------------------------------------------------
def test_mixture_empirical_frequency_matches_weights():
    mix = Mixture.proportional()
    drawn = mix.sample(np.random.default_rng(5), 20000)
    for src, w in zip(mix.sources, mix.weights):
        emp = sum(d.name == src.name for d in drawn) / len(drawn)
        assert abs(emp - w) < 0.02, f"{src.name}: {emp:.4f} vs {w:.4f}"


def test_hours_match_the_paper():
    """论文 Sec. 2.2: 总计 20,854 小时, 其中 EgoDex 829 小时; Stage II 约 50 h 人类 + 4 h 机器人."""
    by = {s.name: s for s in SOURCES}
    assert by["in_the_wild"].hours + by["egodex"].hours == 20854.0
    assert by["egodex"].hours == 829.0
    assert (by["aligned_human"].hours, by["aligned_robot"].hours) == (50.0, 4.0)
    assert {s.stage for s in SOURCES} == {1, 2}


def test_padding_fraction(cfg):
    frac = padding_fraction(cfg)
    assert set(frac) == set(ALL_EMB)
    for name, f in frac.items():
        emb = cfg.embodiments[name]
        assert f == pytest.approx(1 - emb.action_dim / cfg.max_action_dim)
        assert 0.0 <= f < 1.0
    # G1 的三指手动作维度最小, padding 比例最高
    assert frac["g1_trifinger"] == max(frac.values())


def test_all_norm_modes_are_constructible():
    stats = stats_from(_pool(6))
    for mode in NORM_MODES:
        y = Normalizer(mode, stats).forward(_pool(6))
        assert y.shape == (64, 8, 6) and torch.isfinite(y).all()
