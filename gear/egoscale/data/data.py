"""EgoScale 的样本组织: 相机槽位, 人类占位状态, 跨本体 padding 与 mask, 归一化, 数据混合.

上游: NVIDIA/Isaac-GR00T @ 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96,
      gr00t/model/transforms.py L240-L299 (_prepare_state / _prepare_action),
      gr00t/data/transform/state_action.py L98-L213 (Normalizer),
      gr00t/data/transform/concat.py L91-L112 (视频拼接), gr00t/data/embodiment_tags.py L19-L47.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.2, Sec. 2.3, Sec. 2.5.
许可: 上游 Isaac-GR00T 为 Apache-2.0; 本文件为 PyTorch 重写 (re-implements, does not copy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# 论文 Sec. 2.5: 一个头部相机提供第一视角, 两个腕部相机朝掌心. 顺序是模型学到的约定, 不能换.
CAMERA_SLOTS = ("head", "left_wrist", "right_wrist")

NORM_MODES = ("min_max", "q99", "mean_std", "scale", "binary")
INVERTIBLE_MODES = ("min_max", "q99", "mean_std")


# ---------------------------------------------------------------------------
# 本体表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Embodiment:
    """一个本体. 论文 Sec. 2.3: 各本体只换输入输出的 MLP 适配器, backbone 与 DiT 全共享."""

    name: str
    embodiment_id: int  # 选哪一对适配器; EgoScale 的映射未披露, 见 README Sec. 8
    state_dim: int | None  # 未披露, 见 README Sec. 8
    action_dim: int | None  # 未披露 (依赖未披露的 rot_rep), 见 README Sec. 8
    has_proprio: bool  # 人类演示为 False -> 论文 Sec. 2.3 的占位 token 分支
    cameras: tuple[str, ...]


def embodiment_table(state_dims: dict[str, int], action_dims: dict[str, int]) -> dict[str, Embodiment]:
    """四个本体. 维度由调用方给: paper() 给 None, tiny() 给可跑通的非论文值."""
    return {
        # Stage I 的 in-the-wild 数据: 只有头部相机, 没有本体感受
        "human_wild": Embodiment("human_wild", 0, state_dims["human"], action_dims["human"],
                                 False, ("head",)),
        # Stage II 的对齐人类数据: 论文 Sec. 2.2 说用与机器人相同的三路相机配置采集
        "human_aligned": Embodiment("human_aligned", 1, state_dims["human"], action_dims["human"],
                                    False, CAMERA_SLOTS),
        # 论文 Sec. 2.5: 双 7 自由度臂 (相对末端位姿) + 22 自由度 Sharpa 手 (关节空间)
        "r1pro_sharpa": Embodiment("r1pro_sharpa", 2, state_dims["r1pro"], action_dims["r1pro"],
                                   True, CAMERA_SLOTS),
        # 论文 Sec. 3.5 / App. D.1: 更短的臂, 7 自由度三指手
        "g1_trifinger": Embodiment("g1_trifinger", 3, state_dims["g1"], action_dims["g1"],
                                   True, CAMERA_SLOTS),
    }


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    max_state_dim: int | None  # 未披露; 上游是 64, 见 README Sec. 8
    max_action_dim: int | None  # 未披露; 上游是 32, 但装不下 62 维, 见 README Sec. 8
    state_horizon: int | None  # 未披露; 上游 observation_indices = [0] -> 1
    action_horizon: int | None  # 未披露; GR00T N1 用 H = 16, 见 README Sec. 8
    image_hw: tuple[int, int] | None  # 未披露; 上游 resize 到 224x224
    norm_mode: str | None  # 未披露; 上游各配置默认 min_max
    crop_scale: float | None  # 未披露; 上游 0.95
    color_jitter: tuple[float, float, float, float] | None  # 未披露; 上游 (.3,.4,.5,.08)
    embodiments: dict[str, Embodiment] = field(default_factory=dict)
    mixture_weights: np.ndarray | None = None  # 未披露, 见 README Sec. 8


def paper() -> DataConfig:
    """论文规模. 未披露的字段一律 None (占位), 见 README Sec. 8."""
    nones = {k: None for k in ("human", "r1pro", "g1")}
    return DataConfig(
        max_state_dim=None,
        max_action_dim=None,
        state_horizon=None,
        action_horizon=None,
        image_hw=None,
        norm_mode=None,
        crop_scale=None,
        color_jitter=None,
        embodiments=embodiment_table(nones, nones),
        mixture_weights=None,
    )


def tiny() -> DataConfig:
    """CPU 上几秒钟跑通. 下面每个值都不是论文值, 只为让代码路径可执行."""
    return DataConfig(
        max_state_dim=64,  # tiny only, not a paper value (上游 GR00T 的值)
        max_action_dim=64,  # tiny only, not a paper value (上游是 32, 装不下 62)
        state_horizon=1,  # tiny only, not a paper value
        action_horizon=8,  # tiny only, not a paper value
        image_hw=(32, 32),  # tiny only, not a paper value (上游 resize 到 224x224)
        norm_mode="min_max",  # tiny only, not a paper value (上游各配置的默认)
        crop_scale=0.95,  # tiny only, not a paper value (上游 data_config.py L172)
        color_jitter=(0.3, 0.4, 0.5, 0.08),  # tiny only (上游 data_config.py L174-L180)
        embodiments=embodiment_table(
            # 人类: 无本体感受 -> state_dim 0. 动作是 Sec. 3.6 的 full 空间, rotation_6d 下 2x31.
            # R1Pro: 双 7 自由度臂 + 双 22 自由度手 = 58 维本体感受, 动作与人类同构.
            # G1: 双 7 自由度臂 + 双 7 自由度三指手 = 28 维, 动作 2x(9+7) = 32.
            # 这些维度都是本仓库按论文 Sec. 2.5 / Sec. 3.5 的文字描述算的, 非论文披露值.
            state_dims={"human": 0, "r1pro": 58, "g1": 28},
            action_dims={"human": 62, "r1pro": 62, "g1": 32},
        ),
        mixture_weights=None,  # 默认按小时数成比例, 见 Mixture.proportional
    )


# ---------------------------------------------------------------------------
# 1. 归一化 (上游 state_action.py L98-L213 的五种模式与它们各自的退化维处理)
# ---------------------------------------------------------------------------
class Normalizer:
    """逐维归一化. EgoScale 未披露用哪种模式, 所以 mode 必须显式传入, 见 README Sec. 8."""

    def __init__(self, mode: str, stats: dict[str, np.ndarray]) -> None:
        if mode not in NORM_MODES:
            raise ValueError(f"mode 必须是 {NORM_MODES} 之一, 收到 {mode!r}")
        self.mode = mode
        self.stats = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in stats.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stats
        if self.mode == "min_max":
            lo, hi = s["min"], s["max"]
            ok = lo != hi
            out = torch.zeros_like(x)
            out[..., ok] = 2 * (x[..., ok] - lo[ok]) / (hi[ok] - lo[ok]) - 1
            # 上游 L170-L173: min == max 的维度置 0 (注掉了"保留原值"那行)
            return out
        if self.mode == "q99":
            lo, hi = s["q01"], s["q99"]
            ok = lo != hi
            out = torch.zeros_like(x)
            out[..., ok] = 2 * (x[..., ok] - lo[ok]) / (hi[ok] - lo[ok]) - 1
            out[..., ~ok] = x[..., ~ok]  # 上游 L133-L135: 退化维保留原值
            return torch.clamp(out, -1, 1)  # 上游 L136
        if self.mode == "mean_std":
            mu, sd = s["mean"], s["std"]
            ok = sd != 0
            out = torch.zeros_like(x)
            out[..., ok] = (x[..., ok] - mu[ok]) / sd[ok]
            out[..., ~ok] = x[..., ~ok]  # 上游 L149-L151: 退化维保留原值
            return out
        if self.mode == "scale":
            amax = torch.maximum(s["min"].abs(), s["max"].abs())
            ok = amax != 0
            out = torch.zeros_like(x)
            out[..., ok] = x[..., ok] / amax[ok]
            return out
        return (x > 0.5).to(x.dtype)  # binary, 上游 L185-L187

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stats
        if self.mode == "min_max":
            return (x + 1) / 2 * (s["max"] - s["min"]) + s["min"]
        if self.mode == "q99":
            return (x + 1) / 2 * (s["q99"] - s["q01"]) + s["q01"]
        if self.mode == "mean_std":
            return x * s["std"] + s["mean"]
        if self.mode == "binary":
            return (x > 0.5).to(x.dtype)
        # 上游 L193-L213 的 inverse 里没有 "scale" 分支, 直接落到 ValueError
        raise ValueError(f"上游不支持 {self.mode!r} 的逆变换")


def stats_from(x: torch.Tensor) -> dict[str, np.ndarray]:
    """从一批原生量算逐维统计量. EgoScale 未披露统计量来自哪个数据集, 见 README Sec. 8."""
    flat = x.reshape(-1, x.shape[-1])
    return {
        "min": flat.min(0).values.numpy(),
        "max": flat.max(0).values.numpy(),
        "mean": flat.mean(0).numpy(),
        "std": flat.std(0).numpy(),
        "q01": torch.quantile(flat, 0.01, dim=0).numpy(),
        "q99": torch.quantile(flat, 0.99, dim=0).numpy(),
    }


# ---------------------------------------------------------------------------
# 2. padding 与 mask (上游 transforms.py L240-L299)
# ---------------------------------------------------------------------------
def prepare_state(
    state: torch.Tensor | None, max_state_dim: int, state_horizon: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """state 为 None (人类样本) 时走上游 L245-L250: 全 0 + 全 False mask.

    EgoScale 在模型侧把这个位置换成可学习占位 token (论文 Sec. 2.3), 见 ../dit.
    """
    if state is None:
        zeros = torch.zeros(state_horizon, max_state_dim)
        return zeros, torch.zeros_like(zeros, dtype=torch.bool), state_horizon

    assert state.shape[0] == state_horizon, f"{state.shape=} vs {state_horizon=}"
    n = state.shape[-1]
    if n > max_state_dim:
        # 上游 L256-L259: state 超宽是截断, 不是报错
        state, n = state[:, :max_state_dim], max_state_dim
    else:
        state = torch.nn.functional.pad(state, (0, max_state_dim - n))
    mask = torch.zeros_like(state, dtype=torch.bool)
    mask[:, :n] = True
    return state, mask, state.shape[0]


def prepare_action(
    action: torch.Tensor, max_action_dim: int, action_horizon: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """与 state 不同, 上游 L288-L290 对超宽 action 是断言失败: 截断会静默丢掉自由度."""
    assert action.shape[0] == action_horizon, f"{action.shape=} vs {action_horizon=}"
    n = action.shape[-1]
    assert n <= max_action_dim, f"动作维 {n} 超过 max_action_dim {max_action_dim}"
    padded = torch.nn.functional.pad(action, (0, max_action_dim - n))
    mask = torch.zeros(action_horizon, max_action_dim, dtype=torch.bool)
    mask[:, :n] = True
    return padded, mask, action_horizon


def unpad_action(action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """逆过程: 按 mask 截回本体原生维度. 必须在逆归一化之后做, 见 README Sec. 3."""
    n = int(mask[0].sum())
    return action[..., :n]


# ---------------------------------------------------------------------------
# 3. 相机槽位 (上游 concat.py L91-L112 的拼接顺序)
# ---------------------------------------------------------------------------
def build_video(
    frames_by_slot: dict[str, np.ndarray], hw: tuple[int, int], n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """固定槽位拼成 (T, V, H, W, 3); 缺的槽位填黑并在 view_mask 里标 False.

    上游 ConcatTransform 要求所有声明的视频键都存在, 缺一个就断言失败; 填黑 + mask 是本仓库
    按读者契约补的处理, EgoScale 未说, 见 README Sec. 8.
    """
    h, w = hw
    views, mask = [], []
    for slot in CAMERA_SLOTS:
        if slot in frames_by_slot:
            v = frames_by_slot[slot]
            assert v.shape == (n_frames, h, w, 3), f"{slot}: {v.shape}"
            views.append(v)
            mask.append(True)
        else:
            views.append(np.zeros((n_frames, h, w, 3), dtype=np.uint8))
            mask.append(False)
    return np.stack(views, axis=1), np.array(mask, dtype=bool)


def augment_params(cfg: DataConfig) -> dict:
    """图像增广的参数表. 全部是上游 GR00T 的值, EgoScale 未披露自己的, 见 README Sec. 8."""
    return {
        "crop_scale": cfg.crop_scale,  # 上游 data_config.py L172
        "resize_hw": cfg.image_hw,  # 上游 L173 是 224x224 linear
        "color_jitter": cfg.color_jitter,  # 上游 L174-L180
    }


# ---------------------------------------------------------------------------
# 4. 组装一个样本 (上游 transforms.py L301-L338)
# ---------------------------------------------------------------------------
def build_sample(
    emb: Embodiment,
    cfg: DataConfig,
    frames_by_slot: dict[str, np.ndarray],
    language: str,
    state: torch.Tensor | None,
    action: torch.Tensor | None,
    training: bool = True,
) -> dict:
    """一条演示 -> 一个可直接进 batch 的样本."""
    video, view_mask = build_video(frames_by_slot, cfg.image_hw, cfg.state_horizon)
    s, s_mask, _ = prepare_state(
        None if not emb.has_proprio else state, cfg.max_state_dim, cfg.state_horizon
    )
    out = {
        "video": video,
        "view_mask": view_mask,
        "language": language,
        "state": s,
        "state_mask": s_mask,
        "has_proprio": emb.has_proprio,
        "embodiment_id": emb.embodiment_id,
    }
    if training:
        # 上游 L316-L323: action / action_mask 只在训练时存在
        assert action is not None, "训练样本必须给 action"
        a, a_mask, _ = prepare_action(action, cfg.max_action_dim, cfg.action_horizon)
        out["action"], out["action_mask"] = a, a_mask
    return out


def collate(samples: list[dict]) -> dict:
    """沿新的 batch 维堆叠. 上游 transforms.py L54-L82 的 collate 做同一件事."""
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        if k == "language":
            out[k] = vals
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.from_numpy(np.stack(vals)) if isinstance(vals[0], np.ndarray) \
                else torch.as_tensor(vals)
    return out


# ---------------------------------------------------------------------------
# 5. 数据混合 (论文 Sec. 2.2 的各来源小时数; 权重未披露)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    name: str
    hours: float
    embodiment: str
    stage: int


# 论文 Sec. 2.2. in-the-wild 的 20025 是 20854 - 829, 由本仓库相减得到, 见 README Sec. 8.
SOURCES = (
    Source("in_the_wild", 20025.0, "human_wild", 1),
    Source("egodex", 829.0, "human_wild", 1),
    Source("aligned_human", 50.0, "human_aligned", 2),
    Source("aligned_robot", 4.0, "r1pro_sharpa", 2),
)


class Mixture:
    def __init__(self, sources: tuple[Source, ...], weights: np.ndarray | None) -> None:
        self.sources = sources
        if weights is None:
            raise ValueError("混合权重未披露 (见 README Sec. 8); 用 Mixture.proportional() 或显式传入")
        w = np.asarray(weights, dtype=float)
        assert w.shape == (len(sources),) and (w >= 0).all() and w.sum() > 0
        self.weights = w / w.sum()

    @classmethod
    def proportional(cls, sources: tuple[Source, ...] = SOURCES) -> "Mixture":
        """按小时数成比例. 这是本仓库的选择, 不是论文值, 见 README Sec. 8."""
        return cls(sources, np.array([s.hours for s in sources]))

    def sample(self, rng: np.random.Generator, n: int) -> list[Source]:
        idx = rng.choice(len(self.sources), size=n, p=self.weights)
        return [self.sources[i] for i in idx]


def padding_fraction(cfg: DataConfig) -> dict[str, float]:
    """每个本体有多大比例的 action 维度是补出来的. 这是 figs/padding.png 的数据来源."""
    return {
        e.name: 1.0 - e.action_dim / cfg.max_action_dim for e in cfg.embodiments.values()
    }


# ---------------------------------------------------------------------------
# 6. 一次 tiny 运行
# ---------------------------------------------------------------------------
def _fake_pool(emb: Embodiment, cfg: DataConfig, n: int = 64):
    """造一小批该本体的原生 state / action, 用来算归一化统计量.

    统计量必须来自一个数据集而不是单个样本: 只有一个样本时每一维都退化成 min == max,
    min_max 会把整条向量置 0 (上游 L170-L173). 真实的统计量来源 EgoScale 未披露, 见 README Sec. 8.
    """
    state = None if not emb.has_proprio else torch.randn(n, cfg.state_horizon, emb.state_dim)
    return state, torch.randn(n, cfg.action_horizon, emb.action_dim)


def _fake_demo(emb: Embodiment, cfg: DataConfig, rng: np.random.Generator):
    h, w = cfg.image_hw
    frames = {
        slot: rng.integers(0, 256, (cfg.state_horizon, h, w, 3), dtype=np.uint8)
        for slot in emb.cameras
    }
    state = None if not emb.has_proprio else torch.randn(cfg.state_horizon, emb.state_dim)
    action = torch.randn(cfg.action_horizon, emb.action_dim)
    return frames, state, action


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = tiny()
    print(f"config: tiny(), max_state_dim={cfg.max_state_dim}, "
          f"max_action_dim={cfg.max_action_dim}, H={cfg.action_horizon}, "
          f"norm_mode={cfg.norm_mode!r}, image {cfg.image_hw}")
    print(f"camera slots: {CAMERA_SLOTS}")
    print(f"augment: {augment_params(cfg)}")

    samples = []
    for name in ("human_wild", "human_aligned", "r1pro_sharpa", "g1_trifinger"):
        emb = cfg.embodiments[name]
        frames, state, action = _fake_demo(emb, cfg, rng)

        pool_s, pool_a = _fake_pool(emb, cfg)
        norm_a = Normalizer(cfg.norm_mode, stats_from(pool_a))
        a_n = norm_a.forward(action)
        s_n = None if state is None else \
            Normalizer(cfg.norm_mode, stats_from(pool_s)).forward(state)

        smp = build_sample(emb, cfg, frames, "roll the t-shirt and put it into the basket",
                           s_n, a_n)
        samples.append(smp)
        print(f"\n[{name}] id={emb.embodiment_id} has_proprio={emb.has_proprio} "
              f"cameras={emb.cameras}")
        print(f"  video {tuple(smp['video'].shape)} view_mask {smp['view_mask']}")
        print(f"  state {tuple(smp['state'].shape)} native={emb.state_dim} "
              f"mask True={int(smp['state_mask'].sum())} "
              f"all-zero={bool((smp['state'] == 0).all())}")
        print(f"  action {tuple(smp['action'].shape)} native={emb.action_dim} "
              f"mask True/row={int(smp['action_mask'][0].sum())} "
              f"padding={1 - emb.action_dim / cfg.max_action_dim:.1%}")
        back = norm_a.inverse(unpad_action(smp["action"], smp["action_mask"]))
        print(f"  roundtrip max |err| = {float((back - action).abs().max()):.2e}")

    batch = collate(samples)
    print(f"\nbatch: " + ", ".join(
        f"{k} {tuple(v.shape)}" if isinstance(v, torch.Tensor) else f"{k} list[{len(v)}]"
        for k, v in batch.items()))

    mix = Mixture.proportional()
    drawn = mix.sample(rng, 10000)
    freq = {s.name: sum(d.name == s.name for d in drawn) / len(drawn) for s in mix.sources}
    print(f"\nmixture weights (hours-proportional, this repo's choice, not a paper value):")
    for s, w in zip(mix.sources, mix.weights):
        print(f"  {s.name:15s} stage {s.stage}  {s.hours:9.1f} h  w={w:.5f}  "
              f"empirical={freq[s.name]:.5f}")
    print("padding fraction per embodiment: "
          f"{ {k: f'{v:.1%}' for k, v in padding_fraction(cfg).items()} }")

    infer_only = build_sample(cfg.embodiments["r1pro_sharpa"], cfg,
                              *_fake_demo(cfg.embodiments["r1pro_sharpa"], cfg, rng)[:1],
                              "unscrew the cap from the bottle",
                              torch.zeros(cfg.state_horizon, 58), None, training=False)
    print(f"\ntraining=False keys: {sorted(infer_only)}  (no action / action_mask)")


if __name__ == "__main__":
    main()
