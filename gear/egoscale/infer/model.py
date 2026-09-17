"""端到端推理链路: 三路图像 + 指令 -> 条件向量 -> K 步去噪 -> 机器人命令. 只有推理路径.

上游: EgoScale 未开源; 装配对照 NVIDIA/Isaac-GR00T @
      4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 的 gr00t/model/gr00t_n1.py L171-L198.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.3, Sec. 2.5; GR00T N1 arXiv:2503.14734v2 Sec. 2.1.
许可: 上游 Isaac-GR00T 为 Apache-2.0; 本文件为 PyTorch 重写 (re-implements, does not copy).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from gear.egoscale.action.action import ROT_DIM
from gear.egoscale.backbone.model import VisionLanguageBackbone
from gear.egoscale.backbone.model import tiny as backbone_tiny
from gear.egoscale.data.data import (
    CAMERA_SLOTS,
    DataConfig,
    Normalizer,
    build_sample,
    build_video,
    prepare_state,
    stats_from,
    unpad_action,
)
from gear.egoscale.data.data import tiny as data_tiny
from gear.egoscale.dit.model import ActionExpert
from gear.egoscale.dit.model import tiny as dit_tiny


@dataclass
class Observation:
    images: dict[str, np.ndarray]  # 只给该本体实际有的槽位
    instruction: str
    state: torch.Tensor | None  # 原生单位; 人类演示为 None
    embodiment: str


@dataclass
class RobotCommand:
    """论文 Sec. 2.5 定义的两段接口."""

    arm_delta_pose: torch.Tensor  # (H, 2, 3 + R) 相对末端位姿增量, 不需要 IK
    hand_joints: torch.Tensor  # (H, 2, n_hand) 目标关节角


class EgoScalePolicy:
    """把前五个 module 串成一次可执行的推理.

    统计量在构造时给定 (训练集上算好的), 推理时只用不更新 —— 见 ../data Sec. 3.
    """

    def __init__(self, backbone: VisionLanguageBackbone, expert: ActionExpert,
                 data_cfg: DataConfig, action_stats: dict[str, dict],
                 state_stats: dict[str, dict], rot_rep: str,
                 tokenizer=None) -> None:
        self.backbone = backbone.eval()
        self.expert = expert.eval()
        self.cfg = data_cfg
        self.action_norm = {k: Normalizer(data_cfg.norm_mode, v)
                            for k, v in action_stats.items()}
        self.state_norm = {k: Normalizer(data_cfg.norm_mode, v)
                           for k, v in state_stats.items()}
        self.rot_rep = rot_rep
        self.tokenizer = tokenizer or (lambda s: _hash_tokenize(s, backbone.cfg.vocab))

    @torch.no_grad()
    def act(self, obs: Observation) -> tuple[torch.Tensor, dict]:
        """返回 (H, D_native) 的**原生单位**动作, 以及逐环节耗时."""
        emb = self.cfg.embodiments[obs.embodiment]
        t = {}

        t0 = time.perf_counter()
        state_n = None
        if emb.has_proprio:
            assert obs.state is not None, f"{emb.name} 必须给 state"
            state_n = self.state_norm[emb.name].forward(obs.state[None])
        sample = build_sample(emb, self.cfg, obs.images, obs.instruction, state_n, None,
                              training=False)
        t["preprocess_ms"] = 1e3 * (time.perf_counter() - t0)

        images = torch.from_numpy(sample["video"]).float().permute(0, 1, 4, 2, 3) / 255.0
        view_mask = torch.from_numpy(sample["view_mask"])[None]
        ids = self.tokenizer(obs.instruction)[None]
        attn = torch.ones_like(ids, dtype=torch.bool)

        t0 = time.perf_counter()
        # phi 每个 chunk 只算一次, K 步去噪全程复用 (../backbone Sec. 3)
        out = self.backbone(images[0][None], view_mask, ids, attn)
        t["backbone_ms"] = 1e3 * (time.perf_counter() - t0)

        t0 = time.perf_counter()
        action_n = self.expert.sample(
            out["backbone_features"], out["backbone_attention_mask"],
            sample["state"][None], sample["state_mask"][None],
            torch.tensor([emb.embodiment_id]), torch.tensor([emb.has_proprio]),
        )[0]
        t["denoise_ms"] = 1e3 * (time.perf_counter() - t0)

        t0 = time.perf_counter()
        # 顺序不能反: 先逆归一化 (统计量按 padding 后的宽度存), 再按 mask 截维.
        mask = torch.zeros(action_n.shape, dtype=torch.bool)
        mask[:, : emb.action_dim] = True
        action = self.action_norm[emb.name].inverse(unpad_action(action_n, mask))
        t["postprocess_ms"] = 1e3 * (time.perf_counter() - t0)

        t["total_ms"] = sum(t.values())
        t["phi_shape"] = tuple(out["backbone_features"].shape)
        t["n_denoise"] = self.expert.cfg.num_inference_timesteps
        return action, t

    def to_robot_command(self, action: torch.Tensor, embodiment: str) -> RobotCommand:
        """论文 Sec. 2.5: 手臂是相对末端位姿增量, 手是目标关节角."""
        r = ROT_DIM[self.rot_rep]
        per_hand = action.shape[-1] // 2
        n_hand = per_hand - (3 + r)
        assert n_hand > 0, f"动作维 {action.shape[-1]} 装不下两只手的 {3 + r} 维腕部"
        halves = action.reshape(action.shape[0], 2, per_hand)
        return RobotCommand(arm_delta_pose=halves[..., : 3 + r],
                            hand_joints=halves[..., 3 + r:])


def _hash_tokenize(text: str, vocab: int) -> torch.Tensor:
    """占位分词器: 真实实现用 Eagle 的 tokenizer, checkpoint 视为给定 (见 ../backbone)."""
    return torch.tensor([abs(hash(w)) % vocab for w in text.lower().split()] or [0])


# ---------------------------------------------------------------------------
# tiny 装配
# ---------------------------------------------------------------------------
def build_tiny_policy(seed: int = 0) -> tuple[EgoScalePolicy, DataConfig]:
    torch.manual_seed(seed)
    bcfg = backbone_tiny()
    dcfg = replace(dit_tiny(), backbone_embedding_dim=bcfg.d_llm)
    data_cfg = data_tiny()
    assert dcfg.max_action_dim == data_cfg.max_action_dim
    assert dcfg.max_state_dim == data_cfg.max_state_dim
    assert bcfg.image_hw == data_cfg.image_hw
    assert dcfg.max_num_embodiments >= len(data_cfg.embodiments)

    backbone = VisionLanguageBackbone(bcfg)
    expert = ActionExpert(dcfg)

    # 统计量本该来自训练集; 这里用一小批合成数据代替, 见 ../data Sec. 8
    a_stats, s_stats = {}, {}
    for name, emb in data_cfg.embodiments.items():
        a_stats[name] = stats_from(torch.randn(64, dcfg.action_horizon, emb.action_dim))
        s_stats[name] = stats_from(
            torch.randn(64, data_cfg.state_horizon, max(emb.state_dim, 1))
        )
    return EgoScalePolicy(backbone, expert, data_cfg, a_stats, s_stats,
                          rot_rep="rotation_6d"), data_cfg


def fake_observation(data_cfg: DataConfig, embodiment: str,
                     rng: np.random.Generator) -> Observation:
    emb = data_cfg.embodiments[embodiment]
    h, w = data_cfg.image_hw
    images = {s: rng.integers(0, 256, (data_cfg.state_horizon, h, w, 3), dtype=np.uint8)
              for s in emb.cameras}
    state = None if not emb.has_proprio else torch.randn(emb.state_dim)
    return Observation(images=images, instruction="unscrew the cap from the bottle",
                       state=state, embodiment=embodiment)


def main() -> None:
    policy, data_cfg = build_tiny_policy()
    rng = np.random.default_rng(0)
    n_par = sum(p.numel() for p in list(policy.backbone.parameters())
                + list(policy.expert.parameters()))
    print(f"tiny end-to-end policy: {n_par:,} parameters "
          f"(structure values are NOT paper values, see README Sec. 8)")
    print(f"camera slots {CAMERA_SLOTS}, rot_rep={policy.rot_rep!r}, "
          f"K={policy.expert.cfg.num_inference_timesteps}, H={policy.expert.cfg.action_horizon}")

    for name in ("r1pro_sharpa", "g1_trifinger", "human_wild"):
        emb = data_cfg.embodiments[name]
        obs = fake_observation(data_cfg, name, rng)
        action, info = policy.act(obs)
        print(f"\n[{name}] id={emb.embodiment_id} cameras={emb.cameras} "
              f"has_proprio={emb.has_proprio}")
        print(f"  [1] images {sorted(obs.images)} -> video "
              f"({data_cfg.state_horizon},{len(CAMERA_SLOTS)},{data_cfg.image_hw[0]},"
              f"{data_cfg.image_hw[1]},3)")
        print(f"  [2] backbone -> phi {info['phi_shape']}  "
              f"(computed once, reused by all {info['n_denoise']} denoising steps)")
        print(f"  [3] denoise {info['n_denoise']} steps -> "
              f"({policy.expert.cfg.action_horizon},{policy.expert.cfg.max_action_dim}) "
              f"normalized")
        print(f"  [4] un-normalize then un-pad -> {tuple(action.shape)} native")
        if name != "human_wild":
            cmd = policy.to_robot_command(action, name)
            print(f"  [5] arm delta pose {tuple(cmd.arm_delta_pose.shape)}  "
                  f"hand joints {tuple(cmd.hand_joints.shape)}")
            print(f"      arm[0,L,:3] = {cmd.arm_delta_pose[0, 0, :3].numpy().round(3)}")
            print(f"      hand[0,L,:4] = {cmd.hand_joints[0, 0, :4].numpy().round(3)}")
        print(f"  latency: " + "  ".join(
            f"{k.replace('_ms', '')} {v:.1f}ms" for k, v in info.items()
            if k.endswith("_ms")))

    print("\nnote: EgoScale discloses no latency and no parameter count. "
          "GR00T N1 reports 63.9 ms end-to-end on an L40 for a 16-step chunk with 4 denoising "
          "steps (arXiv:2503.14734v2 Sec. 2.1) -- that is GR00T's number, not EgoScale's.")


if __name__ == "__main__":
    main()
