"""三阶段 curriculum 的阶段表、冻结开关与 scaling law 的拟合 / 外推.

上游: EgoScale 未开源, 训练脚本也没有对应的开源实现 (上游 gr00t/experiment/trainer.py 只是
      HuggingFace Trainer 的薄封装). 冻结开关对照 NVIDIA/Isaac-GR00T @
      4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 的
      gr00t/model/backbone/eagle_backbone.py L65-L94 与
      gr00t/model/action_head/flow_matching_action_head.py L217-L254.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.4, Sec. 3.2, Sec. 3.3, App. D.1;
      GR00T N1 arXiv:2503.14734v2 Table 6 (仅作为对照, 不是 EgoScale 的值).
许可: 上游 Isaac-GR00T 为 Apache-2.0; 本文件为 PyTorch 重写 (re-implements, does not copy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from gear.egoscale.backbone.model import VisionLanguageBackbone
from gear.egoscale.backbone.model import tiny as backbone_tiny
from gear.egoscale.dit.model import ActionExpert
from gear.egoscale.dit.model import tiny as dit_tiny
from gear.egoscale.dit.train import flow_matching_loss


# ---------------------------------------------------------------------------
# 1. 阶段表 (论文 Sec. 2.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    name: str
    steps: int
    batch_size: int
    lr: float
    tune_llm: bool
    tune_visual: bool
    tune_dit: bool
    tune_projector: bool
    data: str
    mixture: tuple[float, ...] | None = None  # 未披露, 见 README Sec. 8


# 论文 Sec. 2.4 的原话: "train on 20K hours of egocentric human data for 100K steps with 256
# GB200 GPUs using a global batch size of 8,192 and learning rate 5e-5, fully unfreezing every
# parameter of the VLA model."
STAGE1 = Stage(
    name="I_human_pretrain", steps=100_000, batch_size=8_192, lr=5e-5,
    tune_llm=True, tune_visual=True, tune_dit=True, tune_projector=True,
    data="20,854 h egocentric human video",
)

# "train on the aligned human-robot play dataset for 50K steps with batch size 2,048 and learning
# rate 3e-5, freezing the vision-language backbone while only updating the vision encoder and DiT
# action expert" (Sec. 2.4) + App. D.1: "only the vision encoder, DiT action expert, and
# state-action encoder and decoder are updated". 见 README Sec. 1.x 第 3 条.
STAGE2 = Stage(
    name="II_aligned_midtrain", steps=50_000, batch_size=2_048, lr=3e-5,
    tune_llm=False, tune_visual=True, tune_dit=True, tune_projector=True,
    data="~50 h aligned human + ~4 h robot play",
)


def stage3(mid_trained: bool) -> Stage:
    """"the vision encoder is frozen if mid-training is used and unfrozen otherwise" (Sec. 2.4)."""
    return Stage(
        name=f"III_post_train({'mid' if mid_trained else 'no_mid'})",
        steps=10_000, batch_size=512, lr=3e-5,
        tune_llm=False, tune_visual=not mid_trained, tune_dit=True, tune_projector=True,
        data="task demos, 100/task (Shirt 20, Bottle 4x25)",
    )


STAGES = (STAGE1, STAGE2, stage3(mid_trained=True))

# 论文 Sec. 3.2 比较的四个 checkpoint.
CHECKPOINTS = {
    "no_pretrain": (stage3(mid_trained=False),),
    "midtrain_only": (STAGE2, stage3(mid_trained=True)),
    "human_pretrain": (STAGE1, stage3(mid_trained=False)),
    "human_pretrain_midtrain": (STAGE1, STAGE2, stage3(mid_trained=True)),
}


def optimizer_config() -> dict[str, object]:
    """EgoScale 只披露了 lr 与 batch size; 其余一概未披露, 见 README Sec. 8."""
    return {
        "name": None, "beta1": None, "beta2": None, "eps": None, "weight_decay": None,
        "lr_scheduler": None, "warmup_ratio": None, "grad_clip": None, "ema": None,
        "precision": None,
    }


def groot_n1_optimizer() -> dict[str, object]:
    """GR00T N1 arXiv:2503.14734v2 表 6. **这是 GR00T 的值, 不是 EgoScale 的.**"""
    return {
        "name": "AdamW", "beta1": 0.95, "beta2": 0.999, "eps": 1e-8, "weight_decay": 1e-5,
        "lr_scheduler": "cosine", "warmup_ratio": 0.05, "grad_clip": None, "ema": None,
        "precision": None,
    }


# ---------------------------------------------------------------------------
# 2. 冻结
# ---------------------------------------------------------------------------
def apply_stage(backbone: VisionLanguageBackbone, expert: ActionExpert,
                stage: Stage) -> dict[str, int]:
    """按阶段设置 requires_grad 并把冻结的子模块切到 eval, 返回逐模块的可训练参数量."""
    backbone.set_trainable_parameters(tune_llm=stage.tune_llm, tune_visual=stage.tune_visual)
    expert.set_trainable_parameters(tune_projector=stage.tune_projector,
                                    tune_diffusion_model=stage.tune_dit)
    backbone.set_frozen_modules_to_eval_mode()
    expert.set_frozen_modules_to_eval_mode()

    def n(mod) -> int:
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    # 注意 backbone.post (vlln + VL self-attention) 不在任何一个开关的管辖范围内:
    # 上游 flow_matching_action_head.py L217-L238 只冻 state/action 编解码器与 DiT, 从不冻
    # vlln 与 vl_self_attention, 所以它们在三个阶段里**始终可训练**. 见 README Sec. 1.x.
    return {
        "backbone.vision": n(backbone.vision),
        "backbone.connector": n(backbone.connector),
        "backbone.llm": n(backbone.llm),
        "backbone.post": n(backbone.post),
        "expert.dit": n(expert.dit),
        "expert.state_encoder": n(expert.state_encoder),
        "expert.action_encoder": n(expert.action_encoder),
        "expert.action_decoder": n(expert.action_decoder),
    }


# ---------------------------------------------------------------------------
# 3. scaling law (论文 Sec. 3.3 式 (1))
# ---------------------------------------------------------------------------
PAPER_INTERCEPT = 0.024
PAPER_SLOPE = 0.003
PAPER_R2 = 0.9983

# 论文图 5 右栏柱子上印的平均任务完成分 (论文直接给出的数字).
FIG5_RIGHT = {1: 0.30, 2: 0.45, 4: 0.48, 10: 0.57, 20: 0.71}

# 论文图 5 中栏的验证损失. 论文没有列表, 这四个点是**从图上读的**, 精度约 +-0.0003;
# 1k 那个点在图里被裁掉, 不收录. 见 README Sec. 8.
FIG5_CENTER = {2: 0.0216, 4: 0.0194, 10: 0.0165, 20: 0.0138}


@dataclass(frozen=True)
class ScalingLaw:
    """L(D) = intercept - slope * ln(D), D 的单位是**千小时** (见 README Sec. 1.x 第 1 条)."""

    intercept: float
    slope: float
    r2: float | None = None

    @classmethod
    def paper(cls) -> "ScalingLaw":
        return cls(PAPER_INTERCEPT, PAPER_SLOPE, PAPER_R2)

    @classmethod
    def fit(cls, d_khours, losses) -> "ScalingLaw":
        """在 ln D 上做最小二乘, 同时给出 R^2."""
        x = np.log(np.asarray(d_khours, dtype=float))
        y = np.asarray(losses, dtype=float)
        slope_signed, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope_signed * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return cls(float(intercept), float(-slope_signed), r2)

    def predict(self, d_khours):
        return self.intercept - self.slope * np.log(np.asarray(d_khours, dtype=float))

    def hours_for(self, target_loss: float) -> float:
        """反解: 要把验证损失压到 target_loss 需要多少千小时."""
        return float(math.exp((self.intercept - target_loss) / self.slope))


def paper_law_in_hours(d_hours):
    """把论文式按字面 (D 以小时计) 求值. 存在只是为了展示它给出负损失, 见 README Sec. 1.x."""
    return PAPER_INTERCEPT - PAPER_SLOPE * np.log(np.asarray(d_hours, dtype=float))


# ---------------------------------------------------------------------------
# 4. 一次 tiny 的三阶段循环
# ---------------------------------------------------------------------------
def _tiny_models():
    """tiny backbone 与 tiny expert; expert 的 cross_attention_dim 必须等于 backbone 的宽度."""
    bcfg = backbone_tiny()
    dcfg = replace(dit_tiny(), backbone_embedding_dim=bcfg.d_llm)
    return bcfg, dcfg, VisionLanguageBackbone(bcfg), ActionExpert(dcfg)


def _tiny_batch(bcfg, dcfg, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(b, bcfg.n_views, 3, *bcfg.image_hw, generator=g)
    view_mask = torch.ones(b, bcfg.n_views, dtype=torch.bool)
    ids = torch.randint(0, bcfg.vocab, (b, 6), generator=g)
    attn = torch.ones(b, 6, dtype=torch.bool)
    state = torch.randn(b, dcfg.state_horizon, dcfg.max_state_dim, generator=g)
    action = torch.randn(b, dcfg.action_horizon, dcfg.max_action_dim, generator=g)
    action_mask = torch.zeros_like(action, dtype=torch.bool)
    action_mask[..., :32] = True
    emb = torch.tensor([0, 2])[:b]
    hp = torch.tensor([False, True])[:b]
    return images, view_mask, ids, attn, state, action, action_mask, emb, hp


def run_stage(backbone, expert, stage: Stage, bcfg, dcfg, n_steps: int = 2,
              seed: int = 0) -> dict:
    """跑 n_steps 步, 返回可训练参数量与实际拿到非零梯度的模块集合."""
    trainable = apply_stage(backbone, expert, stage)
    # 清掉上一阶段留下的 .grad: 冻结的参数不会被 opt.zero_grad() 碰到, 不清就会看起来"还在更新"
    for p in list(backbone.parameters()) + list(expert.parameters()):
        p.grad = None
    params = [p for p in list(backbone.parameters()) + list(expert.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=stage.lr)  # 优化器未披露, 见 README Sec. 8
    losses = []
    for step in range(n_steps):
        images, vm, ids, attn, state, action, amask, emb, hp = _tiny_batch(
            bcfg, dcfg, seed=seed + step
        )
        phi_out = backbone(images, vm, ids, attn)
        out = flow_matching_loss(expert, phi_out["backbone_features"],
                                 phi_out["backbone_attention_mask"], state, action, amask,
                                 emb, hp, generator=torch.Generator().manual_seed(seed + step))
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())

    got_grad = {
        name for name, mod in (("backbone.vision", backbone.vision),
                               ("backbone.connector", backbone.connector),
                               ("backbone.llm", backbone.llm),
                               ("backbone.post", backbone.post),
                               ("expert.dit", expert.dit),
                               ("expert.state_encoder", expert.state_encoder),
                               ("expert.action_encoder", expert.action_encoder),
                               ("expert.action_decoder", expert.action_decoder))
        if any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    }
    return {"trainable": trainable, "losses": losses, "got_grad": got_grad}


def main() -> None:
    torch.manual_seed(0)
    bcfg, dcfg, backbone, expert = _tiny_models()
    total = sum(p.numel() for p in list(backbone.parameters()) + list(expert.parameters()))
    print(f"tiny models: backbone + expert = {total:,} parameters "
          f"(structure values are NOT paper values)")

    print("\nstage table (every number below is from paper Sec. 2.4):")
    print(f"  {'stage':28s} {'steps':>8s} {'batch':>7s} {'lr':>8s}  "
          f"{'llm':>5s} {'vis':>5s} {'dit':>5s} {'proj':>5s}")
    for st in (STAGE1, STAGE2, stage3(True), stage3(False)):
        print(f"  {st.name:28s} {st.steps:>8,} {st.batch_size:>7,} {st.lr:>8.0e}  "
              f"{str(st.tune_llm):>5s} {str(st.tune_visual):>5s} {str(st.tune_dit):>5s} "
              f"{str(st.tune_projector):>5s}")

    backbone.train()
    expert.train()
    print("\nrunning 2 tiny steps per stage:")
    for st in STAGES:
        r = run_stage(backbone, expert, st, bcfg, dcfg, n_steps=2)
        tr = sum(r["trainable"].values())
        frozen = sorted(k for k, v in r["trainable"].items() if v == 0)
        print(f"\n  [{st.name}]  lr={st.lr:.0e}  batch(paper)={st.batch_size:,}  "
              f"steps(paper)={st.steps:,}")
        print(f"    trainable {tr:,} / {total:,} ({tr / total:.1%})")
        print(f"    frozen modules: {frozen if frozen else 'none'}")
        print(f"    modules that received gradient: {sorted(r['got_grad'])}")
        print(f"    tiny losses: {[round(x, 4) for x in r['losses']]}")

    print("\noptimizer:")
    print(f"  EgoScale discloses: lr and batch size only -> {optimizer_config()}")
    print(f"  GR00T N1 Table 6 (NOT EgoScale's):          -> {groot_n1_optimizer()}")

    law = ScalingLaw.paper()
    fitted = ScalingLaw.fit(list(FIG5_CENTER), list(FIG5_CENTER.values()))
    print(f"\nscaling law (paper Sec. 3.3 Eq. (1)): "
          f"L = {law.intercept} - {law.slope} * ln(D),  R^2 = {law.r2}")
    print(f"  fitted on the four read-off points:  L = {fitted.intercept:.5f} - "
          f"{fitted.slope:.5f} * ln(D),  R^2 = {fitted.r2:.5f}")
    print(f"  {'D (k hours)':>12s} {'read off':>10s} {'paper law':>10s} "
          f"{'fitted':>10s} {'paper (D in HOURS)':>20s}")
    for d, obs in FIG5_CENTER.items():
        print(f"  {d:>12d} {obs:>10.4f} {float(law.predict(d)):>10.4f} "
              f"{float(fitted.predict(d)):>10.4f} "
              f"{float(paper_law_in_hours(d * 1000)):>20.4f}")
    neg = float(paper_law_in_hours(20854))
    print(f"  taking D literally as hours gives L(20,854 h) = {neg:.4f} < 0 -> impossible, "
          f"so D must be in thousands of hours (README Sec. 1.x row 1)")

    print(f"\n  extrapolation with the paper law:")
    for d in (40, 100, 200):
        print(f"    {d:>4d}k hours -> L = {float(law.predict(d)):.4f}")
    print(f"    to reach L = 0.010 you would need {law.hours_for(0.010):,.0f}k hours "
          f"({law.hours_for(0.010) / 20.854:.1f}x the current dataset)")

    print(f"\n  downstream (paper Fig. 5 right, printed on the bars): "
          f"{FIG5_RIGHT}")
    print(f"  offline loss and real-robot score move together across these 5 points, but that is "
          f"a correlation over 5 samples, not a causal claim")


if __name__ == "__main__":
    main()
