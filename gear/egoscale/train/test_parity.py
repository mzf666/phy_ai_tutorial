"""对齐检查: 阶段超参逐值对齐论文 Sec. 2.4, 冻结集合, scaling law 的拟合与单位冲突.

不复现任何训练曲线, 不承诺收敛.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.4, Sec. 3.2, Sec. 3.3, App. D.1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear.egoscale.train.train import (
    CHECKPOINTS,
    FIG5_CENTER,
    FIG5_RIGHT,
    PAPER_INTERCEPT,
    PAPER_R2,
    PAPER_SLOPE,
    STAGE1,
    STAGE2,
    STAGES,
    ScalingLaw,
    _tiny_batch,
    _tiny_models,
    apply_stage,
    groot_n1_optimizer,
    optimizer_config,
    paper_law_in_hours,
    run_stage,
    stage3,
)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    return _tiny_models()


# --------------------------------------------------------------------------
# 数值对齐 (论文 Sec. 2.4)
# --------------------------------------------------------------------------
def test_stage_hyperparameters_match_the_paper():
    """论文 Sec. 2.4: 100K/8192/5e-5, 50K/2048/3e-5, 10K/512/3e-5."""
    assert (STAGE1.steps, STAGE1.batch_size, STAGE1.lr) == (100_000, 8_192, 5e-5)
    assert (STAGE2.steps, STAGE2.batch_size, STAGE2.lr) == (50_000, 2_048, 3e-5)
    for mid in (True, False):
        s = stage3(mid)
        assert (s.steps, s.batch_size, s.lr) == (10_000, 512, 3e-5)


def test_stage_freeze_flags_match_the_paper():
    """Sec. 2.4 + App. D.1. 见 README Sec. 1.x 第 3 条."""
    assert (STAGE1.tune_llm, STAGE1.tune_visual, STAGE1.tune_dit, STAGE1.tune_projector) == \
        (True, True, True, True)  # "fully unfreezing every parameter"
    assert STAGE2.tune_llm is False and STAGE2.tune_visual is True
    assert STAGE2.tune_dit is True and STAGE2.tune_projector is True
    # "the vision encoder is frozen if mid-training is used and unfrozen otherwise"
    assert stage3(mid_trained=True).tune_visual is False
    assert stage3(mid_trained=False).tune_visual is True
    assert stage3(True).tune_llm is False and stage3(False).tune_llm is False


def test_fig5_right_scores_match_the_paper():
    """论文图 5 右栏柱子上印的数字."""
    assert FIG5_RIGHT == {1: 0.30, 2: 0.45, 4: 0.48, 10: 0.57, 20: 0.71}
    ds = sorted(FIG5_RIGHT)
    scores = [FIG5_RIGHT[d] for d in ds]
    assert scores == sorted(scores), "论文 Sec. 3.3: 单调上升, 未见饱和"


def test_checkpoint_paths_match_the_ablation():
    """论文 Sec. 3.2 比较的四个 checkpoint."""
    assert set(CHECKPOINTS) == {"no_pretrain", "midtrain_only", "human_pretrain",
                                "human_pretrain_midtrain"}
    assert CHECKPOINTS["no_pretrain"] == (stage3(False),)
    assert CHECKPOINTS["human_pretrain_midtrain"] == (STAGE1, STAGE2, stage3(True))
    assert len(CHECKPOINTS["human_pretrain_midtrain"]) == 3


def test_optimizer_is_undisclosed_but_groot_is_not():
    """EgoScale 只给了 lr 与 batch size, 见 README Sec. 8."""
    ego = optimizer_config()
    assert all(v is None for v in ego.values())
    g = groot_n1_optimizer()
    assert (g["name"], g["beta1"], g["beta2"], g["eps"]) == ("AdamW", 0.95, 0.999, 1e-8)
    assert (g["weight_decay"], g["lr_scheduler"], g["warmup_ratio"]) == (1e-5, "cosine", 0.05)
    assert set(ego) == set(g), "两张表的字段必须一一对应, 才看得出缺了什么"


# --------------------------------------------------------------------------
# 冻结
# --------------------------------------------------------------------------
def test_stage1_trains_everything(models):
    _, _, backbone, expert = models
    t = apply_stage(backbone, expert, STAGE1)
    assert all(v > 0 for v in t.values()), t


def test_stage2_freezes_only_the_llm(models):
    _, _, backbone, expert = models
    t = apply_stage(backbone, expert, STAGE2)
    assert t["backbone.llm"] == 0
    for k in ("backbone.vision", "backbone.connector", "expert.dit",
              "expert.state_encoder", "expert.action_encoder", "expert.action_decoder"):
        assert t[k] > 0, k


def test_stage3_with_midtraining_also_freezes_the_vision_tower(models):
    _, _, backbone, expert = models
    t = apply_stage(backbone, expert, stage3(mid_trained=True))
    assert t["backbone.llm"] == 0 and t["backbone.vision"] == 0
    assert t["backbone.connector"] == 0, "上游把 connector (mlp1) 归在 visual 一侧"
    assert t["expert.dit"] > 0 and t["expert.action_decoder"] > 0

    t2 = apply_stage(backbone, expert, stage3(mid_trained=False))
    assert t2["backbone.vision"] > 0, "没有 mid-training 时视觉编码器解冻"


def test_vl_post_process_is_never_frozen(models):
    """上游的冻结开关从不涉及 vlln 与 vl_self_attention. 见 README Sec. 1.x 第 5 条."""
    _, _, backbone, expert = models
    for st in (STAGE1, STAGE2, stage3(True), stage3(False)):
        t = apply_stage(backbone, expert, st)
        assert t["backbone.post"] > 0, st.name


def test_frozen_modules_are_switched_to_eval(models):
    _, _, backbone, expert = models
    backbone.train()
    expert.train()
    apply_stage(backbone, expert, stage3(mid_trained=True))
    assert not backbone.llm.training and not backbone.vision.training
    assert expert.dit.training, "DiT 在三个阶段里都是解冻的, 应保持 train 模式"


def test_trainable_counts_are_monotone_across_stages(models):
    """冻结集合是逐阶段扩大的, 所以可训练参数量单调不增."""
    _, _, backbone, expert = models
    counts = [sum(apply_stage(backbone, expert, st).values()) for st in STAGES]
    assert counts[0] > counts[1] > counts[2]


# --------------------------------------------------------------------------
# scaling law
# --------------------------------------------------------------------------
def test_paper_law_constants():
    law = ScalingLaw.paper()
    assert (law.intercept, law.slope, law.r2) == (PAPER_INTERCEPT, PAPER_SLOPE, PAPER_R2)
    assert (PAPER_INTERCEPT, PAPER_SLOPE, PAPER_R2) == (0.024, 0.003, 0.9983)


def test_fit_recovers_a_synthetic_law_exactly():
    d = np.array([1.0, 2.0, 4.0, 10.0, 20.0])
    y = 0.031 - 0.0042 * np.log(d)
    law = ScalingLaw.fit(d, y)
    assert law.intercept == pytest.approx(0.031, abs=1e-9)
    assert law.slope == pytest.approx(0.0042, abs=1e-9)
    assert law.r2 == pytest.approx(1.0, abs=1e-12)


def test_predict_and_hours_for_are_inverses():
    law = ScalingLaw.paper()
    for d in (1.0, 5.0, 20.854, 100.0):
        assert law.hours_for(float(law.predict(d))) == pytest.approx(d, rel=1e-9)


def test_paper_law_units():
    """论文说 D 的单位是小时, 但按小时代入会得到负损失. 见 README Sec. 1.x 第 1 条."""
    assert float(paper_law_in_hours(20854)) < 0.0
    assert float(paper_law_in_hours(4000)) < 0.0  # 4k 小时就已经为负了

    law = ScalingLaw.paper()
    for d_k, observed in FIG5_CENTER.items():
        assert abs(float(law.predict(d_k)) - observed) < 0.0015, d_k
        assert float(law.predict(d_k)) > 0


def test_fitted_slope_is_rounded_in_the_paper():
    """读图四个点拟合出的斜率约 0.0034, 论文印的 0.003 是一位有效数字. 见 README Sec. 1.x 第 2 条."""
    law = ScalingLaw.fit(list(FIG5_CENTER), list(FIG5_CENTER.values()))
    assert law.r2 > 0.99, "读图点本身就接近一条直线"
    assert 0.0030 < law.slope < 0.0040
    assert round(law.slope, 3) == PAPER_SLOPE
    assert abs(law.intercept - PAPER_INTERCEPT) < 0.001


def test_read_off_points_are_flagged_as_such():
    """FIG5_CENTER 只有 4 个点: 1k 那个点在图里被裁掉, 不收录. 见 README Sec. 8."""
    assert set(FIG5_CENTER) == {2, 4, 10, 20}
    assert 1 not in FIG5_CENTER
    assert set(FIG5_RIGHT) == {1, 2, 4, 10, 20}  # 右栏是论文印出来的, 五个都有


def test_extrapolation_is_monotone_decreasing():
    law = ScalingLaw.paper()
    d = np.array([20.0, 40.0, 100.0, 200.0])
    pred = law.predict(d)
    assert np.all(np.diff(pred) < 0)
    assert np.all(pred > 0), "在这个外推区间里损失还没变负"


# --------------------------------------------------------------------------
# 端到端: 梯度只出现在解冻的模块上
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stage_idx", [0, 1, 2])
def test_gradients_only_reach_unfrozen_modules(stage_idx):
    torch.manual_seed(1)
    bcfg, dcfg, backbone, expert = _tiny_models()
    backbone.train()
    expert.train()
    stage = STAGES[stage_idx]
    r = run_stage(backbone, expert, stage, bcfg, dcfg, n_steps=1)
    frozen = {k for k, v in r["trainable"].items() if v == 0}
    assert r["got_grad"].isdisjoint(frozen), (stage.name, r["got_grad"], frozen)
    assert r["got_grad"], "至少要有模块拿到梯度"
    assert len(r["losses"]) == 1 and np.isfinite(r["losses"][0])


def test_the_three_stages_share_one_objective():
    """三个阶段的 loss 完全一样, 变的只有数据、lr、batch 与冻结集合 (与 pi0.5 不同)."""
    for st in STAGES:
        assert st.tune_dit is True, "DiT 在任何阶段都不冻结, 否则目标就无从优化"
    assert len({(s.lr, s.batch_size) for s in STAGES}) == 3, "三阶段的 lr/batch 组合两两不同"
