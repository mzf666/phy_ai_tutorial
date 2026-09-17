"""对齐检查: rubric 分值逐项对齐附录 B, 两种打分的解析性质, 推理链路的顺序约束, 采样平均.

不复现任何论文得分.
论文: EgoScale arXiv:2602.16710v1 Sec. 2.5, Sec. 3.1, Sec. 3.3, App. B.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear.egoscale.action.action import ROT_DIM
from gear.egoscale.data.data import CAMERA_SLOTS
from gear.egoscale.infer.eval import (
    G1_SUITE,
    N_SAMPLES_PER_TIMESTEP,
    N_SEEDS,
    N_TIMESTEPS_PER_EPISODE,
    N_VALIDATION_EPISODES,
    ONE_SHOT_SUITE,
    POST_TRAINING_SUITE,
    TASKS,
    FakeRobotTask,
    averaged_prediction,
    binary_success,
    human_validation_loss,
    rubric_total,
    rubric_warnings,
    run_episode,
    run_suite,
    run_task,
    score,
    score_additive,
    score_progress,
    validation_budget,
)
from gear.egoscale.infer.model import build_tiny_policy, fake_observation


@pytest.fixture(scope="module")
def policy_and_cfg():
    return build_tiny_policy()


# --------------------------------------------------------------------------
# rubric 的数值对齐 (论文 App. B)
# --------------------------------------------------------------------------
def test_task_set_matches_the_paper():
    assert set(TASKS) == set(POST_TRAINING_SUITE + ONE_SHOT_SUITE + G1_SUITE)
    assert len(POST_TRAINING_SUITE) == 5 and len(ONE_SHOT_SUITE) == 2 and len(G1_SUITE) == 2
    # 论文的编号从 I 跳到 V, 然后是 VIII / IX -- VI 与 VII 不存在. 见 README Sec. 1.x 第 3 条
    numbers = {t.number for t in TASKS.values()}
    assert {"I", "II", "III", "IV", "V", "VIII", "IX"} <= numbers
    assert "VI" not in numbers and "VII" not in numbers


def test_additive_rubrics_that_sum_to_one():
    for key in ("tong", "bottle", "syringe", "pen_in_bin"):
        assert rubric_total(TASKS[key]) == pytest.approx(1.0, abs=1e-9), key


def test_the_three_rubrics_that_do_not_sum_to_one():
    """论文 App. B 的三处: 1.1 / 1.2 / 0.99, 而正文说完成分在 [0,1]. 见 README Sec. 1.x 第 2 条."""
    warn = rubric_warnings()
    assert set(warn) == {"fold_oneshot", "bottle_oneshot", "dish_in_rack"}
    assert warn["fold_oneshot"] == pytest.approx(1.1, abs=1e-9)
    assert warn["bottle_oneshot"] == pytest.approx(1.2, abs=1e-9)
    assert warn["dish_in_rack"] == pytest.approx(0.99, abs=1e-9)


def test_specific_rubric_values():
    """逐项对齐论文 App. B."""
    tong = dict(TASKS["tong"].rubric)
    assert tong["grasps the tongs"] == 0.4
    assert sorted(tong.values()) == [0.2, 0.2, 0.2, 0.4]

    bottle = dict(TASKS["bottle"].rubric)
    assert bottle["grasps the bottle"] == 0.1
    assert bottle["unscrews with at least three continuous rotations"] == 0.5

    syr = [v for _, v in TASKS["syringe"].rubric]
    assert syr == [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1]

    shirt = [v for _, v in TASKS["shirt"].rubric]
    assert shirt == [0.0, 0.3, 0.5, 0.8, 1.0]

    card = [v for _, v in TASKS["card"].rubric]
    assert card == [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    assert all(v == 0.25 for _, v in TASKS["pen_in_bin"].rubric)
    assert all(v == 0.11 for _, v in TASKS["dish_in_rack"].rubric)
    assert len(TASKS["dish_in_rack"].rubric) == 9  # 3 盘 x 3 里程碑


def test_trial_counts():
    """论文 Sec. 3.1: 每个 checkpoint 10 次试验, Bottle 是 4 个瓶子 x 4 = 16."""
    for key in ("shirt", "card", "tong", "syringe", "fold_oneshot", "bottle_oneshot",
                "pen_in_bin", "dish_in_rack"):
        assert TASKS[key].trials == 10, key
    assert TASKS["bottle"].trials == 16
    assert "Sec. 3.1 says 16, App. B says 12" in TASKS["bottle"].note
    assert N_SEEDS == 2


def test_instructions_are_the_paper_strings():
    assert TASKS["shirt"].instruction == "Roll the T-shirt and put it into the basket."
    assert TASKS["bottle"].instruction == "Unscrew the cap from the bottle."
    assert TASKS["bottle_oneshot"].instruction == "Unscrew the cap from the water bottle."
    assert TASKS["pen_in_bin"].instruction == "Marker canister task."
    assert TASKS["dish_in_rack"].instruction == "Put plates on dishrack."


def test_kinds():
    """App. B: 变形体或阶段耦合紧的用 progress, 可分解成独立子技能的用 additive."""
    assert TASKS["shirt"].kind == "progress" and TASKS["card"].kind == "progress"
    for key in ("tong", "bottle", "syringe", "fold_oneshot", "bottle_oneshot",
                "pen_in_bin", "dish_in_rack"):
        assert TASKS[key].kind == "additive", key


# --------------------------------------------------------------------------
# 打分的解析性质
# --------------------------------------------------------------------------
def test_additive_is_additive():
    t = TASKS["syringe"]
    names = [k for k, _ in t.rubric]
    a, b = set(names[:3]), set(names[3:])
    assert score_additive(t, a | b) == pytest.approx(
        score_additive(t, a) + score_additive(t, b), abs=1e-9)


def test_progress_is_not_additive_and_takes_the_furthest_milestone():
    t = TASKS["shirt"]
    names = [k for k, _ in t.rubric]
    early, late = {names[1]}, {names[3]}
    assert score_progress(t, early) == 0.3
    assert score_progress(t, late) == 0.8
    # 取最远而不是求和
    assert score_progress(t, early | late) == 0.8
    assert score_progress(t, early | late) != score_progress(t, early) + score_progress(t, late)
    assert score_progress(t, set()) == 0.0


def test_scores_are_clipped_to_unit_interval():
    """三处 rubric 合计 > 1, 截断是本仓库的选择. 见 README Sec. 1.x 第 2 条."""
    t = TASKS["bottle_oneshot"]
    all_done = {k for k, _ in t.rubric}
    assert rubric_total(t) > 1.0
    assert score_additive(t, all_done) == 1.0
    for task in TASKS.values():
        s = score(task, {k for k, _ in task.rubric})
        assert 0.0 <= s <= 1.0, task.key


def test_binary_success_requires_every_scoring_milestone():
    t = TASKS["tong"]
    names = [k for k, v in t.rubric if v > 0]
    assert binary_success(t, set(names)) == 1
    assert binary_success(t, set(names[:-1])) == 0
    assert binary_success(t, set()) == 0
    # 0 分的里程碑 ("no folding") 不该被算成必须达成的
    shirt = TASKS["shirt"]
    assert binary_success(shirt, {k for k, v in shirt.rubric if v > 0}) == 1


# --------------------------------------------------------------------------
# 验证协议 (论文 Sec. 3.3)
# --------------------------------------------------------------------------
def test_validation_protocol_constants():
    assert (N_VALIDATION_EPISODES, N_TIMESTEPS_PER_EPISODE, N_SAMPLES_PER_TIMESTEP) == \
        (2_000, 20, 16)
    b = validation_budget()
    assert b["forwards_per_episode"] == 320
    assert b["total_forwards"] == 640_000


def test_space_must_be_given_explicitly():
    """论文没说验证损失算在归一化空间还是原生空间. 见 README Sec. 8."""
    gt = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        human_validation_loss(lambda: torch.randn(4, 3), gt, space="whatever")
    human_validation_loss(lambda: torch.randn(4, 3), gt, space="normalized", n_samples=2)


def test_averaging_order_matters():
    """先平均预测再算误差 != 先算 16 个误差再平均. 前者压方差, 后者不压 (Jensen)."""
    g = torch.Generator().manual_seed(0)
    gt = torch.zeros(6, 4)

    def sampler():
        return torch.randn(6, 4, generator=g)

    torch.manual_seed(0)
    g.manual_seed(0)
    avg_then_err = human_validation_loss(sampler, gt, space="normalized", n_samples=16)
    g.manual_seed(0)
    err_then_avg = torch.stack(
        [torch.nn.functional.mse_loss(sampler(), gt) for _ in range(16)]
    ).mean()
    assert float(avg_then_err) < float(err_then_avg)
    # 对零均值噪声, 平均 n 个样本把方差压到 1/n
    assert float(avg_then_err) == pytest.approx(float(err_then_avg) / 16, rel=0.5)


def test_averaged_prediction_shape_and_variance():
    g = torch.Generator().manual_seed(1)
    spread = []
    for n in (1, 16):
        vals = torch.stack([averaged_prediction(
            lambda: torch.randn(3, 5, generator=g), n_samples=n) for _ in range(64)])
        assert vals.shape == (64, 3, 5)
        spread.append(float(vals.std()))
    assert spread[1] < spread[0] / 2, "16 样本平均必须明显压低方差"


# --------------------------------------------------------------------------
# 推理链路
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["r1pro_sharpa", "g1_trifinger", "human_wild"])
def test_act_returns_native_dimensions(policy_and_cfg, name):
    policy, cfg = policy_and_cfg
    emb = cfg.embodiments[name]
    obs = fake_observation(cfg, name, np.random.default_rng(0))
    action, info = policy.act(obs)
    assert action.shape == (policy.expert.cfg.action_horizon, emb.action_dim)
    assert info["n_denoise"] == policy.expert.cfg.num_inference_timesteps
    assert info["total_ms"] > 0


def test_video_always_has_all_camera_slots(policy_and_cfg):
    """human_wild 只有头部相机, 但 video 的槽位数必须固定. 见 ../data."""
    policy, cfg = policy_and_cfg
    obs = fake_observation(cfg, "human_wild", np.random.default_rng(1))
    assert set(obs.images) == {"head"}
    action, info = policy.act(obs)
    # 序列长度里的图像部分 = V * tokens_per_view, 与相机是否真实无关
    n_img = len(CAMERA_SLOTS) * policy.backbone.tokens_per_view
    assert info["phi_shape"][1] > n_img  # 还要加上文本 token


def test_robot_command_split(policy_and_cfg):
    """论文 Sec. 2.5: 手臂是相对末端位姿增量, 手是目标关节角."""
    policy, cfg = policy_and_cfg
    r = ROT_DIM[policy.rot_rep]
    for name, n_hand in (("r1pro_sharpa", 22), ("g1_trifinger", 7)):
        emb = cfg.embodiments[name]
        action = torch.randn(policy.expert.cfg.action_horizon, emb.action_dim)
        cmd = policy.to_robot_command(action, name)
        assert cmd.arm_delta_pose.shape[-1] == 3 + r
        assert cmd.hand_joints.shape[-1] == n_hand
        assert cmd.arm_delta_pose.shape[1] == cmd.hand_joints.shape[1] == 2  # 双手
        # 两段拼回去必须还原原动作
        back = torch.cat((cmd.arm_delta_pose, cmd.hand_joints), dim=-1)
        assert torch.allclose(back.reshape(action.shape), action)


def test_missing_state_for_a_robot_is_rejected(policy_and_cfg):
    policy, cfg = policy_and_cfg
    obs = fake_observation(cfg, "r1pro_sharpa", np.random.default_rng(2))
    obs.state = None
    with pytest.raises(AssertionError):
        policy.act(obs)


# --------------------------------------------------------------------------
# episode 循环
# --------------------------------------------------------------------------
def test_episode_loop_touches_every_step(policy_and_cfg):
    policy, cfg = policy_and_cfg
    task = TASKS["tong"]
    env = FakeRobotTask(task, cfg, np.random.default_rng(3), max_chunks=3)
    r = run_episode(policy, env, task, trial=0)
    assert r["n_infer"] == 3, "每个 chunk 推理一次"
    assert 0.0 <= r["score"] <= 1.0
    assert r["success"] in (0, 1)
    assert r["achieved"] <= {k for k, _ in task.rubric}


def test_execute_steps_shortens_the_executed_chunk(policy_and_cfg):
    """开环执行长度是显式参数: EgoScale 未披露. 见 README Sec. 8."""
    policy, cfg = policy_and_cfg
    task = TASKS["pen_in_bin"]

    class Recorder(FakeRobotTask):
        lengths: list = []

        def step(self, action_chunk):
            Recorder.lengths.append(action_chunk.shape[0])
            return super().step(action_chunk)

    Recorder.lengths = []
    env = Recorder(task, cfg, np.random.default_rng(4), max_chunks=2)
    run_episode(policy, env, task, trial=0, execute_steps=3)
    assert Recorder.lengths == [3, 3]

    Recorder.lengths = []
    env = Recorder(task, cfg, np.random.default_rng(4), max_chunks=2)
    run_episode(policy, env, task, trial=0)
    assert Recorder.lengths == [policy.expert.cfg.action_horizon] * 2


def test_aggregation_over_seeds_and_trials(policy_and_cfg):
    """论文 Sec. 3.1: 两个种子, 每个种子若干次试验, 报两个种子的平均."""
    policy, cfg = policy_and_cfg
    res = run_task(policy, TASKS["pen_in_bin"], cfg, trials=2)
    assert len(res["per_seed"]) == N_SEEDS
    assert res["score"] == pytest.approx(
        float(np.mean([s["score"] for s in res["per_seed"]])), abs=1e-9)
    assert all(s["trials"] == 2 for s in res["per_seed"])
    assert 0.0 <= res["score"] <= 1.0 and 0.0 <= res["success"] <= 1.0


def test_suite_average(policy_and_cfg):
    policy, cfg = policy_and_cfg
    res = run_suite(policy, G1_SUITE, cfg, trials=1)
    assert set(res["tasks"]) == set(G1_SUITE)
    assert res["average_score"] == pytest.approx(
        float(np.mean([r["score"] for r in res["tasks"].values()])), abs=1e-9)


def test_fake_env_has_no_task_semantics(policy_and_cfg):
    """假环境只用来跑通循环; 不同种子必须给出不同结果, 说明它是随机的而不是在"解任务"."""
    policy, cfg = policy_and_cfg
    task = TASKS["syringe"]
    outs = []
    for seed in (0, 1, 2, 3):
        env = FakeRobotTask(task, cfg, np.random.default_rng(seed), max_chunks=4)
        outs.append(run_episode(policy, env, task, trial=0)["score"])
    assert len(set(outs)) > 1, "分数应随种子变化, 它没有任何可解读性"
