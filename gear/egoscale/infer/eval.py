"""两套评测: 离线的人类验证损失协议, 与真机的任务完成分 rubric.

上游: EgoScale 未开源; 评测协议全部来自论文.
论文: EgoScale arXiv:2602.16710v1 Sec. 3.1, Sec. 3.3, Sec. 3.5, App. B.
许可: 本文件为按论文实现 (re-implements, does not copy).

不复现任何论文数字. rubric 的分值、指令、试验数与聚合方式逐项对齐附录 B; 真机部分用一个
与真接口签名一致的假环境跑通循环.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from gear.egoscale.infer.model import EgoScalePolicy, build_tiny_policy, fake_observation

# ---------------------------------------------------------------------------
# 1. 离线: 人类验证损失 (论文 Sec. 3.3)
# ---------------------------------------------------------------------------
N_VALIDATION_EPISODES = 2_000  # 论文 Sec. 3.3
N_TIMESTEPS_PER_EPISODE = 20  # 论文 Sec. 3.3
N_SAMPLES_PER_TIMESTEP = 16  # 论文 Sec. 3.3: 采 16 个样本, 对动作 chunk 取平均


def averaged_prediction(sample_fn, n_samples: int = N_SAMPLES_PER_TIMESTEP) -> torch.Tensor:
    """采 n 个样本后对 chunk 取**平均**, 再交给上层算一次误差.

    顺序不能反: 先平均预测再算误差 (压方差), 而不是先算 n 个误差再平均 (不压).
    见 test_parity.py::test_averaging_order_matters.
    """
    return torch.stack([sample_fn() for _ in range(n_samples)]).mean(0)


def human_validation_loss(
    sample_fn,
    ground_truth: torch.Tensor,
    space: str,
    n_samples: int = N_SAMPLES_PER_TIMESTEP,
) -> torch.Tensor:
    """一个时刻的验证损失 (论文 Sec. 3.3).

    space: "normalized" 或 "native". **论文没说算在哪个空间**, 所以这里是必填参数, 见
    README Sec. 8. 两者数值差一个量级, 不能猜.
    """
    if space not in ("normalized", "native"):
        raise ValueError("space 必须显式给出 'normalized' 或 'native' (论文未披露), "
                         "见 README Sec. 8")
    pred = averaged_prediction(sample_fn, n_samples)
    return torch.nn.functional.mse_loss(pred, ground_truth)


def validation_budget(n_episodes: int = N_VALIDATION_EPISODES) -> dict[str, int]:
    """论文协议一共要跑多少次策略前向."""
    per_episode = N_TIMESTEPS_PER_EPISODE * N_SAMPLES_PER_TIMESTEP
    return {
        "episodes": n_episodes,
        "timesteps_per_episode": N_TIMESTEPS_PER_EPISODE,
        "samples_per_timestep": N_SAMPLES_PER_TIMESTEP,
        "forwards_per_episode": per_episode,
        "total_forwards": n_episodes * per_episode,
    }


# ---------------------------------------------------------------------------
# 2. 真机: 任务与 rubric (论文 Sec. 3.1, App. B)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    key: str
    number: str  # 论文的编号; 注意 VI 与 VII 在论文里不存在
    name: str
    instruction: str
    kind: str  # "additive" 或 "progress"
    rubric: tuple[tuple[str, float], ...]  # (里程碑, 分值)
    trials: int
    platform: str
    note: str = ""


# --- R1Pro + Sharpa 手, 五个 post-training 任务 (论文 Sec. 3.1 与 App. B) ---
TASK_SHIRT = Task(
    key="shirt", number="I", name="Shirt Rolling",
    instruction="Roll the T-shirt and put it into the basket.",
    kind="progress",
    rubric=(("no folding", 0.0), ("initial fold", 0.3),
            ("multiple folds or partial rolling", 0.5),
            ("continuous rolling into a compact shape", 0.8),
            ("places the rolled shirt into the basket", 1.0)),
    trials=10, platform="r1pro_sharpa",
)
TASK_CARD = Task(
    key="card", number="II", name="Card Sorting",
    instruction="Pick up the card and sort it into the correct card holder.",
    kind="progress",
    rubric=(("no successful pickup", 0.0), ("poor grasp, multiple cards lifted", 0.3),
            ("single card picked up", 0.5),
            ("placed with noticeable disturbance or incorrect insertion", 0.7),
            ("correct card placed with minor disturbance", 0.9),
            ("single correct card cleanly placed", 1.0)),
    trials=10, platform="r1pro_sharpa",
)
TASK_TONG = Task(
    key="tong", number="III", name="Tong Fruit Transfer",
    instruction="Use the tong to pick up the fruit and place it into the basket.",
    kind="additive",
    rubric=(("grasps the tongs", 0.4), ("picks up the fruit", 0.2),
            ("places the fruit into the basket", 0.2),
            ("returns the tongs to the table", 0.2)),
    trials=10, platform="r1pro_sharpa",
    note="2 fruits (lemon, plum) x 5 trials each (App. B)",
)
TASK_BOTTLE = Task(
    key="bottle", number="IV", name="Bottle Cap Unscrewing",
    instruction="Unscrew the cap from the bottle.",
    kind="additive",
    rubric=(("grasps the bottle", 0.1),
            ("unscrews with at least three continuous rotations", 0.5),
            ("fully removes the cap", 0.2), ("places the cap on the table", 0.2)),
    trials=16, platform="r1pro_sharpa",
    note="4 bottles x 4 trials = 16; Sec. 3.1 says 16, App. B says 12 -- see README Sec. 1.x",
)
TASK_SYRINGE = Task(
    key="syringe", number="V", name="Syringe Liquid Transfer",
    instruction=("Pick up the syringe, draw liquid from tube A, inject it into tube B, "
                 "and throw the syringe into the trash can."),
    kind="additive",
    rubric=(("picks up the syringe", 0.1), ("correctly aims at tube A", 0.1),
            ("pulls the plunger to draw liquid", 0.2), ("re-aims at tube B", 0.1),
            ("pushes the plunger to inject liquid", 0.2),
            ("hands over or releases the syringe", 0.2),
            ("disposes of the syringe into the trash can", 0.1)),
    trials=10, platform="r1pro_sharpa",
)

# --- one-shot 任务 (论文 App. B). 注意合计分别是 1.1 与 1.2, 见 README Sec. 1.x 第 2 条 ---
TASK_FOLD_ONESHOT = Task(
    key="fold_oneshot", number="VIII", name="One-Shot T-Shirt Folding",
    instruction="Fold the T-shirt.",
    kind="additive",
    rubric=(("folds at least one sleeve", 0.4), ("folds both sleeves", 0.4),
            ("executes a bottom fold", 0.3)),
    trials=10, platform="r1pro_sharpa",
    note="a messy fold subtracts 0.1 (criterion not given); rubric sums to 1.1",
)
TASK_BOTTLE_ONESHOT = Task(
    key="bottle_oneshot", number="IX", name="One-Shot Bottle Cap Unscrewing",
    instruction="Unscrew the cap from the water bottle.",
    kind="additive",
    rubric=(("grasps the bottle", 0.1), ("one to two successful rotations", 0.2),
            ("at least three continuous rotations", 0.5), ("fully removes the cap", 0.2),
            ("places the cap on the table", 0.2)),
    trials=10, platform="r1pro_sharpa",
    note="rubric sums to 1.2",
)

# --- G1 + 三指手, 两个跨本体任务 (论文 App. B) ---
TASK_PEN = Task(
    key="pen_in_bin", number="G1-I", name="Pen in Bin",
    instruction="Marker canister task.",
    kind="additive",
    rubric=(("picks up or opens the canister", 0.25), ("places the canister down stably", 0.25),
            ("picks up the marker", 0.25), ("places the marker into the canister", 0.25)),
    trials=10, platform="g1_trifinger",
)
TASK_DISH = Task(
    key="dish_in_rack", number="G1-II", name="Dish Handover in Rack",
    instruction="Put plates on dishrack.",
    kind="additive",
    rubric=tuple(
        (f"plate {i + 1}: {m}", 0.11)
        for i in range(3)
        for m in ("picks up", "transfers between hands", "places upright into the rack")
    ),
    trials=10, platform="g1_trifinger",
    note="3 plates x 3 milestones x 0.11 = 0.99; the paper says the max is 1.0",
)

TASKS = {t.key: t for t in (TASK_SHIRT, TASK_CARD, TASK_TONG, TASK_BOTTLE, TASK_SYRINGE,
                            TASK_FOLD_ONESHOT, TASK_BOTTLE_ONESHOT, TASK_PEN, TASK_DISH)}

POST_TRAINING_SUITE = ("shirt", "card", "tong", "bottle", "syringe")  # 论文图 4
ONE_SHOT_SUITE = ("fold_oneshot", "bottle_oneshot")  # 论文 Sec. 3.4
G1_SUITE = ("pen_in_bin", "dish_in_rack")  # 论文 Sec. 3.5

N_SEEDS = 2  # 论文 Sec. 3.1: "we train each method using two random training seeds"


# ---------------------------------------------------------------------------
# 3. 打分 (论文 App. B 的两种策略)
# ---------------------------------------------------------------------------
def score_additive(task: Task, achieved: set[str]) -> float:
    """各子技能得分相加, 再截断到 [0, 1].

    截断是本仓库的选择: 论文正文说完成分在 [0, 1], 但 Task VIII / IX / Dish 的 rubric
    分别合计 1.1 / 1.2 / 0.99. 见 README Sec. 1.x 第 2 条.
    """
    total = sum(v for k, v in task.rubric if k in achieved)
    return float(min(max(total, 0.0), 1.0))


def score_progress(task: Task, achieved: set[str]) -> float:
    """取达到的**最远**里程碑对应的分数. 不可加.

    变形体或阶段耦合紧的任务用这个: 中间状态无法独立定义, 硬拆成加分项会出现"折了一半又散开"
    却拿到分的情况.
    """
    best = 0.0
    for name, value in task.rubric:  # rubric 按里程碑顺序排列
        if name in achieved:
            best = value
    return float(min(max(best, 0.0), 1.0))


def score(task: Task, achieved: set[str]) -> float:
    return (score_additive if task.kind == "additive" else score_progress)(task, achieved)


def binary_success(task: Task, achieved: set[str]) -> int:
    """App. B 末段: 只有按指令从头到尾完成才算成功."""
    required = {k for k, v in task.rubric if v > 0.0}
    return int(required <= achieved)


def rubric_total(task: Task) -> float:
    return float(sum(v for _, v in task.rubric))


def rubric_warnings() -> dict[str, float]:
    """rubric 相加不等于 1 的任务. 见 README Sec. 1.x 第 2 条."""
    return {k: rubric_total(t) for k, t in TASKS.items()
            if t.kind == "additive" and abs(rubric_total(t) - 1.0) > 1e-9}


# ---------------------------------------------------------------------------
# 4. 环境接口与假环境
# ---------------------------------------------------------------------------
class RobotTask:
    """真机任务的接口签名. 真实实现要接机器人; 本仓库只提供下面的假环境."""

    def reset(self, trial: int): raise NotImplementedError
    def step(self, action_chunk) -> tuple[object, set[str], bool]: raise NotImplementedError


@dataclass
class FakeRobotTask(RobotTask):
    """与 RobotTask 签名一致的假环境: 里程碑随机达成, **不模拟任务语义**.

    存在的唯一目的是让 episode 循环的每一步都被走过一次. 它产生的分数没有任何可解读性.
    """

    task: Task
    data_cfg: object
    rng: np.random.Generator
    max_chunks: int = 4
    _achieved: set[str] = field(default_factory=set)
    _chunks: int = 0

    def reset(self, trial: int):
        self._achieved = set()
        self._chunks = 0
        return fake_observation(self.data_cfg, self.task.platform, self.rng)

    def step(self, action_chunk):
        self._chunks += 1
        # 按 rubric 顺序推进, 每一步有一定概率达成下一个里程碑 (纯随机, 无语义)
        for name, value in self.task.rubric:
            if name not in self._achieved and value > 0.0:
                if self.rng.random() < 0.5:
                    self._achieved.add(name)
                break
        done = self._chunks >= self.max_chunks
        obs = fake_observation(self.data_cfg, self.task.platform, self.rng)
        return obs, set(self._achieved), done


# ---------------------------------------------------------------------------
# 5. episode 循环与聚合 (论文 Sec. 3.1, Sec. 3.5)
# ---------------------------------------------------------------------------
def run_episode(policy: EgoScalePolicy, env: RobotTask, task: Task, trial: int,
                execute_steps: int | None = None) -> dict:
    """一次试验: reset -> (推理一个 chunk -> 执行 -> 收里程碑)* -> 打分.

    execute_steps 是开环执行长度: 一次推理出 H 步, 实际执行几步. **EgoScale 未披露**,
    见 README Sec. 8; None 表示执行整块.
    """
    obs = env.reset(trial)
    achieved: set[str] = set()
    n_infer = 0
    while True:
        action, _ = policy.act(obs)
        n_infer += 1
        chunk = action if execute_steps is None else action[:execute_steps]
        obs, achieved, done = env.step(chunk)
        if done:
            break
    return {"score": score(task, achieved), "success": binary_success(task, achieved),
            "achieved": achieved, "n_infer": n_infer}


def run_task(policy: EgoScalePolicy, task: Task, data_cfg, seeds=range(N_SEEDS),
             trials: int | None = None, execute_steps: int | None = None) -> dict:
    """论文 Sec. 3.1: 两个随机种子, 每个 checkpoint 若干次试验, 报两个种子的平均."""
    n_trials = task.trials if trials is None else trials
    per_seed = []
    for seed in seeds:
        env = FakeRobotTask(task, data_cfg, np.random.default_rng(1000 + seed))
        rows = [run_episode(policy, env, task, t, execute_steps) for t in range(n_trials)]
        per_seed.append({
            "score": float(np.mean([r["score"] for r in rows])),
            "success": float(np.mean([r["success"] for r in rows])),
            "trials": n_trials,
        })
    return {
        "task": task.key,
        "per_seed": per_seed,
        "score": float(np.mean([s["score"] for s in per_seed])),
        "success": float(np.mean([s["success"] for s in per_seed])),
    }


def run_suite(policy: EgoScalePolicy, suite, data_cfg, **kw) -> dict:
    rows = {k: run_task(policy, TASKS[k], data_cfg, **kw) for k in suite}
    return {
        "tasks": rows,
        "average_score": float(np.mean([r["score"] for r in rows.values()])),
        "average_success": float(np.mean([r["success"] for r in rows.values()])),
    }


# ---------------------------------------------------------------------------
# 6. 一次 tiny 运行
# ---------------------------------------------------------------------------
def main() -> None:
    policy, data_cfg = build_tiny_policy()
    print("=== evaluation (1): human validation loss protocol, paper Sec. 3.3 ===")
    b = validation_budget()
    print(f"  {b['episodes']:,} held-out episodes x {b['timesteps_per_episode']} timesteps "
          f"x {b['samples_per_timestep']} samples = {b['total_forwards']:,} policy forwards")
    rng = np.random.default_rng(0)
    obs = fake_observation(data_cfg, "human_wild", rng)
    gt = policy.act(obs)[0]
    single = torch.stack([torch.nn.functional.mse_loss(policy.act(obs)[0], gt)
                          for _ in range(8)])
    avg16 = torch.stack([human_validation_loss(lambda: policy.act(obs)[0], gt,
                                               space="normalized", n_samples=16)
                         for _ in range(8)])
    print(f"  1-sample loss : mean {single.mean():.5f}  std {single.std():.5f}")
    print(f"  16-sample loss: mean {avg16.mean():.5f}  std {avg16.std():.5f}  "
          f"(averaging the PREDICTIONS, not the errors)")
    print(f"  space must be passed explicitly (paper does not say which) -> README Sec. 8")

    print("\n=== evaluation (2): real-robot rubrics, paper Sec. 3.1 / App. B ===")
    print(f"  {'task':18s} {'no.':>5s} {'kind':>9s} {'items':>6s} {'sum*':>6s} {'trials':>7s}  "
          f"platform      (* additive only; progress milestones are not additive)")
    for t in TASKS.values():
        # progress 的里程碑不可加, 所以"合计"对它没有意义, 打成 "-"
        total = f"{rubric_total(t):>6.2f}" if t.kind == "additive" else f"{'-':>6s}"
        print(f"  {t.name:18.18s} {t.number:>5s} {t.kind:>9s} {len(t.rubric):>6d} "
              f"{total} {t.trials:>7d}  {t.platform}")
    warn = rubric_warnings()
    print(f"\n  rubrics whose items do NOT sum to 1.0: {warn}")
    print(f"  -> scores are clipped to [0, 1]; that clipping is this repo's choice "
          f"(README Sec. 1.x row 2)")

    print("\n  running the episode loop on the FAKE environment (no task semantics):")
    for suite_name, suite in (("post-training", POST_TRAINING_SUITE),
                              ("one-shot", ONE_SHOT_SUITE), ("G1", G1_SUITE)):
        res = run_suite(policy, suite, data_cfg, trials=2)
        per_task = ", ".join(f"{k} {v['score']:.2f}/{v['success']:.2f}"
                             for k, v in res["tasks"].items())
        print(f"    [{suite_name:13s}] avg score {res['average_score']:.3f}  "
              f"avg success {res['average_success']:.3f}")
        print(f"      per task (score/success): {per_task}")
    print("\n  these numbers are meaningless by construction: the fake environment reaches "
          "milestones at random. The loop, the scoring and the aggregation are what is being "
          "exercised.")

    total_trials = sum(TASKS[k].trials for k in POST_TRAINING_SUITE + ONE_SHOT_SUITE + G1_SUITE)
    print(f"\n  a full real evaluation of ONE method: {N_SEEDS} seeds x {total_trials} trials "
          f"= {N_SEEDS * total_trials} robot trials (paper Sec. 3.1)")


if __name__ == "__main__":
    main()
