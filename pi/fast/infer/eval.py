"""pi0-FAST evaluation flow: LIBERO reuses pi0's loop; DROID adds the paper's 44-trial rubric suite (Table II) and a
DROID-shaped stand-in environment. Numbers are not reproduced (repo rule); the procedure is.

Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           examples/libero/main.py (the eval loop, same for pi0 and pi0-FAST), src/openpi/policies/libero_policy.py L52-L69,
           src/openpi/policies/droid_policy.py L10-L18
  paper    FAST, arXiv:2501.09747v1, Sec. VI-B (DROID zero-shot), Appendix D, E, Table II
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Everything generic (Env protocol, EpisodeResult, rubric_score, run_episode with chunked replanning, evaluate, ToyEnv,
LIBERO constants and observation adapter) is pi.pi0.infer.eval. This file only holds the FAST increment.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

from pi.fast.infer.model import DROID_ACTION_DIM, DROID_ACTION_HORIZON_PAPER, DROID_EXECUTE_STEPS, DROID_STATE_DIM, Pi0FASTPolicy, droid_obs_to_raw, fast_image_slots
from pi.pi0.infer.eval import (  # noqa: F401  (re-exported: the LIBERO loop is identical for pi0-FAST)
    LIBERO_ACTION_DIM,
    LIBERO_NUM_TRIALS_PER_TASK,
    LIBERO_REPLAN_STEPS,
    LIBERO_SUITES,
    Env,
    EpisodeResult,
    ToyEnv,
    evaluate,
    libero_obs_to_raw,
    rubric_score,
    run_episode,
)

# Paper Table II, row by row (task, trials). 17 rows, 44 trials; the text says "16 tasks" (README Sec. 8).
DROID_TASKS = (
    ("Put the spoon in the dish rack", 4),
    ("Put carrot in bowl", 4),
    ("Put plate in dish rack", 2),
    ("Wipe the table", 2),
    ("Put the plate on the table", 2),
    ("Clean up the table", 2),
    ("Close the drawer", 4),
    ("Put the stapler on the notebook", 2),
    ("Put stapler in the drawer", 4),
    ("Clean the whiteboard", 2),
    ("Put the marker in the cup", 4),
    ("Put the black sponge in the blue bowl", 2),
    ("Put the red bottle in the black bowl", 2),
    ("Put the watermelon in the purple bowl", 2),
    ("Move the watermelon from the purple bowl to the blue bowl", 2),
    ("Put the tape in the purple bowl", 2),
    ("Put the water bottle on the left side of the table", 2),
)
DROID_TOTAL_TRIALS = 44  # Table II "Total"


def libero_obs_to_raw_fast(obs: dict, prompt: str) -> dict:
    """pi0's LIBERO adapter with the FAST slot name (libero_policy.py L58-L69: same keys, wrist -> wrist_0_rgb, no masking)."""
    return fast_image_slots(libero_obs_to_raw(obs, prompt))


class RubricEnv(Env, Protocol):
    def rubric(self) -> tuple[int, int]:
        """After an episode: (points, max_points) for the task-progress rubric. On a real robot a human fills this in."""


@dataclasses.dataclass
class RubricResult:
    task: str
    episode_idx: int
    points: int
    max_points: int
    score: float
    steps: int
    infer_calls: int


def evaluate_rubric(policy: Pi0FASTPolicy, env: RubricEnv, tasks=DROID_TASKS, *, max_steps: int, replan_steps: int = DROID_EXECUTE_STEPS[0]) -> dict:
    """Paper Sec. VI-B / Appendix E: each trial scored as points / max_points; the reported '% task progress' is the mean
    over all trials (tasks have unequal trial counts, Table II). Per-task means are given too."""
    results = []
    for task_id, (task, trials) in enumerate(tasks):
        for i in range(trials):
            ep = run_episode(policy, env, task_id, i, max_steps=max_steps, replan_steps=replan_steps)
            points, max_points = env.rubric()
            results.append(RubricResult(task, i, points, max_points, rubric_score(points, max_points), ep.steps, ep.infer_calls))
    per_task = {t: float(np.mean([r.score for r in results if r.task == t])) for t, _ in tasks}
    return {"per_task": per_task, "task_progress": float(np.mean([r.score for r in results])), "n_trials": len(results), "episodes": results}


class DROIDToyEnv:
    """DROID-shaped stand-in: observation keys and shapes of droid_policy.py, converted with droid_obs_to_raw on the way
    out, so the adapter is exercised. Not a simulator: images are noise; the scripted rubric has 2 points (joint-0 target
    above 0.1 for `need` consecutive steps, then joint-1 target above 0.1 for `need` steps), done when both are earned.
    With an untrained policy the targets are the (random) current joints, so trials succeed or time out at random."""

    def __init__(self, image_hw=(180, 320), seed: int = 0):
        self.hw, self.rng = image_hw, np.random.default_rng(seed)

    def _obs(self):
        return {"observation/exterior_image_1_left": self.rng.integers(0, 256, (*self.hw, 3), dtype=np.uint8),
                "observation/wrist_image_left": self.rng.integers(0, 256, (*self.hw, 3), dtype=np.uint8),
                "observation/joint_position": self.rng.uniform(-0.5, 0.5, 7).astype(np.float32),
                "observation/gripper_position": np.float32(self.rng.uniform(0, 1))}

    def reset(self, task_id, episode_idx):
        self.prompt = DROID_TASKS[task_id % len(DROID_TASKS)][0]
        self.points, self.hold, self.need = 0, 0, 2 + (task_id + episode_idx) % 3
        return droid_obs_to_raw(self._obs(), self.prompt), self.prompt

    def step(self, action):
        assert action.shape == (DROID_ACTION_DIM,), action.shape
        cond = action[0] > 0.1 if self.points == 0 else action[1] > 0.1
        self.hold = self.hold + 1 if cond else 0
        if self.hold >= self.need:
            self.points, self.hold = self.points + 1, 0
        return droid_obs_to_raw(self._obs(), self.prompt), self.points >= 2, {"points": self.points, "max_points": 2}

    def rubric(self):
        return self.points, 2


# ======================================================================================
#   uv run python -m pi.fast.infer.eval
# ======================================================================================
def main() -> None:
    from pi.fast.infer.model import tiny_fast_policy

    policy, _ = tiny_fast_policy(max_decoding_steps=24)
    env = DROIDToyEnv()
    tasks, max_steps, replan = DROID_TASKS[:3], 40, DROID_EXECUTE_STEPS[0]
    print(f"DROID suite: {len(DROID_TASKS)} rows, {sum(n for _, n in DROID_TASKS)} trials (Table II); here {len(tasks)} tasks, "
          f"max {max_steps} steps, predict {DROID_ACTION_HORIZON_PAPER} execute {replan} (App. D)")
    for task_id, (task, trials) in enumerate(tasks):
        for i in range(trials):
            ep = run_episode(policy, env, task_id, i, max_steps=max_steps, replan_steps=replan)
            p, mp = env.rubric()
            print(f"  task {task_id} '{task}' trial {i}: {p}/{mp} points after {ep.steps} steps, {ep.infer_calls} infer calls")
    out = evaluate_rubric(policy, env, tasks, max_steps=max_steps, replan_steps=replan)
    print(f"per task {{{', '.join(f'{k[:22]}: {v:.2f}' for k, v in out['per_task'].items())}}}  task progress {out['task_progress']:.2f} over {out['n_trials']} trials")
    print(f"LIBERO for reference: {list(LIBERO_SUITES)} x {LIBERO_NUM_TRIALS_PER_TASK} trials, replan {LIBERO_REPLAN_STEPS} of 10; same loop as pi0 (pi.pi0.infer.eval), "
          f"raw via libero_obs_to_raw_fast, state {DROID_STATE_DIM}-dim here vs LIBERO 8, actions {DROID_ACTION_DIM} vs {LIBERO_ACTION_DIM}")


if __name__ == "__main__":
    main()
