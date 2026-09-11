"""pi0 evaluation flow: environment interface, episode loop with chunked replanning, scoring, aggregation.

The numbers are not reproduced (repo rule); the procedure is. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          examples/libero/main.py (the only public eval loop: LIBERO in simulation, L48-L196)
  paper   pi0 arXiv:2410.24164v1 Sec. V (10 episodes per task, normalized score), Appendix D (chunk execution),
          Appendix E (per-task scoring rubrics for the real-robot tasks)
Upstream license: Apache-2.0. This file re-implements, it does not copy. LIBERO itself (the simulator, the BDDL task
files, the 50 initial states per task) is not vendored; `Env` below is the interface an adapter must provide, and
`ToyEnv` is a stand-in so the loop runs on CPU.

Two metrics exist in the pi0 ecosystem and the loop supports both:
  binary success   LIBERO: env reports `done` when the BDDL goal predicate holds; score 1 or 0 (main.py L153-L157)
  rubric score     paper's real-robot tasks: points / max_points in [0, 1], e.g. bussing = objects placed / 12
Aggregation is a plain mean over episodes per task, then over tasks (paper Sec. V-A; main.py L182-L185).
"""

from __future__ import annotations

import collections
import dataclasses
import math
from typing import Protocol

import numpy as np

from pi.pi0.infer.model import Pi0Policy


# ======================================================================================
# 1. LIBERO facts, as used by openpi's eval script. examples/libero/main.py L17-L18, L28-L38, L60-L71, L113-L141.
# ======================================================================================
LIBERO_SUITES = {  # suite -> (number of tasks, max env steps per episode); "longest training demo" comments in main.py
    "libero_spatial": (10, 220),
    "libero_object": (10, 280),
    "libero_goal": (10, 300),
    "libero_10": (10, 520),
    "libero_90": (90, 400),
}
LIBERO_NUM_TRIALS_PER_TASK = 50  # main.py L38; one fixed initial state per trial (L82, L97)
LIBERO_REPLAN_STEPS = 5  # main.py L29: execute 5 of the 50 predicted actions, then re-infer
LIBERO_NUM_STEPS_WAIT = 10  # main.py L37: dummy actions while objects settle
LIBERO_ENV_RESOLUTION = 256  # main.py L18: render size; resized (with pad) to 224 before the policy (L28, L117-L122)
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # main.py L17: zero motion, gripper open
LIBERO_STATE_DIM = 8  # eef position 3 + eef axis-angle 3 + gripper qpos 2 (main.py L133-L139)
LIBERO_ACTION_DIM = 7  # 6 eef deltas + 1 gripper (libero_policy.py L94-L100)
LIBERO_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")  # -> base_0_rgb, left_wrist_0_rgb; both rotated 180 deg (L114-L116)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> axis-angle f32[3]. main.py L199-L214, taken there from robosuite transform_utils."""
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def libero_obs_to_raw(obs: dict, prompt: str) -> dict:
    """LIBERO env observation -> the `raw` dict Pi0Policy.infer takes (../data). main.py L113-L141 + libero_policy.py
    L30-L84: rotate both images 180 deg (training data was stored that way), state = [eef_pos, axis-angle, gripper]."""
    base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]).astype(np.float32)
    return {"images": {"base_0_rgb": base[None], "left_wrist_0_rgb": wrist[None]}, "state": state[None], "prompt": [prompt]}


# ======================================================================================
# 2. The environment interface an adapter must provide, and the two scoring rules.
# ======================================================================================
class Env(Protocol):
    def reset(self, task_id: int, episode_idx: int) -> tuple[dict, str]:
        """-> (raw observation dict for Pi0Policy.infer with batch size 1, task prompt).
        LIBERO: env.reset(); env.set_init_state(initial_states[episode_idx]) (main.py L93-L97)."""

    def step(self, action: np.ndarray) -> tuple[dict, bool, dict]:
        """action f32[native_dim] -> (raw observation, done, info). LIBERO: done = goal predicate satisfied."""


@dataclasses.dataclass
class EpisodeResult:
    task_id: int
    episode_idx: int
    score: float  # 1 / 0 for binary success; points / max_points for a rubric
    steps: int  # env steps taken after the wait period
    infer_calls: int


def rubric_score(points: int, max_points: int) -> float:
    """Paper Appendix E: partial credit, e.g. bussing hard = correctly sorted objects / 12; shirt folding = 1 or 0."""
    assert 0 <= points <= max_points
    return points / max_points


# ======================================================================================
# 3. One episode: chunked replanning. examples/libero/main.py L92-L165.
# ======================================================================================
def run_episode(policy: Pi0Policy, env: Env, task_id: int, episode_idx: int, *, max_steps: int,
                replan_steps: int = LIBERO_REPLAN_STEPS, num_steps_wait: int = 0, dummy_action=None,
                noise_rng: np.random.Generator | None = None) -> EpisodeResult:
    """Run the policy until the env reports done or max_steps is reached. The policy predicts 50 actions per call;
    only the first `replan_steps` are executed before the next observation is taken (openpi LIBERO: 5;
    paper real robots: 25 at 50 Hz, 16 at 20 Hz, Appendix D). No temporal ensembling."""
    obs, prompt = env.reset(task_id, episode_idx)
    plan: collections.deque = collections.deque()
    done, t, infer_calls = False, 0, 0
    for _ in range(num_steps_wait):  # LIBERO: let dropped objects settle before acting (main.py L106-L111)
        obs, done, _ = env.step(np.asarray(dummy_action, np.float32))
    while t < max_steps and not done:
        if not plan:  # finished the previous chunk: observe and re-plan (main.py L127-L148)
            raw = libero_obs_to_raw(obs, prompt) if "agentview_image" in obs else obs
            chunk = policy.infer(raw)["actions"][0]  # [50, native_dim]
            assert len(chunk) >= replan_steps
            plan.extend(chunk[:replan_steps])
            infer_calls += 1
        obs, done, _ = env.step(plan.popleft())
        t += 1
    return EpisodeResult(task_id, episode_idx, float(done), t, infer_calls)


# ======================================================================================
# 4. A whole benchmark: tasks x trials -> per-task and overall score. main.py L75-L186; paper Sec. V-A.
# ======================================================================================
def evaluate(policy: Pi0Policy, env: Env, num_tasks: int, num_trials: int, *, max_steps: int, **episode_kwargs) -> dict:
    per_task, results = {}, []
    for task_id in range(num_tasks):
        eps = [run_episode(policy, env, task_id, i, max_steps=max_steps, **episode_kwargs) for i in range(num_trials)]
        results += eps
        per_task[task_id] = float(np.mean([e.score for e in eps]))  # "Current task success rate", main.py L182
    return {
        "per_task": per_task,
        "mean_over_tasks": float(np.mean(list(per_task.values()))),  # what the tables report (LIBERO README, paper Fig. 7)
        "mean_over_episodes": float(np.mean([e.score for e in results])),  # "Total success rate", main.py L185
        "episodes": results,
    }


# ======================================================================================
# 5. A stand-in environment so the loop runs without LIBERO. Not a simulator: images are noise, "success" is a
#    scripted rule (the episode succeeds once the gripper command has been > 0 for `hold` consecutive steps).
# ======================================================================================
class ToyEnv:
    def __init__(self, native_dim: int = 7, image_hw=(128, 128), seed: int = 0):
        self.d, self.hw, self.rng = native_dim, image_hw, np.random.default_rng(seed)
        self.prompts = ["put the bowl on the plate", "close the drawer", "stack the cups"]

    def _obs(self):
        return {"images": {"base_0_rgb": self.rng.integers(0, 256, (1, *self.hw, 3), dtype=np.uint8),
                           "left_wrist_0_rgb": self.rng.integers(0, 256, (1, *self.hw, 3), dtype=np.uint8)},
                "state": self.rng.standard_normal((1, self.d)).astype(np.float32),
                "prompt": [self.prompt]}

    def reset(self, task_id, episode_idx):
        self.prompt = self.prompts[task_id % len(self.prompts)]
        self.hold, self.need = 0, 3 + (task_id + episode_idx) % 3
        return self._obs(), self.prompt

    def step(self, action):
        self.hold = self.hold + 1 if action[-1] > 0 else 0
        return self._obs(), self.hold >= self.need, {}


# ======================================================================================
#   uv run python -m pi.pi0.infer.eval
# ======================================================================================
def main():
    import torch

    from pi.pi0.data.data import ByteEncoder, NormStats, PromptTokenizer, make_bool_mask
    from pi.pi0.infer.model import RobotSpec, tiny_pi0

    torch.manual_seed(0)
    d = 7
    robot = RobotSpec({"state": NormStats(np.zeros(d, np.float32), np.ones(d, np.float32)),
                       "actions": NormStats(np.zeros(d, np.float32), np.ones(d, np.float32))}, make_bool_mask(6, -1), d)
    policy = Pi0Policy(tiny_pi0(), PromptTokenizer(ByteEncoder()), robot)
    env = ToyEnv(d)
    num_tasks, num_trials, max_steps = 2, 3, 40
    print(f"benchmark: {num_tasks} tasks x {num_trials} trials, max {max_steps} steps, replan every {LIBERO_REPLAN_STEPS} of 50 predicted actions")
    print("LIBERO for reference:", {k: f"{n} tasks, max {m} steps" for k, (n, m) in LIBERO_SUITES.items()}, f"x {LIBERO_NUM_TRIALS_PER_TASK} trials")
    for task_id in range(num_tasks):
        for i in range(num_trials):
            r = run_episode(policy, env, task_id, i, max_steps=max_steps)
            print(f"  task {task_id} '{env.prompt}' episode {i}: {'success' if r.score else 'failure'} after {r.steps} steps, {r.infer_calls} infer calls")
    out = evaluate(policy, env, num_tasks, num_trials, max_steps=max_steps)
    print(f"per task {out['per_task']}  mean over tasks {out['mean_over_tasks']:.2f}  mean over episodes {out['mean_over_episodes']:.2f}")
    print(f"rubric example (paper App. E, bussing hard, 9 of 12 objects sorted): score {rubric_score(9, 12):.3f}")


if __name__ == "__main__":
    main()
