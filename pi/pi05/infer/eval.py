"""pi0.5 evaluation flow: the mock-home rubric suite (paper Appendix B), the language-following protocol (Appendix C),
Hi Robot's Instruction Accuracy / Task Progress (Hi Robot Sec. 5.2), and the two-level episode loop that drives them.
Numbers are not reproduced (repo rule); the procedure is, on a stand-in environment.

Sources of truth:
  paper    pi0.5 arXiv:2504.16054v1 Sec. V-A, V-B, V-E, Appendix B (rubrics, 10 trials per task, 12 locations), Appendix C;
           Hi Robot arXiv:2502.19417v2 Sec. 4.1-4.2 (schedule), 5.1 (tasks), 5.2 (metrics, 20 trials per task per method)
  openpi   no evaluation code for pi0.5 exists upstream; Env / EpisodeResult / rubric_score are pi.pi0.infer.eval,
           RubricResult is pi.fast.infer.eval
License: this file is the repo's own (the upstream pieces it imports are Apache-2.0 re-implementations).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from pi.fast.infer.eval import RubricResult  # noqa: F401  (re-exported)
from pi.pi0.infer.eval import Env, rubric_score  # noqa: F401
from pi.pi05.hier.model import HL_PERIOD_S
from pi.pi05.infer.model import CONTROL_HZ, MOBILE_CAMERAS, Pi05HierPolicy, mobile_obs_to_raw

# ---- Appendix B: four tasks, their high-level prompt (Sec. V-A / Fig. 7 wording), rubric points, 10 trials per task per policy
MOCK_HOME_TASKS = (
    ("Dishes in Sink", "place the dishes in the sink", 8, ("+1 per item picked up (4 items)", "+1 per item placed in the sink (4 items)")),
    ("Items in Drawer", "put the items in the drawer", 4, ("+1 picking up the object", "+1 opening the drawer", "+1 putting the object into the drawer", "+1 closing the drawer (object inside)")),
    ("Laundry Basket", "put the laundry in the laundry basket", 3, ("+1 navigating to and picking up the clothing", "+1 placing the clothing into or on the basket", "+1 clothing fully inside the basket")),
    ("Make Bed", "make the bed", 5, ("+1 blanket straightened to cover the sheets", "+1 first pillow at the head", "+1 second pillow at the head", "+1 blanket very neat", "+1 both pillows very neat")),
)
TRIALS_PER_TASK = 10  # Appendix B "we perform 10 evaluations per task"
LOCATIONS = ("real kitchen",) * 3 + ("real bedroom",) * 3 + ("mock kitchen",) * 3 + ("mock bedroom",) * 3  # 12 locations (Appendix B)
STANDARD_EVALS_PER_POLICY = 40  # Appendix B "a total of 40 evaluations per policy for our standard evaluations"

# ---- Appendix C: language following, two scenarios x five objects, plus five out-of-distribution objects
LANGUAGE_FOLLOWING = {
    "items in the drawer": ("tongs", "wooden serving spoon", "can opener", "scissors", "small yellow mustard"),
    "items in the sink": ("cup", "bowl", "plate", "plastic spoon", "cutting board"),
}
LANGUAGE_FOLLOWING_OOD = ("funnel", "pill bottle", "grill lighter", "lighter", "safety goggles")  # Appendix C, "items in the drawer" with novel objects
CHANCE_LANGUAGE_FOLLOWING = 0.2  # Appendix C: 5 objects, the target placed farther than the distractors -> ~20% for a policy that ignores the command

# ---- Hi Robot Sec. 5.1 / 5.2
HIROBOT_TASKS = (("table bussing", "UR5e, 7-dim"), ("sandwich making", "bimanual ARX, 14-dim"), ("grocery shopping", "mobile ARX, 14-dim state / 16-dim action"))
HIROBOT_TRIALS_PER_TASK = 20


def instruction_accuracy(judgements: list[bool]) -> float:
    """Hi Robot Sec. 5.2: fraction of high-level predictions in a trial the blind evaluator marks consistent with the user's
    command and the current observation. Flat baselines are judged on the intent of their behaviour instead."""
    return float(np.mean(judgements)) if judgements else 0.0


def task_progress(placed: int, total: int) -> float:
    """Hi Robot Sec. 5.2: proportion of objects placed in their correct location / configuration."""
    assert 0 <= placed <= total and total > 0
    return placed / total


# ======================================================================================
# 1. The two-level episode loop. Low level every `execute_steps` control steps (undisclosed; README Sec. 8),
#    high level every hl_period_s or on a user message (Hi Robot Sec. 4.1-4.2).
# ======================================================================================
@dataclasses.dataclass
class HierEpisode:
    task: str
    episode_idx: int
    steps: int
    ll_calls: int
    hl_calls: int
    subtasks: list[str]  # every high-level output, in order
    utterances: list[str]
    points: int
    max_points: int
    score: float


class HierEnv(Env):
    """What the two-level loop needs beyond pi.pi0.infer.eval.Env: the raw observation dict is the mobile one
    ({"front", "rear", "left_wrist", "right_wrist", "state"}); rubric() after the episode."""

    def rubric(self) -> tuple[int, int]: ...


def run_hier_episode(policy: Pi05HierPolicy, env, task: str, episode_idx: int, *, max_steps: int, execute_steps: int,
                     hl_period_s: float = HL_PERIOD_S, control_hz: float = CONTROL_HZ, user_messages: dict[int, str] | None = None) -> HierEpisode:
    """user_messages: {control step: text} interjections; each triggers the high level at that step (Sec. 4.2)."""
    obs, prompt = env.reset(task, episode_idx)
    policy.last_hl_time, policy.subtask, policy.previous_subtask = None, "", None
    policy.hl_period_s = hl_period_s
    plan: list[np.ndarray] = []
    t, done, ll_calls, hl_calls, subtasks, utterances = 0, False, 0, 0, [], []
    while t < max_steps and not done:
        msg = (user_messages or {}).get(t)
        if not plan or msg is not None:  # a message re-plans immediately (the new subtask changes the low-level prompt)
            raw_hl = mobile_obs_to_raw(obs, prompt, hl=True)
            raw_ll = mobile_obs_to_raw(obs, prompt, hl=False)
            out = policy.step(raw_hl, {"images": raw_ll["images"], "state": raw_ll["state"]}, t / control_hz, msg)
            plan = list(out["actions_exec"][0][:execute_steps])
            ll_calls += 1
            if out["hl_ran"]:
                hl_calls += 1
                subtasks.append(out["subtask"])
                if out["utterance"]:
                    utterances.append(out["utterance"])
        obs, done, _ = env.step(plan.pop(0))
        t += 1
    points, max_points = env.rubric()
    return HierEpisode(task, episode_idx, t, ll_calls, hl_calls, subtasks, utterances, points, max_points, rubric_score(points, max_points))


# ======================================================================================
# 2. The three evaluations.
# ======================================================================================
def evaluate_mock_home(policy: Pi05HierPolicy, env, tasks=MOCK_HOME_TASKS, *, trials_per_task: int = TRIALS_PER_TASK, max_steps: int, execute_steps: int, **kw) -> dict:
    """Appendix B / Fig. 7b, 10: per trial points / max_points; per task the mean over its trials; the headline number is the
    mean over tasks ("task progress", in percent)."""
    episodes = []
    for task, _prompt, _max, _rubric in tasks:
        for i in range(trials_per_task):
            episodes.append(run_hier_episode(policy, env, task, i, max_steps=max_steps, execute_steps=execute_steps, **kw))
    per_task = {task: float(np.mean([e.score for e in episodes if e.task == task])) for task, *_ in tasks}
    return {"per_task": per_task, "task_progress": float(np.mean(list(per_task.values()))), "n_trials": len(episodes), "episodes": episodes}


def evaluate_language_following(policy: Pi05HierPolicy, env, scenarios=LANGUAGE_FOLLOWING, *, trials_per_object: int = 1, max_steps: int, execute_steps: int, **kw) -> dict:
    """Appendix C / Fig. 9, 11, 15: each trial names one of five objects; language following rate = the named object was
    selected; success rate = it ended in the right place. Both averaged over the two scenarios."""
    rows = []
    for scenario, objects in scenarios.items():
        for obj in objects:
            for i in range(trials_per_object):
                task = f"{scenario}: {obj}"
                ep = run_hier_episode(policy, env, task, i, max_steps=max_steps, execute_steps=execute_steps, **kw)
                followed, succeeded = env.language_check()
                rows.append((scenario, obj, followed, succeeded, ep))
    per_scenario = {s: {"language_following_rate": float(np.mean([r[2] for r in rows if r[0] == s])), "success_rate": float(np.mean([r[3] for r in rows if r[0] == s]))} for s in scenarios}
    return {"per_scenario": per_scenario,
            "language_following_rate": float(np.mean([v["language_following_rate"] for v in per_scenario.values()])),
            "success_rate": float(np.mean([v["success_rate"] for v in per_scenario.values()])),
            "chance": CHANCE_LANGUAGE_FOLLOWING, "rows": rows}


def evaluate_hirobot(policy: Pi05HierPolicy, env, tasks=HIROBOT_TASKS, *, trials_per_task: int = HIROBOT_TRIALS_PER_TASK, max_steps: int, execute_steps: int, **kw) -> dict:
    """Hi Robot Sec. 5.2: IA and TP per trial (a blind human judges each high-level prediction; the stand-in env judges
    with a script), averaged per task and over tasks (Fig. 5)."""
    per_task, all_ia, all_tp = {}, [], []
    for task, _robot in tasks:
        ia, tp = [], []
        for i in range(trials_per_task):
            ep = run_hier_episode(policy, env, task, i, max_steps=max_steps, execute_steps=execute_steps, **kw)
            ia.append(instruction_accuracy(env.judge(ep.subtasks)))
            tp.append(task_progress(*env.placed()))
        per_task[task] = {"instruction_accuracy": float(np.mean(ia)), "task_progress": float(np.mean(tp))}
        all_ia += ia
        all_tp += tp
    return {"per_task": per_task, "instruction_accuracy": float(np.mean(all_ia)), "task_progress": float(np.mean(all_tp))}


# ======================================================================================
# 3. A stand-in environment. Not a simulator: images are noise; the rubric, the language check and the judge are
#    scripted rules so every branch of the loop runs. Scores mean nothing.
# ======================================================================================
class MockHomeToyEnv:
    def __init__(self, n_dims: int = 19, image_hw=(96, 128), seed: int = 0):
        self.d, self.hw, self.rng = n_dims, image_hw, np.random.default_rng(seed)

    def _obs(self):
        o = {name: self.rng.integers(0, 256, (*self.hw, 3), dtype=np.uint8) for name in MOBILE_CAMERAS}
        o["state"] = self.rng.uniform(-0.5, 0.5, self.d).astype(np.float32)
        return o

    def reset(self, task, episode_idx):
        self.task, self.points, self.hold = task, 0, 0
        self.max_points = next((m for name, _p, m, _r in MOCK_HOME_TASKS if name == task), 3)
        self.need = 4 + episode_idx % 3
        prompt = next((p for name, p, _m, _r in MOCK_HOME_TASKS if name == task), task)
        return self._obs(), prompt

    def step(self, action):
        assert action.shape == (self.d,), action.shape
        # scripted rule: a point every `need` consecutive steps with the base angular-velocity target above 0.1
        self.hold = self.hold + 1 if action[16] > 0.1 else 0
        if self.hold >= self.need and self.points < self.max_points:
            self.points, self.hold = self.points + 1, 0
        return self._obs(), self.points >= self.max_points, {"points": self.points}

    def rubric(self):
        return self.points, self.max_points

    def language_check(self):  # (followed, succeeded): scripted from the points earned
        return self.points >= 1, self.points >= 2

    def judge(self, subtasks):  # a "blind evaluator": here, non-empty subtasks count as consistent
        return [bool(s) for s in subtasks] or [False]

    def placed(self):
        return min(self.points, 3), 3


# ======================================================================================
#   uv run python -m pi.pi05.infer.eval
# ======================================================================================
def main() -> None:
    from pi.pi05.infer.model import tiny_pi05_policy

    policy, robot = tiny_pi05_policy(19, hier=True, max_new_tokens=4)
    env = MockHomeToyEnv()
    print("mock-home suite (Appendix B):")
    for name, prompt, m, rubric in MOCK_HOME_TASKS:
        print(f"  {name:16s} prompt {prompt!r:42s} {m} points: {'; '.join(rubric)}")
    print(f"  {TRIALS_PER_TASK} trials per task per policy, {len(LOCATIONS)} locations, {STANDARD_EVALS_PER_POLICY} standard evaluations per policy")
    ep = run_hier_episode(policy, env, "Items in Drawer", 0, max_steps=120, execute_steps=25, user_messages={30: "not that one"})
    print(f"\none episode: {ep.steps} steps, {ep.ll_calls} low-level calls (every 25 steps or on a message), {ep.hl_calls} high-level calls "
          f"(t=0, every 1 s = 50 steps, + the message at step 30), rubric {ep.points}/{ep.max_points} -> {ep.score:.2f}")
    res = evaluate_mock_home(policy, env, trials_per_task=2, max_steps=60, execute_steps=25)
    print(f"mock home, 2 trials each (toy): per task {{{', '.join(f'{k}: {v:.2f}' for k, v in res['per_task'].items())}}}, task progress {res['task_progress']:.2f}")
    lf = evaluate_language_following(policy, env, max_steps=40, execute_steps=25)
    print(f"language following (toy): LF rate {lf['language_following_rate']:.2f}, success {lf['success_rate']:.2f} (chance {lf['chance']}); scenarios {list(lf['per_scenario'])}")
    hr = evaluate_hirobot(policy, env, trials_per_task=2, max_steps=40, execute_steps=25)
    print(f"Hi Robot metrics (toy): IA {hr['instruction_accuracy']:.2f}, TP {hr['task_progress']:.2f} over {list(hr['per_task'])}")
    print("\nscores are scripted; only the loop, the schedule, the bookkeeping and the aggregation are real")


if __name__ == "__main__":
    main()
