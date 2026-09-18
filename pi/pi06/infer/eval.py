"""pi0.6* evaluation flow: the five tasks with their time limits and success criteria (Sec. VI-A), the two metrics
throughput (successes per hour) and success rate from human ratings (Sec. VI-C), the box-assembly stage breakdown
(Fig. 8 right), the episode loop that enforces the time limit, and the hand-off of every evaluation episode to the
RECAP dataset (Algorithm 1 line 7: evaluation rollouts are training data). Numbers are not reproduced; the procedure is.

Sources of truth:
  paper    pi0.6* arXiv:2511.14759v2 Sec. VI-A (tasks, limits, success definitions), VI-B (baselines), VI-C (throughput =
           successful executions per hour; success from human raters aggregating several quality indicators; box stages:
           pick up a box sheet / build the box / label it / place it in a crate; error bars = standard error),
           Sec. VI-C.2 (iterations: 300 trajectories on four robots per iteration; 600 autonomous + 360 intervention
           trials per iteration for box assembly), Sec. VI-C.4 (600 trajectories per iteration, 97% on strict T-shirt),
           Appendix F "Dataset composition", Algorithm 1 line 7
  openpi   no evaluation code upstream; Env / EpisodeResult are pi.pi0.infer.eval
License: this file is the repo's own.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pi.pi0.infer.eval import Env  # noqa: F401  (interface re-exported)
from pi.pi06.data.data import EpisodeLabels
from pi.pi06.infer.model import CONTROL_HZ, STATIC_CAMERAS, STATIC_DIMS_14, Pi06Policy, static_obs_to_raw


# ---- Sec. VI-A: task table. Prompts are this repo's wording (undisclosed except "make me an espresso", Sec. V-A; README Sec. 8).
@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    time_limit_s: float
    success: str  # the human-judged criterion
    stages: tuple[str, ...] = ()  # box assembly only (Fig. 8 right)
    notes: str = ""


TASKS = (
    Task("laundry (t-shirts and shorts)", "fold the laundry", 200, "one item folded and stacked in the top-right corner of the table", notes="pi0 laundry task; item from a basket, variable initial conditions"),
    Task("laundry (diverse items)", "fold the laundry", 500, "the target item correctly folded and placed on a stack", notes="11 item types; metric reported on the button-up shirt"),
    Task("laundry (targeted failure removal)", "fold the shirt", 200, "folded correctly with the collar centred and facing up", notes="one orange T-shirt, fixed flattened (adversarial) initial condition; strict"),
    Task("cafe (double shot espresso)", "make me an espresso", 200, "all steps completed without critical mistakes (dropping the portafilter, spilling)",
         notes="pick up portafilter, grind, tamp, lock in, bring cup, extract, serve"),
    Task("box assembly", "assemble the box", 600, "from a flattened sheet to an assembled, labelled box stacked in the crate",
         stages=("pick up a box sheet", "build the box", "label the box", "place it in the crate"), notes="real factory deployment"),
)
DIVERSE_LAUNDRY_ITEMS = ("towels", "button-up shirts", "sweaters", "jeans", "T-shirts", "shorts", "polos", "skirts", "long sleeve shirts", "socks", "underwear")  # Sec. VI-A, 11 types
LONG_RUNS = {"espresso": "13 hours straight", "laundry in a new home": "over two hours without interruptions"}  # Sec. I
# ---- Sec. VI-C.2 / VI-C.4 / App. F: data collected per iteration (also the evaluation rollouts)
EPISODES_PER_ITERATION = {
    "laundry (t-shirts and shorts)": {"autonomous": 300, "corrections": 0, "robots": 4},
    "laundry (diverse items)": {"autonomous": 450, "corrections": 287, "robots": None},
    "laundry (targeted failure removal)": {"autonomous": 600, "corrections": 0, "robots": 3, "note": "~1000 autonomous + 280 + 378 correction episodes in total (App. F)"},
    "box assembly": {"autonomous": 600, "corrections": 360, "robots": 3},
    "cafe (double shot espresso)": {"autonomous": 414, "corrections": 429, "robots": None, "note": "single iteration"},
}


def task(name: str) -> Task:
    return next(t for t in TASKS if t.name == name)


# ======================================================================================
# 1. Metrics. Sec. VI-C: throughput = successful executions per hour (captures speed AND success); success rate = fraction
#    of episodes the raters mark successful, aggregated from several quality indicators (rule undisclosed: AND here).
# ======================================================================================
@dataclasses.dataclass
class EpisodeRecord:
    task: str
    episode_idx: int
    steps: int  # control steps at CONTROL_HZ
    duration_s: float
    timed_out: bool
    quality: dict[str, bool]  # the raters' quality indicators
    success: bool
    stages_done: int  # box assembly: how many of the 4 stages completed
    infer_calls: int
    hl_calls: int
    subtasks: list[str]
    is_correction: np.ndarray | None = None  # bool[steps] if an expert intervened (Sec. V-D)

    def to_labels(self, max_episode_len: int) -> EpisodeLabels:
        """Algorithm 1 line 7 / Sec. V-D: the episode joins D_ell with its outcome label and correction flags."""
        return EpisodeLabels(self.task, self.success, max_episode_len, self.steps, self.is_correction)


def success_from_quality(quality: dict[str, bool]) -> bool:
    """Sec. VI-C: "Raters are asked to judge the episode with respect to multiple quality metrics, and we aggregate these
    quality indicators into a success label." The aggregation rule is undisclosed; this repo requires all of them."""
    return bool(quality) and all(quality.values())


def throughput_per_hour(records: list[EpisodeRecord]) -> float:
    """Successful executions per hour of robot time, failures' time included (that is what makes speed count)."""
    hours = sum(r.duration_s for r in records) / 3600.0
    return sum(r.success for r in records) / hours if hours > 0 else 0.0


def success_rate(records: list[EpisodeRecord]) -> float:
    return float(np.mean([r.success for r in records])) if records else 0.0


def standard_error(values) -> float:
    """Error bars in Fig. 7-12 are standard errors."""
    v = np.asarray(values, np.float64)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def stage_success(records: list[EpisodeRecord], n_stages: int = 4) -> list[float]:
    """Fig. 8 right: per-stage success for box assembly = fraction of episodes that completed at least stage k."""
    return [float(np.mean([r.stages_done >= k for r in records])) for k in range(1, n_stages + 1)]


def max_episode_len(t: Task, hz: float = CONTROL_HZ) -> int:
    """T_max for the value normalisation (Sec. V-C 'maximum episode length of the task'): this repo uses the time limit
    (README Sec. 8: whether the paper uses the limit or the longest observed episode is undisclosed)."""
    return int(round(t.time_limit_s * hz))


# ======================================================================================
# 2. The episode loop: chunked execution, subtask at its own rate, the time limit, and the raters' verdict.
# ======================================================================================
class RatedEnv(Env):
    """Beyond pi.pi0.infer.eval.Env: the raw observation is the static one ({"base", "left_wrist", "right_wrist", "state"});
    quality() after the episode returns the raters' indicators; stages() the number of completed stages (box)."""

    def quality(self) -> dict[str, bool]: ...

    def stages(self) -> int: ...


def run_episode(policy: Pi06Policy, env, t: Task, episode_idx: int, *, execute_steps: int, hz: float = CONTROL_HZ, noise_rng=None) -> EpisodeRecord:
    """Execute `execute_steps` of every H-step chunk (undisclosed; README Sec. 8), stop at done or the task's time limit."""
    obs, prompt = env.reset(t.name, episode_idx)
    policy.reset()
    plan: list[np.ndarray] = []
    max_steps = max_episode_len(t, hz)
    k, done, calls, hl_calls, subtasks = 0, False, 0, 0, []
    while k < max_steps and not done:
        if not plan:
            out = policy.infer(static_obs_to_raw(obs, prompt), t_now=k / hz)
            plan = list(out["actions"][0][:execute_steps])
            calls += 1
            if out["hl_ran"]:
                hl_calls += 1
                subtasks.append(out["subtask"])
        obs, done, _ = env.step(plan.pop(0))
        k += 1
    quality = env.quality() if done else {"finished within the time limit": False}
    return EpisodeRecord(t.name, episode_idx, k, k / hz, not done, quality, success_from_quality(quality), env.stages(), calls, hl_calls, subtasks)


def evaluate(policy: Pi06Policy, env, tasks=TASKS, *, trials: int, execute_steps: int, hz: float = CONTROL_HZ) -> dict:
    """Per task: throughput (successes / hour), success rate, standard error over trials, box stage breakdown; every record
    is also returned as RECAP training data (to_labels)."""
    out = {}
    for t in tasks:
        recs = [run_episode(policy, env, t, i, execute_steps=execute_steps, hz=hz) for i in range(trials)]
        entry = {"throughput_per_hour": throughput_per_hour(recs), "success_rate": success_rate(recs), "success_se": standard_error([r.success for r in recs]),
                 "mean_duration_s": float(np.mean([r.duration_s for r in recs])), "records": recs, "labels": [r.to_labels(max_episode_len(t, hz)) for r in recs]}
        if t.stages:
            entry["stage_success"] = stage_success(recs, len(t.stages))
        out[t.name] = entry
    return out


# ======================================================================================
# 3. A stand-in environment. Not a simulator: images are noise; a scripted rule finishes the task after `need`
#    consecutive steps with the left-gripper target above 0.1, stage by stage for box assembly.
# ======================================================================================
class StaticToyEnv:
    def __init__(self, image_hw=(96, 128), seed: int = 0):
        self.hw, self.rng = image_hw, np.random.default_rng(seed)

    def _obs(self):
        o = {name: self.rng.integers(0, 256, (*self.hw, 3), dtype=np.uint8) for name in STATIC_CAMERAS}
        o["state"] = self.rng.uniform(-0.5, 0.5, len(STATIC_DIMS_14)).astype(np.float32)
        return o

    def reset(self, name, episode_idx):
        self.t = task(name)
        self.n_stages = max(len(self.t.stages), 1)
        self.done_stages, self.hold, self.need = 0, 0, 3 + episode_idx % 3
        return self._obs(), self.t.prompt

    def step(self, action):
        assert action.shape == (len(STATIC_DIMS_14),), action.shape
        self.hold = self.hold + 1 if action[6] > 0.1 else 0  # left gripper target
        if self.hold >= self.need and self.done_stages < self.n_stages:
            self.done_stages, self.hold = self.done_stages + 1, 0
        return self._obs(), self.done_stages >= self.n_stages, {"stages": self.done_stages}

    def quality(self):  # scripted raters: two indicators, the second fails every third episode
        return {"task completed": self.done_stages >= self.n_stages, "no critical mistake": self.rng.random() > 0.33}

    def stages(self):
        return self.done_stages


# ======================================================================================
#   uv run python -m pi.pi06.infer.eval
# ======================================================================================
def main() -> None:
    from pi.pi06.infer.model import tiny_pi06_policy

    policy, _ = tiny_pi06_policy(max_new_tokens=4)
    env = StaticToyEnv()
    print("tasks (Sec. VI-A):")
    for t in TASKS:
        print(f"  {t.name:36s} limit {t.time_limit_s:4.0f} s = {max_episode_len(t)} steps @ {CONTROL_HZ} Hz; success: {t.success}" + (f"; stages {t.stages}" if t.stages else ""))
    print("metrics (Sec. VI-C): throughput = successes / hour (failures' time counts); success = raters' quality indicators aggregated (rule undisclosed; all required here)")
    hz = 50
    rec = run_episode(policy, env, task("box assembly"), 0, execute_steps=25, hz=hz)
    print(f"\none box episode (tiny, hz {hz}, execute 25 of each 50-step chunk): {rec.steps} steps = {rec.duration_s:.2f} s, {rec.infer_calls} infer calls, {rec.hl_calls} subtask decodes, "
          f"stages {rec.stages_done}/4, timed out {rec.timed_out}, quality {rec.quality} -> success {rec.success}")
    lab = rec.to_labels(max_episode_len(task("box assembly"), hz))
    print(f"-> RECAP data: EpisodeLabels(success={lab.success}, T_max={lab.max_episode_len}, steps={lab.num_steps}); value targets {[f'{x:.4f}' for x in lab.value_targets(1.0)[0][:3]]} ...")
    small = tuple(dataclasses.replace(t, time_limit_s=2.0) for t in TASKS[:2] + TASKS[4:])  # 2-second limits so the toy loop stays short
    res = evaluate(policy, env, small, trials=4, execute_steps=10, hz=hz)
    print("\ntoy evaluation (numbers are meaningless; the arithmetic is the point):")
    for name, r in res.items():
        line = f"  {name:36s} throughput {r['throughput_per_hour']:7.1f}/h  success {r['success_rate']:.2f} +- {r['success_se']:.2f}  mean duration {r['mean_duration_s']:.2f} s"
        if "stage_success" in r:
            line += f"  stages {[f'{x:.2f}' for x in r['stage_success']]}"
        print(line)
    print(f"\nreference counts (Sec. VI-C.2, App. F): {EPISODES_PER_ITERATION}")


if __name__ == "__main__":
    main()
