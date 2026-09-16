"""Generate figs/eval.png: (a) the replanning rhythm of one toy episode (high level every 1 s or on a message, low level
every k executed steps of a 50-step chunk), from a real run of run_hier_episode; (b) the rubric structure of the four
mock-home tasks and how trials aggregate to the reported task-progress number.
Run: uv run python pi/pi05/infer/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.pi05.infer.eval import MOCK_HOME_TASKS, TRIALS_PER_TASK, MockHomeToyEnv, run_hier_episode  # noqa: E402
from pi.pi05.infer.model import CONTROL_HZ, tiny_pi05_policy  # noqa: E402

OUT = pathlib.Path(__file__).with_name("eval.png")
policy, robot = tiny_pi05_policy(19, hier=True, max_new_tokens=3)
env = MockHomeToyEnv(seed=3)
T, k = 150, 25
calls = []  # (step, hl_ran)
orig_step = policy.step


def logged_step(raw_hl, raw_ll, t_now, user_message=None, noise=None, generator=None):
    out = orig_step(raw_hl, raw_ll, t_now, user_message, noise, generator)
    calls.append((int(round(t_now * CONTROL_HZ)), out["hl_ran"], user_message))
    return out


policy.step = logged_step
env.need = 10**6
orig_reset = env.reset
env.reset = lambda task, i: (orig_reset(task, i), setattr(env, "need", 10**6))[0]
ep = run_hier_episode(policy, env, "Dishes in Sink", 0, max_steps=T, execute_steps=k, user_messages={60: "not that plate"})

fig, (a, b) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [1, 1.2]})
a.set_xlim(-2, T + 2)
a.set_ylim(-0.5, 3.4)
for i, (step, hl, msg) in enumerate(calls):
    a.plot([step, step], [0, 0.4], color="#3d85c6", lw=2)
    y = 0.5 + 0.28 * (i % 3)  # stagger overlapping chunks
    a.add_patch(plt.Rectangle((step, y), robot.action_horizon, 0.22, color="#cfe2f3", ec="#3d85c6", lw=0.8))
    a.add_patch(plt.Rectangle((step, y), k, 0.22, color="#3d85c6"))
    if hl:
        a.plot([step, step], [1.45, 1.85], color="#b45f06", lw=3)
    if msg:
        a.annotate(f"user: '{msg}'", (step, 1.4), (step + 4, 2.7), fontsize=8, color="#990000", arrowprops={"arrowstyle": "->", "color": "#990000"})
for s in range(0, T + 1, int(CONTROL_HZ)):
    a.axvline(s, color="#bbb", ls=":", lw=0.8)
    a.text(s, 3.0, f"{s // int(CONTROL_HZ)} s", fontsize=7.5, ha="center", color="#777")
a.set_yticks([0.2, 0.9, 1.6], [f"low-level call", f"chunk 50 (light) /\nexecuted k = {k} (dark)", "high-level call"], fontsize=8)
a.set_xlabel(f"control step (50 Hz); episode of {ep.steps} steps: {ep.ll_calls} low-level calls, {ep.hl_calls} high-level calls")
a.set_title("(a) replanning rhythm of run_hier_episode (toy env; k = 25 is a placeholder, README Sec. 8; high level every 1 s or on a message, Hi Robot Sec. 4.1-4.2)", fontsize=9)

names = [t for t, *_ in MOCK_HOME_TASKS]
maxp = [m for _, _, m, _ in MOCK_HOME_TASKS]
rng = np.random.default_rng(0)
pts = [rng.integers(0, m + 1, TRIALS_PER_TASK) for m in maxp]  # illustrative random points, NOT results
per_task = [p.mean() / m for p, m in zip(pts, maxp)]
x = np.arange(len(names))
for i, (p, m) in enumerate(zip(pts, maxp)):
    b.scatter(np.full(TRIALS_PER_TASK, i) + rng.uniform(-0.18, 0.18, TRIALS_PER_TASK), p / m, s=18, color="#999", label="one trial: points / max" if i == 0 else None)
b.bar(x, per_task, width=0.5, color="#cfe2f3", edgecolor="#3d85c6", label="per task: mean over 10 trials")
b.axhline(np.mean(per_task), color="#b45f06", ls="--", label=f"task progress = mean over tasks ({np.mean(per_task):.2f})")
b.set_xticks(x, [f"{n}\n{m} points" for n, m in zip(names, maxp)], fontsize=8)
b.set_ylim(0, 1.05)
b.set_ylabel("fraction of rubric points")
b.set_title("(b) Appendix B aggregation: points / max per trial -> mean per task (10 trials) -> mean over the 4 tasks (illustrative random points, not results)", fontsize=9)
b.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT, "calls", calls)
