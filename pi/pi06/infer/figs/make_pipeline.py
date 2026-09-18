"""Generate figs/pipeline.png: one control step of pi0.6* end to end (top row: robot inputs -> subtask at the low rate
-> flow sequence with 'Advantage: positive' -> optional CFG second prefix -> 5 Euler steps -> 14-dim targets) and the
evaluation / data loop as the mirrored bottom row (episode -> time limit -> raters -> throughput / success -> RECAP
labels). Numbers from one real tiny run.
Run: uv run python pi/pi06/infer/figs/make_pipeline.py
"""

import dataclasses
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.pi06.backbone.model import AdaRMSNorm  # noqa: E402
from pi.pi06.data.data import ACTION_DIM  # noqa: E402
from pi.pi06.infer.eval import StaticToyEnv, max_episode_len, run_episode, task  # noqa: E402
from pi.pi06.infer.model import CFG_BETA_RANGE, CONTROL_HZ, LATENCY_H100_MS, STATIC_CAMERAS, STATIC_DIMS_14, static_obs_to_raw, tiny_pi06_policy  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
rng = np.random.default_rng(0)
policy, robot = tiny_pi06_policy(max_new_tokens=6)
for m in policy.model.modules():
    if isinstance(m, AdaRMSNorm):
        torch.nn.init.normal_(m.modulation.weight, std=0.05)
obs = {name: rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for name in STATIC_CAMERAS}
obs["state"] = rng.uniform(-0.5, 0.5, 14).astype(np.float32)
raw = static_obs_to_raw(obs, "make me an espresso")
noise = torch.randn(1, robot.action_horizon, ACTION_DIM)
out = policy.infer(raw, 0.0, noise)
policy.beta = 2.0
out_cfg = policy.infer(raw, 0.0, noise)
policy.beta = 1.0
env = StaticToyEnv()
t_box = dataclasses.replace(task("box assembly"), time_limit_s=1.0)
rec = run_episode(policy, env, t_box, 0, execute_steps=25)
lab = rec.to_labels(max_episode_len(t_box))
tm = out["timing"]

ENC = [
    ("0 robot inputs (Fig. 5)", f"3 cameras: base, left_wrist,\n  right_wrist uint8[96,128,3]\nstate f32[14]: 2 x (6 joints\n  + gripper), {CONTROL_HZ} Hz\nprompt 'make me an espresso'\n  (+ metadata, optional)", "#e8e8e8"),
    ("1 subtask, at the low rate\nSec. V-A", f"every {policy.hl_period_s} s (Hi Robot rule;\n  pi0.6 rate undisclosed)\nlayout hl_prompt -> greedy\n  until '\\n' / EOS\nran now: {out['hl_ran']}, {out['timing'].get('subtask decode (1 tokens)', tm[next(k for k in tm if 'subtask' in k)]):.0f} ms tiny\n-> ell_hat = '{out['subtask']}' (untrained)", "#cfe2f3"),
    ("2 flow sequence (../data)\nSec. V-B, V-D", "'Task: ..., State: ...;\\n'\n+ 'Subtask: ell_hat\\n'\n+ 'Advantage: positive\\n'\n  (deployment: I = True\n   <=> beta = 1, Eq. 2)\n+ 'Action: '", "#cfe2f3"),
    ("3 prefix cache(s)\n(../backbone)", f"expert 0 once: {tm['observation forward pass']:.0f} ms tiny\nbeta > 1 (CFG, App. E):\n  a 2nd sequence with the\n  Advantage line omitted\n  -> 2nd cache ({out_cfg['timing']['observation forward pass x2 (CFG)']:.0f} ms)", "#cfe2f3"),
    ("4 5 Euler steps\ncard Sec. 2; App. E Eq. 13", f"v = v_u + beta (v_c - v_u)\nbeta = 1: v_c only\nbeta in {list(CFG_BETA_RANGE)} 'where useful'\ntiny: beta 2 vs 1 rms diff\n  {float((out_cfg['x_0'] - out['x_0']).pow(2).mean().sqrt()):.3f}\nx_0 f32{tuple(out['x_0'].shape)}", "#cfe2f3"),
    ("5 inverse transforms\n(../pi05/infer)", f"quantile inverse, joints +=\n  state, keep 14 dims\nactions f32{tuple(out['actions'].shape)}\n= 1.0 s of targets @ 50 Hz\nexecute k rows, re-plan\n  (k undisclosed)", "#e8e8e8"),
    ("6 latency", f"card: {LATENCY_H100_MS:.0f} ms / chunk on one\n  H100, 5 steps, 3 cameras\n  (breakdown undisclosed)\ntiny CPU total {tm['total']:.0f} ms\n  ({tm['x5 action forward pass (flow)']:.0f} ms for the 5 steps)", "#d9ead3"),
]
DEC = [
    ("6' RECAP data (Alg. 1 l. 7)", f"EpisodeRecord -> EpisodeLabels\n  success {lab.success}, steps {lab.num_steps},\n  T_max {lab.max_episode_len} (= limit x Hz)\n-> ../value targets, ../train", "#d9ead3"),
    ("5' metrics, Sec. VI-C", "throughput = successes / hour\n  (failures' time counts)\nsuccess rate = raters' label\nerror bars: standard error\nbox: 4-stage breakdown", "#fce5cd"),
    ("4' raters", "quality indicators -> success\n  (aggregation undisclosed;\n   all required here)\ntoy: " + "\n  ".join(f"{k}: {v}" for k, v in rec.quality.items()), "#fce5cd"),
    ("3' time limit (Sec. VI-A)", f"200 s laundry / espresso /\n  strict T-shirt, 500 s diverse,\n  600 s box\ntoy box: limit 1 s = 50 steps,\n  ran {rec.steps} steps, timed out {rec.timed_out}", "#fce5cd"),
    ("2' episode loop", f"reset -> infer every k steps\n  (subtask at its own rate)\n-> env.step at 50 Hz\ntoy: {rec.infer_calls} infer calls,\n  {rec.hl_calls} subtask decode(s)", "#e8e8e8"),
]

fig, ax = plt.subplots(figsize=(28, 9.4))
ax.set_xlim(0, 28)
ax.set_ylim(0, 9.4)
ax.axis("off")
BOX_W, GAP, X0 = 3.55, 0.24, 0.3
Y_ENC, H_ENC = 5.5, 2.75
Y_DEC, H_DEC = 2.2, 2.2


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=7.1, family="monospace")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1))


for i, (t_, b_, c_) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, H_ENC, t_, b_, c_)
    if i:
        arrow(x - GAP, Y_ENC + H_ENC / 2, x, Y_ENC + H_ENC / 2)
x_last = X0 + (len(ENC) - 1) * (BOX_W + GAP)
for i, (t_, b_, c_) in enumerate(DEC):
    x = x_last - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, H_DEC, t_, b_, c_)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + H_DEC / 2, x + BOX_W, Y_DEC + H_DEC / 2)
arrow(x_last + BOX_W / 2, Y_ENC, x_last + BOX_W / 2, Y_DEC + H_DEC)
ax.text(x_last + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + H_DEC) / 2, "deploy,\nevaluate,\ncollect", fontsize=8, va="center")
ax.text(X0, Y_DEC - 0.35, "blue = pi0.6* inference increment (paper Sec. V-A / V-B / V-D, App. E; card Sec. 2), orange = evaluation protocol (Sec. VI-A / VI-C), grey = reused (pi0.5 inverse chain, Hi Robot schedule), green = hand-off.  "
        "No upstream code exists.", fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) the deployed sequence carries 'Advantage: positive' (I = True); the CFG branch is the SAME prefix minus that line;  (ii) beta = 1 must reproduce the conditional model exactly;  "
        "(iii) T_max used for value targets = the task's time limit x 50 Hz (this repo);  (iv) every evaluation episode becomes RECAP data with its human label and correction flags.", fontsize=8.5, va="top")
ax.text(X0, Y_ENC + H_ENC + 0.55, f"pi0.6* inference and evaluation: subtask -> Advantage token -> (CFG) -> 5 Euler steps -> 14-dim targets @ 50 Hz; throughput / success / stages -> RECAP data (tiny model, "
        f"{len(STATIC_DIMS_14)}-dim static bimanual spec)", fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "timing", {k: round(v, 1) for k, v in tm.items()}, "toy episode", rec.steps, rec.timed_out)
