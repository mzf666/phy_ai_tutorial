"""Generate figs/pipeline.png: one infer call from a DROID observation to executable actions, step by step (top), and
the robot-side mirror: execution rhythm and scoring (bottom). Values come from one real tiny run; the token -> action
half uses the postfix a trained model would emit (an untrained model emits no 'Action: ' marker -> zeros).
Run: uv run python pi/fast/infer/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.data.data import discretize_state, paligemma_to_fast  # noqa: E402
from pi.fast.infer.eval import DROID_TASKS  # noqa: E402
from pi.fast.infer.model import DROID_ACTION_HORIZON_PAPER, DROID_EXECUTE_STEPS, droid_obs_to_raw, tiny_fast_policy, to_executable_actions  # noqa: E402
from pi.fast.tokenizer.tokenizer import normalize_quantile, unnormalize_quantile  # noqa: E402
from pi.pi0.data.data import to_absolute_actions  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")

rng = np.random.default_rng(0)
policy, seq = tiny_fast_policy(max_decoding_steps=48)
robot = policy.robot
H, d = robot.action_horizon, robot.native_dim
obs = {"observation/exterior_image_1_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
       "observation/wrist_image_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
       "observation/joint_position": rng.uniform(-0.5, 0.5, 7).astype(np.float32),
       "observation/gripper_position": np.float32(0.3)}
raw = droid_obs_to_raw(obs, DROID_TASKS[10][0])
st = normalize_quantile(raw["state"][0], robot.norm_stats["state"])
bins = discretize_state(st)
out = policy.infer(raw)
tm = out["timing"]
# oracle postfix: what a trained model would emit for a smooth chunk
chunk = np.clip(0.3 * rng.standard_normal((H, d)), -0.9, 0.9).astype(np.float32)
toks, m, ar, _ = seq.tokenize(raw["prompt"][0], st, chunk)
postfix = toks[m & (ar == 1)]
fast_ids = paligemma_to_fast(postfix[len(seq._action_marker) : -2])
rec_norm = seq.extract_actions(postfix, H, d)
rec_un = unnormalize_quantile(rec_norm, robot.norm_stats["actions"])
rec_abs = to_absolute_actions(raw["state"], rec_un[None], robot.delta_mask)[0]
acts = to_executable_actions(postfix[None], raw["state"], robot, seq)[0]


def r3(v, n=4):
    return " ".join(f"{float(x):+.2f}" for x in np.asarray(v).ravel()[:n])


ENC = [
    ("0 DROID observation\ndroid_policy.py L10-L18", f"exterior_image_1_left\n  u8[180,320,3]\nwrist_image_left u8[180,320,3]\njoint_position f32[7]\n  {r3(obs['observation/joint_position'], 3)} ..\ngripper_position f32 {float(obs['observation/gripper_position']):.2f}\nprompt '{raw['prompt'][0][:20]}..'", "#e8e8e8"),
    ("1 droid_obs_to_raw\nL35-L74", f"images: base_0_rgb, wrist_0_rgb\n  u8[1,180,320,3] each\n  (base_1_rgb absent -> black,\n   mask True, L53-L56)\nstate f32[1,8] = joints+gripper\nprompt [str]", "#cfe2f3"),
    ("2 build_fast_batch\n(../data) inference mode", f"resize/pad -> f32[1,224,224,3]\n  x 3 slots\nstate quantile {r3(st, 3)} ..\n  bins {bins[:4].tolist()} ..\nprefix i64[1,180], no postfix\n{tm['data preprocessing']:.1f} ms", "#e8e8e8"),
    ("3 sample_actions\n(../model) L236-L313", f"prefill 768 + prefix, then\n{out['n_steps']} decode steps (cap 48)\ngreedy (T=0; bimanual 0.7)\n{tm['sample_actions (prefill + decode steps)']:.0f} ms = {tm['ms per decode step']:.1f} ms/step\n(tiny, CPU; paper 750 ms 4090)", "#e8e8e8"),
    ("4 generated ids\n", f"tokens i64[1,48]\nthis run (untrained):\n  {out['tokens'][0, :5].tolist()} ..\noracle postfix ({len(postfix)} ids):\n  {postfix[:3].tolist()} .. {postfix[-2:].tolist()}\n  'Action: ' {len(fast_ids)} FAST '|' EOS", "#cfe2f3"),
    ("5 extract_actions\n(../data) L119-L134", f"find 'Action: ' .. '|'\n-> paligemma_to_fast\n-> FAST decode\nf32[{H},{d}] normalized\n  {r3(rec_norm[0])} ..\nno marker -> zeros (this run)", "#cfe2f3"),
    ("6 unnormalize_quantile\ntransforms.py L175-L181", f"(x+1)/2 (q99-q01) + q01\nf32[{H},{d}] physical units\n  {r3(rec_un[0])} ..\nq01/q99 from the checkpoint's\nnorm_stats.json", "#cfe2f3"),
    ("7 to_absolute_actions\n(pi0/data) L226-L245", f"delta dims += raw state\n(mask: 7 joints delta,\n gripper absolute)\n  {r3(rec_abs[0])} ..\n{tm['extract + inverse transforms']:.2f} ms for 5-8", "#e8e8e8"),
    ("8 executable actions\nDroidOutputs L81", f"[..., :8] -> f32[1,{H},{d}]\nrow 0 {r3(acts[0], 3)} ..\nmax |err| vs analytic\n  {np.abs(acts - rec_abs).max():.1e} (same path)\nvs true chunk {np.abs(acts - to_absolute_actions(raw['state'], unnormalize_quantile(chunk[None], robot.norm_stats['actions']), robot.delta_mask)[0]).max():.3f}", "#d9ead3"),
]
DEC = [
    ("8' chunk to the robot\npaper App. D", f"predict {DROID_ACTION_HORIZON_PAPER} steps\nexecute {DROID_EXECUTE_STEPS[0]} or {DROID_EXECUTE_STEPS[1]} open-loop\n(LIBERO: predict 10, execute 5)\nno temporal ensembling", "#cfe2f3"),
    ("7' env.step x k\n(pi0/infer) run_episode", "action f32[8] each step\n-> new observation\nk = executed steps, then\nre-observe and infer again\ndone or max_steps -> stop", "#e8e8e8"),
    ("6' scoring\nTable II, App. E", "DROID: rubric points / max\n  (pick 1, place 1, ...)\n  44 trials over 17 rows\n  -> % task progress\nLIBERO: binary success, 50/task", "#cfe2f3"),
]

fig, ax = plt.subplots(figsize=(27.5, 8.2))
ax.set_xlim(0, 27.5)
ax.set_ylim(0, 8.2)
ax.axis("off")
BOX_W, BOX_H, DEC_H, GAP, X0 = 2.75, 2.5, 1.95, 0.25, 0.3
Y_ENC, Y_DEC = 4.7, 1.7


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=9, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=7.2, family="monospace")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1))


for i, (t, b, c) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_ENC + BOX_H / 2, x, Y_ENC + BOX_H / 2)
x8 = X0 + 8 * (BOX_W + GAP)
for i, (t, b, c) in enumerate(DEC):
    x = x8 - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, DEC_H, t, b, c)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + DEC_H / 2, x + BOX_W, Y_DEC + DEC_H / 2)
arrow(x8 + BOX_W / 2, Y_ENC, x8 + BOX_W / 2, Y_DEC + DEC_H)
ax.text(x8 + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + DEC_H) / 2, "to the robot", fontsize=8, va="center")
# loop back from 7' to 0
x6 = x8 - 2 * (BOX_W + GAP)
ax.add_patch(FancyArrowPatch((x6, Y_DEC + DEC_H / 2), (X0 + BOX_W / 2, Y_ENC), arrowstyle="-|>", mutation_scale=12, color="#b45f06", lw=1.2,
                             connectionstyle="arc3,rad=-0.25"))
ax.text((x6 + X0) / 2, Y_DEC + DEC_H + 0.75, "after k executed steps: new observation, infer again (chunked replanning)", fontsize=8.5, color="#b45f06", ha="center")
ax.text(X0, Y_DEC - 0.35, "blue = pi0-FAST increment (droid_policy.py, transforms.py, paper App. D / E), grey = ../data, ../model, pi0/data, pi0/infer reused, green = output.  "
        "Line refs: openpi@215abfb droid_policy.py unless noted.", fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) H and native_dim in FASTRobotSpec = the (H, D) the tokenizer was fit and the model trained with;  (ii) norm_stats are the checkpoint's quantiles, applied to state before binning and to actions after decoding;  "
        "(iii) delta_mask matches the training-time DeltaActions;  (iv) the executed k <= H.", fontsize=8.5, va="top")
ax.text(X0, Y_ENC + BOX_H + 0.45, "pi0-FAST inference: DROID observation -> prefix -> prefill + token-by-token decoding -> FAST ids -> chunk -> absolute actions  "
        f"(tiny config; DROID scale: {d}-dim, {H}-step chunk, max_token_len 180)", fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "n_steps", out["n_steps"], "postfix", len(postfix))
