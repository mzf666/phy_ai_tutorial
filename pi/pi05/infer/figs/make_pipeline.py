"""Generate figs/pipeline.png: one Pi05Policy.infer call from the mobile manipulator's raw observation to executable
targets (top), the mirrored inverse transforms and the two-level episode loop (bottom). Tiny config, real values.
Run: uv run python pi/pi05/infer/figs/make_pipeline.py
"""

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
from pi.fast.tokenizer.tokenizer import unnormalize_quantile  # noqa: E402
from pi.pi0.data.data import to_absolute_actions  # noqa: E402
from pi.pi05.data.data import ACTION_DIM, LL_IMAGE_KEYS, build_pi05_batch  # noqa: E402
from pi.pi05.infer.model import MOBILE_CAMERAS, mobile_obs_to_raw, tiny_pi05_policy  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
rng = np.random.default_rng(0)
policy, robot = tiny_pi05_policy(19)
obs = {name: rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for name in MOBILE_CAMERAS}
obs["state"] = rng.uniform(-0.5, 0.5, 19).astype(np.float32)
raw = mobile_obs_to_raw(obs, "pick up the plate", hl=False)
noise = torch.randn(1, robot.action_horizon, ACTION_DIM)
out = policy.infer(raw, noise=noise)
o, _ = build_pi05_batch(raw, robot.norm_stats, policy.seq, layout="flow", image_keys=LL_IMAGE_KEYS, action_horizon=robot.action_horizon, delta_mask=robot.delta_mask, train=False)
prompt = policy.seq.text.decode(o.tokenized_prompt[0].tolist())
x0 = out["x_0"].numpy()
a1 = unnormalize_quantile(x0, robot.norm_stats["actions"])
a2 = to_absolute_actions(raw["state"], a1, robot.delta_mask)
tm = out["timing"]


def v3(x):
    return " ".join(f"{float(v):+.2f}" for v in np.asarray(x).reshape(-1)[:3])


ENC = [
    ("0 robot observation\nSec. IV-E", f"front / rear / 2 wrists\n uint8[96,128,3] each\nstate f32[19]: {v3(obs['state'])}\n = 2x(6 joints+gripper),\n base vx vy w, lift z x\nprompt 'pick up the plate'", "#e8e8e8"),
    ("1 mobile_obs_to_raw(hl=False)", f"drop the rear camera\nimages: base_0,\n left_wrist_0, right_wrist_0\n uint8[1,96,128,3]\nstate f32[1,19]", "#cfe2f3"),
    ("2 ../data 'flow' layout\nquantile norm, state bins", f"q01->-1, q99->+1 (L187)\nprompt: '{prompt[:20]}\n {prompt[20:44]}\n ...;\\nAction: '\n{int(o.tokenized_prompt_mask.sum())} tokens; 3 x 256 image\nmask [T, T, T]\n{tm['data preprocessing']:.0f} ms", "#e8e8e8"),
    ("3 SigLIP x 3 + prefix\n(expert 0 once)", f"768 + 200 = 968 tokens\nkv cache 4 layers\nimage enc {tm['image encoders']:.0f} ms\nprefix {tm['observation forward pass']:.0f} ms\n(4090, pi0: 14 + 32 ms)", "#e8e8e8"),
    ("4 10 Euler steps\n../expert (adaRMSNorm)", f"noise f32[1,50,32]\n-> x_0 f32[1,50,32]\n{v3(x0[0,0])} ...\n{tm['x10 action forward pass (flow)']:.0f} ms (4090, pi0: 27 ms)", "#e8e8e8"),
    ("5 inverse transforms\ntransforms.py L175-L245", f"unnormalize_quantile\n-> to_absolute_actions\n (joints, lift += state;\n grippers, base v as is)\n-> [..., :19]\n{tm['inverse transforms']:.0f} ms", "#cfe2f3"),
    ("6 executable targets", f"f32[1,50,19]\nrow 0 joints {v3(out['actions'][0,0])}\n base v {v3(out['actions'][0,0,14:17])}\n-> PD controllers @ 50 Hz\n(Sec. IV-E, no planner)\ntotal {tm['total']:.0f} ms", "#d9ead3"),
]
DEC = [
    ("6' chunk in physical units", f"[50,19]; execute the first\nk rows (k undisclosed),\nthen observe again", "#d9ead3"),
    ("5' absolute <- delta", f"delta dims: a - state\n{v3(a2[0,0,:19] - raw['state'][0])} ...\n(what training targets are)", "#cfe2f3"),
    ("4' normalized <- physical", f"(a - q01)/(q99 - q01)*2 - 1\n= x_0 up to eps: max|d|\n{np.abs((a1 - x0)[..., :19]).max():.1e}", "#cfe2f3"),
    ("loop: ../hier + eval.py", "every 1 s / user message:\n HL (4 cameras) -> subtask\nevery k steps: LL chunk\nepisode -> rubric points\n(Appendix B), LF check (C),\nIA / TP (Hi Robot)", "#f9cb9c"),
]

fig, ax = plt.subplots(figsize=(24, 8.6))
ax.set_xlim(0, 24)
ax.set_ylim(0, 8.6)
ax.axis("off")
BOX_W, BOX_H, DEC_H, GAP, X0 = 3.1, 2.5, 2.0, 0.26, 0.3
Y_ENC, Y_DEC = 4.9, 1.9


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=7.2, family="monospace")


def arrow(x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1, **kw))


for i, (t, b, c) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_ENC + BOX_H / 2, x, Y_ENC + BOX_H / 2)
x_last = X0 + (len(ENC) - 1) * (BOX_W + GAP)
for i, (t, b, c) in enumerate(DEC):
    x = x_last - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, DEC_H, t, b, c)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + DEC_H / 2, x + BOX_W, Y_DEC + DEC_H / 2)
arrow(x_last + BOX_W / 2, Y_ENC, x_last + BOX_W / 2, Y_DEC + DEC_H)
ax.text(x_last + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + DEC_H) / 2, "inverse", fontsize=8, va="center")
ax.text(X0, Y_ENC + BOX_H + 0.5, "pi0.5 inference on the 19-dim mobile manipulator (tiny config, CPU): low level = openpi's pi05 path; the two-level loop wraps it (../hier)",
        fontsize=10.5, weight="bold", va="bottom")
ax.text(X0, Y_DEC - 0.35, "blue = pi0.5 / robot-side increment, grey = ../data, ../hier, pi0 reused, orange = the evaluation loop, green = contract.  "
        "Line refs: openpi@215abfb transforms.py / config.py; Hi Robot App. B.3 for the RTX 4090 numbers.", fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) the same q01 / q99 make the state bins, normalize the actions and undo them;  (ii) to_absolute_actions uses the state the model saw, not a later one;  "
        "(iii) base VELOCITY dims are absolute, never added to a state;  (iv) the executed length k and the 1-s high-level period set the replanning rhythm (README Sec. 8).",
        fontsize=8.5, va="top")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, {k: round(v) for k, v in tm.items()})
