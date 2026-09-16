"""Generate figs/pipeline.png: one post-training step (top: sample -> three-segment sequence -> Fig. 18 mask -> one
forward with two outputs -> CE + alpha MSE -> update) and, underneath, which part of that pipeline pre-training
(alpha = 0) and openpi's flow-only fine-tuning use, plus where every parameter group comes from. Tiny run, real values.
Run: uv run python pi/pi05/train/figs/make_pipeline.py
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
from pi.fast.tokenizer.tokenizer import make_smooth_chunks  # noqa: E402
from pi.pi0.data.data import make_bool_mask  # noqa: E402
from pi.pi0.flow_matching.train import interpolate, sample_timestep  # noqa: E402
from pi.pi0.train.train import EMA, make_optimizer  # noqa: E402
from pi.pi05.data.data import LL_IMAGE_KEYS, build_pi05_batch, tiny_pi05_tokenizer, unit_stats  # noqa: E402
from pi.pi05.hier.model import tiny_pi05  # noqa: E402
from pi.pi05.train.train import ALPHA_POSTTRAIN, init_expert_for_posttraining, joint_forward, joint_loss, make_joint_mask, openpi_libero_config, select_trainable, split_prefix_fast, train_step  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
torch.manual_seed(0)
rng = np.random.default_rng(0)
B, H, d = 1, 10, 7
model = tiny_pi05()
model.proj.action_horizon = H
seq = tiny_pi05_tokenizer(H, d)
raw = {"images": {"base_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8), "left_wrist_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8)},
       "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
       "actions": (0.3 * make_smooth_chunks(B, H + 2, d, rng)).astype(np.float32),
       "prompt": ["put the plate in the sink"]}
obs, actions = build_pi05_batch(raw, unit_stats(d), seq, layout="fast", image_keys=LL_IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=True)
pre, post, n_img = split_prefix_fast(obs)
n_pre, n_post = int(pre[0, n_img:].sum()), int(post[0].sum())
init_expert_for_posttraining(model)
t = sample_timestep(B)
noise = torch.randn_like(actions)
x_t, u_t = interpolate(actions, noise, t)
mask = make_joint_mask(pre, post, H)
with torch.no_grad():
    logits, targets, lm, v_t = joint_forward(model, obs, x_t, t)
    out = joint_loss(model, obs, actions, alpha=ALPHA_POSTTRAIN, t=t, noise=noise)
cfg = openpi_libero_config()
params = select_trainable(model)
opt = make_optimizer(params, cfg)
ema = EMA(model, cfg.ema_decay)
stats = train_step(model, (obs, actions, None), params, opt, ema, cfg, 0, alpha=ALPHA_POSTTRAIN)
S = mask.shape[1] - H


def v3(x):
    return " ".join(f"{float(v):+.2f}" for v in np.asarray(x).reshape(-1)[:3])


ENC = [
    ("0 one action sample\n../data 'fast' + actions", f"images x3 (1 masked)\nprefix {n_pre} tok + FAST {n_post} tok\nactions f32[1,{H},32]\n(same chunk, twice:\n as ids and as floats)", "#e8e8e8"),
    ("1 noise the chunk\nflow_matching/train", f"t ~ Beta(1.5,1)*.999+.001\n  = {float(t[0]):.3f}\nx_t = t*noise + (1-t)*a\nu_t = noise - a\nx_t[0,0] {v3(x_t[0,0])}", "#e8e8e8"),
    ("2 three segments", f"expert 0: [img {n_img} | text 200]\n  ({n_pre} prefix, {n_post} FAST)\nexpert 1: {H} action tokens\n  action_in_proj(x_t)\ncond = time MLP(t)", "#cfe2f3"),
    ("3 make_joint_mask\nFig. 18", f"bool[1,{S}+{H},{S}+{H}]\nprefix<->prefix, FAST->prefix,\nFAST causal, expert->prefix,\nexpert<->expert; expert never\nsees FAST, FAST never sees\nexpert; pads all False", "#cfe2f3"),
    ("4 one forward, 2 outputs\njoint_forward", f"llm([tokens, actions], mask,\n cond=[None, cond])\ntext logits [1,199,257152]\n (positions 0..198 -> 1..199)\nv_t f32[1,{H},32]\n {v3(v_t[0,0])}", "#cfe2f3"),
    ("5 Eq. 1\njoint_loss", f"CE on FAST postfix\n ({int(lm[0].sum())} positions): {float(out['ce'][0]):.2f}\nMSE (v_t - u_t)^2 mean:\n {float(out['mse'][0]):.3f}\nloss = CE + 10 * MSE\n = {float(out['loss'].detach()):.2f}", "#f9cb9c"),
    ("6 update\npi0/train", f"backward, clip 1.0\nAdamW lr {stats['lr']:.1e}\n (openpi pi05_libero:\n warmup 10k -> 5e-5)\nEMA 0.999\ngrad norm {stats['grad_norm']:.1f}", "#d9ead3"),
]
STAGES = [
    ("pre-training (280k, alpha = 0)", "uses 0, 2 (expert 0 only), 3 (no expert block), 4 (text logits), 5 (CE only), 6\nno expert weights exist; every action is FAST tokens; HL / WD samples use the 'text' layout", "#fff2cc"),
    ("post-training (80k, alpha = 10)", "the whole row; expert 1 + proj re-initialised (init_expert_for_posttraining), modulation 0 -> identity expert at step 0\ndata: MM + ME (filtered) as fast+flow, WD / HL(ME) / VI as text", "#cfe2f3"),
    ("openpi flow-only finetune", "uses 0 ('flow' layout, no FAST postfix), 1, 2, 3 (no FAST block), 4 (v_t only), 5 (MSE only), 6\npi05_libero: 30k, batch 256; pi05_full_droid_finetune: 100k, batch 256", "#d9ead3"),
]

fig, ax = plt.subplots(figsize=(24, 9.2))
ax.set_xlim(0, 24)
ax.set_ylim(0, 9.2)
ax.axis("off")
BOX_W, BOX_H, GAP, X0 = 3.1, 2.6, 0.26, 0.3
Y_ENC = 5.6


def box(x, y, w, h, title, body, color, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=fs, family="monospace")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1))


for i, (tt, b, c) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, BOX_H, tt, b, c)
    if i:
        arrow(x - GAP, Y_ENC + BOX_H / 2, x, Y_ENC + BOX_H / 2)
W_ST = 7 * (BOX_W + GAP) - GAP
for j, (tt, b, c) in enumerate(STAGES):
    y = 4.1 - j * 1.25
    ax.add_patch(FancyBboxPatch((X0, y), W_ST, 1.05, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="#444", lw=1))
    ax.text(X0 + 0.12, y + 0.92, tt, ha="left", va="top", fontsize=8.8, weight="bold")
    ax.text(X0 + 0.12, y + 0.58, b, ha="left", va="top", fontsize=7.4, family="monospace")
ax.text(X0, Y_ENC + BOX_H + 0.5, f"pi0.5 post-training, one step (tiny: width 64 / 32, 4 layers, {d}-dim H = {H} chunk; paper: 19-dim, H = 50, alpha = 10, 80k steps after 280k discrete steps)",
        fontsize=10.5, weight="bold", va="bottom")
ax.text(X0, 0.55, "blue = pi0.5 increment (paper Sec. IV-B, IV-D, Appendix E / Fig. 18; no upstream code), grey = pi0 / FAST reused, orange = the loss, green = optimizer (pi0/train).  "
        "Parameter provenance at post-training: SigLIP + embedder + expert 0 from the 280k pre-training (which started from PaliGemma); expert 1 + proj random; modulation zero.",
        fontsize=8.5, va="top")
ax.text(X0, 0.15, "Must agree: (i) the FAST postfix and the continuous actions encode the same normalized delta chunk;  (ii) the expert positions skip the FAST tokens so train == inference;  "
        "(iii) alpha = 0 must run no expert at all (pre-training has none);  (iv) the paper's optimizer is undisclosed: the lr here is openpi's fine-tuning value.", fontsize=8.5, va="top")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "loss", float(out["loss"].detach()), "n_pre", n_pre, "n_post", n_post)
