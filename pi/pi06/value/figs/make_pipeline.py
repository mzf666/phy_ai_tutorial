"""Generate figs/pipeline.png: the value function from observation to V (top row), the advantage -> threshold ->
indicator -> Advantage token chain (middle row), and the training target chain as the mirrored bottom row. Numbers
from one real tiny run (untrained value function, one 12-step success episode).
Run: uv run python pi/pi06/value/figs/make_pipeline.py
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
import pi.pi06.value.train as T  # noqa: E402
from pi.pi06.data.data import C_FAIL_TINY, NUM_BINS, STATIC_IMAGE_KEYS, EpisodeLabels, advantage_text, build_pi06_batch, episode_rewards, tiny_pi06_tokenizer, unit_stats  # noqa: E402
from pi.pi06.value.model import tiny_value_function  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
torch.manual_seed(0)
rng = np.random.default_rng(0)
H, d, T1, Tmax = 10, 7, 12, 40
vf = tiny_value_function().eval()
seq = tiny_pi06_tokenizer(H, d)
lab = EpisodeLabels("fold the shirt", True, Tmax, T1, is_correction=np.array([False] * 8 + [True] * 4))
R, bins = lab.value_targets(C_FAIL_TINY)
r = episode_rewards(T1, True, C_FAIL_TINY) / Tmax
steps = []
for t in range(T1):
    raw = {"images": {k: rng.integers(0, 256, (1, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
           "state": rng.uniform(-0.5, 0.5, (1, d)).astype(np.float32), "prompt": ["fold the shirt"]}
    obs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="value", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, value_bins=[int(bins[t])], metadata=["speed: fast"])
    steps.append(obs)
obs0 = steps[0]
with torch.no_grad():
    h, _, valid, n_img = vf.vlm.forward_prefix(obs0)
    idx = int(vf.readout_index(valid)[0])
    logits, _ = vf(obs0)
    p = torch.softmax(logits, -1)[0]
    V0 = float(p @ vf.bin_values)
    ce0 = float(T.value_loss(vf, obs0)[0])
out_pre = T.label_episode(vf, steps, r, R, cfg=T.pretrain_config(), threshold=0.0, is_correction=lab.is_correction)
out_ft = T.label_episode(vf, steps, r, R, cfg=T.finetune_config(), threshold=0.0, is_correction=lab.is_correction)
pool = np.concatenate([out_ft["advantages"], rng.normal(0, 0.05, 500)])
eps = T.improvement_threshold(pool, T.POSITIVE_FRACTION_FINETUNE, rng)
ind = T.improvement_indicator(out_ft["advantages"], eps, lab.is_correction)
ind_noc = T.improvement_indicator(out_ft["advantages"], eps)
n_text = int(obs0.token_mask[0].sum())

ENC = [
    ("0 observation, layout value\n(../data)", f"images x3 f32[1,448,448,3]\ntokens: '[BOS] Task: fold the\n  shirt speed: fast,\n  State: ...;\\n' = {n_text} tok\nNO Subtask, NO Advantage\n(Sec. V-C: same ell as the VLA)", "#e8e8e8"),
    ("1 Gemma3VLM, small backbone\n(../backbone; 670M, Sec. V-C)", f"vision -> 3x256 image tokens\n+ {obs0.tokens.shape[1]} text -> h f32{tuple(h.shape)}\nimages bidir, text causal\npaper: Gemma 3 1B-size body\n  (698M non-emb), tiny w 32", "#e8e8e8"),
    ("2 readout: last valid token", f"idx = {idx} (768 image cols\n  + {n_text} text - 1)\ncausal text: this token has\n  seen everything\nfeat f32[1,{h.shape[2]}]\n(position undisclosed)", "#cfe2f3"),
    ("3 value head -> 201 logits\nSec. IV-A", f"Linear(w -> {NUM_BINS})\nlogits f32[1,{NUM_BINS}]\nsoftmax = p(V = b | o, ell)\nuntrained: max p {float(p.max()):.4f}\n  (uniform 1/201 = {1/201:.4f})", "#cfe2f3"),
    ("4 expectation\nV = sum_b p(b) v(b)", f"v(b) = -1 + b/200\nV(o_0) = {V0:+.4f}\n(uniform -> -0.5;\n trained target -0.275 =\n -11 steps / T_max 40)", "#cfe2f3"),
]
ADV = [
    ("A1 values along the episode", f"{T1} value calls ->\nV(o_t) = {' '.join(f'{x:+.2f}' for x in out_pre['values'][:4])} ...\n(Sec. V-D: on-the-fly during\n VLA training)", "#e8e8e8"),
    ("A2 advantage, App. F", f"pre-train N = T:\n  A = R_t - V(o_t)\n  = {' '.join(f'{x:+.2f}' for x in out_pre['advantages'][:3])} ...\nfine-tune N = 50:\n  sum r + V(o_t+50) - V(o_t)\n  = {' '.join(f'{x:+.2f}' for x in out_ft['advantages'][:3])} ...\n(equal here: episode < 50)", "#fce5cd"),
    ("A3 threshold eps_ell\nApp. F / Sec. V-D", f"quantile of a 10k-sample\npool of the task's A:\n  30% positive (pre-train)\n  40% (fine-tune), 10% (T-shirt)\nhere (40%, 512 pool):\n  eps = {eps:+.4f}", "#fce5cd"),
    ("A4 indicator I_t, Sec. IV-B", f"I_t = 1[A_t > eps]\n  = {''.join('1' if x else '0' for x in ind_noc)}\ncorrections (steps 8-11)\n  forced True:\n  = {''.join('1' if x else '0' for x in ind)}\nSFT stage: all True (V-D)", "#fce5cd"),
    ("A5 Advantage token\n(../data, 30% dropped)", f"I_t -> '{advantage_text(True).strip()}'\n   / '{advantage_text(False).strip()}'\ninto the policy sequence\nafter Subtask, before Action\n(../train consumes it)", "#d9ead3"),
]
DEC = [
    ("4' target bin (../data)", f"R_t / T_max = {R[0]:+.3f}\n-> bin {int(bins[0])} of 201\n(episode: success, 12 steps,\n T_max 40, Eq. 5)", "#e8e8e8"),
    ("3' Eq. 1: CE(target bin, p)", f"-log p(b = {int(bins[0])} | o_0, ell)\n= {ce0:.3f} (ln 201 = {np.log(201):.3f})\nplus web co-training CE\n  (Sec. V-C, weight undisclosed)", "#cfe2f3"),
    ("2' RECAP refit (Algorithm 1)", "each iteration: V_ell^k\n  from V_pre on all D_ell\n  (not from V^{k-1}, Sec. V-D)\noptimizer / steps undisclosed", "#cfe2f3"),
]

fig, ax = plt.subplots(figsize=(28, 12.4))
ax.set_xlim(0, 28)
ax.set_ylim(0, 12.4)
ax.axis("off")
BOX_W, GAP, X0 = 3.3, 0.24, 0.3
Y_ENC, H_ENC = 9.0, 2.8
Y_ADV, H_ADV = 5.6, 2.75
Y_DEC, H_DEC = 2.3, 2.2


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
for i, (t_, b_, c_) in enumerate(ADV):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ADV, BOX_W, H_ADV, t_, b_, c_)
    if i:
        arrow(x - GAP, Y_ADV + H_ADV / 2, x, Y_ADV + H_ADV / 2)
x_enc_last = X0 + (len(ENC) - 1) * (BOX_W + GAP)
arrow(x_enc_last + BOX_W / 2, Y_ENC, X0 + BOX_W / 2, Y_ADV + H_ADV)
ax.text(x_enc_last - 3.0, Y_ENC - 0.35, "repeat per step t", fontsize=8)
for i, (t_, b_, c_) in enumerate(DEC):
    x = x_enc_last - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, H_DEC, t_, b_, c_)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + H_DEC / 2, x + BOX_W, Y_DEC + H_DEC / 2)
ax.text(X0, Y_DEC - 0.35, "blue = value function (paper Sec. IV-A, V-C), orange = advantage / threshold / indicator (Sec. IV-B, V-D, App. F), grey = reused (../data, ../backbone), green = hand-off to the policy.  No upstream code exists.",
        fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) the value input is the policy's prefix without Subtask / Advantage;  (ii) V, R and r are all in the same normalised units (/ T_max) and the same 201-bin grid (../data);  "
        "(iii) eps_ell is a per-task quantile over a pool, never per episode;  (iv) corrections override the threshold, SFT overrides everything;  (v) each RECAP iteration refits V from the pre-trained V, not from the last one.",
        fontsize=8.5, va="top")
ax.text(X0, Y_ENC + H_ENC + 0.55, f"pi0.6* value function: observation -> 201-bin return distribution -> V -> advantage -> threshold -> Advantage token (tiny: {vf.vlm.cfg.depth}-layer width-{vf.vlm.cfg.width} body, "
        f"12-step success episode, T_max {Tmax}; paper: 670M Gemma 3 body, N = 50, 30% / 40% / 10% positive)", fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "V0", V0, "eps", eps, "ind", ind.astype(int).tolist())
