"""Generate figs/pipeline.png: one KI training step (top row: joint sequence -> prefix + expert tokens -> mask with the
visibility rules -> attention with the stop-gradient -> CE + alpha MSE -> which parameters each term moves) and the
RECAP loop of Algorithm 1 as the bottom row (demos -> SFT -> collect -> V from V_pre -> pi from pi_pre -> repeat).
Numbers from one real tiny run (expert gates perturbed so the stop-gradient is visible).
Run: uv run python pi/pi06/train/figs/make_pipeline.py
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
import pi.pi06.train.train as T  # noqa: E402
from pi.pi0.data.data import make_bool_mask  # noqa: E402
from pi.pi06.backbone.model import AdaRMSNorm, tiny_pi06  # noqa: E402
from pi.pi06.data.data import STATIC_IMAGE_KEYS, build_pi06_batch, tiny_pi06_tokenizer, unit_stats  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
torch.manual_seed(0)
rng = np.random.default_rng(0)
H, d, B = 10, 7, 2
model, seq = tiny_pi06(), tiny_pi06_tokenizer(H, d)
for m in model.modules():
    if isinstance(m, AdaRMSNorm):
        torch.nn.init.normal_(m.modulation.weight, std=0.05)
raw = {"images": {k: rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
       "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32), "actions": rng.uniform(-0.5, 0.5, (B, H + 2, d)).astype(np.float32),
       "prompt": ["make me an espresso", "fold the shirt"]}
obs, actions = build_pi06_batch(raw, unit_stats(d), seq, layout="joint", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=True,
                                subtasks=["grab the portafilter", None], advantages=[True, None])
t = torch.tensor([0.7, 0.3])
noise = torch.randn(actions.shape)
emb, valid, n_img = model.embed_prefix(obs)
mask = model.prefix_mask(obs, n_img, n_expert=H, expert_visible=obs.expert_visible)
S = emb.shape[1]
n_ce = [int(obs.loss_mask[i].sum()) for i in range(B)]
n_vis = [int(obs.expert_visible[i].sum()) for i in range(B)]
out = T.ki_loss(model, obs, actions, t=t, noise=noise, insulate=True)
g = {}
for name, term in (("mse", out["mse"].mean()), ("ce", out["ce"].mean())):
    model.zero_grad(set_to_none=True)
    term.backward(retain_graph=True)
    g[name] = T.grad_norms_by_part(model)
model.zero_grad(set_to_none=True)
o2 = T.ki_loss(model, obs, actions, t=t, noise=noise, insulate=False)
o2["mse"].mean().backward()
g["mse_no_sg"] = T.grad_norms_by_part(model)
model.zero_grad(set_to_none=True)

ENC = [
    ("0 joint batch (../data)", f"sample 0: prefix + Subtask +\n  Advantage: positive + FAST\n  ({n_ce[0]} CE tokens)\nsample 1: prefix + FAST, indicator\n  dropped ({n_ce[1]} CE tokens)\nactions f32[{B},{H},32], has_actions [T, T]\ntau ~ Beta(1.5, 1) (KI App. B)", "#e8e8e8"),
    ("1 tokens: prefix + expert\n(../backbone)", f"prefix emb f32[{B},{S},{emb.shape[2]}]\n  (3x256 image + 200 text)\nexpert: x_t = tau noise + (1-tau) a\n  -> {H} tokens, cond = time MLP\npositions: expert continues\n  after its visible columns", "#e8e8e8"),
    ("2 mask (card Sec. 2;\nKI App. B)", f"bool[{B},{S + H},{S + H}]\nimages bidir, text causal\nexpert rows read images +\n  {n_vis[0]} / {n_vis[1]} visible text cols\n  (never FAST) + itself\nno text / image row reads\n  the expert", "#cfe2f3"),
    ("3 attention with\nstop-gradient (KI Eq. 5-6)", "expert queries attend\n  sg(K_b), sg(V_b) of the\n  backbone tokens and their\n  own K_a, V_a\nbackbone rows: unchanged\ninsulate=True (34 layers)", "#cfe2f3"),
    ("4 two heads", f"text logits only at the {sum(n_ce)}\n  loss positions -> [N, 262144]\n  (Subtask + FAST targets)\nexpert -> v_t f32[{B},{H},32]\n  target u_t = noise - actions", "#cfe2f3"),
    ("5 loss, KI Eq. 4 / paper Eq. 4", f"L = mean CE + alpha mean MSE\n  alpha = 1 (stop-gradient)\nCE {[f'{x:.1f}' for x in out['ce'].tolist()]}\nMSE {[f'{x:.2f}' for x in out['mse'].tolist()]}\nno alpha for the indicator:\n  30% dropout instead (App. F)", "#cfe2f3"),
    ("6 who moves (tiny grad norms)", f"MSE term, insulate=True:\n  backbone {g['mse']['backbone']:.3f}  expert {g['mse']['expert']:.2f}\nMSE term, insulate=False:\n  backbone {g['mse_no_sg']['backbone']:.2f}  expert {g['mse_no_sg']['expert']:.2f}\nCE term: backbone {g['ce']['backbone']:.1f}\n  expert {g['ce']['expert']:.1f}\nAdamW / clip / EMA: pi0 helpers,\n  values undisclosed", "#d9ead3"),
]
LOOP = [
    ("L0 pre-training (Alg. 1 l. 1-2)", "V_pre on D_demo (Eq. 1)\npi_pre on D_demo with\n  V_pre's indicators\n  (30% positive, N = T)\n= offline RL pre-training", "#fce5cd"),
    ("L1 task SFT (l. 3-5)", "D_ell <- demonstrations\nV_ell^0 from V_pre\npi_ell^0 from pi_pre,\n  I = True everywhere\n  (Sec. V-D)", "#fce5cd"),
    ("L2 collect (l. 7)", "deploy pi_ell^{k-1} (../infer)\nautonomous rollouts +\n  optional expert corrections\nhuman outcome labels\nD_ell <- D_ell + new", "#fce5cd"),
    ("L3 refit V (l. 8)", "V_ell^k from V_PRE on all\n  of D_ell (Eq. 1)\nnot from V_ell^{k-1}\n  (Sec. V-D: avoids drift)", "#fce5cd"),
    ("L4 refit pi (l. 9)", "pi_ell^k from pi_PRE with\n  V_ell^k's indicators\n  (40% positive, N = 50;\n   corrections True)\n1-2 iterations in the paper", "#fce5cd"),
    ("L5 result", "throughput > 2x, failures\n  / 2 (Sec. VI-C)\nstrict T-shirt 97% after\n  2 x 600 autonomous eps.", "#d9ead3"),
]

fig, ax = plt.subplots(figsize=(28, 9.6))
ax.set_xlim(0, 28)
ax.set_ylim(0, 9.6)
ax.axis("off")
BOX_W, GAP, X0 = 3.7, 0.24, 0.3
Y_ENC, H_ENC = 5.7, 2.85
Y_DEC, H_DEC = 2.3, 2.3


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
for i, (t_, b_, c_) in enumerate(LOOP):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, H_DEC, t_, b_, c_)
    if i:
        arrow(x - GAP, Y_DEC + H_DEC / 2, x, Y_DEC + H_DEC / 2)
# the loop arrow L4 -> L2
x2, x4 = X0 + 2 * (BOX_W + GAP), X0 + 4 * (BOX_W + GAP)
ax.annotate("", xy=(x2 + BOX_W / 2, Y_DEC), xytext=(x4 + BOX_W / 2, Y_DEC), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1, connectionstyle="arc3,rad=-0.3"))
ax.text((x2 + x4) / 2 + BOX_W / 2, Y_DEC - 0.95, "repeat k = 1..K (both refits from the pre-trained checkpoints)", fontsize=8, ha="center")
ax.text(X0, Y_DEC - 1.35, "blue = KI objective (KI Sec. 5.1-5.2; paper Sec. V-A / V-B; card Sec. 2), orange = RECAP loop (paper Algorithm 1, Sec. IV-C, V-D), grey = reused (../data, ../backbone), green = outcome.  No upstream code exists.",
        fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 1.75, "Must agree: (i) the stop-gradient is inside attention (K_b, V_b for expert queries), so alpha = 1 and the CE is the only signal the backbone gets from actions;  (ii) the CE never depends on the expert (nobody reads it);  "
        "(iii) the expert never reads FAST tokens but does read the Advantage token;  (iv) every RECAP refit starts from the PRE-TRAINED V / pi;  (v) SFT = indicator fixed True, RECAP = indicator from the refit V.", fontsize=8.5, va="top")
ax.text(X0, Y_ENC + H_ENC + 0.55, "pi0.6* training: one Knowledge-Insulation step (CE on Subtask + FAST, alpha = 1 x flow MSE, expert gradient stopped in attention) and the RECAP loop (tiny model, expert gates perturbed)",
        fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "grads", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in g.items()})
