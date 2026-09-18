"""Generate figs/ki.png: (a) gradient norms on the backbone vs the expert from the CE term and the MSE term, with and
without the stop-gradient (tiny model, real numbers); (b) where the flow loss's gradient can travel: the attention
blocks of one layer with the sg() marks of KI Eq. 5-6; (c) the RECAP data / indicator schedule across stages: which
checkpoint each stage starts from, which indicator rule it uses, and the paper's per-iteration episode counts.
Run: uv run python pi/pi06/train/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
import pi.pi06.train.train as T  # noqa: E402
from pi.pi0.data.data import make_bool_mask  # noqa: E402
from pi.pi06.backbone.model import AdaRMSNorm, tiny_pi06  # noqa: E402
from pi.pi06.data.data import STATIC_IMAGE_KEYS, build_pi06_batch, tiny_pi06_tokenizer, unit_stats  # noqa: E402
from pi.pi06.infer.eval import EPISODES_PER_ITERATION  # noqa: E402

OUT = pathlib.Path(__file__).with_name("ki.png")
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
obs, actions = build_pi06_batch(raw, unit_stats(d), seq, layout="joint", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=False,
                                subtasks=["grab the portafilter", "grab the collar"], advantages=[True, False])
t = torch.tensor([0.7, 0.3])
noise = torch.randn(actions.shape)
rows = []
for label, term, ins in (("CE term", "ce", True), ("MSE term, insulate=True (KI)", "mse", True), ("MSE term, insulate=False (joint-training baseline)", "mse", False)):
    model.zero_grad(set_to_none=True)
    T.ki_loss(model, obs, actions, t=t, noise=noise, insulate=ins)[term].mean().backward()
    g = T.grad_norms_by_part(model)
    rows.append((label, g["backbone"], g["expert"]))
model.zero_grad(set_to_none=True)

fig = plt.figure(figsize=(20, 6.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.25])
a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
# (a)
y = np.arange(len(rows))
a.barh(y - 0.18, [r[1] for r in rows], height=0.36, color="#6fa8dc", edgecolor="#444", label="backbone (vision, embedder, experts[0])")
a.barh(y + 0.18, [r[2] for r in rows], height=0.36, color="#c27ba0", edgecolor="#444", label="action expert (experts[1], proj)")
for i, r in enumerate(rows):
    a.text(max(r[1], 1e-3) * 1.05, i - 0.18, f"{r[1]:.3g}", va="center", fontsize=7.5)
    a.text(max(r[2], 1e-3) * 1.05, i + 0.18, f"{r[2]:.3g}", va="center", fontsize=7.5)
a.set_yticks(y, [r[0].replace(" (joint-training baseline)", "\n(joint-training baseline)").replace(" (KI)", "\n(KI)") for r in rows], fontsize=7.5)
a.set_xlim(0, max(max(r[1], r[2]) for r in rows) * 1.25)
a.set_xlabel("gradient norm (tiny model, one batch)")
a.set_title("(a) which weights each loss term moves\n(KI Eq. 4-6: the backbone learns actions only through the FAST CE)", fontsize=9)
a.legend(fontsize=7, loc="lower right")
a.invert_yaxis()
a.grid(axis="x", alpha=0.3)

# (b) the attention block diagram
b.set_xlim(0, 10)
b.set_ylim(0, 10)
b.axis("off")


def blk(x, y, w, h, text, color):
    b.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=color, ec="#444", lw=1))
    b.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


blk(0.5, 7.2, 4, 1.6, "backbone tokens X_b\n(images | text incl. FAST)", "#6fa8dc")
blk(5.5, 7.2, 4, 1.6, "expert tokens X_a\n(50 noisy actions, adaRMSNorm)", "#c27ba0")
blk(0.5, 4.2, 4, 1.6, "Q_b K_b^T -> P_bb\nP_bb V_b", "#cfe2f3")
blk(5.5, 4.2, 4, 1.6, "Q_a [sg(K_b) ; K_a]^T -> [P_ab P_aa]\nP_ab sg(V_b) + P_aa V_a", "#ead1dc")
blk(0.5, 1.2, 4, 1.6, "-> CE on Subtask + FAST\n(backbone weights move)", "#d9ead3")
blk(5.5, 1.2, 4, 1.6, "-> alpha x flow MSE\n(only expert weights move)", "#d9ead3")
for x0, x1 in ((2.5, 2.5), (7.5, 7.5)):
    b.annotate("", xy=(x1, 5.8), xytext=(x0, 7.2), arrowprops=dict(arrowstyle="-|>", color="#333"))
    b.annotate("", xy=(x1, 2.8), xytext=(x0, 4.2), arrowprops=dict(arrowstyle="-|>", color="#333"))
b.annotate("", xy=(5.5, 5.0), xytext=(4.5, 5.0), arrowprops=dict(arrowstyle="-|>", color="#cc0000", lw=1.5))
b.text(5.0, 5.35, "K_b, V_b\nwith sg()", ha="center", fontsize=7.5, color="#cc0000")
b.text(5.0, 9.5, "one layer of Gemma3MoEBlock (KI Eq. 5-6; ../backbone insulate=True)", ha="center", fontsize=8.5, weight="bold")
b.text(5.0, 0.4, "no arrow from X_a back to X_b: nobody reads the expert; the sg() cuts the only remaining path", ha="center", fontsize=7.5)
b.set_title("(b) the stop-gradient sits inside attention, not at the loss", fontsize=9)

# (c) RECAP schedule
c.set_xlim(0, 10)
c.set_ylim(0, 10)
c.axis("off")
stages = [
    ("pre-training", "from Gemma 3 4B + fresh expert", "I from V_pre, 30% positive, N = T", "tens of thousands of demo hours"),
    ("task SFT", "from pi_pre", "I = True", "task demonstrations"),
    ("RECAP iter. 1", "from pi_pre", "I from V^1 (refit from V_pre), 40% positive, N = 50", "+ rollouts (+ corrections)"),
    ("RECAP iter. 2", "from pi_pre", "I from V^2 (refit from V_pre), 40% positive", "+ rollouts (+ corrections)"),
]
for i, (name, init, ind, data) in enumerate(stages):
    yy = 8.6 - i * 2.1
    c.add_patch(FancyBboxPatch((0.2, yy - 0.8), 9.6, 1.7, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fce5cd" if i else "#e8e8e8", ec="#444", lw=1))
    c.text(0.4, yy + 0.55, name, fontsize=9, weight="bold", va="center")
    c.text(0.4, yy + 0.05, f"init: {init}", fontsize=7.5, va="center")
    c.text(0.4, yy - 0.4, f"indicator: {ind}", fontsize=7.5, va="center")
    c.text(9.6, yy + 0.05, f"data: {data}", fontsize=7.5, va="center", ha="right")
lines = [f"{k.replace('laundry (', '').replace(')', '')}: {v['autonomous']} + {v['corrections']}" for k, v in EPISODES_PER_ITERATION.items()]
c.text(5.0, 0.55, "episodes per iteration (Sec. VI-C.2, App. F):  " + "  |  ".join(lines[:3]) + "\n" + "  |  ".join(lines[3:]), fontsize=6.8, ha="center", va="center")
c.set_title("(c) RECAP stages (Algorithm 1, Sec. V-D): every refit starts from the pre-trained checkpoints", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT, rows)
