"""Generate figs/objective.png: (a) the Fig. 18 attention mask on a small sequence (image / prefix text / FAST tokens /
pad / expert tokens); (b) the two-stage curriculum with each stage's data sources and the three disclosed learning-rate
schedules (openpi pi05_libero, openpi pi05_full_droid_finetune, Hi Robot high-level policy).
Run: uv run python pi/pi05/train/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.pi0.train.train import lr_at  # noqa: E402
from pi.pi05.train.train import MIXTURE, POSTTRAIN_STEPS, PRETRAIN_STEPS, hirobot_hl_config, make_joint_mask, openpi_droid_config, openpi_libero_config  # noqa: E402

OUT = pathlib.Path(__file__).with_name("objective.png")

# ---- (a) the mask on a toy sequence: 3 image tokens, 4 prefix text, 4 FAST, 1 pad, 3 expert
labels = ["img"] * 3 + ["txt"] * 4 + ["FAST"] * 4 + ["pad"] + ["exp"] * 3
pre = torch.tensor([[1] * 7 + [0] * 5], dtype=torch.bool)
post = torch.tensor([[0] * 7 + [1] * 4 + [0]], dtype=torch.bool)
m = make_joint_mask(pre, post, 3)[0].numpy()

fig = plt.figure(figsize=(14, 6.2))
a = fig.add_subplot(1, 2, 1)
a.imshow(m, cmap="Blues", vmin=0, vmax=1.3)
a.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7.5)
a.set_yticks(range(len(labels)), labels, fontsize=7.5)
for k in (3, 7, 11, 12):
    a.axhline(k - 0.5, color="#b45f06", lw=1)
    a.axvline(k - 0.5, color="#b45f06", lw=1)
a.set_xlabel("key (may be attended)")
a.set_ylabel("query")
a.set_title("(a) make_joint_mask on a toy sequence (paper Appendix E, Fig. 18)\nprefix bidirectional; FAST -> prefix + causal FAST; expert -> prefix + expert; no one sees the expert", fontsize=9)
a.text(11.6, 1.5, "expert never\nreads FAST", fontsize=7.5, color="#990000", ha="center")

# ---- (b) curriculum + schedules
b = fig.add_subplot(2, 2, 2)
b.set_xlim(0, PRETRAIN_STEPS + POSTTRAIN_STEPS)
b.set_ylim(0, 1)
b.add_patch(plt.Rectangle((0, 0.55), PRETRAIN_STEPS, 0.35, color="#fff2cc", ec="#444"))
b.add_patch(plt.Rectangle((PRETRAIN_STEPS, 0.55), POSTTRAIN_STEPS, 0.35, color="#cfe2f3", ec="#444"))
b.text(PRETRAIN_STEPS / 2, 0.72, f"pre-training: {PRETRAIN_STEPS // 1000}k steps, alpha = 0\n" + ", ".join(f"{s} ({l})" for s, l, _ in MIXTURE["pretrain"]), ha="center", va="center", fontsize=7.5)
b.text(PRETRAIN_STEPS + POSTTRAIN_STEPS / 2, 0.72, f"post-training\n{POSTTRAIN_STEPS // 1000}k, alpha = 10\nexpert added\n(random init)", ha="center", va="center", fontsize=7)
b.text(PRETRAIN_STEPS + POSTTRAIN_STEPS, 0.42, "post-training data:\n" + "\n".join(f"{s} ({l})" for s, l, _ in MIXTURE["posttrain"]), ha="right", va="top", fontsize=6.8)
b.set_yticks([])
b.set_xlabel("gradient steps (paper Sec. IV-D; optimizer, batch and compute undisclosed)")
b.set_title("(b) the two-stage curriculum and its data (Sec. IV-C, IV-D)", fontsize=9)

c = fig.add_subplot(2, 2, 4)
steps = np.arange(0, 100_001, 500)
for name, cfg, n in (("openpi pi05_libero (30k)", openpi_libero_config(), 30_000), ("openpi pi05_full_droid_finetune (100k)", openpi_droid_config(), 100_000), ("Hi Robot high-level policy (steps undisclosed)", hirobot_hl_config(), 100_000)):
    s = steps[steps <= n]
    c.plot(s, [lr_at(int(x), cfg) for x in s], label=name)
c.set_xlabel("step")
c.set_ylabel("learning rate")
c.set_yscale("log")
c.set_yticks([1e-6, 1e-5, 5e-5], ["1e-6", "1e-5", "5e-5"])
c.set_title("(c) the three disclosed schedules (config.py L744-L763, L865-L894; Hi Robot App. C.2)", fontsize=9)
c.legend(fontsize=7, loc="lower right")
c.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
