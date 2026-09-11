"""Generate figs/train.png: the openpi fine-tuning lr schedule, parameter provenance, and the mixture weighting.
Run: uv run python pi/pi0/train/figs/make_figs.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))  # repo root, so `pi` imports

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pi.pi0.train.train import TrainConfig, lr_at, mixture_weights

OUT = pathlib.Path(__file__).with_name("train.png")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))

# (a) schedule
ax = axes[0]
c = TrainConfig()
steps = np.arange(0, 32_001, 50)
ax.plot(steps, [lr_at(int(s), c) for s in steps], color="#4a78c2", lw=2)
ax.axvline(c.warmup_steps, color="#999", ls=":", lw=1)
ax.axvline(c.decay_steps, color="#999", ls=":", lw=1)
ax.annotate(f"step 0: peak/(warmup+1) = {lr_at(0, c):.2e}", (0, lr_at(0, c)), (2500, 4e-6), fontsize=7.5, arrowprops=dict(arrowstyle="-|>", lw=0.8))
ax.annotate(f"step 1000: peak {c.peak_lr:.1e}", (1000, c.peak_lr), (4000, c.peak_lr * 0.98), fontsize=7.5, arrowprops=dict(arrowstyle="-|>", lw=0.8))
ax.annotate(f"step 30000: {c.decay_lr:.1e}, then constant", (30000, c.decay_lr), (15000, 6e-6), fontsize=7.5, arrowprops=dict(arrowstyle="-|>", lw=0.8))
ax.set_xlabel("step")
ax.set_ylabel("learning rate")
ax.set_title("(a) openpi fine-tuning schedule: linear warmup 1k + cosine to 30k\nAdamW b1 0.9 b2 0.95 eps 1e-8 wd 1e-10, clip 1.0, EMA 0.99, batch 32", fontsize=9.5)
ax.set_ylim(0, 2.8e-5)
ax.text(0.98, 0.95, "paper pre-training: 700k steps,\noptimizer settings not disclosed", transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc="#fff8dc", ec="#999", lw=0.7))

# (b) provenance
ax = axes[1]
parts = [("SigLIP", 414.8, "#dbe9ff"), ("embedding", 526.6, "#dbe9ff"), ("Gemma 2B\n(expert 0)", 1981.9, "#dbe9ff"),
         ("Gemma 300M\n(expert 1)", 311.5, "#e3f3e0"), ("projections", 3.2, "#e3f3e0")]
ax.barh([p[0] for p in parts], [p[1] for p in parts], color=[p[2] for p in parts], edgecolor="#444")
for i, p in enumerate(parts):
    ax.text(p[1] + 30, i, f"{p[1]:.1f}M", va="center", fontsize=8)
ax.set_xlim(0, 2600)
ax.invert_yaxis()
ax.set_xlabel("parameters (M)")
ax.set_title("(b) blue: loaded from PaliGemma pt_224 (2923.3M)\ngreen: from scratch (314.7M); all trained by default", fontsize=9.5)

# (c) mixture weighting
ax = axes[2]
counts = {"laundry\n1e8": 10**8, "bussing\n1e7": 10**7, "toast\n1e6": 10**6, "new robot\n1e5": 10**5}
raw = np.array(list(counts.values()), float)
raw /= raw.sum()
w = np.array(list(mixture_weights(counts).values()))
x = np.arange(len(counts))
ax.bar(x - 0.2, raw, 0.4, label="share by count", color="#cccccc", edgecolor="#444")
ax.bar(x + 0.2, w, 0.4, label="paper: ∝ n^0.43", color="#ffe6cc", edgecolor="#444")
ax.set_xticks(x)
ax.set_xticklabels(counts.keys(), fontsize=8)
ax.set_ylabel("sampling weight")
ax.legend(fontsize=8)
ax.set_title("(c) pre-training mixture weighting (paper Sec. V-A)\nillustrative counts; real counts not disclosed", fontsize=9.5)

fig.suptitle("pi0 training recipe (openpi@215abfb config.py / optimizer.py / scripts/train.py; paper Sec. V-A)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
