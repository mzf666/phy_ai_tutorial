"""Generate figs/flow.png: the timestep distribution, the linear path with its constant velocity, and the sampler loop.
Run: uv run python pi/pi0/flow_matching/figs/make_figs.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).with_name("flow.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))

# ---------------- (a) timestep distribution, openpi convention t (1 = noise) ----------------
ax = axes[0]
t = np.linspace(0.001, 1.0, 400)
b = (t - 0.001) / 0.999
pdf = 1.5 * np.sqrt(b) / 0.999
ax.fill_between(t, pdf, alpha=0.25, color="#4a78c2")
ax.plot(t, pdf, color="#4a78c2", lw=2)
ax.axvline(0.6004, color="#c00", ls="--", lw=1)
ax.text(0.6004, 1.55, " mean 0.6004", color="#c00", fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.7)
ax.set_xlabel("t  (openpi: 1 = noise, 0 = clean)")
ax.set_ylabel("density")
ax.set_title("(a) training timestep\nt = Beta(1.5, 1) * 0.999 + 0.001", fontsize=10)
ax.text(0.03, 1.45, "paper: tau = 1 - t,\np(tau) = Beta((s - tau)/s; 1.5, 1), s = 0.999\nnever samples t < 0.001",
        fontsize=7.5, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="#fff8dc", ec="#999", lw=0.7))

# ---------------- (b) linear path for one action dimension ----------------
ax = axes[1]
a, n = -0.8, 1.1  # one scalar action value and one noise value
ts = np.linspace(0, 1, 100)
ax.plot(ts, ts * n + (1 - ts) * a, color="#333", lw=2)
ax.scatter([0, 1], [a, n], color=["#2a8f3a", "#c00"], zorder=3, s=40)
ax.text(0.02, a - 0.2, "t = 0: actions", color="#2a8f3a", fontsize=8.5)
ax.text(0.62, n + 0.08, "t = 1: noise", color="#c00", fontsize=8.5)
for tt in (0.3, 0.7):
    x = tt * n + (1 - tt) * a
    ax.annotate("", xy=(tt - 0.12, x - 0.12 * (n - a)), xytext=(tt, x), arrowprops=dict(arrowstyle="-|>", color="#4a78c2", lw=1.5))
    ax.text(tt + 0.02, x - 0.05, f"x_t, t={tt}", fontsize=7.5)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(a - 0.5, n + 0.4)
ax.set_xlabel("t")
ax.set_ylabel("one action dim (normalized)")
ax.set_title("(b) x_t = t*noise + (1-t)*actions\ntarget u = noise - actions (constant), loss = |v - u|^2", fontsize=10)
ax.text(0.05, n + 0.15, "blue arrows: -0.1 * u, the Euler step\n(network predicts v ~ u)", fontsize=7.5, color="#4a78c2")

# ---------------- (c) sampler loop ----------------
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")


def box(x, y, w, h, title, sub, fc="#f3f3f3", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center", fontsize=fs, weight="bold")
    ax.text(x + w / 2, y + 0.22, sub, ha="center", va="center", fontsize=6.8, family="monospace")


def arrow(x0, y0, x1, y1, label=None, dx=0.08):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1, color="#444"))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=6.8, family="monospace", va="center")


box(0.3, 4.9, 4.2, 0.9, "prefix once (../vlm, expert 0)", "kv_cache 18 x [B,816,1,256]", fc="#dbe9ff")
box(5.3, 4.9, 4.4, 0.9, "noise ~ N(0, I)", "x_1 f32[B,50,32], t = 1.0", fc="#dbe9ff")
box(0.3, 2.3, 9.4, 2.1, "repeat 10x   (t = 1.0, 0.9, ..., 0.1)", "", fc="#fff3e0")
box(0.6, 2.75, 2.6, 1.15, "embed_suffix", "(state, x_t, t)\n-> f32[B,51,1024]")
box(3.5, 2.75, 3.0, 1.15, "suffix_forward + decode", "attend into kv_cache\n-> v_t f32[B,50,32]")
box(6.8, 2.75, 2.6, 1.15, "Euler", "x_t += -0.1 * v_t\nt += -0.1")
arrow(3.2, 3.32, 3.5, 3.32)
arrow(6.5, 3.32, 6.8, 3.32)
arrow(2.4, 4.9, 2.4, 4.4)
arrow(7.5, 4.9, 7.5, 4.4)
box(0.3, 0.5, 9.4, 0.9, "x_0 = actions (normalized, delta, 32-dim)", "-> ../data inverse transforms -> execute 25 (50 Hz) or 16 (20 Hz) steps open-loop", fc="#e3f3e0")
arrow(5.0, 2.3, 5.0, 1.4)
ax.set_title("(c) sample_actions: 10 Euler steps, 27 ms on RTX 4090 (paper Table I)", fontsize=10)

fig.suptitle("pi0 flow matching (openpi@215abfb pi0.py L188-L279)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
