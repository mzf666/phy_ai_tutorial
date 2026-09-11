"""Generate figs/architecture.png: the pi0 suffix embedding, one two-expert Gemma layer, and the three-block mask.
Run: uv run python pi/pi0/action_expert/figs/make_figs.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).with_name("architecture.png")

fig = plt.figure(figsize=(15, 8.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.6)
ax.axis("off")

BLUE, GREY, ORANGE, GREEN = "#dbe9ff", "#f3f3f3", "#ffe6cc", "#e3f3e0"


def box(x, y, w, h, title, sub, fc=GREY, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#444", lw=1.1))
    ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center", fontsize=fs, weight="bold")
    ax.text(x + w / 2, y + 0.26, sub, ha="center", va="center", fontsize=7.4, family="monospace")


def arrow(x0, y0, x1, y1, label=None, dx=0.08, dy=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444"))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, fontsize=7.2, family="monospace", va="center")


# ================= left column: suffix embedding (pi0.py L140-L186) =================
ax.text(2.6, 8.3, "suffix embedding  (ActionProjections.embed_suffix)", ha="center", fontsize=11, weight="bold")
box(0.3, 7.1, 2.1, 0.8, "state q_t", "f32[B,32]  (../data)", fc=BLUE)
box(2.75, 7.1, 2.3, 0.8, "noisy actions A_t^tau", "f32[B,50,32]", fc=BLUE)
box(0.3, 5.75, 2.1, 0.8, "state_proj", "Linear 32->1024 (+bias)")
box(2.75, 5.75, 2.3, 0.8, "action_in_proj  (W1)", "Linear 32->1024")
arrow(1.35, 7.1, 1.35, 6.55)
arrow(3.9, 7.1, 3.9, 6.55)
box(3.55, 4.45, 1.5, 0.8, "posemb_sincos", "tau f32[B] -> f32[B,1024]\nperiod 4e-3..4.0", fc=ORANGE)
box(2.2, 3.0, 2.85, 1.0, "concat(W1 a, phi(tau)) -> W2 -> swish -> W3",
    "mlp_in 2048->1024, mlp_out 1024->1024\nf32[B,50,2048] -> f32[B,50,1024]", fs=8.5)
arrow(3.05, 5.75, 3.05, 4.0, "f32[B,50,1024]", dx=-1.35)
arrow(4.3, 4.45, 4.3, 4.0)
ax.plot([1.35, 1.35], [5.75, 2.3], color="#444", lw=1.1)
ax.text(1.42, 4.3, "f32[B,1,1024]", fontsize=7.2, family="monospace")
arrow(3.6, 3.0, 3.6, 2.3)
box(0.3, 1.5, 4.75, 0.8, "suffix tokens = [state | action x 50]", "f32[B,51,1024]  input_mask all True  ar_mask=[1,1,0x49]", fc=GREEN)
arrow(1.35, 2.3, 1.35, 2.3)  # no-op keeps the vertical line ending on the box
ax.text(2.7, 0.95, "-> expert 1 of every MoE layer;\nafter the stack: action_out_proj Linear 1024->32 on the last 50 tokens -> v_t f32[B,50,32]",
        ha="center", fontsize=7.6, va="center")

# ================= middle column: one MoE layer (gemma.py L284-L333) =================
mx = 5.75
ax.text(mx + 2.35, 8.3, "one of 18 layers  (MoEBlock)", ha="center", fontsize=11, weight="bold")
box(mx, 7.1, 2.2, 0.8, "expert 0 tokens", "prefix f32[B,816,2048]", fc=BLUE)
box(mx + 2.5, 7.1, 2.2, 0.8, "expert 1 tokens", "suffix f32[B,51,1024]", fc=GREEN)
box(mx, 5.85, 2.2, 0.85, "RMSNorm_0 -> q/kv proj_0", "q[B,816,8,256] k,v[B,816,1,256]")
box(mx + 2.5, 5.85, 2.2, 0.85, "RMSNorm_1 -> q/kv proj_1", "q[B,51,8,256] k,v[B,51,1,256]")
arrow(mx + 1.1, 7.1, mx + 1.1, 6.7)
arrow(mx + 3.6, 7.1, mx + 3.6, 6.7)
box(mx, 4.35, 4.7, 1.1, "concat on sequence -> RoPE -> ONE attention",
    "MQA 8 q heads / 1 kv head; q,k,v [B,867,...] (+ kv_cache [B,816,1,256])\nmask bool[B,867,867] train, [B,51,867] inference", fc=ORANGE)
arrow(mx + 1.1, 5.85, mx + 1.1, 5.45)
arrow(mx + 3.6, 5.85, mx + 3.6, 5.45)
box(mx, 2.95, 2.2, 0.95, "split -> out proj_0 -> +res", "RMSNorm_0 -> GeGLU_0 (16384) -> +res")
box(mx + 2.5, 2.95, 2.2, 0.95, "split -> out proj_1 -> +res", "RMSNorm_1 -> GeGLU_1 (4096) -> +res")
arrow(mx + 1.1, 4.35, mx + 1.1, 3.9)
arrow(mx + 3.6, 4.35, mx + 3.6, 3.9)
arrow(mx + 1.1, 2.95, mx + 1.1, 2.45, "f32[B,816,2048]", dx=-1.55)
arrow(mx + 3.6, 2.95, mx + 3.6, 2.45, "f32[B,51,1024]", dx=0.1)
ax.text(mx + 2.35, 1.95, "experts touch only inside the orange box; widths may differ, head layout must match",
        ha="center", fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#999", lw=0.8))
ax.text(mx + 2.35, 1.1, "inference: xs=[prefix, None] once -> kv_cache; then xs=[None, suffix] x 10 steps\n"
        "training: xs=[prefix, suffix] in one pass (joint_forward)", ha="center", fontsize=8, va="center")

# ================= right column: the three-block mask =================
rx = 11.05
ax.text(rx + 1.75, 8.3, "attention mask (867 x 867)", ha="center", fontsize=11, weight="bold")
n = 12  # schematic: 8 prefix, 1 state, 3 actions
ar = np.array([0] * 8 + [1, 1, 0, 0])
c = np.cumsum(ar)
m = (c[None, :] <= c[:, None]).astype(float)
mask_ax = fig.add_axes([(rx + 0.45) / 15, 3.55 / 8.6, 3.5 / 15, 3.5 / 8.6])
mask_ax.imshow(m, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
mask_ax.set_xticks([3.5, 8, 10])
mask_ax.set_xticklabels(["prefix (816)", "state", "actions (50)"], fontsize=7.5)
mask_ax.set_yticks([3.5, 8, 10])
mask_ax.set_yticklabels(["prefix", "state", "actions"], fontsize=7.5)
mask_ax.set_xlabel("key", fontsize=8)
mask_ax.set_ylabel("query", fontsize=8, labelpad=2)
for s in (7.5, 8.5):
    mask_ax.axhline(s, color="#c00", lw=0.8)
    mask_ax.axvline(s, color="#c00", lw=0.8)
mask_ax.tick_params(length=0)
ax.text(rx + 1.75, 2.9, "ar_mask = [0 x 816 | 1 | 1, 0 x 49]\nblock ids 0 | 1 | 2 ; dark = may attend",
        ha="center", fontsize=8, family="monospace", va="center")
ax.text(rx + 1.75, 1.75, "prefix: bidirectional, never sees\n  state / actions (pre-training modalities)\n"
        "state: sees prefix + itself\nactions: see everything, bidirectional",
        ha="center", fontsize=8, va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#999", lw=0.8))

ax.set_title("pi0 action expert: suffix embedding, two-expert layer, blockwise mask  (openpi@215abfb, paper Appendix B)",
             fontsize=12, pad=4, y=0.975)
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
