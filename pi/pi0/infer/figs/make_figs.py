"""Generate figs/pipeline.png: one pi0 inference call, which module runs how many times, with paper Table I timings.
Run: uv run python pi/pi0/infer/figs/make_figs.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).with_name("pipeline.png")

fig, ax = plt.subplots(figsize=(15, 6.4))
ax.set_xlim(0, 15)
ax.set_ylim(0, 6.4)
ax.axis("off")

C = dict(data="#e8e8e8", vlm="#dbe9ff", ae="#e3f3e0", fm="#ffe6cc", out="#f5e6ff")


def box(x, y, w, h, title, sub, fc, fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#444", lw=1.1))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="center", fontsize=fs, weight="bold")
    ax.text(x + w / 2, y + 0.3, sub, ha="center", va="center", fontsize=7.2, family="monospace")


def arrow(x0, y0, x1, y1, label=None, dy=0.12):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, fontsize=7, family="monospace", ha="center")


# row 1: raw -> data -> observation
box(0.3, 4.6, 2.4, 1.2, "raw robot inputs", "uint8 images {slot: [B,h,w,3]}\nstate f32[B,d]  prompt [str]", C["out"])
arrow(2.7, 5.2, 3.3, 5.2)
box(3.3, 4.6, 2.9, 1.2, "build_batch  (../data)  x1", "resize+pad 224, normalize,\npad to 32, tokenize 48, masks", C["data"])
arrow(6.2, 5.2, 6.8, 5.2)
ax.text(6.5, 5.42, "Observation", fontsize=7, family="monospace", ha="center")
box(6.8, 4.6, 3.6, 1.2, "SigLIP x3  +  embedder x1  (../vlm)", "3 x [B,256,2048] + [B,48,2048]\n-> prefix [B,816,2048]", C["vlm"])
arrow(10.4, 5.2, 11.0, 5.2)
box(11.0, 4.6, 3.7, 1.2, "llm xs=[prefix, None] x1  (expert 0)", "18 layers -> kv_cache\n18 x [B,816,1,256]", C["ae"])

# row 2: the loop
box(0.3, 2.3, 14.4, 1.75, "repeat 10x   t = 1.0 ... 0.1        (../flow_matching sample_actions)", "", C["fm"])
box(0.6, 2.6, 3.2, 1.0, "embed_suffix  (../action_expert)", "(state, x_t, t) -> [B,51,1024]", C["ae"])
arrow(3.8, 3.1, 4.3, 3.1)
box(4.3, 2.6, 4.0, 1.0, "llm xs=[None, suffix]  (expert 1, attends into cache)", "[B,51,1024] -> [B,51,1024]", C["ae"])
arrow(8.3, 3.1, 8.8, 3.1)
box(8.8, 2.6, 2.6, 1.0, "decode", "action_out_proj -> v_t [B,50,32]", C["ae"])
arrow(11.4, 3.1, 11.9, 3.1)
box(11.9, 2.6, 2.5, 1.0, "Euler", "x_t += -0.1 * v_t", C["fm"])
arrow(12.85, 4.6, 12.85, 4.05)
ax.text(12.95, 4.33, "kv_cache", fontsize=7, family="monospace", ha="left", va="center")

# row 3: output
arrow(7.5, 2.3, 7.5, 1.75)
ax.text(7.6, 2.03, "x_0 [B,50,32]", fontsize=7, family="monospace", ha="left", va="center")
box(3.3, 0.55, 8.4, 1.2, "to_executable_actions  (../flow_matching)  x1", "unnormalize -> joints += q_t -> [:, :, :d]   ->  actions f32[B,50,d], absolute, physical units", C["out"])
ax.text(12.0, 1.15, "-> execute rows 0..24 (50 Hz)\n   or 0..15 (20 Hz), open-loop,\n   then observe and call again",
        fontsize=8, va="center", ha="left")

# paper timing overlay
ax.text(0.3, 6.15, "paper Table I (RTX 4090, 3 cameras): image encoders 14 ms | observation forward 32 ms | x10 action forward 27 ms | total 73 ms on-board (86 ms off-board)",
        fontsize=8.2, va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#fff8dc", ec="#999", lw=0.7))
ax.text(0.3, 0.25, "params: SigLIP 414.8M + embedding 526.6M + expert 0 1981.9M + expert 1 311.5M + projections 3.2M = 3,238,048,528  (paper: 3.3B)",
        fontsize=8.2, va="center", family="monospace")

ax.set_title("pi0 inference call graph (openpi@215abfb pi0.py sample_actions, policy.py infer)", fontsize=12, pad=20)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
