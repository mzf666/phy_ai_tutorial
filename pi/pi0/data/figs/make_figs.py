"""Generate figs/pipeline.png: the pi0 data pipeline with the tensor contract on every edge.
Run: uv run python pi/pi0/data/figs/make_figs.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).with_name("pipeline.png")

# (label, sublabel) for each stage, top to bottom; edges carry the tensor spec leaving the stage.
STAGES = [
    ("raw robot sample", "images {name: uint8[B,h,w,3]}  state f32[B,d]  actions f32[B,>=50,d]  prompt str"),
    ("(a) robot adapter", "rename cameras to 3 slots; missing slot -> zeros + image_mask=False"),
    ("(b) DeltaActions", "joint dims: a_t' - q_t ; gripper dims unchanged  (mask per robot)"),
    ("(c) Normalize (z-score)", "(x - mean) / (std + 1e-6), per-robot stats, state and actions"),
    ("(d) ResizeImages", "aspect-preserving bilinear resize + black pad -> uint8[B,224,224,3]"),
    ("(e) TokenizePrompt", "BOS + sentencepiece(prompt) + '\\n' ; pad/truncate to 48"),
    ("(f) PadStatesAndActions", "zero-pad last dim d -> 32"),
    ("(g) in-model: to [-1,1] (+ augment if train)", "crop 95% / rotate ±5° (non-wrist) ; HSV jitter p=0.5 (all)"),
    ("Observation + actions", "images f32[B,224,224,3]x3  image_masks bool[B]x3  state f32[B,32]\n"
                              "tokenized_prompt i64[B,48]  mask bool[B,48]  |  actions f32[B,50,32]"),
]

fig, ax = plt.subplots(figsize=(11, 12.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(STAGES) * 1.35 + 0.3)
ax.axis("off")

box_w, box_h = 8.4, 0.95
for i, (title, sub) in enumerate(STAGES):
    y = (len(STAGES) - 1 - i) * 1.35 + 0.4
    is_io = i in (0, len(STAGES) - 1)
    face = "#dbe9ff" if is_io else "#f3f3f3"
    ax.add_patch(FancyBboxPatch((0.8, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=face, ec="#444", lw=1.2))
    ax.text(1.0, y + box_h - 0.22, title, fontsize=11.5, weight="bold", va="center")
    ax.text(1.0, y + 0.3, sub, fontsize=8.6, va="center", family="monospace")
    if i < len(STAGES) - 1:
        ax.annotate("", xy=(5.0, y - 0.38), xytext=(5.0, y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#444"))

ax.set_title("pi0 data pipeline (openpi@215abfb order)", fontsize=13, pad=10)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
