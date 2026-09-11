"""Generate figs/architecture.png: SigLIP + Gemma 2B over the pi0 prefix, tensor contract on every edge.
Run: uv run python pi/pi0/vlm/figs/make_figs.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).with_name("architecture.png")

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, title, sub, fc="#f3f3f3"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#444", lw=1.1))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + 0.3, sub, ha="center", va="center", fontsize=7.8, family="monospace")


def arrow(x0, y0, x1, y1, label=None, dx=0.08):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444"))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, fontsize=7.6, family="monospace", va="center")


# --- three camera slots -> three SigLIP passes (shared weights) ---
cams = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
for i, c in enumerate(cams):
    x = 0.4 + i * 2.55
    box(x, 5.9, 2.3, 0.95, c, "f32[B,224,224,3] in [-1,1]", fc="#dbe9ff")
    arrow(x + 1.15, 5.9, x + 1.15, 4.95)
    box(x, 4.0, 2.3, 0.95, "SigLIP So400m/14", "patch14 -> 256 tok, w1152, d27\nhead Linear 1152->2048")
    arrow(x + 1.15, 4.0, x + 1.15, 3.1, "f32[B,256,2048]")

# --- prompt tokens -> Gemma embedding ---
box(8.3, 5.9, 2.3, 0.95, "tokenized_prompt", "i64[B,48] + mask bool[B,48]", fc="#dbe9ff")
arrow(9.45, 5.9, 9.45, 4.95)
box(8.3, 4.0, 2.3, 0.95, "Gemma embed", "table[257152,2048] * sqrt(2048)")
arrow(9.45, 4.0, 9.45, 3.1, "f32[B,48,2048]")

# --- concat = prefix ---
box(0.4, 2.15, 10.2, 0.95, "prefix = concat(img0, img1, img2, prompt)",
    "emb f32[B,816,2048]   input_mask bool[B,816] (image_masks x256, token_mask)   ar_mask = all False")
arrow(5.5, 2.15, 5.5, 1.35, "mask = make_attn_mask(input_mask, ar_mask): bool[B,816,816]   positions = cumsum(input_mask)-1", dx=0.15)

# --- Gemma 2B ---
box(0.4, 0.3, 10.2, 1.05, "Gemma 2B  (18 x [RMSNorm -> MQA attn (8 q heads, 1 kv head, 256) + RoPE -> RMSNorm -> GeGLU 16384])",
    "out: hidden f32[B,816,2048]  |  kv_cache: 18 x (k, v) f32[B,816,1,256]  -> consumed by the action expert")

# --- side note ---
ax.text(11.0, 3.6, "one prefix pass per\naction chunk;\nthe 10 denoising steps\nreuse kv_cache",
        fontsize=8.5, va="center", ha="left", color="#333",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8dc", ec="#999", lw=0.8))

ax.set_title("pi0 VLM backbone (PaliGemma structure), openpi@215abfb", fontsize=12.5, pad=8)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print("wrote", OUT)
