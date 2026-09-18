"""Generate figs/gemma3.png: (a) the attention mask of one tiny joint sequence at a local layer (sliding window) and at a
global layer, with the expert rows; (b) the 34-layer L/G pattern of Gemma 3 4B with the RoPE base and window per layer;
(c) parameter budget pi0.5 vs pi0.6 (backbone, vision, embedding, expert) with the inferred expert width.
Run: uv run python pi/pi06/backbone/figs/make_figs.py
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
import pi.pi06.backbone.model as M  # noqa: E402
from pi.fast.tokenizer.tokenizer import make_smooth_chunks  # noqa: E402
from pi.pi05.expert.model import expert_param_count as pi05_expert_count  # noqa: E402
from pi.pi0.vlm.model import GEMMA_2B, GEMMA_300M  # noqa: E402
from pi.pi06.data.data import SEG_ACTION, SEG_ADVANTAGE, SEG_PREFIX, SEG_SUBTASK, tiny_pi06_tokenizer  # noqa: E402

OUT = pathlib.Path(__file__).with_name("gemma3.png")
rng = np.random.default_rng(0)
H, d = 4, 3
seq = tiny_pi06_tokenizer(H, d, max_len=96)
state = rng.uniform(-0.5, 0.5, d).astype(np.float32)
actions = (0.3 * make_smooth_chunks(1, H, d, rng)[0]).astype(np.float32)
toks, mask, seg, loss = seq.tokenize("fold", state, layout="joint", actions=actions, subtask="grab", advantage=True)
n = int(mask.sum())
n_img, E = 6, H  # 6 stand-in image tokens (2 cameras x 3), E expert tokens
tm = torch.from_numpy(mask)[None]
vis = torch.from_numpy(np.isin(seg, (SEG_PREFIX, SEG_SUBTASK, SEG_ADVANTAGE)) & mask)[None]
m = M.make_pi06_mask(n_img, tm, n_expert=E, expert_visible=vis)[0]
S = n_img + n
keep = list(range(n_img + n)) + list(range(n_img + tm.shape[1], n_img + tm.shape[1] + E))
m = m[keep][:, keep].numpy()
pos = torch.cat([torch.arange(n_img + n), n_img + n + torch.arange(E)])[None]
win = 8
sl = M.sliding_mask(pos, pos, win)[0].numpy()
seg_r = seg[:n]
bounds = [0, n_img] + [n_img + int(np.argmax(seg_r == s)) for s in (SEG_SUBTASK, SEG_ADVANTAGE, SEG_ACTION)] + [S]
labels = ["images", "prefix", "Subtask", "Advantage", "FAST", "expert"]

fig = plt.figure(figsize=(20, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.15, 1.0])
a = fig.add_subplot(gs[0])
ga, gb = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
img = np.zeros((*m.shape, 3))
img[m & sl] = (0.15, 0.55, 0.25)  # visible at a local layer
img[m & ~sl] = (0.75, 0.88, 0.75)  # visible only at global layers (cut by the window)
img[~m] = (0.97, 0.97, 0.97)
a.imshow(img, interpolation="nearest")
for bnd in bounds[1:-1]:
    a.axvline(bnd - 0.5, color="#444", lw=0.8)
    a.axhline(bnd - 0.5, color="#444", lw=0.8)
mids = [(bounds[i] + bounds[i + 1]) / 2 - 0.5 for i in range(5)] + [S + E / 2 - 0.5]
a.set_xticks(mids, labels, fontsize=8)
a.set_yticks(mids, labels, fontsize=8)
a.set_title(f"(a) one joint sequence (6 image + {n} text + {E} expert tokens): dark = visible at a local layer (window {win}),\n"
            "light = visible at global layers only, white = never. Images bidirectional, text causal, expert never reads FAST,\nnobody reads the expert (card Sec. 2; paper Sec. V-A; KI App. B).", fontsize=8.5)

# (b) layer pattern
cfg = M.GEMMA3_4B
types = ["G" if cfg.is_global(i) else "L" for i in range(cfg.depth)]
ga.bar(range(cfg.depth), [1] * cfg.depth, color=["#e06666" if t == "G" else "#6fa8dc" for t in types], edgecolor="#444")
for i, t in enumerate(types):
    ga.text(i, 0.5, t, ha="center", va="center", fontsize=7, color="white", weight="bold")
ga.set_yticks([])
ga.set_xticks(range(0, cfg.depth, 3))
ga.set_xlabel("layer index (34 layers; _gemma.py L34, L39-L46)")
ga.set_title("(b) Gemma 3 4B: 5 local : 1 global. local = sliding window 1024, RoPE base 10k;\nglobal = full attention, RoPE base 1M with positions / 8 (L240-L244)", fontsize=8.5)
ga.text(0, 1.08, "L = local (29 layers)   G = global (5 layers: 5, 11, 17, 23, 29)", fontsize=8)
ga.set_ylim(0, 1.3)

# (c) parameter budget
pc = M.backbone_param_count()
cand = M.expert_width_candidates()[0]
with torch.device("meta"):
    vis_pi06 = sum(p.numel() for p in M.VisionEmbed(M.SIGLIP_400M_448, 2560).parameters())
rows = {
    "pi0.5 (PaliGemma + 300M expert)": [414_803_696, GEMMA_2B.vocab_size * GEMMA_2B.width, 2_923_335_408 - 414_803_696 - GEMMA_2B.vocab_size * GEMMA_2B.width, pi05_expert_count(GEMMA_300M)],
    "pi0.6 (Gemma 3 4B + ~860M expert)": [vis_pi06, pc["embedding"], pc["non_embedding"], cand[2]],
}
names = ["vision (SigLIP)", "embedding table", "LM layers (non-embedding)", "action expert"]
colors = ["#93c47d", "#ffd966", "#6fa8dc", "#c27ba0"]
left = np.zeros(2)
for j in range(4):
    vals = np.array([rows[k][j] for k in rows]) / 1e9
    gb.barh(list(rows), vals, left=left, color=colors[j], edgecolor="#444", label=names[j])
    for i, v in enumerate(vals):
        gb.text(left[i] + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5)
    left += vals
gb.set_xlabel("parameters (billions)")
gb.set_title(f"(c) parameter budget: 3.35B -> {left[1]:.2f}B\n(report Table 1: 417M / 675M / 3,209M; card: expert ~860M\n-> width {cand[0]}, mlp {cand[1]}, inferred, README Sec. 8)", fontsize=8.5)
gb.legend(fontsize=7.5, loc="lower right")
gb.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT, {k: [f"{x/1e6:.0f}M" for x in v] for k, v in rows.items()})
