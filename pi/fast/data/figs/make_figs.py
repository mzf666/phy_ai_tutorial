"""Generate figs/masks.png: the ar_mask / loss_mask bands of one 180-token FAST sequence and the attention mask they
imply, next to pi0's three-block mask for comparison.
Run: uv run python pi/fast/data/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.data.data import ByteTextCodec, FASTSequenceTokenizer, tiny_fast_tokenizer  # noqa: E402
from pi.fast.tokenizer.tokenizer import make_smooth_chunks  # noqa: E402

OUT = pathlib.Path(__file__).with_name("masks.png")
rng = np.random.default_rng(0)
H, d = 10, 7
seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, d), max_len=180)
state = make_smooth_chunks(1, 1, d, rng)[0, 0]
actions = make_smooth_chunks(1, H, d, rng)[0]
tok, m, ar, lm = seq.tokenize("pick up the red block", state, actions)
n_pre, n_real = int(((ar == 0) & m).sum()), int(m.sum())


def make_attn_mask(input_mask, ar_mask):  # pi0_fast.py L23-L48, one sequence
    c = np.cumsum(ar_mask)
    return (c[None, :] <= c[:, None]) & (input_mask[None, :] & input_mask[:, None])


fig = plt.figure(figsize=(16, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 5], width_ratios=[1.6, 1, 1])

# ---- (a) the three bands ----
ax = fig.add_subplot(gs[0, 0])
bands = np.stack([m.astype(int), ar, lm.astype(int)])
ax.imshow(bands, cmap="Blues", aspect="auto", vmin=0, vmax=1.4)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["token_mask", "ar_mask", "loss_mask"], fontsize=8)
ax.set_xticks([0, n_pre, n_real, 180])
ax.set_xticklabels(["0\nBOS", f"{n_pre}\n'Action: '", f"{n_real}\nEOS", "180\npad"], fontsize=7.5)
ax.set_title(f"(a) one sequence: prefix {n_pre} (Task + State) | postfix {n_real - n_pre} (Action ids) | pad {180 - n_real}", fontsize=9.5)

# ---- (b) FAST attention mask over the 180-token sequence (the 768 image keys are visible to every query, not drawn) ----
ax = fig.add_subplot(gs[1, 0])
A = make_attn_mask(m, ar)
ax.imshow(A, cmap="Greys", aspect="equal", interpolation="nearest")
for p in (n_pre, n_real):
    ax.axhline(p - 0.5, color="#c00", lw=0.8)
    ax.axvline(p - 0.5, color="#c00", lw=0.8)
ax.set_xticks([0, n_pre, n_real, 180])
ax.set_xticklabels(["0", f"{n_pre}", f"{n_real}", "180"], fontsize=7.5)
ax.set_yticks([0, n_pre, n_real])
ax.set_yticklabels(["BOS", "'Action: '", "EOS"], fontsize=7.5)
ax.set_xlabel("key   (prefix | postfix | pad); every query also sees all 768 image tokens")
ax.set_ylabel("query")
ax.set_title("(b) pi0-FAST sequence: prefix one bidirectional block,\npostfix causal token by token, pad masked (pi0_fast.py L23-L48)", fontsize=9)

# ---- (c) pi0's suffix side for comparison: 48 prompt | 1 state | 50 actions ----
ax = fig.add_subplot(gs[1, 1])
pi0_in = np.ones(48 + 51, bool)
pi0_ar = np.concatenate([np.zeros(48, int), [1], [1], np.zeros(49, int)])
Ap = make_attn_mask(pi0_in, pi0_ar)
ax.imshow(Ap, cmap="Greys", aspect="equal", interpolation="nearest")
for p in (48, 49):
    ax.axhline(p - 0.5, color="#c00", lw=0.8)
    ax.axvline(p - 0.5, color="#c00", lw=0.8)
ax.set_xticks([0, 48, 99])
ax.set_xticklabels(["0", "48 | 49\nstate", "99"], fontsize=7.5)
ax.set_yticks([0, 49])
ax.set_yticklabels(["prompt", "actions"], fontsize=7.5)
ax.set_xlabel("key   (prompt | state | 50 action tokens)")
ax.set_title("(c) pi0 for comparison: three blocks, the 50 action\ntokens attend to each other bidirectionally (pi0.py)", fontsize=9)

# ---- (d) zoom on the FAST postfix corner ----
ax = fig.add_subplot(gs[1, 2])
z0, z1 = n_pre - 12, n_real + 4
ax.imshow(A[z0:z1, z0:z1], cmap="Greys", aspect="equal", interpolation="nearest")
ax.axhline(12 - 0.5, color="#c00", lw=0.8)
ax.axvline(12 - 0.5, color="#c00", lw=0.8)
ax.set_xticks([0, 12, 12 + n_real - n_pre])
ax.set_xticklabels(["prefix\nend", "'Action: '", "EOS | pad"], fontsize=7.5)
ax.tick_params(axis="x", pad=2)
ax.set_yticks([])
ax.set_title("(d) zoom: each postfix token sees the whole prefix\nand the postfix tokens before it (lower triangle)", fontsize=9)
ax = fig.add_subplot(gs[0, 1:])
ax.axis("off")
ax.text(0, 0.5, "ar_mask feeds make_attn_mask: cumsum(ar) is 0 on images + prefix (one block, bidirectional) and increases by 1 per postfix token\n(strict causal). loss_mask == (ar_mask == 1): cross-entropy exactly on the tokens the model must generate. pi0 has 3 blocks; FAST has 1 + one per token.",
        fontsize=8.2, va="center")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
