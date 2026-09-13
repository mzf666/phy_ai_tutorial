"""Generate figs/tokenizer.png: the five-step FAST pipeline on one synthetic chunk, the sparse quantized DCT matrix,
and the compression / reconstruction trade-off versus the rounding scale.
Run: uv run python pi/fast/tokenizer/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.tokenizer.eval import sweep_scale  # noqa: E402
from pi.fast.tokenizer.tokenizer import FASTTokenizer, dct, make_smooth_chunks, pretokenize, quantize  # noqa: E402

OUT = pathlib.Path(__file__).with_name("tokenizer.png")
rng = np.random.default_rng(0)
H, D = 50, 7
train = list(make_smooth_chunks(64, H, D, rng))
test = make_smooth_chunks(32, H, D, rng)
tok = FASTTokenizer.fit(train, scale=10, vocab_size=1024)
a = test[0]
ids = tok(a)[0]
rec = tok.decode([ids], time_horizon=H, action_dim=D)[0]
q = quantize(dct(a, axis=0), tok.scale)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw=dict(width_ratios=[1.25, 0.9, 1.1]))

# ---------------- (a) chunk vs reconstruction, plus the pipeline as text ----------------
ax = axes[0]
t = np.arange(H) / H
for d, col in zip(range(3), ["#4a78c2", "#2a8f3a", "#c0392b"]):
    ax.plot(t, a[:, d], color=col, lw=2, label=f"dim {d}" if d == 0 else None)
    ax.plot(t, rec[:, d], color=col, lw=1, ls="--")
ax.plot([], [], color="#333", lw=2, label="normalized chunk a")
ax.plot([], [], color="#333", lw=1, ls="--", label="decode(encode(a))")
ax.set_xlabel("time in the 1 s chunk (H = 50 steps @ 50 Hz)")
ax.set_ylabel("normalized action (q01 -> -1, q99 -> +1)")
ax.set_ylim(-1.25, 1.55)
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_title(f"(a) one chunk [H={H}, D={D}] -> {len(ids)} tokens (naive binning: {H * D})", fontsize=10)
steps = ("1. DCT-II along time, per dim (ortho)\n"
         "2. Q = round(10 * C)   <- the only lossy step\n"
         "3. flatten [H, D] row-major: low frequencies first\n"
         "4. chr(Q - min_token) -> string\n"
         "5. byte-level BPE (GPT-2 pre-tokenizer) -> ids")
ax.text(0.02, -1.18, steps, fontsize=7.6, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff8dc", ec="#999", lw=0.7))

# ---------------- (b) the quantized DCT matrix ----------------
ax = axes[1]
last = int(np.nonzero(q.any(axis=1))[0].max())
show = q[: max(24, last + 1)]
vmax = max(1, np.abs(show).max())
im = ax.imshow(show, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
for i in range(show.shape[0]):
    for j in range(show.shape[1]):
        if show[i, j] != 0:
            ax.text(j, i, str(show[i, j]), ha="center", va="center", fontsize=6.5, color="k")
ax.set_xlabel("action dim")
ax.set_ylabel("frequency index k (row 0 = mean * sqrt(H))")
ax.set_title(f"(b) Q = round(10 * DCT): {int((q != 0).sum())} of {q.size} nonzero\n(first {show.shape[0]} of {H} rows; rows {show.shape[0]}..{H - 1} all zero)", fontsize=10)
ax.set_xticks(range(D))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

# ---------------- (c) trade-off vs scale ----------------
ax = axes[2]
rows = sweep_scale(train, test, scales=(1, 2, 5, 10, 20, 50), vocab_size=1024)
sc = [r.scale for r in rows]
ax.plot(sc, [r.tokens_per_chunk for r in rows], "o-", color="#4a78c2", lw=2)
ax.set_xscale("log")
ax.set_xlabel("rounding scale gamma (paper / FAST+: 10)")
ax.set_ylabel("tokens per chunk", color="#4a78c2")
ax.axhline(H * D, color="#999", ls=":", lw=1)
ax.text(1.05, H * D * 0.93, f"naive binning = H*D = {H * D}", fontsize=7.5, color="#666")
ax.axvline(10, color="#c00", ls="--", lw=1)
ax2 = ax.twinx()
ax2.plot(sc, [r.mse for r in rows], "s--", color="#c0392b", lw=1.5)
ax2.set_yscale("log")
ax2.set_ylabel("reconstruction MSE (normalized space)", color="#c0392b")
ax.set_title("(c) compression vs fidelity, tokenizer refit per gamma\n(synthetic smooth chunks; shape of paper Fig. 12, not its numbers)", fontsize=10)

fig.suptitle("FAST tokenizer: normalized action chunk -> DCT -> quantize -> flatten -> byte-level BPE -> tokens", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
