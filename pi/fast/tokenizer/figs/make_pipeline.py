"""Generate figs/pipeline.png: the FAST tokenizer as a step-by-step flow chart (encode on top, decode below),
with the shape and a concrete example at every step, taken from a real tiny run.
Run: uv run python pi/fast/tokenizer/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False  # the example strings contain "$" and control characters
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.tokenizer.tokenizer import (  # noqa: E402
    FASTTokenizer, QuantileStats, dct, ints_to_text, make_smooth_chunks, normalize_quantile, pretokenize, quantize,
)

OUT = pathlib.Path(__file__).with_name("pipeline.png")

# ---- one real tiny run: a 20 Hz, 4-dim robot, 1-second chunk ----
rng = np.random.default_rng(0)
H, D = 20, 4
chunks = make_smooth_chunks(64, H, D, rng)
tok = FASTTokenizer.fit(list(chunks), scale=10, vocab_size=320)
stats = QuantileStats(q01=np.array([-1.5, -0.8, 0.0, 0.0]), q99=np.array([1.5, 0.8, 0.08, 1.0]))  # pretend units
a_norm = chunks[0]
a_raw = (a_norm + 1) / 2 * (stats.q99 - stats.q01) + stats.q01
C = dct(a_norm, axis=0)
Q = quantize(C, tok.scale)
flat = Q.reshape(-1)
text = ints_to_text(flat, tok.min_token)
words = pretokenize(text)
ids = tok(a_norm)[0]
rec = tok.decode([ids], time_horizon=H, action_dim=D)[0]


def fmt_row(r, w=5):
    return " ".join(f"{v:>{w}.2f}" if isinstance(v, float) or isinstance(v, np.floating) else f"{int(v):>{w}d}" for v in r)


fig, ax = plt.subplots(figsize=(22.5, 7.6))
ax.set_xlim(0, 22.5)
ax.set_ylim(0, 7.6)
ax.axis("off")

BOX_W, BOX_H = 2.85, 2.55
DEC_H = 1.55
Y_ENC, Y_DEC = 4.6, 1.5
X0, GAP = 0.35, 0.28


def box(x, y, title, shape, body, fc, h=BOX_H):
    ax.add_patch(FancyBboxPatch((x, y), BOX_W, h, boxstyle="round,pad=0.04", fc=fc, ec="#444", lw=1))
    ax.text(x + BOX_W / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.6, weight="bold")
    ax.text(x + BOX_W / 2, y + h - 0.58, shape, ha="center", va="top", fontsize=8.2, color="#1a4d8f", family="monospace")
    ax.text(x + 0.12, y + h - 0.95, body, ha="left", va="top", fontsize=7.1, family="monospace", linespacing=1.25)


def arrow(x1, x2, y, label, up=True, color="#333"):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color))
    ax.text((x1 + x2) / 2, y + (0.14 if up else -0.14), label, ha="center", va="bottom" if up else "top", fontsize=7.4, color=color)


# ---------------- encode row ----------------
enc = [
    ("0. raw action chunk", f"float[H={H}, D={D}]  (rad, m, ...)",
     "1 s of teleop targets @ 20 Hz\nrows = time steps, cols = joints\n\n" + "\n".join(fmt_row(a_raw[i]) for i in range(4)) + "\n  ...  (20 rows)"),
    ("1. quantile normalize", "float[H, D] ~ [-1, 1]",
     "(x - q01)/(q99 - q01) * 2 - 1\nq01/q99 per dim from the train set\nno clipping\n\n" + "\n".join(fmt_row(a_norm[i]) for i in range(4)) + "\n  ..."),
    ("2. DCT-II along time", "float[H, D]  (frequency x dim)",
     "C = M @ a, M orthonormal, per dim\nrow k = frequency k\nrow 0 = mean * sqrt(H)\n\n" + "\n".join(fmt_row(C[i]) for i in range(4)) + "\n  ...  (rows 8..19 ~ 0)"),
    ("3. round(gamma * C)", f"int[H, D], gamma = {tok.scale}",
     f"the only lossy step\nnonzero: {int((Q != 0).sum())} of {Q.size}\n\n" + "\n".join(fmt_row(Q[i], 4) for i in range(5)) + "\n  ...  (all zero)"),
    ("4. flatten, low freq first", f"int[H*D = {H * D}]",
     "row-major over [H, D]:\nfreq 0 of all dims,\nthen freq 1 of all dims, ...\n\n" + " ".join(str(v) for v in flat[:8]) + "\n" + " ".join(str(v) for v in flat[8:16]) + "\n...  0 0 0 0 0 0 0 0"),
    ("5. chr(q - min_token)", f"str, {len(text)} chars; min_token = {tok.min_token}",
     "each int -> one unicode char\ncode = q - min_token, 0 -> chr(%d)\nGPT-2 pre-tokenizer splits it into\n%d 'words' (by unicode class)\n\ncodes: %s\n       %s ..." % (-tok.min_token, len(words), " ".join(str(ord(c)) for c in text[:8]), " ".join(str(ord(c)) for c in text[8:16]))),
    ("6. byte-level BPE", f"int[n = {len(ids)}]  <<  {H * D}",
     "utf-8 bytes -> 256 byte chars,\nmerge by rank, never across words\n\n" + " ".join(str(v) for v in ids[:7]) + "\n" + " ".join(str(v) for v in ids[7:14]) + "\n" + " ".join(str(v) for v in ids[14:]) + "\n\n-> action tokens for the VLM"),
]
xs = [X0 + i * (BOX_W + GAP) for i in range(len(enc))]
fcs = ["#f4f4f4", "#e8f0fb", "#e8f0fb", "#fde9d9", "#e8f0fb", "#e8f0fb", "#e3f3e3"]
for x, (t, sh, b), fc in zip(xs, enc, fcs):
    box(x, Y_ENC, t, sh, b, fc)
for i in range(len(enc) - 1):
    arrow(xs[i] + BOX_W + 0.02, xs[i + 1] - 0.02, Y_ENC + BOX_H / 2, "")

# ---------------- decode row (mirror) ----------------
dec = [
    ("6'. BPE decode", "int[n] -> str",
     "ids -> byte chars -> utf-8\nfails if the model emits\na non-utf-8 byte sequence"),
    ("5'. ord(c) + min_token", f"str -> int[{H * D}]",
     "must give exactly H*D ints,\nelse upstream prints an error\nand returns a ZERO chunk"),
    ("4'. reshape(H, D)", "int[H, D]",
     "H and D must be supplied by\nthe caller: the tokens carry\nno shape information"),
    ("3'. / gamma", "float[H, D]",
     "back to the coefficient grid;\nrounding error <= 0.5/gamma\nper coefficient"),
    ("2'. inverse DCT (M^T)", "float[H, D] ~ [-1, 1]",
     f"orthonormal, so chunk error\n<= 0.5/gamma * sqrt(H)\nthis run: max|err| = {np.abs(rec - a_norm).max():.3f}"),
    ("1'. quantile unnormalize", "float[H, D]  (rad, m, ...)",
     "(x + 1)/2 * (q99 - q01) + q01\nthen pi.pi0.data.to_absolute_actions\nand cut to the native dim"),
]
xs_dec = [xs[6 - i] for i in range(len(dec))]
for x, (t, sh, b) in zip(xs_dec, dec):
    box(x, Y_DEC, t, sh, b, "#fbf7e6", h=DEC_H)
for i in range(len(dec) - 1):
    arrow(xs_dec[i] - 0.02, xs_dec[i + 1] + BOX_W + 0.02, Y_DEC + DEC_H / 2, "")
# down arrow from encode box 6 to decode box 6'
ax.add_patch(FancyArrowPatch((xs[6] + BOX_W / 2, Y_ENC - 0.02), (xs[6] + BOX_W / 2, Y_DEC + DEC_H + 0.02), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333"))
ax.text(xs[6] + BOX_W / 2 + 0.12, (Y_ENC + Y_DEC + DEC_H) / 2, "VLM generates\nthese ids\n(../model)", fontsize=7.4, va="center")

# ---------------- codebook note ----------------
ax.text(X0, Y_DEC - 0.35,
        f"codebook = (gamma, min_token, BPE vocab + merges) is one unit: fit() computes min_token from the training set and trains the merges on those "
        f"character codes; encode and decode must use the same three values (FAST+: gamma 10, min_token -354, vocab 2048).",
        fontsize=8.2, va="top", color="#333")
ax.text(X0, Y_DEC - 0.72,
        "encode: paper Algorithm 1 / fast_hf L43-L58.   decode: fast_hf L60-L96.   fit: fast_hf L99-L150.   Example values are from a real run of tokenizer.py main() (tiny, synthetic).",
        fontsize=7.6, va="top", color="#666")
ax.text(X0, 7.45, "FAST tokenizer: how one action chunk becomes tokens (top, left to right) and how tokens become actions again (bottom, right to left)",
        fontsize=11.5, weight="bold", va="top")

fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("wrote", OUT)
