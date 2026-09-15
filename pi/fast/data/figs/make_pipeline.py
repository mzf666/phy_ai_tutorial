"""Generate figs/pipeline.png: raw sample -> one token sequence, step by step (top), and the inverse that cuts the actions
back out of a generated sequence (bottom). Values come from one real tiny run. Steps shared with pi0 are grey.
Run: uv run python pi/fast/data/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.data.data import ACTION_DIM, ByteTextCodec, FASTSequenceTokenizer, build_fast_batch, discretize_state, fast_to_paligemma  # noqa: E402
from pi.fast.tokenizer.tokenizer import QuantileStats, make_smooth_chunks, normalize_quantile  # noqa: E402
from pi.pi0.data.data import make_bool_mask, to_delta_actions  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")

rng = np.random.default_rng(0)
B, H, d = 1, 10, 7
raw = {
    "images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8), "wrist_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
    "state": rng.uniform(-0.6, 0.6, (B, d)).astype(np.float32),
    "actions": (rng.uniform(-0.6, 0.6, (B, 1, d)) + 0.3 * make_smooth_chunks(B, H, d, rng)).astype(np.float32),
    "prompt": ["Pick_up the red block"],
}
q01, q99 = np.full(d, -1.0), np.full(d, 1.0)
stats = {"state": QuantileStats(q01, q99), "actions": QuantileStats(q01, q99)}
mask = make_bool_mask(6, -1)
codec = ByteTextCodec()
# fit the stand-in FAST tokenizer on chunks drawn like the example (delta + normalized), so the example is in-distribution
_fit = [normalize_quantile(to_delta_actions(s0, a0, mask)[0], stats["actions"]) for s0, a0 in
        ((rng.uniform(-0.6, 0.6, (1, d)).astype(np.float32), (rng.uniform(-0.6, 0.6, (1, 1, d)) + 0.3 * make_smooth_chunks(1, H, d, rng)).astype(np.float32)) for _ in range(64))]
from pi.fast.tokenizer.tokenizer import FASTTokenizer  # noqa: E402
seq = FASTSequenceTokenizer(codec, FASTTokenizer.fit(_fit, scale=10, vocab_size=400), max_len=180)
obs, actions = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=mask, train=False)
state_n = normalize_quantile(raw["state"][0], stats["state"])
delta = to_delta_actions(raw["state"], raw["actions"], mask)[0]
act_n = normalize_quantile(delta, stats["actions"])
fast_ids = np.asarray(seq.fast(act_n[None])[0])
tok, m, ar, lm = (x[0].numpy() for x in (obs.tokenized_prompt, obs.tokenized_prompt_mask, obs.token_ar_mask, obs.token_loss_mask))
n_pre, n_real = int(((ar == 0) & m).sum()), int(m.sum())
rec = seq.extract_actions(tok, H, d)


def row(v, w=5):
    return " ".join(f"{x:>{w}.2f}" for x in v)


fig, ax = plt.subplots(figsize=(22.5, 8.6))
ax.set_xlim(0, 22.5)
ax.set_ylim(0, 8.6)
ax.axis("off")
BOX_W, BOX_H, DEC_H, GAP, X0 = 2.85, 3.0, 1.75, 0.28, 0.35
Y_ENC, Y_DEC = 5.0, 1.6


def box(x, y, title, shape, body, fc, h=BOX_H):
    ax.add_patch(FancyBboxPatch((x, y), BOX_W, h, boxstyle="round,pad=0.04", fc=fc, ec="#444", lw=1))
    ax.text(x + BOX_W / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.4, weight="bold")
    ax.text(x + BOX_W / 2, y + h - 0.56, shape, ha="center", va="top", fontsize=7.9, color="#1a4d8f", family="monospace")
    ax.text(x + 0.1, y + h - 0.92, body, ha="left", va="top", fontsize=6.9, family="monospace", linespacing=1.22)


def arrow(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333"))


GREY, BLUE, ORANGE, GREEN, CREAM = "#f1f1f1", "#e8f0fb", "#fde9d9", "#e3f3e3", "#fbf7e6"
enc = [
    ("0. raw sample", "images uint8[128,160,3] x2\nstate f32[7], actions f32[H+, 7]", f"prompt {raw['prompt'][0]!r}\nstate  {row(raw['state'][0][:4])} ..\nactions[0] {row(raw['actions'][0, 0, :4])} ..\ncameras: base_0, wrist_0\n(no base_1)", GREY),
    ("1. slots + delta (pi0)", "images -> 3 slots, all mask True\nactions f32[H=10, 7]", f"base_1_rgb := black image,\nimage_mask = True (FAST!)\njoints -= state, gripper kept\ndelta[0] {row(delta[0, :4])} ..", GREY),
    ("2. quantile normalize", "state, actions ~ [-1, 1]", f"(x - q01)/(q99 - q01)*2 - 1\nstate_n {row(state_n[:4])} ..\nact_n[0] {row(act_n[0, :4])} ..\n(pi0: z-score instead)", BLUE),
    ("3. state -> 256 bins -> text", "int[7] -> str", f"digitize(x, 256 edges in [-1,1)) - 1\nbins {discretize_state(state_n).tolist()}\n\nprefix text:\n'Task: pick up the red block,\n State: {' '.join(map(str, discretize_state(state_n)[:4]))} ..;\\n'\n-> {n_pre} ids incl. BOS", BLUE),
    ("4. FAST(actions) -> ids", f"f32[10, 7] -> int[{len(fast_ids)}]", f"../tokenizer on the NATIVE dim\n(before padding to 32)\nids {fast_ids[:6].tolist()} ..\n\nmap into PaliGemma tail:\n257152 - 1 - 128 - id\n-> {fast_to_paligemma(fast_ids[:3]).tolist()} ..", BLUE),
    ("5. postfix + masks + pad", f"tokens int[180] = {n_pre} + {n_real - n_pre} + pad", f"'Action: ' + ids + '|' + EOS\nar_mask   0 x {n_pre} | 1 x {n_real - n_pre} | 0 pad\nloss_mask F x {n_pre} | T x {n_real - n_pre} | F pad\ntoken_mask T x {n_real} | F x {180 - n_real}\n\nprefix: bidirectional, no loss\npostfix: causal, CE loss", ORANGE),
    ("6. FASTObservation", "imgs f32[3][224,224,3] state f32[32]\n+ 4 x int/bool[180]", "images resized + padded (pi0)\nstate zero-padded to 32 but the\nFAST model never reads it\n\ntokenized_prompt, _mask,\ntoken_ar_mask, token_loss_mask\n-> ../model, ../train", GREEN),
]
xs = [X0 + i * (BOX_W + GAP) for i in range(len(enc))]
for x, (t, sh, b, fc) in zip(xs, enc):
    box(x, Y_ENC, t, sh, b, fc)
for i in range(len(enc) - 1):
    arrow(xs[i] + BOX_W + 0.02, xs[i + 1] - 0.02, Y_ENC + BOX_H / 2)

dec = [
    ("6'. generated ids", "int[L] from ../model", "prefix + whatever the VLM wrote\n(ideally 'Action: ' ids '|' EOS)"),
    ("5'. find 'Action: ' .. '|'", "int[n] action span", "no 'Action: ' -> ZERO chunk (upstream)\ntext id inside the span -> zeros"),
    ("4'. tail -> FAST ids -> decode", f"int[n] -> f32[10, 7]", f"257152 - 1 - 128 - id (same map)\n../tokenizer decode(H=10, D=7)\nmax|rec - act_n| = {np.abs(rec - act_n).max():.3f}"),
    ("2'. unnormalize + absolute", "f32[10, 7] -> robot units", "(x+1)/2*(q99-q01)+q01\njoints += state; cut to native dim\n(../infer)"),
]
xs_dec = [xs[6], xs[5], xs[4], xs[2]]
for x, (t, sh, b) in zip(xs_dec, dec):
    box(x, Y_DEC, t, sh, b, CREAM, h=DEC_H)
for i in range(len(dec) - 1):
    arrow(xs_dec[i] - 0.02, xs_dec[i + 1] + BOX_W + 0.02, Y_DEC + DEC_H / 2)
ax.add_patch(FancyArrowPatch((xs[6] + BOX_W / 2, Y_ENC - 0.02), (xs[6] + BOX_W / 2, Y_DEC + DEC_H + 0.02), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333"))
ax.text(xs[6] + BOX_W / 2 + 0.12, (Y_ENC + Y_DEC + DEC_H) / 2, "VLM prefills the\nprefix, generates\nthe postfix\n(../model)", fontsize=7.4, va="center")

ax.text(X0, Y_DEC - 0.35, "grey = identical to pi0 (imported from pi.pi0.data); blue = FAST-specific; orange = the one new artefact (sequence + 3 masks). "
        "Encode: openpi@215abfb tokenizer.py L64-L117, transforms.py L270-L288.  Decode: tokenizer.py L119-L134.  Order: data_loader.py L183-L190 with config.py L150-L159.", fontsize=7.8, va="top", color="#444")
ax.text(X0, Y_DEC - 0.7, "Example: one LIBERO-like sample (7-dim, action_horizon 10, max_token_len 180) through a real tiny run of data.py main(); the text codec is a byte stand-in for PaliGemma's SentencePiece.", fontsize=7.6, va="top", color="#666")
ax.text(X0, 8.45, "pi0-FAST data: how a raw sample becomes one token sequence with three masks (top), and how a generated sequence becomes actions again (bottom)", fontsize=11.5, weight="bold", va="top")
fig.savefig(OUT, dpi=125, bbox_inches="tight")
print("wrote", OUT)
