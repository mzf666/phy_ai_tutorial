"""Generate figs/decode.png: (a) the structure of inference cost, pi0 vs pi0-FAST: how many sequential transformer
forwards and on how many parameters each; (b) the per-step attention window over the cache in one real tiny decode.
Run: uv run python pi/fast/model/figs/make_figs.py
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
from pi.fast.data.data import ACTION_DIM, ByteTextCodec, FASTSequenceTokenizer, build_fast_batch, tiny_fast_tokenizer  # noqa: E402
from pi.fast.model.model import Pi0FAST, left_to_right_align, tiny  # noqa: E402
from pi.fast.tokenizer.tokenizer import QuantileStats  # noqa: E402
from pi.pi0.data.data import make_bool_mask  # noqa: E402
from pi.pi0.vlm.model import make_attn_mask  # noqa: E402

OUT = pathlib.Path(__file__).with_name("decode.png")

# ---- (b) real tiny run: the cache window of each decode step -------------------------------------------------------
torch.manual_seed(0)
rng = np.random.default_rng(0)
B, H, d = 1, 10, 7
model = Pi0FAST(*tiny()).eval()
raw = {
    "images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8), "wrist_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
    "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
    "prompt": ["Pick_up the red block"],
}
stats = {"state": QuantileStats(np.full(d, -1.0), np.full(d, 1.0)), "actions": QuantileStats(np.full(d, -1.0), np.full(d, 1.0))}
seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, d), max_len=180)
obs, _ = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=False)
N_STEPS = 12
with torch.no_grad():
    emb, m, ar = model.embed_inputs(obs)
    emb_r, mask_r, attn_r = left_to_right_align(emb, m, make_attn_mask(m, ar))
prefill_size = mask_r.shape[1]
prefill_len = int(mask_r.sum())
prefix_start = prefill_size - prefill_len
n_img = 3 * 256
# window[step, col] = may the query of decode step `step` see cache column `col`  (pi0_fast.py L294-L298)
cols = np.arange(prefill_size + N_STEPS)
window = np.zeros((N_STEPS, prefill_size + N_STEPS), dtype=bool)
for s in range(N_STEPS):
    window[s] = (cols >= prefix_start) & (cols < prefill_size + s + 1)

# ---- (a) structure of inference cost: disclosed values only ---------------------------------------------------------
# pi0: one prefix forward on the 2B VLM (816 image + 48 text tokens), then 10 Euler steps on the 300M expert
#      (pi0 arXiv:2410.24164v1 Appendix B; openpi pi0.py num_steps=10). pi0-FAST: one prefill on 2B (768 + 250 tokens),
#      then 30-60 decode steps on 2B (FAST Sec. VI-E). Latency 100 ms vs 750 ms on an RTX 4090 (Sec. VI-E).
fig, (axa, axb) = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
pi0_steps = [("prefill 2B", 2.9, "#6fa8dc")] + [("Euler step 300M", 0.3, "#93c47c")] * 10
fast_lo = [("prefill 2B", 2.9, "#6fa8dc")] + [("decode step 2B", 2.9, "#e06666")] * 30
fast_hi = [("prefill 2B", 2.9, "#6fa8dc")] + [("decode step 2B", 2.9, "#e06666")] * 60
for y, (name, steps) in enumerate([("pi0-FAST, bimanual (~60 tokens)", fast_hi), ("pi0-FAST, single arm (~30 tokens)", fast_lo), ("pi0 (flow matching)", pi0_steps)]):
    for i, (_, p, c) in enumerate(steps):
        axa.bar(i, p, width=0.85, bottom=y * 3.6, color=c, edgecolor="none")
    axa.text(-1.2, y * 3.6 + 1.4, name, ha="right", va="center", fontsize=9)
axa.set_xlim(-22, 62)
axa.set_xticks([0, 10, 20, 30, 40, 50, 60])
axa.set_ylim(-0.4, 11.2)
axa.set_yticks([])
axa.set_xlabel("sequential transformer forwards per action chunk (bar height = parameters run at that step, B)")
axa.set_title("(a) why inference is slower: every action token is one more 2B forward", fontsize=10)
axa.text(12, 3.6 * 2 + 0.9, "paper Sec. VI-E: ~100 ms", fontsize=8.5, color="#38761d", ha="left")
axa.text(31, 3.6 * 1 + 3.25, "paper Sec. VI-E: ~750 ms (RTX 4090)", fontsize=8.5, color="#990000", ha="center")
axa.text(31, 3.6 * 0 + 3.25, "30-60 tokens: Sec. VI-E; 2B / 300M: pi0 Appendix B", fontsize=8.5, color="#333", ha="center")
for c, lab in (("#6fa8dc", "prefill (images + text), 2B backbone"), ("#93c47c", "flow-matching Euler step, 300M expert"), ("#e06666", "autoregressive decode step, 2B backbone")):
    axa.bar(0, 0, color=c, label=lab)
axa.legend(loc="upper right", fontsize=8, frameon=False)

ZOOM = 30  # show the last ZOOM prefix columns and the generated ones; everything from prefix_start on is visible anyway
axb.imshow(window, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1.4)
axb.set_xlabel(f"KV-cache column, zoomed: last {ZOOM} of {prefill_len} prefix cols + generated; pads [0, {prefix_start}) never visible\n"
               f"x = the column written at that step (token of step-1); each step sees one more column")
axb.set_ylabel("decode step")
axb.set_title(f"(b) cache window per decode step, tiny run: cols [{prefix_start}, {prefill_size}+step+1)", fontsize=10)
axb.axvline(prefill_size - 0.5, color="#990000", lw=0.8, ls="--")
axb.text(prefill_size - ZOOM / 2, -0.9, f"prefix text (cols {prefill_size - ZOOM}..{prefill_size - 1})", ha="center", fontsize=8)
axb.text(prefill_size + N_STEPS / 2, -0.9, "generated tokens", ha="center", fontsize=8, color="#990000")
for s_ in range(N_STEPS):
    axb.text(prefill_size + s_, s_, "x", ha="center", va="center", fontsize=7, color="white")  # the column written at this step
axb.set_xlim(prefill_size - ZOOM - 0.5, prefill_size + N_STEPS - 0.5)
axb.set_ylim(N_STEPS - 0.5, -1.4)
fig.suptitle("openpi@215abfb pi0_fast.py sample_actions L236-L313; gemma_fast.py cache L165-L206", fontsize=9, y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT, "prefix_start", prefix_start, "prefill_len", prefill_len)
