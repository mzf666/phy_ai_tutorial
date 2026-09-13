"""Generate figs/pipeline.png: FASTObservation -> generated action tokens, step by step (top), and the mirrored
inverse (tokens -> actions) with the two stop conditions (bottom). Values come from one real tiny run.
Run: uv run python pi/fast/model/figs/make_pipeline.py
"""

import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.data.data import ACTION_DIM, EOS_ID, ByteTextCodec, FASTSequenceTokenizer, build_fast_batch, paligemma_to_fast, tiny_fast_tokenizer  # noqa: E402
from pi.fast.model.model import MAX_DECODING_STEPS, Pi0FAST, left_to_right_align, tiny  # noqa: E402
from pi.fast.tokenizer.tokenizer import QuantileStats  # noqa: E402
from pi.pi0.data.data import make_bool_mask  # noqa: E402
from pi.pi0.vlm.model import make_attn_mask  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")

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
n_text = int(obs.tokenized_prompt_mask.sum())

with torch.no_grad():
    img0 = model.img(obs.images["base_0_rgb"])
    emb, input_mask, ar_mask = model.embed_inputs(obs)
    attn = make_attn_mask(input_mask, ar_mask)
    emb_r, mask_r, attn_r = left_to_right_align(emb, input_mask, attn)
    prefill_size = mask_r.shape[1]
    prefill_len = int(mask_r.sum())
    prefix_start = prefill_size - prefill_len
    positions = mask_r.long().cumsum(1) - 1
    t0 = time.perf_counter()
    pre_logits, cache = model.llm(emb_r, positions, attn_r)
    t_prefill = (time.perf_counter() - t0) * 1e3
    logit = model.logits_head(pre_logits[:, -1:])
    top = torch.topk(logit[0, 0], 3)
    token = logit[0, 0].argmax()
    probs = torch.softmax(logit[0, 0] / 0.7, -1)
    col = torch.arange(prefill_size + MAX_DECODING_STEPS)
    step_ms = []
    tok = token.view(1, 1)
    for step in range(1, 4):
        n_cols = prefill_size + step
        cm = (col[None, None, :n_cols] >= prefix_start) & (col[None, None, :n_cols] < n_cols)
        t0 = time.perf_counter()
        lg, cache = model.decode_step(tok, torch.tensor([[prefill_len + step]]), cache, cm)
        step_ms.append((time.perf_counter() - t0) * 1e3)
        tok = lg[0, 0].argmax().view(1, 1)
    tokens, n_steps = model.sample_actions(obs, max_decoding_steps=32)
acts = seq.extract_actions(tokens[0].numpy(), H, d)
marker = seq._action_marker


def v3(x):
    return " ".join(f"{float(v):+.2f}" for v in x[:3])


ENC = [
    ("0 FASTObservation\n(../data, inference)", f"images 3x f32[1,224,224,3]\nimage_masks all True\ntokenized_prompt i64[1,180]\n  real {n_text} = prefix only\ntoken_ar_mask all 0\nstate f32[1,32]: not read", "#e8e8e8"),
    ("1 SigLIP (pi0/vlm)\nshared over 3 cameras", f"f32[1,256,64] per camera\ncam0 token0: {v3(img0[0,0])}\n3 x 256 = 768 image tokens", "#e8e8e8"),
    ("2 embed tokens + concat\nembed_inputs L159-L195", f"tokens: lookup * sqrt(64)\nemb f32[1,948,64]\ninput_mask bool[1,948]\n  True: 768 + {n_text} = {prefill_len}\nar_mask i64[1,948]: all 0", "#cfe2f3"),
    ("3 make_attn_mask\n(pi0/vlm) L23-L48", f"bool[1,948,948]\nattendable {attn[0].float().mean():.3f}\n= ({prefill_len}/948)^2\nall bidirectional:\nno postfix at inference", "#cfe2f3"),
    ("4 left_to_right_align\nL51-L64", f"roll by -{prefill_len}\nprefix_start {prefix_start}\nprefill_len {prefill_len}\npositions: pads -1,\nreal 0..{prefill_len-1}", "#cfe2f3"),
    ("5 prefill: Gemma once\n(pi0/vlm) L265-L267", f"pre_logits f32[1,948,64]\ncache 4 layers x (k,v)\n  [1,948,1,16]\nlast col: {v3(pre_logits[0,-1])}\n{t_prefill:.0f} ms (tiny, CPU)", "#cfe2f3"),
    ("6 tied head on last col\nlogits = h . E^T  L120-L121", f"f32[1,1,257152]\ntop-3 ids {top.indices.tolist()}\nvalues {v3(top.values)}\nno head parameters", "#cfe2f3"),
    ("7 sample one token\nL279-L284", f"greedy: argmax -> {int(token)}\nT=0.7: p(argmax) {float(probs[token]):.3f}\n  categorical(logit/T)\ntokens[:,0] = {int(token)}", "#f9cb9c"),
    ("8 decode step (x N)\nL292-L301", f"embed 1 token f32[1,1,64]\nposition prefill_len+step+1\n  = {prefill_len+1}, {prefill_len+2}, ...\ncache_mask cols [{prefix_start}, 949+step)\ncache grows 949, 950, ...\n{np.mean(step_ms):.1f} ms/step (tiny, CPU)", "#cfe2f3"),
    ("9 output\nL310-L313", f"tokens i64[1,256]\nran {n_steps} steps (cap 32 here,\n  {MAX_DECODING_STEPS} upstream)\nfirst ids {tokens[0,:4].tolist()}\nunfilled columns = 0", "#d9ead3"),
]
DEC = [
    ("9' generated ids", f"i64[256]\n{tokens[0,:6].tolist()} ...", "#d9ead3"),
    ("8' stop conditions\nL288-L289, L307", f"all samples emitted EOS={EOS_ID}\n  -> stop (samples that\n  stopped earlier keep\n  generating junk)\nor step == cap -> hard stop", "#cfe2f3"),
    ("7' extract_actions\n(../data) L119-L134", f"find 'Action: ' ids\n  {marker[:4]}...\ntake until '|' / EOS\nnot found -> zeros", "#cfe2f3"),
    ("6' paligemma_to_fast\n257152-1-128-t", f"tail ids -> [0, 2048)\nout of range -> zeros\n(text where actions\n should be)", "#cfe2f3"),
    ("5' FAST decode\n(../tokenizer)", f"BPE -> ints -> /gamma\n-> IDCT -> f32[{H},{d}]\nthis run: all zero?\n  {bool((acts == 0).all())} (random weights,\n  no marker)", "#cfe2f3"),
]

fig, ax = plt.subplots(figsize=(30, 8.2))
ax.set_xlim(0, 30)
ax.set_ylim(0, 8.2)
ax.axis("off")
BOX_W, BOX_H, DEC_H, GAP, X0 = 2.72, 2.35, 1.95, 0.24, 0.3
Y_ENC, Y_DEC = 4.6, 1.65


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=9, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=7.6, family="monospace")


def arrow(x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1, **kw))


for i, (t, b, c) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_ENC + BOX_H / 2, x, Y_ENC + BOX_H / 2)
# loop arrow 8 -> 7
x7, x8 = X0 + 7 * (BOX_W + GAP), X0 + 8 * (BOX_W + GAP)
ax.add_patch(FancyArrowPatch((x8 + BOX_W / 2, Y_ENC + BOX_H), (x7 + BOX_W / 2, Y_ENC + BOX_H), arrowstyle="-|>", mutation_scale=12,
                             color="#b45f06", lw=1.2, connectionstyle="arc3,rad=0.35"))
ax.text((x7 + x8 + BOX_W) / 2, Y_ENC + BOX_H + 0.55, "next logits -> sample again", ha="center", fontsize=8, color="#b45f06")

x_last = X0 + 9 * (BOX_W + GAP)
for i, (t, b, c) in enumerate(DEC):
    x = x_last - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, DEC_H, t, b, c)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + DEC_H / 2, x + BOX_W, Y_DEC + DEC_H / 2)
arrow(x_last + BOX_W / 2, Y_ENC, x_last + BOX_W / 2, Y_DEC + DEC_H)
ax.text(x_last + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + DEC_H) / 2, "inverse", fontsize=8, va="center")

ax.text(X0, Y_DEC - 0.35, "blue = pi0-FAST increment (pi0_fast.py), grey = pi0/vlm and ../data reused, orange = stochastic / irreversible, green = output.  "
        "Line refs: openpi@215abfb pi0_fast.py unless noted; L120-L121 = gemma_fast.py.", fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) the 2048 embedding rows [254976, 257023] are both the action input embedding and the action logits (tied head);  "
        "(ii) extract_actions needs the same (H, D, tokenizer) that built the training postfix;  (iii) cache size = prefill_size + max_decoding_steps;  "
        "(iv) decode position is prefill_len+step+1 (training: prefill_len+step, README Sec. 8).", fontsize=8.5, va="top")
ax.text(X0, Y_ENC + BOX_H + 1.0, "pi0-FAST model: one prefill over [3 x 256 image tokens | prefix text + state bins], then one 2B forward per generated action token  "
        f"(tiny config: width 64, 4 layers; LIBERO-scale sequence 768 + 180 = 948, {H}-step {d}-dim chunk)", fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "prefill_len", prefill_len, "prefix_start", prefix_start, "n_steps", n_steps, "first ids", tokens[0, :4].tolist())
