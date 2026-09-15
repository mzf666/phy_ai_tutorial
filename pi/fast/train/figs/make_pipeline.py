"""Generate figs/pipeline.png: one batch from FASTObservation to one parameter update, step by step (top), and the
parameter side, full fine-tuning vs LoRA (bottom). Values come from one real tiny run.
Run: uv run python pi/fast/train/figs/make_pipeline.py
"""

import dataclasses
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.data.data import ACTION_DIM, PALIGEMMA_VOCAB_SIZE, ByteTextCodec, FASTSequenceTokenizer, build_fast_batch, tiny_fast_tokenizer  # noqa: E402
from pi.fast.model.model import Pi0FAST, paper, tiny  # noqa: E402
from pi.fast.tokenizer.tokenizer import QuantileStats, make_smooth_chunks  # noqa: E402
from pi.fast.train.train import apply_lora, cross_entropy, lora_param_count, paper_config, select_trainable, target_logits  # noqa: E402
from pi.pi0.data.data import make_bool_mask  # noqa: E402
from pi.pi0.train.train import EMA, clip_and_step, lr_at, make_optimizer  # noqa: E402
from pi.pi0.vlm.model import make_attn_mask  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")

torch.manual_seed(0)
rng = np.random.default_rng(0)
B, H, d = 2, 10, 7
model = Pi0FAST(*tiny())
cfg = dataclasses.replace(paper_config(), warmup_steps=2, batch_size=B)
stats = {"state": QuantileStats(np.full(d, -1.0), np.full(d, 1.0)), "actions": QuantileStats(np.full(d, -1.0), np.full(d, 1.0))}
seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, d), max_len=180)
raw = {"images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
       "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
       "actions": (0.3 * make_smooth_chunks(B, H + 2, d, rng)).astype(np.float32),
       "prompt": ["pick up the red block", "close the drawer"]}
obs, _ = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=True, generator=torch.Generator().manual_seed(0))
n_real, n_loss = obs.tokenized_prompt_mask.sum(1).tolist(), obs.token_loss_mask.sum(1).tolist()

emb, input_mask, ar_mask = model.embed_inputs(obs)
attn = make_attn_mask(input_mask, ar_mask)
logits, targets, loss_mask = target_logits(model, obs)
per_sample = cross_entropy(logits, targets, loss_mask)
loss = per_sample.mean()
params = select_trainable(model, "full")
opt = make_optimizer(params, cfg)
ema = EMA(model, cfg.ema_decay)
before = {n: p.detach().clone() for n, p in model.named_parameters()}
loss.backward()
grad_norm = clip_and_step(params, opt, cfg, 2)
ema.update(model)
step_norm = float(torch.sqrt(sum(((p.detach() - before[n]) ** 2).sum() for n, p in model.named_parameters())))
first_loss = [int(torch.nonzero(loss_mask[b])[0]) for b in range(B)]
n_full = sum(p.numel() for p in model.parameters())
lora = apply_lora(Pi0FAST(*tiny()))
n_lora = sum(p.numel() for n, p in lora.named_parameters() if "lora" in n)
n_train_lora = sum(p.numel() for p in select_trainable(lora, "lora"))
n_img = sum(p.numel() for p in lora.img.parameters())
with torch.device("meta"):
    pm = apply_lora(Pi0FAST(*paper()))
p_lora = lora_param_count(pm.cfg)

ENC = [
    ("0 FASTObservation (train)\n../data build_fast_batch", f"tokenized_prompt i64[2,180]\n  real {n_real}\ntoken_ar_mask: postfix 1\ntoken_loss_mask: postfix\n  loss tokens {n_loss}\nimages 3x f32[2,224,224,3]", "#e8e8e8"),
    ("1 embed_inputs\n(../model) L159-L195", f"emb f32[2,948,64]\ninput_mask valid {input_mask.sum(1).tolist()}\nar_mask sum {ar_mask.sum(1).tolist()}\nmake_attn_mask bool[2,948,948]\n(prefix bidir, postfix causal)", "#e8e8e8"),
    ("2 drop the last token\nL215-L218", "emb[:, :-1]  f32[2,947,64]\nmask[:, :-1, :-1]\npositions[:, :-1]\nlast token predicts\nnothing", "#cfe2f3"),
    ("3 Gemma forward\n(pi0/vlm), return_prelogits", "pre_logits f32[2,947,64]\nleft-aligned, no cache,\nno right alignment\n(teacher forcing)", "#e8e8e8"),
    ("4 logits at targets only\nL222-L226", f"pre_logits[:, -179:]\n-> logits_head\nf32[2,179,{PALIGEMMA_VOCAB_SIZE}]\n(not 947: the 768 image\n positions predict nothing)", "#cfe2f3"),
    ("5 shift by one\nL209-L213, L231", f"targets = tokens[:, 1:]\n  i64[2,179]\nloss_mask = loss[:, 1:]\n  bool[2,179], sum {loss_mask.sum(1).tolist()}\nfirst loss pos {first_loss}\n  target id {[int(targets[b, first_loss[b]]) for b in range(B)]} ('A')", "#cfe2f3"),
    ("6 cross entropy\nL227-L233", f"log_softmax, gather target\n* loss_mask, / count per sample\nper sample {[f'{v:.2f}' for v in per_sample.tolist()]}\nmean {loss:.2f}\n(random init; ln V = {np.log(PALIGEMMA_VOCAB_SIZE):.2f})", "#cfe2f3"),
    ("7 backward + clip\n(pi0/train) clip_and_step", f"grad over {n_full:,} params\nglobal norm {grad_norm:.2f}\nclipped to 1.0\n(optimizer.py L74)", "#e8e8e8"),
    ("8 AdamW + EMA\n(pi0/train)", f"lr_at(step 2) = {lr_at(2, cfg):.1e}\n(warmup then constant)\nAdamW (0.9, 0.95)\n|delta params| {step_norm:.2e}\nEMA 0.99 (paper 0.999)", "#d9ead3"),
]
DEC = [
    ("full fine-tune\nconfig.py L493 nnx.Nothing", f"trainable = all {n_full:,} (tiny)\n= 2,923,335,408 (paper)\nEMA on", "#d9ead3"),
    ("LoRA: apply_lora\ngemma_fast.py L53-L72", f"per head, rank 16, alpha 16\n+{n_lora:,} params (tiny)\n+{p_lora:,} (paper)\nA, B ~ N(0, 0.01^2)", "#cfe2f3"),
    ("LoRA: select_trainable\npi0_fast.py L127-L131", f"freeze llm minus lora\ntrainable {n_train_lora:,} (tiny)\n= lora + SigLIP {n_img:,}\npaper: 27,869,184 + 414,803,696\nEMA off (L741)", "#cfe2f3"),
]

fig, ax = plt.subplots(figsize=(27.5, 8.0))
ax.set_xlim(0, 27.5)
ax.set_ylim(0, 8.0)
ax.axis("off")
BOX_W, BOX_H, DEC_H, GAP, X0 = 2.75, 2.45, 1.95, 0.25, 0.3
Y_ENC, Y_DEC = 4.5, 1.6


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=9, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=7.6, family="monospace")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1))


for i, (t, b, c) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_ENC + BOX_H / 2, x, Y_ENC + BOX_H / 2)
x7 = X0 + 7 * (BOX_W + GAP)
for i, (t, b, c) in enumerate(DEC):
    x = x7 - (2 - i) * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, DEC_H, t, b, c)
    if i:
        arrow(x - GAP, Y_DEC + DEC_H / 2, x, Y_DEC + DEC_H / 2)
arrow(x7 + BOX_W / 2, Y_DEC + DEC_H, x7 + BOX_W / 2, Y_ENC)
ax.text(x7 + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + DEC_H) / 2, "which params\nreceive the step", fontsize=8, va="center")
ax.text(x7 - 2 * (BOX_W + GAP), Y_DEC + DEC_H + 0.35, "parameter side (bottom row): what a step updates", fontsize=9, style="italic", va="bottom")
ax.text(X0, Y_DEC - 0.35, "blue = pi0-FAST increment (pi0_fast.py, lora.py), grey = ../model, ../data, pi0/train reused, green = the update.  Line refs: openpi@215abfb pi0_fast.py unless noted.",
        fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) loss_mask comes from ../data and marks the postfix incl. 'Action: ', '|', EOS;  (ii) the logits head is the tied embedding, so CE updates the 2048 action rows;  "
        "(iii) the training positions are prefill_len+step, the ../model decode positions are +1 (README Sec. 8 of ../model);  (iv) images get gradient only through attention, never through the loss directly.",
        fontsize=8.5, va="top")
ax.text(X0, Y_ENC + BOX_H + 0.45, "pi0-FAST training step: next-token CE on the action tokens only, logits formed at target positions only  "
        f"(tiny config, LIBERO-scale: {H}-step {d}-dim chunk, max_token_len 180, batch {B})", fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "loss", float(loss.detach()), "grad_norm", grad_norm, "n_lora", n_lora, "paper lora", p_lora)
