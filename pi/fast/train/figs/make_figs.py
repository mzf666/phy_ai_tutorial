"""Generate figs/train.png: (a) the paper's warmup-then-constant schedule vs openpi's LIBERO cosine schedule;
(b) which input positions carry loss after the shift by one, from one real tiny batch; (c) trainable parameters,
full vs LoRA, at paper size.
Run: uv run python pi/fast/train/figs/make_figs.py
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
from pi.fast.model.model import paper  # noqa: E402
from pi.fast.tokenizer.tokenizer import QuantileStats, make_smooth_chunks  # noqa: E402
from pi.fast.train.train import libero_config, lora_param_count, paper_config  # noqa: E402
from pi.pi0.data.data import make_bool_mask  # noqa: E402
from pi.pi0.train.train import lr_at  # noqa: E402

OUT = pathlib.Path(__file__).with_name("train.png")

# ---- (b) real tiny batch: token / target / loss layout after the shift ----------------------------------------------
rng = np.random.default_rng(0)
B, H, d = 1, 10, 7
seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, d), max_len=180)
raw = {"images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
       "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
       "actions": (0.3 * make_smooth_chunks(B, H + 2, d, rng)).astype(np.float32),
       "prompt": ["pick up the red block"]}
stats = {"state": QuantileStats(np.full(d, -1.0), np.full(d, 1.0)), "actions": QuantileStats(np.full(d, -1.0), np.full(d, 1.0))}
obs, _ = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=False)
tok_mask, ar, lm = obs.tokenized_prompt_mask[0].numpy(), obs.token_ar_mask[0].numpy(), obs.token_loss_mask[0].numpy()
L = len(tok_mask)
cls = np.where(~tok_mask, 0, np.where(ar == 0, 1, 2))  # 0 pad, 1 prefix, 2 postfix
n_pre, n_post = int((cls == 1).sum()), int((cls == 2).sum())
ZOOM = (n_pre - 6, n_pre + n_post + 4)

fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(19, 4.6), gridspec_kw={"width_ratios": [1.1, 1.5, 0.9]})

# ---- (a) schedules -------------------------------------------------------------------------------------------------
steps = np.arange(0, 100_001, 250)
pc, lc = paper_config(), libero_config()
axa.plot(steps, [lr_at(s, pc) for s in steps], color="#990000", label="paper App. C / pi0_fast_full_droid_finetune: warmup 1k -> 5e-5 constant")
axa.plot(steps, [lr_at(s, lc) for s in steps], color="#3d85c6", label="openpi pi0_fast_libero: 2.5e-5 cosine -> 2.5e-6 at 30k (global default)")
axa.axvline(lc.num_train_steps, color="#3d85c6", lw=0.8, ls="--")
axa.axvline(pc.num_train_steps, color="#990000", lw=0.8, ls="--")
axa.text(lc.num_train_steps, 5.3e-5, " 30k", color="#3d85c6", fontsize=8)
axa.text(pc.num_train_steps, 5.3e-5, "100k ", color="#990000", fontsize=8, ha="right")
axa.set_ylim(0, 5.8e-5)
axa.set_xlabel("step")
axa.set_ylabel("learning rate")
axa.set_title("(a) two schedules in play (paper: DROID 240k, LIBERO 40k steps)", fontsize=9.5)
axa.legend(fontsize=7, loc="center right", frameon=False)

# ---- (b) loss positions ------------------------------------------------------------------------------------------------
colors = {0: "#f3f3f3", 1: "#cfe2f3", 2: "#f9cb9c"}
rows = [("input token j", cls), ("target = token j+1", np.append(cls[1:], 0)), ("loss_mask[j+1]", np.where(np.append(lm[1:], False), 3, 0))]
colors[3] = "#e06666"
for r, (name, arr) in enumerate(rows):
    for j in range(ZOOM[0], ZOOM[1]):
        axb.add_patch(plt.Rectangle((j, 2 - r), 1, 0.8, color=colors[int(arr[j])], ec="white", lw=0.5))
    axb.text(ZOOM[0] - 0.5, 2 - r + 0.4, name, ha="right", va="center", fontsize=8.5)
axb.set_xlim(ZOOM[0] - 9, ZOOM[1])
axb.set_ylim(-0.3, 3.4)
axb.set_yticks([])
axb.set_xlabel(f"token position j inside the 180-token sequence (image positions 0..767 come before; prefix {n_pre} tokens, postfix {n_post})")
axb.set_title(f"(b) after the shift, position j predicts token j+1: {n_post} loss positions = the last prefix token + {n_post - 1} postfix tokens", fontsize=9.5)
for c, lab in ((1, "prefix (prompt + state)"), (2, "postfix (Action: ... | EOS)"), (0, "pad"), (3, "in the loss")):
    axb.bar(0, 0, color=colors[c], label=lab)
axb.legend(fontsize=7.5, loc="upper left", ncol=4, frameon=False)
axb.axvline(n_pre - 1 + 0.5, color="#333", lw=0.8, ls="--")
axb.text(n_pre - 1 + 0.5, 3.0, "last prefix token predicts 'A' of 'Action: '", fontsize=7.5, ha="center")

# ---- (c) trainable parameters at paper size ---------------------------------------------------------------------------
_, gcfg = paper()
n_lora = lora_param_count(gcfg)
full = 2_923_335_408
siglip = 414_803_696
bars = [("full fine-tune\ntrainable", full, "#990000"), ("LoRA: trainable\n(lora + SigLIP)", n_lora + siglip, "#e06666"), ("LoRA: lora\nonly", n_lora, "#f9cb9c"), ("LoRA: frozen\n(llm base)", full - siglip, "#cccccc")]
for i, (name, v, c) in enumerate(bars):
    axc.bar(i, v, color=c)
    axc.text(i, v * 1.15, f"{v/1e6:,.1f}M" if v < 1e9 else f"{v/1e9:.2f}B", ha="center", fontsize=8)
axc.set_yscale("log")
axc.set_ylim(1e7, 1e10)
axc.set_yticks([1e7, 1e8, 1e9, 1e10], labels=["10M", "100M", "1B", "10B"])
axc.set_xticks(range(len(bars)))
axc.set_xticklabels([b[0] for b in bars], fontsize=7.5)
axc.set_ylabel("parameters (log)")
axc.set_title("(c) what a step updates, paper size", fontsize=9.5)
fig.suptitle("openpi@215abfb pi0_fast.py compute_loss L197-L233, get_freeze_filter L127-L131; config.py L699-L742, L831-L860; lora.py", fontsize=9, y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT, "prefix", n_pre, "postfix", n_post, "lora", n_lora)
