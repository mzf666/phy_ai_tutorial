"""Generate figs/pipeline.png: one denoising step of the pi0.5 action expert, step by step (top), and the pi0 expert's
corresponding step underneath so every replaced piece is visible (bottom). Values from one real tiny run.
Run: uv run python pi/pi05/expert/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, ActionProjections, MoEGemma, posemb_sincos, tiny_experts  # noqa: E402
from pi.pi0.vlm.model import make_attn_mask  # noqa: E402
from pi.pi05.expert.model import AdaMoEGemma, AdaRMSNorm, Pi05ActionProjections, expert_param_count, suffix_forward  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
torch.manual_seed(0)
B = 1
vlm_cfg, exp_cfg = tiny_experts()
llm = AdaMoEGemma((vlm_cfg, exp_cfg)).eval()
proj = Pi05ActionProjections(exp_cfg).eval()
pi0_proj = ActionProjections(exp_cfg).eval()
n = lambda m: sum(p.numel() for p in m.parameters())
prefix_emb = torch.randn(B, 60, vlm_cfg.width)
prefix_mask = torch.ones(B, 60, dtype=torch.bool)
x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
tau = torch.full((B,), 0.7)
with torch.no_grad():
    (_, _), kv = llm([prefix_emb, None], prefix_mask.long().cumsum(1) - 1, make_attn_mask(prefix_mask, torch.zeros(60, dtype=torch.bool)))
    phi = posemb_sincos(tau, exp_cfg.width, 4e-3, 4.0)
    cond = proj.time_cond(tau)
    tokens, smask, sar, _ = proj.embed_suffix(x_t, tau)
    h, gate = llm.layers[0].experts[1].pre_attention_norm(tokens, cond)
    out0 = suffix_forward(llm, kv, prefix_mask, tokens, smask, sar, cond)
    v0 = proj.decode(out0)
    for m in llm.modules():
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    mod = llm.layers[0].experts[1].pre_attention_norm.modulation(cond)[0].view(3, -1)
    out1 = suffix_forward(llm, kv, prefix_mask, tokens, smask, sar, cond)
    v1 = proj.decode(out1)
    pi0_tokens, _, pi0_ar = pi0_proj.embed_suffix(torch.randn(B, ACTION_DIM), x_t, tau)


def v3(x):
    return " ".join(f"{float(v):+.2f}" for v in x.reshape(-1)[:3])


ENC = [
    ("0 inputs", f"x_t f32[1,50,32]:\n  {v3(x_t[0,0])}\ntau f32[1] = {float(tau[0]):.1f}\nprefix kv cache: 4 layers\n  x [1,60,1,16] (expert 0)\nNO state input here\n(it is text in the prompt)", "#e8e8e8"),
    ("1 phi(tau) = posemb_sincos\npi0.py L161 (as pi0)", f"f32[1,{exp_cfg.width}]\n{v3(phi[0])} ...\nperiods 4e-3 .. 4.0", "#e8e8e8"),
    ("2 time MLP -> cond\nL164-L167; App. E", f"swish(W2 swish(W1 phi))\ncond f32[1,{exp_cfg.width}]\n{v3(cond[0])} ...\nrms {cond.pow(2).mean().sqrt():.3f}\none vector per sample", "#cfe2f3"),
    ("3 action_in_proj(x_t)\nL159, L168", f"tokens f32[1,50,{exp_cfg.width}]\n{v3(tokens[0,0])} ...\nno concat with time,\nno time MLP on tokens\nar_mask [1, 0 x 49] (L182)", "#cfe2f3"),
    ("4 AdaRMSNorm (per layer x2)\ngemma.py L113-L131", f"normed*(1+scale)+shift\n(scale,shift,gate) =\n  Dense(cond) zero-init\nat init all 0:\n  |scale| {float(mod[0].abs().max()):.2f} after perturb\nreturns gate [1,1,{exp_cfg.width}]", "#cfe2f3"),
    ("5 gated residual\nL311, L330, L453-L459", "x + gate * attn(norm(x))\nx + gate * mlp(norm(x))\ngate = 0 at init\n-> every layer identity\n(expert 0: plain residual)", "#cfe2f3"),
    ("6 final AdaRMSNorm\nL382, L410", f"out f32[1,50,{exp_cfg.width}]\nat init == plain_norm(\n  action_in_proj(x_t)):\n  {torch.allclose(out0, llm._plain_final(tokens), atol=1e-5)}\ngate discarded", "#cfe2f3"),
    ("7 action_out_proj\nL212 / L269", f"v_t f32[1,50,32]\nall 50 tokens decoded\ninit: {v3(v0[0,0])}\nperturbed: {v3(v1[0,0])}", "#d9ead3"),
]
PI0 = [
    ("0 inputs (pi0)", "x_t, tau AND state f32[1,32]\n(state_proj -> 1 token)", "#e8e8e8"),
    ("1 phi(tau)", "same posemb_sincos", "#e8e8e8"),
    ("2 concat(action, time)\npi0.py L172-L177", f"[1,50,2w] -> mlp_in, swish,\nmlp_out -> [1,50,w]\ntau enters ONCE, here", "#f4cccc"),
    ("3 suffix tokens", f"[state | 50 actions] = 51\nar_mask [1, 1, 0 x 49]\n{v3(pi0_tokens[0,1])} ...", "#f4cccc"),
    ("4 RMSNorm (learned scale)\ngemma.py L119-L125", "normed * (1 + scale)\nno cond, no gate", "#f4cccc"),
    ("5 plain residual", "x + attn(norm(x))\nx + mlp(norm(x))\nrandom init mixes prefix\ninto actions from step 0", "#f4cccc"),
    ("6 final RMSNorm", "[1,51,w]", "#f4cccc"),
    ("7 action_out_proj", "last 50 of 51 tokens\n(state token dropped)", "#d9ead3"),
]

fig, ax = plt.subplots(figsize=(25, 8.4))
ax.set_xlim(0, 25)
ax.set_ylim(0, 8.4)
ax.axis("off")
BOX_W, BOX_H, DEC_H, GAP, X0 = 2.82, 2.45, 1.75, 0.24, 0.3
Y_ENC, Y_DEC = 4.9, 2.2


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + 0.1, y + h - 0.72, body, ha="left", va="top", fontsize=7.2, family="monospace")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1))


for i, (t, b, c) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_ENC + BOX_H / 2, x, Y_ENC + BOX_H / 2)
for i, (t, b, c) in enumerate(PI0):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, DEC_H, t, b, c)
    if i:
        arrow(x - GAP, Y_DEC + DEC_H / 2, x, Y_DEC + DEC_H / 2)
ax.text(X0, Y_ENC + BOX_H + 0.55, f"pi0.5 action expert, one denoising step (tiny: expert width {exp_cfg.width}, 4 layers; paper: width 1024, 18 layers): top = pi0.5, bottom = pi0's same step for contrast",
        fontsize=10.5, weight="bold", va="bottom")
ax.text(X0, Y_DEC + DEC_H + 0.12, "pi0 (pi.pi0.action_expert), red = what pi0.5 replaces", fontsize=9, va="bottom", color="#990000")
ax.text(X0, Y_DEC - 0.35, "blue = pi0.5 increment (openpi@215abfb pi0.py / gemma.py line refs), grey = unchanged from pi0, green = output.  "
        f"Params at paper size: expert1 {expert_param_count(exp_cfg.__class__(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256, vocab_size=1)):,} (pi0 311,464,960); projections 2,165,792 (pi0 3,248,160).",
        fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) cond is computed once per denoising step and shared by all 2 x 18 + 1 adaptive norms;  (ii) the same prefix kv cache serves all 10 steps (unchanged);  "
        "(iii) at zero init the expert output does not depend on tau or on the prefix, so post-training starts from 'the VLM plus a passive expert'.", fontsize=8.5, va="top")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT)
