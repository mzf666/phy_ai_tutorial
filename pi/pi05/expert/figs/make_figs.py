"""Generate figs/adarms.png: (a) how much the expert's hidden state depends on tau after each layer, for pi0.5 (adaRMSNorm,
modulation perturbed to N(0, 0.05^2) since it is exactly 0 at init) and for pi0 (tau injected once at the entry); (b) the
paper-size parameter breakdown of both experts. Tiny configs, one real run.
Run: uv run python pi/pi05/expert/figs/make_figs.py
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
from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, ActionProjections, MoEGemma, tiny_experts  # noqa: E402
from pi.pi0.vlm.model import GemmaConfig, make_attn_mask  # noqa: E402
from pi.pi05.expert.model import AdaMoEGemma, AdaRMSNorm, Pi05ActionProjections  # noqa: E402

OUT = pathlib.Path(__file__).with_name("adarms.png")
torch.manual_seed(0)
B = 4
vlm_cfg, exp_cfg = tiny_experts()
prefix_emb = torch.randn(B, 60, vlm_cfg.width)
prefix_mask = torch.ones(B, 60, dtype=torch.bool)
pmask = make_attn_mask(prefix_mask, torch.zeros(60, dtype=torch.bool))
ppos = prefix_mask.long().cumsum(1) - 1
x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
state = torch.randn(B, ACTION_DIM)
taus = (1.0, 0.5)


def per_layer(llm, tokens, smask, sar, kv, cond=None):
    """Hidden state of expert 1 after each layer (before the final norm)."""
    b, s = tokens.shape[:2]
    mask = torch.cat([prefix_mask[:, None, :].expand(-1, s, -1), make_attn_mask(smask, sar)], -1)
    pos = prefix_mask.long().sum(-1, keepdim=True) + smask.long().cumsum(-1) - 1
    xs, outs = [None, tokens], []
    for i, layer in enumerate(llm.layers):
        xs, _ = layer(xs, pos, mask, kv[i], cond) if cond is not None or isinstance(llm, AdaMoEGemma) else layer(xs, pos, mask, kv[i])
        outs.append(xs[1].clone())
    return outs


with torch.no_grad():
    # pi0.5: zero init (identity) and perturbed modulation
    llm5 = AdaMoEGemma((vlm_cfg, exp_cfg)).eval()
    proj5 = Pi05ActionProjections(exp_cfg).eval()
    (_, _), kv5 = llm5([prefix_emb, None], ppos, pmask)
    curves = {}
    for label, std in (("pi0.5, zero init", 0.0), ("pi0.5, modulation ~ N(0, 0.05^2)", 0.05), ("pi0.5, modulation ~ N(0, 0.2^2)", 0.2)):
        for m in llm5.modules():
            if isinstance(m, AdaRMSNorm):
                torch.nn.init.normal_(m.modulation.weight, std=std) if std > 0 else torch.nn.init.zeros_(m.modulation.weight)
        hs = []
        for tau in taus:
            tokens, smask, sar, cond = proj5.embed_suffix(x_t, torch.full((B,), tau))
            hs.append(per_layer(llm5, tokens, smask, sar, kv5, cond))
        curves[label] = [float((a - b).pow(2).mean().sqrt() / (a.pow(2).mean().sqrt() + 1e-8)) for a, b in zip(*hs)]
    # pi0: entry injection, random init
    llm0 = MoEGemma((vlm_cfg, exp_cfg)).eval()
    proj0 = ActionProjections(exp_cfg).eval()
    (_, _), kv0 = llm0([prefix_emb, None], ppos, pmask)
    hs = []
    for tau in taus:
        tokens, smask, sar = proj0.embed_suffix(state, x_t, torch.full((B,), tau))
        hs.append(per_layer(llm0, tokens, smask, sar, kv0))
    curves["pi0 (entry injection), random init"] = [float((a - b).pow(2).mean().sqrt() / (a.pow(2).mean().sqrt() + 1e-8)) for a, b in zip(*hs)]

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
layers = np.arange(1, exp_cfg.depth + 1)
for label, ys in curves.items():
    a.plot(layers, ys, marker="o", label=label, ls="--" if label.startswith("pi0 ") else "-")
a.set_xticks(layers)
a.set_xlabel("after layer")
a.set_ylabel("|h(tau=1) - h(tau=0.5)| / |h|   (expert 1 hidden state)")
a.set_title("(a) tau-sensitivity per layer, tiny expert (width 32, 4 layers)\nopenpi@215abfb gemma.py L113-L131, pi0.py L159-L169", fontsize=9.5)
a.legend(fontsize=8)
a.grid(alpha=0.3)

paper = GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256, vocab_size=1)
w, f = paper.width, paper.mlp_dim
attn = 18 * (w * 8 * 256 + w * 2 * 256 + 8 * 256 * w)
mlp = 18 * (2 * w * f + f * w)
norm_pi0 = 37 * w
ada = 37 * (3 * w * w + 3 * w)
proj_pi0, proj_pi05 = 3_248_160, 2_165_792
parts = {"attention q / kv / out": (attn, attn), "GeGLU": (mlp, mlp), "norms": (norm_pi0, ada), "projections": (proj_pi0, proj_pi05)}
left = np.zeros(2)
for name, (p0, p5) in parts.items():
    b.barh(["pi0 expert + proj", "pi0.5 expert + proj"], [p0, p5], left=left, label=name, edgecolor="#444")
    left += np.array([p0, p5])
for y, tot in enumerate(left):
    b.text(tot + 5e6, y, f"{int(tot):,}", va="center", fontsize=8)
b.set_xlim(0, left.max() * 1.25)
b.set_xlabel("parameters (paper size)")
b.set_title("(b) 37 adaptive norms add 116.5M (paper still says '300M')\nREADME Sec. 1.5", fontsize=9.5)
b.legend(fontsize=8, loc="lower right")
b.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for k, v in curves.items():
    print(k, [f"{x:.3f}" for x in v])
print("totals", left)
