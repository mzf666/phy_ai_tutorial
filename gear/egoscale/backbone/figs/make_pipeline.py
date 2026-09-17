"""Generate figs/pipeline.png: three camera views + one instruction -> the conditioning vectors the
DiT cross-attends to (top row), and what is and is not invertible / cacheable on the way back
(bottom row). Values come from one real tiny run.
Run: uv run python gear/egoscale/backbone/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.backbone.model import (  # noqa: E402
    N15_SOURCES,
    VisionLanguageBackbone,
    n15,
    tiny,
)

FWD = "#2c5f8a"
INV = "#8a5a2c"
DEAD = "#8f8f8f"


def run_tiny():
    torch.manual_seed(0)
    cfg = tiny()
    m = VisionLanguageBackbone(cfg).eval()
    b, lang = 2, 7
    images = torch.randn(b, cfg.n_views, 3, *cfg.image_hw)
    view_mask = torch.tensor([[True, True, True], [True, False, False]])
    ids = torch.randint(0, cfg.vocab, (b, lang))
    attn = torch.ones(b, lang, dtype=torch.bool)
    attn[1, :2] = False
    with torch.no_grad():
        img = m.encode_images(images)
        h, valid = m.encode_sequence(images, view_mask, ids, attn)
        out = m(images, view_mask, ids, attn)
    full = tiny()
    full.select_layer = full.llm_layers_total
    dropped = sum(p.numel() for p in VisionLanguageBackbone(full).parameters()) - \
        sum(p.numel() for p in m.parameters())
    return dict(cfg=cfg, m=m, lang=lang, img=img, h=h, valid=valid, out=out, dropped=dropped)


def box(ax, x, y, w, h, title, shape, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", linewidth=1.6,
                                edgecolor=color, facecolor=color + "10", mutation_aspect=0.45))
    ax.text(x + w / 2, y + h - 0.030, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.076, shape, ha="center", va="top",
            fontsize=8.8, family="monospace", color="#333333")
    ax.text(x + 0.010, y + h - 0.122, "\n".join(body), ha="left", va="top",
            fontsize=8.2, family="monospace", color="#444444", linespacing=1.6)


def arrow(ax, x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color, shrinkA=0, shrinkB=0))


def main():
    d = run_tiny()
    cfg, m = d["cfg"], d["m"]
    n_img = cfg.n_views * m.tokens_per_view
    seq = n_img + d["lang"]
    npar = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731

    fig, ax = plt.subplots(figsize=(30.0, 10.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n, gap = 7, 0.011
    w = (1 - 0.02 - (n - 1) * gap) / n
    xs = [0.01 + i * (w + gap) for i in range(n)]
    yf, yi, hh = 0.555, 0.145, 0.275

    fwd = [
        ("1. three camera views", f"images (B,{cfg.n_views},3,{cfg.image_hw[0]},{cfg.image_hw[1]})", [
            "head + left wrist + right wrist",
            "fixed slot order (paper Sec. 2.5)",
            "black-filled slots STILL get tokens,",
            "so the sequence length is constant",
            "tiny uses 32x32 (upstream 224x224)",
        ], FWD),
        ("2. vision tower", f"-> (B*V,{m.vision.n_patch},{cfg.d_vit})", [
            f"patch {cfg.patch} -> grid {m.vision.grid}",
            f"{m.vision.n_patch} patch tokens per view",
            f"{cfg.vit_layers} blocks, {cfg.vit_heads} heads",
            f"params {npar(m.vision):,}",
            "weights are GIVEN, not trained here",
        ], FWD),
        ("3. pixel shuffle", f"-> (B*V,{m.tokens_per_view},{cfg.d_vit * 4})", [
            "space -> channel REARRANGEMENT",
            f"ratio {cfg.downsample_ratio}: tokens /4, width x4",
            "no interpolation, no element lost",
            "N1.5 turns this OFF ('nops')",
            "upstream L297-L309",
        ], FWD),
        ("4. MLP connector", f"-> (B,{n_img},{cfg.d_llm})", [
            f"{cfg.mlp_connector_layers}-layer variant (LayerNorm + 2 Linear)",
            "N1.5 uses the 1-layer variant",
            f"params {npar(m.connector):,}",
            f"{cfg.n_views} views x {m.tokens_per_view} = {n_img} image tokens",
            "upstream L138-L156",
        ], FWD),
        ("5. merge with text", f"-> (B,{seq},{cfg.d_llm})", [
            f"{n_img} image + {d['lang']} text tokens",
            "text tokenizer pads on the LEFT",
            "black slots + pads -> mask False",
            f"valid per sample {d['valid'].sum(1).tolist()}",
            "upstream transforms.py L51, L172-L177",
        ], FWD),
        ("6. LLM, middle layer", f"h (B,{seq},{cfg.d_llm})", [
            f"{cfg.llm_layers_total} layers total, take layer {cfg.select_layer}",
            "layers above it are POPPED at build",
            f"saves {d['dropped']:,} params (not just a fwd)",
            f"kept params {npar(m.llm):,}",
            "N1.5: select_layer = 12",
        ], FWD),
        ("7. VL post-process", f"phi_t (B,{seq},{cfg.d_llm})", [
            "LayerNorm + self-attention blocks",
            "use_vlln switches BOTH or NEITHER",
            "NO mask here (upstream takes none)",
            f"params {npar(m.post):,}; N1.5: 4 layers",
            f"phi std {d['out']['backbone_features'].std():.3f}",
        ], FWD),
    ]
    for (title, shape, body, color), x in zip(fwd, xs):
        box(ax, x, yf, w, hh, title, shape, body, color)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, yf + hh / 2, b, yf + hh / 2, FWD)

    inv = [
        ("1'. what the DiT gets", "phi_t + its mask", [
            "cross-attention keys/values, plus",
            "encoder_attention_mask to drop pads",
            "THIS is where masking finally bites",
            f"mask valid {d['valid'].sum(1).tolist()} / {seq}",
            "see ../dit",
        ], INV),
        ("2'. computed ONCE per chunk", "cache across K steps", [
            "phi_t does not depend on the noise",
            "level, so the K denoising steps reuse it",
            "this is the payoff of cross-attention",
            "over concatenating vision tokens into",
            "the DiT's own self-attention",
        ], INV),
        ("3'. mask leaks here", "unmasked self-attention", [
            "the VL self-attention takes NO mask,",
            "so pads and black slots DO influence",
            "valid positions before the DiT ever",
            "sees them (upstream behaviour, pinned",
            "by a test) -- README Sec. 1.x row 5",
        ], DEAD),
        ("4'. dropped layers: gone", "layers > select_layer", [
            "they are never constructed, so there",
            "is no way to recover the final-layer",
            "representation from this model",
            "the choice is structural, not a flag",
            "upstream eagle_backbone.py L59-L60",
        ], DEAD),
        ("5'. pixel shuffle: invertible", "rearrangement", [
            "element multiset is unchanged, so an",
            "exact inverse exists -- but nothing in",
            "the pipeline needs it: the connector",
            "that follows is not invertible",
            "",
        ], INV),
        ("6'. vision weights: frozen", "stage dependent", [
            "stage I: everything unfrozen",
            "stage II: LLM frozen, vision updated",
            "stage III: vision frozen iff mid-trained",
            "frozen modules MUST be set to eval()",
            "paper Sec. 2.4; upstream L65-L94",
        ], INV),
        ("7'. pretraining: out of scope", "checkpoint is given", [
            "SigLIP2-400M + Qwen3-1.7B (N1.5) or",
            "SigLIP-2 + SmolLM2 (N1) -- their",
            "recipes are stated as facts only",
            "EgoScale never says which one it used",
            "README Sec. 8",
        ], DEAD),
    ]
    for (title, shape, body, color), x in zip(inv, xs[::-1]):
        box(ax, x, yi, w, hh, title, shape, body, color)
    for a, b in zip(xs[::-1][:-1], xs[::-1][1:]):
        arrow(ax, a, yi + hh / 2, b + w, yi + hh / 2, INV)

    ax.text(0.5, 0.988, "EgoScale conditioning path: three views + one instruction -> phi_t",
            ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax.text(0.5, 0.950,
            "top row = forward (shared by training and inference)     bottom row = what happens to "
            "phi_t downstream and what cannot be undone, grey = lossy or out of scope     "
            "numbers from one real tiny run",
            ha="center", va="top", fontsize=10, color="#555555")
    ax.text(0.5, 0.078,
            "Coupling that must be kept together: the camera slot order, the per-view token count "
            "and backbone_attention_mask are one unit -- reorder the slots and every learned "
            "position convention breaks.     "
            "EgoScale discloses NONE of these structural values; the tiny run uses toy sizes, and "
            f"the N1.5 column (select_layer={n15().select_layer}, d_llm={n15().d_llm}, "
            f"vl layers={n15().vl_attn_layers}) comes from the released checkpoint's config.json, "
            "not from the paper (README Sec. 8).",
            ha="center", va="center", fontsize=9.4, color="#222222",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3f3f3", edgecolor="#bcbcbc"))
    ax.text(0.5, 0.014,
            "paper arXiv:2602.16710v1 Sec. 2.3   |   Isaac-GR00T@4af2b62 "
            "gr00t/model/backbone/eagle_backbone.py L29-L133, "
            "eagle2_hg_model/modeling_eagle2_5_vl.py L138-L156 L297-L339   |   "
            f"N1.5 values: {len(N15_SOURCES)} fields from nvidia/GR00T-N1.5-3B config.json "
            "(accessed 2026-09-17)",
            ha="center", va="center", fontsize=8.6, color="#777777")

    over = [ln for _, _, body, _ in fwd + inv for ln in body if len(ln) > 43]
    assert not over, f"这些行会溢出框: {over}"

    out = pathlib.Path(__file__).with_name("pipeline.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
