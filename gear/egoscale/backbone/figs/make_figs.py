"""Generate figs/select_layer.png: what the two structural knobs of the backbone actually buy --
(a) taking the middle layer removes parameters, it does not just skip a forward pass,
(b) pixel shuffle and the number of camera slots set the sequence length the DiT must cross-attend
    to, and
(c) the disclosed numbers, each with its source, including which ones are NOT EgoScale's.
Run: uv run python gear/egoscale/backbone/figs/make_figs.py
"""

import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.backbone.model import (  # noqa: E402
    N15_SOURCES,
    VisionLanguageBackbone,
    n15,
    tiny,
)

C_KEPT = "#2c5f8a"
C_DROP = "#c9c9c9"
C_LAT = "#d0503f"


def sweep_select_layer():
    """Real measurements on the tiny config: parameters kept and measured forward time."""
    base = tiny()
    ks, params, ms = [], [], []
    images = torch.randn(2, base.n_views, 3, *base.image_hw)
    view_mask = torch.ones(2, base.n_views, dtype=torch.bool)
    ids = torch.randint(0, base.vocab, (2, 7))
    attn = torch.ones(2, 7, dtype=torch.bool)
    for k in range(1, base.llm_layers_total + 1):
        cfg = tiny()
        cfg.select_layer = k
        m = VisionLanguageBackbone(cfg).eval()
        with torch.no_grad():
            m(images, view_mask, ids, attn)  # warm up
            t0 = time.perf_counter()
            for _ in range(20):
                m(images, view_mask, ids, attn)
            dt = (time.perf_counter() - t0) / 20
        ks.append(k)
        params.append(sum(p.numel() for p in m.parameters()))
        ms.append(1e3 * dt)
    return np.array(ks), np.array(params), np.array(ms)


def sequence_lengths():
    """Where the DiT's cross-attention sequence length comes from."""
    rows = []
    for shuffle in (False, True):
        cfg = tiny()
        cfg.use_pixel_shuffle = shuffle
        m = VisionLanguageBackbone(cfg)
        for n_views in (1, 3):
            rows.append((shuffle, n_views, m.tokens_per_view,
                         n_views * m.tokens_per_view, 7))
    return rows


def main():
    ks, params, ms = sweep_select_layer()
    cfg = tiny()

    fig, axes = plt.subplots(1, 3, figsize=(20.5, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})

    # --- (a) parameters and latency vs select_layer -------------------------
    ax = axes[0]
    full = params[-1]
    ax.bar(ks, params, color=C_KEPT, label="parameters actually constructed")
    ax.bar(ks, full - params, bottom=params, color=C_DROP,
           label="never constructed (popped at build)")
    ax.axvline(cfg.select_layer, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(cfg.select_layer + 0.12, full * 0.46, f"tiny uses\nselect_layer={cfg.select_layer}",
            ha="left", va="center", fontsize=9.5, color="#444444")
    ax.set_xlabel(f"select_layer (LLM has {cfg.llm_layers_total} layers in the tiny config)",
                  fontsize=10)
    ax.set_ylabel("total backbone parameters", fontsize=10)
    ax.set_title("(a) the middle layer is a STRUCTURAL choice\n"
                 "upstream pops the layers above it (eagle_backbone.py L59-L60)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(ks, ms, color=C_LAT, marker="o", linewidth=1.8, label="measured forward (ms)")
    ax2.set_ylabel("measured forward time, ms (CPU, batch 2)", fontsize=10, color=C_LAT)
    ax2.tick_params(axis="y", colors=C_LAT)
    ax2.set_ylim(0, max(ms) * 1.4)
    ax2.legend(fontsize=9, loc="upper left")

    # --- (b) sequence length ------------------------------------------------
    ax = axes[1]
    rows = sequence_lengths()
    labels = [f"{'shuffle' if s else 'no shuffle'}\n{v} view(s)" for s, v, *_ in rows]
    img_tok = [r[3] for r in rows]
    txt_tok = [r[4] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, img_tok, color=C_KEPT, label="image tokens")
    ax.bar(x, txt_tok, bottom=img_tok, color="#e0a33e", label="text tokens")
    for i, (im, tx) in enumerate(zip(img_tok, txt_tok)):
        ax.text(i, im + tx + 0.8, f"{im + tx}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, max(a + b for a, b in zip(img_tok, txt_tok)) * 1.25)
    ax.set_ylabel("cross-attention sequence length the DiT sees", fontsize=10)
    ax.set_title("(b) where the sequence length comes from\n"
                 "black-filled slots still occupy tokens",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # --- (c) disclosed numbers with their sources ---------------------------
    ax = axes[2]
    ax.axis("off")
    n = n15()
    lines = [
        ("EgoScale (arXiv:2602.16710v1)", "", "#7a4fa3"),
        ("  backbone structure", "NOT DISCLOSED anywhere in the paper", "#7a4fa3"),
        ("  Sec. 2.3 says", "'similar to GR00T N1'", "#7a4fa3"),
        ("  App. D.1 says", "adapters 'following GR00T-N1 and N1.5'", "#7a4fa3"),
        ("", "", "#000000"),
        ("GR00T N1 (arXiv:2503.14734v2 Sec. 2.1)", "", C_KEPT),
        ("  image tokens per frame", "64  (224x224 + pixel shuffle)", C_KEPT),
        ("  LLM layer used", "12th", C_KEPT),
        ("  total / VLM parameters", "2.2B / 1.34B", C_KEPT),
        ("  end-to-end, 16-step chunk", "63.9 ms on an L40", C_KEPT),
        ("", "", "#000000"),
        ("GR00T-N1.5-3B config.json (2026-09-17)", "", C_LAT),
    ]
    for key in N15_SOURCES:
        lines.append((f"  {key}", repr(getattr(n, key)), C_LAT))
    lines += [
        ("", "", "#000000"),
        ("None of the rows below the first block", "", "#555555"),
        ("are EgoScale's disclosed values.", "See README Sec. 8.", "#555555"),
    ]
    for i, (k, v, c) in enumerate(lines):
        y = 0.94 - i * 0.050
        ax.text(0.0, y, k, fontsize=9.2, family="monospace", color=c,
                fontweight="bold" if not k.startswith("  ") and k else "normal")
        ax.text(0.56, y, v, fontsize=9.2, family="monospace", color=c)
    ax.set_title("(c) every number with its source", fontsize=12.5, fontweight="bold")

    fig.suptitle("EgoScale backbone: the two structural knobs (which LLM layer, how many image "
                 "tokens) and where their values actually come from",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "(a) and (b) are measured on this repo's tiny config, not paper numbers. The paper "
             "reports that the middle layer is both faster and better downstream (GR00T N1 Sec. 2.1); "
             "this repo cannot reproduce that claim without the real checkpoint, so it is quoted in "
             "(c) rather than plotted.",
             ha="center", va="bottom", fontsize=9.2, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])

    out = pathlib.Path(__file__).with_name("select_layer.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
