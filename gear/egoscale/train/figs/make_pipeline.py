"""Generate figs/pipeline.png: the three stages left to right (top row -- data, steps, batch, lr and
the parameters that are actually trainable, measured on a real tiny run), and what each stage does
NOT do (bottom row -- the frozen set, and the gap it leaves that the next stage exists to close).
Run: uv run python gear/egoscale/train/figs/make_pipeline.py
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
from gear.egoscale.train.train import (  # noqa: E402
    STAGE1,
    STAGE2,
    ScalingLaw,
    _tiny_models,
    run_stage,
    stage3,
)

FWD = "#2c5f8a"
FRZ = "#8a5a2c"


def run_tiny():
    torch.manual_seed(0)
    bcfg, dcfg, backbone, expert = _tiny_models()
    backbone.train()
    expert.train()
    total = sum(p.numel() for p in list(backbone.parameters()) + list(expert.parameters()))
    out = {}
    for st in (STAGE1, STAGE2, stage3(True)):
        out[st.name] = run_stage(backbone, expert, st, bcfg, dcfg, n_steps=2)
    return total, out


def box(ax, x, y, w, h, title, shape, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", linewidth=1.8,
                                edgecolor=color, facecolor=color + "10", mutation_aspect=0.55))
    ax.text(x + w / 2, y + h - 0.034, title, ha="center", va="top",
            fontsize=12, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.086, shape, ha="center", va="top",
            fontsize=9.5, family="monospace", color="#333333")
    ax.text(x + 0.014, y + h - 0.140, "\n".join(body), ha="left", va="top",
            fontsize=9.0, family="monospace", color="#444444", linespacing=1.65)


def arrow(ax, x0, y0, x1, y1, color, label=""):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=color, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, y0 + 0.022, label, ha="center", va="bottom",
                fontsize=8.6, color=color, linespacing=1.3)


def main():
    total, runs = run_tiny()
    law = ScalingLaw.paper()

    fig, ax = plt.subplots(figsize=(22.0, 11.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n, gap = 3, 0.035
    w = (1 - 0.04 - (n - 1) * gap) / n
    xs = [0.02 + i * (w + gap) for i in range(n)]
    yf, yi, hh = 0.505, 0.105, 0.355

    stages = (STAGE1, STAGE2, stage3(True))
    titles = ("Stage I - human pretraining", "Stage II - aligned mid-training",
              "Stage III - post-training")
    extras = (
        ["256 GB200 GPUs (paper Sec. 2.4)",
         "fully unfreezing EVERY parameter",
         "gives the general manipulation prior",
         f"paper law: L(20k h) = {float(law.predict(20)):.4f}"],
        ["freezes the vision-language backbone",
         "(the LLM half); vision encoder, DiT and",
         "the state/action adapters keep training",
         "grounds the prior in robot sensing"],
        ["vision encoder frozen BECAUSE stage II",
         "ran; unfrozen otherwise, to make room",
         "for a new embodiment (paper Sec. 2.4)",
         "one demo is enough for some tasks"],
    )
    for st, title, extra, x in zip(stages, titles, extras, xs):
        r = runs[st.name]
        tr = sum(r["trainable"].values())
        body = [
            f"{st.data}",
            f"steps {st.steps:,}   batch {st.batch_size:,}   lr {st.lr:.0e}",
            f"trainable in this repo's tiny run:",
            f"  {tr:,} / {total:,}  ({tr / total:.0%})",
            "",
        ] + extra
        box(ax, x, yf, w, hh, title, f"tune: llm={st.tune_llm} vis={st.tune_visual} "
            f"dit={st.tune_dit} proj={st.tune_projector}", body, FWD)
    arrow(ax, xs[0] + w, yf + hh / 2, xs[1], yf + hh / 2, FWD, "same\nloss")
    arrow(ax, xs[1] + w, yf + hh / 2, xs[2], yf + hh / 2, FWD, "same\nloss")

    gaps = (
        ("what stage I does NOT give you", "frozen: nothing", [
            "the human action space is aligned only",
            "in the abstract: no robot proprioception,",
            "no wrist cameras in the wild data,",
            "no guarantee the commands are executable",
            "-> that is what stage II is for",
            "(paper Sec. 3.2: pretrain-only already",
            " beats midtrain-only on most tasks)",
        ]),
        ("what stage II does NOT give you", "frozen: backbone.llm", [
            "only ~4 hours of robot data and 344",
            "generic play tasks -- nothing about the",
            "five evaluation tasks specifically",
            "-> that is what stage III is for",
            "but the motion primitives it does expose",
            "are enough for ONE-SHOT transfer",
            "(paper Sec. 3.4: 0.88 on Fold Shirt)",
        ]),
        ("what stage III does NOT give you", "frozen: llm + vision + connector", [
            "no new perception: the vision tower is",
            "frozen once mid-training has run, so all",
            "adaptation happens in the DiT and the",
            "embodiment adapters",
            "NOTE: backbone.post (vlln + VL self-attn)",
            "is never frozen by any switch upstream",
            "-> README Sec. 1.x row 5",
        ]),
    )
    for (title, shape, body), x in zip(gaps, xs):
        box(ax, x, yi, w, hh, title, shape, body, FRZ)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, yi + hh / 2, b, yi + hh / 2, FRZ)

    ax.text(0.5, 0.985, "EgoScale three-stage curriculum: one objective, three data sets, "
                        "three freezing sets",
            ha="center", va="top", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.945,
            "top row = what each stage trains     bottom row = what it deliberately leaves out, "
            "which is why the next stage exists     "
            "every step / batch / lr below is from paper Sec. 2.4; the trainable counts are "
            "measured on this repo's tiny models",
            ha="center", va="top", fontsize=10.5, color="#555555")
    ax.text(0.5, 0.062,
            "The loss never changes across the three stages -- unlike pi0.5, which switches "
            "objective between its two stages. What changes is the data, the learning rate, the "
            "batch size and the frozen set.     "
            "The optimizer, schedule, warmup, weight decay, gradient clipping and mixture weights "
            "are ALL undisclosed (README Sec. 8); the tiny run uses AdamW only so that it can run.",
            ha="center", va="center", fontsize=9.6, color="#222222",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3f3f3", edgecolor="#bcbcbc"))
    ax.text(0.5, 0.012,
            "paper arXiv:2602.16710v1 Sec. 2.4 / Sec. 3.2 / App. D.1   |   freezing switches after "
            "Isaac-GR00T@4af2b62 eagle_backbone.py L65-L94, flow_matching_action_head.py L217-L254",
            ha="center", va="center", fontsize=8.8, color="#777777")

    all_bodies = [c for _, _, c in gaps] + [
        [f"{st.data}",
         f"steps {st.steps:,}   batch {st.batch_size:,}   lr {st.lr:.0e}"] + extra
        for st, extra in zip(stages, extras)
    ]
    over = [ln for body in all_bodies for ln in body if len(ln) > 46]
    assert not over, f"这些行会溢出框: {over}"

    out = pathlib.Path(__file__).with_name("pipeline.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
