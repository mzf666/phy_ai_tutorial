"""Generate figs/eval.png: the three things that decide what an EgoScale number means --
(a) additive vs progress-based rubrics have different shapes, and progress is NOT additive;
(b) three of the paper's additive rubrics do not sum to 1 while the text says scores are in [0,1];
(c) the 16-sample average in the validation protocol is not optional: the scaling law's adjacent
    points are ~0.0025 apart, which single-sample noise would swamp.
Run: uv run python gear/egoscale/infer/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.infer.eval import (  # noqa: E402
    N_SAMPLES_PER_TIMESTEP,
    TASKS,
    averaged_prediction,
    rubric_total,
    rubric_warnings,
    score_additive,
    score_progress,
)
from gear.egoscale.train.train import FIG5_CENTER  # noqa: E402

C_ADD = "#2c5f8a"
C_PROG = "#d0503f"
C_WARN = "#e0a33e"


def main():
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.6),
                             gridspec_kw={"width_ratios": [1.2, 1.05, 1.1]})

    # --- (a) the two scoring shapes ----------------------------------------
    ax = axes[0]
    add_task, prog_task = TASKS["syringe"], TASKS["shirt"]
    add_names = [k for k, v in add_task.rubric if v > 0]
    prog_names = [k for k, v in prog_task.rubric]

    add_curve = [score_additive(add_task, set(add_names[: i + 1]))
                 for i in range(len(add_names))]
    prog_curve = [score_progress(prog_task, set(prog_names[: i + 1]))
                  for i in range(len(prog_names))]
    ax.step(range(1, len(add_curve) + 1), add_curve, where="post", color=C_ADD,
            linewidth=2.2, marker="o", label=f"additive: {add_task.name} (7 items)")
    ax.step(range(1, len(prog_curve) + 1), prog_curve, where="post", color=C_PROG,
            linewidth=2.2, marker="s", linestyle="--",
            label=f"progress: {prog_task.name} (5 milestones)")

    # progress is not additive: two separate milestones do not add up
    early, late = {prog_names[1]}, {prog_names[3]}
    ax.scatter([2], [score_progress(prog_task, early | late)], s=160, facecolor="none",
               edgecolor=C_PROG, linewidth=2.0, zorder=5)
    ax.annotate(f"reaching milestones 1 and 3 gives "
                f"{score_progress(prog_task, early | late):.1f},\n"
                f"not {score_progress(prog_task, early) + score_progress(prog_task, late):.1f} "
                f"-- progress takes the FURTHEST one",
                xy=(2, score_progress(prog_task, early | late)), xytext=(3.4, 0.08),
                fontsize=9.3, color=C_PROG, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_PROG, linewidth=1.3))
    ax.set_xlabel("milestones reached, in order", fontsize=10)
    ax.set_ylabel("task completion score", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("(a) two scoring strategies, two different shapes\n"
                 "paper App. B", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # --- (b) rubrics that do not sum to 1 -----------------------------------
    ax = axes[1]
    additive = [t for t in TASKS.values() if t.kind == "additive"]
    names = [t.name for t in additive]
    totals = [rubric_total(t) for t in additive]
    warn = rubric_warnings()
    colors = [C_WARN if t.key in warn else C_ADD for t in additive]
    y = np.arange(len(additive))
    ax.barh(y, totals, color=colors, height=0.6)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.4)
    for i, (t, total) in enumerate(zip(additive, totals)):
        ax.text(total + 0.02, i, f"{total:.2f}", va="center", fontsize=9.5,
                fontweight="bold", color=colors[i])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t.number}  {n}" for t, n in zip(additive, names)], fontsize=9.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xlabel("sum of the rubric items", fontsize=10)
    ax.set_title("(b) three additive rubrics do not sum to 1\n"
                 "the text says scores are in [0, 1] -- this repo clips",
                 fontsize=12.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # --- (c) why 16 samples ------------------------------------------------
    ax = axes[2]
    g = torch.Generator().manual_seed(0)
    gt = torch.zeros(8, 16)

    def sampler():
        return torch.randn(8, 16, generator=g)

    reps = 400
    spreads = {}
    for n in (1, 2, 4, 8, 16, 32):
        g.manual_seed(1)
        losses = [float(torch.nn.functional.mse_loss(averaged_prediction(sampler, n), gt))
                  for _ in range(reps)]
        spreads[n] = (float(np.mean(losses)), float(np.std(losses)))
    ns = sorted(spreads)
    means = [spreads[n][0] for n in ns]
    stds = [spreads[n][1] for n in ns]
    ax.errorbar(ns, means, yerr=stds, color=C_ADD, marker="o", linewidth=2.0, capsize=5,
                label="mean +- std of the estimated loss")
    ax.plot(ns, [means[0] / n for n in ns], color=C_PROG, linestyle="--", linewidth=1.8,
            label="1/n reference")
    ax.axvline(N_SAMPLES_PER_TIMESTEP, color="#333333", linestyle=":", linewidth=1.4)
    ax.text(N_SAMPLES_PER_TIMESTEP * 1.08, means[0] * 0.55,
            f"the paper uses n = {N_SAMPLES_PER_TIMESTEP}", fontsize=9.5, color="#333333")

    gap = abs(FIG5_CENTER[10] - FIG5_CENTER[20])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    # text.parse_math is off, so the default LaTeX-style log tick labels would print raw
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0e}"))
    ax.set_xlabel("samples averaged per timestep (paper Sec. 3.3)", fontsize=10)
    ax.set_ylabel("estimated loss (arbitrary units, log scale)", fontsize=10)
    ax.set_title("(c) averaging the PREDICTIONS is what shrinks the noise\n"
                 f"the scaling law's adjacent points differ by only ~{gap:.4f}",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("EgoScale evaluation: what a completion score is, where the paper's rubrics do "
                 "not add up, and why the validation protocol averages 16 samples",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "(a) and (b) come straight from App. B; no real-robot number is reproduced anywhere "
             "in this repo. (c) is a property of averaging i.i.d. samples, measured here on "
             "synthetic noise -- it shows why the order matters: average the predictions first, "
             "then take one error, rather than averaging 16 errors. See README Sec. 1.x and Sec. 8.",
             ha="center", va="bottom", fontsize=9.2, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])

    out = pathlib.Path(__file__).with_name("eval.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
