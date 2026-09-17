"""Generate figs/scaling.png: the paper's log-linear law, read three ways --
(a) the units in Eq. (1) cannot be hours: taken literally the law goes negative;
(b) what the law extrapolates to, and what it would cost to keep going;
(c) offline validation loss and real-robot score move together across the five runs -- over five
    points, and as a correlation only.
Run: uv run python gear/egoscale/train/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.train.train import (  # noqa: E402
    FIG5_CENTER,
    FIG5_RIGHT,
    PAPER_R2,
    ScalingLaw,
    paper_law_in_hours,
)

C_PAPER = "#d0503f"
C_FIT = "#2c5f8a"
C_OBS = "#e0a33e"
C_BAD = "#8f8f8f"


def main():
    law = ScalingLaw.paper()
    fitted = ScalingLaw.fit(list(FIG5_CENTER), list(FIG5_CENTER.values()))

    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.6),
                             gridspec_kw={"width_ratios": [1.15, 1.1, 1.0]})

    # --- (a) the units ------------------------------------------------------
    ax = axes[0]
    grid_k = np.geomspace(0.5, 40, 200)
    ax.plot(grid_k, law.predict(grid_k), color=C_PAPER, linewidth=2.2,
            label="Eq. (1) with D in THOUSANDS of hours")
    ax.plot(grid_k, paper_law_in_hours(grid_k * 1000), color=C_BAD, linewidth=2.0,
            linestyle="--", label="Eq. (1) with D taken literally, in hours")
    ax.scatter(list(FIG5_CENTER), list(FIG5_CENTER.values()), color=C_OBS, s=90, zorder=5,
               edgecolor="white", linewidth=1.2, label="Fig. 5 centre, READ OFF the plot")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}k"))
    bad20 = float(paper_law_in_hours(20854))
    ax.annotate(f"L(20,854 h) = {bad20:.4f} < 0\na loss cannot be negative",
                xy=(20, bad20), xytext=(1.1, 0.0035), ha="left",
                fontsize=9.5, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, linewidth=1.3))
    ax.set_xlabel("human pretraining data", fontsize=10)
    ax.set_ylabel("human validation loss (MSE)", fontsize=10)
    ax.set_title("(a) Eq. (1) says 'hours', but only 'thousands of hours' works\n"
                 "README Sec. 1.x row 1", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)

    # --- (b) extrapolation --------------------------------------------------
    ax = axes[1]
    grid = np.geomspace(1, 400, 300)
    ax.plot(grid, law.predict(grid), color=C_PAPER, linewidth=2.2,
            label=f"paper: 0.024 - 0.003 ln D  (R2 = {PAPER_R2})")
    ax.plot(grid, fitted.predict(grid), color=C_FIT, linewidth=1.8, linestyle="--",
            label=f"least squares on the read-off points:\n"
                  f"{fitted.intercept:.5f} - {fitted.slope:.5f} ln D  "
                  f"(R2 = {fitted.r2:.4f})")
    ax.scatter(list(FIG5_CENTER), list(FIG5_CENTER.values()), color=C_OBS, s=90, zorder=5,
               edgecolor="white", linewidth=1.2)
    ax.axvspan(1, 20.854, color="#2c5f8a", alpha=0.07)
    ax.text(4.5, law.predict(1.2), "measured range\n(1k - 20.854k h)", fontsize=9.5,
            color="#2c5f8a", ha="center")
    for d in (40, 100, 200):
        ax.scatter([d], [float(law.predict(d))], color=C_PAPER, s=45, marker="x")
        ax.text(d, float(law.predict(d)) - 0.0012, f"{d}k\n{float(law.predict(d)):.4f}",
                fontsize=8.8, ha="center", color=C_PAPER)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10, 20, 40, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}k"))
    ax.set_xlabel("human pretraining data (thousands of hours, log scale)", fontsize=10)
    ax.set_ylabel("human validation loss (MSE)", fontsize=10)
    need = law.hours_for(0.010)
    ax.set_title("(b) every doubling buys a fixed drop\n"
                 f"reaching L = 0.010 would take {need:,.0f}k hours "
                 f"({need / 20.854:.1f}x today's data)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=8.8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # --- (c) offline loss vs real-robot score -------------------------------
    ax = axes[2]
    ds = sorted(FIG5_RIGHT)
    scores = [FIG5_RIGHT[d] for d in ds]
    ax.plot(ds, scores, color=C_FIT, marker="o", linewidth=2.0, markersize=8,
            label="avg. task completion score (Fig. 5 right)")
    for d, s in zip(ds, scores):
        ax.text(d, s + 0.03, f"{s:.2f}", ha="center", fontsize=9.5, fontweight="bold",
                color=C_FIT)
    ax.set_xscale("log")
    ax.set_xticks(ds)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}k"))
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("human pretraining data (thousands of hours)", fontsize=10)
    ax.set_ylabel("avg. task completion score", fontsize=10, color=C_FIT)
    ax.tick_params(axis="y", colors=C_FIT)

    ax2 = ax.twinx()
    ax2.plot(list(FIG5_CENTER), list(FIG5_CENTER.values()), color=C_PAPER, marker="s",
             linewidth=2.0, markersize=7, linestyle="--",
             label="validation loss (Fig. 5 centre, read off)")
    ax2.set_ylabel("human validation loss (MSE)", fontsize=10, color=C_PAPER)
    ax2.tick_params(axis="y", colors=C_PAPER)
    ax2.invert_yaxis()
    ax.set_title("(c) the two move together -- over FIVE points\n"
                 "the paper reports a correlation, not a causal claim",
                 fontsize=12.5, fontweight="bold")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8.8, loc="lower right")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    fig.suptitle("EgoScale Sec. 3.3: a log-linear scaling law, its units, its extrapolation, "
                 "and how much it is really claiming",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "The orange points are READ OFF Fig. 5 centre (about +-0.0003); the paper publishes "
             "only the fitted line and its R2, so any conclusion drawn from them inherits that "
             "precision. The 1k point is cropped in the published figure and is not included. "
             "The scores in (c) are printed on the bars by the paper. See README Sec. 8.",
             ha="center", va="bottom", fontsize=9.2, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])

    out = pathlib.Path(__file__).with_name("scaling.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
