"""Generate figs/action_spaces.png: the three human pretraining action spaces of paper Sec. 3.6 --
their vector layout (left), how much of the hand each one actually pins down, measured by running
the same retargeting NLP with full vs fingertip-only supervision (middle), and the task completion
scores the paper reports for them (right).
Run: uv run python gear/egoscale/action/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.action.action import (  # noqa: E402
    _FINGERTIP_IDX,
    ACTION_SPACES,
    N_JOINTS,
    N_KEYPOINTS_ROBOT,
    ActionConfig,
    RetargetWeights,
    ToyHand22,
    action_dim,
    retarget_chunk,
    tiny,
)

# paper arXiv:2602.16710v1 Fig. 8, task completion score, read off the published bars.
FIG8 = {
    "Wrist Only": {"Card": 0.56, "Tong": 0.24, "Bottle": 0.26},
    "Fingertip": {"Card": 0.17, "Tong": 0.76, "Bottle": 0.55},
    "Full (Retargeted Joints)": {"Card": 0.74, "Tong": 0.79, "Bottle": 0.61},
}
COLORS = {"Wrist Only": "#8e6fc4", "Fingertip": "#e0a33e", "Full (Retargeted Joints)": "#d0503f"}
SLICE_COLORS = {"wrist": "#2c5f8a", "fingertips": "#e0a33e", "joints": "#d0503f"}


def exact_cfg() -> ActionConfig:
    """Position term only, no filtering: isolates what the supervision itself determines."""
    return ActionConfig(**{
        **tiny().__dict__,
        "alpha": 1.0,
        "weights": RetargetWeights(w_pos=1.0, w_smooth=0.0, w_reg=0.0),
    })


def supervision_experiment():
    """Same NLP, same targets, only the supervised keypoint set differs.

    Full: all 20 keypoints (paper App. D). Fingertip: only the 5 TIP keypoints, which is all the
    Sec. 3.6 fingertip representation carries before its MLP.
    """
    hand = ToyHand22()
    rng = np.random.default_rng(3)
    q_star = hand.clamp(hand.q_rest + 0.35 * rng.normal(size=(24, N_JOINTS)))
    kp = np.stack([hand.fk(q) for q in q_star])

    tips_only = np.zeros(N_KEYPOINTS_ROBOT)
    tips_only[_FINGERTIP_IDX] = 1.0

    q_full, _ = retarget_chunk(kp, hand, exact_cfg())
    q_tips, _ = retarget_chunk(kp, hand, exact_cfg(), kp_weight=tips_only)
    return hand, np.abs(q_full - q_star).mean(0), np.abs(q_tips - q_star).mean(0)


def main():
    hand, err_full, err_tips = supervision_experiment()
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1.25, 1.0]})

    # --- (a) vector layout -------------------------------------------------
    ax = axes[0]
    layout = {
        "wrist_only": [("wrist", 9)],
        "fingertip": [("wrist", 9), ("fingertips", 45)],
        "full": [("wrist", 9), ("joints", 22)],
    }
    for row, space in enumerate(ACTION_SPACES):
        left = 0.0
        for name, width in layout[space]:
            ax.barh(row, width, left=left, height=0.5, color=SLICE_COLORS[name],
                    edgecolor="white", linewidth=1.2)
            ax.text(left + width / 2, row, f"{name}\n{width}", ha="center", va="center",
                    fontsize=9.5, color="white", fontweight="bold")
            left += width
        ax.text(left + 1.5, row, f"x 2 hands = {action_dim(space, 'rotation_6d')}",
                ha="left", va="center", fontsize=10.5, fontweight="bold", color="#333333")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["wrist_only", "fingertip", "full\n(EgoScale default)"], fontsize=11)
    ax.set_xlim(0, 78)
    ax.set_xlabel("dims per hand (rotation_6d: 3 translation + 6 rotation per pose)", fontsize=10)
    ax.set_title("(a) what each action space carries", fontsize=12.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # --- (b) supervision sufficiency --------------------------------------
    ax = axes[1]
    idx = np.arange(N_JOINTS)
    ax.bar(idx - 0.2, err_tips, width=0.4, color=SLICE_COLORS["fingertips"],
           label="supervise 5 fingertips only")
    ax.bar(idx + 0.2, err_full, width=0.4, color=SLICE_COLORS["joints"],
           label="supervise all 20 keypoints")
    ax.set_yscale("log")
    # text.parse_math is off, so the default LaTeX-style log tick labels would print raw
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0e}"))
    ax.set_ylim(top=10 ** np.ceil(np.log10(err_tips.max())) * 3)
    ax.set_xticks(idx[::2])
    ax.set_xlabel("joint index (thumb 0-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-21)",
                  fontsize=10)
    ax.set_ylabel("mean |q - q*| over 24 random poses  (rad, log scale)", fontsize=10)
    ax.set_title(
        f"(b) same NLP, same targets, different supervision\n"
        f"fingertips only: {err_tips.mean():.3f} rad mean error   |   "
        f"all keypoints: {err_full.mean():.1e} rad",
        fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # --- (c) paper numbers -------------------------------------------------
    ax = axes[2]
    tasks = ["Card", "Tong", "Bottle"]
    x = np.arange(len(tasks))
    for i, (rep, scores) in enumerate(FIG8.items()):
        vals = [scores[t] for t in tasks]
        bars = ax.bar(x + (i - 1) * 0.27, vals, width=0.26, color=COLORS[rep], label=rep)
        ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("task completion score", fontsize=10)
    ax.set_title("(c) paper arXiv:2602.16710v1 Fig. 8\n(reported, NOT reproduced here)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "EgoScale Sec. 3.6: the hand action space decides how much finger information the "
        "pretraining signal carries",
        fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "(b) is a mechanism check on this repo's ToyHand22, not a paper number: five fingertip "
             "positions leave whole joints unconstrained, so mapping them back to joint commands is "
             "under-determined -- which is the failure mode Sec. 3.6 reports as unstable grasps. "
             "Joint limits and the URDF are toy values (README Sec. 8).",
             ha="center", va="bottom", fontsize=9.2, color="#555555")

    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    out = pathlib.Path(__file__).with_name("action_spaces.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
