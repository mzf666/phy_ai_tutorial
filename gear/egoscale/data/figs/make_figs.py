"""Generate figs/padding.png: what cross-embodiment padding costs, and why action_mask cannot be
dropped -- (a) how much of each embodiment's action vector is padding, (b) the same prediction error
scored with the masked denominator vs a naive mean over the padded width, (c) the state mask, where
the human branch is entirely False.
Run: uv run python gear/egoscale/data/figs/make_figs.py
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
from gear.egoscale.data.data import (  # noqa: E402
    padding_fraction,
    prepare_action,
    prepare_state,
    tiny,
)

ORDER = ("human_wild", "human_aligned", "r1pro_sharpa", "g1_trifinger")
C_NATIVE = "#2c5f8a"
C_PAD = "#c9c9c9"
C_MASKED = "#d0503f"
C_NAIVE = "#e0a33e"


def losses(cfg, seed=0):
    """Same per-dimension error everywhere; only the denominator differs.

    masked  = (err^2 * mask).sum() / mask.sum()          <- upstream flow_matching_action_head L341-L343
    naive   = (err^2 * mask).mean()                      <- averaging over the padded width instead
    The padded dims contribute zero error (they are zero in both prediction and target), so the
    naive denominator inflates by exactly 1 / (1 - padding fraction).
    """
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name in ORDER:
        emb = cfg.embodiments[name]
        err = torch.randn(cfg.action_horizon, emb.action_dim, generator=g)
        padded, mask, _ = prepare_action(err, cfg.max_action_dim, cfg.action_horizon)
        sq = padded**2 * mask
        out[name] = (float(sq.sum() / mask.sum()), float(sq.mean()))
    return out


def main():
    cfg = tiny()
    frac = padding_fraction(cfg)
    loss = losses(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 6.2),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    x = np.arange(len(ORDER))
    labels = [n.replace("_", "\n") for n in ORDER]

    # --- (a) how much of the action vector is padding -----------------------
    ax = axes[0]
    native = [cfg.embodiments[n].action_dim for n in ORDER]
    pad = [cfg.max_action_dim - d for d in native]
    ax.bar(x, native, color=C_NATIVE, label="native action dims")
    ax.bar(x, pad, bottom=native, color=C_PAD, label="zero padding")
    for i, n in enumerate(ORDER):
        ax.text(i, cfg.max_action_dim + 1.2, f"{frac[n]:.0%} padding",
                ha="center", fontsize=10, fontweight="bold",
                color=C_MASKED if frac[n] > 0.2 else "#555555")
        ax.text(i, native[i] / 2, str(native[i]), ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, cfg.max_action_dim + 6)
    ax.set_ylabel(f"dims in the padded action vector (max_action_dim = {cfg.max_action_dim})",
                  fontsize=10)
    ax.set_title("(a) one width for four embodiments", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # --- (b) why the mask is the loss denominator ---------------------------
    ax = axes[1]
    masked = [loss[n][0] for n in ORDER]
    naive = [loss[n][1] for n in ORDER]
    ax.bar(x - 0.2, masked, width=0.38, color=C_MASKED,
           label="(err^2 * mask).sum() / mask.sum()   <- upstream")
    ax.bar(x + 0.2, naive, width=0.38, color=C_NAIVE,
           label="(err^2 * mask).mean()   <- naive, wrong denominator")
    for i, n in enumerate(ORDER):
        ax.text(i + 0.2, naive[i] + 0.02, f"x{naive[i] / masked[i]:.2f}",
                ha="center", fontsize=9.5, color="#7a5a10", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("loss on identical per-dimension error", fontsize=10)
    ax.set_title("(b) the same error, two denominators\n"
                 "the naive one shrinks by exactly (1 - padding fraction)",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, max(masked + naive) * 1.35)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # --- (c) the state mask -------------------------------------------------
    ax = axes[2]
    rows = []
    for name in ORDER:
        emb = cfg.embodiments[name]
        state = None if not emb.has_proprio else torch.randn(cfg.state_horizon, emb.state_dim)
        _, mask, _ = prepare_state(state, cfg.max_state_dim, cfg.state_horizon)
        rows.append(mask[0].numpy().astype(float))
    ax.imshow(np.stack(rows), aspect="auto", cmap="Blues", vmin=0, vmax=1.4,
              interpolation="nearest")
    ax.set_yticks(x)
    ax.set_yticklabels([n.replace("_", " ") for n in ORDER], fontsize=10)
    ax.set_xlabel(f"state dimension (max_state_dim = {cfg.max_state_dim})", fontsize=10)
    for i, name in enumerate(ORDER):
        emb = cfg.embodiments[name]
        note = ("no proprioception -> learnable placeholder token (paper Sec. 2.3)"
                if not emb.has_proprio else f"{emb.state_dim} real dims, rest padded")
        ax.text(1.0, i, note, ha="left", va="center", fontsize=8.8,
                color="white" if emb.has_proprio else "#7a4fa3", fontweight="bold")
    ax.set_title("(c) state_mask: the human rows are entirely False",
                 fontsize=12.5, fontweight="bold")

    fig.suptitle("EgoScale Sec. 2.3: padding is what lets humans and two robots share one tensor -- "
                 "action_mask is what keeps their losses comparable",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "(b) uses identical i.i.d. errors on every real dimension, so any difference between "
             "the bars comes only from the denominator. Dropping the mask would make the G1's loss "
             "look best simply because half its vector is zeros it is not asked to predict. "
             "max_action_dim / max_state_dim are undisclosed; tiny uses "
             f"{cfg.max_action_dim} / {cfg.max_state_dim} (README Sec. 8).",
             ha="center", va="bottom", fontsize=9.2, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])

    out = pathlib.Path(__file__).with_name("padding.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
