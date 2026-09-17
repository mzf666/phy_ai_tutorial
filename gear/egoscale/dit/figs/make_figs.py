"""Generate figs/dit.png: the three things that make this action expert what it is --
(a) the token layout and which blocks see phi at all,
(b) the flow-matching path: straight line, constant velocity, so an oracle field is exact for any K,
(c) the Beta(1.5, 1) timestep distribution, which puts the training mass on the high-noise end.
Run: uv run python gear/egoscale/dit/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.dit.model import ActionExpert, n15, tiny  # noqa: E402
from gear.egoscale.dit.train import add_noise, oracle_integrate, sample_time  # noqa: E402

C_STATE = "#7a4fa3"
C_FUT = "#e0a33e"
C_ACT = "#d0503f"
C_CROSS = "#2c5f8a"
C_SELF = "#9ab8d0"


def main():
    torch.manual_seed(0)
    cfg = tiny()
    n = n15()
    m = ActionExpert(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.4),
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.05]})

    # --- (a) token layout + which blocks see phi ---------------------------
    ax = axes[0]
    segs = [("state", n.state_horizon, C_STATE),
            ("future_tokens", n.num_target_vision_tokens, C_FUT),
            ("action", n.action_horizon, C_ACT)]
    left = 0
    for name, width, color in segs:
        ax.barh(1.0, width, left=left, height=0.42, color=color, edgecolor="white")
        if width >= 8:
            ax.text(left + width / 2, 1.0, f"{name}\n{width}", ha="center", va="center",
                    fontsize=9.5, color="white", fontweight="bold")
        else:  # 太窄的段放到条外面, 否则文字会被裁掉
            ax.text(left + width / 2, 1.30, f"{name} ({width})", ha="center", va="bottom",
                    fontsize=9.5, color=color, fontweight="bold")
        left += width
    ax.text(left + 0.8, 1.0, f"= {left} tokens\n(N1.5)", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color="#333333")
    ax.annotate("only these are decoded\ninto the velocity",
                xy=(left - n.action_horizon / 2, 0.76), xytext=(left - 8, 0.30),
                fontsize=9.5, color=C_ACT, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_ACT, linewidth=1.4))

    for i in range(n.num_layers):
        cross = i % 2 == 0
        ax.add_patch(Rectangle((i * 3.0, -1.35), 2.6, 0.55,
                               color=C_CROSS if cross else C_SELF))
    ax.text(0, -0.65, f"DiT blocks (N1.5: {n.num_layers} layers), left to right:",
            fontsize=10, color="#333333")
    ax.text(0, -1.62, "dark = cross-attention on phi     light = self-attention only     "
                      "every block gets tau through AdaLN",
            fontsize=9.2, color="#555555")
    ax.set_xlim(-1, 52)
    ax.set_ylim(-2.1, 1.5)
    ax.axis("off")
    ax.set_title("(a) token layout and where phi enters\n"
                 "phi is computed once and reused by all K denoising steps",
                 fontsize=12.5, fontweight="bold")

    # --- (b) the path is a straight line ------------------------------------
    ax = axes[1]
    a = torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(1))
    e = torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(2))
    taus = torch.linspace(0, 1, 101)
    dist_to_action = [float((add_noise(a, e, t.expand(1))[0] - a).norm()) for t in taus]
    dist_to_noise = [float((add_noise(a, e, t.expand(1))[0] - e).norm()) for t in taus]
    ax.plot(taus, dist_to_noise, color="#888888", linewidth=1.8, label="distance to the noise eps")
    ax.plot(taus, dist_to_action, color=C_ACT, linewidth=1.8, label="distance to the action A")
    ax.axvline(0.0, color="#bbbbbb", linestyle=":", linewidth=1.0)
    ax.axvline(1.0, color="#bbbbbb", linestyle=":", linewidth=1.0)
    for step in range(n.num_inference_timesteps):
        t = step / n.num_inference_timesteps
        ax.axvline(t, color=C_CROSS, linestyle="--", linewidth=1.1)
    ax.text(0.5, max(dist_to_noise) * 0.93,
            f"dashed: the K = {n.num_inference_timesteps} taus the sampler visits\n"
            "note it never evaluates tau = 1",
            ha="center", fontsize=9.5, color=C_CROSS)

    errs = [float((oracle_integrate(a, e, k) - a).abs().max()) for k in (1, 2, 4, 8, 50)]
    ax.text(0.5, max(dist_to_noise) * 0.68,
            "oracle velocity A - eps, max |A_hat - A|:\n"
            + "   ".join(f"K={k}: {err:.0e}" for k, err in zip((1, 2, 4, 8, 50), errs)),
            ha="center", fontsize=9.2, color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"))
    ax.set_xlabel("tau", fontsize=10)
    ax.set_ylabel("||A_tau - endpoint||", fontsize=10)
    ax.set_title("(b) the path is a straight line\n"
                 "constant velocity, so any K is exact for an ORACLE field",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # --- (c) the timestep distribution --------------------------------------
    ax = axes[2]
    tau = sample_time(200000, cfg, generator=torch.Generator().manual_seed(4)).numpy()
    ax.hist(tau, bins=80, density=True, color=C_CROSS, alpha=0.75,
            label="sample_time() empirical")
    grid = np.linspace(0.0, 1.0, 400)
    # u = s(1 - tau); density of tau = s * alpha * u^(alpha-1) on the transformed variable
    alpha, s = cfg.noise_beta_alpha, cfg.noise_s
    dens = s * alpha * np.clip(s * (1 - grid), 0, None) ** (alpha - 1)
    ax.plot(grid, dens, color=C_ACT, linewidth=2.0, label="Beta(1.5, 1) pushed through tau=(s-u)/s")
    ax.axvline(tau.mean(), color="#333333", linestyle="--", linewidth=1.3)
    ax.text(tau.mean() + 0.02, ax.get_ylim()[1] * 0.85, f"mean {tau.mean():.3f}",
            fontsize=9.5, color="#333333")
    ax.axvline(0.5, color="#bbbbbb", linestyle=":", linewidth=1.2)
    ax.set_xlabel("tau  (0 = pure noise, 1 = the action)", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.set_title("(c) training mass sits on the HIGH-NOISE end\n"
                 f"{(tau < 0).mean() * 100:.2f}% of samples land at tau < 0 "
                 f"because s = {s} < 1",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("EgoScale / GR00T action expert: what the shared DiT actually does with phi, "
                 "the noise path and the timestep",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "(a) uses the released GR00T-N1.5-3B config (1 + 32 + 16 = 49 tokens, 16 layers), "
             "NOT EgoScale's -- the paper discloses none of it. (b) and (c) are exact properties of "
             "the flow-matching formulation, measured here, not paper numbers. "
             "The velocity sign follows the upstream code (A - eps), not GR00T N1 Eq. (1). "
             "See README Sec. 1.x and Sec. 8.",
             ha="center", va="bottom", fontsize=9.2, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])

    out = pathlib.Path(__file__).with_name("dit.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
