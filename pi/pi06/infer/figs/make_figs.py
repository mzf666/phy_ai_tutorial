"""Generate figs/eval.png: (a) the throughput arithmetic: successes per hour as a function of success rate and mean
episode duration, with the five tasks' time limits marked (Sec. VI-A / VI-C; no paper numbers are plotted);
(b) CFG on the tiny model: how far x_0(beta) moves from x_0(1) as beta grows, with the paper's beta range (App. E);
(c) the per-iteration data budget of each task (Sec. VI-C.2, App. F).
Run: uv run python pi/pi06/infer/figs/make_figs.py
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
from pi.pi06.backbone.model import AdaRMSNorm  # noqa: E402
from pi.pi06.data.data import ACTION_DIM  # noqa: E402
from pi.pi06.infer.eval import EPISODES_PER_ITERATION, TASKS  # noqa: E402
from pi.pi06.infer.model import CFG_BETA_RANGE, STATIC_CAMERAS, static_obs_to_raw, tiny_pi06_policy  # noqa: E402

OUT = pathlib.Path(__file__).with_name("eval.png")
fig, (a, b, c) = plt.subplots(1, 3, figsize=(19, 5.4))

# (a) throughput = 3600 * p / mean duration (s): one curve per success rate
dur = np.linspace(30, 700, 200)
for p_, col in ((0.5, "#cc0000"), (0.7, "#e69138"), (0.9, "#6aa84f"), (1.0, "#3d85c6")):
    a.plot(dur, 3600 * p_ / dur, color=col, label=f"success rate {p_:.0%}")
for lim, label in ((200, "200 s: laundry, strict T-shirt, espresso"), (500, "500 s: diverse laundry"), (600, "600 s: box assembly")):
    a.axvline(lim, color="#999", ls=":", lw=1)
    a.text(lim - 4, 59, label, fontsize=6.5, rotation=90, va="top", ha="right", color="#555")
a.set_xlabel("mean episode duration (s), failures included")
a.set_ylabel("throughput (successes / hour)")
a.set_title("(a) throughput = 3600 x success rate / mean duration (Sec. VI-C):\nspeed and reliability in one number; dotted = task time limits (Sec. VI-A)", fontsize=9)
a.set_ylim(0, 60)
a.legend(fontsize=7.5)
a.grid(alpha=0.3)

# (b) CFG on the tiny model
rng = np.random.default_rng(0)
policy, robot = tiny_pi06_policy(max_new_tokens=3)
for m in policy.model.modules():
    if isinstance(m, AdaRMSNorm):
        torch.nn.init.normal_(m.modulation.weight, std=0.05)
obs = {name: rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for name in STATIC_CAMERAS}
obs["state"] = rng.uniform(-0.5, 0.5, 14).astype(np.float32)
raw = static_obs_to_raw(obs, "make me an espresso")
noise = torch.randn(1, robot.action_horizon, ACTION_DIM)
ref = policy.infer(raw, 0.0, noise)["x_0"]
betas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
diffs = []
for beta in betas:
    policy.beta = beta
    diffs.append(float((policy.infer(raw, 0.0, noise)["x_0"] - ref).pow(2).mean().sqrt()))
policy.beta = 1.0
b.plot(betas, diffs, "o-", color="#3d85c6")
b.axvspan(*CFG_BETA_RANGE, color="#d9ead3", alpha=0.6, label=f"paper range beta in {list(CFG_BETA_RANGE)} (App. E)")
b.axvline(1.0, color="#333", ls="--", lw=1, label="beta = 1: deployment default (I = True, Sec. V-D)")
b.set_xlabel("beta")
b.set_ylabel("rms |x_0(beta) - x_0(1)| (normalised action units)")
b.set_title("(b) CFG on the tiny model (expert gates perturbed, untrained):\nv = v_u + beta (v_c - v_u), Eq. 13; the shift grows linearly-ish with beta", fontsize=9)
b.legend(fontsize=7.5)
b.grid(alpha=0.3)

# (c) data per iteration
names = list(EPISODES_PER_ITERATION)
auto = [EPISODES_PER_ITERATION[n]["autonomous"] for n in names]
corr = [EPISODES_PER_ITERATION[n]["corrections"] for n in names]
y = np.arange(len(names))
c.barh(y, auto, color="#9fc5e8", edgecolor="#444", label="autonomous episodes / iteration")
c.barh(y, corr, left=auto, color="#f4cccc", edgecolor="#444", label="episodes with expert corrections / iteration")
for i, n in enumerate(names):
    r = EPISODES_PER_ITERATION[n]
    c.text(auto[i] + corr[i] + 10, i, f"{auto[i]} + {corr[i]}" + (f", {r['robots']} robots" if r.get("robots") else "") + (f"\n({r['note'][:40]})" if r.get("note") else ""), fontsize=6.5, va="center")
c.set_yticks(y, [n.replace(" (", "\n(") for n in names], fontsize=7.5)
c.set_xlim(0, 1700)
c.set_xlabel("episodes per RECAP iteration")
c.set_title("(c) experience collected per iteration (Sec. VI-C.2, App. F)", fontsize=9)
c.legend(fontsize=7, loc="lower right")
c.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT, "cfg diffs", [round(d, 3) for d in diffs])
