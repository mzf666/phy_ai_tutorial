"""Generate figs/advantage.png: (a) the 201-bin value distribution the head outputs at three steps of a tiny episode
(untrained: near-uniform) next to the Eq. 1 target bins; (b) the two advantage estimators on a SYNTHETIC critic
(V = target + a smooth error, labelled as such): whole-episode (N = T) vs N-step, and where corrections override;
(c) the per-task threshold as a quantile of a synthetic advantage pool: 30% / 40% / 10% positive.
Run: uv run python pi/pi06/value/figs/make_figs.py
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
import pi.pi06.value.train as T  # noqa: E402
from pi.pi06.data.data import C_FAIL_TINY, STATIC_IMAGE_KEYS, EpisodeLabels, bin_values, build_pi06_batch, episode_rewards, tiny_pi06_tokenizer, unit_stats  # noqa: E402
from pi.pi06.value.model import tiny_value_function  # noqa: E402

OUT = pathlib.Path(__file__).with_name("advantage.png")
torch.manual_seed(0)
rng = np.random.default_rng(0)
H, d, T1, Tmax = 10, 7, 12, 40
vf = tiny_value_function().eval()
seq = tiny_pi06_tokenizer(H, d)
lab = EpisodeLabels("fold the shirt", True, Tmax, T1)
R, bins = lab.value_targets(C_FAIL_TINY)
vals = bin_values()

fig, (a, b, c) = plt.subplots(1, 3, figsize=(19, 5.2))
# (a) distributions
for t, col in ((0, "#3d85c6"), (6, "#6aa84f"), (11, "#cc0000")):
    raw = {"images": {k: rng.integers(0, 256, (1, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
           "state": rng.uniform(-0.5, 0.5, (1, d)).astype(np.float32), "prompt": ["fold the shirt"]}
    obs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="value", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False)
    with torch.no_grad():
        p = vf.distribution(obs)[0].numpy()
    a.plot(vals, p, color=col, lw=1, label=f"step {t}: p(b | o_t) (untrained), V = {float(p @ vals):+.3f}; target bin {bins[t]} (dashed, R_t/T_max = {R[t]:+.3f})")
    a.axvline(vals[bins[t]], color=col, ls="--", lw=1)
a.axhline(1 / 201, color="#999", ls=":", lw=1)
a.text(-0.98, 1 / 201 * 1.02, "uniform 1/201", fontsize=7, color="#666")
a.set_xlabel("v(b), b = 0..200 (Sec. IV-A, 201 bins over [-1, 0])")
a.set_ylabel("p(V = b | o_t, ell)")
a.set_title("(a) the head's return distribution vs the Eq. 1 target bin\n(12-step success episode, T_max 40; tiny untrained head)", fontsize=9)
a.legend(fontsize=6.5, loc="upper left")
a.set_ylim(0, a.get_ylim()[1] * 1.25)

# (b) synthetic critic
t = np.arange(T1)
V_syn = R + 0.12 * np.sin(np.linspace(0, 2 * np.pi, T1)) - 0.05  # SYNTHETIC: target + a smooth error, not a trained model
r = episode_rewards(T1, True, C_FAIL_TINY) / Tmax
A_T = T.advantage_whole_episode(R, V_syn)
A_3 = T.advantage_nstep(r, V_syn, n=3)
A_50 = T.advantage_nstep(r, V_syn, n=T.LOOKAHEAD_N)
b.plot(t, R, "k-", lw=1, label="target R_t / T_max")
b.plot(t, V_syn, "o-", color="#999", lw=1, label="SYNTHETIC critic V(o_t) = target + smooth error")
b.plot(t, A_T, "s-", color="#3d85c6", label="A, N = T (pre-training): R_t/T_max - V(o_t)")
b.plot(t, A_50, "^--", color="#6aa84f", label=f"A, N = 50 (post-training) = N = T here (T < 50)")
b.plot(t, A_3, "v-", color="#e69138", label="A, N = 3 (for contrast): bootstraps V(o_t+3)")
eps = T.improvement_threshold(np.concatenate([A_T, rng.normal(0, 0.05, 500)]), 0.4)
b.axhline(eps, color="#cc0000", ls="--", lw=1, label=f"eps (40% positive of a 512 pool) = {eps:+.3f}")
corr = np.zeros(T1, bool)
corr[8:] = True
b.fill_between(t, -1.05, 0.35, where=corr, color="#f4cccc", alpha=0.5, label="correction steps: I_t forced True")
b.set_ylim(-1.05, 0.35)
b.set_xlabel("step t")
b.set_title("(b) advantage estimators (App. F) on a synthetic critic\n(nothing here is a trained value; it shows the arithmetic)", fontsize=9)
b.legend(fontsize=6.5, loc="lower right")
b.grid(alpha=0.3)

# (c) thresholds
pool = rng.normal(-0.02, 0.08, 10_000)
c.hist(pool, bins=80, color="#d9d9d9", edgecolor="#999")
for frac, col, name in ((0.30, "#3d85c6", "pre-training 30%"), (0.40, "#6aa84f", "fine-tuning 40%"), (0.10, "#cc0000", "T-shirt task 10%")):
    e = T.improvement_threshold(pool, frac, rng)
    c.axvline(e, color=col, lw=1.5, label=f"{name}: eps = {e:+.3f} -> {(pool > e).mean():.0%} positive")
c.set_xlabel("advantage A (synthetic pool of 10k datapoints, App. F)")
c.set_ylabel("count")
c.set_title("(c) eps_ell as a quantile of the task's advantage pool\n(App. F: 30% / 40% / 10%; 10k-sample estimate)", fontsize=9)
c.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
