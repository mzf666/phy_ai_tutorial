"""Generate figs/latency.png: (a) per-call timing vs number of decode steps from real tiny runs (sample_actions is
everything and grows linearly); (b) execution rhythm: predict 15, execute 8 or 15 open-loop, where the infer calls fall.
Run: uv run python pi/fast/infer/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.infer.model import DROID_ACTION_HORIZON_PAPER, DROID_EXECUTE_STEPS, droid_obs_to_raw, tiny_fast_policy  # noqa: E402

OUT = pathlib.Path(__file__).with_name("latency.png")

rng = np.random.default_rng(0)
obs = {"observation/exterior_image_1_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
       "observation/wrist_image_left": rng.integers(0, 256, (180, 320, 3), dtype=np.uint8),
       "observation/joint_position": rng.uniform(-0.5, 0.5, 7).astype(np.float32),
       "observation/gripper_position": np.float32(0.3)}
raw = droid_obs_to_raw(obs, "put the marker in the cup")
caps = [8, 16, 32, 64]
rows = []
for cap in caps:
    policy, _ = tiny_fast_policy(max_decoding_steps=cap)
    policy.infer(raw)  # warm-up
    t = policy.infer(raw)["timing"]
    rows.append((t["data preprocessing"], t["sample_actions (prefill + decode steps)"], t["extract + inverse transforms"]))
rows = np.array(rows)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
bottom = np.zeros(len(caps))
for j, (name, c) in enumerate((("data preprocessing", "#93c47c"), ("sample_actions (prefill + N decode steps)", "#e06666"), ("extract + inverse transforms", "#6fa8dc"))):
    axa.bar(range(len(caps)), rows[:, j], bottom=bottom, color=c, label=name)
    bottom += rows[:, j]
for i, cap in enumerate(caps):
    axa.text(i, bottom[i] + 2, f"{bottom[i]:.0f} ms", ha="center", fontsize=8)
axa.set_xticks(range(len(caps)))
axa.set_xticklabels([f"{c} steps" for c in caps])
axa.set_ylabel("ms per infer call (tiny config, CPU, float32)")
axa.set_title("(a) an untrained tiny model runs to the cap: cost is the decode loop, linear in steps", fontsize=9.5)
axa.legend(fontsize=7.5, frameon=False, loc="upper left")
axa.text(0.02, 0.62, "paper Sec. VI-E, RTX 4090, bf16:\npi0-FAST ~750 ms / chunk (30-60 tokens)\npi0 ~100 ms / chunk (10 Euler steps)", transform=axa.transAxes, fontsize=8, va="top")

H = DROID_ACTION_HORIZON_PAPER
T = 45
for r, k in enumerate(DROID_EXECUTE_STEPS):
    y = 1 - r
    n_calls = -(-T // k)
    for c in range(n_calls):
        start = c * k
        # predicted chunk (light) and executed part (dark)
        axb.add_patch(plt.Rectangle((start, y + 0.05), min(H, T - start), 0.7, color="#f4cccc", ec="none"))
        axb.add_patch(plt.Rectangle((start, y + 0.05), min(k, T - start), 0.7, color="#e06666", ec="none"))
        axb.plot([start, start], [y - 0.02, y + 0.82], color="#333", lw=1.2)
        axb.text(start + 0.2, y + 0.86, "infer", fontsize=7, color="#333")
    axb.text(-0.8, y + 0.4, f"predict {H}, execute {k}\n{n_calls} infer calls / {T} steps", ha="right", va="center", fontsize=8.5)
axb.set_xlim(-14, T)
axb.set_xticks(range(0, T + 1, 5))
axb.set_ylim(-0.3, 2.1)
axb.set_yticks([])
axb.set_xlabel("control step (DROID control frequency not disclosed -> chunk duration in seconds unknown, README Sec. 8)")
axb.set_title("(b) execution rhythm, paper App. D: light = predicted chunk, dark = executed open-loop, bar = infer call", fontsize=9.5)
fig.suptitle("openpi@215abfb policy.py Policy.infer L67-L106 (times sample_actions only); paper Sec. VI-E, Appendix D", fontsize=9, y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT, rows.round(1).tolist())
