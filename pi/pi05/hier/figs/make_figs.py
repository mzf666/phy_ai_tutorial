"""Generate figs/hier.png: (a) sequential transformer forwards per second of control for pi0, pi0-FAST and pi0.5
(high level once per second + low level once per chunk), with the model size each forward runs on; (b) a 3-second
episode timeline of the Hi Robot schedule, taken from a real tiny run of HierarchicalPolicy.step.
Run: uv run python pi/pi05/hier/figs/make_figs.py
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
from pi.pi05.data.data import HL_IMAGE_KEYS, LL_IMAGE_KEYS, tiny_pi05_tokenizer  # noqa: E402
from pi.pi05.hier.model import HierarchicalPolicy, split_response, tiny_pi05  # noqa: E402

OUT = pathlib.Path(__file__).with_name("hier.png")

# ---- (a) forward counts per second, paper sizes. Chunk = 50 steps at 50 Hz = 1 s (pi0.5 Sec. IV-E); executed length
# undisclosed, so assume one low-level call per second (a lower bound). pi0: 1 prefill + 10 expert steps per chunk
# (pi0 paper). pi0-FAST: 1 prefill + 30-60 tokens per chunk (FAST Sec. VI-E). pi0.5 high level: 1 prefill + T tokens
# per second, T = subtask length (say 8), Hi Robot App. B.3 for the per-token cost.
BACKBONE, EXPERT = 2.92, 0.43  # billions: PaliGemma (SigLIP + Gemma 2B), pi0.5 expert with adaRMSNorm (../expert)
bars = {
    "pi0": [("prefill (2.9B)", 1, BACKBONE), ("10 expert steps (0.31B)", 10, 0.31)],
    "pi0-FAST": [("prefill (2.9B)", 1, BACKBONE), ("~45 token steps (2.9B)", 45, BACKBONE)],
    "pi0.5 (HL + LL)": [("HL prefill (2.9B)", 1, BACKBONE), ("HL ~8 token steps (2.9B)", 8, BACKBONE), ("LL prefill (2.9B)", 1, BACKBONE), ("10 expert steps (0.43B)", 10, EXPERT)],
}
fig, (a, b) = plt.subplots(2, 1, figsize=(13, 8.2), gridspec_kw={"height_ratios": [1, 1.1]})
colors = plt.cm.tab20(np.linspace(0, 1, 12))
ci = 0
for y, (name, parts) in enumerate(bars.items()):
    left = 0
    for label, n, size in parts:
        a.barh(name, n * size, left=left, color=colors[ci], edgecolor="#444", label=f"{name}: {label}, {n} x {size:.2f}B = {n * size:.1f}B")
        ci += 1
        left += n * size
    a.text(left + 1, y, f"{left:.0f}B params-forwards / s", va="center", fontsize=8.5)
a.legend(fontsize=7, loc="upper right")
a.set_xlim(0, 175)
a.set_xlabel("sequential forwards per second of control, weighted by the parameters each one runs through (billions)")
a.set_title("(a) what one second of control costs in sequential transformer forwards (paper sizes; pi0.5 HL once per second, Hi Robot App. B.3: 4090 prefill 47 ms + 13.2 ms / token)", fontsize=9)
a.grid(axis="x", alpha=0.3)

# ---- (b) timeline from a real tiny run
torch.manual_seed(0)
rng = np.random.default_rng(0)
model = tiny_pi05()
seq = tiny_pi05_tokenizer(10, 7)
images = {k: rng.integers(0, 256, (1, 64, 80, 3), dtype=np.uint8) for k in HL_IMAGE_KEYS}
state = rng.uniform(-0.5, 0.5, (1, 19)).astype(np.float32)
raw_hl = {"images": images, "state": state, "prompt": ["clean the kitchen"]}
raw_ll = {"images": {k: images[k] for k in LL_IMAGE_KEYS}, "state": state}
pol = HierarchicalPolicy(model, seq, max_new_tokens=8)
script = ["pick up the plate", "put the plate in the sink", "put it back respond: sorry", "pick up the cup"]
pol.high_level = lambda raw, msg=None, generator=None: (*split_response(script.pop(0)), 8)  # scripted subtasks: random weights give no sentence
events = []
t, chunk_s, k = 0.0, 0.5, 0  # a chunk every 0.5 s here (executed length undisclosed; 25 of 50 steps at 50 Hz as in pi0)
while t < 3.0:
    msg = "that's not the plate" if abs(t - 1.5) < 1e-9 else None
    out = pol.step(raw_hl, raw_ll, t, msg)
    events.append((t, out["hl_ran"], out["subtask"], out["utterance"], msg))
    if abs(t - 2.0) < 1e-9:
        pol.resume()
        events.append((t + 0.01, False, pol.subtask + "  (resume)", None, None))
    t = round(t + chunk_s, 3)
b.set_xlim(-0.1, 3.3)
b.set_ylim(-0.6, 2.4)
for t, hl, sub, utt, msg in events:
    if "resume" in sub:
        b.text(t, 0.55, "resume() -> " + sub.split("  ")[0], fontsize=7.5, color="#38761d", ha="left")
        continue
    b.plot([t, t], [0, 0.3], color="#3d85c6", lw=2)
    if hl:
        b.plot([t, t], [1, 1.35], color="#b45f06", lw=3)
        b.text(t + 0.02, 1.42 + 0.32 * (int(round(t / 0.5)) % 2), f"HL -> '{sub}'" + (f"\n  utterance '{utt}'" if utt else ""), fontsize=7.5, ha="left", va="bottom")
    if msg:
        b.annotate(f"user: '{msg}'", (t, 1.0), (t - 0.05, 2.15), fontsize=8, color="#990000", arrowprops={"arrowstyle": "->", "color": "#990000"}, ha="right")
b.hlines([0, 1], -0.1, 3.3, color="#bbb", lw=0.8)
b.set_yticks([0.15, 1.15], ["low level:\n1 chunk / 0.5 s", "high level:\nevery 1 s or on a message"], fontsize=8)
b.set_xlabel("time (s)")
b.set_title("(b) Hi Robot schedule in a 3-second episode (tiny run of HierarchicalPolicy.step; subtasks scripted since random weights decode nothing)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for e in events:
    print(e)
