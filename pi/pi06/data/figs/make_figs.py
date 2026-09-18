"""Generate figs/segments.png: (a) who may read which segment: the token-level visibility matrix of one tiny "joint"
sequence (rows: a query from each segment + the expert's 50 action tokens; columns: segments), i.e. the data-level
contract ../train and ../backbone turn into attention masks; (b) the value-target chain for a success and a failure
episode of the same length; (c) the 200-token budget per layout for the paper robot (14 dims).
Run: uv run python pi/pi06/data/figs/make_figs.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.tokenizer.tokenizer import make_smooth_chunks  # noqa: E402
from pi.pi06.data.data import (  # noqa: E402
    C_FAIL_TINY,
    MAX_TOKEN_LEN,
    SEG_ACTION,
    SEG_ADVANTAGE,
    SEG_MARKER,
    SEG_PREFIX,
    SEG_SUBTASK,
    SEG_TEXT,
    EpisodeLabels,
    bin_values,
    expert_visible,
    tiny_pi06_tokenizer,
)

OUT = pathlib.Path(__file__).with_name("segments.png")
rng = np.random.default_rng(0)
H, d = 10, 7
seq = tiny_pi06_tokenizer(H, d)
state = rng.uniform(-0.5, 0.5, d).astype(np.float32)
actions = (0.3 * make_smooth_chunks(1, H, d, rng)[0]).astype(np.float32)
toks, mask, seg, loss = seq.tokenize("make a double espresso", state, layout="joint", actions=actions, subtask="grab the portafilter", advantage=True, metadata="speed: fast")
n = int(mask.sum())
seg = seg[:n]
names = {SEG_PREFIX: "prefix (Task+meta+State)", SEG_SUBTASK: "Subtask: ...", SEG_ADVANTAGE: "Advantage: positive", SEG_ACTION: "Action: <FAST> |"}
order = [SEG_PREFIX, SEG_SUBTASK, SEG_ADVANTAGE, SEG_ACTION]
starts = {s: int(np.argmax(seg == s)) for s in order}
ends = {s: int(n - np.argmax(seg[::-1] == s)) for s in order}

fig, (a, b, c) = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw={"width_ratios": [1.25, 1.1, 1.2]})
# (a) visibility: a query token in segment i (its LAST token) vs every key column; plus the expert row
M = np.zeros((5, n))
for i, s in enumerate(order):
    q = ends[s] - 1
    M[i, : q + 1] = 1  # causal text: sees every earlier valid token (card Sec. 2)
M[4] = expert_visible(seg, mask[:n]).astype(float)
a.imshow(M, cmap="Greens", vmin=0, vmax=1.4, aspect="auto", interpolation="nearest")
for s in order[1:]:
    a.axvline(starts[s] - 0.5, color="#444", lw=0.8)
a.set_xticks([(starts[s] + ends[s]) / 2 - 0.5 for s in order], [names[s].replace(" (Task+meta+State)", "") for s in order], fontsize=7.5, rotation=12)
a.set_yticks(range(5), [f"query: last token of\n{names[s]}" for s in order] + ["50 expert action tokens\n(../train, ../backbone)"], fontsize=7.5)
a.set_title(f"(a) who reads what in one 'joint' sequence ({n} tokens; text causal, card Sec. 2;\nexpert never reads FAST: paper Sec. V-A, KI App. B). CE on Subtask + FAST only.", fontsize=9)
a.set_xlabel("key column (token index)")
for i, s in enumerate(order):
    a.text(n + 1, i, f"loss:\n{'CE' if loss[ends[s]-1] else 'none'}", fontsize=7, va="center", ha="left", color="#333")
a.set_xlim(-0.5, n + 22)

# (b) value targets
T, Tmax = 12, 40
vs, _ = EpisodeLabels("t", True, Tmax, T).value_targets(C_FAIL_TINY)
vf, _ = EpisodeLabels("t", False, Tmax, T).value_targets(C_FAIL_TINY)
vf_raw = np.cumsum(np.r_[-np.ones(T - 1), -C_FAIL_TINY][::-1])[::-1] / Tmax
b.plot(range(T), vs, "o-", color="#38761d", label="success: v = -(T - t) / T_max")
b.plot(range(T), vf_raw, "s--", color="#cc0000", alpha=0.5, label=f"failure, before clip: (-(T-t) - C_fail) / T_max")
b.plot(range(T), vf, "s-", color="#cc0000", label="failure: clipped to -1")
b.axhline(0, color="#333", lw=0.8)
b.axhline(-1, color="#333", lw=0.8)
b.set_ylim(-2.4, 0.15)
b.set_xlabel("step t (episode of 12 steps, T_max = 40)")
b.set_ylabel("normalised return = value target")
b.set_title(f"(b) Eq. 5 -> per-task normalised return (Sec. V-C), {len(bin_values())} bins over [-1, 0]\n(C_fail undisclosed; tiny uses {C_FAIL_TINY:.0f})", fontsize=9)
b.legend(fontsize=7.5, loc="lower right")
b.grid(alpha=0.3)

# (c) token budget, paper robot
state14 = np.zeros(14, np.float32)
seq14 = tiny_pi06_tokenizer(H, 14, seed=3, max_len=400)
act14 = (0.3 * make_smooth_chunks(1, H, 14, rng)[0]).astype(np.float32)
rows = []
kw_all = dict(subtask="fold the left flap inwards", advantage=True, metadata="speed: fast")
for label, layout, kw in [("value / hl_prompt", "value", {}), ("flow (+subtask+adv)", "flow", kw_all), ("joint (+subtask+adv)", "joint", dict(actions=act14, **kw_all)),
                          ("joint, adv dropped", "joint", dict(actions=act14, subtask=kw_all["subtask"], advantage=None, metadata="speed: fast")), ("text (caption)", "text", dict(target_text="a dog catches a frisbee"))]:
    _, m_, s_, _ = seq14.tokenize("assemble the box", state14, layout=layout, **kw)
    nn_ = int(m_.sum())
    parts = [int(((s_ == sid) & m_).sum()) for sid in (SEG_PREFIX, SEG_SUBTASK, SEG_ADVANTAGE, SEG_ACTION, SEG_TEXT, SEG_MARKER)]
    rows.append((label, parts, max(MAX_TOKEN_LEN - nn_, 0)))
labels = [r[0] for r in rows][::-1]
P = np.array([r[1] + [r[2]] for r in rows])[::-1]
pnames = ["prefix", "Subtask", "Advantage", "FAST", "text target", "'Action: ' marker", "pad"]
colors = ["#9fc5e8", "#f9cb9c", "#b4a7d6", "#93c47d", "#ffd966", "#e6b8af", "#eeeeee"]
left = np.zeros(len(rows))
for j in range(7):
    c.barh(labels, P[:, j], left=left, color=colors[j], edgecolor="#444", label=pnames[j])
    left += P[:, j]
c.axvline(MAX_TOKEN_LEN, color="#333", ls=":", lw=1)
c.set_xlim(0, max(MAX_TOKEN_LEN + 4, left.max() + 4))
c.set_xlabel("tokens (byte codec; SentencePiece is shorter)")
c.set_title(f"(c) the {MAX_TOKEN_LEN}-token budget per layout, paper robot (14-dim state, H = {H} here)\nmax_token_len undisclosed for pi0.6, pi0.5's 200 kept", fontsize=9)
c.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
c.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for r in rows:
    print(r)
