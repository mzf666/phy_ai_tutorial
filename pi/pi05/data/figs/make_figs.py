"""Generate figs/layouts.png: (a) how the 200-token budget is used by the three layouts for the tiny 7-dim robot and
for a 19-dim state (the paper's mobile manipulator), against pi0's 48; (b) the HL / LL camera slot masks.
Run: uv run python pi/pi05/data/figs/make_figs.py
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
from pi.pi05.data.data import HL_IMAGE_KEYS, LL_IMAGE_KEYS, MAX_TOKEN_LEN, hl_target_text, tiny_pi05_tokenizer, with_control_mode  # noqa: E402

OUT = pathlib.Path(__file__).with_name("layouts.png")
rng = np.random.default_rng(0)
H = 10
rows = []
for d in (7, 19):
    seq = tiny_pi05_tokenizer(H, d, seed=d, max_len=400)  # count without the 200 cut; the cut is drawn as a line
    state = rng.uniform(-0.5, 0.5, d).astype(np.float32)
    actions = (0.3 * make_smooth_chunks(1, H, d, rng)[0]).astype(np.float32)
    prompt = with_control_mode("put the dishes in the sink", "joint")
    n_task = len(seq.text.encode("Task: " + prompt + ", State: ", add_bos=True))
    n_state = len(seq.text.encode(" ".join(map(str, np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1)) + ";\n"))
    for layout, kw in [("flow", {}), ("fast", {"actions": actions}), ("text", {"target_text": hl_target_text("pick up the plate", [("plate", (0.4, 0.1, 0.9, 0.2))])})]:
        toks, mask, ar, _ = seq.tokenize(prompt, state, layout=layout, **kw)
        n_post = int((ar == 1).sum())
        n_marker = len(seq._action_marker) if layout == "flow" else 0
        rows.append((f"{d}-dim, {layout}", n_task, n_state, n_marker, n_post, max(MAX_TOKEN_LEN - int(mask.sum()), 0)))

fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
labels = [r[0] for r in rows][::-1]
parts = np.array([r[1:] for r in rows])[::-1]
names = ["BOS + 'Task: <prompt>, State: '", "state bins + ';\\n'", "'Action: ' (flow prefix)", "postfix (FAST ids / text target)", "pad"]
colors = ["#9fc5e8", "#3d85c6", "#f9cb9c", "#93c47d", "#eeeeee"]
left = np.zeros(len(rows))
for j in range(5):
    a.barh(labels, parts[:, j], left=left, color=colors[j], edgecolor="#444", label=names[j])
    left += parts[:, j]
a.axvline(48, color="#cc0000", ls="--", lw=1)
a.text(48.5, len(rows) - 0.5, "pi0 max_token_len 48", color="#cc0000", fontsize=8, va="center")
a.axvline(MAX_TOKEN_LEN, color="#333", ls=":", lw=1)
a.set_xlim(0, max(MAX_TOKEN_LEN + 4, left.max() + 4))
a.text(MAX_TOKEN_LEN + 1, len(rows) - 1, "beyond 200: truncated\nupstream (L40-L48)", fontsize=7.5, va="center")
a.set_xlabel("tokens (byte codec; SentencePiece is shorter)")
a.set_title(f"(a) the {MAX_TOKEN_LEN}-token budget per layout (H = {H}; openpi@215abfb pi0_config.py L39, tokenizer.py L22-L48)", fontsize=9.5)
a.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
a.grid(axis="x", alpha=0.3)

slots = list(HL_IMAGE_KEYS)
m = np.array([[1, 1, 1, 1], [1 if s in LL_IMAGE_KEYS else 0 for s in slots], [1, 0, 1, 1]])
b.imshow(m, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
b.set_xticks(range(4), [s.replace("_rgb", "") for s in slots], rotation=20, fontsize=8)
b.set_yticks(range(3), ["HL: all 4 cameras", "LL: wrist + forward", "example: rear camera\nunplugged"], fontsize=8)
for i in range(3):
    for j in range(4):
        b.text(j, i, "True" if m[i, j] else "False\n(black)", ha="center", va="center", fontsize=8)
b.set_title("(b) image_masks per slot (paper Sec. IV-E;\ndroid_policy.py L48-L51 rule: missing -> False)", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for r in rows:
    print(r)
