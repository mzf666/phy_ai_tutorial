"""Generate figs/pipeline.png: raw sample -> Pi05Observation step by step (top, with the three postfix layouts side by
side at the tokenization step), and the mirrored inverse (bottom). Values come from one real tiny run.
Run: uv run python pi/pi05/data/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.fast.data.data import discretize_state  # noqa: E402
from pi.fast.tokenizer.tokenizer import make_smooth_chunks  # noqa: E402
from pi.pi0.data.data import make_bool_mask, to_delta_actions  # noqa: E402
from pi.pi05.data.data import LL_IMAGE_KEYS, MAX_TOKEN_LEN, build_pi05_batch, hl_target_text, state_prefix_text, tiny_pi05_tokenizer, unit_stats, with_control_mode  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")

rng = np.random.default_rng(0)
B, H, d = 1, 10, 7
raw = {
    "images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8), "left_wrist_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
    "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
    "actions": (0.3 * make_smooth_chunks(B, H + 3, d, rng)).astype(np.float32),
    "prompt": [with_control_mode("pick up the pillow", "joint")],
}
seq = tiny_pi05_tokenizer(H, d)
delta_mask = make_bool_mask(6, -1)
stats = unit_stats(d)
delta = to_delta_actions(raw["state"], raw["actions"][:, :H], delta_mask)
prefix = state_prefix_text(raw["prompt"][0], raw["state"][0])
target = hl_target_text("pick up the pillow", [("pillow", (0.40, 0.11, 0.91, 0.19))])
outs = {}
for layout, kw in [("flow", {}), ("fast", {}), ("text", {"target_text": [target]})]:
    obs, actions = build_pi05_batch(raw, stats, seq, layout=layout, image_keys=LL_IMAGE_KEYS, action_horizon=H, delta_mask=delta_mask, train=False, **kw)
    m, ar = obs.tokenized_prompt_mask[0].numpy(), obs.token_ar_mask[0].numpy()
    outs[layout] = (obs, actions, int(((ar == 0) & m).sum()), int((ar == 1).sum()))
obs, actions, n_pre_flow, _ = outs["flow"]
_, _, n_pre, n_fast = outs["fast"]
_, _, _, n_text = outs["text"]
toks_fast = outs["fast"][0].tokenized_prompt[0].numpy()
rec = seq.extract_actions(toks_fast, H, d)
toks_text = outs["text"][0].tokenized_prompt[0].numpy()
txt = seq.extract_text(toks_text[n_pre:])
bins = discretize_state(raw["state"][0])


def v3(x):
    return " ".join(f"{float(v):+.2f}" for v in np.asarray(x).reshape(-1)[:3])


ENC = [
    ("0 raw sample", f"images: base_0, left_wrist\n  uint8[1,128,160,3]\nstate f32[1,{d}]: {v3(raw['state'][0])}\nactions f32[1,{H+3},{d}] absolute\nprompt: 'pick up the pillow\n  <control mode> joint\n  <control mode>'", "#e8e8e8"),
    ("1 camera slots\n(LL: 3 of the 4)", f"LL_IMAGE_KEYS: base_0,\n  left_wrist, right_wrist\nright_wrist missing:\n  black + mask False\n(droid_policy L48-L51,\n same rule as pi0)", "#cfe2f3"),
    ("2 delta + quantile\n(pi0/data, fast/tokenizer)", f"delta: joints -= state\n  {v3(delta[0,0])}\nquantile q01->-1, q99->+1\n(config.py L187)\nstate normalized:\n  {v3(raw['state'][0])}", "#e8e8e8"),
    ("3 state -> 256 bins -> text\n(fast/data) tokenizer.py L26", f"bins: {' '.join(map(str, bins))}\nprefix text:\n 'Task: pick up the pillow\n  <control mode> joint\n  <control mode>, State:\n  {' '.join(map(str, bins[:4]))} ...;\\n'", "#cfe2f3"),
    ("4 layout: postfix\nL28 / FAST L83-L87 / Fig. 4", f"flow : + 'Action: ' (prefix)\n  {n_pre_flow} tokens, no postfix\nfast : + 'Action: '+FAST+'|'\n  +EOS = {n_fast} postfix tokens\ntext : + target + EOS\n  = {n_text} postfix tokens", "#cfe2f3"),
    ("5 pad to max_len 200\npi0_config.py L39", f"tokens i64[1,200]\ntoken_mask bool[1,200]\nar_mask i64: 0 prefix, 1 postfix\nloss_mask: True on postfix\nprefix {n_pre} real tokens", "#cfe2f3"),
    ("6 pad state / actions to 32\ntransforms.py L328-L340", f"state f32[1,32]\n  (model does NOT read it)\nactions f32[1,{H},32]\n  dims {d}..31 zero", "#e8e8e8"),
    ("7 images -> [-1,1], augment\nmodel.py L176-L181 = App. E", f"f32[1,224,224,3] x 3\ntrain: crop .95, rot +-5,\n color b.3 c.4 s.5\n(wrist: color only)", "#e8e8e8"),
    ("8 Pi05Observation", f"images {{3 slots}}\nimage_masks [T, T, F]\nstate f32[1,32]\ntokenized_prompt i64[1,200]\n + mask / ar / loss masks\nactions f32[1,{H},32] (train)", "#d9ead3"),
]
DEC = [
    ("8' generated ids", "i64[N] from ../hier\n(HL: text; pre-train: FAST)", "#d9ead3"),
    ("7' extract_text\n(text / hl_prompt layouts)", f"cut at EOS, drop pad / BOS\n-> '{txt[:22]}...\n   ...{txt[-26:]}'", "#cfe2f3"),
    ("6' parse_hl_text\nFig. 4 format", "-> subtask 'pick up the pillow'\n   boxes [('pillow',\n   (0.40, 0.11, 0.91, 0.19))]\nno 'Subtask:' -> whole text", "#cfe2f3"),
    ("5' extract_actions\n(fast/data) L119-L134", f"find 'Action: ' .. '|'\n-> FAST decode f32[{H},{d}]\nmax|err| {np.abs(rec - actions[0,:,:d].numpy()).max():.3f} (rounding)\nno marker -> zeros", "#cfe2f3"),
    ("4' unnormalize + absolute\n(../infer)", "quantile inverse,\njoints += raw state,\nkeep native dims", "#e8e8e8"),
]

fig, ax = plt.subplots(figsize=(28, 8.6))
ax.set_xlim(0, 28)
ax.set_ylim(0, 8.6)
ax.axis("off")
BOX_W, BOX_H, DEC_H, GAP, X0 = 2.82, 2.5, 2.0, 0.24, 0.3
Y_ENC, Y_DEC = 4.9, 1.9


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=7.2, family="monospace")


def arrow(x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1, **kw))


for i, (t, b, c) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_ENC + BOX_H / 2, x, Y_ENC + BOX_H / 2)
x_last = X0 + (len(ENC) - 1) * (BOX_W + GAP)
for i, (t, b, c) in enumerate(DEC):
    x = x_last - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, DEC_H, t, b, c)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + DEC_H / 2, x + BOX_W, Y_DEC + DEC_H / 2)
arrow(x_last + BOX_W / 2, Y_ENC, x_last + BOX_W / 2, Y_DEC + DEC_H)
ax.text(x_last + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + DEC_H) / 2, "inverse", fontsize=8, va="center")

ax.text(X0, Y_DEC - 0.35, "blue = pi0.5 increment (tokenizer.py L22-L48, config.py L126-L138, paper Sec. IV-C / Fig. 4), grey = pi0/data and fast reused, green = contract.  "
        "Line refs: openpi@215abfb tokenizer.py unless noted.", fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) q01 / q99 of the checkpoint's norm_stats are used for both the state bins and the action targets;  (ii) the prefix of every layout is "
        "token-identical, only the postfix differs;  (iii) 'flow' puts 'Action: ' in the bidirectional prefix, 'fast' in the causal postfix (README Sec. 8);  "
        "(iv) extract_actions needs the same (H, D, FAST tokenizer) as training.", fontsize=8.5, va="top")
ax.text(X0, Y_ENC + BOX_H + 0.55, f"pi0.5 data: state as text in the prompt, one prefix and three postfixes (tiny: {d}-dim robot, H = {H}, byte codec, max_token_len {MAX_TOKEN_LEN}; "
        "paper robot: 18 / 19 dims, H = 50, 4 cameras for HL, 3 for LL)", fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "prefix", n_pre, "flow", n_pre_flow, "fast postfix", n_fast, "text postfix", n_text)
