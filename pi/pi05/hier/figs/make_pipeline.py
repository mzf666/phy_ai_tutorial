"""Generate figs/pipeline.png: the high level (top row: 4 cameras + task prompt -> prefill -> token by token -> subtask
text -> strip 'respond:'), the low level (bottom row: 3 cameras + subtask + state bins -> prefill -> 10 Euler steps ->
chunk) and Hi Robot's schedule between them. Tiny config, values from one real run.
Run: uv run python pi/pi05/hier/figs/make_pipeline.py
"""

import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON  # noqa: E402
from pi.pi0.flow_matching.model import sample_actions as euler_sample  # noqa: E402
from pi.pi05.data.data import HL_IMAGE_KEYS, LL_IMAGE_KEYS, build_pi05_batch, parse_hl_text, tiny_pi05_tokenizer  # noqa: E402
from pi.pi05.hier.model import HL_PERIOD_S, split_response, tiny_pi05  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
torch.manual_seed(0)
rng = np.random.default_rng(0)
d = 19
model = tiny_pi05()
seq = tiny_pi05_tokenizer(10, 7)
images = {k: rng.integers(0, 256, (1, 96, 128, 3), dtype=np.uint8) for k in HL_IMAGE_KEYS}
state = rng.uniform(-0.5, 0.5, (1, d)).astype(np.float32)
raw_hl = {"images": images, "state": state, "prompt": ["clean the kitchen"]}
with torch.no_grad():
    obs_hl, _ = build_pi05_batch(raw_hl, None, seq, layout="hl_prompt", image_keys=HL_IMAGE_KEYS, delta_mask=None, train=False)
    emb, mask, ar = model.embed_prefix(obs_hl)
    n_hl_valid, S_hl = int(mask.sum()), emb.shape[1]
    t0 = time.perf_counter()
    tokens, n_steps = model.sample_text(obs_hl, max_new_tokens=16)
    hl_ms = (time.perf_counter() - t0) * 1e3
    text = seq.extract_text(tokens[0].numpy())
    subtask = split_response(parse_hl_text(text)[0])[0] or "pick up the plate"  # random weights give no sentence; use the example subtask
    raw_ll = {"images": {k: images[k] for k in LL_IMAGE_KEYS}, "state": state, "prompt": [subtask]}
    obs_ll, _ = build_pi05_batch(raw_ll, None, seq, layout="flow", image_keys=LL_IMAGE_KEYS, delta_mask=None, train=False)
    emb2, mask2, _ = model.embed_prefix(obs_ll)
    n_ll_valid, S_ll = int(mask2.sum()), emb2.shape[1]
    kv, pm = model.prefix_cache(obs_ll)
    noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM)
    t0 = time.perf_counter()
    x0 = euler_sample(model.make_velocity_fn(kv, pm), noise, 10)
    ll_ms = (time.perf_counter() - t0) * 1e3
prompt_hl = seq.text.decode(obs_hl.tokenized_prompt[0].tolist())
prompt_ll = seq.text.decode(obs_ll.tokenized_prompt[0].tolist())


def v3(x):
    return " ".join(f"{float(v):+.2f}" for v in np.asarray(x).reshape(-1)[:3])


HL = [
    ("H0 high-level inputs\nSec. IV-E", f"4 cameras (front, rear,\n 2 wrists) uint8[1,96,128,3]\nstate f32[1,{d}] (normalized)\nprompt 'clean the kitchen'\n(+ user message, if any)", "#e8e8e8"),
    ("H1 ../data 'hl_prompt'\ntokenizer.py L22-L28", f"'{prompt_hl[:26]}\n {prompt_hl[26:52]}\n {prompt_hl[52:78]}...'\ni64[1,200], {int(obs_hl.tokenized_prompt_mask.sum())} real, ar all 0", "#e8e8e8"),
    ("H2 prefix + right align\npi0.py L106-L137; fast L51-L64", f"4 x 256 img + 200 txt = {S_hl}\nvalid {n_hl_valid}\nprefix-LM mask (all\n bidirectional at inference)", "#cfe2f3"),
    ("H3 prefill: expert 0 once\nL233-L237 / fast L265-L267", f"kv cache 4 layers x\n [1,{S_hl},1,16]\nlast col -> tied head\n -> logits [1,1,257152]", "#cfe2f3"),
    ("H4 decode x T\nfast L273-L307", f"1 token per step, expert 0\ngreedy (temperature 0)\nstop: EOS or cap\nthis run: {n_steps} steps, {hl_ms:.0f} ms\n(paper 2B: ~13 ms/token, 4090)", "#f9cb9c"),
    ("H5 text -> command\n../data parse_hl_text;\nHi Robot Sec. 4.2", f"extract_text -> parse\n 'Subtask: ...' (+ boxes)\nsplit_response: strip\n 'respond: ...' -> TTS\ncommand = '{subtask}'", "#cfe2f3"),
]
LL = [
    ("L0 low-level inputs\nSec. IV-E", f"3 cameras (front, 2 wrists)\nstate f32[1,{d}] (normalized)\nprompt = command from H5", "#e8e8e8"),
    ("L1 ../data 'flow'\ntokenizer.py L28", f"'Task: {subtask},\n State: ...;\\nAction: '\n{int(obs_ll.tokenized_prompt_mask.sum())} real tokens, ar all 0\nno FAST postfix", "#e8e8e8"),
    ("L2 prefix cache\nL233-L237", f"3 x 256 + 200 = {S_ll}\nvalid {n_ll_valid}\nexpert 0 once", "#cfe2f3"),
    ("L3 10 Euler steps\n../expert; L239-L278", f"x_1 = noise f32[1,50,32]\neach: embed_suffix(x_t, t)\n -> cond -> expert 1 (50 tok)\n -> v_t; x -= 0.1 v_t\n{ll_ms:.0f} ms (tiny, CPU)", "#cfe2f3"),
    ("L4 chunk", f"x_0 f32[1,50,32]\n{v3(x0[0,0])} ...\n-> ../infer: unnormalize,\n absolute, native dims", "#d9ead3"),
]

fig, ax = plt.subplots(figsize=(24, 9.6))
ax.set_xlim(0, 24)
ax.set_ylim(0, 9.6)
ax.axis("off")
BOX_W, BOX_H, GAP, X0 = 3.3, 2.3, 0.3, 0.3
Y_HL, Y_LL = 6.3, 1.9


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + 0.1, y + h - 0.85, body, ha="left", va="top", fontsize=7.2, family="monospace")


def arrow(x0, y0, x1, y1, color="#333", **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.1, **kw))


for i, (t, b, c) in enumerate(HL):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_HL, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_HL + BOX_H / 2, x, Y_HL + BOX_H / 2)
for i, (t, b, c) in enumerate(LL):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_LL, BOX_W, BOX_H, t, b, c)
    if i:
        arrow(x - GAP, Y_LL + BOX_H / 2, x, Y_LL + BOX_H / 2)
# command flows from H5 down to L0
x5 = X0 + 5 * (BOX_W + GAP) + BOX_W / 2
x0c = X0 + BOX_W / 2
ym = (Y_HL + Y_LL + BOX_H) / 2
ax.plot([x5, x5, x0c], [Y_HL, ym, ym], color="#b45f06", lw=1.4)
arrow(x0c, ym, x0c, Y_LL + BOX_H, color="#b45f06")
ax.text((x5 + x0c) / 2, ym + 0.12, "subtask text becomes the low-level prompt", ha="center", fontsize=9, color="#b45f06")
# schedule note
ax.text(X0 + 3.2 * (BOX_W + GAP), ym - 0.4,
        f"schedule (Hi Robot Sec. 4.1-4.2): high level at t = 0, then every {HL_PERIOD_S:.0f} s or immediately on a user message;\n"
        "low level every action chunk (50 steps at 50 Hz, executed length undisclosed); 'respond:' text goes to TTS, never to the low level;\n"
        "after an interjection the user can signal resume() -> previous command.", fontsize=8.5, va="center")
ax.text(X0, Y_HL + BOX_H + 0.5, "pi0.5 hierarchical inference: ONE model, two calls per second (tiny: width 64 / 32, 4 layers, byte codec; paper: 2B + 428M expert)",
        fontsize=10.5, weight="bold", va="bottom")
ax.text(X0, Y_LL - 0.35, "blue = pi0.5 increment (openpi@215abfb pi0.py / pi0_fast.py line refs), grey = ../data and pi0 reused, orange = stochastic / irreversible, green = output.",
        fontsize=8.5, va="top")
ax.text(X0, Y_LL - 0.75, "Must agree: (i) both levels share every weight, only the prompt, the cameras and which expert runs differ;  (ii) the high-level prompt and the low-level prompt "
        "use the same state bins from the same norm_stats;  (iii) the text decoded by H4 must be parsed with the format the 'text' layout was trained with (../data Sec. 1.3).",
        fontsize=8.5, va="top")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "hl steps", n_steps, "hl ms", round(hl_ms), "ll ms", round(ll_ms))
