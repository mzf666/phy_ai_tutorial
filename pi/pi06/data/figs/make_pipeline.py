"""Generate figs/pipeline.png: a raw sample -> Pi06Observation step by step (top row; the tokenization box shows the
five segments of the "joint" layout), the RL-label chain (middle row, Eq. 5 -> returns -> normalised -> 201 bins), and
the mirrored inverse (bottom). Every number comes from one real tiny run.
Run: uv run python pi/pi06/data/figs/make_pipeline.py
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
from pi.pi06.data.data import (  # noqa: E402
    ADVANTAGE_DROPOUT,
    C_FAIL_TINY,
    MAX_TOKEN_LEN,
    NUM_BINS,
    SEG_ACTION,
    SEG_ADVANTAGE,
    SEG_PREFIX,
    SEG_SUBTASK,
    STATIC_IMAGE_KEYS,
    EpisodeLabels,
    build_pi06_batch,
    episode_rewards,
    returns,
    tiny_pi06_tokenizer,
    unit_stats,
)

OUT = pathlib.Path(__file__).with_name("pipeline.png")
rng = np.random.default_rng(0)
B, H, d = 1, 10, 7
raw = {
    "images": {"base_0_rgb": rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8), "left_wrist_0_rgb": rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8)},
    "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
    "actions": (0.3 * make_smooth_chunks(B, H + 3, d, rng)).astype(np.float32),
    "prompt": ["make a double espresso"],
}
seq = tiny_pi06_tokenizer(H, d)
delta_mask = make_bool_mask(6, -1)
delta = to_delta_actions(raw["state"], raw["actions"][:, :H], delta_mask)
bins = discretize_state(raw["state"][0])
obs, actions = build_pi06_batch(raw, unit_stats(d), seq, layout="joint", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=delta_mask, train=False,
                                subtasks=["grab the portafilter"], advantages=[True], metadata=["speed: fast"])
t, m, s, l = obs.tokens[0].numpy(), obs.token_mask[0].numpy(), obs.segment[0].numpy(), obs.loss_mask[0].numpy()
n = int(m.sum())
cnt = {k: int(((s == v) & m).sum()) for k, v in [("prefix", SEG_PREFIX), ("subtask", SEG_SUBTASK), ("adv", SEG_ADVANTAGE), ("fast", SEG_ACTION)]}
n_vis = int(obs.expert_visible[0].sum())
rec = seq.extract_actions(t, H, d)
gen_sub = seq.extract_subtask(t[(s == SEG_SUBTASK) & m])
flow_obs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="flow", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=delta_mask, train=False,
                               subtasks=["grab the portafilter"], advantages=[True], metadata=["speed: fast"])
n_flow = int(flow_obs.token_mask[0].sum())
lab = EpisodeLabels("make a double espresso", success=True, max_episode_len=40, num_steps=12, is_correction=np.array([False] * 8 + [True] * 4))
r = episode_rewards(12, True, C_FAIL_TINY)
R = returns(r)
v, vb = lab.value_targets(C_FAIL_TINY)
v_f, vb_f = EpisodeLabels("x", False, 40, 12).value_targets(C_FAIL_TINY)


def v3(x):
    return " ".join(f"{float(u):+.2f}" for u in np.asarray(x).reshape(-1)[:3])


ENC = [
    ("0 raw sample", f"images: base_0, left_wrist\n  uint8[1,96,128,3]\nstate f32[1,{d}]: {v3(raw['state'][0])}\nactions f32[1,{H+3},{d}] absolute\nprompt 'make a double\n  espresso' + metadata\n  'speed: fast'\nsubtask 'grab the portafilter'\nindicator I_t = True", "#e8e8e8"),
    ("1 camera slots (3 static /\n4 mobile), 448x448", "STATIC_IMAGE_KEYS: base_0,\n  left_wrist, right_wrist\nright_wrist missing ->\n  black + mask False\nresize_with_pad -> 448x448\n(card Sec. 2; pi0.5: 224)", "#cfe2f3"),
    ("2 delta + quantile\n(pi0/data, fast/tokenizer)", f"delta: joints -= state\n  {v3(delta[0,0])}\nquantile q01->-1, q99->+1\nstate bins (256):\n  {' '.join(map(str, bins[:5]))} ...", "#e8e8e8"),
    ("3 prefix text\n(pi05/data + metadata)", "'Task: make a double\n  espresso speed: fast,\n  State: 89 90 74 ...;\\n'\nell = ell_t + s (Sec. V-A)\nsame text feeds the\nvalue function (layout value)", "#cfe2f3"),
    ("4 five segments, all causal\nSec. V-A/V-B, KI App. B", f"[BOS] prefix        {cnt['prefix']:3d} tok  in\nSubtask: ...\\n      {cnt['subtask']:3d} tok  CE\nAdvantage: positive\\n {cnt['adv']:3d} tok  in\nAction: <FAST> | EOS {cnt['fast']:3d} tok  CE\n= {n} real tokens\nexpert reads {n_vis} (never FAST)", "#cfe2f3"),
    ("5 advantage dropout 30%\nApp. F", f"drop_indicator(I_t):\n  p = {ADVANTAGE_DROPOUT}: segment omitted\n  -> pi(a | o, ell)  (uncond)\n  else kept -> pi(a | I, o, ell)\nreplaces alpha of Eq. 3;\nenables CFG (../infer)", "#cfe2f3"),
    ("6 pad to max_len 200,\ndims to 32", f"tokens i64[1,{MAX_TOKEN_LEN}]\ntoken_mask, loss_mask bool\nsegment i64 (SEG_* ids)\nstate f32[1,32]\nactions f32[1,{H},32]\n  dims {d}..31 zero", "#e8e8e8"),
    ("7 Pi06Observation", f"images {{3 slots}} f32[1,448,448,3]\nimage_masks [T, T, F]\ntokens / token_mask / segment\n  / loss_mask [1,200]\nhas_actions [True]\nadvantage i8 [1]\nexpert_visible [1,200]", "#d9ead3"),
]
LAB = [
    ("L0 episode labels\nSec. IV step 1", "outcome: success (human)\nT_max(task) = 40 steps\n12 steps, steps 8..11 are\n  human corrections\n(is_correction bool[12])", "#e8e8e8"),
    ("L1 reward Eq. 5", f"r_t = -1 per step,\n 0 at a successful end,\n -C_fail at a failed end\nr = {' '.join(f'{x:.0f}' for x in r[:4])} ... {r[-1]:.0f}\nC_fail undisclosed (tiny {C_FAIL_TINY:.0f})", "#fce5cd"),
    ("L2 return R_t = sum r", f"R = {' '.join(f'{x:.0f}' for x in R[:5])} ... 0\n= -(T - t): steps left\nfailure: R_0 = -{11 + C_FAIL_TINY:.0f}", "#fce5cd"),
    ("L3 normalise by T_max\nSec. V-C -> (-1, 0)", f"v = R / 40, clip [-1, 0]\n= {' '.join(f'{x:.3f}' for x in v[:3])} ... 0\nfailure: all -1 (clipped)", "#fce5cd"),
    ("L4 201 bins, Sec. IV-A", f"bin = round((v + 1) * 200)\n= {' '.join(map(str, vb[:4]))} ... {vb[-1]}\nfailure: {vb_f[0]} everywhere\nvalue_bin i64[1] -> ../value\n  Eq. 1 CE target", "#fce5cd"),
]
DEC = [
    ("7' generated ids", "i64[N] from ../backbone\n(subtask decode / FAST)", "#d9ead3"),
    ("6' extract_subtask", f"cut at '\\n' or EOS,\nstrip 'Subtask: '\n-> '{gen_sub}'", "#cfe2f3"),
    ("5' extract_actions\n(fast/data)", f"find 'Action: ' .. '|'\n-> FAST decode f32[{H},{d}]\nmax|err| {np.abs(rec - actions[0,:,:d].numpy()).max():.3f}\nno marker -> zeros", "#cfe2f3"),
    ("4' bin -> value", f"bin_to_value: v(b) = -1 + b/200\n|v(bin(v)) - v| <= 0.0025\nexpectation over bins\n  gives V (../value)", "#fce5cd"),
    ("3' unnormalise + absolute\n(../infer)", "quantile inverse,\njoints += raw state,\nkeep native dims", "#e8e8e8"),
]

fig, ax = plt.subplots(figsize=(28, 12.2))
ax.set_xlim(0, 28)
ax.set_ylim(0, 12.2)
ax.axis("off")
BOX_W, GAP, X0 = 3.2, 0.24, 0.3
Y_ENC, H_ENC = 8.9, 2.75
Y_LAB, H_LAB = 5.55, 2.35
Y_DEC, H_DEC = 2.2, 2.2


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=8.6, weight="bold")
    ax.text(x + 0.1, y + h - 0.78, body, ha="left", va="top", fontsize=7.1, family="monospace")


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.1))


for i, (t_, b_, c_) in enumerate(ENC):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_ENC, BOX_W, H_ENC, t_, b_, c_)
    if i:
        arrow(x - GAP, Y_ENC + H_ENC / 2, x, Y_ENC + H_ENC / 2)
for i, (t_, b_, c_) in enumerate(LAB):
    x = X0 + i * (BOX_W + GAP)
    box(x, Y_LAB, BOX_W, H_LAB, t_, b_, c_)
    if i:
        arrow(x - GAP, Y_LAB + H_LAB / 2, x, Y_LAB + H_LAB / 2)
x_last = X0 + (len(ENC) - 1) * (BOX_W + GAP)
for i, (t_, b_, c_) in enumerate(DEC):
    x = x_last - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, H_DEC, t_, b_, c_)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + H_DEC / 2, x + BOX_W, Y_DEC + H_DEC / 2)
# labels row feeds the contract (value_bin) and the inverse row
x_lab_end = X0 + (len(LAB) - 1) * (BOX_W + GAP) + BOX_W
arrow(x_lab_end, Y_LAB + H_LAB / 2, x_last, Y_ENC + 0.3)
ax.text(x_lab_end + 0.1, Y_LAB + H_LAB / 2 + 0.55, "value_bin i64[B]\n(layout 'value')", fontsize=7.5, va="bottom")
arrow(x_last + BOX_W / 2, Y_ENC, x_last + BOX_W / 2, Y_DEC + H_DEC)
ax.text(x_last + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + H_DEC) / 2 + 0.4, "inverse", fontsize=8, va="center")

ax.text(X0, Y_DEC - 0.35, "blue = pi0.6* sequence increment (paper Sec. V-A / V-B, App. F; card Sec. 2), orange = RL labels (Eq. 5, Sec. IV-A, V-C), grey = pi0.5 / pi0 / FAST reused, green = contract.  "
        "No upstream code exists for pi0.6; pi0.5 prefix / FAST postfix follow openpi@215abfb tokenizer.py.", fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, "Must agree: (i) the value function reads the SAME prefix (Task + metadata + State) as the policy, without subtask / advantage;  (ii) the expert's visible columns = PREFIX + SUBTASK + ADVANTAGE (+ MARKER), "
        "never the FAST tokens;  (iii) advantage dropout happens in data, the loss has no alpha;  (iv) T_max, C_fail and the bin grid used for the value targets are the ones ../value decodes with;  "
        f"(v) a flow sequence = the joint one minus FAST ({n_flow} vs {n} tokens here).", fontsize=8.5, va="top")
ax.text(X0, Y_ENC + H_ENC + 0.55, f"pi0.6* data: five causal segments with the advantage token, three KI sample kinds, and Eq. 5 -> 201-bin value targets (tiny: {d}-dim arm, H = {H}, byte codec, max_token_len {MAX_TOKEN_LEN}, {NUM_BINS} bins; "
        "paper robot: 14 dims, H = 50, 3 cameras at 448x448)", fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "tokens", n, cnt, "expert reads", n_vis, "flow", n_flow)
