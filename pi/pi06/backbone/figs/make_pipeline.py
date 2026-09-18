"""Generate figs/pipeline.png: one low-level inference through the pi0.6 backbone, step by step (top row), and the
subtask-decoding path as the mirrored bottom row. Shapes and numbers from one real tiny run.
Run: uv run python pi/pi06/backbone/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
import pi.pi06.backbone.model as M  # noqa: E402
from pi.fast.data.data import EOS_ID  # noqa: E402
from pi.pi06.data.data import STATIC_IMAGE_KEYS, build_pi06_batch, tiny_pi06_tokenizer, unit_stats  # noqa: E402

OUT = pathlib.Path(__file__).with_name("pipeline.png")
torch.manual_seed(0)
B, H, d = 1, 10, 7
cfg, ecfg = M.tiny_experts()
model = M.tiny_pi06().eval()
seq = tiny_pi06_tokenizer(H, d)
rng = np.random.default_rng(0)
raw = {"images": {k: rng.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
       "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32), "prompt": ["make a double espresso"]}
obs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="flow", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False,
                          subtasks=["grab the portafilter"], advantages=[True])
with torch.no_grad():
    patches = model.vision.img(obs.images["base_0_rgb"])
    vis = model.vision(obs.images["base_0_rgb"])
    emb, valid, n_img = model.embed_prefix(obs)
    mask = model.prefix_mask(obs, n_img)
    pos = valid.long().cumsum(1) - 1
    ti = n_img + 20
    keys_global = int(mask[0, ti].sum())
    keys_local = int((mask[0, ti] & M.sliding_mask(pos, pos, cfg.sliding_window)[0, ti]).sum())
    h, cache, valid, n_img = model.forward_prefix(obs)
    noise = torch.randn(B, H, 32)
    vt = model.make_velocity_fn(cache, valid)(noise, torch.ones(B))
    x0 = model.sample_actions(obs, noise, M.NUM_DENOISING_STEPS)
    hobs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="hl_prompt", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False)
    toks, steps = model.sample_text(hobs, max_new_tokens=8, stop_ids=(EOS_ID, seq.newline_id))
n_text = int(obs.token_mask[0].sum())
pat = "".join("G" if M.GEMMA3_4B.is_global(i) else "L" for i in range(M.GEMMA3_4B.depth))
pc = M.backbone_param_count()
cand = M.expert_width_candidates()[0]

ENC = [
    ("0 Pi06Observation\n(../data, layout flow)", f"images x3 f32[1,448,448,3]\n  right_wrist masked\ntokens i64[1,200], {n_text} real:\n  Task/State + Subtask\n  + Advantage + 'Action: '", "#e8e8e8"),
    ("1 SigLIP So400m/14 @ 448\nno head (card Sec. 2)", f"patch 14 -> 32x32 = 1024\n-> f32[1,{patches.shape[1]},w_v]\n(pi0.5: 224 -> 256)\ntiny w_v {patches.shape[2]}; paper 1152", "#cfe2f3"),
    ("2 2x2 avg-pool -> 256,\nRMSNorm, linear -> LM width", f"VisionExit L202-L231:\n  {patches.shape[1]} -> {vis.shape[1]} tokens\nmm_soft_embedding_norm\nmm_input_projection\n  1152 -> 2560 (paper)\n-> f32[1,{vis.shape[1]},{vis.shape[2]}]", "#cfe2f3"),
    ("3 prefix = images | text\nembedder 262,144 x width", f"emb f32[1,{emb.shape[1]},{emb.shape[2]}]\n  = 3x256 image + 200 text\nvalid {int(valid[0].sum())} of {emb.shape[1]}\n  (masked camera: 256 cols\n   invalid, pi0 rule)\npositions = cumsum(valid)-1", "#e8e8e8"),
    ("4 mask: images bidir,\ntext causal (card Sec. 2)", f"bool[1,{mask.shape[1]},{mask.shape[2]}]\nimage row: all images\ntext row t: images +\n  text <= t\nexpert rows (later): images\n  + visible text + itself", "#cfe2f3"),
    ("5 34 layers, pattern\nLLLLLG, GQA 8/4 x 256", f"local: window 1024, RoPE 10k\nglobal: RoPE 1M, pos/8\nQK-norm; post-norms\ntiny: 6 layers, window {cfg.sliding_window}:\n  text row 20 keeps {keys_local}\n  of {keys_global} keys (local)\n  vs all {keys_global} (global)", "#cfe2f3"),
    ("6 prefix KV cache", f"{len(cache)} layers x (k, v, pos)\nk f32[1,{cache[0][0].shape[1]},{cache[0][0].shape[2]},{cache[0][0].shape[3]}]\n  = [B, S, kv_heads, head_dim]\npos i64[1,{cache[0][2].shape[1]}] (needed by\n  the sliding layers)\ncomputed once per chunk", "#e8e8e8"),
    ("7 action expert (34 layers,\n~860M) x 5 Euler steps", f"x_t f32[1,{H},32], tau ->\n  adaRMSNorm expert, reads\n  all {int(valid[0].sum())} valid prefix cols\nv_t f32{tuple(vt.shape)}\n5 steps 1.0 -> 0 (card; pi0.5: 10)\nx_0 f32{tuple(x0.shape)}", "#cfe2f3"),
    ("8 to the robot\n(../infer)", "quantile inverse, delta ->\n  absolute, 14 dims @ 50 Hz\n63 ms / chunk on one H100\n  with 3 cameras (card)", "#d9ead3"),
]
DEC = [
    ("8' subtask text\n-> next low-level prefix", "extract_subtask (../data):\n  cut at '\\n' / EOS, strip\n  'Subtask: '", "#d9ead3"),
    ("7' greedy decode loop", f"right-aligned prefill (pi0.5),\n  cache window grows 1/step\ntiny: {steps} step(s), ids\n  {toks[0, :steps].tolist()}\nstop ids: EOS, '\\n'\nmax {M.MAX_NEW_TOKENS} tokens (undisclosed)", "#cfe2f3"),
    ("6' tied logits head", "h[:, -1] @ E^T\n-> f32[1, 262144]\n(Gemma 3 vocab; PaliGemma\n was 257,152)", "#cfe2f3"),
    ("5' same 34-layer stack,\nexpert absent", "xs = [prefix, None]\nsliding + global layers\nas in step 5", "#e8e8e8"),
    ("4' layout hl_prompt\n(../data)", "prefix only: Task/State\n+ metadata; the model\nwrites 'Subtask: ...\\n'", "#e8e8e8"),
]

fig, ax = plt.subplots(figsize=(30, 9.2))
ax.set_xlim(0, 30)
ax.set_ylim(0, 9.2)
ax.axis("off")
BOX_W, GAP, X0 = 3.05, 0.24, 0.3
Y_ENC, H_ENC = 5.3, 2.75
Y_DEC, H_DEC = 2.0, 2.15


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
x_last = X0 + (len(ENC) - 1) * (BOX_W + GAP)
for i, (t_, b_, c_) in enumerate(DEC):
    x = x_last - i * (BOX_W + GAP)
    box(x, Y_DEC, BOX_W, H_DEC, t_, b_, c_)
    if i:
        arrow(x + BOX_W + GAP, Y_DEC + H_DEC / 2, x + BOX_W, Y_DEC + H_DEC / 2)
arrow(x_last + BOX_W / 2, Y_ENC, x_last + BOX_W / 2, Y_DEC + H_DEC)
ax.text(x_last + BOX_W / 2 + 0.08, (Y_ENC + Y_DEC + H_DEC) / 2, "high level\n(lower rate)", fontsize=8, va="center")
ax.text(X0, Y_DEC - 0.35, "blue = pi0.6 backbone increment (google-deepmind/gemma@0513283 _gemma.py L221-L246, _modules.py L36-L52 / L150-L270 / L400-L490, vision/_vision.py L202-L231; card Sec. 2), grey = pi0 / pi0.5 reused, green = hand-off.  "
        "No pi0.6 code exists upstream.", fontsize=8.5, va="top")
ax.text(X0, Y_DEC - 0.75, f"Must agree: (i) expert heads = backbone heads (8 q / 4 kv x 256) so one attention serves both;  (ii) the cache stores key positions: local layers re-apply the window at every step;  "
        f"(iii) global layers divide positions by 8 before RoPE (base 1M), local layers base 10k;  (iv) Gemma 3 4B closed form: non-embedding {pc['non_embedding']/1e6:,.0f}M (report 3,209M), embedding {pc['embedding']/1e6:,.0f}M;  "
        f"(v) 860M expert <=> width {cand[0]}, mlp {cand[1]} under the adaRMSNorm block ({cand[2]/1e6:.0f}M) - inferred, README Sec. 8.", fontsize=8.5, va="top")
ax.text(X0, Y_ENC + H_ENC + 0.55, f"pi0.6 backbone: SigLIP 448 -> 256 tokens, Gemma 3 4B ({pat}), causal text, 34-layer expert, 5 denoising steps (tiny: 6 layers, window {cfg.sliding_window}, width {cfg.width} / expert {ecfg.width})",
        fontsize=10.5, weight="bold", va="bottom")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT, "valid", int(valid[0].sum()), "local/global keys", keys_local, keys_global, "decode steps", steps)
