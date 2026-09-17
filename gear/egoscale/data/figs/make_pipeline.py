"""Generate figs/pipeline.png: one raw demonstration -> one batch-ready sample (top row, with the
human and robot branches split at the state box), and the mirrored inverse (bottom row). Values come
from one real tiny run.
Run: uv run python gear/egoscale/data/figs/make_pipeline.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.data.data import (  # noqa: E402
    CAMERA_SLOTS,
    Normalizer,
    augment_params,
    build_sample,
    build_video,
    collate,
    prepare_action,
    stats_from,
    tiny,
    unpad_action,
)

FWD = "#2c5f8a"
INV = "#8a5a2c"
HUM = "#7a4fa3"
ROB = "#b5442f"


def v(x, n=3):
    return np.array2string(np.asarray(x), precision=n, suppress_small=True)


def run_tiny():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = tiny()
    h, w = cfg.image_hw
    out = {}
    for name in ("human_wild", "r1pro_sharpa", "g1_trifinger"):
        emb = cfg.embodiments[name]
        frames = {s: rng.integers(0, 256, (cfg.state_horizon, h, w, 3), dtype=np.uint8)
                  for s in emb.cameras}
        pool_a = torch.randn(64, cfg.action_horizon, emb.action_dim)
        norm = Normalizer(cfg.norm_mode, stats_from(pool_a))
        native = torch.randn(cfg.action_horizon, emb.action_dim)
        state = None if not emb.has_proprio else torch.randn(cfg.state_horizon, emb.state_dim)
        if state is not None:
            pool_s = torch.randn(64, cfg.state_horizon, emb.state_dim)
            state = Normalizer(cfg.norm_mode, stats_from(pool_s)).forward(state)
        smp = build_sample(emb, cfg, frames, "roll the t-shirt and put it into the basket",
                           state, norm.forward(native))
        back = norm.inverse(unpad_action(smp["action"], smp["action_mask"]))
        out[name] = dict(emb=emb, frames=frames, sample=smp, norm=norm,
                         native=native, back=back)
    out["cfg"] = cfg
    out["batch"] = collate([out[n]["sample"] for n in
                            ("human_wild", "r1pro_sharpa", "g1_trifinger")])
    return out


def box(ax, x, y, w, h, title, shape, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", linewidth=1.6,
                                edgecolor=color, facecolor=color + "10", mutation_aspect=0.45))
    ax.text(x + w / 2, y + h - 0.030, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.076, shape, ha="center", va="top",
            fontsize=8.8, family="monospace", color="#333333")
    ax.text(x + 0.010, y + h - 0.122, "\n".join(body), ha="left", va="top",
            fontsize=8.2, family="monospace", color="#444444", linespacing=1.6)


def arrow(ax, x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color, shrinkA=0, shrinkB=0))


def main():
    d = run_tiny()
    cfg = d["cfg"]
    hw, rw, gw = d["human_wild"], d["r1pro_sharpa"], d["g1_trifinger"]
    hs, rs, gs = hw["sample"], rw["sample"], gw["sample"]

    fig, ax = plt.subplots(figsize=(30.0, 10.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n, gap = 7, 0.011
    w = (1 - 0.02 - (n - 1) * gap) / n
    xs = [0.01 + i * (w + gap) for i in range(n)]
    yf, yi, hh = 0.555, 0.145, 0.275

    fwd = [
        ("1. raw demonstration", "per-slot frames + state + action", [
            "one episode, 30 FPS (paper Sec. 2.2)",
            f"human_wild has {len(hw['emb'].cameras)} camera(s)",
            f"r1pro/g1 have {len(rw['emb'].cameras)} cameras",
            "language: the task instruction",
            "paper Sec. 2.2 / Sec. 2.5",
        ], FWD),
        ("2. camera slots", f"video (T,{len(CAMERA_SLOTS)},H,W,3) u8", [
            "fixed order: head | left | right",
            "missing slot -> BLACK + view_mask",
            f"human_wild view_mask {v(hs['view_mask'])}",
            f"r1pro      view_mask {v(rs['view_mask'])}",
            "upstream concat.py L91-L112",
        ], FWD),
        ("3. image augment", f"-> {cfg.image_hw} (tiny)", [
            f"crop scale {augment_params(cfg)['crop_scale']}",
            f"resize {cfg.image_hw} (upstream 224x224)",
            f"color jitter {cfg.color_jitter}",
            "ALL upstream GR00T values;",
            "EgoScale's own: UNDISCLOSED (Sec. 8)",
        ], FWD),
        ("4a. state: HUMAN", f"state ({cfg.state_horizon},{cfg.max_state_dim})", [
            "no proprioception exists in ego video",
            "-> all zeros, state_mask all False",
            f"mask True count = {int(hs['state_mask'].sum())}",
            "model side swaps in a LEARNABLE",
            "placeholder token (paper Sec. 2.3)",
        ], HUM),
        ("4b. state: ROBOT", f"state ({cfg.state_horizon},{cfg.max_state_dim})", [
            f"r1pro native {rw['emb'].state_dim} (2x7 arm + 2x22 hand)",
            f"g1    native {gw['emb'].state_dim} (2x7 arm + 2x7 hand)",
            f"r1pro mask True = {int(rs['state_mask'].sum())}",
            f"g1    mask True = {int(gs['state_mask'].sum())}",
            "over-wide state is TRUNCATED (L256)",
        ], ROB),
        ("5. action padding", f"action ({cfg.action_horizon},{cfg.max_action_dim})", [
            f"human/r1pro native {rw['emb'].action_dim} -> pad to {cfg.max_action_dim}",
            f"g1 native {gw['emb'].action_dim} -> "
            f"{1 - gw['emb'].action_dim / cfg.max_action_dim:.0%} padding",
            "over-wide action ASSERTS (L288-L290):",
            "truncating an action drops a DoF",
            "action_mask carries the loss denominator",
        ], FWD),
        ("6. collate a batch", "stack on a new dim", [
            f"video {tuple(d['batch']['video'].shape)}",
            f"action {tuple(d['batch']['action'].shape)}",
            f"embodiment_id {v(d['batch']['embodiment_id'].numpy())}",
            f"has_proprio {v(d['batch']['has_proprio'].numpy())}",
            "one batch, three embodiments",
        ], FWD),
    ]
    for (title, shape, body, color), x in zip(fwd, xs):
        box(ax, x, yf, w, hh, title, shape, body, color)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, yf + hh / 2, b, yf + hh / 2, FWD)

    inv = [
        ("1'. native command", "-> this embodiment", [
            "r1pro: 2x(3 trans + 6 rot + 22 joints)",
            "g1:    2x(3 trans + 6 rot + 7 joints)",
            f"r1pro roundtrip max err "
            f"{float((rw['back'] - rw['native']).abs().max()):.1e}",
            f"g1    roundtrip max err "
            f"{float((gw['back'] - gw['native']).abs().max()):.1e}",
            "see ../action for what the dims mean",
        ], INV),
        ("2'. un-normalize", "same statistics as forward", [
            f"mode = {cfg.norm_mode!r} (tiny; UNDISCLOSED)",
            "min_max / mean_std: exactly invertible",
            "q99: lossy outside [q01,q99] (clamp)",
            "scale: upstream has NO inverse at all",
            "upstream state_action.py L193-L213",
        ], INV),
        ("3'. un-pad by mask", f"{cfg.max_action_dim} -> native dims", [
            "ORDER MATTERS: un-normalize THEN un-pad",
            "(statistics are stored at padded width)",
            f"g1 keeps dims 0..{gw['emb'].action_dim - 1}, drops the rest",
            "padded dims were never supervised",
            "see ../dit for the masked loss",
        ], INV),
        ("4'. batch -> sample", "split the batch dim", [
            "per-sample embodiment_id selects which",
            "MLP adapter pair decodes it",
            f"ids in this batch: {v(d['batch']['embodiment_id'].numpy())}",
            "upstream embodiment_tags.py L42-L47",
            "",
        ], INV),
        ("5'. placeholder: no inverse", "human state", [
            "there is nothing to recover: the human",
            "never had proprioception",
            "at inference the robot always has it,",
            "so this branch is training-only",
            "paper Sec. 2.3",
        ], "#8f8f8f"),
        ("6'. augment: no inverse", "images", [
            "crop + jitter are lossy on purpose",
            "inference uses the deterministic path",
            "(resize only, no random crop/jitter)",
            "EgoScale does not state its eval-time",
            "preprocessing -> README Sec. 8",
        ], "#8f8f8f"),
        ("7'. no action at inference", "training=False", [
            "upstream apply_single only builds",
            "action / action_mask when training",
            "(transforms.py L316-L323)",
            f"inference keys: video, view_mask,",
            "language, state, state_mask, emb_id",
        ], INV),
    ]
    for (title, shape, body, color), x in zip(inv, xs[::-1]):
        box(ax, x, yi, w, hh, title, shape, body, color)
    for a, b in zip(xs[::-1][:-1], xs[::-1][1:]):
        arrow(ax, a, yi + hh / 2, b + w, yi + hh / 2, INV)

    ax.text(0.5, 0.988, "EgoScale sample construction: one demonstration -> one batch that mixes "
                        "humans and two robots",
            ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax.text(0.5, 0.950,
            "top row = forward (training-data preprocessing); boxes 4a/4b are the ONLY place the "
            "human and robot paths differ     bottom row = inverse (deployment), grey = not "
            "invertible     numbers from one real tiny run",
            ha="center", va="top", fontsize=10, color="#555555")
    ax.text(0.5, 0.078,
            "Coupling that must be kept together: the normalization statistics, the padded width, "
            "and action_mask are one unit -- un-normalizing at the wrong width or dropping the mask "
            "silently changes the data.     "
            f"max_state_dim / max_action_dim / the norm mode / H are all UNDISCLOSED; tiny uses "
            f"{cfg.max_state_dim} / {cfg.max_action_dim} / {cfg.norm_mode!r} / {cfg.action_horizon} "
            "only to make the code path run (README Sec. 8).",
            ha="center", va="center", fontsize=9.4, color="#222222",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3f3f3", edgecolor="#bcbcbc"))
    ax.text(0.5, 0.014,
            "paper arXiv:2602.16710v1 Sec. 2.2 / Sec. 2.3 / Sec. 2.5   |   mechanism after "
            "Isaac-GR00T@4af2b62 gr00t/model/transforms.py L240-L299, "
            "gr00t/data/transform/state_action.py L98-L213",
            ha="center", va="center", fontsize=8.6, color="#777777")

    over = [ln for _, _, body, _ in fwd + inv for ln in body if len(ln) > 43]
    assert not over, f"这些行会溢出框: {over}"

    out = pathlib.Path(__file__).with_name("pipeline.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
