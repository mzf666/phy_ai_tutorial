"""Generate figs/pipeline.png: one real inference end to end (top row, with measured per-stage
latency from a real tiny run), and the assembly constraints read backwards (bottom row -- what each
step depends on, and what breaks if the order is swapped).
Run: uv run python gear/egoscale/infer/figs/make_pipeline.py
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
from gear.egoscale.data.data import CAMERA_SLOTS  # noqa: E402
from gear.egoscale.infer.model import build_tiny_policy, fake_observation  # noqa: E402

FWD = "#2c5f8a"
BACK = "#8a5a2c"


def v(x, n=3):
    return np.array2string(np.asarray(x), precision=n, suppress_small=True)


def run_tiny():
    policy, cfg = build_tiny_policy()
    rng = np.random.default_rng(0)
    obs = fake_observation(cfg, "r1pro_sharpa", rng)
    action, info = policy.act(obs)
    cmd = policy.to_robot_command(action, "r1pro_sharpa")
    emb = cfg.embodiments["r1pro_sharpa"]
    n_par = sum(p.numel() for p in list(policy.backbone.parameters())
                + list(policy.expert.parameters()))
    return dict(policy=policy, cfg=cfg, obs=obs, action=action, info=info, cmd=cmd,
                emb=emb, n_par=n_par)


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
    policy, cfg, info, emb = d["policy"], d["cfg"], d["info"], d["emb"]
    dcfg = policy.expert.cfg

    fig, ax = plt.subplots(figsize=(30.0, 10.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n, gap = 7, 0.011
    w = (1 - 0.02 - (n - 1) * gap) / n
    xs = [0.01 + i * (w + gap) for i in range(n)]
    yf, yi, hh = 0.555, 0.145, 0.275

    fwd = [
        ("1. observation", "images + text + state", [
            f"slots present: {len(d['obs'].images)} of {len(CAMERA_SLOTS)}",
            "fixed order: head | left | right wrist",
            f"state ({emb.state_dim},) native units",
            f"instruction: '{d['obs'].instruction[:22]}...'",
            "paper Sec. 2.5",
        ], FWD),
        ("2. normalize + assemble", f"video (T,{len(CAMERA_SLOTS)},H,W,3)", [
            "state normalized with the TRAINING",
            "statistics, then padded to max_state_dim",
            "missing camera slots are black-filled",
            f"{info['preprocess_ms']:.2f} ms",
            "see ../data",
        ], FWD),
        ("3. backbone", f"phi {info['phi_shape']}", [
            "three views + instruction -> one",
            "conditioning sequence, middle LLM layer",
            f"computed ONCE for the whole chunk",
            f"{info['backbone_ms']:.2f} ms",
            "see ../backbone",
        ], FWD),
        ("4. state token", f"({dcfg.state_horizon},{dcfg.input_embedding_dim})", [
            "embodiment-specific MLP by embodiment_id",
            f"id = {emb.embodiment_id}, has_proprio = {emb.has_proprio}",
            "a human sample would get the learnable",
            "placeholder token instead (Sec. 2.3)",
            "see ../dit",
        ], FWD),
        ("5. K-step denoise", f"({dcfg.action_horizon},{dcfg.max_action_dim})", [
            f"A <- N(0,I), then {info['n_denoise']} Euler steps",
            "every step reuses the SAME phi",
            "output is in the NORMALIZED space",
            f"{info['denoise_ms']:.2f} ms for all {info['n_denoise']} steps",
            "see ../dit",
        ], FWD),
        ("6. un-normalize, un-pad", f"({dcfg.action_horizon},{emb.action_dim})", [
            "ORDER: un-normalize FIRST, then un-pad",
            "(statistics are stored at padded width)",
            f"{dcfg.max_action_dim} -> {emb.action_dim} native dims",
            f"{info['postprocess_ms']:.2f} ms",
            "see ../data",
        ], FWD),
        ("7. robot command", "arm + hand", [
            f"arm {tuple(d['cmd'].arm_delta_pose.shape)} relative EEF delta",
            f"hand {tuple(d['cmd'].hand_joints.shape)} target joint angles",
            f"arm[0,L,:3] = {v(d['cmd'].arm_delta_pose[0, 0, :3].numpy())}",
            "no IK, no extra calibration (Sec. 2.5)",
            f"total {info['total_ms']:.2f} ms (CPU, tiny)",
        ], FWD),
    ]
    for (title, shape, body, color), x in zip(fwd, xs):
        box(ax, x, yf, w, hh, title, shape, body, color)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, yf + hh / 2, b, yf + hh / 2, FWD)

    back = [
        ("7'. what the robot needs", "two interfaces", [
            "the arm controller wants an INCREMENT,",
            "the hand controller wants an ANGLE",
            "mixing them up is silent: both are just",
            "floats of the right shape",
            "paper Sec. 2.5",
        ]),
        ("6'. why this order", "un-normalize -> un-pad", [
            "the statistics vector has max_action_dim",
            "entries; slicing to the native width first",
            "would pair dim i with statistic i anyway,",
            "but any future per-embodiment padding",
            "offset would then silently misalign",
        ]),
        ("5'. what K costs", "K forwards of the DiT", [
            "the backbone runs once, the DiT K times",
            f"here: {info['backbone_ms']:.2f} ms vs "
            f"{info['denoise_ms']:.2f} ms",
            "that ratio is why cross-attention beats",
            "concatenating vision into self-attention",
            "GR00T N1 uses K = 4",
        ]),
        ("4'. what needs W_0", "absolute wrist pose", [
            "to turn dW back into an absolute pose you",
            "need the wrist pose AT INFERENCE TIME,",
            "not the one mid-execution -- otherwise the",
            "whole chunk drifts",
            "see ../action",
        ]),
        ("3'. what phi carries", "and what it does not", [
            "phi is conditioning only: it never sees",
            "the noisy action, so it cannot be refined",
            "between denoising steps",
            "its padding mask is never applied upstream",
            "-> ../dit README Sec. 1.x row 1",
        ]),
        ("2'. statistics are state", "training-time artefact", [
            "the normalizer is NOT part of the model",
            "checkpoint in this repo -- ship it with the",
            "weights or the policy silently outputs",
            "actions in the wrong units",
            "EgoScale never says where theirs came from",
        ]),
        ("1'. open-loop execution", "H steps, execute n", [
            f"one inference gives H = {dcfg.action_horizon} steps;",
            "how many are actually executed before",
            "re-planning is UNDISCLOSED (README Sec. 8)",
            "that number sets the re-planning rhythm",
            "and therefore the closed-loop behaviour",
        ]),
    ]
    for (title, shape, body), x in zip(back, xs[::-1]):
        box(ax, x, yi, w, hh, title, shape, body, BACK)
    for a, b in zip(xs[::-1][:-1], xs[::-1][1:]):
        arrow(ax, a, yi + hh / 2, b + w, yi + hh / 2, BACK)

    ax.text(0.5, 0.988, "EgoScale end-to-end inference: three views and one sentence -> "
                        "an arm increment and a hand pose",
            ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax.text(0.5, 0.950,
            "top row = one real forward pass on this repo's tiny models, with measured latency     "
            "bottom row = the same chain read backwards: what each step depends on and what breaks "
            "if the order is swapped",
            ha="center", va="top", fontsize=10, color="#555555")
    ax.text(0.5, 0.078,
            f"This tiny policy has {d['n_par']:,} parameters and runs in "
            f"{info['total_ms']:.1f} ms on a CPU -- neither number means anything about EgoScale, "
            "which discloses no parameter count and no latency at all.     "
            "GR00T N1 reports 63.9 ms end to end on an L40 for a 16-step chunk with 4 denoising "
            "steps (arXiv:2503.14734v2 Sec. 2.1); that is GR00T's number (README Sec. 8).",
            ha="center", va="center", fontsize=9.4, color="#222222",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3f3f3", edgecolor="#bcbcbc"))
    ax.text(0.5, 0.014,
            "paper arXiv:2602.16710v1 Sec. 2.3 / Sec. 2.5   |   assembly after Isaac-GR00T@4af2b62 "
            "gr00t/model/gr00t_n1.py L171-L198",
            ha="center", va="center", fontsize=8.6, color="#777777")

    over = [ln for _, _, body, _ in fwd for ln in body if len(ln) > 43]
    over += [ln for _, _, body in back for ln in body if len(ln) > 43]
    assert not over, f"这些行会溢出框: {over}"

    out = pathlib.Path(__file__).with_name("pipeline.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
