"""Generate figs/pipeline.png: egocentric sensor streams -> cross-embodiment action chunk step by
step (top row), and the mirrored inverse (bottom row, grey = not invertible). Values come from one
real tiny run.
Run: uv run python gear/egoscale/action/figs/make_pipeline.py
"""

import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from gear.egoscale.action.action import (  # noqa: E402
    _FINGERTIP_IDX,
    ToyHand22,
    _synthetic_stream,
    build_action_chunk,
    decode_se3,
    human_keypoints_in_wrist_frame,
    relative_wrist_motion,
    retarget_chunk,
    tiny,
    wrist_pose_world,
)

FWD = "#2c5f8a"
INV = "#8a5a2c"
DEAD = "#8f8f8f"


def v(x, n=3):
    return np.array2string(np.asarray(x), precision=n, suppress_small=True)


def run_tiny():
    torch.manual_seed(0)
    cfg, hand = tiny(), ToyHand22()
    T_wc, H_c, _ = _synthetic_stream(cfg, hand)
    W_w = wrist_pose_world(T_wc, H_c)
    dW = relative_wrist_motion(W_w)
    kp = human_keypoints_in_wrist_frame(T_wc, H_c, W_w)

    t0 = time.time()
    qs, infos = [], []
    for h in range(2):
        q, info = retarget_chunk(kp[:, h].double().numpy(), hand, cfg)
        qs.append(q)
        infos.append(info)
    ms = 1e3 * (time.time() - t0) / (2 * cfg.n_frames)

    q_hand = torch.from_numpy(np.stack(qs, axis=1)).float()
    action = build_action_chunk(dW, "full", cfg.rot_rep, q_hand=q_hand)
    back = decode_se3(action[:, :9], cfg.rot_rep)
    tips = torch.from_numpy(
        np.stack(
            [
                [hand.fk_pose(q_hand[t, h].numpy())[_FINGERTIP_IDX] for h in range(2)]
                for t in range(cfg.n_frames)
            ]
        )
    ).float()
    return dict(cfg=cfg, T_wc=T_wc, H_c=H_c, W_w=W_w, dW=dW, kp=kp, q_hand=q_hand,
                action=action, back=back, tips=tips, info=infos[0], ms=ms)


def box(ax, x, y, w, h, title, shape, body, color):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006", linewidth=1.6,
                       edgecolor=color, facecolor=color + "10", mutation_aspect=0.45)
    )
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.072, shape, ha="center", va="top",
            fontsize=8.8, family="monospace", color="#333333")
    ax.text(x + 0.010, y + h - 0.115, "\n".join(body), ha="left", va="top",
            fontsize=8.2, family="monospace", color="#444444", linespacing=1.6)


def arrow(ax, x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color, shrinkA=0, shrinkB=0))


def main():
    d = run_tiny()
    cfg = d["cfg"]
    fig, ax = plt.subplots(figsize=(30.0, 9.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n, gap = 7, 0.011
    w = (1 - 0.02 - (n - 1) * gap) / n
    xs = [0.01 + i * (w + gap) for i in range(n)]
    yf, yi, hh = 0.545, 0.135, 0.285

    fwd = [
        ("1. raw streams", "T_wc (8,4,4) | H_c (8,2,21,4,4)", [
            "SLAM camera pose + 21 hand",
            "keypoint poses, 30 FPS, camera fr.",
            f"T_wc[0].t = {v(d['T_wc'][0, :3, 3].numpy())}",
            f"H_c[0,L,0].t = {v(d['H_c'][0, 0, 0, :3, 3].numpy())}",
            "paper Sec. 2.1 Raw Sensor Streams",
        ], FWD),
        ("2. wrist in world", "W_w (8,2,4,4)", [
            "W_t = T_t^{w<-c} . H_t^{c,0}",
            "wrist is keypoint index 0",
            f"W[0,L].t = {v(d['W_w'][0, 0, :3, 3].numpy())}",
            f"W[7,L].t = {v(d['W_w'][7, 0, :3, 3].numpy())}",
            "paper Sec. 2.1",
        ], FWD),
        ("3. relative motion", "dW (8,2,4,4)", [
            "dW_t = (W_0)^-1 . W_t",
            "reference = frame 0 OF THIS CHUNK",
            f"dW[0] == I : "
            f"{torch.allclose(d['dW'][0], torch.eye(4).expand(2, 4, 4), atol=1e-5)}",
            f"dW[7,L].t = {v(d['dW'][7, 0, :3, 3].numpy())}",
            "invariant to the SLAM world frame",
        ], FWD),
        ("4. keypoints in wrist", "kp (8,2,20,3)", [
            "(W_t)^-1 T_t H_t^{c,i}, drop wrist",
            "21 -> 20, same frame as robot FK",
            f"kp[0,L,midTIP]={v(d['kp'][0, 0, 11].numpy(), 3)}",
            f"palm scale s = {d['info']['scale']:.4f}",
            "paper App. D",
        ], FWD),
        ("5. per-frame NLP", "q_raw (8,2,22)", [
            "min w_pos . SUM ||FK(q) - s.p||^2",
            "s.t. URDF joint limits (the only one)",
            "CasADi + IPOPT, warm start = prev",
            f"converged {int(d['info']['converged'].sum())}/8, "
            f"resid[-1] {d['info']['residual'][-1]:.1e}",
            f"{d['ms']:.1f} ms/frame (CPU, 1 core)",
        ], FWD),
        ("6. exponential filter", "q (8,2,22)", [
            "q~_t = a.q_t + (1-a).q~_{t-1}",
            f"a = {cfg.alpha}  (tiny only, NOT paper)",
            f"q[0,L,:3] = {v(d['q_hand'][0, 0, :3].numpy())}",
            f"q[7,L,:3] = {v(d['q_hand'][7, 0, :3].numpy())}",
            "paper App. D",
        ], FWD),
        ("7. action chunk (full)", "action (8,62)", [
            "per hand: 3 trans + 6 rot + 22 jt",
            "two hands -> 2 x 31 = 62",
            f"a[1,:3] (trans)={v(d['action'][1, :3].numpy())}",
            f"a[1,9:12] (jt)={v(d['action'][1, 9:12].numpy())}",
            "paper Sec. 3.6 default repr.",
        ], FWD),
    ]
    for (title, shape, body, color), x in zip(fwd, xs):
        box(ax, x, yf, w, hh, title, shape, body, color)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, yf + hh / 2, b, yf + hh / 2, FWD)

    inv = [
        ("1'. robot command", "62 -> arm + hand", [
            "arm: dW IS the relative EEF cmd",
            "hand: 22 dims ARE target joint angles",
            "paper Sec. 2.5: arms relative EEF,",
            "Sharpa hand in joint space",
            "-> no IK, no extra calibration",
        ], INV),
        ("2'. absolute wrist", "W_t (8,2,4,4)", [
            "W_t = W_0 . dW_t",
            "W_0 must be the wrist pose AT",
            "INFERENCE TIME; a mid-execution",
            "pose makes the whole chunk drift",
            f"max |W_0.dW - W| = "
            f"{float((d['W_w'][0:1] @ d['dW'] - d['W_w']).abs().max()):.1e}",
        ], INV),
        ("3'. decode SE(3)", "(8,62) -> dW (8,4,4)", [
            "rotation_6d: Gram-Schmidt on 2 rows",
            "decode(encode(X)) == X exactly",
            f"max elementwise err = "
            f"{float((d['back'] - d['dW'][:, 0]).abs().max()):.1e}",
            "rot_rep is UNDISCLOSED, must be",
            "passed explicitly: README Sec. 8",
        ], INV),
        ("4'. filter: NOT invertible", "q (8,2,22)", [
            "first-order low pass is lossy",
            "a < 1 cannot recover q_raw",
            f"this run: max |q - q_raw| = "
            f"{float(np.abs(d['q_hand'][:, 0].numpy() - d['info']['raw']).max()):.3f} rad",
            "(the robot consumes q, so the",
            "inverse path never needs q_raw)",
        ], DEAD),
        ("5'. retarget: NOT invertible", "22 -> 20x3 only", [
            "FK(q) recovers keypoints, but human",
            "bone lengths and the 21st point are",
            "projected away: many-to-one",
            "on solver failure: keep the previous",
            f"frame's solution (fallbacks: {d['info']['n_fallback']})",
        ], DEAD),
        ("6'. world frame: gone", "dW -x-> W_w", [
            "the SLAM world origin is arbitrary",
            "and not comparable across videos;",
            "dW cancels it, so it cannot be",
            "restored -- that cancellation is why",
            "the representation is relative (2.1)",
        ], DEAD),
        ("7'. fingertip branch", "tips (8,2,5,4,4)", [
            "Sec. 3.6 ablation: supervise the 5",
            "fingertips only, then an MLP -> 22",
            f"tip[0,L,thumb]={v(d['tips'][0, 0, 0, :3, 3].numpy(), 3)}",
            "joint feasibility is not guaranteed",
            "-> unstable grasps; see fig 2",
        ], INV),
    ]
    for (title, shape, body, color), x in zip(inv, xs[::-1]):
        box(ax, x, yi, w, hh, title, shape, body, color)
    for a, b in zip(xs[::-1][:-1], xs[::-1][1:]):
        arrow(ax, a, yi + hh / 2, b + w, yi + hh / 2, INV)

    ax.text(0.5, 0.988, "EgoScale human action representation: egocentric sensor streams -> "
                        "cross-embodiment action chunk",
            ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax.text(0.5, 0.950,
            "top row = forward (training-data preprocessing)     bottom row = inverse (deployment); "
            "grey boxes are steps that cannot be inverted     "
            "all numbers from one real tiny run: H=8 frames, 2 hands, rotation_6d",
            ha="center", va="top", fontsize=10, color="#555555")
    ax.text(0.5, 0.072,
            "Coupling that must be kept together: frame 0 of the chunk is BOTH the reference of dW "
            "and the W_0 the inverse needs -- store them as a pair.     "
            "The palm scale s is per-chunk and must be recomputed.     "
            "a and the objective weights are undisclosed (README Sec. 8); the tiny values only make "
            "the code path run.",
            ha="center", va="center", fontsize=9.4, color="#222222",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3f3f3", edgecolor="#bcbcbc"))
    ax.text(0.5, 0.014,
            "paper arXiv:2602.16710v1 Sec. 2.1 / Sec. 3.6 / App. D   |   rotation encoding "
            "convention after Isaac-GR00T@4af2b62 gr00t/data/transform/state_action.py L29-L95",
            ha="center", va="center", fontsize=8.6, color="#777777")

    over = [ln for _, _, body, _ in fwd + inv for ln in body if len(ln) > 40]
    assert not over, f"这些行会溢出框: {over}"

    out = pathlib.Path(__file__).with_name("pipeline.png")
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
