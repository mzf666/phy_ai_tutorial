"""Generate figs/pipeline.png: one inference (top row, the K-step Euler loop unrolled) and the
mirrored training forward (bottom row, same modules but Beta-sampled timestep, velocity target and
mask-weighted loss). Values come from one real tiny run.
Run: uv run python gear/egoscale/dit/figs/make_pipeline.py
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
from gear.egoscale.dit.model import ActionExpert, tiny  # noqa: E402
from gear.egoscale.dit.train import flow_matching_loss  # noqa: E402

FWD = "#2c5f8a"
TRN = "#8a5a2c"
DEAD = "#8f8f8f"


def v(x, n=3):
    return np.array2string(np.asarray(x), precision=n, suppress_small=True)


def run_tiny():
    torch.manual_seed(0)
    cfg = tiny()
    m = ActionExpert(cfg)
    b, s_len = 4, 19
    phi = torch.randn(b, s_len, cfg.backbone_embedding_dim)
    phi_mask = torch.ones(b, s_len, dtype=torch.bool)
    phi_mask[1, -4:] = False
    state = torch.randn(b, cfg.state_horizon, cfg.max_state_dim)
    state_mask = torch.ones_like(state, dtype=torch.bool)
    action = torch.randn(b, cfg.action_horizon, cfg.max_action_dim)
    action_mask = torch.zeros_like(action, dtype=torch.bool)
    native = [62, 62, 32, 62]
    for i, n in enumerate(native):
        action_mask[i, :, :n] = True
    emb_id = torch.tensor([2, 0, 3, 1])
    has_proprio = torch.tensor([True, False, True, False])

    m.eval()
    with torch.no_grad():
        sf = m.encode_state(state, emb_id, has_proprio)
        af = m.encode_action(torch.randn_like(action), torch.full((b,), 250), emb_id)
        tok = m.build_tokens(sf, af)
        t0 = time.perf_counter()
        sampled = m.sample(phi, phi_mask, state, state_mask, emb_id, has_proprio)
        ms = 1e3 * (time.perf_counter() - t0)
    m.train()
    out = flow_matching_loss(m, phi, phi_mask, state, action, action_mask, emb_id,
                             has_proprio, generator=torch.Generator().manual_seed(3))
    return dict(cfg=cfg, m=m, phi=phi, phi_mask=phi_mask, sf=sf, af=af, tok=tok,
                sampled=sampled, ms=ms, out=out, native=native, b=b, s_len=s_len,
                action_mask=action_mask)


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
    cfg, m, out = d["cfg"], d["m"], d["out"]
    n_tok = cfg.state_horizon + cfg.num_target_vision_tokens + cfg.action_horizon

    fig, ax = plt.subplots(figsize=(30.0, 10.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n, gap = 7, 0.011
    w = (1 - 0.02 - (n - 1) * gap) / n
    xs = [0.01 + i * (w + gap) for i in range(n)]
    yf, yi, hh = 0.555, 0.145, 0.275

    fwd = [
        ("1. condition + state", f"phi ({d['b']},{d['s_len']},{cfg.backbone_embedding_dim})", [
            "phi comes from ../backbone, computed",
            "ONCE and reused by all K steps",
            f"phi_mask valid {d['phi_mask'].sum(1).tolist()}",
            "(upstream never applies that mask!)",
            f"state {tuple((d['b'], cfg.state_horizon, cfg.max_state_dim))}",
        ], FWD),
        ("2. state -> token", f"-> ({d['b']},{cfg.state_horizon},{cfg.input_embedding_dim})", [
            "CategorySpecificMLP by embodiment_id",
            "has_proprio=False -> LEARNABLE",
            "placeholder token (paper Sec. 2.3)",
            "samples 1 and 3 are human demos",
            "their state values are irrelevant",
        ], FWD),
        ("3. A <- N(0, I)", f"({d['b']},{cfg.action_horizon},{cfg.max_action_dim})", [
            "pure noise, the starting point of the",
            "flow-matching path",
            f"H = {cfg.action_horizon}, padded width {cfg.max_action_dim}",
            f"native dims per sample {d['native']}",
            "upstream L376-L382",
        ], FWD),
        ("4. action -> tokens", f"-> ({d['b']},{cfg.action_horizon},{cfg.input_embedding_dim})", [
            "W1 -> concat(sin/cos tau) -> W2+swish",
            "-> W3, all embodiment specific",
            "+ position_embedding (add_pos_embed)",
            "tau enters as a 1000-bucket integer",
            "upstream L56-L98, L318-L321",
        ], FWD),
        ("5. token layout", f"-> ({d['b']},{n_tok},{cfg.input_embedding_dim})", [
            f"[state {cfg.state_horizon} | future {cfg.num_target_vision_tokens} | "
            f"action {cfg.action_horizon}]",
            "N1.5: 1 + 32 + 16 = 49 tokens",
            "future_tokens get NO supervision here",
            "(they belong to the FLARE objective)",
            "upstream L325-L327",
        ], FWD),
        ("6. DiT", f"-> ({d['b']},{n_tok},{cfg.hidden_size})", [
            f"{cfg.num_layers} blocks: even = cross-attn on phi,",
            "odd = self-attn only (interleave)",
            "timestep enters via AdaLN in EVERY block",
            f"{m.dit.n_cross} cross / "
            f"{len(m.dit.blocks) - m.dit.n_cross} self",
            "upstream L281-L305",
        ], FWD),
        ("7. K-step Euler", f"A ({d['b']},{cfg.action_horizon},{cfg.max_action_dim})", [
            f"K = {cfg.num_inference_timesteps}: A <- A + (1/K) * V",
            f"tau visits "
            f"{[round(s / cfg.num_inference_timesteps, 2) for s in range(cfg.num_inference_timesteps)]}",
            "note: tau = 1.0 is NEVER evaluated",
            f"{d['ms']:.1f} ms for the whole loop (CPU)",
            "upstream L384-L403",
        ], FWD),
    ]
    for (title, shape, body, color), x in zip(fwd, xs):
        box(ax, x, yf, w, hh, title, shape, body, color)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, yf + hh / 2, b, yf + hh / 2, FWD)

    trn = [
        ("1'. sample tau", "tau (B,)", [
            "u ~ Beta(1.5, 1); tau = (s - u)/s",
            f"s={cfg.noise_s}; run {v(out['tau'].detach().numpy(), 2)}",
            "mass sits on SMALL tau = high noise",
            "about 0.15% of samples get tau < 0",
            "upstream L256-L258",
        ], TRN),
        ("2'. add noise", "A_tau (B,H,D)", [
            "A_tau = (1 - tau)*eps + tau*A",
            "tau=0 -> pure noise, tau=1 -> the action",
            "the path is a straight line, so the",
            "velocity along it is constant",
            "upstream L307",
        ], TRN),
        ("3'. velocity target", "A - eps", [
            "upstream regresses A - eps (L308).",
            "GR00T N1 Eq. (1) prints eps - A, which",
            "contradicts its own Euler update;",
            "this repo follows the CODE",
            "README Sec. 1.x row 2",
        ], TRN),
        ("4'. discretize tau", "bucket (B,)", [
            f"(tau * {cfg.num_timestep_buckets}).long()",
            f"this run {out['bucket'].tolist()}",
            "a sinusoidal encoder, not a lookup",
            "table -- so bucket = -1 does not crash",
            "upstream L311",
        ], TRN),
        ("5'. same forward", "pred (B,H,D)", [
            "identical modules to the inference path;",
            "only the timestep source differs",
            "one forward per training step, versus",
            f"K = {cfg.num_inference_timesteps} at inference",
            "upstream L313-L345",
        ], TRN),
        ("6'. masked MSE", "scalar", [
            "(MSE * action_mask).sum() / mask.sum()",
            f"mask True/row {d['action_mask'][:, 0].sum(-1).tolist()} of {cfg.max_action_dim}",
            f"loss {out['loss'].item():.4f}",
            "a plain .mean() would make padded",
            "embodiments look better (see ../data)",
        ], TRN),
        ("7'. no second objective", "single loss", [
            "no cross-entropy term, no auxiliary",
            "head: unlike pi0.5's joint CE+MSE",
            "the only other tokens in the sequence",
            "(future_tokens) are unsupervised here",
            "paper Sec. 2.3",
        ], DEAD),
    ]
    for (title, shape, body, color), x in zip(trn, xs):
        box(ax, x, yi, w, hh, title, shape, body, color)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w, yi + hh / 2, b, yi + hh / 2, TRN)

    ax.text(0.5, 0.988, "EgoScale action expert: one inference (top) and one training step "
                        "(bottom) through the same modules",
            ha="center", va="top", fontsize=15.5, fontweight="bold")
    ax.text(0.5, 0.950,
            "top row = K-step Euler sampling     bottom row = the training forward; it is NOT a "
            "mirror image here because flow matching's inverse is the sampler itself -- the two "
            "rows differ only in where tau comes from and what is regressed     "
            "numbers from one real tiny run",
            ha="center", va="top", fontsize=10, color="#555555")
    ax.text(0.5, 0.078,
            "Coupling that must be kept together: the timestep bucketisation, the Beta parameters "
            "and the velocity sign are one unit -- flip the sign and the Euler update walks away "
            "from the data.     "
            f"EgoScale discloses none of the structural values; tiny uses {cfg.num_layers} DiT "
            f"layers / W={cfg.input_embedding_dim} / K={cfg.num_inference_timesteps}, and the N1.5 "
            "column in figs/dit.png comes from the released checkpoint (README Sec. 8).",
            ha="center", va="center", fontsize=9.4, color="#222222",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#f3f3f3", edgecolor="#bcbcbc"))
    ax.text(0.5, 0.014,
            "paper arXiv:2602.16710v1 Sec. 2.3 / App. D.1   |   Isaac-GR00T@4af2b62 "
            "gr00t/model/action_head/flow_matching_action_head.py L256-L404, "
            "cross_attention_dit.py L31-L307",
            ha="center", va="center", fontsize=8.6, color="#777777")

    over = [ln for _, _, body, _ in fwd + trn for ln in body if len(ln) > 43]
    assert not over, f"这些行会溢出框: {over}"

    out_path = pathlib.Path(__file__).with_name("pipeline.png")
    fig.savefig(out_path, dpi=125, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
