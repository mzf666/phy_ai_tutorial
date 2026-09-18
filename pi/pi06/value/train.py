"""pi0.6* value-function training and the advantage machinery: Eq. 1 (cross entropy over 201 return bins), the web-data
co-training term, the two advantage estimators (pre-training: A = R_t - V(o_t); post-training: N = 50-step lookahead),
the per-task improvement threshold epsilon_ell as a quantile (30% / 40% / 10% positive), the indicator I_t with human
corrections forced True and the SFT stage fixed True, and one training step.

Sources of truth:
  paper    pi0.6* arXiv:2511.14759v2 Eq. 1 (Sec. IV-A), Sec. IV-B (I_t = 1[A > eps_ell]; corrections forced True),
           Sec. V-C (co-train on a small mixture of multi-modal web data), Sec. V-D (eps_ell = 30th percentile of
           predicted values in pre-training; I_t = True during SFT; value function run on-the-fly during VLA training;
           both V and pi finetuned from the pre-trained checkpoint), Appendix F (advantage estimation with N = 50;
           pre-training uses N = T; thresholds 30% / 40% / 10%; 10k datapoints for the estimate), Algorithm 1
  openpi   no value-function code upstream; optimizer / EMA / clipping helpers are pi.pi0.train (openpi optimizer.py)
Licenses: Apache-2.0 for the openpi pieces. Re-implements, does not copy.

Every optimisation hyper-parameter of the value function is undisclosed (README Sec. 8): `paper_config()` carries
None; the tiny run uses pi0's TrainConfig values only to execute the code path.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from pi.fast.train.train import cross_entropy
from pi.pi0.train.train import EMA, TrainConfig, clip_and_step, lr_at, make_optimizer  # noqa: F401  (re-exported)
from pi.pi06.data.data import SEG_TEXT, Pi06Observation
from pi.pi06.value.model import ValueFunction

LOOKAHEAD_N = 50  # App. F: "We use N = 50 lookahead to calculate this advantage" (post-training)
POSITIVE_FRACTION_PRETRAIN = 0.30  # App. F / Sec. V-D: ~30% of demonstration data has positive advantage (30th percentile of values)
POSITIVE_FRACTION_FINETUNE = 0.40  # App. F: ~40% of the evaluation rollouts in each iteration
POSITIVE_FRACTION_TSHIRT = 0.10  # App. F: T-shirt / shorts laundry, only ~10% positive (demonstrations slow but reliable)
THRESHOLD_SAMPLE_SIZE = 10_000  # App. F: "calculated on a random sample of 10k datapoints"


# ======================================================================================
# 1. Configurations. Everything about the optimiser is undisclosed.
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class ValueTrainConfig:
    stage: str  # "pretrain" | "finetune"
    positive_fraction: float
    lookahead: int | None  # None = N = T (whole-episode return)
    optimizer: TrainConfig | None  # None = undisclosed (README Sec. 8)
    note: str = ""


def pretrain_config() -> ValueTrainConfig:
    return ValueTrainConfig("pretrain", POSITIVE_FRACTION_PRETRAIN, None, None, "Sec. V-D / App. F: whole-episode return, 30% positive; optimizer undisclosed")


def finetune_config(positive_fraction: float = POSITIVE_FRACTION_FINETUNE) -> ValueTrainConfig:
    return ValueTrainConfig("finetune", positive_fraction, LOOKAHEAD_N, None, "App. F: N = 50 lookahead, 40% positive (10% for T-shirts); from the pre-trained checkpoint")


def tiny_config() -> TrainConfig:
    return TrainConfig(warmup_steps=2, peak_lr=1e-3, decay_steps=10, decay_lr=1e-4, batch_size=2, num_train_steps=3)  # tiny only, not paper values


# ======================================================================================
# 2. Eq. 1: cross entropy between the discretised empirical return and p_phi(V | o, ell). Plus the co-training term of
#    Sec. V-C: on "text" layout samples the backbone predicts the text target through the tied head, so the value
#    backbone keeps its VLM knowledge ("to prevent overfitting").
# ======================================================================================
def value_loss(vf: ValueFunction, obs: Pi06Observation) -> torch.Tensor:
    """Per-sample CE f32[B]: -log p_phi(V = R^B_t | o_t, ell) (Eq. 1). obs is a "value" layout batch with value_bin."""
    logits, _ = vf(obs)
    return F.cross_entropy(logits.float(), obs.value_bin, reduction="none")


def cotrain_text_loss(vf: ValueFunction, obs: Pi06Observation) -> torch.Tensor:
    """Per-sample CE f32[B] on a "text" layout batch: next-token prediction of the SEG_TEXT target through the tied head
    (only positions whose NEXT token carries a loss get a logit). Sec. V-C; the mixture weight is undisclosed."""
    _, h = vf(obs)
    n_img = h.shape[1] - obs.tokens.shape[1]
    logits = vf.vlm.logits_head(h[:, n_img : n_img + obs.tokens.shape[1] - 1])
    targets = obs.tokens[:, 1:]
    return cross_entropy(logits, targets, obs.loss_mask[:, 1:] & (obs.segment[:, 1:] == SEG_TEXT))


# ======================================================================================
# 3. Advantages. Sec. III: A^pi(o_t, a_t) = E[sum_{t'=t}^{t+N-1} r_t' + V(o_{t+N})] - V(o_t). All quantities are in the
#    normalised units of ../data (rewards / T_max, values in [-1, 0]).
#    App. F pre-training: "A = sum_{t'} r_t' - V(o_t), setting N = T for each episode": the whole remaining return
#    minus the value, one value call per sample (the paper writes the sum from t' = 0; read as t' = t, README Sec. 8).
#    App. F post-training: N = 50 with the bootstrap V(o_{t+N}); past the episode end the sum runs to T and no
#    bootstrap is added (the terminal reward already carries the outcome).
# ======================================================================================
def advantage_whole_episode(norm_returns: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A_t = R_t / T_max - V(o_t). norm_returns, values f32[T + 1] -> f32[T + 1]."""
    return np.asarray(norm_returns, np.float32) - np.asarray(values, np.float32)


def advantage_nstep(norm_rewards: np.ndarray, values: np.ndarray, n: int = LOOKAHEAD_N) -> np.ndarray:
    """A_t = sum_{t'=t}^{min(t+n-1, T)} r_t' / T_max + [t + n <= T] V(o_{t+n}) - V(o_t). f32[T + 1]."""
    r, v = np.asarray(norm_rewards, np.float32), np.asarray(values, np.float32)
    T1 = len(r)
    c = np.concatenate([[0.0], np.cumsum(r)])  # c[k] = sum r[:k]
    out = np.empty(T1, np.float32)
    for t in range(T1):
        end = min(t + n, T1)
        boot = v[t + n] if t + n < T1 else 0.0
        out[t] = c[end] - c[t] + boot - v[t]
    return out


def improvement_threshold(advantages: np.ndarray, positive_fraction: float, rng: np.random.Generator | None = None,
                          sample_size: int = THRESHOLD_SAMPLE_SIZE) -> float:
    """eps_ell such that ~positive_fraction of the (sampled) advantages exceed it (App. F: 30% / 40% / 10%; 10k sample).
    Sec. V-D phrases the pre-training rule as the 30th percentile of predicted values; App. F as 30% positive advantage."""
    a = np.asarray(advantages, np.float32).reshape(-1)
    if rng is not None and len(a) > sample_size:
        a = rng.choice(a, sample_size, replace=False)
    return float(np.quantile(a, 1.0 - positive_fraction))


def improvement_indicator(advantages: np.ndarray, threshold: float, is_correction: np.ndarray | None = None, sft: bool = False) -> np.ndarray:
    """I_t = 1[A_t > eps_ell] (Sec. IV-B), True on human-correction steps (Sec. IV-B, V-D), all True in the SFT stage (Sec. V-D)."""
    a = np.asarray(advantages, np.float32)
    if sft:
        return np.ones(a.shape, bool)
    ind = a > threshold
    if is_correction is not None:
        ind = ind | np.asarray(is_correction, bool)
    return ind


@torch.no_grad()
def episode_values(vf: ValueFunction, step_batches) -> np.ndarray:
    """Run the value function over an episode given as a list of "value" layout batches (one step per row) -> f32[T + 1].
    Sec. V-D: this is the on-the-fly call during VLA training (one call per sample in pre-training)."""
    vf.eval()
    return np.concatenate([vf.value(obs).cpu().numpy() for obs in step_batches]).astype(np.float32)


def label_episode(vf: ValueFunction, step_batches, norm_rewards: np.ndarray, norm_returns: np.ndarray, *, cfg: ValueTrainConfig,
                  threshold: float, is_correction: np.ndarray | None = None, sft: bool = False) -> dict:
    """Values -> advantages (whole-episode or N-step per cfg) -> indicators for one episode. The threshold comes from
    improvement_threshold over the task's data (10k sample), not from this episode alone."""
    v = episode_values(vf, step_batches)
    a = advantage_whole_episode(norm_returns, v) if cfg.lookahead is None else advantage_nstep(norm_rewards, v, cfg.lookahead)
    return {"values": v, "advantages": a, "indicators": improvement_indicator(a, threshold, is_correction, sft)}


# ======================================================================================
# 4. One training step of the value function (Eq. 1 + optional web co-training). Optimiser: pi0's helpers, values undisclosed.
# ======================================================================================
def train_step(vf: ValueFunction, value_batch: Pi06Observation, params, optimizer, ema: EMA | None, cfg: TrainConfig, step: int,
               text_batch: Pi06Observation | None = None, cotrain_weight: float = 1.0) -> dict[str, float]:
    """loss = mean CE over the 201 bins (+ cotrain_weight x mean text CE). cotrain_weight is undisclosed (README Sec. 8)."""
    vf.train()
    ce = value_loss(vf, value_batch).mean()
    loss = ce
    text_ce = torch.zeros(())
    if text_batch is not None:
        text_ce = cotrain_text_loss(vf, text_batch).mean()
        loss = loss + cotrain_weight * text_ce
    loss.backward()
    grad_norm = clip_and_step(params, optimizer, cfg, step)
    if ema is not None:
        ema.update(vf)
    return {"loss": float(loss.detach()), "value_ce": float(ce.detach()), "text_ce": float(text_ce.detach()), "grad_norm": grad_norm, "lr": lr_at(step, cfg)}


# ======================================================================================
# 5. Walk one episode through labels -> values -> advantages -> indicators, and one training step.
#    uv run python -m pi.pi06.value.train
# ======================================================================================
def main():
    from pi.pi06.data.data import C_FAIL_TINY, STATIC_IMAGE_KEYS, EpisodeLabels, build_pi06_batch, episode_rewards, tiny_pi06_tokenizer, unit_stats
    from pi.pi06.value.model import tiny_value_function

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    H, d, T1, Tmax = 10, 7, 12, 40
    vf = tiny_value_function()
    seq = tiny_pi06_tokenizer(H, d)
    lab = EpisodeLabels("fold the shirt", success=True, max_episode_len=Tmax, num_steps=T1, is_correction=np.array([False] * 8 + [True] * 4))
    v_target, bins = lab.value_targets(C_FAIL_TINY)
    r_norm = episode_rewards(T1, True, C_FAIL_TINY) / Tmax
    print(f"episode: {T1} steps, success, T_max {Tmax}, corrections at steps 8-11; value targets {[f'{x:.3f}' for x in v_target[:4]]} ... {v_target[-1]:.1f}, bins {bins[:4].tolist()} ... {bins[-1]}")
    steps = []
    for t in range(T1):
        raw = {"images": {k: rng.integers(0, 256, (1, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
               "state": rng.uniform(-0.5, 0.5, (1, d)).astype(np.float32), "prompt": ["fold the shirt"]}
        obs, _ = build_pi06_batch(raw, unit_stats(d), seq, layout="value", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, value_bins=[int(bins[t])])
        steps.append(obs)
    print(f"[values]     {T1} value calls (layout value, {int(steps[0].token_mask[0].sum())} text tokens each)")
    for cfg in (pretrain_config(), finetune_config()):
        out = label_episode(vf, steps, r_norm, v_target, cfg=cfg, threshold=0.0, is_correction=lab.is_correction)
        a = out["advantages"]
        print(f"[{cfg.stage:9s}] V(o_t) {[f'{x:.3f}' for x in out['values'][:3]]} ...  A_t ({'N = T' if cfg.lookahead is None else f'N = {cfg.lookahead}'}) {[f'{x:+.3f}' for x in a[:3]]} ... {a[-1]:+.3f}")
    a = out["advantages"]
    pool = np.concatenate([a, rng.normal(0, 0.05, 500)])  # stand-in for the task's 10k-sample pool
    for name, frac in (("pretrain 30%", POSITIVE_FRACTION_PRETRAIN), ("finetune 40%", POSITIVE_FRACTION_FINETUNE), ("T-shirt 10%", POSITIVE_FRACTION_TSHIRT)):
        eps = improvement_threshold(pool, frac, rng)
        ind = improvement_indicator(a, eps, lab.is_correction)
        print(f"[threshold]  {name}: eps = {eps:+.4f} -> {int((pool > eps).sum())}/{len(pool)} of the pool positive; this episode I_t = {''.join('1' if x else '0' for x in ind)} (steps 8-11 forced True: corrections)")
    print(f"[sft]        I_t = {''.join('1' if x else '0' for x in improvement_indicator(a, 0.0, sft=True))} (Sec. V-D: fixed True in the SFT stage)")
    # one training step: Eq. 1 on a 2-step batch + web co-training on a caption
    vb, _ = build_pi06_batch({"images": {k: rng.integers(0, 256, (2, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
                              "state": rng.uniform(-0.5, 0.5, (2, d)).astype(np.float32), "prompt": ["fold the shirt"] * 2},
                             unit_stats(d), seq, layout="value", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=True, value_bins=[int(bins[0]), int(bins[-1])])
    tb, _ = build_pi06_batch({"images": {k: rng.integers(0, 256, (2, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
                              "state": rng.uniform(-0.5, 0.5, (2, d)).astype(np.float32), "prompt": ["caption the image"] * 2},
                             unit_stats(d), seq, layout="text", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=True, target_text=["a dog catches a frisbee", "a cup on a table"])
    cfg = tiny_config()
    params = [p for p in vf.parameters() if p.requires_grad]
    opt = make_optimizer(params, cfg)
    ema = EMA(vf, 0.99)
    for step in range(cfg.num_train_steps):
        m = train_step(vf, vb, params, opt, ema, cfg, step, text_batch=tb)
        print(f"[train {step}]  loss {m['loss']:.3f} = value CE {m['value_ce']:.3f} (ln 201 = {np.log(201):.3f}) + text CE {m['text_ce']:.3f}; grad norm {m['grad_norm']:.2f}; lr {m['lr']:.1e} (tiny only)")
    print("\n../train: the policy's KI objective consumes these indicators as the Advantage token; RECAP refits V from V_pre each iteration")


if __name__ == "__main__":
    main()
