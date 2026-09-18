"""pi0.6* training: the Knowledge-Insulation joint objective (cross entropy on subtask + FAST tokens, plus alpha = 1 times
the flow-matching MSE of the expert, with the expert's gradient stopped inside attention), the three sample kinds and
their loss masks, the indicator modes of the three RECAP stages (offline-RL pre-training / SFT with I = True / RECAP
iterations), the RECAP loop of Algorithm 1 (collect -> refit V from V_pre -> refit pi from pi_pre), one training step,
and the disclosed facts about baselines, results and cost.

Sources of truth:
  paper    pi0.6* arXiv:2511.14759v2 Sec. IV-B (Eq. 3; corrections forced True), IV-C / Algorithm 1 (pre-train V and pi
           on D_demo; per task: SFT, then K iterations of collect / V from V_pre / pi from pi_pre), Sec. V-A (KI recipe,
           factorised log-likelihood, expert does not read FAST), V-B (Eq. 4: CE of discrete actions + alpha_eta flow loss;
           indicator dropped at random instead of tuning alpha), V-D (SFT with I = True; V and pi finetuned from the
           pre-trained checkpoint; one iteration often enough), VI-B (baselines), VI-C (results), App. C (Eq. 9 with
           w(eta) = e^{-eta/2} subsumed in alpha), App. D (PPO / SPO baseline Eq. 11, eta = 0.01), App. F
  KI       Knowledge Insulation arXiv:2505.23705v1 Sec. 5.1 (Eq. 4: L = -sum M^ell log p + alpha M^act ||omega - a - f||^2),
           Sec. 5.2 (Eq. 5-6: stop-gradient on K_b, V_b for expert queries; "we can simply set alpha = 1"), Sec. 7 (~20% more
           compute per step), Fig. 6b (pi0 needs 7.5x the steps), App. B (timestep sampling as pi0)
  card     pi0.6 model card Sec. 2 ("trained with Knowledge Insulation ... the gradient from the action expert does not
           flow back to the main VLM backbone")
  openpi   https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479 has no pi0.6 / KI
           training code; optimizer / EMA / clipping helpers are pi.pi0.train (optimizer.py), the CE is pi.fast.train,
           timestep sampling and the interpolation are pi.pi0.flow_matching.train
Licenses: Apache-2.0 for the openpi pieces. Re-implements, does not copy.

Every optimisation hyper-parameter (lr, batch, steps, EMA, compute) is undisclosed (README Sec. 8): `paper` configs carry
None; the tiny run uses pi0's TrainConfig values only to execute the code path.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import torch

from pi.pi0.flow_matching.train import interpolate, sample_timestep
from pi.pi0.train.train import EMA, TrainConfig, clip_and_step, lr_at, make_optimizer  # noqa: F401  (re-exported)
from pi.pi06.backbone.model import AdaRMSNorm, Pi06
from pi.pi06.data.data import ADVANTAGE_DROPOUT, Pi06Observation, drop_indicator
from pi.pi06.value.model import ValueFunction
from pi.pi06.value.train import POSITIVE_FRACTION_FINETUNE, POSITIVE_FRACTION_PRETRAIN, ValueTrainConfig, finetune_config as value_finetune_config, improvement_threshold, label_episode, pretrain_config as value_pretrain_config

ALPHA = 1.0  # KI Sec. 5.2: with the stop-gradient "we can simply set alpha = 1 in (4), since now the diffusion loss term applies to an independent set of weights"
KI_EXTRA_COMPUTE = 0.20  # KI Sec. 7: "increases computational cost by about 20% during training"
KI_STEPS_VS_PI0 = 7.5  # KI Fig. 6b: pi0 (flow only) needs 7.5x as many steps to reach similar performance


# ======================================================================================
# 1. Configurations. The three RECAP stages differ only in where the indicator comes from (Sec. V-D, Algorithm 1).
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class Pi06TrainConfig:
    stage: str  # "pretrain" | "sft" | "recap"
    indicator: str  # "value" (I_t = 1[A > eps] from the value function) | "true" (fixed True)
    advantage_dropout: float  # App. F: 30%
    positive_fraction: float | None  # threshold rule for this stage (None: no threshold)
    init_from: str  # which checkpoint the stage starts from
    alpha: float = ALPHA
    insulate: bool = True  # KI stop-gradient
    optimizer: TrainConfig | None = None  # undisclosed (README Sec. 8)
    note: str = ""


def pretrain_config() -> Pi06TrainConfig:
    return Pi06TrainConfig("pretrain", "value", ADVANTAGE_DROPOUT, POSITIVE_FRACTION_PRETRAIN, "Gemma 3 4B + fresh expert",
                           note="Algorithm 1 l. 2: offline RL pre-training on D_demo with V_pre's indicators (Sec. V-D); optimizer undisclosed")


def sft_config() -> Pi06TrainConfig:
    return Pi06TrainConfig("sft", "true", ADVANTAGE_DROPOUT, None, "pi_pre", note="Algorithm 1 l. 5 / Sec. V-D: I_t fixed True on the task's demonstrations = SFT")


def recap_config(positive_fraction: float = POSITIVE_FRACTION_FINETUNE) -> Pi06TrainConfig:
    return Pi06TrainConfig("recap", "value", ADVANTAGE_DROPOUT, positive_fraction, "pi_pre",
                           note="Algorithm 1 l. 9 / Sec. V-D: from pi_pre on all of D_ell with V_ell^k's indicators (40% positive; 10% T-shirt)")


def tiny_config() -> TrainConfig:
    return TrainConfig(warmup_steps=2, peak_lr=1e-3, decay_steps=10, decay_lr=1e-4, batch_size=2, num_train_steps=2)  # tiny only, not paper values


# ======================================================================================
# 2. The KI joint forward. One pass over [images | text | expert tokens] with ../backbone's mask (images bidirectional,
#    text causal, expert reads PREFIX + SUBTASK + ADVANTAGE and itself, never FAST; nobody reads the expert) and the
#    stop-gradient inside attention (insulate=True). The text logits are computed only where a loss applies
#    (Subtask + FAST tokens, KI M^ell), the expert's velocity only matters where has_actions (M^act).
# ======================================================================================
def ki_forward(model: Pi06, obs: Pi06Observation, x_t: torch.Tensor | None, t: torch.Tensor | None, *, insulate: bool = True):
    """-> (logits f32[N, V] at the N positions whose NEXT token carries a CE loss, targets i64[N], sample_index i64[N],
    v_t f32[B, H, 32] or None). x_t=None: text branch only (a VLM-data batch)."""
    emb, valid, n_img = model.embed_prefix(obs)
    b, L = obs.tokens.shape
    pos0 = valid.long().cumsum(1) - 1
    if x_t is None:
        outs, _ = model.llm([emb, None], pos0, model.prefix_mask(obs, n_img), None)
    else:
        tokens, _m, _ar, cond = model.proj.embed_suffix(x_t, t)
        e = tokens.shape[1]
        mask = model.prefix_mask(obs, n_img, n_expert=e, expert_visible=obs.expert_visible)
        # expert positions continue after the tokens it can see (as at inference, where no FAST tokens exist)
        n_seen = valid[:, :n_img].long().sum(1, keepdim=True) + obs.expert_visible.long().sum(1, keepdim=True)
        pos1 = n_seen + torch.arange(e, device=emb.device)[None]
        outs, _ = model.llm([emb, tokens], torch.cat([pos0, pos1], 1), mask, None, cond, insulate=insulate)
    h = outs[0][:, n_img : n_img + L - 1]  # position i predicts token i + 1
    lm = obs.loss_mask[:, 1:]
    logits = model.logits_head(h[lm])  # [N, V]: only the loss positions go through the 262,144-way head
    targets = obs.tokens[:, 1:][lm]
    sample_index = torch.arange(b, device=emb.device)[:, None].expand(-1, L - 1)[lm]
    v_t = None if x_t is None else model.proj.decode(outs[1])
    return logits, targets, sample_index, v_t


def per_sample_ce(logits: torch.Tensor, targets: torch.Tensor, sample_index: torch.Tensor, b: int) -> torch.Tensor:
    """CE per sample normalised by its number of loss tokens (pi.fast.train.cross_entropy's rule) -> f32[B]; 0 for a sample without loss tokens."""
    if logits.shape[0] == 0:
        return torch.zeros(b, device=logits.device)
    tok = -torch.log_softmax(logits.float(), -1).gather(-1, targets[:, None])[:, 0]
    ce = torch.zeros(b, device=logits.device).index_add(0, sample_index, tok)
    cnt = torch.zeros(b, device=logits.device).index_add(0, sample_index, torch.ones_like(tok))
    return ce / cnt.clamp(min=1.0)


def ki_loss(model: Pi06, obs: Pi06Observation, actions: torch.Tensor | None, *, alpha: float = ALPHA, t: torch.Tensor | None = None,
            noise: torch.Tensor | None = None, insulate: bool = True) -> dict:
    """KI Eq. 4 / paper Eq. 4: mean_b CE_b + alpha * mean_{b: has_actions} MSE_b. MSE per sample = mean over H x 32 of
    (v_t - u_t)^2 with u_t = noise - actions (pi0 convention); alpha_eta of App. C is a constant here (README Sec. 8)."""
    b = obs.tokens.shape[0]
    if actions is None or not bool(obs.has_actions.any()):
        logits, targets, idx, _ = ki_forward(model, obs, None, None, insulate=insulate)
        ce = per_sample_ce(logits, targets, idx, b)
        return {"loss": ce.mean(), "ce": ce, "mse": torch.zeros(b, device=ce.device)}
    x_t, u_t = interpolate(actions, noise, t)
    logits, targets, idx, v_t = ki_forward(model, obs, x_t, t, insulate=insulate)
    ce = per_sample_ce(logits, targets, idx, b)
    mse = (v_t - u_t).pow(2).mean(dim=(1, 2))
    m = obs.has_actions.to(mse.dtype)
    return {"loss": ce.mean() + alpha * (mse * m).sum() / m.sum().clamp(min=1.0), "ce": ce, "mse": mse}


def backbone_parameter_names(model: Pi06) -> list[str]:
    """Everything the KI stop-gradient protects: vision, embedder, experts[0] of every layer, final_norms[0]."""
    return [n for n, _ in model.named_parameters() if n.startswith(("vision.", "embedder.")) or ".experts.0." in n or n.startswith("llm.final_norms.0")]


def expert_parameter_names(model: Pi06) -> list[str]:
    return [n for n, _ in model.named_parameters() if ".experts.1." in n or n.startswith("llm.final_norms.1") or n.startswith("proj.")]


def grad_norms_by_part(model: Pi06) -> dict[str, float]:
    bb, ex = set(backbone_parameter_names(model)), set(expert_parameter_names(model))
    out = {"backbone": 0.0, "expert": 0.0}
    for n, p in model.named_parameters():
        if p.grad is not None:
            out["backbone" if n in bb else "expert" if n in ex else "other"] = out.get("backbone" if n in bb else "expert" if n in ex else "other", 0.0) + float(p.grad.pow(2).sum())
    return {k: v**0.5 for k, v in out.items()}


# ======================================================================================
# 3. One policy training step.
# ======================================================================================
def train_step(model: Pi06, batch, params, optimizer, ema: EMA | None, cfg: TrainConfig, step: int, *, alpha: float = ALPHA, insulate: bool = True,
               generator: torch.Generator | None = None) -> dict[str, float]:
    """batch = (Pi06Observation "joint" / "text", actions f32[B, H, 32] or None). Timestep ~ Beta(1.5, 1) (KI App. B = pi0)."""
    obs, actions = batch
    model.train()
    t = noise = None
    if actions is not None:
        t = sample_timestep(actions.shape[0], generator, device=actions.device)
        noise = torch.randn(actions.shape, generator=generator, device=actions.device)
    out = ki_loss(model, obs, actions, alpha=alpha, t=t, noise=noise, insulate=insulate)
    out["loss"].backward()
    grad_norm = clip_and_step(params, optimizer, cfg, step)
    if ema is not None:
        ema.update(model)
    return {"loss": float(out["loss"].detach()), "ce": float(out["ce"].mean().detach()), "mse": float(out["mse"].mean().detach()), "grad_norm": grad_norm, "lr": lr_at(step, cfg)}


# ======================================================================================
# 4. Indicators for a batch of episodes, per stage (Sec. V-D, App. F): "true" = SFT; "value" = threshold over the pool of
#    the stage's advantages (30% pre-training / 40% RECAP), corrections forced True, then the 30% dropout.
# ======================================================================================
@dataclasses.dataclass
class Episode:
    """One episode as RECAP data: per-step "value" batches (for V), the labels, and whatever ../data needs to rebuild the
    policy sequence (raw inputs, subtasks, actions) - kept opaque here as `steps`."""

    labels: object  # pi.pi06.data.EpisodeLabels
    value_batches: list  # list of "value" layout Pi06Observation, one row per step
    norm_rewards: np.ndarray
    norm_returns: np.ndarray
    steps: list  # per-step (raw, subtask, actions) for build_pi06_batch(layout="joint")


def stage_indicators(vf: ValueFunction | None, episodes: list[Episode], cfg: Pi06TrainConfig, vcfg: ValueTrainConfig | None, rng: np.random.Generator) -> list[np.ndarray]:
    """-> per episode, an object array of True / False / None (None = dropped) ready for build_pi06_batch(advantages=...)."""
    if cfg.indicator == "true":
        return [np.array([drop_indicator(True, rng, cfg.advantage_dropout) for _ in range(e.labels.num_steps)], dtype=object) for e in episodes]
    assert vf is not None and vcfg is not None and cfg.positive_fraction is not None
    labelled = [label_episode(vf, e.value_batches, e.norm_rewards, e.norm_returns, cfg=vcfg, threshold=0.0) for e in episodes]
    pool = np.concatenate([l["advantages"] for l in labelled])
    eps = improvement_threshold(pool, cfg.positive_fraction, rng)
    out = []
    for e, l in zip(episodes, labelled):
        ind = l["advantages"] > eps
        if e.labels.is_correction is not None:
            ind = ind | e.labels.is_correction
        out.append(np.array([drop_indicator(bool(i), rng, cfg.advantage_dropout) for i in ind], dtype=object))
    return out


# ======================================================================================
# 5. Algorithm 1. The three subroutines are injected so the loop stays readable and the tiny run can use stand-ins:
#    collect(policy_state) -> list[Episode]; fit_value(v_pre_state, dataset) -> vf; fit_policy(pi_pre_state, dataset, vf, cfg) -> model.
#    Both refits start from the PRE-TRAINED checkpoints every iteration (Sec. V-D), never from the previous iteration.
# ======================================================================================
def recap(pi_pre: Pi06, v_pre: ValueFunction, demos: list[Episode], *, collect, fit_value, fit_policy, iterations: int, log=print) -> dict:
    """Algorithm 1 lines 3-10 for one task ell. Returns the final policy, value function and the aggregated dataset."""
    pi_pre_state, v_pre_state = copy.deepcopy(pi_pre.state_dict()), copy.deepcopy(v_pre.state_dict())
    dataset = list(demos)  # l. 3: D_ell <- demonstrations
    vf = fit_value(v_pre_state, dataset)  # l. 4: V_ell^0 from V_pre
    policy = fit_policy(pi_pre_state, dataset, vf, sft_config())  # l. 5: pi_ell^0 from pi_pre, I = True (Sec. V-D: SFT)
    log(f"[recap] iteration 0: {len(dataset)} demonstration episodes -> V_ell^0 (from V_pre), pi_ell^0 (from pi_pre, SFT)")
    history = [{"iteration": 0, "episodes": len(dataset)}]
    for k in range(1, iterations + 1):  # l. 6-10
        new = collect(policy)  # l. 7: autonomous rollouts (+ optional expert corrections), human outcome labels
        dataset = dataset + list(new)
        vf = fit_value(v_pre_state, dataset)  # l. 8: from V_pre on ALL of D_ell
        policy = fit_policy(pi_pre_state, dataset, vf, recap_config())  # l. 9: from pi_pre with V_ell^k's indicators
        history.append({"iteration": k, "episodes": len(dataset), "new": len(new), "successes": int(sum(e.labels.success for e in new))})
        log(f"[recap] iteration {k}: +{len(new)} episodes ({history[-1]['successes']} successes) -> {len(dataset)} total; V and pi refit from the pre-trained checkpoints")
    return {"policy": policy, "value": vf, "dataset": dataset, "history": history}


# ======================================================================================
# 6. Facts: baselines, results, cost (Sec. VI-B, VI-C, App. D; KI Sec. 7). Stated, not reproduced.
# ======================================================================================
BASELINES = (
    ("pre-trained pi0.5", "no RL, no RECAP (Sec. VI-B)"),
    ("pre-trained pi0.6", "supervised pre-training, no advantage indicator"),
    ("RL pre-trained pi0.6*", "offline-RL pre-training with V_pre and I_t (Sec. V-D)"),
    ("pi0.6* offline RL + SFT", "the RL pre-trained model finetuned on the task's demonstrations with I = True"),
    ("AWR", "advantage-weighted regression on the same data (Sec. VI-C.3): reasonable success, much slower policies"),
    ("PPO (SPO trust region)", "App. D Eq. 10-11: single-step diffusion likelihood bound, separate AR / flow ratios, trust region eta = 0.01; stable but weak"),
)
RESULTS = (
    ("throughput", "more than doubles on diverse laundry and espresso from on-robot data (offline RL + SFT -> final); T-shirt task +50% over two iterations (Fig. 7, 9)"),
    ("failure rate", "reduced by about 2x (Sec. VI-C.1)"),
    ("success rate", "90%+ on every task except diverse laundry; laundry > 90% after the first iteration (Fig. 8, 10)"),
    ("box assembly", "about 90% on folding and labelling within 600 s after two iterations; 2x throughput after the second iteration (Fig. 8-10)"),
    ("failure-mode removal", "strict T-shirt task: 97% after two iterations of 600 autonomous trajectories, no interventions (Fig. 12)"),
    ("long runs", "espresso 13 hours straight; laundry in a new home over two hours (Sec. I)"),
)
COST = (
    ("KI training overhead", f"about {KI_EXTRA_COMPUTE:.0%} more compute per step than flow-only; pi0 needs {KI_STEPS_VS_PI0}x the steps (KI Sec. 7, Fig. 6b)"),
    ("pre-training data", "tens of thousands of hours of demonstrations, many robots (Sec. IV); composition follows pi0.5 + more platforms (card Sec. 3)"),
    ("per-task experience", "see ../infer EPISODES_PER_ITERATION (Sec. VI-C.2, App. F)"),
    ("optimizer / lr / batch / steps / EMA / GPUs / hours", "undisclosed (README Sec. 8)"),
)


# ======================================================================================
# 7. One KI step, the stop-gradient check, and a two-iteration RECAP loop with stand-in subroutines, all tiny.
#    uv run python -m pi.pi06.train.train
# ======================================================================================
def main():
    from pi.pi0.data.data import make_bool_mask
    from pi.pi06.backbone.model import tiny_pi06
    from pi.pi06.data.data import C_FAIL_TINY, STATIC_IMAGE_KEYS, EpisodeLabels, build_pi06_batch, episode_rewards, tiny_pi06_tokenizer, unit_stats
    from pi.pi06.value.model import tiny_value_function
    from pi.pi06.value.train import train_step as value_train_step

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    H, d, B = 10, 7, 2
    model, vf, seq = tiny_pi06(), tiny_value_function(), tiny_pi06_tokenizer(H, d)
    stats, dmask = unit_stats(d), make_bool_mask(6, -1)

    def raw_batch(b, prompts):
        return {"images": {k: rng.integers(0, 256, (b, 96, 128, 3), dtype=np.uint8) for k in ("base_0_rgb", "left_wrist_0_rgb")},
                "state": rng.uniform(-0.5, 0.5, (b, d)).astype(np.float32), "actions": rng.uniform(-0.5, 0.5, (b, H + 2, d)).astype(np.float32), "prompt": prompts}

    # --- one KI batch: a joint sample with subtask + Advantage: positive, and a joint sample with the indicator dropped ---
    obs, actions = build_pi06_batch(raw_batch(B, ["make me an espresso", "fold the shirt"]), stats, seq, layout="joint", image_keys=STATIC_IMAGE_KEYS, action_horizon=H,
                                    delta_mask=dmask, train=True, subtasks=["grab the portafilter", None], advantages=[True, None])
    t = sample_timestep(B)
    noise = torch.randn(actions.shape)
    print(f"[batch]  joint layout: sample 0 = prefix + Subtask + Advantage + FAST ({int(obs.loss_mask[0].sum())} CE tokens), sample 1 = prefix + FAST ({int(obs.loss_mask[1].sum())} CE tokens); "
          f"expert reads {obs.expert_visible[0].sum().item()} / {obs.expert_visible[1].sum().item()} text columns; has_actions {obs.has_actions.tolist()}")
    out = ki_loss(model, obs, actions, alpha=ALPHA, t=t, noise=noise, insulate=True)
    print(f"[loss]   CE {[f'{x:.2f}' for x in out['ce'].tolist()]} (per sample, per token) + alpha {ALPHA:.0f} x MSE {[f'{x:.3f}' for x in out['mse'].tolist()]} = {float(out['loss']):.3f}   (KI Eq. 4, alpha = 1 because of the stop-gradient)")
    # --- the stop-gradient: MSE alone must not move the backbone. At init the zero-initialised adaRMSNorm gates already cut
    #     the expert off from the prefix (../backbone), so perturb them first; otherwise both settings would read 0.
    for m in model.modules():
        if isinstance(m, AdaRMSNorm):
            torch.nn.init.normal_(m.modulation.weight, std=0.05)
    for ins in (True, False):
        model.zero_grad(set_to_none=True)
        o = ki_loss(model, obs, actions, alpha=ALPHA, t=t, noise=noise, insulate=ins)
        (o["mse"].mean()).backward()
        g = grad_norms_by_part(model)
        print(f"[sg]     insulate={ins!s:5s}: grad norm from the MSE term alone -> backbone {g['backbone']:.4f}, expert {g['expert']:.4f}   ({'KI Eq. 5-6: expert queries see sg(K_b), sg(V_b)' if ins else 'joint-training baseline of KI Fig. 4'})")
    model.zero_grad(set_to_none=True)
    o = ki_loss(model, obs, actions, alpha=ALPHA, t=t, noise=noise, insulate=True)
    (o["ce"].mean()).backward()
    g = grad_norms_by_part(model)
    print(f"[sg]     CE term alone -> backbone {g['backbone']:.4f}, expert {g['expert']:.4f}   (the backbone learns actions through the FAST tokens)")
    # --- a VLM-data batch: text target only, no expert pass ---
    tobs, _ = build_pi06_batch(raw_batch(B, ["caption the image"] * B), stats, seq, layout="text", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=dmask, train=True,
                               target_text=["a dog catches a frisbee", "a cup on a table"])
    o = ki_loss(model, tobs, None)
    print(f"[vlm]    text layout: CE {[f'{x:.2f}' for x in o['ce'].tolist()]}, MSE {o['mse'].tolist()} (M^act = 0: no expert pass)")
    # --- two optimizer steps ---
    cfg = tiny_config()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, cfg)
    for step in range(cfg.num_train_steps):
        m = train_step(model, (obs, actions), params, opt, None, cfg, step)
        print(f"[step {step}] loss {m['loss']:.3f} = CE {m['ce']:.3f} + MSE {m['mse']:.3f}; grad norm {m['grad_norm']:.2f}; lr {m['lr']:.1e} (tiny only)")

    # --- Algorithm 1 with stand-in subroutines (tiny counts) ---
    def make_episode(success: bool, n_steps: int, corrections: bool = False) -> Episode:
        lab = EpisodeLabels("fold the shirt", success, 40, n_steps, np.array([False] * (n_steps - 3) + [True] * 3) if corrections else None)
        R, bins = lab.value_targets(C_FAIL_TINY)
        r = episode_rewards(n_steps, success, C_FAIL_TINY) / 40
        vb, steps = [], []
        for k in range(n_steps):
            rb = raw_batch(1, ["fold the shirt"])
            vb.append(build_pi06_batch(rb, stats, seq, layout="value", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False, value_bins=[int(bins[k])])[0])
            steps.append(rb)
        return Episode(lab, vb, r, R, steps)

    def collect(policy):
        return [make_episode(bool(rng.random() < 0.6), int(rng.integers(4, 7)), corrections=bool(rng.random() < 0.5)) for _ in range(3)]

    def fit_value(v_pre_state, dataset):
        v = tiny_value_function()
        v.load_state_dict(v_pre_state)
        p = [q for q in v.parameters() if q.requires_grad]
        o = make_optimizer(p, cfg)
        for i, e in enumerate(dataset[:4]):
            value_train_step(v, e.value_batches[0], p, o, None, cfg, i)
        return v.eval()

    def fit_policy(pi_pre_state, dataset, vf_k, stage_cfg):
        pi = tiny_pi06()
        pi.load_state_dict(pi_pre_state)
        vcfg = value_pretrain_config() if stage_cfg.stage == "pretrain" else value_finetune_config(stage_cfg.positive_fraction or POSITIVE_FRACTION_FINETUNE)
        inds = stage_indicators(vf_k if stage_cfg.indicator == "value" else None, dataset, stage_cfg, vcfg, rng)
        n_true = sum(int(sum(1 for x in i if x is True)) for i in inds)
        n_drop = sum(int(sum(1 for x in i if x is None)) for i in inds)
        n_all = sum(len(i) for i in inds)
        p = [q for q in pi.parameters() if q.requires_grad]
        o = make_optimizer(p, cfg)
        e = dataset[0]
        jb, ja = build_pi06_batch(e.steps[0], stats, seq, layout="joint", image_keys=STATIC_IMAGE_KEYS, action_horizon=H, delta_mask=dmask, train=True, advantages=[inds[0][0]])
        train_step(pi, (jb, ja), p, o, None, cfg, 0)
        print(f"         fit_policy[{stage_cfg.stage}]: indicators over {n_all} steps: {n_true} True, {n_all - n_true - n_drop} False, {n_drop} dropped ({stage_cfg.advantage_dropout:.0%}); from {stage_cfg.init_from}")
        return pi.eval()

    demos = [make_episode(True, 5) for _ in range(2)]
    res = recap(model, vf, demos, collect=collect, fit_value=fit_value, fit_policy=fit_policy, iterations=2)
    print(f"[recap]  history {res['history']}")
    print("\nbaselines (Sec. VI-B):", "; ".join(f"{n}: {d}" for n, d in BASELINES[:2]), "...")
    print("results (Sec. VI-C):", RESULTS[0][1])


if __name__ == "__main__":
    main()
