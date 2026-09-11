"""pi0 training loop: warmup + cosine schedule, AdamW, global-norm clipping, EMA, parameter provenance, mixture weights.

Minimal PyTorch re-implementation of openpi's JAX training code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/training/config.py (TrainConfig L465-L556), src/openpi/training/optimizer.py,
          src/openpi/training/weight_loaders.py, scripts/train.py (init_train_state L85-L133, train_step L137-L191)
  paper   pi0 arXiv:2410.24164v1 Sec. V-A (mixture weighting n^0.43, post-training), Sec. VI-A (700k / 160k steps)
Upstream license: Apache-2.0. This file re-implements, it does not copy.

Every number below is openpi's FINE-TUNING default (e.g. pi0_libero: 30k steps from the pi0_base checkpoint).
The paper's pre-training optimizer settings are not disclosed (README gap ledger). The loss is
../flow_matching/train.py compute_loss; the model is ../infer Pi0. Sharding, bf16 casting of frozen params,
checkpointing and logging are systems engineering and are not reproduced.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from pi.pi0.flow_matching.train import compute_loss, make_train_velocity_fn, sample_timestep
from pi.pi0.infer.model import Pi0


# ======================================================================================
# 1. Hyperparameters. openpi@215abfb config.py L465-L556 (TrainConfig), optimizer.py L15-L31, L65-L85.
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    warmup_steps: int = 1_000  # optimizer.py L19
    peak_lr: float = 2.5e-5  # L20
    decay_steps: int = 30_000  # L21; total schedule length, warmup included (optax semantics)
    decay_lr: float = 2.5e-6  # L22
    b1: float = 0.9  # L69
    b2: float = 0.95  # L70
    eps: float = 1e-8  # L71
    weight_decay: float = 1e-10  # L73; applied to every trainable parameter, no mask (train.py L88)
    clip_gradient_norm: float = 1.0  # L74
    ema_decay: float | None = 0.99  # config.py L489; None for the LoRA configs (L697)
    batch_size: int = 32  # config.py L505
    num_train_steps: int = 30_000  # config.py L510


def lr_at(step: int, cfg: TrainConfig) -> float:
    """optax.warmup_cosine_decay_schedule(init=peak/(warmup+1), peak, warmup_steps, decay_steps, end=decay_lr).
    Linear warmup from peak/(warmup+1) at step 0 to peak at step warmup; cosine from peak to decay_lr between
    warmup and decay_steps; constant decay_lr afterwards. optimizer.py L24-L31."""
    init = cfg.peak_lr / (cfg.warmup_steps + 1)
    if step < cfg.warmup_steps:
        return init + (cfg.peak_lr - init) * step / cfg.warmup_steps
    n = max(cfg.decay_steps - cfg.warmup_steps, 1)
    frac = min((step - cfg.warmup_steps) / n, 1.0)
    alpha = cfg.decay_lr / cfg.peak_lr
    return cfg.peak_lr * ((1 - alpha) * 0.5 * (1 + math.cos(math.pi * frac)) + alpha)


# ======================================================================================
# 2. Which parameters come from PaliGemma and which start from scratch. weight_loaders.py L58-L73, L76-L104.
# ======================================================================================
def split_params(pi0: Pi0) -> dict[str, list[str]]:
    """PaliGemmaWeightLoader overwrites every parameter whose name exists in pt_224.npz and keeps the rest as
    initialized. In pi0 that means: img, embedder, expert 0 of every layer, final_norms[0] are loaded; expert 1
    (named with a "_1" suffix upstream, gemma.py L443-L451), final_norms[1] and the five projections are fresh."""
    loaded, fresh = [], []
    for name, _ in pi0.named_parameters():
        is_expert0 = ".experts.0." in name or name.startswith("llm.final_norms.0.")
        (loaded if name.startswith(("img.", "embedder.")) or is_expert0 else fresh).append(name)
    return {"from_paligemma": loaded, "from_scratch": fresh}


def select_trainable(pi0: Pi0, freeze: str = "nothing") -> list[torch.nn.Parameter]:
    """openpi default: nothing is frozen (config.py L492 `nnx.Nothing`), full fine-tuning of all 3.24B parameters.
    The LoRA configs freeze every non-LoRA parameter inside `llm` (pi0_config.py L88-L117); LoRA layers are not
    implemented here, so only "nothing" mirrors upstream. "llm" is offered for experiments: it freezes both
    experts (not SigLIP, not the projections), which is the non-LoRA part of that filter."""
    assert freeze in ("nothing", "llm")
    for name, p in pi0.named_parameters():
        p.requires_grad_(not (freeze == "llm" and name.startswith("llm.")))
    return [p for p in pi0.parameters() if p.requires_grad]


# ======================================================================================
# 3. Optimizer, clipping, EMA. optimizer.py L81-L85; train.py L160-L175.
# ======================================================================================
def make_optimizer(params, cfg: TrainConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(params, lr=cfg.peak_lr, betas=(cfg.b1, cfg.b2), eps=cfg.eps, weight_decay=cfg.weight_decay)


def clip_and_step(params, optimizer: torch.optim.Optimizer, cfg: TrainConfig, step: int) -> float:
    """Set this step's lr, clip the global grad norm to clip_gradient_norm, apply AdamW. Returns the pre-clip norm."""
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step, cfg)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_gradient_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


class EMA:
    """ema <- decay * ema + (1 - decay) * param, over ALL parameters (frozen ones included; they just do not move).
    train.py L112-L113 (init = a copy of params), L169-L175. Inference uses these weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def kernel_param_norm(model: torch.nn.Module) -> float:
    """train.py L177-L185: global norm of parameters with ndim > 1, excluding bias / scale / pos_embedding /
    input_embedding by name."""
    skip = ("bias", "scale", "pos_embedding", "input_embedding")
    ks = [p for n, p in model.named_parameters() if p.ndim > 1 and n.split(".")[-1] not in skip]
    return float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in ks)))


# ======================================================================================
# 4. One training step. train.py L137-L191.
# ======================================================================================
def train_step(pi0: Pi0, batch, params, optimizer, ema: EMA | None, cfg: TrainConfig, step: int,
               generator: torch.Generator | None = None) -> dict[str, float]:
    """batch = (Observation, actions f32[B, 50, 32]) from ../data build_batch(train=True).
    loss = mean over batch, horizon and action dim of the flow-matching loss (train.py L150-L151)."""
    obs, actions = batch
    pi0.train()
    b = actions.shape[0]
    t = sample_timestep(b, generator, device=actions.device)
    noise = torch.randn(actions.shape, generator=generator, device=actions.device)
    prefix_emb, prefix_mask, prefix_ar = pi0.embed_prefix(obs)
    v = make_train_velocity_fn(pi0.llm, pi0.proj, prefix_emb, prefix_mask, prefix_ar, obs.state)
    loss = compute_loss(v, actions, noise, t).mean()
    loss.backward()
    grad_norm = clip_and_step(params, optimizer, cfg, step)
    if ema is not None:
        ema.update(pi0)
    return {"loss": float(loss.detach()), "grad_norm": grad_norm, "param_norm": kernel_param_norm(pi0), "lr": lr_at(step, cfg)}


@torch.no_grad()
def evaluate_loss(pi0: Pi0, batches, seed: int = 0) -> float:
    """Held-out flow-matching loss with a fixed seed for t and noise (this repo's addition; openpi has no
    validation loop). Same metric as the training loss."""
    pi0.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for obs, actions in batches:
        t = sample_timestep(actions.shape[0], g)
        noise = torch.randn(actions.shape, generator=g)
        prefix_emb, prefix_mask, prefix_ar = pi0.embed_prefix(obs)
        v = make_train_velocity_fn(pi0.llm, pi0.proj, prefix_emb, prefix_mask, prefix_ar, obs.state)
        losses.append(compute_loss(v, actions, noise, t).mean())
    return float(torch.stack(losses).mean())


# ======================================================================================
# 5. Pre-training mixture weights. Paper Sec. V-A: each (task, robot) combination weighted by n^0.43.
#    openpi's public code trains on one dataset and has no implementation of this; formula from the paper.
# ======================================================================================
def mixture_weights(counts: dict[str, int], power: float = 0.43) -> dict[str, float]:
    """counts: samples (timesteps) per (task, robot) combination -> normalized sampling weights ∝ n^power."""
    w = {k: float(n) ** power for k, n in counts.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


# ======================================================================================
# 6. A few tiny steps end to end.  uv run python -m pi.pi0.train.train
# ======================================================================================
def main():
    from pi.pi0.data.data import ByteEncoder, NormStats, PromptTokenizer, build_batch, make_bool_mask
    from pi.pi0.infer.model import tiny_pi0

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B, d = 4, 7
    pi0 = tiny_pi0()
    cfg = TrainConfig(warmup_steps=2, decay_steps=8, batch_size=B, num_train_steps=6)  # schedule squeezed for the demo
    groups = split_params(pi0)
    n = lambda names: sum(p.numel() for nm, p in pi0.named_parameters() if nm in set(names))
    print(f"params from PaliGemma checkpoint: {n(groups['from_paligemma']):,}   from scratch: {n(groups['from_scratch']):,}")
    print(f"config (paper-size defaults except the squeezed schedule): {cfg}")
    print("lr schedule:", " ".join(f"{s}:{lr_at(s, cfg):.2e}" for s in range(0, 10)))

    tok = PromptTokenizer(ByteEncoder())
    stats = {"state": NormStats(np.zeros(d, np.float32), np.ones(d, np.float32)), "actions": NormStats(np.zeros(d, np.float32), np.ones(d, np.float32))}
    mask = make_bool_mask(6, -1)

    def batch(seed):
        r = np.random.default_rng(seed)
        raw = {"images": {"base_0_rgb": r.integers(0, 256, (B, 96, 128, 3), dtype=np.uint8)},
               "state": r.standard_normal((B, d)).astype(np.float32),
               "actions": r.standard_normal((B, 60, d)).astype(np.float32),
               "prompt": ["fold the towel"] * B}
        return build_batch(raw, stats, tok, delta_mask=mask, train=True, generator=torch.Generator().manual_seed(seed))

    params = select_trainable(pi0, "nothing")
    opt = make_optimizer(params, cfg)
    ema = EMA(pi0, cfg.ema_decay)
    held_out = [batch(100), batch(101)]
    print(f"held-out loss before: {evaluate_loss(pi0, held_out):.4f}")
    g = torch.Generator().manual_seed(0)
    for step in range(cfg.num_train_steps):
        info = train_step(pi0, batch(step), params, opt, ema, cfg, step, g)
        drift = max(float((ema.shadow[nm] - p.detach()).abs().max()) for nm, p in pi0.named_parameters())
        print(f"step {step}: lr {info['lr']:.2e}  loss {info['loss']:.4f}  grad_norm {info['grad_norm']:.3f}  param_norm {info['param_norm']:.2f}  max|ema - param| {drift:.2e}")
    print(f"held-out loss after:  {evaluate_loss(pi0, held_out):.4f}  (tiny random model, {cfg.num_train_steps} steps: no claim beyond 'the loop runs')")
    print("mixture weights, paper n^0.43 for e.g. {laundry: 1e8, bussing: 1e7, toast: 1e6}:",
          {k: f"{v:.3f}" for k, v in mixture_weights({"laundry": 10**8, "bussing": 10**7, "toast": 10**6}).items()},
          " (raw shares would be 0.901 / 0.090 / 0.009)")


if __name__ == "__main__":
    main()
