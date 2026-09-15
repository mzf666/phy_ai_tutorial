"""pi0-FAST training: next-token cross entropy on the action tokens only (shift by one, logits only at target
positions, per-sample normalization), the paper's warmup-then-constant schedule, and the per-head LoRA variant with
its freeze rule.

Re-implementation (PyTorch). Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/models/pi0_fast.py: compute_loss L197-L233, get_freeze_filter L127-L131
           src/openpi/models/gemma_fast.py: gemma_2b_lora L53-L72; src/openpi/models/lora.py L11-L30, L43-L65, L96-L150
           src/openpi/training/config.py: pi0_fast_libero L699-L720, _low_mem_finetune L721-L742,
           pi0_fast_full_droid_finetune L831-L860, TrainConfig defaults L488-L511; optimizer.py L15-L31, L65-L85
  paper    FAST, arXiv:2501.09747v1, Sec. VI-F (5x fewer GPU hours), Fig. 9, Appendix C (optimizer), D (DROID), E (LIBERO)
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Optimizer, schedule, clipping, EMA, parameter-norm logging are pi.pi0.train.train (identical upstream code path);
the model is ../model Pi0FAST. This file only holds the increment: the loss, LoRA, the freeze rule, one train step.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi.fast.data.data import FASTObservation
from pi.fast.model.model import Pi0FAST
from pi.pi0.train.train import EMA, TrainConfig, clip_and_step, kernel_param_norm, lr_at, make_optimizer
from pi.pi0.vlm.model import ExpertAttnProj, GeGLU, GemmaConfig, make_attn_mask

LORA_RANK = 16  # gemma_fast.py L68-L69
LORA_ALPHA = 16.0  # gemma_fast.py L68-L69; scaling alpha / rank = 1.0 (lora.py L30, rslora False)
LORA_INIT_STD = 0.01  # lora.py L20: BOTH A and B ~ N(0, 0.01^2); no zero init of B (README Sec. 8)


# ======================================================================================
# 1. Configs. paper_config = openpi pi0_fast_full_droid_finetune (the one config whose schedule matches paper App. C);
#    libero_config = openpi pi0_fast_libero (global defaults). Paper values that differ are noted per field.
# ======================================================================================
def paper_config() -> TrainConfig:
    return TrainConfig(
        warmup_steps=1_000,  # config.py L849; paper App. C "1k steps"
        peak_lr=5e-5,  # L850; paper 5e-5
        decay_steps=1_000_000,  # L851: nominal, never reached in 100k steps
        decay_lr=5e-5,  # L852 == peak: cosine with equal ends = constant after warmup; paper "constant"
        b1=0.9, b2=0.95, eps=1e-8,  # optimizer.py L69-L71; paper (0.9, 0.95)
        weight_decay=1e-10,  # optimizer.py L73 (openpi default); paper "without weight decay" -> README Sec. 8
        clip_gradient_norm=1.0,  # optimizer.py L74; paper 1
        ema_decay=0.99,  # config.py L490 (openpi default, not overridden); paper 0.999 -> README Sec. 8
        batch_size=256,  # L855; paper DROID 256
        num_train_steps=100_000,  # L854 (fine-tune from pi0_fast_base); paper DROID 240k from scratch -> Sec. 8
    )


def libero_config() -> TrainConfig:
    """pi0_fast_libero (config.py L699-L720): everything at the openpi global default, i.e. the pi0 fine-tuning
    schedule 2.5e-5 cosine -> 2.5e-6 over 30k steps, batch 32. The paper's LIBERO run is 40k steps (App. E)."""
    return TrainConfig(num_train_steps=30_000)  # L719


def lora_config() -> TrainConfig:
    """pi0_fast_libero_low_mem_finetune (config.py L721-L742): libero_config with EMA off (L741)."""
    return dataclasses.replace(libero_config(), ema_decay=None)


# ======================================================================================
# 2. Loss. pi0_fast.py L205-L233.
# ======================================================================================
def target_logits(model: Pi0FAST, obs: FASTObservation):
    """-> (logits f32[B, L-1, vocab], targets i64[B, L-1], loss_mask bool[B, L-1]); L = max_token_len.
    Inputs drop the last token (L215-L218); logits are formed only for the last L-1 positions, the ones that predict
    the shifted token sequence (L222-L226). Left-aligned, no cache: the training forward."""
    emb, input_mask, ar_mask = model.embed_inputs(obs)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = input_mask.long().cumsum(1) - 1
    pre_logits, _ = model.llm(emb[:, :-1], positions[:, :-1], attn_mask[:, :-1, :-1])
    targets = obs.tokenized_prompt[:, 1:]  # L210-L213 (one-hot upstream; ids here)
    logits = model.logits_head(pre_logits[:, -targets.shape[1] :])  # L224-L226
    return logits, targets, obs.token_loss_mask[:, 1:]  # L231


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample: -sum_t mask_t * log_softmax(logits_t)[target_t] / max(sum_t mask_t, 1). L227-L233. -> f32[B]"""
    logp = F.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets[..., None])[..., 0]  # L232 (sum(targets * logp) with one-hot targets)
    m = loss_mask.to(token_logp.dtype)
    return -(token_logp * m).sum(-1) / m.sum(-1).clamp(min=1.0)


# ======================================================================================
# 3. LoRA on the last two axes of every attention / FFN weight, per head. lora.py L43-L65, L96-L150.
# ======================================================================================
def _lora(shape: tuple[int, ...]) -> nn.Parameter:
    return nn.Parameter(torch.randn(shape) * LORA_INIT_STD)


class LoRAAttnProj(ExpertAttnProj):
    """q [N, D, H] -> A [N, D, r], B [N, r, H]; kv [2, K, D, H] -> [2, K, D, r], [2, K, r, H];
    out [N, H, D] -> [N, H, r], [N, r, D]. gemma_fast.py L137-L163 with lora.py Einsum."""

    def __init__(self, cfg: GemmaConfig, rank: int = LORA_RANK, alpha: float = LORA_ALPHA):
        super().__init__(cfg)
        n, k, d, h = cfg.num_heads, cfg.num_kv_heads, cfg.width, cfg.head_dim
        self.scaling = alpha / rank  # lora.py L30
        self.q_lora_a, self.q_lora_b = _lora((n, d, rank)), _lora((n, rank, h))
        self.kv_lora_a, self.kv_lora_b = _lora((2, k, d, rank)), _lora((2, k, rank, h))
        self.attn_vec_lora_a, self.attn_vec_lora_b = _lora((n, h, rank)), _lora((n, rank, d))

    def qkv(self, x):
        q, k, v = super().qkv(x)
        q = q + torch.einsum("btnr,nrh->btnh", torch.einsum("btd,ndr->btnr", x, self.q_lora_a), self.q_lora_b) * self.scaling
        kv = torch.einsum("btckr,ckrh->btckh", torch.einsum("btd,ckdr->btckr", x, self.kv_lora_a), self.kv_lora_b) * self.scaling
        return q, k + kv[:, :, 0], v + kv[:, :, 1]

    def out(self, encoded):
        lora = torch.einsum("btnr,nrd->btd", torch.einsum("btnh,nhr->btnr", encoded, self.attn_vec_lora_a), self.attn_vec_lora_b)
        return super().out(encoded) + lora * self.scaling


class LoRAGeGLU(GeGLU):
    """gating [2, D, F] -> A [2, D, r], B [2, r, F]; linear [F, D] -> A [F, r], B [r, D]. lora.py L109-L150."""

    def __init__(self, cfg: GemmaConfig, rank: int = LORA_RANK, alpha: float = LORA_ALPHA):
        super().__init__(cfg)
        self.scaling = alpha / rank
        self.gating_lora_a, self.gating_lora_b = _lora((2, cfg.width, rank)), _lora((2, rank, cfg.mlp_dim))
        self.linear_lora_a, self.linear_lora_b = _lora((cfg.mlp_dim, rank)), _lora((rank, cfg.width))

    def forward(self, x):
        s = self.scaling
        gate = x @ self.gating_einsum[0] + (x @ self.gating_lora_a[0]) @ self.gating_lora_b[0] * s
        up = x @ self.gating_einsum[1] + (x @ self.gating_lora_a[1]) @ self.gating_lora_b[1] * s
        act = F.gelu(gate, approximate="tanh") * up
        return act @ self.linear + (act @ self.linear_lora_a) @ self.linear_lora_b * s


def apply_lora(model: Pi0FAST, rank: int = LORA_RANK, alpha: float = LORA_ALPHA) -> Pi0FAST:
    """Swap every GemmaBlock's attn / mlp for the LoRA subclass, keeping the base weights (gemma_2b_lora)."""
    for layer in model.llm.layers:
        attn, mlp = LoRAAttnProj(model.cfg, rank, alpha), LoRAGeGLU(model.cfg, rank, alpha)
        attn.load_state_dict(layer.attn.state_dict(), strict=False)
        mlp.load_state_dict(layer.mlp.state_dict(), strict=False)
        layer.attn, layer.mlp = attn, mlp
    return model


def lora_param_count(cfg: GemmaConfig, rank: int = LORA_RANK) -> int:
    """Closed form of README Sec. 1.3 (per layer x depth)."""
    n, k, d, h, f = cfg.num_heads, cfg.num_kv_heads, cfg.width, cfg.head_dim, cfg.mlp_dim
    per_layer = n * (d + h) * rank + 2 * k * (d + h) * rank + n * (h + d) * rank + 2 * (d + f) * rank + (f + d) * rank
    return per_layer * cfg.depth


# ======================================================================================
# 4. Freeze rule. pi0_fast.py L127-L131: LoRA variant freezes All(PathRegex(".*llm.*"), Not(PathRegex(".*lora.*")));
#    otherwise nnx.Nothing (config.py L493). Note the regex only matches `llm`: SigLIP stays trainable under LoRA.
# ======================================================================================
def select_trainable(model: Pi0FAST, mode: str = "full") -> list[nn.Parameter]:
    assert mode in ("full", "lora")
    for name, p in model.named_parameters():
        frozen = mode == "lora" and name.startswith("llm.") and "lora" not in name
        p.requires_grad_(not frozen)
    return [p for p in model.parameters() if p.requires_grad]


# ======================================================================================
# 5. One training step and a held-out loss. Same structure as pi.pi0.train.train_step.
# ======================================================================================
def train_step(model: Pi0FAST, obs: FASTObservation, params, optimizer, ema: EMA | None, cfg: TrainConfig, step: int) -> dict[str, float]:
    model.train()
    logits, targets, loss_mask = target_logits(model, obs)
    loss = cross_entropy(logits, targets, loss_mask).mean()
    loss.backward()
    grad_norm = clip_and_step(params, optimizer, cfg, step)
    if ema is not None:
        ema.update(model)
    return {"loss": float(loss.detach()), "grad_norm": grad_norm, "param_norm": kernel_param_norm(model), "lr": lr_at(step, cfg),
            "loss_tokens": float(loss_mask.sum())}


@torch.no_grad()
def evaluate_loss(model: Pi0FAST, batches) -> float:
    """Held-out CE with the same masking (this repo's addition; openpi has no validation loop)."""
    model.eval()
    return float(torch.stack([cross_entropy(*target_logits(model, obs)).mean() for obs in batches]).mean())


# ======================================================================================
# 6. A few tiny steps end to end.  uv run python -m pi.fast.train.train
# ======================================================================================
def main() -> None:
    import math

    import numpy as np

    from pi.fast.data.data import ACTION_DIM, PALIGEMMA_VOCAB_SIZE, ByteTextCodec, FASTSequenceTokenizer, build_fast_batch, tiny_fast_tokenizer
    from pi.fast.model.model import tiny
    from pi.fast.tokenizer.tokenizer import QuantileStats, make_smooth_chunks
    from pi.pi0.data.data import make_bool_mask

    torch.manual_seed(0)
    B, H, d = 2, 10, 7  # LIBERO-like (config.py L711)
    model = Pi0FAST(*tiny())
    cfg = dataclasses.replace(paper_config(), warmup_steps=2, batch_size=B, num_train_steps=5)  # schedule squeezed for the demo
    print(f"paper config (pi0_fast_full_droid_finetune): {paper_config()}")
    print("lr schedule, squeezed warmup 2:", " ".join(f"{s}:{lr_at(s, cfg):.1e}" for s in range(6)), " (constant after warmup: peak == decay_lr)")

    stats = {"state": QuantileStats(np.full(d, -1.0), np.full(d, 1.0)), "actions": QuantileStats(np.full(d, -1.0), np.full(d, 1.0))}
    seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, d), max_len=180)

    def batch(seed):
        r = np.random.default_rng(seed)
        raw = {"images": {"base_0_rgb": r.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
               "state": r.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
               "actions": (0.3 * make_smooth_chunks(B, H + 2, d, r)).astype(np.float32),
               "prompt": ["pick up the red block", "close the drawer"]}
        obs, _ = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=True,
                                  generator=torch.Generator().manual_seed(seed))
        return obs

    obs = batch(0)
    n_real, n_loss = obs.tokenized_prompt_mask.sum(1).tolist(), obs.token_loss_mask.sum(1).tolist()
    print(f"\n[input]  tokenized_prompt i64 {tuple(obs.tokenized_prompt.shape)}  real tokens {n_real}  loss (postfix) tokens {n_loss}")
    with torch.no_grad():
        emb, input_mask, ar_mask = model.embed_inputs(obs)
        print(f"[embed]  emb {tuple(emb.shape)} -> inputs emb[:, :-1] {tuple(emb[:, :-1].shape)}, mask[:, :-1, :-1]")
        logits, targets, loss_mask = target_logits(model, obs)
        print(f"[logits] {tuple(logits.shape)} = [B, max_len-1, vocab]: head applied to the last {logits.shape[1]} of {emb.shape[1]-1} positions only")
        print(f"[target] tokens[:, 1:] {tuple(targets.shape)}  loss_mask[:, 1:] {tuple(loss_mask.shape)}  loss tokens per sample {loss_mask.sum(1).tolist()}")
        per_sample = cross_entropy(logits, targets, loss_mask)
        print(f"[loss]   per sample {[f'{v:.3f}' for v in per_sample.tolist()]}  (random weights, ln vocab = {math.log(PALIGEMMA_VOCAB_SIZE):.3f})  mean {per_sample.mean():.3f}")

    params = select_trainable(model, "full")
    opt = make_optimizer(params, cfg)
    ema = EMA(model, cfg.ema_decay)
    held_out = [batch(100)]
    print(f"\n[full fine-tune] trainable {sum(p.numel() for p in params):,} of {sum(p.numel() for p in model.parameters()):,}; held-out loss before {evaluate_loss(model, held_out):.3f}")
    for step in range(cfg.num_train_steps):
        info = train_step(model, batch(step), params, opt, ema, cfg, step)
        print(f"  step {step}: lr {info['lr']:.1e}  loss {info['loss']:.3f}  grad_norm {info['grad_norm']:.2f}  loss_tokens {info['loss_tokens']:.0f}  param_norm {info['param_norm']:.2f}")
    print(f"  held-out loss after {evaluate_loss(model, held_out):.3f}  (tiny random model, {cfg.num_train_steps} steps: no claim beyond 'the loop runs')")

    lora = apply_lora(Pi0FAST(*tiny()))
    lp = select_trainable(lora, "lora")
    n_lora = sum(p.numel() for n, p in lora.named_parameters() if "lora" in n)
    n_img = sum(p.numel() for p in lora.img.parameters())
    print(f"\n[lora]   added {n_lora:,} params (closed form {lora_param_count(lora.cfg):,}); trainable {sum(p.numel() for p in lp):,} = lora + SigLIP {n_img:,}; "
          f"frozen {sum(p.numel() for p in lora.parameters()) - sum(p.numel() for p in lp):,} (llm base incl. embedding)")
    lcfg = dataclasses.replace(lora_config(), warmup_steps=2, batch_size=B)
    lopt = make_optimizer(lp, lcfg)
    info = train_step(lora, batch(0), lp, lopt, None, lcfg, 0)
    print(f"  one LoRA step: loss {info['loss']:.3f}  grad_norm {info['grad_norm']:.2f}  (EMA off, config.py L741)")


if __name__ == "__main__":
    main()
