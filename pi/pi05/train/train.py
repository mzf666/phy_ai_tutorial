"""pi0.5 training: the joint objective (cross entropy on text + FAST action tokens, plus alpha times the flow-matching
MSE of the adaRMSNorm expert), the three-block attention mask that keeps the two action representations apart,
the two-stage curriculum (280k discrete pre-training, 80k post-training with a freshly initialised expert), the
disclosed fine-tuning configurations, and one training step.

Minimal PyTorch re-implementation. Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/models/pi0.py compute_loss L188-L214 (flow-only, adarms_cond), src/openpi/models/pi0_fast.py L205-L233
           (target logits + CE, via pi.fast.train), src/openpi/training/config.py L744-L763 (pi05_libero), L799-L827
           (pi05_aloha_pen_uncap), L865-L894 (pi05_full_droid_finetune), L895-L918 (pi05_droid_finetune), optimizer.py
  paper    pi0.5 arXiv:2504.16054v1 Sec. IV-B (Eq. 1, alpha), IV-C (pre-training data), IV-D (280k / 80k, alpha = 10,
           random-init expert), V-B (40k scaling runs), Appendix E (timestep sampling, Fig. 18 mask);
           Hi Robot arXiv:2502.19417v2 Appendix C.1-C.3 (high-level policy optimizer, batch, compute)
Upstream license: Apache-2.0. This file re-implements, it does not copy. Upstream has no joint objective; that part
follows the paper and every inferred detail is in README Sec. 8.

Optimizer, schedule, clipping and EMA are pi.pi0.train; the CE is pi.fast.train; timestep sampling and the
interpolation are pi.pi0.flow_matching.train. Sharding, bf16 and checkpointing are not reproduced.
"""

from __future__ import annotations

import dataclasses

import torch

from pi.fast.train.train import cross_entropy
from pi.pi0.flow_matching.train import interpolate, sample_timestep
from pi.pi0.train.train import EMA, TrainConfig, clip_and_step, lr_at, make_optimizer  # noqa: F401  (re-exported)
from pi.pi05.data.data import Pi05Observation
from pi.pi05.expert.model import AdaMoEBlock, AdaRMSNorm, Pi05ActionProjections
from pi.pi05.hier.model import Pi05

ALPHA_PRETRAIN = 0.0  # paper Sec. IV-B: "pre-train our model as a standard VLM transformer by mapping actions to text tokens (alpha = 0)"
ALPHA_POSTTRAIN = 10.0  # paper Sec. IV-D: "alpha = 10.0 for 80k additional steps"
PRETRAIN_STEPS = 280_000  # Sec. IV-D
POSTTRAIN_STEPS = 80_000  # Sec. IV-D
SCALING_STEPS = 40_000  # Sec. V-B: per-location-count post-training runs


# ======================================================================================
# 1. Configurations. The paper discloses steps and alpha only; openpi discloses its fine-tuning runs; Hi Robot its
#    high-level policy. `optimizer=None` = undisclosed (README Sec. 8).
# ======================================================================================
@dataclasses.dataclass(frozen=True)
class Pi05TrainConfig:
    stage: str  # "pretrain" | "posttrain" | "flow_only" | "hl_text"
    alpha: float | None  # weight of the flow MSE (None: not applicable)
    num_train_steps: int
    optimizer: TrainConfig | None  # None = undisclosed, see README Sec. 8
    note: str = ""


def pretrain_config() -> Pi05TrainConfig:
    return Pi05TrainConfig("pretrain", ALPHA_PRETRAIN, PRETRAIN_STEPS, None, "Sec. IV-C/D: FAST + text tokens only, no expert; optimizer undisclosed")


def posttrain_config() -> Pi05TrainConfig:
    return Pi05TrainConfig("posttrain", ALPHA_POSTTRAIN, POSTTRAIN_STEPS, None, "Sec. IV-D: CE + 10 * flow MSE, expert random-init; optimizer undisclosed")


def scaling_config() -> Pi05TrainConfig:
    return Pi05TrainConfig("posttrain", ALPHA_POSTTRAIN, SCALING_STEPS, None, "Sec. V-B: 40k post-training steps per location count")


def openpi_libero_config() -> TrainConfig:
    """config.py L744-L763: warmup 10k then constant 5e-5 (decay_lr == peak_lr), batch 256, EMA 0.999, 30k steps.
    Flow-only (the released pi05 path), action_horizon 10, discrete_state_input=False."""
    return TrainConfig(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5, clip_gradient_norm=1.0, ema_decay=0.999, batch_size=256, num_train_steps=30_000)


def openpi_droid_config() -> TrainConfig:
    """config.py L865-L894: warmup 1k then constant 5e-5, batch 256, 100k steps, EMA default 0.99 (L489), H 16, 32 dims."""
    return TrainConfig(warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5, clip_gradient_norm=1.0, ema_decay=0.99, batch_size=256, num_train_steps=100_000)


def openpi_droid_small_config() -> TrainConfig:
    """config.py L895-L918: 20k steps, batch 32, from the pi05_droid checkpoint; schedule = TrainConfig defaults (cosine 2.5e-5 -> 2.5e-6)."""
    return TrainConfig(batch_size=32, num_train_steps=20_000)


def openpi_aloha_config() -> TrainConfig:
    """config.py L799-L827: 20k steps, batch 64, from pi05_base; default schedule."""
    return TrainConfig(batch_size=64, num_train_steps=20_000)


def hirobot_hl_config() -> TrainConfig:
    """Hi Robot App. C.2: AdamW beta (0.9, 0.95), no weight decay, clip 1, EMA 0.999, warmup 1k then constant 1e-5, batch 512.
    Steps undisclosed (~2 h on 8 x H100, App. C.3); num_train_steps below is a placeholder (README Sec. 8)."""
    return TrainConfig(warmup_steps=1_000, peak_lr=1e-5, decay_steps=1_000_000, decay_lr=1e-5, b1=0.9, b2=0.95, weight_decay=0.0,
                       clip_gradient_norm=1.0, ema_decay=0.999, batch_size=512, num_train_steps=0)  # undisclosed, see README Sec. 8


# ======================================================================================
# 2. The three-block mask. Paper Appendix E / Fig. 18: prefix bidirectional; FAST tokens attend the prefix and causally
#    to earlier FAST tokens; expert tokens attend the prefix and each other, NOT the FAST tokens; nothing attends the
#    expert. make_attn_mask's cumsum rule cannot skip a block, so the mask is built explicitly.
# ======================================================================================
def make_joint_mask(prefix_valid: torch.Tensor, post_valid: torch.Tensor, n_expert: int) -> torch.Tensor:
    r"""prefix_valid, post_valid: bool[B, S] over the SAME token axis ([images | tokens]), marking which columns are valid
    prefix tokens and which are valid FAST / text postfix tokens; n_expert = E expert tokens appended after them.
    -> bool[B, S + E, S + E], True where the row (query) may attend the column (key).

    One sample with 2 image tokens, 3 prefix text tokens, 2 FAST tokens, 1 pad, and E = 3 expert tokens:

        prefix_valid = [1 1 1 1 1 0 0 0]        post_valid = [0 0 0 0 0 1 1 0]

                               <------- S = 8 -------->  <-- E = 3 -->
                               I0 I1 T0 T1 T2 F0 F1  .   E0 E1 E2
                              +------------------------+-------------+
                 /    I0      | x  x  x  x  x  .  .  . | .  .  .     |
                 |    I1      | x  x  x  x  x  .  .  . | .  .  .     |
          prefix |    T0      | x  x  x  x  x  .  .  . | .  .  .     |  <- prefix never
                 |    T1      | x  x  x  x  x  .  .  . | .  .  .     |     sees the expert
                 \    T2      | x  x  x  x  x  .  .  . | .  .  .     |
                              |     (1) bidirectional  |             |
          FAST   /    F0      | x  x  x  x  x  x  .  . | .  .  .     |
                 \    F1      | x  x  x  x  x  x  x  . | .  .  .     |
                              | (2) prefix (3) causal  |             |
          pad          .      | .  .  .  .  .  .  .  . | .  .  .     |
                              +------------------------+-------------+
                 /    E0      | x  x  x  x  x  .  .  . | x  x  x     |
          expert |    E1      | x  x  x  x  x  .  .  . | x  x  x     |
                 \    E2      | x  x  x  x  x  .  .  . | x  x  x     |
                              | (4) prefix  (5) blind  | (6) full    |
                              +------------------------+-------------+

    (5) and the empty top-right block are the point of this mask: the flow branch must not read the FAST tokens
    (they are the discrete answer to the same chunk), and the text / FAST cross entropy must not depend on whether
    an expert block is present, so that alpha = 0 pre-training and alpha = 10 post-training compute the same CE.
    Padding columns need no special case: both `valid` vectors are 0 there, so every block leaves them False.
    Matching this, the expert's position ids skip the FAST tokens as well (see `joint_forward`): at inference no
    FAST tokens exist, so the expert must not count them here either.
    """
    b, s = prefix_valid.shape
    dev = prefix_valid.device
    idx = torch.arange(s, device=dev)
    causal = idx[None, :] <= idx[:, None]  # [S, S]
    pre_r, pre_c = prefix_valid[:, :, None], prefix_valid[:, None, :]
    post_r, post_c = post_valid[:, :, None], post_valid[:, None, :]
    tok = (pre_r & pre_c) | (post_r & pre_c) | (post_r & post_c & causal[None])  # prefix <-> prefix, FAST -> prefix, FAST -> earlier FAST
    n = s + n_expert
    mask = torch.zeros(b, n, n, dtype=torch.bool, device=dev)
    mask[:, :s, :s] = tok
    mask[:, s:, :s] = pre_c.expand(-1, n_expert, -1)  # expert -> prefix only (never the FAST tokens)
    mask[:, s:, s:] = True  # expert <-> expert
    return mask  # token rows never see expert columns (mask[:, :s, s:] stays False)


# ======================================================================================
# 3. The joint forward and Eq. 1.
# ======================================================================================
def split_prefix_fast(obs: Pi05Observation):
    """The observation's token sequence holds prefix (ar 0) and postfix (ar 1) tokens left-aligned. Returns the boolean
    masks over the WHOLE [images | tokens] axis: which columns are valid prefix, which are valid FAST / text postfix."""
    n_img = 256 * len(obs.images)
    img_valid = torch.cat([obs.image_masks[k][:, None].expand(-1, 256) for k in obs.images], 1)
    tok_valid = obs.tokenized_prompt_mask
    tok_post = obs.token_ar_mask.bool() & tok_valid
    prefix = torch.cat([img_valid, tok_valid & ~tok_post], 1)
    post = torch.cat([torch.zeros_like(img_valid), tok_post], 1)
    return prefix, post, n_img


def joint_forward(model: Pi05, obs: Pi05Observation, x_t: torch.Tensor | None, t: torch.Tensor | None):
    """One pass with both outputs. -> (text_logits f32[B, L-1, V], targets i64[B, L-1], text_loss_mask bool[B, L-1],
    v_t f32[B, 50, 32] or None). x_t=None runs expert 0 only (pre-training, alpha = 0)."""
    emb, input_mask, _ar = model.embed_prefix(obs)  # [B, S, W], S = n_img*256 + L
    prefix_valid, post_valid, n_img = split_prefix_fast(obs)
    assert prefix_valid.shape == input_mask.shape
    L = obs.tokenized_prompt.shape[1]
    # expert-0 positions: cumsum over valid tokens (pi0.py L208)
    pos0 = input_mask.long().cumsum(1) - 1
    if x_t is None:
        # only the two token blocks: this is exactly pi.fast.train's forward (prefix-LM over prefix, causal postfix)
        mask = make_joint_mask(prefix_valid, post_valid, 0)
        (h0, _), _ = model.llm([emb, None], pos0, mask)
        v_t = None
    else:
        tokens, smask, sar, cond = model.proj.embed_suffix(x_t, t)
        mask = make_joint_mask(prefix_valid, post_valid, tokens.shape[1])
        # expert positions continue from the number of valid NON-FAST tokens, as at inference where no FAST tokens exist (README Sec. 8)
        pos1 = prefix_valid.long().sum(1, keepdim=True) + torch.arange(tokens.shape[1], device=emb.device)[None]
        (h0, h1), _ = model.llm([emb, tokens], torch.cat([pos0, pos1], 1), mask, cond=cond)
        v_t = model.proj.decode(h1)
    # target logits only for the last L-1 positions (pi0_fast.py L222-L226 via pi.fast.train.target_logits): position i predicts token i+1
    text_logits = model.logits_head(h0[:, n_img : n_img + L - 1])
    targets = obs.tokenized_prompt[:, 1:]
    return text_logits, targets, obs.token_loss_mask[:, 1:], v_t


def joint_loss(model: Pi05, obs: Pi05Observation, actions: torch.Tensor | None, *, alpha: float, t: torch.Tensor | None = None,
               noise: torch.Tensor | None = None, has_actions: torch.Tensor | None = None) -> dict:
    """Eq. 1: mean_b CE_b + alpha * mean_{b: has_actions} MSE_b. CE per sample normalised by its postfix length
    (pi.fast.train.cross_entropy); MSE per sample = mean over 50 x 32 of (v_t - u_t)^2 (pi0.py L212-L214, then the mean
    the training loop takes). alpha == 0: no expert pass at all."""
    b = obs.tokenized_prompt.shape[0]
    if alpha == 0.0 or actions is None:
        logits, targets, lm, _ = joint_forward(model, obs, None, None)
        ce = cross_entropy(logits, targets, lm)
        return {"loss": ce.mean(), "ce": ce, "mse": torch.zeros(b, device=ce.device)}
    x_t, u_t = interpolate(actions, noise, t)
    logits, targets, lm, v_t = joint_forward(model, obs, x_t, t)
    ce = cross_entropy(logits, targets, lm)
    mse = (v_t - u_t).pow(2).mean(dim=(1, 2))
    if has_actions is None:
        has_actions = torch.ones(b, dtype=torch.bool, device=mse.device)
    m = has_actions.to(mse.dtype)
    mse_mean = (mse * m).sum() / m.sum().clamp(min=1.0)
    return {"loss": ce.mean() + alpha * mse_mean, "ce": ce, "mse": mse}


# ======================================================================================
# 4. Post-training starts from a fresh expert (Sec. IV-D). Expert 0, SigLIP and the embedder keep the pre-trained weights.
# ======================================================================================
def expert_parameter_names(model: Pi05) -> list[str]:
    return [n for n, _ in model.named_parameters() if ".experts.1." in n or n.startswith("llm.final_norms.1") or n.startswith("proj.")]


@torch.no_grad()
def init_expert_for_posttraining(model: Pi05, seed: int = 0) -> list[str]:
    """Re-initialise expert 1 (every layer's experts[1] and final_norms[1]) and the projections; AdaRMSNorm modulations
    stay zero (../expert Sec. 1.1), so the expert starts as the identity map. Returns the names that changed."""
    g = torch.Generator().manual_seed(seed)
    names = expert_parameter_names(model)
    for layer in model.llm.layers:
        assert isinstance(layer, AdaMoEBlock)
        e1 = layer.experts[1]
        for lin in (e1.attn.q_einsum, e1.attn.kv_einsum, e1.attn.attn_vec_einsum):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * lin.weight.shape[1] ** -0.5)
        e1.mlp.gating_einsum.copy_(torch.randn(e1.mlp.gating_einsum.shape, generator=g) * e1.mlp.gating_einsum.shape[1] ** -0.5)
        e1.mlp.linear.copy_(torch.randn(e1.mlp.linear.shape, generator=g) * e1.mlp.linear.shape[0] ** -0.5)
    for m in model.modules():
        if isinstance(m, AdaRMSNorm):
            m.modulation.weight.zero_()
            m.modulation.bias.zero_()
    fresh = Pi05ActionProjections(model.proj.cfg, model.proj.action_dim, model.proj.action_horizon)
    model.proj.load_state_dict(fresh.state_dict())
    return names


def select_trainable(model: Pi05, freeze: str = "nothing") -> list[torch.nn.Parameter]:
    """openpi default: nothing frozen (config.py L492); Hi Robot App. C.1: full model unfrozen. "vlm" is offered for
    experiments: train only the expert + projections."""
    assert freeze in ("nothing", "vlm")
    expert = set(expert_parameter_names(model))
    for n, p in model.named_parameters():
        p.requires_grad_(freeze == "nothing" or n in expert)
    return [p for p in model.parameters() if p.requires_grad]


# ======================================================================================
# 5. One step, and a held-out loss.
# ======================================================================================
def train_step(model: Pi05, batch, params, optimizer, ema: EMA | None, cfg: TrainConfig, step: int, *, alpha: float,
               generator: torch.Generator | None = None) -> dict[str, float]:
    """batch = (Pi05Observation with the "fast" / "text" layout, actions f32[B, 50, 32] or None, has_actions bool[B] or None)."""
    obs, actions, has_actions = batch
    model.train()
    if actions is not None and alpha > 0:
        t = sample_timestep(actions.shape[0], generator, device=actions.device)
        noise = torch.randn(actions.shape, generator=generator, device=actions.device)
    else:
        t = noise = None
    out = joint_loss(model, obs, actions, alpha=alpha, t=t, noise=noise, has_actions=has_actions)
    out["loss"].backward()
    grad_norm = clip_and_step(params, optimizer, cfg, step)
    if ema is not None:
        ema.update(model)
    return {"loss": float(out["loss"].detach()), "ce": float(out["ce"].mean().detach()), "mse": float(out["mse"].mean().detach()),
            "grad_norm": grad_norm, "lr": lr_at(step, cfg)}


@torch.no_grad()
def evaluate_loss(model: Pi05, batches, *, alpha: float, seed: int = 0) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for obs, actions, has_actions in batches:
        t = noise = None
        if actions is not None and alpha > 0:
            t = sample_timestep(actions.shape[0], g)
            noise = torch.randn(actions.shape, generator=g)
        losses.append(joint_loss(model, obs, actions, alpha=alpha, t=t, noise=noise, has_actions=has_actions)["loss"])
    return float(torch.stack(losses).mean())


# ======================================================================================
# 6. The data mixture, as facts (paper Sec. IV-C, IV-D; weights undisclosed).
# ======================================================================================
MIXTURE = {
    "pretrain": (("MM", "fast", "mobile manipulators, ~400 h, ~100 homes"), ("ME", "fast", "static arms in many homes"), ("CE", "fast", "lab cross-embodiment incl. OXE"),
                 ("HL", "text", "subtask (+ bbox) labels on MM / ME / CE"), ("WD", "text", "CapsFusion, COCO, Cambrian-7M, PixMo, VQAv2, indoor bbox")),
    "posttrain": (("MM", "fast+flow", "successful episodes below a length threshold"), ("ME", "fast+flow", "same filter"),
                  ("HL(ME)", "text", "the ME slice of HL"), ("WD", "text", "kept to preserve semantics"), ("VI", "text", "verbal-instruction demonstrations")),
}


# ======================================================================================
# 7. One post-training step end to end with the tiny config.  uv run python -m pi.pi05.train.train
# ======================================================================================
def main():
    import numpy as np

    from pi.fast.tokenizer.tokenizer import make_smooth_chunks
    from pi.pi0.data.data import make_bool_mask
    from pi.pi05.data.data import ACTION_DIM, LL_IMAGE_KEYS, build_pi05_batch, tiny_pi05_tokenizer, unit_stats
    from pi.pi05.hier.model import tiny_pi05

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B, H, d = 2, 10, 7  # tiny FAST tokenizer at LIBERO scale; the paper chunk is 50 x 19
    model = tiny_pi05()
    seq = tiny_pi05_tokenizer(H, d)
    raw = {"images": {"base_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8), "left_wrist_0_rgb": rng.integers(0, 256, (B, 64, 80, 3), dtype=np.uint8)},
           "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
           "actions": (0.3 * make_smooth_chunks(B, H + 2, d, rng)).astype(np.float32),
           "prompt": ["put the plate in the sink", "pick up the cup"]}
    obs, actions = build_pi05_batch(raw, unit_stats(d), seq, layout="fast", image_keys=LL_IMAGE_KEYS, action_horizon=H, delta_mask=make_bool_mask(6, -1), train=True)
    model.proj.action_horizon = H  # tiny chunk of 10 steps (paper: 50)
    prefix_valid, post_valid, n_img = split_prefix_fast(obs)
    print(f"[batch]  tokens {tuple(obs.tokenized_prompt.shape)}: prefix {prefix_valid[:, n_img:].sum(1).tolist()} + FAST postfix {post_valid.sum(1).tolist()} per sample; actions {tuple(actions.shape)}")

    # --- stage 1: pre-training, alpha = 0 ---
    out0 = joint_loss(model, obs, actions, alpha=ALPHA_PRETRAIN)
    print(f"[pretrain] alpha 0: loss = CE only = {float(out0['loss'].detach()):.3f}  (per sample {[f'{v:.2f}' for v in out0['ce'].detach().tolist()]}); no expert pass")

    # --- stage 2: post-training, alpha = 10, fresh expert ---
    changed = init_expert_for_posttraining(model)
    print(f"[posttrain] re-initialised {len(changed)} expert / projection tensors; modulation stays 0 -> expert is the identity at step 0")
    t = sample_timestep(B)
    noise = torch.randn_like(actions)
    x_t, u_t = interpolate(actions, noise, t)
    mask = make_joint_mask(prefix_valid, post_valid, H)
    P = prefix_valid.shape[1]
    f0 = int(post_valid[0].nonzero()[0])
    print(f"[mask]   {tuple(mask.shape)} = [B, {P} + {H}, ...]; FAST->prefix {bool(mask[0, f0, :n_img][prefix_valid[0, :n_img]].all())}, FAST->expert {bool(mask[0, f0, P:].any())}, "
          f"expert->FAST {bool(mask[0, P, f0])}, expert->prefix {bool(mask[0, P, 0])}, expert<->expert {bool(mask[0, P:, P:].all())}, prefix->expert {bool(mask[0, 0, P:].any())}")
    logits, targets, lm, v_t = joint_forward(model, obs, x_t, t)
    print(f"[forward] text_logits {tuple(logits.shape)} (last L-1 positions), loss positions {lm.sum(1).tolist()}; v_t {tuple(v_t.shape)}")
    out = joint_loss(model, obs, actions, alpha=ALPHA_POSTTRAIN, t=t, noise=noise)
    print(f"[loss]   CE {float(out['ce'].mean().detach()):.3f} + {ALPHA_POSTTRAIN:.0f} x MSE {float(out['mse'].mean().detach()):.3f} = {float(out['loss'].detach()):.3f}")

    # --- one update with openpi's libero schedule (the paper's optimizer is undisclosed) ---
    cfg = openpi_libero_config()
    params = select_trainable(model)
    opt = make_optimizer(params, cfg)
    ema = EMA(model, cfg.ema_decay)
    stats = train_step(model, (obs, actions, None), params, opt, ema, cfg, step=0, alpha=ALPHA_POSTTRAIN)
    print(f"[step 0] {stats}  (lr from openpi pi05_libero: warmup to 5e-5 over 10k, then constant)")
    print(f"[configs] pretrain {pretrain_config()}\n          posttrain {posttrain_config()}\n          hirobot HL {hirobot_hl_config()}")


if __name__ == "__main__":
    main()
