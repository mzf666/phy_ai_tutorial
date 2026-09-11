"""pi0 action expert: Gemma 300M weights that share attention with the VLM, plus the state / action / timestep embedding.

Minimal PyTorch re-implementation of openpi's JAX code. Source of truth:
  openpi  https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
          src/openpi/models/pi0.py (embed_suffix, forward paths), src/openpi/models/gemma.py (two-expert Attention / Block / Module)
  paper   pi0 arXiv:2410.24164v1 Sec. III, Appendix B, Appendix D
Upstream license: Apache-2.0 (openpi, big_vision). This file re-implements, it does not copy.

What this module covers (inference only, as every model.py in this repo): (state, noisy_actions, timestep) -> 51
suffix tokens in the expert width (ActionProjections, used once at entry and once at exit); a Gemma stack where every
layer holds one set of weights per expert and the experts meet only inside attention; the inference-style suffix
forward against a prefix KV cache; decoding the last 50 tokens to the velocity field. The training-time joint forward
over prefix + suffix belongs to train.py (added later); timestep sampling, the loss and the Euler loop live in
../flow_matching.

Building blocks (RMSNorm, ExpertAttnProj, attend, apply_rope, GeGLU, GemmaBlock, make_attn_mask) are imported from
../vlm/model.py; the single-expert GemmaBlock there is reused as "one expert's slot" of a two-expert block.

How this relates to ../vlm (README Sec. 1.4). pi0's language model is ONE 18-layer transformer whose every layer holds
two independent sets of weights: expert 0 = Gemma 2B (from the PaliGemma checkpoint, the ../vlm weights) and
expert 1 = Gemma 300M (from scratch, added here). Image and prompt tokens go through expert 0, state and action tokens
through expert 1; norms, projections and MLPs are never shared. The two meet only inside attention, where expert 1's
queries can read expert 0's keys / values. So the VLM is not a "base" that the expert sits on top of: the two run side
by side through the same depth. In code, ../vlm's single-expert `Gemma` is the special case of `MoEGemma` with expert 1
absent (test_parity asserts this bit-for-bit); the final model in ../infer uses `MoEGemma`, and keeps from ../vlm only
SigLIP, the vocabulary Embedder and embed_prefix. This mirrors openpi, which builds a single
`_gemma.Module(configs=[paligemma_config, action_expert_config])` (pi0.py L73-L80) and no standalone Gemma 2B.
At inference expert 0 runs once per action chunk (xs=[prefix, None] -> kv cache) and expert 1 runs 10 times
(xs=[None, suffix]); in training both run together in one pass (xs=[prefix, suffix], train.py).
Attribute names follow the openpi parameter tree; expert i > 0 gets the "_i" suffix in openpi (gemma.py L443-L451),
here it is experts[i].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi.pi0.vlm.model import GEMMA_2B, GEMMA_300M, GemmaConfig, GemmaBlock, RMSNorm, apply_rope, attend, make_attn_mask, tiny_gemma

ACTION_DIM = 32  # pi0_config.py L25
ACTION_HORIZON = 50  # pi0_config.py L26


def tiny_expert() -> GemmaConfig:
    """CPU-sized action expert. Narrower than tiny_gemma (32 vs 64) on purpose: the paper expert is also narrower than
    the VLM (1024 vs 2048). Heads / kv heads / head_dim must match tiny_gemma (gemma.py L165-L168)."""
    return GemmaConfig(width=32, depth=4, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=16, vocab_size=1)


# ======================================================================================
# 1. Timestep embedding. openpi@215abfb pi0.py L47-L63.
# ======================================================================================
def posemb_sincos(pos: torch.Tensor, embedding_dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """pos float32[B] -> float32[B, embedding_dim] = concat(sin(2*pi*pos/period), cos(2*pi*pos/period)).
    period is log-uniform from min_period to max_period over embedding_dim/2 channels. pi0 calls this with
    (timestep, 1024, 4e-3, 4.0): the slowest channel completes 1/4 of a cycle over tau in [0, 1], the fastest 250.

    Why (README Sec. 1.3): this is the Transformer sinusoidal positional encoding with the integer position replaced
    by a continuous scalar in [0, 1]. A bare scalar appended to a 1024-wide activation carries almost no signal and an
    MLP is insensitive to small differences in it; spreading tau over many frequencies (Fourier features) gives slow
    channels that say roughly where tau is and fast channels that resolve differences of ~1e-3. sin/cos pairs make
    every tau uniquely decodable and make shifts in tau linear maps of the embedding. The range is chosen for
    tau in [0, 1] (upstream comment: "sensitivity in the range [0, 1]"): max_period 4 keeps the slowest channel
    monotone on [0, 1]; min_period 4e-3 resolves the smallest training tau (0.001) and the Euler step (0.1) easily.
    DDPM uses the same function on integer t in 0..1000; flow matching makes t continuous, so the periods rescale.

    What `timestep` is: the flow-matching noise level tau, one scalar per sample, unrelated to the robot control
    step t. Training draws tau ~ Beta(1.5, 1) * 0.999 + 0.001 per sample (pi0.py L197); inference walks the fixed grid
    1.0, 0.9, ..., 0.1 (L228, L278). openpi convention: tau = 1 is pure noise, tau = 0 is the clean chunk, the
    opposite of the paper's text (L226-L227). tau enters the network only here, mixed into the 50 action tokens
    at the entry; no layer sees it again."""
    assert embedding_dim % 2 == 0
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    x = pos.float()[:, None] * (2 * math.pi / period)[None, :]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


# ======================================================================================
# 2. Two-expert Gemma. openpi@215abfb gemma.py L158-L249 (Attention), L284-L333 (Block), L340-L411 (Module).
#    Every expert owns its own norms, q/kv/out projections and MLP; q, k, v of all present experts are concatenated
#    along the sequence axis and go through ONE attention. Nothing else is shared.
#
#    "MoE" here is NOT the LLM kind (Mixtral / DeepSeek): there is no router and no gating parameter. Which expert a
#    token uses is fixed by modality when the inputs are assembled (xs[0] = image + prompt tokens, xs[1] = state +
#    action tokens) and is the same in all 18 layers; the forward below just indexes `experts[i]` by list position.
#    Also unlike LLM MoE, which only splits the FFN, the whole layer is split here (norms, q/k/v, out proj, FFN);
#    only the softmax step is shared. The paper says "two sets of weights (also known as experts [45])", borrowing
#    the word, not the mechanism. Think "modality-specific parameters", see README Sec. 1.5.
# ======================================================================================
class MoEBlock(nn.Module):
    def __init__(self, cfgs: tuple[GemmaConfig, ...]):
        super().__init__()
        c0 = cfgs[0]
        assert all((c.num_heads, c.num_kv_heads, c.head_dim) == (c0.num_heads, c0.num_kv_heads, c0.head_dim) for c in cfgs)
        self.cfgs = cfgs
        # experts[i] holds pre_attention_norm / attn / pre_ffw_norm / mlp for expert i; its own forward is not used.
        self.experts = nn.ModuleList(GemmaBlock(c) for c in cfgs)

    def forward(self, xs, positions, mask, kv_cache=None):
        """xs: list over experts of x_i[B, T_i, w_i] or None. positions int64[B, sum T_i], mask bool[B, sum T_i, S].
        Returns (list of updated x_i or None, (k, v) over all present tokens incl. cache)."""
        present = [(i, x) for i, x in enumerate(xs) if x is not None]
        qs, ks, vs = [], [], []
        for i, x in present:  # step 1: per-expert norm + projection into the shared head space (L172-L199)
            e = self.experts[i]
            q, k, v = e.attn.qkv(e.pre_attention_norm(x))
            qs.append(q), ks.append(k), vs.append(v)
        q, k, v = torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1)  # step 2: concat along sequence (L201)
        q = apply_rope(q, positions) * self.cfgs[0].head_dim**-0.5
        k = apply_rope(k, positions)
        enc, kv = attend(q, k, v, mask, kv_cache)  # step 3: one attention for everybody (L216-L231)
        out = list(xs)
        start = 0
        for i, x in present:  # step 4-5: split back, per-expert out projection, residual, per-expert MLP (L233-L247, L314-L330)
            e = self.experts[i]
            end = start + x.shape[1]
            x = x + e.attn.out(enc[:, start:end])
            out[i] = x + e.mlp(e.pre_ffw_norm(x))
            start = end
        return out, kv


class MoEGemma(nn.Module):
    """Decoder stack with one set of weights per expert. gemma.py L340-L411.
    forward(xs, positions, mask, kv_cache=None) -> (list of final-normed hidden states or None, kv_cache list over layers).
    This is THE pi0 transformer: experts[0] of every layer carries the Gemma 2B weights that ../vlm's `Gemma` introduced,
    experts[1] the 300M action expert. `xs[i] = None` means expert i has no tokens this call (inference: prefix pass is
    [prefix, None], each denoising step is [None, suffix]); training (train.py) passes [prefix, suffix] together.
    The VLM's vocabulary embedding belongs to expert 0 (gemma.py L355-L359); ../vlm's PaliGemma already owns one, so
    `with_embedder` is off by default here and ../infer decides where it lives."""

    def __init__(self, cfgs: tuple[GemmaConfig, ...], with_embedder: bool = False):
        super().__init__()
        assert all(c.depth == cfgs[0].depth for c in cfgs)  # L352
        self.cfgs = cfgs
        self.embedder = None
        if with_embedder:
            from pi.pi0.vlm.model import Embedder

            self.embedder = Embedder(cfgs[0])
        self.layers = nn.ModuleList(MoEBlock(cfgs) for _ in range(cfgs[0].depth))
        self.final_norms = nn.ModuleList(RMSNorm(c.width) for c in cfgs)  # L382

    def forward(self, xs, positions, mask, kv_cache=None):
        xs = list(xs)
        new_cache = []
        for i, layer in enumerate(self.layers):
            xs, kv = layer(xs, positions, mask, None if kv_cache is None else kv_cache[i])
            new_cache.append(kv)
        return [None if x is None else n(x) for n, x in zip(self.final_norms, xs)], new_cache


# ======================================================================================
# 3. The robotics-specific projections and the suffix. openpi@215abfb pi0.py L92-L100, L140-L186, L212.
#    These five Linear layers are NOT the action expert. They sit outside the transformer: four of them run once
#    at the entry (32-dim state / actions / scalar tau -> 51 tokens of width 1024, before layer 0) and one runs once
#    at the exit (last 50 tokens after layer 17 -> 32-dim velocity). In openpi they hang off the Pi0 object, not off
#    the llm (pi0.py L92-L100). The action expert proper, Gemma 300M, lives in every layer as
#    MoEGemma.layers[i].experts[1], i = 0..17, and the suffix tokens go through it 18 times. Paper Appendix B lists
#    the two as separate additions: "(1) additional input and output projections ... (3) a second, smaller set of
#    weights for the action expert". See README Sec. 1.6.
# ======================================================================================
class ActionProjections(nn.Module):
    """state_proj, action_in_proj, action_time_mlp_in/out (entry) and action_out_proj (exit). 3,248,160 params at
    paper size; the 311M-parameter expert itself is in MoEGemma."""

    def __init__(self, cfg: GemmaConfig = GEMMA_300M, action_dim: int = ACTION_DIM, action_horizon: int = ACTION_HORIZON):
        super().__init__()
        w = cfg.width
        self.cfg, self.action_dim, self.action_horizon = cfg, action_dim, action_horizon
        self.state_proj = nn.Linear(action_dim, w)  # L97; state and action share the 32-dim padded layout (../data)
        self.action_in_proj = nn.Linear(action_dim, w)  # L92, W1 in Appendix B
        self.action_time_mlp_in = nn.Linear(2 * w, w)  # L98, W2
        self.action_time_mlp_out = nn.Linear(w, w)  # L99, W3
        self.action_out_proj = nn.Linear(w, action_dim)  # L100

    def embed_suffix(self, state, noisy_actions, timestep):
        """state f32[B, 32], noisy_actions f32[B, 50, 32], timestep f32[B] in [0, 1]
        -> (tokens f32[B, 51, w], input_mask bool[B, 51] all True, ar_mask bool[51] = [1, 1, 0 x 49]). pi0.py L140-L186."""
        b = state.shape[0]
        state_token = self.state_proj(state)[:, None, :]  # [B, 1, w], L153
        action_tokens = self.action_in_proj(noisy_actions)  # [B, 50, w], L159
        time_emb = posemb_sincos(timestep, self.cfg.width, min_period=4e-3, max_period=4.0)  # [B, w], L161
        time_tokens = time_emb[:, None, :].expand(-1, self.action_horizon, -1)  # L172
        x = torch.cat([action_tokens, time_tokens], dim=-1)  # [B, 50, 2w], L173
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))  # W3 swish(W2 concat(...)), L174-L176
        tokens = torch.cat([state_token, x], dim=1)  # [B, 51, w]
        input_mask = torch.ones(b, 1 + self.action_horizon, dtype=torch.bool, device=state.device)  # L155, L180
        # state opens a block (prefix must not see it), the first action opens the action block, the rest join it. L157, L182
        ar_mask = torch.tensor([True, True] + [False] * (self.action_horizon - 1), device=state.device)
        return tokens, input_mask, ar_mask

    def decode(self, suffix_out):
        """f32[B, 51, w] -> velocity field f32[B, 50, 32]; the state token's output is dropped. pi0.py L212."""
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])


# ======================================================================================
# 4. How the suffix meets the cached prefix at inference. openpi@215abfb pi0.py L239-L269 (sample_actions.step).
#    The training-time counterpart (prefix and suffix in ONE pass of 867 tokens, xs=[prefix, suffix], pi0.py
#    L202-L211) is training code and lives in train.py, not here.
# ======================================================================================
def suffix_forward(llm: MoEGemma, kv_cache, prefix_mask, suffix_emb, suffix_mask, suffix_ar):
    """Suffix only, attending into the cached prefix keys / values: expert 0 is absent. Returns suffix_out [B, 51, w1].
    Mask rows are the 51 suffix queries; columns are 816 cached prefix keys (visible iff valid) then 51 suffix keys."""
    b, s = suffix_emb.shape[:2]
    prefix_cols = prefix_mask[:, None, :].expand(-1, s, -1)  # [B, 51, 816], L249
    suffix_cols = make_attn_mask(suffix_mask, suffix_ar)  # [B, 51, 51], L246
    mask = torch.cat([prefix_cols, suffix_cols], dim=-1)  # [B, 51, 867], L252
    positions = prefix_mask.long().sum(-1, keepdim=True) + suffix_mask.long().cumsum(-1) - 1  # L259
    (prefix_out, suffix_out), _ = llm([None, suffix_emb], positions, mask, kv_cache=kv_cache)
    assert prefix_out is None
    return suffix_out


PI0_EXPERTS = (GEMMA_2B, GEMMA_300M)  # pi0_config.py L21-L22: paligemma_variant gemma_2b, action_expert_variant gemma_300m


def tiny_experts() -> tuple[GemmaConfig, GemmaConfig]:
    return tiny_gemma(), tiny_expert()


# ======================================================================================
# 5. Walk one inference step with the tiny config: suffix embedding, prefix cache, cached suffix forward, decode.
#    uv run python -m pi.pi0.action_expert.model
# ======================================================================================
def main():
    from pi.pi0.vlm.model import IMAGE_KEYS, PaliGemma, tiny_vit

    torch.manual_seed(0)
    B = 2
    vlm_cfg, exp_cfg = tiny_experts()
    # PaliGemma here only supplies the image encoder and the prompt embedding table (its own single-expert Gemma
    # layers are unused); the transformer that both experts run through is the MoEGemma below, whose experts[0]
    # is where the Gemma 2B weights live in the real model. ../infer assembles the final object; this main keeps
    # the two pieces visible. See README Sec. 1.4 for the vlm / action_expert relationship.
    pg = PaliGemma(tiny_vit(), vlm_cfg).eval()
    llm = MoEGemma((vlm_cfg, exp_cfg)).eval()
    proj = ActionProjections(exp_cfg).eval()
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"tiny configs: vlm expert={vlm_cfg}\n              action expert={exp_cfg}")
    print(f"params: MoEGemma expert0 {sum(n(l.experts[0]) for l in llm.layers):,}  expert1 {sum(n(l.experts[1]) for l in llm.layers):,}"
          f"  ActionProjections {n(proj):,}")

    # --- inputs: what ../data produces. state is consumed HERE, not by the VLM ---
    images = {k: torch.rand(B, 224, 224, 3) * 2 - 1 for k in IMAGE_KEYS}
    image_masks = {k: torch.ones(B, dtype=torch.bool) for k in IMAGE_KEYS}
    image_masks["right_wrist_0_rgb"][:] = False
    tokens = torch.randint(3, vlm_cfg.vocab_size, (B, 48))
    token_mask = torch.zeros(B, 48, dtype=torch.bool)
    token_mask[:, :9] = True
    state = torch.randn(B, ACTION_DIM)
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)  # at inference the first x_t is pure noise (../flow_matching)
    timestep = torch.ones(B)  # openpi convention: tau = 1 is noise, the Euler loop walks 1.0 -> 0.0 in steps of 0.1
    print("\n[input]  state f32", tuple(state.shape), " x_t f32", tuple(x_t.shape), " timestep f32", tuple(timestep.shape))

    with torch.no_grad():
        # --- 1. prefix from ../vlm: embed once, run expert 0 once, keep the kv cache (pi0.py L233-L237) ---
        prefix_emb, prefix_mask, prefix_ar = pg.embed_prefix(images, image_masks, tokens, token_mask)
        print("[prefix] emb", tuple(prefix_emb.shape), f" valid {int(prefix_mask[0].sum())}/{prefix_mask.shape[1]}")
        pmask = make_attn_mask(prefix_mask, prefix_ar)
        ppos = prefix_mask.long().cumsum(1) - 1
        (p_out, none_out), kv_cache = llm([prefix_emb, None], ppos, pmask)
        assert none_out is None
        print("[prefix] MoEGemma xs=[prefix, None]: out", tuple(p_out.shape), " kv_cache", len(kv_cache), "x", tuple(kv_cache[0][0].shape),
              " = [B, 816, kv_heads, head_dim]; this is all the VLM does per action chunk")

        # --- 2. suffix embedding, step by step (ActionProjections.embed_suffix unrolled) ---
        st = proj.state_proj(state)[:, None, :]
        print("[suffix] state_proj(state)          ", tuple(st.shape), " = [B, 1, w]")
        at_ = proj.action_in_proj(x_t)
        print("[suffix] action_in_proj(x_t)        ", tuple(at_.shape), " = [B, 50, w]")
        te = posemb_sincos(timestep, exp_cfg.width, 4e-3, 4.0)
        print("[suffix] posemb_sincos(timestep)    ", tuple(te.shape), f" = [B, w]; sample0 first/last: {te[0, 0]:.3f} {te[0, -1]:.3f}")
        cat = torch.cat([at_, te[:, None, :].expand(-1, ACTION_HORIZON, -1)], -1)
        print("[suffix] concat(action, time)       ", tuple(cat.shape), " = [B, 50, 2w]")
        mixed = proj.action_time_mlp_out(F.silu(proj.action_time_mlp_in(cat)))
        print("[suffix] mlp_out(swish(mlp_in(.)))  ", tuple(mixed.shape), " = [B, 50, w]")
        suffix_emb, suffix_mask, suffix_ar = proj.embed_suffix(state, x_t, timestep)
        assert torch.allclose(suffix_emb, torch.cat([st, mixed], 1))
        print("[suffix] tokens", tuple(suffix_emb.shape), " input_mask", tuple(suffix_mask.shape), " ar_mask", suffix_ar[:4].int().tolist(), "...")

        # --- 3. suffix forward against the cache (suffix_forward unrolled): mask and positions, then the layers ---
        P = prefix_emb.shape[1]
        prefix_cols = prefix_mask[:, None, :].expand(-1, suffix_emb.shape[1], -1)
        suffix_cols = make_attn_mask(suffix_mask, suffix_ar)
        mask = torch.cat([prefix_cols, suffix_cols], -1)
        positions = prefix_mask.long().sum(-1, keepdim=True) + suffix_mask.long().cumsum(-1) - 1
        print("[cache]  mask", tuple(mask.shape), " = [B, 51, 816 + 51]; positions", tuple(positions.shape), f" start at {int(positions[0, 0])}")
        print(f"[cache]  mask rows: state sees actions? {bool(mask[0, 0, P+1:].any())}; action sees state? {bool(mask[0, 1, P])};"
              f" actions bidirectional? {bool(mask[0, 1:, P+1:].all())}; masked camera visible? {bool(mask[0, :, 512:768].any())}")
        xs = [None, suffix_emb]
        for i, layer in enumerate(llm.layers):
            xs, kv = layer(xs, positions, mask, kv_cache[i])
            if i == 0:
                print("[cache]  layer 0: expert0 out", xs[0], " expert1 out", tuple(xs[1].shape), " k/v seen", tuple(kv[0].shape), " = [B, 867, kv_heads, head_dim]")
        s_out = suffix_forward(llm, kv_cache, prefix_mask, suffix_emb, suffix_mask, suffix_ar)
        assert torch.allclose(s_out, llm.final_norms[1](xs[1]))
        v_t = proj.decode(s_out)
        print("[cache]  suffix_out", tuple(s_out.shape), " -> decode -> v_t", tuple(v_t.shape), " = [B, 50, 32]")
    print("\n../flow_matching repeats steps 2-3 ten times with x_t <- x_t - 0.1 * v_t, timestep <- timestep - 0.1;"
          " the training-time joint forward is in train.py")


if __name__ == "__main__":
    main()
