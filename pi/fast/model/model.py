"""pi0-FAST model: PaliGemma with no action expert. SigLIP image tokens + the token sequence from ../data go through
Gemma 2B under a prefix-LM mask; the word-embedding matrix transposed is the logits head; actions are generated token by
token with a KV cache and cut back out by ../data's extract_actions.

Re-implementation (PyTorch). Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/models/pi0_fast.py: make_attn_mask L23-L48, left_to_right_align L51-L64, Pi0FAST.__init__ L134-L157,
           embed_inputs L159-L195, sample_actions L236-L313; Pi0FASTConfig L76-L98
           src/openpi/models/gemma_fast.py: Embedder.decode (tied head) L120-L121, fixed-size KV cache L165-L183,
           Module.__call__ L302-L418 (return_prelogits, decode)
  paper    FAST, arXiv:2501.09747v1, Sec. VI-A (no model changes), Sec. VI-E (inference cost), Appendix C (decoding)
License of the upstream code: Apache-2.0 (openpi). This file re-implements, it does not copy.

Everything that is PaliGemma (SigLIP, Gemma blocks, RMSNorm, RoPE, grouped-query attention with a KV cache, the
prefix-LM mask) is imported from pi.pi0.vlm.model. This file only holds the increment: the tied logits head, the
image + token-sequence assembly with a data-given ar_mask, right alignment, and the autoregressive decoding loop.
Training forward / loss live in ../train; this file is inference only.
"""

from __future__ import annotations

import dataclasses
import time

import torch
import torch.nn as nn

from pi.fast.data.data import EOS_ID, IMAGE_KEYS, PALIGEMMA_VOCAB_SIZE, FASTObservation
from pi.pi0.vlm.model import GEMMA_2B, SIGLIP_SO400M_14, Gemma, GemmaConfig, SigLIP, ViTConfig, make_attn_mask, tiny_gemma, tiny_vit

MAX_DECODING_STEPS = 256  # pi0_fast.py L241; the KV cache is sized prefill_size + this (L263)


# ======================================================================================
# Configs. paper(): pi0_fast.py L137-L156 (gemma_2b + So400m/14). tiny(): pi0/vlm's tiny shapes, full vocab.
# ======================================================================================
def paper() -> tuple[ViTConfig, GemmaConfig]:
    return SIGLIP_SO400M_14, GEMMA_2B  # Pi0FASTConfig.paligemma_variant = "gemma_2b" (pi0_fast.py L79)


def tiny() -> tuple[ViTConfig, GemmaConfig]:
    """CPU-sized. The vocab stays 257,152 because ../data maps action tokens into the PaliGemma tail (ids >= 254976);
    the embedding is then 257152 x 64 = 16.5M floats, fine on CPU. Logits are only ever computed at a few positions."""
    return tiny_vit(), dataclasses.replace(tiny_gemma(), vocab_size=PALIGEMMA_VOCAB_SIZE)


# ======================================================================================
# 1. Right alignment. pi0_fast.py L51-L64. Inference only.
# ======================================================================================
def left_to_right_align(x: torch.Tensor, input_mask: torch.Tensor, attn_mask: torch.Tensor):
    """Per sample, roll the sequence so the real tokens end at the last column and padding sits in front.
    x f32[B, S, W], input_mask bool[B, S], attn_mask bool[B, S, S] -> same shapes.
    seqlen = (index of the last True in input_mask) + 1; everything is rolled by -seqlen (rows and columns of attn_mask).
    Attention relations are unchanged: make_attn_mask only depends on cumsum and validity, both roll along."""
    b, s = input_mask.shape
    ar = torch.arange(s, device=x.device)
    seqlen = (input_mask.long() * ar).amax(dim=1) + 1  # L60
    out_x, out_m, out_a = torch.empty_like(x), torch.empty_like(input_mask), torch.empty_like(attn_mask)
    for i in range(b):  # upstream vmaps over the batch; a loop keeps it readable
        k = -int(seqlen[i])
        out_x[i] = torch.roll(x[i], k, dims=0)
        out_m[i] = torch.roll(input_mask[i], k, dims=0)
        out_a[i] = torch.roll(attn_mask[i], (k, k), dims=(0, 1))
    return out_x, out_m, out_a


# ======================================================================================
# 2. The model. pi0_fast.py L134-L195 + gemma_fast.py L120-L121, L344-L418.
# ======================================================================================
class Pi0FAST(nn.Module):
    """PaliGemma (SigLIP + Gemma) and nothing else. Parameters: 2,923,335,408 at paper size (test_parity.py)."""

    def __init__(self, vit_cfg: ViTConfig = SIGLIP_SO400M_14, gemma_cfg: GemmaConfig = GEMMA_2B):
        super().__init__()
        assert vit_cfg.out_dim == gemma_cfg.width
        self.img = SigLIP(vit_cfg)  # pi0_fast.py L147-L156
        self.llm = Gemma(gemma_cfg)  # pi0_fast.py L139-L146, with embedder
        self.cfg = gemma_cfg

    # --- tied head: gemma_fast.py L120-L121 (decode = x . E^T) --------------------------------------------------
    def logits_head(self, x: torch.Tensor) -> torch.Tensor:
        """f32[..., width] -> f32[..., vocab]. No separate weight, no bias: the embedding matrix transposed."""
        return x @ self.llm.embedder.input_embedding.t()

    # --- input assembly: pi0_fast.py L159-L195 ------------------------------------------------------------------
    def embed_inputs(self, obs: FASTObservation):
        """-> (emb f32[B, S, W], input_mask bool[B, S], ar_mask i64[B, S]); S = 3 * 256 + max_len.
        Images in IMAGE_KEYS order with ar 0 (L179), then the whole token sequence with the data-given ar_mask (L188).
        obs.state is not read (there is no state projection in pi0-FAST)."""
        embs, masks, ars = [], [], []
        for k in IMAGE_KEYS:
            t = self.img(obs.images[k])  # [B, 256, W]
            embs.append(t)
            masks.append(obs.image_masks[k][:, None].expand(-1, t.shape[1]))
            ars.append(torch.zeros(t.shape[:2], dtype=torch.long, device=t.device))
        embs.append(self.llm.embed(obs.tokenized_prompt))  # embed_only=True (L185): lookup * sqrt(width)
        masks.append(obs.tokenized_prompt_mask)
        ars.append(obs.token_ar_mask.long())
        return torch.cat(embs, 1), torch.cat(masks, 1), torch.cat(ars, 1)

    # --- full-sequence forward: gemma_fast.py L344-L413 with return_prelogits=True -------------------------------
    def forward(self, obs: FASTObservation):
        """One forward over the whole (left-aligned) sequence -> (pre_logits f32[B, S, W], input_mask, ar_mask).
        Logits are deliberately not formed here ([B, S, 257152] is 1.95 GB at S=948, B=2); apply logits_head where
        needed. ../train gathers the target positions first (pi0_fast.py L222-L226)."""
        emb, input_mask, ar_mask = self.embed_inputs(obs)
        mask = make_attn_mask(input_mask, ar_mask)
        positions = input_mask.long().cumsum(1) - 1
        pre_logits, _ = self.llm(emb, positions, mask)
        return pre_logits, input_mask, ar_mask

    # --- one decoding step: pi0_fast.py L292-L301 ----------------------------------------------------------------
    def decode_step(self, token: torch.Tensor, position: torch.Tensor, cache, cache_mask: torch.Tensor):
        """token i64[B, 1], position i64[B, 1], cache = list over layers of (k, v) [B, S_cache, K, H],
        cache_mask bool[B, 1, S_cache + 1] = which cache columns (plus itself, the last column) this query may see
        -> (logits f32[B, 1, vocab], cache grown by one column)."""
        x = self.llm.embed(token)
        h, cache = self.llm(x, position, cache_mask, cache)
        return self.logits_head(h), cache

    # --- autoregressive decoding: pi0_fast.py L236-L313 ----------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, obs: FASTObservation, *, max_decoding_steps: int = MAX_DECODING_STEPS, temperature: float = 0.0,
                       generator: torch.Generator | None = None) -> tuple[torch.Tensor, int]:
        """-> (tokens i64[B, max_decoding_steps], n_steps). tokens[:, 0] is the first token after the prefix;
        columns after the stop are 0 (L271). Stops when every sample has produced EOS (L288-L289) or at the cap."""
        emb, input_mask, ar_mask = self.embed_inputs(obs)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        emb, input_mask, attn_mask = left_to_right_align(emb, input_mask, attn_mask)  # L254-L256
        b, prefill_size = input_mask.shape
        prefill_len = input_mask.long().sum(1)  # [B] real tokens per sample (L258)
        prefix_start = prefill_size - prefill_len  # [B] first real column after right alignment (L259)

        # prefill: one forward over the right-aligned prefix; the cache holds all prefill_size columns (pads masked)
        positions = input_mask.long().cumsum(1) - 1  # L264: real tokens get 0..prefill_len-1
        pre_logits, cache = self.llm(emb, positions, attn_mask)
        last_logit = self.logits_head(pre_logits[:, -1:])  # L270: last column = last real token after right alignment

        tokens = torch.zeros(b, max_decoding_steps, dtype=torch.long, device=emb.device)  # L271
        has_eos = torch.zeros(b, dtype=torch.bool, device=emb.device)
        col = torch.arange(prefill_size + max_decoding_steps, device=emb.device)
        step = 0
        while step < max_decoding_steps:
            if temperature > 0.0:  # L279-L284
                probs = torch.softmax(last_logit[:, 0] / temperature, dim=-1)
                token = torch.multinomial(probs, 1, generator=generator)  # [B, 1]
            else:
                token = last_logit[:, 0].argmax(-1, keepdim=True)
            tokens[:, step] = token[:, 0]  # L285
            has_eos |= token[:, 0] == EOS_ID  # L288
            step += 1
            if bool(has_eos.all()):  # L289, L307: stop once the whole batch has emitted EOS
                break
            # one step: the new token is written at cache column prefill_size + step - 1; it may see columns
            # [prefix_start, prefill_size + step) (L294-L298). Upstream position is prefill_len + step (L293, with its
            # step counted from 0: prefill_len + step + 1), one more than the training-time position; see README Sec. 8.
            position = (prefill_len + step)[:, None]
            n_cols = prefill_size + step
            cache_mask = (col[None, None, :n_cols] >= prefix_start[:, None, None]) & (col[None, None, :n_cols] < n_cols)
            last_logit, cache = self.decode_step(token, position, cache, cache_mask)
        return tokens, step


# ======================================================================================
# 3. Walk a prefill + a few decode steps with the tiny config and print every intermediate shape.
#    uv run python -m pi.fast.model.model
# ======================================================================================
def main() -> None:
    import numpy as np

    from pi.fast.data.data import ACTION_DIM, ByteTextCodec, FASTSequenceTokenizer, build_fast_batch, tiny_fast_tokenizer
    from pi.fast.tokenizer.tokenizer import QuantileStats
    from pi.pi0.data.data import make_bool_mask

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B, H, d = 2, 10, 7  # LIBERO-like: 7-dim, action_horizon 10 (config.py L711)
    vit_cfg, gemma_cfg = tiny()
    model = Pi0FAST(vit_cfg, gemma_cfg).eval()
    n_img, n_llm, n_emb = (sum(p.numel() for p in m.parameters()) for m in (model.img, model.llm, model.llm.embedder))
    print(f"tiny configs: vit={vit_cfg}\n              gemma={gemma_cfg}")
    print(f"params: SigLIP {n_img:,}  Gemma {n_llm:,} (of which embedding {n_emb:,}); logits head adds 0 (tied)")

    # --- inputs: what ../data produces in inference mode (actions=None -> prefix only) ---
    raw = {  # no "actions" key = inference mode: the sequence is prefix only
        "images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8), "wrist_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
        "state": rng.uniform(-0.5, 0.5, (B, d)).astype(np.float32),
        "prompt": ["Pick_up the red block", "close the drawer"],
    }
    stats = {"state": QuantileStats(np.full(d, -1.0), np.full(d, 1.0)), "actions": QuantileStats(np.full(d, -1.0), np.full(d, 1.0))}
    seq = FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, d), max_len=180)
    obs, _ = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=False)
    print(f"\n[input]  images x{len(obs.images)} f32 {tuple(obs.images['base_0_rgb'].shape)}  tokenized_prompt i64 {tuple(obs.tokenized_prompt.shape)}"
          f"  real tokens per sample {obs.tokenized_prompt_mask.sum(1).tolist()}  ar_mask any? {bool(obs.token_ar_mask.any())}")

    with torch.no_grad():
        # --- 1. assembly ---
        emb, input_mask, ar_mask = model.embed_inputs(obs)
        print(f"[embed]  emb {tuple(emb.shape)} = [B, 3*256 + max_len, width]  input_mask {tuple(input_mask.shape)}  ar_mask {tuple(ar_mask.shape)}")
        print(f"[embed]  valid per sample {input_mask.sum(1).tolist()} = 768 image + prefix text; ar_mask sum {ar_mask.sum(1).tolist()} (inference: all prefix)")
        attn_mask = make_attn_mask(input_mask, ar_mask)
        print(f"[mask]   attn_mask {tuple(attn_mask.shape)}  fraction attendable {attn_mask[0].float().mean():.3f} (= (valid/S)^2, all bidirectional)")

        # --- 2. right alignment ---
        emb_r, mask_r, attn_r = left_to_right_align(emb, input_mask, attn_mask)
        prefill_size = mask_r.shape[1]
        prefill_len = mask_r.long().sum(1)
        prefix_start = prefill_size - prefill_len
        print(f"[align]  first real column: before {[int(m.nonzero()[0]) for m in input_mask]}  after {prefix_start.tolist()}  (prefill_size {prefill_size}, prefill_len {prefill_len.tolist()})")

        # --- 3. prefill ---
        positions = mask_r.long().cumsum(1) - 1
        t0 = time.perf_counter()
        pre_logits, cache = model.llm(emb_r, positions, attn_r)
        t_prefill = time.perf_counter() - t0
        print(f"[prefill] pre_logits {tuple(pre_logits.shape)}  cache {len(cache)} layers x (k, v) {tuple(cache[0][0].shape)} = [B, prefill_size, kv_heads, head_dim]  {t_prefill*1e3:.0f} ms")
        last_logit = model.logits_head(pre_logits[:, -1:])
        print(f"[head]   logits of the last column {tuple(last_logit.shape)} = [B, 1, vocab]; argmax {last_logit[:, 0].argmax(-1).tolist()}")

        # --- 4. a few decode steps by hand (what sample_actions loops) ---
        col = torch.arange(prefill_size + MAX_DECODING_STEPS)
        token = last_logit[:, 0].argmax(-1, keepdim=True)
        for step in range(1, 4):
            position = (prefill_len + step)[:, None]
            n_cols = prefill_size + step
            cache_mask = (col[None, None, :n_cols] >= prefix_start[:, None, None]) & (col[None, None, :n_cols] < n_cols)
            t0 = time.perf_counter()
            last_logit, cache = model.decode_step(token, position, cache, cache_mask)
            dt = time.perf_counter() - t0
            print(f"[step {step-1}] token {token[:, 0].tolist()}  position {position[:, 0].tolist()}  cache_mask {tuple(cache_mask.shape)} visible cols {cache_mask[0, 0].sum().item()}"
                  f"  cache now {tuple(cache[0][0].shape)}  {dt*1e3:.1f} ms")
            token = last_logit[:, 0].argmax(-1, keepdim=True)

    # --- 5. the public entry point: greedy, then temperature 0.7 ---
    t0 = time.perf_counter()
    tokens, n_steps = model.sample_actions(obs, max_decoding_steps=32)
    dt = time.perf_counter() - t0
    print(f"\n[sample] greedy: tokens {tuple(tokens.shape)} i64, ran {n_steps} steps (cap 32; paper cap {MAX_DECODING_STEPS}), {dt*1e3:.0f} ms total, {dt/n_steps*1e3:.1f} ms/step")
    print(f"         first 8 ids of sample 0: {tokens[0, :8].tolist()}  (random weights: not action ids, EOS={EOS_ID} unlikely)")
    g = torch.Generator().manual_seed(0)
    tokens_t, n_t = model.sample_actions(obs, max_decoding_steps=32, temperature=0.7, generator=g)
    print(f"[sample] temperature 0.7: {n_t} steps, first 8 ids of sample 0: {tokens_t[0, :8].tolist()}")
    acts = seq.extract_actions(tokens[0].numpy(), H, d)
    print(f"[actions] extract_actions(tokens[0]) -> {tuple(acts.shape)}, all zero? {bool((acts == 0).all())} (no 'Action: ' marker in random output -> zeros fallback, ../data Sec. 3)")
    print("\nreal model: 30-60 steps x 2B backbone = ~750 ms per chunk on an RTX 4090 (paper Sec. VI-E); see ../infer")


if __name__ == "__main__":
    main()
