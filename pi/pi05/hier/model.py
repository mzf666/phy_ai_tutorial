"""pi0.5 hierarchical inference: one model, two levels. High level = autoregressive decoding of a subtask sentence
(prefix-LM over expert 0, tied logits head, KV cache); low level = 10 flow-matching steps of the adaRMSNorm expert
conditioned on that sentence. Plus Hi Robot's scheduling: rerun the high level every second or on a user message,
strip the verbal response before handing the command to the low level, switch back after an interjection.

Minimal PyTorch re-implementation. Sources of truth:
  openpi   https://github.com/Physical-Intelligence/openpi @ 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/models/pi0.py (Pi0.__init__ L66-L103 pi05 branch, embed_prefix L106-L137, sample_actions L216-L279),
           src/openpi/models/pi0_fast.py (sample_actions L236-L313: the text decoding loop), gemma_fast.py L120-L121 (tied head)
  paper    pi0.5 arXiv:2504.16054v1 Sec. IV-A (factorisation), IV-B (decode text, then 10 denoising steps), IV-E (cameras);
           Hi Robot arXiv:2502.19417v2 Sec. 4.1 (1-second rule), 4.2 (interjections, verbal response), Fig. 6
Upstream license: Apache-2.0. This file re-implements, it does not copy. Upstream has no high-level inference code;
that part follows the papers (README Sec. 8 lists every inferred detail).

Inference only (repo rule). The joint training forward and loss are in ../train.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch
import torch.nn as nn

from pi.fast.data.data import EOS_ID, PALIGEMMA_VOCAB_SIZE
from pi.fast.model.model import left_to_right_align
from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON, tiny_experts
from pi.pi0.flow_matching.model import VelocityFn, sample_actions as euler_sample
from pi.pi0.vlm.model import GEMMA_2B, SIGLIP_SO400M_14, Embedder, GemmaConfig, SigLIP, ViTConfig, make_attn_mask, tiny_vit
from pi.pi05.data.data import HL_IMAGE_KEYS, LL_IMAGE_KEYS, Pi05Observation, Pi05SequenceTokenizer, build_pi05_batch, parse_hl_text
from pi.pi05.expert.model import PI05_EXPERTS, AdaMoEGemma, Pi05ActionProjections, suffix_forward

HL_PERIOD_S = 1.0  # Hi Robot Sec. 4.1: "rerun high-level inference ... when one second has elapsed"; pi0.5's own rate undisclosed
MAX_NEW_TOKENS = 64  # undisclosed, see README Sec. 8; subtasks are a few words
RESPOND_MARKER = "respond:"  # Hi Robot Fig. 6 ("respond: Done! ...", "respond: Sorry!"); exact format undisclosed


# ======================================================================================
# 1. The model. pi0.py L66-L103 with pi05=True: SigLIP, the vocabulary embedder (also the logits head), the two-expert
#    stack with adaptive norms on expert 1 (../expert), and the four projections.
# ======================================================================================
class Pi05(nn.Module):
    def __init__(self, vit_cfg: ViTConfig = SIGLIP_SO400M_14, experts: tuple[GemmaConfig, GemmaConfig] = PI05_EXPERTS,
                 action_dim: int = ACTION_DIM, action_horizon: int = ACTION_HORIZON):
        super().__init__()
        assert vit_cfg.out_dim == experts[0].width
        self.img = SigLIP(vit_cfg)  # L81-L90
        self.embedder = Embedder(experts[0])  # gemma.py L354-L358
        self.llm = AdaMoEGemma(experts)  # L73-L80 with adarms=True
        self.proj = Pi05ActionProjections(experts[1], action_dim, action_horizon)  # L92-L95, L100
        self.action_horizon = action_horizon

    # --- tied head (gemma_fast.py L120-L121): used by the high level only ---
    def logits_head(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.embedder.input_embedding.t()

    # --- prefix: images in the observation's slot order, then the token sequence with its data-given ar mask ---
    def embed_prefix(self, obs: Pi05Observation):
        """-> (emb f32[B, n_img*256 + max_len, W], input_mask bool[B, S], ar_mask i64[B, S]). pi0.py L106-L137;
        the token ar mask comes from ../data (all 0 for the "flow" / "hl_prompt" layouts)."""
        embs, masks, ars = [], [], []
        for k in obs.images:
            t = self.img(obs.images[k])
            embs.append(t)
            masks.append(obs.image_masks[k][:, None].expand(-1, t.shape[1]))
            ars.append(torch.zeros(t.shape[:2], dtype=torch.long, device=t.device))
        e = self.embedder.encode(obs.tokenized_prompt)
        embs.append(e)
        masks.append(obs.tokenized_prompt_mask)
        ars.append(obs.token_ar_mask.long())
        return torch.cat(embs, 1), torch.cat(masks, 1), torch.cat(ars, 1)

    def prefix_cache(self, obs: Pi05Observation):
        """Expert 0 once over the prefix -> (kv_cache, prefix_mask). pi0.py L233-L237."""
        emb, mask, ar = self.embed_prefix(obs)
        (_, none), kv = self.llm([emb, None], mask.long().cumsum(1) - 1, make_attn_mask(mask, ar))
        assert none is None
        return kv, mask

    # --- low level: flow matching on the adaRMSNorm expert. pi0.py L239-L271 with adarms_cond ---
    def make_velocity_fn(self, kv_cache, prefix_mask) -> VelocityFn:
        def v(x_t, t):
            tokens, smask, sar, cond = self.proj.embed_suffix(x_t, t)
            return self.proj.decode(suffix_forward(self.llm, kv_cache, prefix_mask, tokens, smask, sar, cond))

        return v

    def sample_actions(self, obs: Pi05Observation, noise: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """Observation (layout "flow") + noise f32[B, 50, 32] -> normalized action chunk f32[B, 50, 32].
        Prefix once, then num_steps Euler steps (pi.pi0.flow_matching); paper Sec. IV-B: 10 steps."""
        kv, prefix_mask = self.prefix_cache(obs)
        return euler_sample(self.make_velocity_fn(kv, prefix_mask), noise, num_steps)

    # --- high level: text decoding on expert 0. pi0_fast.py L236-L313, same loop as pi.fast.model ---
    def decode_step(self, token, position, cache, cache_mask):
        x = self.embedder.encode(token)
        (h, _), cache = self.llm([x, None], position, cache_mask, cache)
        return self.logits_head(h), cache

    @torch.no_grad()
    def sample_text(self, obs: Pi05Observation, *, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor, int]:
        """Observation (layout "hl_prompt") -> (tokens i64[B, max_new_tokens], n_steps). Greedy by default; stops when
        every sample has emitted EOS or at the cap. Right alignment, prefill and the cache window follow pi0_fast.py."""
        emb, input_mask, ar_mask = self.embed_prefix(obs)
        emb, input_mask, attn_mask = left_to_right_align(emb, input_mask, make_attn_mask(input_mask, ar_mask))
        b, prefill_size = input_mask.shape
        prefill_len = input_mask.long().sum(1)
        prefix_start = prefill_size - prefill_len
        positions = input_mask.long().cumsum(1) - 1
        (pre, _), cache = self.llm([emb, None], positions, attn_mask)
        last_logit = self.logits_head(pre[:, -1:])
        tokens = torch.zeros(b, max_new_tokens, dtype=torch.long, device=emb.device)
        has_eos = torch.zeros(b, dtype=torch.bool, device=emb.device)
        col = torch.arange(prefill_size + max_new_tokens, device=emb.device)
        step = 0
        while step < max_new_tokens:
            if temperature > 0.0:
                token = torch.multinomial(torch.softmax(last_logit[:, 0] / temperature, -1), 1, generator=generator)
            else:
                token = last_logit[:, 0].argmax(-1, keepdim=True)
            tokens[:, step] = token[:, 0]
            has_eos |= token[:, 0] == EOS_ID
            step += 1
            if bool(has_eos.all()):
                break
            position = (prefill_len + step)[:, None]  # upstream's +1 offset, see pi.fast.model README Sec. 8
            n_cols = prefill_size + step
            cache_mask = (col[None, None, :n_cols] >= prefix_start[:, None, None]) & (col[None, None, :n_cols] < n_cols)
            last_logit, cache = self.decode_step(token, position, cache, cache_mask)
        return tokens, step


def tiny_pi05() -> Pi05:
    """Tiny SigLIP + tiny two-expert Gemma with the full 257,152 vocabulary (../data maps FAST ids into its tail)."""
    vlm_cfg, exp_cfg = tiny_experts()
    return Pi05(tiny_vit(), (dataclasses.replace(vlm_cfg, vocab_size=PALIGEMMA_VOCAB_SIZE), exp_cfg))


# ======================================================================================
# 2. Hi Robot's verbal response. Sec. 4.2: the high-level output may carry an utterance u_t, which is spoken to the
#    user and removed before the text goes to the low level. Fig. 6 writes it as 'respond: ...'.
# ======================================================================================
def split_response(text: str) -> tuple[str, str | None]:
    """'pick up the bowl' -> ('pick up the bowl', None); 'respond: Sorry!' -> ('', 'Sorry!');
    'put it back respond: Whoops, sorry' -> ('put it back', 'Whoops, sorry'). Empty command = keep the previous one."""
    low = text.lower()
    i = low.find(RESPOND_MARKER)
    if i < 0:
        return text.strip(), None
    return text[:i].strip(), text[i + len(RESPOND_MARKER):].strip() or None


# ======================================================================================
# 3. The two-level policy with Hi Robot's schedule. Sec. 4.1-4.2.
# ======================================================================================
class HierarchicalPolicy:
    def __init__(self, model: Pi05, seq: Pi05SequenceTokenizer, *, hl_period_s: float = HL_PERIOD_S, hl_keys=HL_IMAGE_KEYS,
                 ll_keys=LL_IMAGE_KEYS, num_steps: int = 10, max_new_tokens: int = MAX_NEW_TOKENS, action_horizon: int = ACTION_HORIZON):
        self.model, self.seq = model.eval(), seq
        self.hl_period_s, self.hl_keys, self.ll_keys = hl_period_s, tuple(hl_keys), tuple(ll_keys)
        self.num_steps, self.max_new_tokens, self.action_horizon = num_steps, max_new_tokens, action_horizon
        self.subtask, self.previous_subtask, self.last_hl_time = "", None, None

    def hl_due(self, t_now: float, user_message: str | None) -> bool:
        """First call, a user message (Sec. 4.2 "triggered immediately"), or hl_period_s elapsed (Sec. 4.1)."""
        return self.last_hl_time is None or user_message is not None or (t_now - self.last_hl_time) >= self.hl_period_s

    @torch.no_grad()
    def high_level(self, raw_hl: dict, user_message: str | None = None, generator=None) -> tuple[str, str | None, int]:
        """-> (command for the low level, utterance or None, decode steps). The user message is appended to the
        high-level prompt (format undisclosed, README Sec. 8)."""
        raw = dict(raw_hl)
        if user_message is not None:
            raw["prompt"] = [f"{p} {user_message}" for p in raw_hl["prompt"]]
        obs, _ = build_pi05_batch(raw, None, self.seq, layout="hl_prompt", image_keys=self.hl_keys, action_horizon=self.action_horizon, delta_mask=None, train=False)
        tokens, n = self.model.sample_text(obs, max_new_tokens=self.max_new_tokens, generator=generator)
        subtask, _boxes = parse_hl_text(self.seq.extract_text(tokens[0].numpy()))
        command, utterance = split_response(subtask)
        return command, utterance, n

    @torch.no_grad()
    def low_level(self, raw_ll: dict, subtask: str, noise: torch.Tensor | None = None) -> torch.Tensor:
        raw = {**raw_ll, "prompt": [subtask] * raw_ll["state"].shape[0]}
        obs, _ = build_pi05_batch(raw, None, self.seq, layout="flow", image_keys=self.ll_keys, action_horizon=self.action_horizon, delta_mask=None, train=False)
        if noise is None:
            noise = torch.randn(obs.state.shape[0], self.action_horizon, self.model.proj.action_dim)
        return self.model.sample_actions(obs, noise, self.num_steps)

    def step(self, raw_hl: dict, raw_ll: dict, t_now: float, user_message: str | None = None, noise=None, generator=None) -> dict:
        """One control-loop step: maybe the high level, always the low level. raw_* carry ALREADY NORMALIZED state
        (norm_stats=None here); ../infer wraps this with the robot's quantile stats and the inverse transforms."""
        timing, utterance, hl_ran, n_tokens = {}, None, False, 0
        if self.hl_due(t_now, user_message):
            t0 = time.perf_counter()
            command, utterance, n_tokens = self.high_level(raw_hl, user_message, generator)
            timing[f"high-level (prefill + {n_tokens} tokens)"] = (time.perf_counter() - t0) * 1e3
            if command:
                if user_message is not None and command != self.subtask:
                    self.previous_subtask = self.subtask  # remember what the interjection replaced (Sec. 4.2)
                self.subtask = command
            self.last_hl_time, hl_ran = t_now, True
        t0 = time.perf_counter()
        actions = self.low_level(raw_ll, self.subtask, noise)
        timing[f"low-level (prefix + {self.num_steps} steps)"] = (time.perf_counter() - t0) * 1e3
        return {"subtask": self.subtask, "utterance": utterance, "hl_ran": hl_ran, "hl_tokens": n_tokens, "actions": actions, "timing": timing}

    def resume(self) -> str:
        """The user signals the interjection is fulfilled: go back to the previous command (Sec. 4.2)."""
        if self.previous_subtask is not None:
            self.subtask, self.previous_subtask = self.previous_subtask, None
        return self.subtask


# ======================================================================================
# 4. One high-level + one low-level call with the tiny config, every shape and the forward counts.
#    uv run python -m pi.pi05.hier.model
# ======================================================================================
def main():
    from pi.pi05.data.data import tiny_pi05_tokenizer

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    H, d = ACTION_HORIZON, 19  # the paper's 19-dim mobile manipulator; state already normalized
    model = tiny_pi05()
    seq = tiny_pi05_tokenizer(10, 7)  # the FAST tokenizer is unused at inference; any one will do
    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"tiny Pi05 params: img {n(model.img):,}  embedder {n(model.embedder):,}  llm {n(model.llm):,}  proj {n(model.proj):,}  total {n(model):,}")
    hl_images = {k: rng.integers(0, 256, (1, 96, 128, 3), dtype=np.uint8) for k in HL_IMAGE_KEYS}
    state = rng.uniform(-0.5, 0.5, (1, d)).astype(np.float32)
    raw_hl = {"images": hl_images, "state": state, "prompt": ["clean the kitchen"]}
    raw_ll = {"images": {k: hl_images[k] for k in LL_IMAGE_KEYS}, "state": state}

    with torch.no_grad():
        # --- high level, unrolled ---
        obs, _ = build_pi05_batch(raw_hl, None, seq, layout="hl_prompt", image_keys=HL_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False)
        emb, mask, ar = model.embed_prefix(obs)
        print(f"\n[HL prefix] {len(obs.images)} cameras x 256 + {obs.tokenized_prompt.shape[1]} tokens = {emb.shape[1]}; valid {int(mask.sum())}; ar all 0? {not ar.any()}")
        print(f"[HL prompt] {seq.text.decode(obs.tokenized_prompt[0].tolist())!r}")
        t0 = time.perf_counter()
        tokens, n_steps = model.sample_text(obs, max_new_tokens=24)
        dt = (time.perf_counter() - t0) * 1e3
        text = seq.extract_text(tokens[0].numpy())
        print(f"[HL decode] {n_steps} steps (cap 24), {dt:.0f} ms = 1 prefill + {n_steps - 1} single-token forwards on expert 0 (2B at paper size)")
        print(f"[HL text]   ids {tokens[0, :6].tolist()} ... -> {text[:40]!r} (random weights: not a sentence) -> split_response -> {split_response(parse_hl_text(text)[0])[0][:30]!r}")

        # --- low level, unrolled ---
        subtask = "pick up the plate"
        obs_ll, _ = build_pi05_batch({**raw_ll, "prompt": [subtask]}, None, seq, layout="flow", image_keys=LL_IMAGE_KEYS, action_horizon=H, delta_mask=None, train=False)
        emb, mask, ar = model.embed_prefix(obs_ll)
        print(f"\n[LL prefix] {len(obs_ll.images)} cameras (rear masked: {[int(v[0]) for v in obs_ll.image_masks.values()]}) + tokens = {emb.shape[1]}; valid {int(mask.sum())}")
        print(f"[LL prompt] {seq.text.decode(obs_ll.tokenized_prompt[0].tolist())!r}")
        kv, pmask = model.prefix_cache(obs_ll)
        v = model.make_velocity_fn(kv, pmask)
        noise = torch.randn(1, H, ACTION_DIM)
        t0 = time.perf_counter()
        x_0 = euler_sample(v, noise, 10)
        dt = (time.perf_counter() - t0) * 1e3
        print(f"[LL flow]   prefix once (expert 0) + 10 x expert 1 over 50 tokens -> x_0 {tuple(x_0.shape)}, {dt:.0f} ms; == model.sample_actions? {torch.allclose(x_0, model.sample_actions(obs_ll, noise))}")

    # --- the scheduled policy ---
    pol = HierarchicalPolicy(model, seq, max_new_tokens=24)
    print("\n[schedule] Hi Robot rule: HL at t=0, then every 1 s or on a user message")
    for t_now, msg in [(0.0, None), (0.4, None), (1.05, None), (1.3, "that's not trash"), (1.6, None)]:
        out = pol.step(raw_hl, raw_ll, t_now, msg)
        print(f"  t={t_now:.2f}s msg={msg!r:20s} hl_ran={out['hl_ran']!s:5s} hl_tokens={out['hl_tokens']:2d} subtask={out['subtask'][:24]!r:28s} utterance={out['utterance']!r:8s} timing={{{', '.join(f'{k}: {v:.0f} ms' for k, v in out['timing'].items())}}}")
    print(f"  resume() -> {pol.resume()[:24]!r}")
    print("\n../infer wraps this with the robot's norm_stats and the inverse transforms, and runs the mock-home evaluation")


if __name__ == "__main__":
    main()
