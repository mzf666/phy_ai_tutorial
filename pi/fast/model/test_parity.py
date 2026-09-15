"""Alignment checks for the pi0-FAST model: paper-size parameter count, the tied head, prefix-LM mask over
[images | prefix | postfix], cached step-by-step decoding == full-sequence forward, right alignment as a permutation,
upstream's decode-position offset, EOS early stop, greedy determinism, and the sample_actions output contract.
Paper-size models are built on the `meta` device, so no 3B-parameter tensors are allocated."""

import numpy as np
import pytest
import torch

from pi.fast.data.data import (
    ACTION_DIM,
    EOS_ID,
    PALIGEMMA_VOCAB_SIZE,
    ByteTextCodec,
    FASTObservation,
    FASTSequenceTokenizer,
    build_fast_batch,
    tiny_fast_tokenizer,
)
from pi.fast.model.model import MAX_DECODING_STEPS, Pi0FAST, left_to_right_align, paper, tiny
from pi.fast.tokenizer.tokenizer import QuantileStats, make_smooth_chunks
from pi.pi0.data.data import make_bool_mask
from pi.pi0.vlm.model import make_attn_mask

H, D, B = 10, 7, 2
N_IMG = 3 * 256


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Pi0FAST(*tiny()).eval()


@pytest.fixture(scope="module")
def seq():
    return FASTSequenceTokenizer(ByteTextCodec(), tiny_fast_tokenizer(H, D), max_len=180)


def make_obs(seq, with_actions: bool) -> FASTObservation:
    rng = np.random.default_rng(1)
    raw = {
        "images": {"base_0_rgb": rng.integers(0, 256, (B, 128, 160, 3), dtype=np.uint8)},
        "state": rng.uniform(-0.5, 0.5, (B, D)).astype(np.float32),
        "prompt": ["pick up the red block", "close the drawer now"],
    }
    if with_actions:
        raw["actions"] = (0.3 * make_smooth_chunks(B, H, D, rng)).astype(np.float32)
    stats = {"state": QuantileStats(np.full(D, -1.0), np.full(D, 1.0)), "actions": QuantileStats(np.full(D, -1.0), np.full(D, 1.0))}
    obs, _ = build_fast_batch(raw, stats, seq, action_horizon=H, action_dim=ACTION_DIM, delta_mask=make_bool_mask(6, -1), train=False)
    return obs


def prefix_only(obs: FASTObservation) -> FASTObservation:
    """Drop the postfix from a training-mode observation (what tokenize(prompt, state, None) would have produced)."""
    keep = obs.token_ar_mask == 0
    return FASTObservation(obs.images, obs.image_masks, obs.state, obs.tokenized_prompt * keep, obs.tokenized_prompt_mask & keep,
                           torch.zeros_like(obs.token_ar_mask), torch.zeros_like(obs.token_loss_mask))


# ---------------------------------------------------------------- parameter count / shapes
def test_paper_parameter_count_is_paligemma_only():
    with torch.device("meta"):
        m = Pi0FAST(*paper())
    emb = m.llm.embedder.input_embedding.numel()
    assert n_params(m.img) == 414_803_696
    assert emb == 526_647_296 == PALIGEMMA_VOCAB_SIZE * 2048
    assert n_params(m.llm) - emb == 1_981_884_416
    assert n_params(m) == 2_923_335_408  # no action expert, no projections, no separate head


def test_logits_head_is_the_transposed_embedding(model):
    assert n_params(model) == n_params(model.img) + n_params(model.llm)  # the head owns no parameters
    x = torch.randn(B, 3, model.cfg.width)
    ref = x @ model.llm.embedder.input_embedding.t()
    out = model.logits_head(x)
    assert out.shape == (B, 3, PALIGEMMA_VOCAB_SIZE)
    assert torch.allclose(out, ref)


def test_embed_inputs_layout_and_prefix_lm_mask(model, seq):
    obs = make_obs(seq, with_actions=True)
    emb, input_mask, ar_mask = model.embed_inputs(obs)
    S = N_IMG + obs.tokenized_prompt.shape[1]
    assert emb.shape == (B, S, model.cfg.width) and input_mask.shape == (B, S) and ar_mask.shape == (B, S)
    assert not ar_mask[:, :N_IMG].any() and torch.equal(ar_mask[:, N_IMG:], obs.token_ar_mask.long())
    assert torch.equal(input_mask[:, N_IMG:], obs.tokenized_prompt_mask) and input_mask[:, :N_IMG].all()
    mask = make_attn_mask(input_mask, ar_mask)
    i = 0
    pre = (ar_mask[i] == 0) & input_mask[i]
    post = ar_mask[i] == 1
    pad = ~input_mask[i]
    assert mask[i][pre][:, pre].all()  # images + prefix text: fully bidirectional
    assert not mask[i][pre][:, post].any()  # the prefix never sees an action token
    sub = mask[i][post][:, post]
    assert torch.equal(sub, torch.tril(torch.ones_like(sub)))  # postfix: causal, token by token
    assert mask[i][post][:, pre].all()  # every action token sees the whole prefix
    assert not mask[i][pad].any() and not mask[i][:, pad].any()


# ---------------------------------------------------------------- analytic properties
def test_cached_decode_matches_full_forward(model, seq):
    """Feeding the postfix tokens one at a time through the cache (training-consistent positions) gives the same
    logits as one forward over the whole left-aligned sequence."""
    obs = make_obs(seq, with_actions=True)
    with torch.no_grad():
        pre_logits, input_mask, ar_mask = model(obs)
        full = model.logits_head(pre_logits)
        emb, m_pre, ar_pre = model.embed_inputs(prefix_only(obs))
        emb, m_pre, attn = left_to_right_align(emb, m_pre, make_attn_mask(m_pre, ar_pre))
        prefill_size = m_pre.shape[1]
        prefill_len = m_pre.long().sum(1)
        prefix_start = prefill_size - prefill_len
        h, cache = model.llm(emb, m_pre.long().cumsum(1) - 1, attn)
        logit = model.logits_head(h[:, -1:])
        col = torch.arange(prefill_size + MAX_DECODING_STEPS)
        post_ids = [obs.tokenized_prompt[b][ar_mask[b, N_IMG:] == 1] for b in range(B)]  # per sample: lengths differ
        n_post = min(len(p) for p in post_ids)
        assert n_post > 1
        for step in range(n_post):
            # full-forward logits at the position that predicts postfix token `step` = last prefix token + step
            for b in range(B):
                pos = int(prefill_len[b]) - 1 + step
                assert torch.allclose(full[b, pos], logit[b, 0], atol=1e-4, rtol=1e-4), (step, b)
            if step == n_post - 1:
                break
            token = torch.stack([p[step : step + 1] for p in post_ids])  # [B, 1]
            position = (prefill_len + step)[:, None]  # training-time position of this token (README Sec. 8)
            n_cols = prefill_size + step + 1
            cache_mask = (col[None, None, :n_cols] >= prefix_start[:, None, None]) & (col[None, None, :n_cols] < n_cols)
            logit, cache = model.decode_step(token, position, cache, cache_mask)


def test_right_align_is_a_permutation(model, seq):
    obs = make_obs(seq, with_actions=False)
    with torch.no_grad():
        emb, m, ar = model.embed_inputs(obs)
        attn = make_attn_mask(m, ar)
        h_left, _ = model.llm(emb, m.long().cumsum(1) - 1, attn)
        emb_r, m_r, attn_r = left_to_right_align(emb, m, attn)
        h_right, _ = model.llm(emb_r, m_r.long().cumsum(1) - 1, attn_r)
    for b in range(B):
        n = int(m[b].sum())
        assert m_r[b, -n:].all() and not m_r[b, :-n].any()  # real tokens now end at the last column
        assert torch.allclose(h_left[b][m[b]], h_right[b][m_r[b]], atol=1e-5)
        assert torch.allclose(h_right[b, -1], h_left[b][m[b]][-1], atol=1e-5)  # last column = last real token


def test_upstream_decode_positions_are_shifted_by_one(model, seq, monkeypatch):
    """pi0_fast.py L293: the step-th generated token gets position prefill_len + step + 1, one more than the
    training-time position prefill_len + step (recorded as a gap, README Sec. 8)."""
    obs = make_obs(seq, with_actions=False)
    seen = []
    orig = model.decode_step

    def spy(token, position, cache, cache_mask):
        seen.append(position[:, 0].clone())
        return orig(token, position, cache, cache_mask)

    monkeypatch.setattr(model, "decode_step", spy)
    _, n = model.sample_actions(obs, max_decoding_steps=3)
    prefill_len = N_IMG + obs.tokenized_prompt_mask.sum(1)
    assert n == 3 and len(seen) == 3  # upstream also runs one forward after the last sampled token (L291-L303)
    for step, pos in enumerate(seen):
        assert torch.equal(pos, prefill_len + step + 1)


def test_eos_stops_the_loop_and_pads_with_zeros(model, seq, monkeypatch):
    obs = make_obs(seq, with_actions=False)
    calls = {"n": 0}
    orig = model.logits_head

    def head(x):
        out = orig(x)
        calls["n"] += 1
        if calls["n"] >= 3:  # from the third logits (prefill + 2 steps) on, every sample prefers EOS
            out = out.clone()
            out[..., EOS_ID] = out.max() + 100.0
        return out

    monkeypatch.setattr(model, "logits_head", head)
    tokens, n = model.sample_actions(obs, max_decoding_steps=16)
    assert n == 3 and tokens.shape == (B, 16)
    assert (tokens[:, 2] == EOS_ID).all() and not (tokens[:, :2] == EOS_ID).any()
    assert not tokens[:, 3:].any()  # unfilled columns stay 0 (pi0_fast.py L271)


def test_greedy_is_argmax_and_temperature_is_reproducible(model, seq, monkeypatch):
    obs = make_obs(seq, with_actions=False)
    logits_seen = []
    orig = model.logits_head

    def head(x):
        out = orig(x)
        logits_seen.append(out[:, 0].argmax(-1))
        return out

    monkeypatch.setattr(model, "logits_head", head)
    tokens, n = model.sample_actions(obs, max_decoding_steps=4)
    assert torch.equal(tokens[:, :n], torch.stack(logits_seen[:n], 1))  # greedy = argmax of each step's logits
    t2, _ = model.sample_actions(obs, max_decoding_steps=4)
    assert torch.equal(tokens, t2)
    a, _ = model.sample_actions(obs, max_decoding_steps=4, temperature=0.7, generator=torch.Generator().manual_seed(3))
    b, _ = model.sample_actions(obs, max_decoding_steps=4, temperature=0.7, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_sample_actions_output_contract(model, seq):
    obs = make_obs(seq, with_actions=False)
    tokens, n = model.sample_actions(obs, max_decoding_steps=5)
    assert tokens.dtype == torch.long and tokens.shape == (B, 5) and 1 <= n <= 5
    assert int(tokens.min()) >= 0 and int(tokens.max()) < PALIGEMMA_VOCAB_SIZE
    acts = seq.extract_actions(tokens[0].numpy(), H, D)  # random weights: no marker -> zeros fallback
    assert acts.shape == (H, D)
