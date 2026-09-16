"""Alignment checks for pi0.5 hierarchical inference: paper parameter count, cached text decoding vs the full forward,
greedy determinism and EOS early stop, low-level sampling vs a hand-written Euler loop and the camera masks,
split_response, and the Hi Robot schedule (1 s, user message, resume)."""

import numpy as np
import pytest
import torch

from pi.fast.data.data import EOS_ID
from pi.pi0.action_expert.model import ACTION_DIM, ACTION_HORIZON
from pi.pi0.vlm.model import make_attn_mask
from pi.pi05.data.data import HL_IMAGE_KEYS, LL_IMAGE_KEYS, build_pi05_batch, tiny_pi05_tokenizer
from pi.pi05.hier.model import PI05_EXPERTS, HierarchicalPolicy, Pi05, split_response, tiny_pi05

D = 19


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_pi05().eval()


@pytest.fixture(scope="module")
def seq():
    return tiny_pi05_tokenizer(10, 7)


@pytest.fixture(scope="module")
def raws():
    rng = np.random.default_rng(5)
    images = {k: rng.integers(0, 256, (1, 64, 80, 3), dtype=np.uint8) for k in HL_IMAGE_KEYS}
    state = rng.uniform(-0.5, 0.5, (1, D)).astype(np.float32)
    return {"images": images, "state": state, "prompt": ["clean the kitchen"]}, {"images": {k: images[k] for k in LL_IMAGE_KEYS}, "state": state}


def test_paper_parameter_count():
    from pi.pi0.vlm.model import SIGLIP_SO400M_14

    with torch.device("meta"):
        m = Pi05(SIGLIP_SO400M_14, PI05_EXPERTS)
    assert sum(p.numel() for p in m.parameters()) == 3_353_433_872


# ---------------------------------------------------------------- high level
def test_cached_decoding_matches_full_forward(model, seq, raws):
    raw_hl, _ = raws
    obs, _ = build_pi05_batch(raw_hl, None, seq, layout="hl_prompt", image_keys=HL_IMAGE_KEYS, delta_mask=None, train=False)
    T = 4
    with torch.no_grad():
        # (a) step-by-step with the cache, training-time positions (prefill_len + step), no right alignment (no padding needed: B=1)
        emb, mask, ar = model.embed_prefix(obs)
        prefill_len = int(mask.sum())
        (pre, _), cache = model.llm([emb, None], mask.long().cumsum(1) - 1, make_attn_mask(mask, ar))
        logit = model.logits_head(pre[:, prefill_len - 1 : prefill_len])  # last real token (left-aligned, B=1)
        toks, step_logits = [], []
        S = mask.shape[1]
        for step in range(T):
            tok = logit[:, 0].argmax(-1, keepdim=True)
            toks.append(int(tok))
            n_cols = S + step + 1
            cache_mask = torch.zeros(1, 1, n_cols, dtype=torch.bool)
            cache_mask[0, 0, :S] = mask[0]
            cache_mask[0, 0, S:] = True
            logit, cache = model.decode_step(tok, torch.tensor([[prefill_len + step]]), cache, cache_mask)
            step_logits.append(logit[0, 0])
        # (b) one full forward over prefix + the generated tokens as a causal postfix
        ids = obs.tokenized_prompt.clone()
        m2 = obs.tokenized_prompt_mask.clone()
        ar2 = obs.token_ar_mask.clone()
        n_text = int(m2.sum())
        ids[0, n_text : n_text + T] = torch.tensor(toks)
        m2[0, n_text : n_text + T] = True
        ar2[0, n_text : n_text + T] = 1
        obs2 = type(obs)(obs.images, obs.image_masks, obs.state, ids, m2, ar2, obs.token_loss_mask)
        emb2, mask2, ar_2 = model.embed_prefix(obs2)
        (h, _), _ = model.llm([emb2, None], mask2.long().cumsum(1) - 1, make_attn_mask(mask2, ar_2))
        n_img = emb2.shape[1] - ids.shape[1]
        full = model.logits_head(h[0, n_img + n_text : n_img + n_text + T])
    for s in range(T - 1):  # logits produced right after generated token s == full-forward logits at that token's position
        assert torch.allclose(step_logits[s], full[s], atol=2e-3), s


def test_greedy_is_deterministic_and_eos_stops(model, seq, raws, monkeypatch):
    raw_hl, _ = raws
    obs, _ = build_pi05_batch(raw_hl, None, seq, layout="hl_prompt", image_keys=HL_IMAGE_KEYS, delta_mask=None, train=False)
    a, na = model.sample_text(obs, max_new_tokens=6)
    b, nb = model.sample_text(obs, max_new_tokens=6)
    assert torch.equal(a, b) and na == nb == 6 and a.shape == (1, 6)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    c, _ = model.sample_text(obs, max_new_tokens=6, temperature=1.0, generator=g1)
    d, _ = model.sample_text(obs, max_new_tokens=6, temperature=1.0, generator=g2)
    assert torch.equal(c, d)
    # force EOS on the third step
    calls = {"n": 0}
    orig = model.logits_head

    def head(x):
        out = orig(x)
        calls["n"] += 1
        if calls["n"] == 3:
            out = out.clone()
            out[..., EOS_ID] = 1e9
        return out

    monkeypatch.setattr(model, "logits_head", head)
    toks, n = model.sample_text(obs, max_new_tokens=10)
    assert n == 3 and int(toks[0, 2]) == EOS_ID and bool((toks[0, 3:] == 0).all())


# ---------------------------------------------------------------- low level
def test_sample_actions_matches_manual_euler_and_masks(model, seq, raws):
    raw_hl, raw_ll = raws
    obs, _ = build_pi05_batch({**raw_ll, "prompt": ["pick up the plate"]}, None, seq, layout="flow", image_keys=LL_IMAGE_KEYS, delta_mask=None, train=False)
    assert list(obs.images) == list(LL_IMAGE_KEYS) and all(bool(v.all()) for v in obs.image_masks.values())
    noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM)
    with torch.no_grad():
        kv, pm = model.prefix_cache(obs)
        v = model.make_velocity_fn(kv, pm)
        x, t = noise, 1.0
        for _ in range(10):
            x = x + (-0.1) * v(x, torch.full((1,), t))
            t -= 0.1
        assert torch.allclose(model.sample_actions(obs, noise, 10), x, atol=1e-5)
    # a low-level observation built on the 4-slot layout with the rear camera absent: mask False on base_1
    obs4, _ = build_pi05_batch({**raw_ll, "prompt": ["x"]}, None, seq, layout="flow", image_keys=HL_IMAGE_KEYS, delta_mask=None, train=False)
    assert [bool(v[0]) for v in obs4.image_masks.values()] == [True, False, True, True]


# ---------------------------------------------------------------- Hi Robot pieces
def test_split_response():
    assert split_response("pick up the bowl") == ("pick up the bowl", None)
    assert split_response("respond: Sorry!") == ("", "Sorry!")
    assert split_response("put it back respond: Whoops, sorry") == ("put it back", "Whoops, sorry")
    assert split_response("Respond: ") == ("", None)


def test_schedule_one_second_user_message_and_resume(model, seq, raws, monkeypatch):
    raw_hl, raw_ll = raws
    pol = HierarchicalPolicy(model, seq, max_new_tokens=2)
    script = ["pick up the plate", "put it back respond: sorry", "open the drawer"]

    def hl(raw, msg=None, generator=None):
        cmd, utt = split_response(script.pop(0))
        return cmd, utt, 2

    monkeypatch.setattr(pol, "high_level", hl)
    o0 = pol.step(raw_hl, raw_ll, 0.0)
    assert o0["hl_ran"] and o0["subtask"] == "pick up the plate" and o0["actions"].shape == (1, ACTION_HORIZON, ACTION_DIM)
    o1 = pol.step(raw_hl, raw_ll, 0.5)
    assert not o1["hl_ran"] and o1["subtask"] == "pick up the plate"
    o2 = pol.step(raw_hl, raw_ll, 0.7, user_message="that's not trash")
    assert o2["hl_ran"] and o2["subtask"] == "put it back" and o2["utterance"] == "sorry"
    assert pol.resume() == "pick up the plate"
    o3 = pol.step(raw_hl, raw_ll, 1.75)
    assert o3["hl_ran"] and o3["subtask"] == "open the drawer"
