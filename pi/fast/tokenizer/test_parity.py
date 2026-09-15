"""Alignment checks for the FAST tokenizer: analytic DCT, quantile normalization, flatten order, GPT-2 pre-tokenization,
byte-level BPE round trips, reconstruction bound, and (if downloaded) the released FAST+ vocabulary. CPU, seconds."""

import math

import numpy as np
import pytest

from pi.fast.tokenizer.tokenizer import (
    BPE,
    BYTE_ALPHABET,
    FAST_PLUS_MIN_TOKEN,
    FAST_PLUS_VOCAB_SIZE,
    FASTTokenizer,
    QuantileStats,
    bytes_to_unicode,
    dct,
    dct_matrix,
    fast_plus_available,
    idct,
    ints_to_text,
    make_smooth_chunks,
    normalize_quantile,
    pretokenize,
    quantize,
    text_to_ints,
    unnormalize_quantile,
)


def test_dct_matrix_is_orthonormal_and_matches_the_formula():
    n = 16
    C = dct_matrix(n)
    assert np.allclose(C @ C.T, np.eye(n), atol=1e-12)
    x = np.random.default_rng(0).normal(size=n)
    direct = np.array([sum(x[i] * math.cos(math.pi * (2 * i + 1) * k / (2 * n)) for i in range(n)) for k in range(n)])
    direct *= math.sqrt(2.0 / n)
    direct[0] /= math.sqrt(2.0)  # scipy's norm="ortho" scaling of the DC term
    assert np.allclose(C @ x, direct)
    assert np.allclose(idct(dct(x[:, None], axis=0), axis=0)[:, 0], x)


def test_dct_of_a_constant_is_only_the_dc_term():
    x = np.full((10, 3), 0.4)
    c = dct(x, axis=0)
    assert np.allclose(c[0], 0.4 * math.sqrt(10))
    assert np.allclose(c[1:], 0.0)


def test_quantile_normalization_maps_q01_q99_to_unit_interval_and_inverts():
    stats = QuantileStats(q01=np.array([-2.0, 0.0]), q99=np.array([2.0, 1.0]))
    assert np.allclose(normalize_quantile(stats.q01, stats), -1.0, atol=1e-5)
    assert np.allclose(normalize_quantile(stats.q99, stats), 1.0, atol=1e-5)
    x = np.array([[3.0, 0.25], [-2.5, 0.9]])  # first row has an outlier beyond q99: no clipping
    z = normalize_quantile(x, stats)
    assert z[0, 0] > 1.0
    assert np.allclose(unnormalize_quantile(z, stats), x)
    padded = np.concatenate([z, np.zeros((2, 3))], axis=-1)  # 32-dim style padding passes through
    assert np.allclose(unnormalize_quantile(padded, stats)[:, 2:], 0.0)


def test_quantize_rounds_half_to_even_like_np_around():
    assert quantize(np.array([0.25, 0.35, -0.25]), 10).tolist() == [2, 4, -2]


def test_flatten_is_frequency_major_and_characters_invert():
    q = np.arange(12).reshape(4, 3) - 5  # [H=4, D=3]
    text = ints_to_text(q.reshape(-1), min_token=-5)
    assert [ord(c) for c in text[:3]] == [0, 1, 2]  # frequency 0 of all 3 dims comes first
    assert np.array_equal(text_to_ints(text, -5).reshape(4, 3), q)
    assert ints_to_text(np.array([-9]), min_token=-5) == chr(0)  # clamp at 0 like upstream


def test_pretokenize_matches_gpt2_regex_cases():
    assert pretokenize("abc123!! xy") == ["abc", "123", "!!", " xy"]
    assert pretokenize("a  b") == ["a", " ", " b"]  # \s+(?!\S) leaves the last space for the word
    assert pretokenize("a  ") == ["a", "  "]
    assert pretokenize("it's") == ["it", "'s"]
    assert pretokenize("Ţ" * 5 + "ţ" * 2) == ["Ţ" * 5 + "ţ" * 2]  # letters of any script form one word
    for s in ["", "   ", "a b c", "\x00\x01Ab9 ", "Ţ" * 40 + "\x00" + "Ţ" * 3]:
        assert "".join(pretokenize(s)) == s


def test_byte_alphabet_is_the_gpt2_map():
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256
    assert m[ord("!")] == "!" and m[0] == chr(256)
    assert len(BYTE_ALPHABET) == 256


def test_bpe_train_encode_decode_round_trip_and_vocab_budget():
    texts = ["ŢŢŢŢŢŢab" * 3, "ŢŢŢŢcd" * 2, "abcd"]
    bpe = BPE.train(texts, vocab_size=270)
    assert len(bpe.vocab) <= 270 and len(bpe.merges) == len(bpe.vocab) - 256
    assert bpe.merges[0] in {("Å", "¢"), ("Å¢", "Å¢")} or "Å¢" in bpe.merges[0][0]  # zero runs merge first
    for t in texts + ["Ţxyz"]:
        assert bpe.decode(bpe.encode(t)) == t


def test_merges_never_cross_pretokenization_boundaries():
    texts = ["ab 12 ab 12"] * 4  # letters, numbers and spaces are separate words
    bpe = BPE.train(texts, vocab_size=262)
    ascii_letters, digits = set("abcdefghijklmnopqrstuvwxyz"), set("0123456789")
    for tok in bpe.vocab:
        if len(tok) > 1:
            assert not (set(tok) & ascii_letters and set(tok) & digits), tok


def test_fit_encode_decode_reconstruction_within_quantization_bound():
    rng = np.random.default_rng(1)
    H, D = 20, 4
    chunks = make_smooth_chunks(48, H, D, rng)
    tok = FASTTokenizer.fit(list(chunks), scale=10, vocab_size=320)
    ids = tok(chunks[:8])
    assert len(ids) == 8 and all(isinstance(i, int) for i in ids[0])
    assert max(len(t) for t in ids) < H * D  # compresses relative to naive per-step binning
    rec = tok.decode(ids, time_horizon=H, action_dim=D, on_error="raise")
    assert rec.shape == (8, H, D)
    assert np.abs(rec - chunks[:8]).max() <= 0.5 / 10 * math.sqrt(H) + 1e-9  # per-coefficient error <= 0.05, idct orthonormal
    assert ((rec - chunks[:8]) ** 2).mean() < (0.5 / 10) ** 2


def test_decode_falls_back_to_zeros_like_upstream():
    rng = np.random.default_rng(2)
    tok = FASTTokenizer.fit(list(make_smooth_chunks(16, 10, 2, rng)), vocab_size=300)
    good = tok(make_smooth_chunks(1, 10, 2, rng))[0]
    bad = good[:-3]  # wrong number of coefficients
    out = tok.decode([bad], time_horizon=10, action_dim=2)
    assert out.shape == (1, 10, 2) and np.all(out == 0)
    with pytest.raises(ValueError):
        tok.decode([bad], time_horizon=10, action_dim=2, on_error="raise")


@pytest.mark.skipif(not fast_plus_available(), reason="FAST+ files not downloaded into .upstream/fast_hf/")
def test_released_fast_plus_vocabulary_round_trips_exactly():
    tok = FASTTokenizer.from_hf_dir()
    assert tok.vocab_size == FAST_PLUS_VOCAB_SIZE and tok.min_token == FAST_PLUS_MIN_TOKEN and tok.scale == 10
    assert len(tok.bpe.merges) == 2048 - 256
    assert tok.bpe.decode(tok.bpe.vocab and [tok.bpe.vocab["Å¢"]]) == chr(354)  # first merge is the 0 coefficient
    rng = np.random.default_rng(3)
    x = make_smooth_chunks(4, 50, 7, rng)  # a 50 Hz single arm
    ids = tok(x)
    assert all(len(t) < 350 / 4 for t in ids)  # far below naive 50 * 7 = 350
    q = tok.coefficients(x)
    rec_q = np.stack([text_to_ints(tok.bpe.decode(t), tok.min_token).reshape(50, 7) for t in ids])
    assert np.array_equal(rec_q, q)  # BPE is lossless; the only loss is the rounding
    rec = tok.decode(ids, time_horizon=50, action_dim=7, on_error="raise")
    assert np.abs(rec - x).max() <= 0.05 * math.sqrt(50) + 1e-9
