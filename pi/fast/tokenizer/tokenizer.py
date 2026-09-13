"""FAST action tokenizer: quantile-normalized action chunk -> DCT-II -> scale & round -> flatten -> byte-level BPE.

Re-implementation (NumPy) of the released tokenizer. Sources of truth:
  fast_hf  https://huggingface.co/physical-intelligence/fast  processing_action_tokenizer.py
           (`UniversalActionProcessor`; accessed 2026-09-13; Apache-2.0), processor_config.json,
           tokenizer.json (the FAST+ vocabulary)
  openpi   https://github.com/Physical-Intelligence/openpi  commit 215abfb217dbac7d5f1273282331b9b1866c0479
           src/openpi/models/tokenizer.py L51-L140 (the caller), src/openpi/transforms.py L141-L181 (quantile norm)
  paper    FAST: Efficient Action Tokenization for Vision-Language-Action Models, arXiv:2501.09747v1,
           Sec. V, Algorithm 1, Fig. 4
This file re-implements, it does not copy, and it does not import scipy or tokenizers.

Read top to bottom:
  1. quantile normalization      2. DCT-II            3. quantize / flatten / integers <-> characters
  4. byte-level BPE (train, encode, decode, load FAST+)   5. FASTTokenizer   6. main()
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import pathlib
import unicodedata
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# Constants.
# --------------------------------------------------------------------------------------
DEFAULT_SCALE = 10  # gamma in Algorithm 1. paper Sec. V-B; fast_hf L20 default `scale: float = 10`
DEFAULT_VOCAB_SIZE = 1024  # paper Sec. V-B single-dataset setting; fast_hf L21 default
FAST_PLUS_VOCAB_SIZE = 2048  # released universal tokenizer, processor_config.json
FAST_PLUS_MIN_TOKEN = -354  # processor_config.json
FAST_PLUS_SCALE = 10  # processor_config.json
FAST_PLUS_DIR = pathlib.Path(__file__).resolve().parents[3] / ".upstream" / "fast_hf"  # gitignored, optional


# --------------------------------------------------------------------------------------
# 1. Quantile normalization. openpi@215abfb transforms.py L141-L145 (normalize), L175-L181 (unnormalize);
#    selected for every non-pi0 model by training/config.py L187. Paper Sec. V-B: 1st / 99th quantile -> [-1, 1].
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class QuantileStats:
    """Per-dimension 1% and 99% quantiles of the training set, float[d], d = the robot's native dim.
    openpi stores them as `q01` / `q99` in norm_stats.json next to `mean` / `std`."""

    q01: np.ndarray
    q99: np.ndarray


def normalize_quantile(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    """[..., d] -> [..., d]; maps q01 -> -1 and q99 -> +1 linearly, no clipping (outliers land outside [-1, 1])."""
    d = x.shape[-1]
    q01, q99 = stats.q01[..., :d], stats.q99[..., :d]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: QuantileStats) -> np.ndarray:
    """Inverse. Accepts x wider than the stats (padded model output): extra dims pass through unchanged."""
    d = stats.q01.shape[-1]
    head = (x[..., :d] + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
    return np.concatenate([head, x[..., d:]], axis=-1) if x.shape[-1] > d else head


# --------------------------------------------------------------------------------------
# 2. DCT-II with orthonormal scaling. fast_hf L52 `dct(action_chunk, axis=1, norm="ortho")`, L91 `idct(...)`.
#    C[k, n] = s_k cos(pi (2n + 1) k / (2N)), s_0 = sqrt(1/N), s_k = sqrt(2/N). C is orthogonal, so idct = C^T.
# --------------------------------------------------------------------------------------
def dct_matrix(n: int) -> np.ndarray:
    """float64[n, n]; `dct_matrix(n) @ x` equals scipy.fft.dct(x, type=2, norm="ortho") for x of length n."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c[0] /= math.sqrt(2.0)
    return c


def dct(x: np.ndarray, axis: int = -2) -> np.ndarray:
    """DCT-II along `axis` (default: the time axis of an [..., H, D] chunk). Same shape."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    y = x @ dct_matrix(x.shape[-1]).T
    return np.moveaxis(y, -1, axis)


def idct(x: np.ndarray, axis: int = -2) -> np.ndarray:
    """Inverse of `dct` (DCT-III with the same orthonormal scaling)."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    y = x @ dct_matrix(x.shape[-1])
    return np.moveaxis(y, -1, axis)


# --------------------------------------------------------------------------------------
# 3. Quantize, flatten, and turn integers into a string the BPE can eat.
#    fast_hf L53: np.around(dct_coeff * scale) -- numpy rounds half to even (2.5 -> 2, 3.5 -> 4).
#    fast_hf L56: "".join(map(chr, np.maximum(elem.flatten() - min_token, 0)))
#      `elem.flatten()` on [H, D] is row-major: all D dims of frequency 0, then all D dims of frequency 1, ...
#      This is the paper's "low-frequency components first" order (Sec. V-B), chosen so the autoregressive
#      model commits to the overall shape of the chunk before its details.
#    fast_hf L82: np.array(list(map(ord, decoded))) + min_token, then reshape(-1, action_dim).
# --------------------------------------------------------------------------------------
def quantize(coeff: np.ndarray, scale: float) -> np.ndarray:
    """float[...] -> int64[...]: round(scale * coeff), half to even. The only lossy step of the tokenizer."""
    return np.around(coeff * scale).astype(np.int64)


def ints_to_text(ints: np.ndarray, min_token: int) -> str:
    """int64[n] -> str of n characters, character code = value - min_token, clamped at 0 like upstream.
    The clamp only matters if a coefficient is below the minimum seen when the tokenizer was fit."""
    codes = np.maximum(np.asarray(ints).reshape(-1) - min_token, 0)
    return "".join(map(chr, codes.tolist()))


def text_to_ints(text: str, min_token: int) -> np.ndarray:
    """Inverse: str -> int64[len(text)]."""
    return np.fromiter((ord(c) for c in text), dtype=np.int64, count=len(text)) + min_token


# --------------------------------------------------------------------------------------
# 4. Byte-level BPE. Upstream trains `tokenizers.ByteLevelBPETokenizer()` with default settings
#    (fast_hf L134-L149); the released tokenizer.json confirms pre_tokenizer = ByteLevel(use_regex=true).
#    Two consequences that change token boundaries and therefore cannot be skipped:
#      (a) every character is first UTF-8 encoded and each byte mapped to a printable unicode character
#          (GPT-2's bytes_to_unicode); the BPE alphabet is those 256 byte characters;
#      (b) the string is split into "words" by the GPT-2 regex
#              's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
#          and merges never cross word boundaries. For quantized DCT coefficients this is a side effect
#          (chr(48..57) count as "numbers", chr(65..90) as "letters", ...), but it is what was trained.
# --------------------------------------------------------------------------------------
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte -> printable unicode character map (256 entries). Printable ASCII and Latin-1 letters map to
    themselves; the remaining 68 bytes map to U+0100.. in order."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_B2U = bytes_to_unicode()
_U2B = {u: b for b, u in _B2U.items()}
BYTE_ALPHABET = tuple(sorted(_B2U.values()))  # the 256 single-character tokens of any byte-level BPE

# Rust regex `\s` = Unicode White_Space (what the `tokenizers` crate uses). Python's str.isspace() also
# includes U+001C-U+001F, which Rust does not, so we list the set explicitly.
_WHITESPACE = frozenset("\t\n\x0b\x0c\r \x85\xa0 " + "".join(map(chr, range(0x2000, 0x200B))) + "    　")
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _is_letter(c: str) -> bool:
    return unicodedata.category(c)[0] == "L"


def _is_number(c: str) -> bool:
    return unicodedata.category(c)[0] == "N"


def _is_space(c: str) -> bool:
    return c in _WHITESPACE


def pretokenize(text: str) -> list[str]:
    """Split like the GPT-2 regex (see above), without the `regex` package. Concatenating the output gives
    back `text` exactly."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "'":
            for suf in _CONTRACTIONS:
                if text.startswith(suf, i):
                    out.append(suf)
                    i += len(suf)
                    break
            else:
                suf = None
            if suf is not None:
                continue
        j = i
        if text[i] == " " and i + 1 < n and not _is_space(text[i + 1]):
            j = i + 1  # the optional leading space of ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`
        c = text[j]
        if _is_space(c):  # here j == i: a whitespace run
            k = i
            while k < n and _is_space(text[k]):
                k += 1
            if k < n and k - i > 1:
                k -= 1  # `\s+(?!\S)`: leave the last space to prefix the next word
            out.append(text[i:k])
            i = k
            continue
        pred = _is_letter if _is_letter(c) else _is_number if _is_number(c) else (lambda ch: not (_is_space(ch) or _is_letter(ch) or _is_number(ch)))
        k = j
        while k < n and pred(text[k]):
            k += 1
        out.append(text[i:k])
        i = k
    return out


def _word_to_byte_chars(word: str) -> list[str]:
    return [_B2U[b] for b in word.encode("utf-8")]


class BPE:
    """Byte-level BPE with the same vocab / merges format as HF tokenizers' `tokenizer.json`.

    vocab:  token string -> id.  merges: ordered list of (left, right) pairs; the index is the merge rank.
    Encoding a word applies the lowest-ranked applicable merge repeatedly (the standard greedy algorithm).
    """

    def __init__(self, vocab: dict[str, int], merges: Sequence[tuple[str, str]]):
        self.vocab = dict(vocab)
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: r for r, m in enumerate(self.merges)}

    # ---- construction -------------------------------------------------------------------------------
    @classmethod
    def from_hf_json(cls, path: str | pathlib.Path) -> "BPE":
        """Load a HF `tokenizer.json` (e.g. the released FAST+ vocabulary)."""
        blob = json.loads(pathlib.Path(path).read_text())
        merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in blob["model"]["merges"]]
        return cls(blob["model"]["vocab"], merges)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        initial_alphabet: Sequence[str] = BYTE_ALPHABET,
        max_token_length: int = 10_000,
    ) -> "BPE":
        """Train on raw strings. Mirrors fast_hf L134-L145 `BpeTrainer(vocab_size, min_frequency=2, special_tokens=[],
        initial_alphabet=..., max_token_length=10000)` on top of ByteLevel pre-tokenization.

        Alphabet ids are assigned in sorted character order, merges get increasing ids after that; ties in pair
        frequency go to the smaller (left id, right id) pair. Both follow HF's trainer as far as it is documented
        (see README gap ledger)."""
        words: collections.Counter[tuple[str, ...]] = collections.Counter()
        for text in texts:
            for w in pretokenize(text):
                words[tuple(_word_to_byte_chars(w))] += 1
        alphabet = set(initial_alphabet)
        for w in words:
            alphabet.update(w)
        vocab = {c: i for i, c in enumerate(sorted(alphabet))}
        merges: list[tuple[str, str]] = []
        seqs = {w: list(w) for w in words}
        while len(vocab) < vocab_size:
            pairs: collections.Counter[tuple[str, str]] = collections.Counter()
            for w, seq in seqs.items():
                cnt = words[w]
                for a, b in zip(seq, seq[1:]):
                    pairs[(a, b)] += cnt
            if not pairs:
                break
            best = max(pairs.items(), key=lambda kv: (kv[1], (-vocab[kv[0][0]], -vocab[kv[0][1]])))
            (a, b), freq = best
            if freq < min_frequency or len(a) + len(b) > max_token_length:
                break
            new = a + b
            vocab[new] = len(vocab)
            merges.append((a, b))
            for w, seq in seqs.items():
                if len(seq) < 2:
                    continue
                out, i = [], 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                seqs[w] = out
        return cls(vocab, merges)

    # ---- encode / decode ----------------------------------------------------------------------------
    def _merge_word(self, symbols: list[str]) -> list[str]:
        while len(symbols) > 1:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            symbols = symbols[:best] + [symbols[best] + symbols[best + 1]] + symbols[best + 2 :]
        return symbols

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in pretokenize(text):
            for tok in self._merge_word(_word_to_byte_chars(w)):
                ids.append(self.vocab[tok])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """ids -> text. Raises KeyError on an unknown id and UnicodeDecodeError on bytes that are not UTF-8
        (the model may emit any token; see FASTTokenizer.decode for the upstream fallback)."""
        chars = "".join(self.id_to_token[int(i)] for i in ids)
        return bytes(_U2B[c] for c in chars).decode("utf-8")


# --------------------------------------------------------------------------------------
# 5. The tokenizer. fast_hf `UniversalActionProcessor`.
# --------------------------------------------------------------------------------------
class FASTTokenizer:
    """Action chunk [H, D] in normalized space <-> list of BPE token ids in [0, vocab_size).

    Attributes mirror fast_hf L17-L39: `scale` (gamma), `min_token` (offset that makes every quantized coefficient
    a non-negative character code), and the (time_horizon, action_dim) needed to reshape at decode time."""

    def __init__(self, bpe: BPE, scale: float = DEFAULT_SCALE, min_token: int = 0, *, time_horizon: int | None = None, action_dim: int | None = None):
        self.bpe, self.scale, self.min_token = bpe, scale, min_token
        self.time_horizon, self.action_dim = time_horizon, action_dim
        self.called_time_horizon, self.called_action_dim = time_horizon, action_dim

    @property
    def vocab_size(self) -> int:
        return len(self.bpe.vocab)

    # ---- encode (fast_hf L43-L58) -------------------------------------------------------------------
    def coefficients(self, chunk: np.ndarray) -> np.ndarray:
        """[B, H, D] or [H, D] -> int64[B, H, D] quantized DCT coefficients (the sparse matrix of paper Fig. 4)."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 2:
            chunk = chunk[None]
        assert chunk.ndim == 3, "expected [batch, time, action_dim] or [time, action_dim]"
        return quantize(dct(chunk, axis=1), self.scale)

    def __call__(self, chunk: np.ndarray) -> list[list[int]]:
        q = self.coefficients(chunk)
        self.called_time_horizon, self.called_action_dim = q.shape[1], q.shape[2]
        return [self.bpe.encode(ints_to_text(elem, self.min_token)) for elem in q]

    # ---- decode (fast_hf L60-L96) -------------------------------------------------------------------
    def decode(self, tokens: Sequence[Sequence[int]], *, time_horizon: int | None = None, action_dim: int | None = None, on_error: str = "zeros") -> np.ndarray:
        """list of B token lists -> float64[B, H, D]. H, D resolve as: argument -> constructor -> last encode call.
        on_error="zeros" reproduces upstream (fast_hf L91-L94: print and return a zero chunk); "raise" re-raises."""
        H = time_horizon or self.time_horizon or self.called_time_horizon
        D = action_dim or self.action_dim or self.called_action_dim
        assert H is not None and D is not None, "pass time_horizon and action_dim, or encode once first"
        self.called_time_horizon, self.called_action_dim = H, D
        out = []
        for ids in tokens:
            try:
                ints = text_to_ints(self.bpe.decode(ids), self.min_token)
                q = ints.reshape(-1, D)  # fast_hf L83: reshape(-1, action_dim), then assert shape == (H, D)
                if q.shape != (H, D):
                    raise ValueError(f"decoded {q.shape[0] * D} coefficients, expected {H * D}")
            except Exception as e:  # noqa: BLE001 -- upstream catches everything
                if on_error == "raise":
                    raise
                print(f"Error decoding tokens: {e}")
                q = np.zeros((H, D), dtype=np.int64)
            out.append(idct(q / self.scale, axis=0))
        return np.stack(out)

    # ---- fit (fast_hf L99-L150) ---------------------------------------------------------------------
    @classmethod
    def fit(cls, chunks: Sequence[np.ndarray], scale: float = DEFAULT_SCALE, vocab_size: int = DEFAULT_VOCAB_SIZE, *, min_frequency: int = 2, time_horizon: int | None = None, action_dim: int | None = None) -> "FASTTokenizer":
        """Train a new tokenizer on a list of [H_i, D] normalized chunks (lengths may differ).
        min_token = min over all quantized coefficients; the alphabet must fit in vocab_size (fast_hf L112-L118)."""
        qs = [quantize(dct(np.asarray(c, dtype=np.float64), axis=0), scale).reshape(-1) for c in chunks]
        allq = np.concatenate(qs)
        min_token, max_token = int(allq.min()), int(allq.max())
        min_vocab = max_token - min_token
        assert min_vocab <= vocab_size, f"vocab_size {vocab_size} too small for the coefficient range {min_vocab}"
        texts = [ints_to_text(q, min_token) for q in qs]
        bpe = BPE.train(texts, vocab_size=vocab_size, min_frequency=min_frequency)
        return cls(bpe, scale=scale, min_token=min_token, time_horizon=time_horizon, action_dim=action_dim)

    @classmethod
    def from_hf_dir(cls, directory: str | pathlib.Path = FAST_PLUS_DIR) -> "FASTTokenizer":
        """Load the released FAST+ from a directory holding processor_config.json and tokenizer.json."""
        directory = pathlib.Path(directory)
        cfg = json.loads((directory / "processor_config.json").read_text())
        bpe = BPE.from_hf_json(directory / "tokenizer.json")
        return cls(bpe, scale=cfg["scale"], min_token=cfg["min_token"], time_horizon=cfg.get("time_horizon"), action_dim=cfg.get("action_dim"))


def fast_plus_available(directory: str | pathlib.Path = FAST_PLUS_DIR) -> bool:
    d = pathlib.Path(directory)
    return (d / "processor_config.json").exists() and (d / "tokenizer.json").exists()


# --------------------------------------------------------------------------------------
# Synthetic data for main / eval / tests: smooth trajectories (a few low-frequency cosines plus small noise),
# already in [-1, 1]. Stands in for a normalized 1-second chunk; NOT robot data.
# --------------------------------------------------------------------------------------
def make_smooth_chunks(n: int, horizon: int, dim: int, rng: np.random.Generator, noise: float = 0.01) -> np.ndarray:
    t = np.linspace(0.0, 1.0, horizon)[None, :, None]
    freq = rng.uniform(0.2, 1.5, size=(n, 1, dim, 3))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, dim, 3))
    amp = rng.uniform(0.1, 0.5, size=(n, 1, dim, 3))
    x = (amp * np.cos(2 * np.pi * freq * t[..., None] + phase)).sum(-1)
    x = x + rng.normal(0, noise, size=x.shape)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(0)
    H, D = 20, 4  # tiny: a 20 Hz robot with 4 action dims, 1-second chunks
    chunks = make_smooth_chunks(64, H, D, rng)
    print(f"fit on {len(chunks)} synthetic chunks of shape [{H}, {D}], scale={DEFAULT_SCALE}, vocab_size=320")
    tok = FASTTokenizer.fit(list(chunks), scale=DEFAULT_SCALE, vocab_size=320)
    print(f"  min_token={tok.min_token}  alphabet={len(BYTE_ALPHABET)} byte chars  merges={len(tok.bpe.merges)}  vocab={tok.vocab_size}")

    a = chunks[0]
    print(f"\nchunk a: {a.shape} {a.dtype}, range [{a.min():.2f}, {a.max():.2f}]")
    c = dct(a, axis=0)
    print(f"1. DCT-II along time -> C {c.shape}; |C| by frequency row (dim 0): {np.round(np.abs(c[:, 0]), 2).tolist()}")
    q = quantize(c, tok.scale)
    print(f"2. round(scale * C) -> Q {q.shape} int64; nonzero {int((q != 0).sum())}/{q.size}; first rows:\n{q[:4]}")
    flat = q.reshape(-1)
    print(f"3. flatten row-major -> {flat.shape}: {flat[:12].tolist()} ... (freq 0 of all {D} dims, then freq 1, ...)")
    text = ints_to_text(flat, tok.min_token)
    print(f"4. chr(q - min_token) -> str of {len(text)} chars; pretokenize -> {len(pretokenize(text))} words")
    ids = tok(a)[0]
    print(f"5. byte-level BPE -> {len(ids)} tokens (naive binning would be {H * D}): {ids[:16]} ...")
    rec = tok.decode([ids], time_horizon=H, action_dim=D)[0]
    print(f"\ndecode -> {rec.shape}; max |err| = {np.abs(rec - a).max():.4f} (bound 0.5/scale * sqrt(H) = {0.5 / tok.scale * math.sqrt(H):.4f}), MSE = {((rec - a) ** 2).mean():.2e}")

    if fast_plus_available():
        fp = FASTTokenizer.from_hf_dir()
        big = make_smooth_chunks(4, 50, 7, rng)  # a 50 Hz, 7-dim arm
        n_tok = [len(t) for t in fp(big)]
        rec = fp.decode(fp(big), time_horizon=50, action_dim=7)
        print(f"\nFAST+ (released): vocab {fp.vocab_size}, min_token {fp.min_token}; 50x7 chunks -> {n_tok} tokens (naive 350); max |err| {np.abs(rec - big).max():.4f}")
    else:
        print(f"\nFAST+ files not found in {FAST_PLUS_DIR}; download processor_config.json + tokenizer.json from HF to try it.")


if __name__ == "__main__":
    main()
