"""Tokenizer evaluation: tokens per chunk (compression) and reconstruction error versus the rounding scale gamma.

This is the paper's Appendix B / Fig. 12 trade-off and Table I's "avg token count", on synthetic smooth chunks
(no robot data is downloaded). Numbers are not the paper's; the shape of the curve is the point.
Run: uv run python -m pi.fast.tokenizer.eval
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pi.fast.tokenizer.tokenizer import FASTTokenizer, fast_plus_available, make_smooth_chunks


@dataclasses.dataclass
class TokenizerMetrics:
    scale: float
    vocab_size: int
    tokens_per_chunk: float  # mean over chunks; naive binning = H * D
    naive_tokens: int
    mse: float  # mean squared reconstruction error in normalized space
    max_abs_err: float


def evaluate_tokenizer(tok: FASTTokenizer, chunks: np.ndarray) -> TokenizerMetrics:
    """chunks float[N, H, D] -> metrics. Uses on_error="raise": an evaluation must not silently score zeros."""
    N, H, D = chunks.shape
    ids = tok(chunks)
    rec = tok.decode(ids, time_horizon=H, action_dim=D, on_error="raise")
    err = rec - chunks
    return TokenizerMetrics(tok.scale, tok.vocab_size, float(np.mean([len(t) for t in ids])), H * D, float((err**2).mean()), float(np.abs(err).max()))


def sweep_scale(train: list[np.ndarray], test: np.ndarray, scales=(1, 2, 5, 10, 20, 50), vocab_size: int = 1024) -> list[TokenizerMetrics]:
    """Refit a tokenizer per gamma (as the paper does for Fig. 12) and evaluate on held-out chunks."""
    return [evaluate_tokenizer(FASTTokenizer.fit(train, scale=s, vocab_size=vocab_size), test) for s in scales]


def print_table(rows: list[TokenizerMetrics], title: str) -> None:
    print(f"\n{title}")
    print(f"{'scale':>6} {'vocab':>6} {'tokens/chunk':>13} {'naive':>6} {'ratio':>6} {'MSE':>10} {'max|err|':>9}")
    for r in rows:
        print(f"{r.scale:>6g} {r.vocab_size:>6d} {r.tokens_per_chunk:>13.1f} {r.naive_tokens:>6d} {r.naive_tokens / r.tokens_per_chunk:>6.1f} {r.mse:>10.2e} {r.max_abs_err:>9.4f}")


def main() -> None:
    rng = np.random.default_rng(0)
    H, D = 50, 7  # a 50 Hz single arm, like the paper's UR5 bussing rows in Table I
    train = list(make_smooth_chunks(64, H, D, rng))
    test = make_smooth_chunks(32, H, D, rng)
    print_table(sweep_scale(train, test), f"FAST fit per scale on {len(train)} synthetic [{H}, {D}] chunks, evaluated on {len(test)} held-out chunks")
    print("paper Table I (gamma = 10, dataset-specific fit): Bussing 7-dim 20 Hz -> 28 tokens (naive 140); Shirt Fold 14-dim 50 Hz -> 53 (naive 700)")
    if fast_plus_available():
        fp = FASTTokenizer.from_hf_dir()
        print_table([evaluate_tokenizer(fp, test)], "released FAST+ (vocab 2048, min_token -354) on the same held-out chunks")


if __name__ == "__main__":
    main()
