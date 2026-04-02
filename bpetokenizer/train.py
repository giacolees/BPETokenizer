"""High-level BPE training pipeline."""

from __future__ import annotations

import os

from ._constants import BASE_VOCAB_SIZE
from ._utils import init_from_counts
from .algorithms.indexed_heap import indexed_heap_bpe
from .pretokenize import pretokenize


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on *input_path* and return the vocabulary and merge list.

    Parameters
    ----------
    input_path:
        Path to a UTF-8 encoded text corpus.
    vocab_size:
        Target vocabulary size (must be > 256 + len(special_tokens)).
    special_tokens:
        Strings to treat as atomic tokens.  They are assigned IDs
        256, 257, … before any merge IDs.

    Returns
    -------
    tokens
        Vocabulary table ``{id: bytes}``.
    merges
        Ordered list of ``(bytes_a, bytes_b)`` merge pairs.
    """
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    counts = pretokenize(text, special_tokens)
    counts, word_vocab, tokens, pair_counts, pair_to_words = init_from_counts(
        counts, special_tokens
    )

    merge_start = BASE_VOCAB_SIZE + len(special_tokens)
    num_merges = vocab_size - merge_start

    merge_ids, tokens = indexed_heap_bpe(
        counts, word_vocab, tokens, pair_counts, pair_to_words, merge_start, num_merges
    )

    merges: list[tuple[bytes, bytes]] = [(tokens[a], tokens[b]) for a, b in merge_ids]
    return tokens, merges
