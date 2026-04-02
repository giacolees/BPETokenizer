"""Internal helpers: vocab initialisation and model persistence."""

from __future__ import annotations

import pickle
from collections import defaultdict

from ._constants import BASE_VOCAB_SIZE


def _build_index(
    counts: dict[str, int],
    word_vocab: dict[str, tuple[int, ...]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[str]]]:
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for token, freq in counts.items():
        seq = word_vocab[token]
        for a, b in zip(seq, seq[1:]):
            pair_counts[(a, b)] += freq

    pair_to_words: dict[tuple[int, int], set[str]] = defaultdict(set)
    for word, seq in word_vocab.items():
        for j in range(len(seq) - 1):
            pair_to_words[(seq[j], seq[j + 1])].add(word)

    return pair_counts, pair_to_words


def init_from_counts(
    counts: dict[str, int],
    special_tokens: list[str] | None = None,
) -> tuple[
    dict[str, int],
    dict[str, tuple[int, ...]],
    dict[int, bytes],
    dict[tuple[int, int], int],
    dict[tuple[int, int], set[str]],
]:
    """Initialise all data structures required by BPE training from word-frequency *counts*.

    Returns
    -------
    counts
        Unchanged word-frequency mapping.
    word_vocab
        Maps each word string to its current token-ID sequence (mutated during training).
    tokens
        Vocabulary table ``{id: bytes}``, seeded with 256 single-byte entries plus any
        special tokens.
    pair_counts
        Aggregate bigram frequencies across the corpus.
    pair_to_words
        Inverted index mapping each bigram to the set of words that contain it.
    """
    if special_tokens is None:
        special_tokens = []

    tokens: dict[int, bytes] = {j: bytes([j]) for j in range(BASE_VOCAB_SIZE)}
    for idx, st in enumerate(special_tokens):
        tokens[BASE_VOCAB_SIZE + idx] = st.encode("utf-8")

    word_vocab: dict[str, tuple[int, ...]] = {word: tuple(word.encode("utf-8")) for word in counts}
    pair_counts, pair_to_words = _build_index(counts, word_vocab)
    return counts, word_vocab, tokens, pair_counts, pair_to_words


def load_counts(
    path: str,
    special_tokens: list[str] | None = None,
) -> tuple[
    dict[str, int],
    dict[str, tuple[int, ...]],
    dict[int, bytes],
    dict[tuple[int, int], int],
    dict[tuple[int, int], set[str]],
]:
    """Load pre-tokenized counts from *path* and initialise BPE data structures."""
    with open(path, "rb") as f:
        counts: dict[str, int] = pickle.load(f)
    return init_from_counts(counts, special_tokens)


def save_model(path: str, merges: list[tuple[int, int]], tokens: dict[int, bytes]) -> None:
    """Persist *merges* and *tokens* to *path* as a pickle file."""
    with open(path, "wb") as f:
        pickle.dump({"merges": merges, "tokens": tokens}, f)
    print(f"Saved model: {len(merges)} merges, {len(tokens)} tokens → {path}")
