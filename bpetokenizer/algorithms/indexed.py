"""Indexed BPE: inverted index limits each merge step to only affected words."""

from __future__ import annotations

from tqdm import tqdm


def indexed_bpe(
    counts: dict[str, int],
    word_vocab: dict[str, tuple[int, ...]],
    tokens: dict[int, bytes],
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[str]],
    merge_start: int,
    num_merges: int,
) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Run BPE merges using an inverted index to avoid scanning the full corpus.

    ``pair_to_words`` maps each bigram to the set of words that currently contain
    it.  Only those words are visited when a pair is merged, and the index is
    kept in sync after each update.

    Parameters
    ----------
    counts:
        Word-frequency mapping (not mutated).
    word_vocab:
        Current token-ID sequence per word (mutated in-place).
    tokens:
        Vocabulary table ``{id: bytes}`` (mutated in-place).
    pair_counts:
        Aggregate bigram frequencies (mutated in-place).
    pair_to_words:
        Inverted index (mutated in-place to stay in sync).
    merge_start:
        ID to assign to the first merged token.
    num_merges:
        Number of merge operations to perform.

    Returns
    -------
    merge_ids
        Ordered list of ``(id_a, id_b)`` merge pairs.
    tokens
        Updated vocabulary table.
    """
    i = merge_start
    merge_ids: list[tuple[int, int]] = []

    for _ in tqdm(range(num_merges), desc="indexed_bpe"):
        if not any(v > 0 for v in pair_counts.values()):
            break
        a, b = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merge_ids.append((a, b))
        tokens[i] = tokens[a] + tokens[b]

        for token in list(pair_to_words[(a, b)]):
            freq = counts[token]
            seq = list(word_vocab[token])
            new_seq: list[int] = []
            j = 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                    if new_seq:
                        pair_counts[(new_seq[-1], a)] -= freq
                        pair_counts[(new_seq[-1], i)] += freq
                    if j + 2 < len(seq):
                        pair_counts[(b, seq[j + 2])] -= freq
                        pair_counts[(i, seq[j + 2])] += freq
                    pair_counts[(a, b)] -= freq
                    new_seq.append(i)
                    j += 2
                else:
                    new_seq.append(seq[j])
                    j += 1

            for k in range(len(seq) - 1):
                pair_to_words[(seq[k], seq[k + 1])].discard(token)
            for k in range(len(new_seq) - 1):
                pair_to_words[(new_seq[k], new_seq[k + 1])].add(token)
            word_vocab[token] = tuple(new_seq)
        i += 1

    return merge_ids, tokens
