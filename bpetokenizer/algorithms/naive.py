"""Naive BPE: O(V × M) — scans all words every merge step."""

from __future__ import annotations

from tqdm import tqdm


def naive_bpe(
    counts: dict[str, int],
    word_vocab: dict[str, tuple[int, ...]],
    tokens: dict[int, bytes],
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[str]],  # accepted but unused
    merge_start: int,
    num_merges: int,
) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Run BPE merges using a naive full-corpus scan on each step.

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
        Inverted index (accepted for interface compatibility; not used).
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

    for _ in tqdm(range(num_merges), desc="naive_bpe"):
        if not any(v > 0 for v in pair_counts.values()):
            break
        a, b = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merge_ids.append((a, b))
        tokens[i] = tokens[a] + tokens[b]

        for token, freq in counts.items():
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
            word_vocab[token] = tuple(new_seq)
        i += 1

    return merge_ids, tokens
