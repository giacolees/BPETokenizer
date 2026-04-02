"""Heap BPE: O(log N) best-pair lookup via a lazy max-heap; still scans all words."""

from __future__ import annotations

from tqdm import tqdm

from bpetokenizer._heap_compat import heapify_max, heappop_max, heappush_max


def heap_bpe(
    counts: dict[str, int],
    word_vocab: dict[str, tuple[int, ...]],
    tokens: dict[int, bytes],
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[str]],  # accepted but unused
    merge_start: int,
    num_merges: int,
) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Run BPE merges using a lazy max-heap for O(log N) best-pair selection.

    Stale heap entries (whose stored frequency no longer matches ``pair_counts``)
    are skipped on pop rather than removed eagerly.

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

    heap_pc: list[tuple[int, tuple[int, int]]] = [(v, k) for k, v in pair_counts.items()]
    heapify_max(heap_pc)

    for _ in tqdm(range(num_merges), desc="heap_bpe"):
        while heap_pc:
            val, (a, b) = heappop_max(heap_pc)
            if val == pair_counts[(a, b)] and val > 0:
                break
        else:
            break

        merge_ids.append((a, b))
        tokens[i] = tokens[a] + tokens[b]

        for token, freq in counts.items():
            seq = list(word_vocab[token])
            new_seq: list[int] = []
            j = 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == a and seq[j + 1] == b:
                    if new_seq:
                        left = new_seq[-1]
                        pair_counts[(left, a)] -= freq
                        pair_counts[(left, i)] += freq
                        heappush_max(heap_pc, (pair_counts[(left, a)], (left, a)))
                        heappush_max(heap_pc, (pair_counts[(left, i)], (left, i)))
                    if j + 2 < len(seq):
                        right = seq[j + 2]
                        pair_counts[(b, right)] -= freq
                        pair_counts[(i, right)] += freq
                        heappush_max(heap_pc, (pair_counts[(b, right)], (b, right)))
                        heappush_max(heap_pc, (pair_counts[(i, right)], (i, right)))
                    pair_counts[(a, b)] -= freq
                    heappush_max(heap_pc, (pair_counts[(a, b)], (a, b)))
                    new_seq.append(i)
                    j += 2
                else:
                    new_seq.append(seq[j])
                    j += 1
            word_vocab[token] = tuple(new_seq)
        i += 1

    return merge_ids, tokens
