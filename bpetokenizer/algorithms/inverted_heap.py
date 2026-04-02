"""Inverted-heap BPE: heap + inverted index — the production-grade algorithm."""

from __future__ import annotations

from tqdm import tqdm

from bpetokenizer._heap_compat import heapify_max, heappop_max, heappush_max


def inverted_heap_bpe(
    counts: dict[str, int],
    word_vocab: dict[str, tuple[int, ...]],
    tokens: dict[int, bytes],
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[str]],
    merge_start: int,
    num_merges: int,
) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Run BPE merges using a lazy max-heap and an inverted index.

    Combines two optimisations:

    * **Heap** — O(log N) best-pair selection with lazy deletion of stale entries.
    * **Inverted index** — each merge only visits the words that actually contain
      the merged pair, avoiding a full-corpus scan.

    This is the algorithm used by :func:`bpetokenizer.train_bpe`.

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

    heap_pc: list[tuple[int, tuple[int, int]]] = [(v, k) for k, v in pair_counts.items()]
    heapify_max(heap_pc)

    for _ in tqdm(range(num_merges), desc="inverted_heap_bpe"):
        # Find the highest-frequency non-stale pair; stop early if none remain.
        while heap_pc:
            val, (a, b) = heappop_max(heap_pc)
            if val == pair_counts[(a, b)] and val > 0:
                break
        else:
            break

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

            for k in range(len(seq) - 1):
                pair_to_words[(seq[k], seq[k + 1])].discard(token)
            for k in range(len(new_seq) - 1):
                pair_to_words[(new_seq[k], new_seq[k + 1])].add(token)
            word_vocab[token] = tuple(new_seq)
        i += 1

    return merge_ids, tokens
