"""Compare the four BPE algorithm implementations on a pre-tokenized corpus.

Requires ``data/pretokenized_counts.pkl`` (produced by
``scripts/pretokenize_corpus.py``).  Outputs ``benchmark.png``.

Usage
-----
    uv run python benchmark.py
"""

import time

import matplotlib.pyplot as plt

from bpetokenizer._utils import load_counts
from bpetokenizer.algorithms import heap_bpe, indexed_bpe, indexed_heap_bpe, naive_bpe

NUM_MERGES = 5000
COUNTS_PATH = "data/pretokenized_counts.pkl"

CONFIGS = [
    ("Naive", naive_bpe),
    ("Heap", heap_bpe),
    ("Inverted Index", indexed_bpe),
    ("Inverted Index + Heap", indexed_heap_bpe),
]


def main() -> None:
    times: list[float] = []
    labels: list[str] = []

    for label, fn in CONFIGS:
        print(f"Running {label}...")
        counts, word_vocab, tokens, pair_counts, pair_to_words = load_counts(COUNTS_PATH)
        t0 = time.perf_counter()
        fn(counts, word_vocab, tokens, pair_counts, pair_to_words, 256, NUM_MERGES)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        labels.append(label)
        print(f"  {label}: {elapsed:.2f}s")

    naive_time = times[0]
    speedups = [naive_time / t for t in times]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, times, color="steelblue", alpha=0.8)
    ax.set_ylabel("Total time (s)")
    ax.set_title(f"Total processing time — {NUM_MERGES} merges")
    ax.tick_params(axis="x", rotation=15)
    for bar, speedup in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="steelblue",
        )
    fig.tight_layout()
    fig.savefig("benchmark.png", dpi=150)
    print("Saved benchmark.png")
    plt.show()


if __name__ == "__main__":
    main()
