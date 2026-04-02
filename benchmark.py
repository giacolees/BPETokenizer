"""Comprehensive benchmark for all four BPE algorithm implementations.

Measures wall time at five merge counts, generates three plots, and prints a
summary table with speedups.  Results are cached to ``data/benchmark_cache.json``
so individual algorithm runs can be resumed without re-running everything.

Usage
-----
    uv run python benchmark.py [--merges N [N ...]] [--no-cache]

Outputs
-------
    data/benchmark_cache.json   raw timing data
    benchmark_time.png          time vs. merge count (log-log)
    benchmark_speedup.png       speedup vs. Naive at every merge count
    benchmark_throughput.png    throughput (merges / second)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from bpetokenizer._utils import load_counts
from bpetokenizer.algorithms import heap_bpe, indexed_bpe, indexed_heap_bpe, naive_bpe

COUNTS_PATH = "data/pretokenized_counts.pkl"
CACHE_PATH = "data/benchmark_cache.json"
ASSETS_DIR = "assets"
SPECIAL = ["<|endoftext|>"]
MERGE_START = 257

DEFAULT_MERGES = [500, 1000, 2000, 5000, 9743]

ALGORITHMS = [
    ("Naive", naive_bpe),
    ("Heap", heap_bpe),
    ("Inverted Index", indexed_bpe),
    ("Inverted Index + Heap", indexed_heap_bpe),
]

COLORS = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
MARKERS = ["o", "s", "^", "D"]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_once(fn, num_merges: int) -> float:
    counts, word_vocab, tokens, pair_counts, pair_to_words = load_counts(
        COUNTS_PATH, special_tokens=SPECIAL
    )
    t0 = time.perf_counter()
    fn(counts, word_vocab, tokens, pair_counts, pair_to_words, MERGE_START, num_merges)
    return time.perf_counter() - t0


def run_benchmark(merge_counts: list[int], cache: dict, no_cache: bool) -> dict:
    for label, fn in ALGORITHMS:
        if label not in cache:
            cache[label] = {}
        for n in merge_counts:
            key = str(n)
            if not no_cache and key in cache[label]:
                print(f"  [{label}] {n:>5} merges — cached ({cache[label][key]:.2f}s)")
                continue
            print(f"  [{label}] {n:>5} merges … ", end="", flush=True)
            t = run_once(fn, n)
            cache[label][key] = t
            print(f"{t:.2f}s")
            Path(CACHE_PATH).write_text(json.dumps(cache, indent=2))
    return cache


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_time(cache: dict, merge_counts: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, _), color, marker in zip(ALGORITHMS, COLORS, MARKERS):
        xs = [n for n in merge_counts if str(n) in cache.get(label, {})]
        ys = [cache[label][str(n)] for n in xs]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2, markersize=7, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of merges", fontsize=12)
    ax.set_ylabel("Wall time (s)", fontsize=12)
    ax.set_title("BPE training time vs. merge count", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(merge_counts)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/benchmark_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved {ASSETS_DIR}/benchmark_time.png")


def plot_speedup(cache: dict, merge_counts: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    naive_times = cache.get("Naive", {})

    for (label, _), color, marker in zip(ALGORITHMS, COLORS, MARKERS):
        xs, ys = [], []
        for n in merge_counts:
            key = str(n)
            if key in cache.get(label, {}) and key in naive_times:
                xs.append(n)
                ys.append(naive_times[key] / cache[label][key])
        linestyle = ":" if label == "Naive" else "-"
        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=7,
            label=label,
            linestyle=linestyle,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Number of merges", fontsize=12)
    ax.set_ylabel("Speedup vs. Naive  (higher is better)", fontsize=12)
    ax.set_title("Speedup relative to the Naive algorithm", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(merge_counts)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/benchmark_speedup.png", dpi=150)
    plt.close(fig)
    print(f"Saved {ASSETS_DIR}/benchmark_speedup.png")


def plot_throughput(cache: dict, merge_counts: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, _), color, marker in zip(ALGORITHMS, COLORS, MARKERS):
        xs, ys = [], []
        for n in merge_counts:
            key = str(n)
            if key in cache.get(label, {}):
                xs.append(n)
                ys.append(n / cache[label][key])
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2, markersize=7, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of merges", fontsize=12)
    ax.set_ylabel("Throughput (merges / second)", fontsize=12)
    ax.set_title("BPE training throughput vs. merge count", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(merge_counts)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/benchmark_throughput.png", dpi=150)
    plt.close(fig)
    print(f"Saved {ASSETS_DIR}/benchmark_throughput.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(cache: dict, merge_counts: list[int]) -> None:
    labels = [lbl for lbl, _ in ALGORITHMS]
    col = 14

    print("\n" + "=" * 70)
    print("WALL TIME (seconds)")
    print("=" * 70)
    header = f"{'Merges':<8}" + "".join(f"{lbl:>{col}}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for n in merge_counts:
        row = f"{n:<8}"
        for lbl in labels:
            t = cache.get(lbl, {}).get(str(n))
            row += f"{t:>{col - 1}.2f}s" if t is not None else f"{'—':>{col}}"
        print(row)

    print("\n" + "=" * 70)
    print("SPEEDUP vs. NAIVE")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for n in merge_counts:
        ref = cache.get("Naive", {}).get(str(n))
        if ref is None:
            continue
        row = f"{n:<8}"
        for lbl in labels:
            t = cache.get(lbl, {}).get(str(n))
            row += f"{ref / t:>{col - 1}.1f}×" if t is not None else f"{'—':>{col}}"
        print(row)

    print("\n" + "=" * 70)
    print("THROUGHPUT (merges / second)")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for n in merge_counts:
        row = f"{n:<8}"
        for lbl in labels:
            t = cache.get(lbl, {}).get(str(n))
            row += f"{n / t:>{col - 1}.0f} " if t is not None else f"{'—':>{col}}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="BPE algorithm benchmark")
    parser.add_argument(
        "--merges",
        nargs="+",
        type=int,
        default=DEFAULT_MERGES,
        metavar="N",
        help="Merge counts to benchmark",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore cached results and re-run everything"
    )
    args = parser.parse_args()

    merge_counts = sorted(args.merges)

    cache: dict = {}
    if not args.no_cache and Path(CACHE_PATH).exists():
        cache = json.loads(Path(CACHE_PATH).read_text())
        print(f"Loaded cache from {CACHE_PATH}")

    print(f"\nRunning benchmark: {merge_counts} merges\n")
    cache = run_benchmark(merge_counts, cache, args.no_cache)

    print("\nGenerating plots …")
    Path(ASSETS_DIR).mkdir(exist_ok=True)
    plot_time(cache, merge_counts)
    plot_speedup(cache, merge_counts)
    plot_throughput(cache, merge_counts)

    print_summary(cache, merge_counts)


if __name__ == "__main__":
    main()
