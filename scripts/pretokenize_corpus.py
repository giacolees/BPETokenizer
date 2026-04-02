#!/usr/bin/env python3
"""Parallel pre-tokenization of a large text corpus.

Splits the file into chunks at special-token boundaries, counts word
frequencies in parallel, and saves the result as a pickle file.

Usage
-----
    uv run python scripts/pretokenize_corpus.py <input> <output> [options]

Example
-------
    uv run python scripts/pretokenize_corpus.py \\
        data/TinyStoriesV2-GPT4-train.txt data/pretokenized_counts.pkl \\
        --special-tokens "<|endoftext|>" --num-processes 16
"""

import argparse
import pickle
from collections import Counter
from multiprocessing import Pool

from bpetokenizer.pretokenize import find_chunk_boundaries, process_chunk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a corpus and save word-frequency counts."
    )
    parser.add_argument("input", help="Path to the input text file.")
    parser.add_argument("output", help="Path for the output .pkl file.")
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        metavar="TOKEN",
        help="Special tokens to split on (default: '<|endoftext|>').",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        metavar="N",
        help="Number of worker processes (default: 16).",
    )
    args = parser.parse_args()

    split_token = args.special_tokens[0].encode() if args.special_tokens else b""

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_processes, split_token)

    chunk_args = [
        (args.input, start, end, args.special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    counts: Counter[str] = Counter()
    with Pool(args.num_processes) as pool:
        for partial in pool.map(process_chunk, chunk_args):
            counts.update(partial)

    with open(args.output, "wb") as f:
        pickle.dump(counts, f)

    print(f"Saved {len(counts):,} unique word types to '{args.output}'.")


if __name__ == "__main__":
    main()
