# BPETokenizer

A from-scratch implementation of Byte Pair Encoding (BPE) tokenization in Python, featuring four progressively optimised training algorithms and a full end-to-end pipeline from raw text to encoded tokens.

## Overview

BPE tokenization works by iteratively merging the most frequent adjacent byte pair in a corpus until a target vocabulary size is reached. This repo implements the algorithm four times — each version fixing a bottleneck of the previous one — to make the optimisation story concrete and measurable.

| Algorithm | Best-pair selection | Word scan | Relative speed |
|---|---|---|---|
| **Naive** | `max()` over all pairs | All words every step | 1× (baseline) |
| **Heap** | Lazy max-heap, O(log N) | All words every step | ~2–3× |
| **Inverted Index** | `max()` over all pairs | Only words containing the pair | ~5–10× |
| **Inverted Index + Heap** | Lazy max-heap, O(log N) | Only words containing the pair | ~15–30× |

The **Inverted Index + Heap** algorithm is used by the public `train_bpe` API.

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<you>/BPETokenizer.git
cd BPETokenizer
uv sync
```

## Quick start

```python
from bpetokenizer import Tokenizer, train_bpe

# Train on any plain-text corpus
tokens, merges = train_bpe("corpus.txt", vocab_size=10_000, special_tokens=["<|endoftext|>"])

# Encode and decode
tokenizer = Tokenizer(tokens, merges, special_tokens=["<|endoftext|>"])
ids = tokenizer.encode("Hello, world! <|endoftext|>")
text = tokenizer.decode(ids)
assert text == "Hello, world! <|endoftext|>"

# Memory-efficient encoding of large files
with open("corpus.txt") as f:
    for token_id in tokenizer.encode_iterable(f):
        ...
```

## Export to JSON (web demo)

```bash
uv run python scripts/export_json.py corpus.txt 2000 tokenizer.json
```

## Large-corpus pre-tokenization (parallel)

For multi-GB corpora, pre-tokenize in parallel and save the word-frequency counts:

```bash
uv run python scripts/pretokenize_corpus.py \
    data/TinyStoriesV2-GPT4-train.txt \
    data/pretokenized_counts.pkl \
    --special-tokens "<|endoftext|>" \
    --num-processes 16
```

## Benchmarking

```bash
# Requires data/pretokenized_counts.pkl (see above)
uv run python benchmark.py
```

Saves a bar chart to `benchmark.png` with per-algorithm timing and speedup labels.

## Development

```bash
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Project structure

```
bpetokenizer/           # installable package
├── __init__.py         # public API: Tokenizer, train_bpe
├── tokenizer.py        # Tokenizer class (encode / decode)
├── train.py            # train_bpe() pipeline
├── pretokenize.py      # GPT-2 regex pre-tokenization + parallel helpers
├── _utils.py           # vocab initialisation and model persistence
├── _constants.py       # BASE_VOCAB_SIZE = 256
└── algorithms/         # four BPE implementations
    ├── naive.py
    ├── heap.py
    ├── indexed.py
    └── indexed_heap.py
scripts/                # CLI entry points
tests/                  # pytest suite
benchmark.py            # algorithm comparison
```

## How it works

**Pre-tokenization** splits the raw text on special tokens, then applies the GPT-2 Unicode-aware regex to produce a `Counter` of word-string frequencies.

**Initialisation** seeds a vocabulary table with 256 single-byte entries (one per possible byte value), appends special tokens at IDs 256+, and builds `pair_counts` (aggregate bigram frequencies) and `pair_to_words` (inverted index: bigram → set of words containing it).

**Training** runs `num_merges = vocab_size − 256 − len(special_tokens)` iterations.  Each iteration picks the highest-frequency bigram, creates a new token whose bytes are the concatenation of the two parents, and updates `pair_counts` and `pair_to_words` in place.

**Inference** (`Tokenizer.encode`) splits on special tokens, applies the GPT-2 regex, then BPE-encodes each chunk greedily by merge rank.
