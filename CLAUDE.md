# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra dev          # install all dependencies including dev tools
uv run pytest tests/ -v      # run test suite
uv run pytest tests/test_tokenizer.py::TestTokenizer  # run a single class
uv run ruff check .          # lint
uv run ruff format .         # format
uv run python benchmark.py   # algorithm comparison (needs data/pretokenized_counts.pkl)
uv run python scripts/export_json.py corpus.txt 2000 tokenizer.json
uv run python scripts/pretokenize_corpus.py data/corpus.txt data/counts.pkl
```

## Architecture

The package is `bpetokenizer/`. Public API: `Tokenizer` and `train_bpe` (both exported from `__init__.py`).

### Data flow

1. **Pre-tokenization** (`pretokenize.py`): Splits text on special tokens, applies the GPT-2 Unicode regex (`PAT`) to produce a `Counter[str]` of word-string frequencies. `process_chunk` is the multiprocessing worker used by `scripts/pretokenize_corpus.py`.

2. **Initialisation** (`_utils.py` → `init_from_counts`): Converts the word-frequency counter into five data structures passed into every BPE algorithm:
   - `counts`: `{word: freq}` — never mutated during training
   - `word_vocab`: `{word: tuple[int, ...]}` — current token-ID sequence per word; mutated each merge
   - `tokens`: `{id: bytes}` — vocabulary table; seeded with 256 single-byte entries (`_constants.BASE_VOCAB_SIZE`) then special tokens at IDs 256+
   - `pair_counts`: `{(a, b): freq}` — aggregate bigram frequencies; mutated each merge
   - `pair_to_words`: `{(a, b): set[word]}` — inverted index; mutated each merge (indexed algorithms only)

3. **BPE training** (`algorithms/`): Each algorithm accepts the same 7-argument signature and returns `(merge_ids, tokens)`. All stop early if no more positive-frequency pairs remain (guard against small corpora).

4. **Inference** (`tokenizer.py`): `Tokenizer.__init__` builds a `merge_rank` dict `{(bytes_a, bytes_b): index}`. Encoding splits on special tokens, applies the GPT-2 regex, then BPE-encodes each chunk greedily by merge rank.

5. **`train_bpe`** (`train.py`): Orchestrates steps 1–3 using `indexed_heap_bpe` and converts integer merge IDs to `(bytes, bytes)` pairs for the `Tokenizer`.

### BPE algorithms (`bpetokenizer/algorithms/`)

| Module | Strategy | Best-pair | Word scan |
|---|---|---|---|
| `naive.py` | Baseline | `max()` O(N) | All words |
| `heap.py` | Lazy max-heap | O(log N) | All words |
| `indexed.py` | Inverted index | `max()` O(N) | Words containing pair only |
| `indexed_heap.py` | Heap + index | O(log N) | Words containing pair only |

The heap uses **lazy deletion**: stale entries are pushed when `pair_counts` updates; on pop, entries are skipped until `val == pair_counts[(a, b)] and val > 0`.

### Key constants

- `BASE_VOCAB_SIZE = 256` — single-byte alphabet size; merge IDs start at `256 + len(special_tokens)`
- Special tokens occupy IDs `256, 257, …` before any merge tokens

## Data directory

The `data/` directory is gitignored. Scripts expect:
- `data/<corpus>.txt` — raw training text
- `data/pretokenized_counts.pkl` — output of `scripts/pretokenize_corpus.py`; input to `benchmark.py` and individual algorithm `__main__` blocks
