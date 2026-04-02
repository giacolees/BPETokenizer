#!/usr/bin/env python3
"""Train a BPE tokenizer and export it to JSON for the web demo.

Usage
-----
    uv run python scripts/export_json.py <input_text> [vocab_size] [output_json]

Example
-------
    uv run python scripts/export_json.py corpus.txt 2000 tokenizer.json
"""

import json
import sys
from pathlib import Path

from bpetokenizer import train_bpe


def export_tokenizer_json(
    input_path: str,
    vocab_size: int = 2000,
    output_path: str = "tokenizer.json",
    special_tokens: list[str] | None = None,
) -> None:
    """Train BPE on *input_path* and write the result to *output_path* as JSON.

    The JSON schema is:
    ``{"startId": int, "vocab": {"<id>": [byte, ...]}, "merges": [[idA, idB], ...]}``
    """
    if special_tokens is None:
        special_tokens = []

    print(f"Training BPE tokenizer on '{input_path}' (vocab_size={vocab_size})...")
    tokens, merges = train_bpe(input_path, vocab_size, special_tokens)

    bytes_to_id: dict[bytes, int] = {v: k for k, v in tokens.items()}
    vocab_json = {str(id_): list(b) for id_, b in tokens.items()}
    merges_json = [[bytes_to_id[ba], bytes_to_id[bb]] for ba, bb in merges]

    data = {
        "startId": 256 + len(special_tokens),
        "vocab": vocab_json,
        "merges": merges_json,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = Path(output_path).stat().st_size / 1024
    print(
        f"Exported {len(tokens)} tokens and {len(merges)} merges "
        f"to '{output_path}' ({size_kb:.1f} KB)."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    _input_path = sys.argv[1]
    _vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    _output_path = sys.argv[3] if len(sys.argv) > 3 else "tokenizer.json"

    export_tokenizer_json(_input_path, _vocab_size, _output_path)
