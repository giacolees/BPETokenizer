"""Pre-tokenization: split raw text into word-frequency counts using the GPT-2 regex."""

from __future__ import annotations

import os
import re as stdlib_re
from collections import Counter
from typing import BinaryIO

import regex

# GPT-2 pre-tokenization pattern (Unicode-aware via the `regex` library)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Return byte offsets that divide *file* into roughly equal chunks.

    Boundaries are snapped forward to the next occurrence of *split_special_token*
    so that no special token is split across two chunks.  Fewer chunks than
    *desired_num_chunks* may be returned if boundaries collapse.
    """
    assert isinstance(split_special_token, bytes), "split_special_token must be bytes"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize(text: str, special_tokens: list[str]) -> Counter[str]:
    """Return a ``Counter`` of pre-tokenized word frequencies.

    Special tokens are used only as split boundaries and are not counted.
    """
    special_set = set(special_tokens)
    counts: Counter[str] = Counter()
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pat = "|".join(stdlib_re.escape(s) for s in sorted_specials)
        segments = stdlib_re.split(f"({split_pat})", text)
    else:
        segments = [text]
    for seg in segments:
        if seg not in special_set:
            counts.update(match.group() for match in regex.finditer(PAT, seg))
    return counts


def process_chunk(args: tuple[str, int, int, list[str]]) -> Counter[str]:
    """Worker function for parallel pre-tokenization (used with ``multiprocessing.Pool``)."""
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as cf:
        cf.seek(start)
        chunk = cf.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize(chunk, special_tokens)
