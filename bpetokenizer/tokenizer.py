"""Inference-time BPE tokenizer: encode and decode text."""

from __future__ import annotations

import re as stdlib_re
from collections.abc import Iterable

import regex

# GPT-2 pre-tokenization pattern (must match the one used during training)
_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """BPE tokenizer built from a trained vocabulary and merge list.

    Parameters
    ----------
    vocab:
        Mapping from token ID to raw bytes, as returned by :func:`train_bpe`.
    merges:
        Ordered list of ``(bytes_a, bytes_b)`` merge pairs, as returned by
        :func:`train_bpe`.  Earlier entries have higher priority.
    special_tokens:
        Optional list of special token strings (e.g. ``["<|endoftext|>"]``).
        These are encoded as atomic units rather than being split by the regex.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: idx for idx, pair in enumerate(merges)
        }
        self.special_tokens: list[str] = special_tokens or []
        self._special_set: set[str] = set(self.special_tokens)

        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(stdlib_re.escape(s) for s in sorted_specials)
            self._special_pattern: stdlib_re.Pattern[str] | None = stdlib_re.compile(f"({pattern})")
        else:
            self._special_pattern = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_chunk(self, word_bytes: bytes) -> list[int]:
        """BPE-encode a single pre-tokenized word given as raw bytes."""
        tokens = [bytes([b]) for b in word_bytes]
        while len(tokens) > 1:
            best_rank = float("inf")
            best_idx = -1
            for idx in range(len(tokens) - 1):
                pair = (tokens[idx], tokens[idx + 1])
                rank = self.merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = idx
            if best_rank == float("inf"):
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2 :]
        return [self.bytes_to_id[t] for t in tokens]

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of token IDs."""
        ids: list[int] = []
        parts = self._special_pattern.split(text) if self._special_pattern else [text]
        for part in parts:
            if not part:
                continue
            if part in self._special_set:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                for match in regex.finditer(_PAT, part):
                    ids.extend(self._encode_chunk(match.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Memory-efficient encoding: yield token IDs one line at a time."""
        for line in iterable:
            yield from self.encode(line)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
