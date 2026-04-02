"""Tests for the BPE tokenizer."""

from __future__ import annotations

import os
import tempfile

import pytest

from bpetokenizer import Tokenizer, train_bpe
from bpetokenizer._constants import BASE_VOCAB_SIZE
from bpetokenizer.pretokenize import pretokenize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_corpus(text: str, repeat: int = 20) -> str:
    """Return *text* repeated enough times for BPE to find meaningful merges."""
    return (text + "\n") * repeat


def _train_small(text: str, vocab_size: int = 300, special_tokens: list[str] | None = None):
    special_tokens = special_tokens or []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    try:
        return train_bpe(path, vocab_size, special_tokens)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# pretokenize
# ---------------------------------------------------------------------------


class TestPretokenize:
    def test_counts_words(self):
        # The GPT-2 regex treats space-prefixed words as distinct tokens:
        # "hello world hello" → "hello", " world", " hello"
        counts = pretokenize("hello world hello", special_tokens=[])
        assert counts["hello"] == 1
        assert counts[" world"] == 1
        assert counts[" hello"] == 1

    def test_special_tokens_excluded(self):
        counts = pretokenize("hello <|end|> world", special_tokens=["<|end|>"])
        assert "<|end|>" not in counts
        assert counts["hello"] == 1
        assert counts[" world"] == 1

    def test_empty_string(self):
        counts = pretokenize("", special_tokens=[])
        assert len(counts) == 0


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        corpus = _tiny_corpus("the quick brown fox jumps over the lazy dog")
        tokens, merges = _train_small(corpus, vocab_size=300)
        return Tokenizer(tokens, merges)

    def test_encode_returns_ints(self, tokenizer):
        ids = tokenizer.encode("hello")
        assert all(isinstance(i, int) for i in ids)

    def test_decode_roundtrip(self, tokenizer):
        text = "the fox"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_empty_string(self, tokenizer):
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

    def test_ids_in_vocab(self, tokenizer):
        ids = tokenizer.encode("quick brown")
        assert all(i in tokenizer.vocab for i in ids)

    def test_encode_iterable(self, tokenizer):
        lines = ["the quick", "brown fox"]
        list_ids = [id_ for line in lines for id_ in tokenizer.encode(line)]
        iter_ids = list(tokenizer.encode_iterable(iter(lines)))
        assert list_ids == iter_ids


class TestTokenizerSpecialTokens:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        special = ["<|endoftext|>"]
        # Use a richer corpus so there are enough unique bigrams to satisfy
        # the requested number of merges (vocab_size - 257 = 43).
        base = "the quick brown fox jumps over the lazy dog"
        corpus = _tiny_corpus(base, repeat=30) + " <|endoftext|>\n" * 10
        tokens, merges = _train_small(corpus, vocab_size=300, special_tokens=special)
        return Tokenizer(tokens, merges, special_tokens=special)

    def test_special_token_is_single_id(self, tokenizer):
        ids = tokenizer.encode("<|endoftext|>")
        assert len(ids) == 1

    def test_special_token_round_trip(self, tokenizer):
        text = "hello <|endoftext|> world"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_special_token_id_is_256(self, tokenizer):
        (special_id,) = tokenizer.encode("<|endoftext|>")
        assert special_id == BASE_VOCAB_SIZE  # first ID after single-byte tokens


# ---------------------------------------------------------------------------
# train_bpe
# ---------------------------------------------------------------------------

_RICH_CORPUS = _tiny_corpus(
    "the quick brown fox jumps over the lazy dog pack my box with five dozen liquor jugs",
    repeat=30,
)


class TestTrainBpe:
    def test_returns_correct_types(self):
        tokens, merges = _train_small(_RICH_CORPUS, vocab_size=270)
        assert isinstance(tokens, dict)
        assert isinstance(merges, list)
        assert all(isinstance(v, bytes) for v in tokens.values())
        assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in merges)

    def test_vocab_size(self):
        target = 280
        tokens, merges = _train_small(_RICH_CORPUS, vocab_size=target)
        assert len(tokens) == target

    def test_num_merges(self):
        target = 280
        tokens, merges = _train_small(_RICH_CORPUS, vocab_size=target)
        assert len(merges) == target - BASE_VOCAB_SIZE

    def test_single_byte_tokens_present(self):
        tokens, _ = _train_small(_RICH_CORPUS, vocab_size=260)
        for byte_val in range(BASE_VOCAB_SIZE):
            assert tokens[byte_val] == bytes([byte_val])
