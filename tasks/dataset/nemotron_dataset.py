"""
Streaming loader for nvidia/Nemotron-Pretraining-Specialized-v1.1.

The dataset has a single `text` column (raw pretraining text, no chat template).
We stream it so only the rows that the training loop actually consumes are
downloaded — no full 100M-row download required.

Only rows whose *character* length falls within [min_chars, max_chars] are kept.
Character length is a fast proxy for token length (approx chars / 3.5 for LLaDA2
tokenizer) and avoids calling the tokenizer during filtering.

Supported subsets
-----------------
    Nemotron-Pretraining-Code-Concepts           (15.2M rows)
    Nemotron-Pretraining-Economics               (345k  rows)
    Nemotron-Pretraining-Formal-Logic            (491k  rows)
    Nemotron-Pretraining-Multiple-Choice         (3.53M rows)
    Nemotron-Pretraining-Unconditional-Algorithmic (181k rows)
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

from datasets import IterableDataset, load_dataset

DATASET_REPO = "nvidia/Nemotron-Pretraining-Specialized-v1.1"

ALL_SUBSETS = [
    "Nemotron-Pretraining-Code-Concepts",
    "Nemotron-Pretraining-Economics",
    "Nemotron-Pretraining-Formal-Logic",
    "Nemotron-Pretraining-Multiple-Choice",
    "Nemotron-Pretraining-Unconditional-Algorithmic",
]

# Chars-per-token ratio for LLaDA2's tokenizer (used to convert target token
# lengths into a character-level pre-filter).  Conservative estimate keeps FP
# misses low without discarding too many borderline examples.
_CHARS_PER_TOKEN = 3.5


def _token_len_to_chars(token_len: int) -> int:
    return int(token_len * _CHARS_PER_TOKEN)


class NemotronStreamingDataset:
    """
    Iterable wrapper around Nemotron-Pretraining-Specialized-v1.1.

    Yields dicts with key ``"text"`` (str).  Applies a character-length filter
    so that only samples likely to have >= min_token_len tokens are returned.

    Parameters
    ----------
    subsets:
        One or more subset names.  Pass ``None`` to use all subsets.
    min_token_len:
        Drop examples whose char length is below ``min_token_len * _CHARS_PER_TOKEN``.
        Set to 0 to disable the lower bound.
    max_token_len:
        Drop examples whose char length exceeds ``max_token_len * _CHARS_PER_TOKEN``.
        Set to 0 to disable the upper bound.
    transform:
        Optional callable applied to each raw dict before yielding.
        Receives ``{"text": str, "uuid": str, "license": str, ...}`` and should
        return a list of sample dicts (same convention as other transforms in
        this repo so they compose cleanly with the existing dataloader).
    shuffle_buffer:
        If > 0, use a shuffle buffer of this size (``IterableDataset.shuffle``).
    seed:
        Random seed for shuffle buffer.
    """

    def __init__(
        self,
        subsets: Optional[list[str]] = None,
        min_token_len: int = 0,
        max_token_len: int = 0,
        transform: Optional[Callable[[dict[str, Any]], list[dict[str, Any]]]] = None,
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.subsets = subsets if subsets is not None else ALL_SUBSETS
        for s in self.subsets:
            if s not in ALL_SUBSETS:
                raise ValueError(f"Unknown subset {s!r}. Valid: {ALL_SUBSETS}")

        self.min_chars = _token_len_to_chars(min_token_len) if min_token_len > 0 else 0
        self.max_chars = _token_len_to_chars(max_token_len) if max_token_len > 0 else 0
        self.transform = transform
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_subset(self, subset: str) -> IterableDataset:
        raw = load_dataset(
            DATASET_REPO,
            name=subset,
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        if not isinstance(raw, IterableDataset):
            raise TypeError(f"Expected IterableDataset for subset {subset!r}, got {type(raw)}")
        return raw

    def _filter(self, row: dict[str, Any]) -> bool:
        text: str = row.get("text", "")
        n = len(text)
        if self.min_chars > 0 and n < self.min_chars:
            return False
        if self.max_chars > 0 and n > self.max_chars:
            return False
        return True

    def _build_stream(self) -> IterableDataset:
        """Merge all requested subsets into a single IterableDataset."""
        streams = [self._load_subset(s) for s in self.subsets]

        if len(streams) == 1:
            merged = streams[0]
        else:
            # interleave_datasets is memory-safe and streams round-robin
            from datasets import interleave_datasets
            merged = interleave_datasets(streams, seed=self.seed)

        if self.min_chars > 0 or self.max_chars > 0:
            merged = merged.filter(self._filter)

        if self.shuffle_buffer > 0:
            merged = merged.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        return merged

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._build_stream():
            if self.transform is not None:
                for sample in self.transform(row):
                    yield sample
            else:
                yield {"text": row["text"]}

    def as_iterable_dataset(self) -> IterableDataset:
        """Return the raw filtered IterableDataset (without transform applied)."""
        return self._build_stream()


def build_nemotron_streaming_dataset(
    subsets: Optional[list[str]] = None,
    min_token_len: int = 0,
    max_token_len: int = 0,
    transform: Optional[Callable] = None,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
) -> NemotronStreamingDataset:
    """
    Factory used by the training script.

    Compatible with veomni's ``IterativeDataset`` wrapper via ``__iter__``.
    """
    return NemotronStreamingDataset(
        subsets=subsets,
        min_token_len=min_token_len,
        max_token_len=max_token_len,
        transform=transform,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )
