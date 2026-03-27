"""
Inspect nvidia/Nemotron-Pretraining-Specialized-v1.1 without downloading the full dataset.

Usage:
    python scripts/inspect_nemotron.py
    python scripts/inspect_nemotron.py --subset Nemotron-Pretraining-Code-Concepts --n 5
    python scripts/inspect_nemotron.py --all-subsets --length-stats
    python scripts/inspect_nemotron.py --length-stats --tokenizer ./LLaDA2.0-mini
"""
import argparse
import itertools
import textwrap

from datasets import IterableDataset, load_dataset

SUBSETS = [
    "Nemotron-Pretraining-Code-Concepts",
    "Nemotron-Pretraining-Economics",
    "Nemotron-Pretraining-Formal-Logic",
    "Nemotron-Pretraining-Multiple-Choice",
    "Nemotron-Pretraining-Unconditional-Algorithmic",
]

DATASET_REPO = "nvidia/Nemotron-Pretraining-Specialized-v1.1"
STATS_ROWS = 200


def _stream(subset: str) -> IterableDataset:
    """Return a streaming IterableDataset for one subset (no full download)."""
    raw = load_dataset(
        DATASET_REPO,
        name=subset,
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    if not isinstance(raw, IterableDataset):
        raise TypeError(f"Expected IterableDataset, got {type(raw)}")
    return raw


def inspect_subset(subset: str, n: int, tokenizer_path: str | None, length_stats: bool) -> None:
    print(f"\n{'='*70}")
    print(f"Subset : {subset}")
    print(f"{'='*70}")

    ds = _stream(subset)
    samples = list(itertools.islice(ds, n))

    if not samples:
        print("  [no samples returned]")
        return

    # --- Column schema ---
    first = samples[0]
    print(f"\nColumns ({len(first)} total):")
    for col, val in first.items():
        val_repr = repr(val)[:120] + ("..." if len(repr(val)) > 120 else "")
        print(f"  {col!r:40s} -> {type(val).__name__}: {val_repr}")

    # --- Sample rows ---
    print(f"\nSample rows (n={n}):")
    for i, row in enumerate(samples):
        text: str = row.get("text", "")
        meta: dict = row.get("metadata", {}) or {}
        uuid: str = row.get("uuid", "?")
        print(f"\n  [{i}] uuid={uuid}  len(text)={len(text)} chars")
        print(f"       text preview: {textwrap.shorten(text, width=200)!r}")
        for k, v in meta.items():
            v_short = textwrap.shorten(str(v), width=100)
            print(f"       metadata.{k}: {v_short!r}")

    # --- Optional length stats ---
    if not length_stats:
        return

    print(f"\nCharacter-length distribution (first {STATS_ROWS} rows):")
    char_lengths = [len(r.get("text", "")) for r in itertools.islice(_stream(subset), STATS_ROWS)]
    char_lengths.sort()
    _print_percentiles(char_lengths, "chars")

    if tokenizer_path:
        print(f"\nTokenised-length distribution using {tokenizer_path!r}:")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tok_lengths = [
            len(tok(r["text"], add_special_tokens=False)["input_ids"])
            for r in itertools.islice(_stream(subset), STATS_ROWS)
        ]
        tok_lengths.sort()
        _print_percentiles(tok_lengths, "tokens")


def _print_percentiles(vals: list[int], unit: str) -> None:
    def q(p: int) -> int:
        return vals[max(0, int(len(vals) * p / 100) - 1)]
    print(
        f"  n={len(vals)}  min={vals[0]}  p25={q(25)}  "
        f"median={q(50)}  p75={q(75)}  p90={q(90)}  max={vals[-1]}  ({unit})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Nemotron-Pretraining-Specialized-v1.1 via streaming"
    )
    parser.add_argument(
        "--subset", default=SUBSETS[0], choices=SUBSETS,
        help="Which subset to inspect (default: Code-Concepts)",
    )
    parser.add_argument(
        "--all-subsets", action="store_true",
        help="Inspect all 5 subsets (overrides --subset)",
    )
    parser.add_argument(
        "-n", type=int, default=3,
        help=f"Sample rows to print per subset (default: 3)",
    )
    parser.add_argument(
        "--length-stats", action="store_true",
        help=f"Print char/token length percentiles over first {STATS_ROWS} rows",
    )
    parser.add_argument(
        "--tokenizer", default=None, metavar="PATH",
        help="HF repo or local path to tokenizer (enables token-length stats)",
    )
    args = parser.parse_args()

    targets = SUBSETS if args.all_subsets else [args.subset]
    for subset in targets:
        inspect_subset(subset, args.n, args.tokenizer, args.length_stats)


if __name__ == "__main__":
    main()
