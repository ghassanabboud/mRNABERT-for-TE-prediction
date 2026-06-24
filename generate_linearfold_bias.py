#!/usr/bin/env python3
"""
Pre-compute LinearFold secondary-structure bias at the token level.

For each sequence in the input CSV (space-separated tokens: single nucleotides
for UTR regions, 3-letter codons for CDS), runs LinearFold and maps the
resulting dot-bracket structure to token-level pair arrays stored in a .npz.

Core logic lives in bias.linearfold; this script is the CLI entry point.

Usage:
    # Single file
    python generate_linearfold_bias.py path/to/train.csv
    python generate_linearfold_bias.py path/to/train.csv -o train_lf_bias.npz

    # Directory: all CSVs combined into one .npz
    python generate_linearfold_bias.py path/to/data_dir/
    python generate_linearfold_bias.py path/to/data_dir/ -o all_lf_bias.npz

    python generate_linearfold_bias.py ... --max-tokens 1022 --jobs 4
    python generate_linearfold_bias.py ... --linearfold /path/to/linearfold
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from bias.linearfold import process_one

DEFAULT_LINEARFOLD = "../LinearFold/linearfold"


def _process_csv(
    csv_path: Path,
    max_tokens: int,
    jobs: int,
    executable: str,
) -> dict:
    """Process one CSV and return a {tx_id: pairs_array} dict."""
    df = pd.read_csv(csv_path)
    missing = {"tx_id", "sequence"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    tx_ids = df["tx_id"].astype(str).tolist()
    sequences = df["sequence"].tolist()
    n = len(tx_ids)
    print(f"  {csv_path.name}: {n} sequences")

    arrays: dict = {}
    skipped = 0

    if jobs == 1:
        for tx_id, seq in tqdm(zip(tx_ids, sequences), total=n, leave=False):
            try:
                _, pairs = process_one(tx_id, seq, max_tokens, executable)
                arrays[tx_id] = pairs
            except Exception as e:
                print(f"\n  Failed {tx_id}: {e}")
                skipped += 1
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(process_one, tx_id, seq, max_tokens, executable): tx_id
                for tx_id, seq in zip(tx_ids, sequences)
            }
            for fut in tqdm(as_completed(futures), total=n, leave=False):
                tx_id = futures[fut]
                try:
                    _, pairs = fut.result()
                    arrays[tx_id] = pairs
                except Exception as e:
                    print(f"\n  Failed {tx_id}: {e}")
                    skipped += 1

    if skipped:
        print(f"  Skipped {skipped} sequences in {csv_path.name}")
    return arrays


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute LinearFold secondary-structure token-level pair arrays."
    )
    parser.add_argument(
        "input",
        help="Input CSV file, or directory containing CSV files (all combined into one .npz).",
    )
    parser.add_argument(
        "--output", "-o",
        help=(
            "Output .npz path. "
            "Default for a file: <stem>_lf_bias.npz next to the file. "
            "Default for a directory: <dir_name>_lf_bias.npz next to the directory."
        ),
    )
    parser.add_argument(
        "--linearfold",
        default=DEFAULT_LINEARFOLD,
        help=f"Path to the LinearFold executable (default: {DEFAULT_LINEARFOLD})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1022,
        help=(
            "Truncate to this many tokens before folding (default: 1022, matching "
            "model_max_length=1024 minus CLS and SEP). Set 0 to fold the full sequence."
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {input_path}")
        out_path = (
            Path(args.output) if args.output
            else input_path.parent / (input_path.name + "_lf_bias.npz")
        )
        print(f"Processing {len(csv_files)} CSV files from {input_path}/")
        if args.max_tokens > 0:
            print(f"Truncating to first {args.max_tokens} tokens per sequence")
        combined: dict = {}
        for csv_path in csv_files:
            arrays = _process_csv(csv_path, args.max_tokens, args.jobs, args.linearfold)
            overlap = combined.keys() & arrays.keys()
            if overlap:
                print(f"  Warning: {len(overlap)} duplicate tx_ids from {csv_path.name} (overwriting)")
            combined.update(arrays)
    else:
        out_path = (
            Path(args.output) if args.output
            else input_path.with_name(input_path.stem + "_lf_bias.npz")
        )
        print(f"Processing {input_path.name}")
        if args.max_tokens > 0:
            print(f"Truncating to first {args.max_tokens} tokens per sequence")
        combined = _process_csv(input_path, args.max_tokens, args.jobs, args.linearfold)

    np.savez_compressed(out_path, **combined)
    print(f"Saved {len(combined)} pair arrays → {out_path}")


if __name__ == "__main__":
    main()
