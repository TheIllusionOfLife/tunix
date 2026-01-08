#!/usr/bin/env python3
"""
Dataset distribution report for local parquet files under ./data.
Focuses on response length (chars) and basic language heuristics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List, Tuple

import pyarrow.parquet as pq


HANGUL = re.compile(r"[\uAC00-\uD7AF]")
CJK = re.compile(r"[\u4E00-\u9FFF]")
CYR = re.compile(r"[\u0400-\u04FF]")


@dataclass
class LengthStats:
    total: int = 0
    empty: int = 0
    total_len: int = 0
    min_len: int | None = None
    max_len: int = 0
    lt_50: int = 0
    gt_4000: int = 0
    hangul: int = 0
    cjk: int = 0
    cyr: int = 0


def iter_strings(table, col_name: str) -> Iterable[str]:
    col = table[col_name]
    for i in range(table.num_rows):
        yield col[i].as_py()


def update_stats(stats: LengthStats, text: str | None) -> None:
    stats.total += 1
    if not text:
        stats.empty += 1
        return
    l = len(text)
    stats.total_len += l
    stats.min_len = l if stats.min_len is None else min(stats.min_len, l)
    stats.max_len = max(stats.max_len, l)
    if l < 50:
        stats.lt_50 += 1
    if l > 4000:
        stats.gt_4000 += 1
    if HANGUL.search(text):
        stats.hangul += 1
    if CJK.search(text):
        stats.cjk += 1
    if CYR.search(text):
        stats.cyr += 1


def collect_lengths(pf: pq.ParquetFile, col_name: str, max_rows: int | None) -> List[int]:
    lengths: List[int] = []
    rows_seen = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[col_name])
        for s in iter_strings(table, col_name):
            if s:
                lengths.append(len(s))
            else:
                lengths.append(0)
            rows_seen += 1
            if max_rows is not None and rows_seen >= max_rows:
                return lengths
    return lengths


def percentile(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def report_file(path: Path, col_name: str, label: str, max_rows: int | None) -> None:
    pf = pq.ParquetFile(path)
    cols = pf.schema.names
    if col_name not in cols:
        print(f"{label}: missing '{col_name}' column. cols={cols}")
        return

    stats = LengthStats()
    rows_seen = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[col_name])
        for s in iter_strings(table, col_name):
            update_stats(stats, s)
            rows_seen += 1
            if max_rows is not None and rows_seen >= max_rows:
                break
        if max_rows is not None and rows_seen >= max_rows:
            break

    lengths = collect_lengths(pf, col_name, max_rows)
    lengths_sorted = sorted(lengths)

    mean_len = 0.0
    non_empty = stats.total - stats.empty
    if non_empty > 0:
        mean_len = stats.total_len / non_empty

    print("=" * 72)
    print(f"{label}")
    print(f"rows_total: {pf.metadata.num_rows}")
    if max_rows is not None:
        print(f"rows_sampled: {stats.total}")
    print(f"empty: {stats.empty}")
    print(f"len_min: {stats.min_len} len_max: {stats.max_len} len_mean: {mean_len:.1f}")
    print(f"len_lt_50: {stats.lt_50} ({stats.lt_50 / max(1, stats.total):.2%})")
    print(f"len_gt_4000: {stats.gt_4000} ({stats.gt_4000 / max(1, stats.total):.2%})")
    print(f"hangul_any: {stats.hangul} ({stats.hangul / max(1, stats.total):.2%})")
    print(f"cjk_any: {stats.cjk} ({stats.cjk / max(1, stats.total):.2%})")
    print(f"cyr_any: {stats.cyr} ({stats.cyr / max(1, stats.total):.2%})")
    print("percentiles (chars):")
    for p in [1, 5, 10, 50, 75, 80, 90, 95, 96, 97, 98, 99]:
        print(f"  p{p}: {percentile(lengths_sorted, p)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report response length distributions for parquet files.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows scanned per file (for speed). Default: all rows.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing parquet files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = [
        (data_dir / "glaiveai_90k_part1.parquet", "response", "glaiveai_90k_part1.parquet:response"),
        (data_dir / "glaiveai_90k_part2.parquet", "response", "glaiveai_90k_part2.parquet:response"),
        (data_dir / "glaiveai_continuation_100k.parquet", "response", "glaiveai_continuation_100k.parquet:response"),
        (Path("archive/data/cot_collection_10k.parquet"), "target", "cot_collection_10k.parquet:target"),
        (Path("archive/data/cot_collection_10k.parquet"), "rationale", "cot_collection_10k.parquet:rationale"),
        (Path("archive/data/openo1_sft_english_20k.parquet"), "output", "openo1_sft_english_20k.parquet:output"),
        (Path("archive/data/raiden_deepseek_r1.parquet"), "response", "raiden_deepseek_r1.parquet:response"),
    ]

    for path, col_name, label in files:
        if not path.exists():
            print(f"{label}: not found")
            continue
        report_file(path, col_name, label, args.max_rows)


if __name__ == "__main__":
    main()
