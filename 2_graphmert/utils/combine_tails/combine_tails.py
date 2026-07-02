#!/usr/bin/env python3
# coding=utf-8
"""
combine_tails.py  — GraphMERT Pipeline Step 7

Merges and deduplicates the shard CSVs from predict_tails_llm.py into a
single KG CSV. No LLM filtering is done here — run fact_score.py separately
as the quality gate (after seed KG generation and after GraphMERT expansion).

Usage (direct):
  python utils/combine_tails/combine_tails.py \\
    --pred_dir   ${OUTPUT_BASE}/graphmert/graphmert_kg/predictions \\
    --output_dir ${OUTPUT_BASE}/graphmert/graphmert_kg/combined

Or via environment variables (for SLURM array jobs):
  export PRED_INPUT_DIR=...
  export OUT_DIR=...
  python utils/combine_tails/combine_tails.py
"""

import os
import logging
import argparse

import pandas as pd


logger = logging.getLogger("combine_tails")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ARRAY_ID    = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
ARRAY_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))


def parse_args():
    ap = argparse.ArgumentParser(description="Merge and deduplicate predicted tail CSVs into a single KG")
    ap.add_argument("--pred_dir", default=os.getenv("PRED_INPUT_DIR", ""),
                    help="Directory containing shard CSVs from predict_tails_llm.py")
    ap.add_argument("--output_dir", default=os.getenv("OUT_DIR", ""),
                    help="Output directory for combined KG file")
    ap.add_argument("--take_subset", action="store_true",
                    default=os.getenv("TAKE_SUBSET", "0").lower() in ("1", "true", "yes"))
    return ap.parse_args()


def load_all_shard_csvs(pred_dir: str) -> pd.DataFrame:
    """Load and concatenate the per-tail shard CSVs from predict_tails_llm.py.
    Each shard writes two files:
      predictions_shard{N}_of{M}.csv  — per-tail rows (id/head/relation/tail)
      queries_shard{N}_of{M}.csv      — per-query rows (head/relation/tails_json)
    We want the per-tail rows only; filter on the "predictions_shard" prefix to
    exclude queries. (predict_tails_llm.py names the var `out_exploded`, but the
    filename has no "exploded" substring, so upstream dc5bb46's `"exploded" in f`
    filter would match nothing on this branch — orchestration renamed the output.)
    """
    csv_files = sorted([
        os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
        if f.endswith(".csv") and "predictions_shard" in f
    ])
    if not csv_files:
        raise FileNotFoundError(
            f"No predictions_shard*.csv files found in {pred_dir}. "
            f"Expected predict_tails_llm.py to have written "
            f"predictions_shard{{N}}_of{{M}}.csv there."
        )
    logger.info("Loading %d shard CSVs from %s", len(csv_files), pred_dir)
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Total rows after concat: %d", len(df))
    return df


def main():
    args = parse_args()

    if not args.pred_dir:
        raise ValueError("--pred_dir is required (or set PRED_INPUT_DIR env var)")
    if not args.output_dir:
        raise ValueError("--output_dir is required (or set OUT_DIR env var)")

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_all_shard_csvs(args.pred_dir)
    if args.take_subset:
        df = df.head(1000)
        logger.info("take_subset=True, using first 1000 rows")

    df = df[df["tail"].notna() & (df["tail"].astype(str).str.strip() != "")]
    before = len(df)
    df = df.drop_duplicates(subset=["head", "relation", "tail"])
    logger.info("Deduplicated: %d → %d rows", before, len(df))

    out_csv = os.path.join(args.output_dir, "final_kg_combined.csv")
    df.to_csv(out_csv, index=False)
    logger.info("Saved combined KG: %s (%d rows)", out_csv, len(df))


if __name__ == "__main__":
    main()
