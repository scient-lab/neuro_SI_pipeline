#!/usr/bin/env python3
# coding=utf-8
"""
clean_llm_relations.py  — GraphMERT Pipeline Step 3b

Cleans the output of add_llm_relations.py: filters relations to only
the allowed set, deduplicates, and optionally drops empty rows.

Usage:
  python utils/relation_matching/clean_llm_relations.py \\
    --input_dir   ${OUTPUT_BASE}/graphmert/llm_relations/relations_all \\
    --output_dir  ${OUTPUT_BASE}/graphmert/llm_relations/relations_cleaned \\
    --tokenizer   ${OUTPUT_BASE}/graphmert/stable_tokenizer

This produces two datasets: relations_cleaned_train and relations_cleaned_eval.
(The split is controlled by --eval_split_pct.)
"""

import os
import re
import json
import logging
import argparse
from typing import Any, Dict, List

import sys
from pathlib import Path

from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import AutoTokenizer

# Pipeline config loader (repo root, 3 levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline_config import get_relations  # noqa: E402


logger = logging.getLogger("clean_llm_relations")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALLOWED_RELATIONS = set(get_relations())


def parse_args():
    ap = argparse.ArgumentParser(description="Clean LLM relation predictions")
    ap.add_argument("--input_dir", required=True,
                    help="Dataset directory from add_llm_relations.py")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for cleaned dataset")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to stable tokenizer")
    ap.add_argument("--eval_split_pct", type=int, default=5,
                    help="Percentage of rows to hold out as eval (default: 5)")
    ap.add_argument("--nonempty_only", action="store_true", default=True,
                    help="Drop rows with empty cleaned_relations_json (default: True)")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--subset", type=int, default=0,
                    help="If > 0, only use this many rows (for debugging)")
    return ap.parse_args()


def _safe_json_loads(x: Any) -> Any:
    if x is None: return None
    if isinstance(x, (dict, list)): return x
    if isinstance(x, str):
        s = x.strip()
        if not s: return None
        try: return json.loads(s)
        except Exception: return None
    return None


def clean_relations_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans relations_json column:
    - Parse JSON output from model
    - Keep only allowed relations
    - Lowercase heads
    - Write compact JSON to cleaned_relations_json
    """
    cleaned_list = []
    for raw in examples.get("relations_json", []):
        parsed = _safe_json_loads(raw)
        if not isinstance(parsed, dict):
            cleaned_list.append("{}")
            continue
        cleaned: Dict[str, List[str]] = {}
        for head, rels in parsed.items():
            head_lower = str(head).strip().lower()
            if not head_lower:
                continue
            if isinstance(rels, str):
                rels = [rels]
            if not isinstance(rels, list):
                continue
            valid = sorted({str(r).strip().lower() for r in rels if str(r).strip().lower() in ALLOWED_RELATIONS})
            if valid:
                cleaned[head_lower] = valid
        cleaned_list.append(json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False))
    examples["cleaned_relations_json"] = cleaned_list
    return examples


def main():
    args = parse_args()

    logger.info("Loading dataset from: %s", args.input_dir)
    dataset = load_from_disk(args.input_dir)
    logger.info("Loaded %d rows, cols=%s", len(dataset), dataset.column_names)

    if args.subset > 0:
        n = min(args.subset, len(dataset))
        dataset = dataset.select(range(n))
        logger.info("Using subset of %d rows", n)

    logger.info("Cleaning relations...")
    dataset = dataset.map(
        clean_relations_batch,
        batched=True,
        batch_size=256,
        num_proc=args.num_workers,
        load_from_cache_file=False,
        desc="Cleaning relations",
    )

    if args.nonempty_only:
        before = len(dataset)
        dataset = dataset.filter(
            lambda ex: ex["cleaned_relations_json"] != "{}",
            num_proc=args.num_workers,
        )
        logger.info("Dropped %d empty rows. Remaining: %d", before - len(dataset), len(dataset))

    # Split into train/eval
    n = len(dataset)
    eval_n = max(1, int(n * args.eval_split_pct / 100))
    train_n = n - eval_n
    train_ds = dataset.select(range(train_n))
    eval_ds  = dataset.select(range(train_n, n))

    os.makedirs(args.output_dir, exist_ok=True)
    train_out = os.path.join(args.output_dir, "relations_cleaned_train")
    eval_out  = os.path.join(args.output_dir, "relations_cleaned_eval")

    train_ds.save_to_disk(train_out)
    eval_ds.save_to_disk(eval_out)

    logger.info("Saved train (%d rows) → %s", len(train_ds), train_out)
    logger.info("Saved eval  (%d rows) → %s", len(eval_ds),  eval_out)
    logger.info("NEXT STEP: run run_dataset_preprocessing.py")


if __name__ == "__main__":
    main()
