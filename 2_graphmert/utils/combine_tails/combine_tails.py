#!/usr/bin/env python3
# coding=utf-8
"""
combine_tails.py  — GraphMERT Pipeline Step 7

Combines the shard CSVs from predict_tails_llm.py into a single KG CSV,
then filters to scientifically plausible triples using an LLM.

Usage (direct):
  python utils/combine_tails/combine_tails.py \\
    --pred_dir   ${OUTPUT_BASE}/graphmert/graphmert_kg/predictions \\
    --output_dir ${OUTPUT_BASE}/graphmert/graphmert_kg/combined \\
    --model_id   /path/to/qwen3-14b

Or via environment variables (for SLURM array jobs):
  export PRED_INPUT_DIR=...
  export OUT_DIR=...
  export MODEL_ID=...
  python utils/combine_tails/combine_tails.py
"""

import os
import json
import logging
import re
import ast
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Pipeline config loader (repo root, 3 levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline_config import get_phase_param  # noqa: E402
import pandas as pd

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from combine_tokens_prompts import SYSTEM_CONTEXT as SYSTEM_PROMPT


def build_user_prompt(head: str, relation: str, tail: str) -> str:
    return f"Head: {head}\nRelation: {relation}\nTail: {tail}"

logger = logging.getLogger("combine_tails")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _sanitize_cuda_visible_devices_for_vllm() -> None:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if "MIG-" in cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_sanitize_cuda_visible_devices_for_vllm()

ARRAY_ID    = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
ARRAY_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))


def parse_args():
    ap = argparse.ArgumentParser(description="Combine predicted tails into a final KG")
    ap.add_argument("--pred_dir", default=os.getenv("PRED_INPUT_DIR", ""),
                    help="Directory containing shard CSVs from predict_tails_llm.py")
    ap.add_argument("--output_dir", default=os.getenv("OUT_DIR", ""),
                    help="Output directory for combined KG files")
    ap.add_argument("--model_id", default=os.getenv("MODEL_ID", ""),
                    help="Path to local vLLM model for scientific plausibility filtering")
    ap.add_argument("--tokenizer", default=os.getenv("TOKENIZER_PATH", ""),
                    help="Path to tokenizer (defaults to model_id if not set)")
    ap.add_argument("--internal_microbatch", type=int, default=int(os.getenv("INTERNAL_MICROBATCH", "256")))
    ap.add_argument("--log_every", type=int, default=int(os.getenv("LOG_EVERY", "250")))
    ap.add_argument("--take_subset", action="store_true",
                    default=os.getenv("TAKE_SUBSET", "0").lower() in ("1", "true", "yes"))
    return ap.parse_args()


def load_all_shard_csvs(pred_dir: str) -> pd.DataFrame:
    """Load and concatenate all shard CSVs from predict_tails_llm.py."""
    csv_files = sorted([
        os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
        if f.endswith(".csv") and "exploded" in f
    ])
    if not csv_files:
        raise FileNotFoundError(f"No exploded shard CSVs found in {pred_dir}")
    logger.info("Loading %d shard CSVs from %s", len(csv_files), pred_dir)
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Total rows after concat: %d", len(df))
    return df


def filter_scientific_triples(df: pd.DataFrame, llm: LLM, tokenizer, microbatch: int,
                              no_think: bool = True) -> pd.DataFrame:
    """Use LLM to filter triples to scientifically plausible ones only.

    no_think: append "/no_think" to suppress Qwen3 <think>. This step is
    plausibility classification (yes/no), max_tokens=10 in sampling — no
    space for thinking anyway. Default True. configs/default.yaml::
    graphmert.combine_tails_no_think.
    """
    results = []
    t0 = time.time()
    think_suffix = " /no_think" if no_think else ""

    valid = df[df["tail"].notna() & (df["tail"].astype(str).str.strip() != "")].copy()
    logger.info("Rows with non-empty tails: %d  think=%s", len(valid), "OFF" if no_think else "ON")

    for start in range(0, len(valid), microbatch):
        batch = valid.iloc[start:start + microbatch]
        prompts = []
        for _, row in batch.iterrows():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(
                    str(row["head"]), str(row["relation"]), str(row["tail"])
                ) + think_suffix},
            ]
            prompts.append(messages)

        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
        outputs = llm.chat(prompts, sampling_params=sampling)

        for row_idx, out in enumerate(outputs):
            text = (out.outputs[0].text if out.outputs else "").strip().lower()
            is_valid = text.startswith("yes") or text.startswith("true")
            row = batch.iloc[row_idx].to_dict()
            row["llm_valid"] = is_valid
            results.append(row)

        if (start + microbatch) % (microbatch * 10) == 0:
            elapsed = time.time() - t0
            logger.info("Filtered %d/%d rows (%.1f rows/s)", start + microbatch, len(valid),
                        (start + microbatch) / max(elapsed, 1e-6))

    return pd.DataFrame(results)


def main():
    args = parse_args()

    if not args.pred_dir:
        raise ValueError("--pred_dir is required (or set PRED_INPUT_DIR env var)")
    if not args.output_dir:
        raise ValueError("--output_dir is required (or set OUT_DIR env var)")
    if not args.model_id:
        raise ValueError("--model_id is required (or set MODEL_ID env var)")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = args.tokenizer or args.model_id

    df = load_all_shard_csvs(args.pred_dir)
    if args.take_subset:
        df = df.head(1000)
        logger.info("take_subset=True, using first 1000 rows")

    logger.info("Loading tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    logger.info("Loading vLLM model: %s", args.model_id)
    # Tunable per profile (graphmert.combine_tails_max_model_len). 4096 is
    # safe; raise on 80 GB cards for higher concurrent batch.
    combine_max_model_len = get_phase_param('graphmert', 'combine_tails_max_model_len', 4096)
    combine_tp_size = get_phase_param('graphmert', 'combine_tails_tensor_parallel_size', 1)
    combine_gpu_mem = get_phase_param('graphmert', 'combine_tails_gpu_memory_utilization', 0.90)
    logger.info("vLLM init: max_model_len=%d tp_size=%d gpu_mem_util=%s",
                combine_max_model_len, combine_tp_size, combine_gpu_mem)
    llm = LLM(model=args.model_id, trust_remote_code=True,
              max_model_len=combine_max_model_len,
              tensor_parallel_size=combine_tp_size,
              gpu_memory_utilization=combine_gpu_mem)

    # Qwen3 thinking control. configs/default.yaml::graphmert.combine_tails_no_think
    try:
        import os as _os, sys as _sys
        _repo_root = _os.environ.get("REPO_ROOT") or _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "..")
        )
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from pipeline_config import get_phase_param
        no_think = bool(get_phase_param('graphmert', 'combine_tails_no_think', True))
    except Exception as e:
        logger.warning("could not read graphmert.combine_tails_no_think (%s) — defaulting True", e)
        no_think = True

    filtered = filter_scientific_triples(df, llm, tokenizer, args.internal_microbatch, no_think=no_think)

    scientific_only = filtered[filtered["llm_valid"] == True].drop(columns=["llm_valid"])
    out_csv = os.path.join(args.output_dir, "final_kg_scientific_only.csv")
    scientific_only.to_csv(out_csv, index=False)
    logger.info("Saved scientific-only KG: %s (%d rows)", out_csv, len(scientific_only))

    all_out = os.path.join(args.output_dir, "final_kg_all.csv")
    filtered.to_csv(all_out, index=False)
    logger.info("Saved all (with llm_valid flag): %s (%d rows)", all_out, len(filtered))


if __name__ == "__main__":
    main()
