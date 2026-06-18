#!/usr/bin/env python3
"""
fact_score.py  — GraphMERT Pipeline Step 8 (Two-LLM Validation)

Validates triples using two separate LLM families simultaneously.
A triple is kept only when BOTH models agree it is valid.

Usage:
  python utils/llm_scores/fact_score.py \\
    --input_csv   ${OUTPUT_BASE}/graphmert/graphmert_kg/combined/final_kg_scientific_only.csv \\
    --output_csv  ${OUTPUT_BASE}/final_kg/validated_final_kg.csv \\
    --model_ids   /path/to/mistral-nemo-12b /path/to/qwen3-14b

Expected output: 15,000–50,000 validated triples.
"""

import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any

import pandas as pd
from vllm import LLM, SamplingParams

from prompts_scores import system_prompt_validity_score as FACT_CHECK_SYSTEM_PROMPT


def build_fact_check_user_prompt(head: str, relation: str, tail: str,
                                  no_think: bool = False) -> str:
    # validate_predictions is a JUDGMENT task — graphmert.validate_predictions_
    # no_think defaults False so Qwen3's <think> stays on. Two-LLM consensus
    # benefits from each model showing its reasoning so the merge step
    # (combine_tails) can weight votes by reasoning quality. Flip to True via
    # config only if running a non-reasoning model where /no_think is no-op.
    suffix = " /no_think" if no_think else ""
    return f"Head: {head}\nRelation: {relation}\nTail: {tail}{suffix}"


logger = logging.getLogger("fact_score")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Two-LLM validation of KG triples")
    ap.add_argument("--input_csv", required=True,
                    help="Input CSV with columns: head, relation, tail")
    ap.add_argument("--output_csv", required=True,
                    help="Output CSV path for validated triples")
    ap.add_argument("--model_ids", nargs="+", required=True,
                    help="Two model paths to use for validation (must provide exactly 2)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    return ap.parse_args()


def score_triples(df: pd.DataFrame, model_id: str, batch_size: int,
                  max_model_len: int, tensor_parallel_size: int,
                  no_think: bool = False) -> List[bool]:
    """Returns a list of booleans — True if model judges the triple as valid.

    no_think: when True, append '/no_think' control token to the user prompt
    so Qwen3-class models skip <think>. Defaults False because this is a
    judgment task where the reasoning trace genuinely helps consensus.
    Override via configs/default.yaml::graphmert.validate_predictions_no_think.
    """
    logger.info("Loading model: %s  (think=%s)", model_id, "OFF" if no_think else "ON")
    llm = LLM(model=model_id, max_model_len=max_model_len,
               tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)

    results = []
    t0 = time.time()

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        prompts = []
        for _, row in batch.iterrows():
            messages = [
                {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": build_fact_check_user_prompt(
                    str(row["head"]), str(row["relation"]), str(row["tail"]),
                    no_think=no_think,
                )},
            ]
            prompts.append(messages)

        outputs = llm.chat(prompts, sampling_params=sampling)
        for out in outputs:
            text = (out.outputs[0].text if out.outputs else "").strip().lower()
            results.append(text.startswith("yes") or text.startswith("true"))

        done = start + len(batch)
        if done % (batch_size * 10) == 0 or done == len(df):
            elapsed = time.time() - t0
            logger.info("Scored %d/%d rows (%.1f rows/s) [%s]",
                        done, len(df), done / max(elapsed, 1e-6), os.path.basename(model_id))

    del llm
    return results


def main():
    args = parse_args()

    if len(args.model_ids) != 2:
        raise ValueError("Exactly 2 model_ids required for two-LLM validation")

    logger.info("Loading input CSV: %s", args.input_csv)
    df = pd.read_csv(args.input_csv)
    logger.info("Input triples: %d", len(df))

    for col in ["head", "relation", "tail"]:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column: '{col}'")

    # Read Qwen3 thinking control. Default False — judgment task benefits
    # from <think>. Override via configs/default.yaml::graphmert.
    # fact_score_no_think (or per-profile).
    try:
        import os as _os, sys as _sys
        _repo_root = _os.environ.get("REPO_ROOT") or _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "..")
        )
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from pipeline_config import get_phase_param
        no_think = bool(get_phase_param('graphmert', 'fact_score_no_think', False))
    except Exception as e:
        logger.warning("could not read graphmert.fact_score_no_think (%s) — defaulting False", e)
        no_think = False

    # Score with both models
    scores_1 = score_triples(df, args.model_ids[0], args.batch_size,
                              args.max_model_len, args.tensor_parallel_size,
                              no_think=no_think)
    scores_2 = score_triples(df, args.model_ids[1], args.batch_size,
                              args.max_model_len, args.tensor_parallel_size,
                              no_think=no_think)

    df["valid_model_1"] = scores_1
    df["valid_model_2"] = scores_2
    df["both_agree_valid"] = df["valid_model_1"] & df["valid_model_2"]

    validated = df[df["both_agree_valid"]].drop(
        columns=["valid_model_1", "valid_model_2", "both_agree_valid"]
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    validated.to_csv(args.output_csv, index=False)

    logger.info("Input triples:    %d", len(df))
    logger.info("Validated (both): %d (%.1f%%)", len(validated),
                100 * len(validated) / max(len(df), 1))
    logger.info("Output saved to: %s", args.output_csv)

    if len(validated) < 15_000:
        logger.warning(
            "Only %d validated triples — below expected range (15k–50k). "
            "Consider expanding your seed KG or adjusting the model.",
            len(validated)
        )


if __name__ == "__main__":
    main()
