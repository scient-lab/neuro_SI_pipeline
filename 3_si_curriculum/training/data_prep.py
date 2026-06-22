#!/usr/bin/env python3
"""
data_prep.py — SI Pipeline Step 4a

Converts validated Q&A JSON into a `DatasetDict({"train", "test"})` with a
`text` column ready for TRL `SFTTrainer`. TRL handles its own tokenization
via `dataset_text_field='text'`; we deliberately do NOT pre-tokenize here.

Ported from bottom-up-superintelligence/data/tokenization.py
(2026-06-03 commit) with these reconciliations:
 - Input schema is the fork's `curriculum_verified.json` (per-item fields:
   `question`, `answer`, `explanation` or `thinking_trace`, optional
   `question_and_explanation`, `source_concept`, `target_concept`, `paths`).
   Upstream expects only `question_and_explanation` and does the split
   itself; we accept both layouts so existing verified JSONs from
   verify_questions.py work unchanged.
 - Uses `tokenizer.apply_chat_template` so the chat template tracks the
   active base model. Replaces the hand-rolled `CHAT_TEMPLATE` constant
   that hardcoded one specific format (bug #7 — chat-template / model
   mismatch with the trainer's response-token lookup).
 - Wraps in `DatasetDict({"train", "test"})` with a small held-out
   eval slice. Previously this script saved a flat `Dataset` which
   crashed `trainer.py` (`dataset['train']` → KeyError on a flat dataset:
   bug #4) and `dataset_text_field='text'` had no `text` column to read
   (bug #5).

Usage:
  python training/data_prep.py \\
    --input_file  ${OUTPUT_BASE}/SI/QA_items/verified/merged_concise.json \\
    --output_path ${OUTPUT_BASE}/SI/QA_items/training_data/ \\
    --model_name  /path/to/base_model
"""

import os
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare SFT training data from Q&A JSON")
    ap.add_argument("--input_file", required=True,
                    help="Validated Q&A JSON file (output of verify_questions.py)")
    ap.add_argument("--output_path", required=True,
                    help="Output directory for DatasetDict (train+test splits)")
    ap.add_argument("--model_name", required=True,
                    help="Base model path or HF repo ID (for tokenizer chat template)")
    ap.add_argument("--eval_frac", type=float, default=0.05,
                    help="Fraction of items held out for the 'test' split (default: 0.05)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for the train/test split")
    ap.add_argument("--cache_dir", default=None,
                    help="HF cache directory (optional)")
    return ap.parse_args()


def _split_question_and_explanation(qae: str) -> Tuple[str, str, str]:
    """Parse upstream's structured `<Question>...<Options>...<Explanation>...<Answer>:...`
    blob. Returns (prompt, cot, answer)."""
    question = qae.split("<Question>")[1].split("</Question>")[0]
    options = qae.split("<Options>")[1].split("</Options>")[0]
    answer_option = random.choice(["A", "B", "C", "D"])
    answer_instruction = (
        "Please only output the choice letter in the answer field e.g. "
        f"Final Answer: {answer_option}"
    )
    prompt = question + "\nOptions:" + options + "\n" + answer_instruction
    cot = qae.split("<Explanation>")[1].split("</Explanation>")[0]
    answer = qae.split("<Answer>:")[1].split("</Answer>")[0]
    return prompt.strip(), cot.strip(), answer.strip()


def _build_text(item: Dict, tokenizer) -> str:
    """Build the chat-template-formatted text for a single Q&A item.
    Accepts BOTH input layouts: (a) upstream's combined
    `question_and_explanation` blob, OR (b) the fork's separate
    `question` / `answer` / `explanation`|`thinking_trace` fields.
    """
    qae = item.get("question_and_explanation")
    if qae:
        prompt, cot, answer = _split_question_and_explanation(qae)
    else:
        prompt = str(item.get("question", "")).strip()
        cot = str(item.get("thinking_trace") or item.get("explanation") or "").strip()
        answer = str(item.get("answer", "")).strip()
        if not prompt or not answer:
            return ""

    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "<think>\n" + cot + "\n</think>\nFinal Answer: " + answer,
            },
        ],
        tokenize=False,
    )


def main():
    args = parse_args()

    logger.info("Loading data from: %s", args.input_file)
    with open(args.input_file, "r", encoding="utf-8") as f:
        items: List[Dict] = json.load(f)
    logger.info("Loaded %d items", len(items))

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    texts: List[str] = []
    for item in items:
        text = _build_text(item, tokenizer)
        if text:
            texts.append(text)
    logger.info("Formatted %d / %d items (dropped %d empty)",
                len(texts), len(items), len(items) - len(texts))

    if not texts:
        raise RuntimeError(
            f"No items survived formatting from {args.input_file}. "
            f"Check that the JSON has either 'question_and_explanation' "
            f"or 'question'+'answer'+('explanation'|'thinking_trace') fields."
        )

    random.seed(args.seed)
    random.shuffle(texts)
    n_eval = max(1, int(round(len(texts) * args.eval_frac))) if len(texts) > 1 else 0
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:]
    if not train_texts:
        # tiny smoke fixtures: avoid empty train split
        train_texts = texts
        eval_texts = texts[:1]
    logger.info("Split: train=%d  test=%d", len(train_texts), len(eval_texts))

    ds = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "test": Dataset.from_dict({"text": eval_texts}),
    })

    os.makedirs(args.output_path, exist_ok=True)
    ds.save_to_disk(args.output_path)
    logger.info("DatasetDict saved to: %s  (train=%d test=%d)",
                args.output_path, len(ds["train"]), len(ds["test"]))


if __name__ == "__main__":
    main()
