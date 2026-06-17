#!/usr/bin/env python3
"""
find_heads_positions.py  — GraphMERT Pipeline Step 2b

Merges all entity-discovery chunk outputs and computes exact token positions
for each entity mention using the stable tokenizer.

Usage:
  python utils/entity_discovery/find_heads_positions.py \\
    --heads_chunks_dir  ${OUTPUT_BASE}/graphmert/dataset_with_heads/chunks \\
    --output_dir        ${OUTPUT_BASE}/graphmert/dataset_with_heads \\
    --tokenizer         ${OUTPUT_BASE}/graphmert/stable_tokenizer
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any

from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Merge entity chunks and find exact token positions")
    ap.add_argument("--heads_chunks_dir", required=True,
                    help="Directory containing chunk_0, chunk_1, ... subdirs from entity_discovery.py")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for merged dataset with exact positions")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to stable tokenizer")
    ap.add_argument("--num_workers", type=int, default=32)
    return ap.parse_args()


def find_exact_positions(examples: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Refine head positions to exact token indices by re-tokenizing each snippet.
    head_positions format: {entity_string: approx_token_pos, ...}
    """
    updated = []
    for input_ids, hp_raw in zip(examples["input_ids"], examples["head_positions"]):
        try:
            hp = json.loads(hp_raw) if isinstance(hp_raw, str) else hp_raw
        except Exception:
            hp = {}

        if not isinstance(hp, dict) or not hp:
            updated.append(json.dumps({}))
            continue

        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        text_lower = text.lower()

        exact = {}
        for ent, approx_pos in hp.items():
            ent_lower = ent.lower()
            char_pos = text_lower.find(ent_lower)
            if char_pos < 0:
                continue
            # Tokenize prefix to get exact token index
            prefix_ids = tokenizer.encode(text[:char_pos], add_special_tokens=False)
            tok_pos = min(len(prefix_ids), 511)
            exact[ent_lower] = tok_pos

        updated.append(json.dumps(exact))
    examples["head_positions"] = updated
    return examples


def main():
    args = parse_args()

    # Discover and merge chunks
    chunks_dir = args.heads_chunks_dir
    chunk_dirs = sorted([
        os.path.join(chunks_dir, d) for d in os.listdir(chunks_dir)
        if os.path.isdir(os.path.join(chunks_dir, d)) and d.startswith("chunk_")
    ])
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk_* subdirectories found in {chunks_dir}")

    logger.info("Found %d chunks: %s", len(chunk_dirs), [os.path.basename(d) for d in chunk_dirs])
    datasets_list = [load_from_disk(d) for d in chunk_dirs]
    merged = concatenate_datasets(datasets_list)
    logger.info("Merged dataset: %d rows", len(merged))

    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Computing exact token positions...")
    merged_with_positions = merged.map(
        lambda examples: find_exact_positions(examples, tokenizer),
        batched=True,
        batch_size=256,
        num_proc=args.num_workers,
        load_from_cache_file=False,
        desc="Finding exact positions",
    )

    # Write the Dataset directly to args.output_dir (no hardcoded subdir).
    # Previous version forced a "neuro_heads_all_with_positions" subdir which
    # (a) made every consumer hardcode that path and (b) baked "neuro_" into
    # downstream code, breaking domain portability.
    os.makedirs(args.output_dir, exist_ok=True)
    merged_with_positions.save_to_disk(args.output_dir)
    logger.info("Saved merged dataset with positions: %s (%d rows)", args.output_dir, len(merged_with_positions))


if __name__ == "__main__":
    main()
