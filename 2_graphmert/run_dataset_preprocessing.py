#!/usr/bin/env python3
"""
run_dataset_preprocessing.py  — GraphMERT Pipeline Step 4

Runs the co-occurrence grounding step that builds GraphMERT training samples.

This replaces the old two-step workflow (graphmert_bridge.py → preprocessing).
The bridge logic is now fully integrated here.

Usage:
  python run_dataset_preprocessing.py \\
    --yaml_file   launch_configs/args_mlm.yaml \\
    --seed_kg_path  ${OUTPUT_BASE}/final_seedkg/neuroscience_kg.csv \\
    --train_src     ${OUTPUT_BASE}/graphmert/llm_relations/relations_cleaned_train \\
    --eval_src      ${OUTPUT_BASE}/graphmert/llm_relations/relations_cleaned_eval \\
    --tokenizer     ${OUTPUT_BASE}/graphmert/stable_tokenizer \\
    --output_dir    ${OUTPUT_BASE}/graphmert/preprocessed

Outputs:
  ${OUTPUT_BASE}/graphmert/preprocessed/
    relation_map.json
    ready_for_training_train/
    ready_for_training_eval/
"""

import argparse
import logging
import os
import sys

from utils import dataset_preprocessing_utils

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_YAML = os.path.join(
    os.path.dirname(__file__), "launch_configs", "args_mlm.yaml"
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="GraphMERT dataset preprocessing (with integrated bridge)"
    )
    ap.add_argument("--yaml_file", default=DEFAULT_YAML,
                    help="Path to args_mlm.yaml (default: launch_configs/args_mlm.yaml)")
    ap.add_argument("--seed_kg_path", default=None,
                    help="Path to seed KG CSV (head, relation, tail columns)")
    ap.add_argument("--train_src", default=None,
                    help="Path to cleaned train LLM-relations HF dataset dir")
    ap.add_argument("--eval_src", default=None,
                    help="Path to cleaned eval LLM-relations HF dataset dir")
    ap.add_argument("--tokenizer", default=None,
                    help="Path to stable_tokenizer (output of run_tokenization.py)")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write grounded training datasets")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    yaml_file = args.yaml_file
    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(os.path.dirname(__file__), yaml_file)
    if not os.path.isfile(yaml_file):
        logger.error("YAML config not found: %s", yaml_file)
        sys.exit(1)

    logger.info("Using YAML config: %s", yaml_file)
    dataset_preprocessing_utils.main(
        yaml_file=yaml_file,
        seed_kg_path=args.seed_kg_path,
        train_src=args.train_src,
        eval_src=args.eval_src,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
    )
    logger.info("Preprocessing done. Run 'python run_mlm.py --yaml_file %s' next.", yaml_file)
