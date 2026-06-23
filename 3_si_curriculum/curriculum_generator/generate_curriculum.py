#!/usr/bin/env python3
"""
generate_curriculum.py — SI Pipeline Step 2a

Generates a curriculum of multi-hop Q&A pairs from the annotated KG.
Uses the hop-distance manifest produced by calculate_hops.py.

Usage:
  python curriculum_generator/generate_curriculum.py \\
    --manifest_path ${OUTPUT_BASE}/final_kg/all_hops_detailed.csv \\
    --output_dir    ${OUTPUT_BASE}/SI/QA_items \\
    --min_hops 2 \\
    --max_hops 3 \\
    --target_count 5000 \\
    --api_key $GOOGLE_API_KEY

Requires GOOGLE_API_KEY for Gemini API access.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import time
import random
import argparse
from pathlib import Path

# Pipeline config loader (repo root, 2 levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import get_phase_param  # noqa: E402

# Profile-driven CLI defaults: read from configs/profiles/<SI_PROFILE>.yaml::curriculum
# with fallbacks if SI_PROFILE is unset. CLI flags still override at parse time.
_DEFAULT_TARGET_COUNT = get_phase_param('curriculum', 'num_questions', 5000)
_DEFAULT_HOP_RANGE    = get_phase_param('curriculum', 'hop_range', [2, 3])
# Checkpoint cadence: persist curriculum.json every N successful questions.
# Default 100 matches the original hardcoded value (preserves paper/pilot
# behavior). Smoke profile drops to 5 so a mid-run crash doesn't lose work.
_CHECKPOINT_EVERY     = get_phase_param('curriculum', 'checkpoint_every', 100)
_DEFAULT_MIN_HOPS     = _DEFAULT_HOP_RANGE[0] if isinstance(_DEFAULT_HOP_RANGE, (list, tuple)) and len(_DEFAULT_HOP_RANGE) >= 1 else 2
_DEFAULT_MAX_HOPS     = _DEFAULT_HOP_RANGE[1] if isinstance(_DEFAULT_HOP_RANGE, (list, tuple)) and len(_DEFAULT_HOP_RANGE) >= 2 else 3
import logging
from datetime import datetime
from typing import List, Set, Tuple, Dict
from collections import deque

from curriculum_generator.generate_questions import QAGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Generate Q&A curriculum from KG hop manifest")
    ap.add_argument("--manifest_path", required=True,
                    help="Path to all_hops_detailed.csv (output of calculate_hops.py)")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for generated Q&A JSON files")
    ap.add_argument("--min_hops", type=int, default=_DEFAULT_MIN_HOPS,
                    help=f"Minimum hop distance (default: {_DEFAULT_MIN_HOPS}; from configs/profiles/<SI_PROFILE>.yaml::curriculum.hop_range)")
    ap.add_argument("--max_hops", type=int, default=_DEFAULT_MAX_HOPS,
                    help=f"Maximum hop distance (default: {_DEFAULT_MAX_HOPS}; from configs/profiles/<SI_PROFILE>.yaml::curriculum.hop_range)")
    ap.add_argument("--target_count", type=int, default=_DEFAULT_TARGET_COUNT,
                    help=f"Target number of Q&A items to generate (default: {_DEFAULT_TARGET_COUNT}; from configs/profiles/<SI_PROFILE>.yaml::curriculum.num_questions)")
    ap.add_argument("--api_key", default=os.environ.get("GOOGLE_API_KEY", ""),
                    help="Google API key for Gemini (or set GOOGLE_API_KEY env var)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def get_path_signature(paths: List[dict]) -> Tuple:
    return tuple((p["start"], p["relation"], p["end"]) for p in paths)


def load_paths_from_manifest(manifest_path: str, min_k: int, max_k: int) -> List[Dict]:
    print(f"Loading paths from manifest: {manifest_path}")
    loaded_paths = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hop_count = int(row.get("hop_distance", 0))
            except (ValueError, TypeError):
                continue
            if not (min_k <= hop_count <= max_k):
                continue

            path_steps = []
            i = 0
            while True:
                h = row.get(f"head_{i}") or row.get("head") if i == 0 else row.get(f"head_{i}")
                r = row.get(f"relation_{i}") or row.get("relation") if i == 0 else row.get(f"relation_{i}")
                t = row.get(f"tail_{i}") or row.get("tail") if i == 0 else row.get(f"tail_{i}")
                if not h or not r or not t:
                    break
                path_steps.append({"start": h, "relation": r, "end": t})
                i += 1
                if i > max_k:
                    break

            if path_steps:
                loaded_paths.append({
                    "hop_count": hop_count,
                    "path": path_steps,
                })

    print(f"Loaded {len(loaded_paths)} paths (hops {min_k}–{max_k})")
    return loaded_paths


def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("GOOGLE_API_KEY is required (--api_key or env var)")

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    paths = load_paths_from_manifest(args.manifest_path, args.min_hops, args.max_hops)
    if not paths:
        raise RuntimeError(f"No paths found for hops {args.min_hops}–{args.max_hops}. "
                           f"Check the manifest file: {args.manifest_path}")

    random.shuffle(paths)

    generator = QAGenerator(api_key=args.api_key)

    results = []
    seen_signatures: Set = set()
    attempts = 0
    max_attempts = args.target_count * 5

    # Stable filename so curriculum.sh / verify_questions.py can consume
    # without globbing. Hop range and timestamp moved into the JSON
    # metadata (added below at save time). Original filename was
    # `curriculum_dataset_hop_{min}_to_{max}_{timestamp}.json` which broke
    # the downstream hardcoded `curriculum.json` consumer in
    # scripts/phases/curriculum.sh:95 (audit bug #3).
    out_file = os.path.join(args.output_dir, "curriculum.json")

    path_queue = deque(paths)

    while len(results) < args.target_count and attempts < max_attempts:
        if not path_queue:
            path_queue = deque(random.sample(paths, len(paths)))

        path_data = path_queue.popleft()
        sig = get_path_signature(path_data["path"])

        if sig in seen_signatures:
            attempts += 1
            continue

        try:
            qa_item = generator.generate_from_path(path_data)
            if qa_item is not None:
                qa_item["hop_count"] = path_data["hop_count"]
                results.append(qa_item)
                seen_signatures.add(sig)
                if len(results) % _CHECKPOINT_EVERY == 0:
                    logger.info("Generated %d / %d items", len(results), args.target_count)
                    # Save checkpoint
                    with open(out_file, "w") as f:
                        json.dump(results, f, indent=2)
        except Exception as e:
            logger.warning("Generation failed: %s", e)
            time.sleep(1)

        attempts += 1

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Generated %d Q&A items", len(results))
    logger.info("Saved to: %s", out_file)


if __name__ == "__main__":
    main()
