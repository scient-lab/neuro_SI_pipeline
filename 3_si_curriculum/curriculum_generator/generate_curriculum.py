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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# Parallel Gemini workers for question generation. Paper=1 (sequential, safe).
# Smoke/pilot=3-5 (parallel, faster). Respects Gemini API rate limits.
_PARALLEL_WORKERS     = get_phase_param('curriculum', 'parallel_generation_workers', 1)
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

    # Parallel Gemini question generation using ThreadPoolExecutor.
    # Paper-grade: 1 worker (sequential, safe for API quotas).
    # Smoke/pilot: 3-5 workers (parallel, faster: 3-4h → ~1h for 100q).
    # Rate-limit backoff: Gemini 429 errors trigger exponential sleep.
    logger.info(f"Starting Q&A generation with {_PARALLEL_WORKERS} parallel worker(s)")

    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
        futures = {}  # Map future → path_data
        pending_paths = deque(random.sample(paths, len(paths)))  # Refilled as we go

        # Prime the executor with initial batch
        while len(futures) < _PARALLEL_WORKERS and len(results) < args.target_count and attempts < max_attempts:
            if not pending_paths:
                pending_paths = deque(random.sample(paths, len(paths)))

            path_data = pending_paths.popleft()
            sig = get_path_signature(path_data["path"])

            if sig not in seen_signatures:
                future = executor.submit(generator.generate_from_path, path_data)
                futures[future] = (path_data, sig)
                attempts += 1

        # Process results as they complete
        while futures and len(results) < args.target_count and attempts < max_attempts:
            for future in as_completed(futures):
                path_data, sig = futures.pop(future)

                try:
                    qa_item = future.result()
                    if qa_item is not None:
                        qa_item["hop_count"] = path_data["hop_count"]
                        results.append(qa_item)
                        seen_signatures.add(sig)
                        logger.info(f"Generated {len(results)} / {args.target_count} items")

                        if len(results) % _CHECKPOINT_EVERY == 0:
                            with open(out_file, "w") as f:
                                json.dump(results, f, indent=2)
                            logger.info(f"Checkpoint saved at {len(results)} items")
                except Exception as e:
                    # Rate limit: exponential backoff on 429 (Gemini quota)
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        logger.warning(f"Rate limited (429): {e}. Backing off...")
                        sleep_time = 4.0 * (2 ** min(3, attempts // 10))  # Max 32s sleep
                        time.sleep(sleep_time)
                    else:
                        logger.warning(f"Generation failed: {e}")

                # Refill executor with next path if we haven't hit target
                if len(results) < args.target_count and attempts < max_attempts:
                    if not pending_paths:
                        pending_paths = deque(random.sample(paths, len(paths)))

                    if pending_paths:
                        path_data = pending_paths.popleft()
                        sig = get_path_signature(path_data["path"])

                        if sig not in seen_signatures:
                            future = executor.submit(generator.generate_from_path, path_data)
                            futures[future] = (path_data, sig)
                            attempts += 1

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Generated %d Q&A items", len(results))
    logger.info("Saved to: %s", out_file)


if __name__ == "__main__":
    main()
