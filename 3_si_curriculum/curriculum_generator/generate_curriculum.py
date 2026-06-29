#!/usr/bin/env python3
"""
generate_curriculum.py — SI Pipeline curriculum generation

Generates a curriculum of multi-hop Q&A items from the annotated KG (the hop-distance
manifest produced by calculate_hops.py).

Two modes, selected by --stage:
  all            legacy single-pass: per path run the full pipeline (pair -> quality ->
                 trace -> correctness) and write curriculum.json. Unchanged behaviour, kept
                 for smoke / backward-compat.
  pair           generate bare QA pairs -> curriculum.jsonl (stage:pair). Over-provisions by
                 curriculum.expected_yield so ~target_count survive the downstream checks.
  validate_pair  stream curriculum.jsonl; non-Gemini pair_check on stage:pair records ->
                 stage:validated_pair | drop.
  item           stream curriculum.jsonl; add a reasoning trace to stage:validated_pair
                 records -> stage:item.

The 2-LLM item consensus (stage:item -> verified|drop) is a separate step (verify_questions.py).

Requires GOOGLE_API_KEY for the Gemini stages (all/pair/item). The validate_pair stage uses
the OpenAI-SDK pair_check client (OPENAI_API_KEY / a local vLLM base_url) instead.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import math
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple, Dict

# Pipeline config loader (repo root, 2 levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import get_phase_param  # noqa: E402

from curriculum_generator.generate_questions import QAGenerator
from curriculum_generator import curriculum_io as cio

# Profile-driven CLI defaults: read from configs/profiles/<SI_PROFILE>.yaml::curriculum
# with fallbacks if SI_PROFILE is unset. CLI flags still override at parse time.
_DEFAULT_TARGET_COUNT = get_phase_param('curriculum', 'num_questions', 5000)
_DEFAULT_HOP_RANGE    = get_phase_param('curriculum', 'hop_range', [2, 3])
# Checkpoint cadence: persist curriculum.json every N successful questions (legacy `all`).
# Default 100 matches the original hardcoded value (preserves paper/pilot behavior).
_CHECKPOINT_EVERY     = get_phase_param('curriculum', 'checkpoint_every', 100)
# Parallel Gemini workers for question generation. Paper=1 (sequential, safe);
# smoke/pilot=3-5 (parallel, faster). Respects Gemini API rate limits.
_PARALLEL_WORKERS     = get_phase_param('curriculum', 'parallel_generation_workers', 1)
_DEFAULT_MIN_HOPS     = _DEFAULT_HOP_RANGE[0] if isinstance(_DEFAULT_HOP_RANGE, (list, tuple)) and len(_DEFAULT_HOP_RANGE) >= 1 else 2
_DEFAULT_MAX_HOPS     = _DEFAULT_HOP_RANGE[1] if isinstance(_DEFAULT_HOP_RANGE, (list, tuple)) and len(_DEFAULT_HOP_RANGE) >= 2 else 3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Generate Q&A curriculum from KG hop manifest")
    ap.add_argument("--manifest_path", default="",
                    help="Path to all_hops_detailed.csv (output of calculate_hops.py); "
                         "required for --stage all/pair, unused by validate_pair/item")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for the curriculum (jsonl/json + stats)")
    ap.add_argument("--stage", choices=["all", "pair", "validate_pair", "item"], default="all",
                    help="Pipeline stage (default all = legacy single-pass curriculum.json)")
    ap.add_argument("--stats_path", default="",
                    help="curriculum_stats.json path (default: <output_dir>/curriculum_stats.json)")
    ap.add_argument("--min_hops", type=int, default=_DEFAULT_MIN_HOPS,
                    help=f"Minimum hop distance (default: {_DEFAULT_MIN_HOPS}; from curriculum.hop_range)")
    ap.add_argument("--max_hops", type=int, default=_DEFAULT_MAX_HOPS,
                    help=f"Maximum hop distance (default: {_DEFAULT_MAX_HOPS}; from curriculum.hop_range)")
    ap.add_argument("--target_count", type=int, default=_DEFAULT_TARGET_COUNT,
                    help=f"Target number of FINAL Q&A items (default: {_DEFAULT_TARGET_COUNT}; from curriculum.num_questions)")
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
                loaded_paths.append({"hop_count": hop_count, "path": path_steps})

    print(f"Loaded {len(loaded_paths)} paths (hops {min_k}–{max_k})")
    return loaded_paths


def _load_paths_or_die(args) -> List[Dict]:
    if not args.manifest_path:
        raise ValueError("--manifest_path is required for this stage (all/pair)")
    paths = load_paths_from_manifest(args.manifest_path, args.min_hops, args.max_hops)
    if not paths:
        raise RuntimeError(f"No paths found for hops {args.min_hops}–{args.max_hops}. "
                           f"Check the manifest file: {args.manifest_path}")
    random.shuffle(paths)
    return paths


def _backoff_on_rate_limit(e: Exception, attempts: int) -> None:
    """Exponential backoff on Gemini 429 / RESOURCE_EXHAUSTED; log + continue otherwise."""
    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
        sleep_time = 4.0 * (2 ** min(3, attempts // 10))  # max 32s
        logger.warning("Rate limited (429): %s. Backing off %.0fs...", e, sleep_time)
        time.sleep(sleep_time)
    else:
        logger.warning("Generation failed: %s", e)


# --- Stage: all (legacy single-pass) -------------------------------------

def run_stage_all(args, generator: QAGenerator) -> None:
    paths = _load_paths_or_die(args)

    results: List[Dict] = []
    seen_signatures: Set = set()
    attempts = 0
    max_attempts = args.target_count * 5
    out_file = os.path.join(args.output_dir, "curriculum.json")

    logger.info("Starting Q&A generation (stage=all) with %d parallel worker(s)", _PARALLEL_WORKERS)

    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
        futures = {}
        pending_paths = deque(random.sample(paths, len(paths)))

        while len(futures) < _PARALLEL_WORKERS and len(results) < args.target_count and attempts < max_attempts:
            if not pending_paths:
                pending_paths = deque(random.sample(paths, len(paths)))
            path_data = pending_paths.popleft()
            sig = get_path_signature(path_data["path"])
            if sig not in seen_signatures:
                future = executor.submit(generator.generate_from_path, path_data)
                futures[future] = (path_data, sig)
                attempts += 1

        while futures and len(results) < args.target_count and attempts < max_attempts:
            for future in as_completed(futures):
                path_data, sig = futures.pop(future)
                try:
                    qa_item = future.result()
                    if qa_item is not None:
                        qa_item["hop_count"] = path_data["hop_count"]
                        results.append(qa_item)
                        seen_signatures.add(sig)
                        logger.info("Generated %d / %d items", len(results), args.target_count)
                        if len(results) % _CHECKPOINT_EVERY == 0:
                            with open(out_file, "w") as f:
                                json.dump(results, f, indent=2)
                            logger.info("Checkpoint saved at %d items", len(results))
                except Exception as e:
                    _backoff_on_rate_limit(e, attempts)

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


# --- Stage: pair (bare QA pairs -> curriculum.jsonl) ---------------------

def run_stage_pair(args, generator: QAGenerator, jsonl_path: str, stats_path: str) -> None:
    paths = _load_paths_or_die(args)
    expected_yield = float(get_phase_param('curriculum', 'expected_yield', 0.55) or 0.55)
    pair_target = math.ceil(args.target_count / max(expected_yield, 0.01))
    logger.info("pair stage: generating ~%d pairs (target_count=%d / expected_yield=%.2f)",
                pair_target, args.target_count, expected_yield)

    seen: Set = set()
    generated = 0
    attempts = 0
    max_attempts = pair_target * 5
    idx = 0

    with cio.open_jsonl_writer(jsonl_path) as out:
        with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
            while generated < pair_target and attempts < max_attempts:
                # Build a batch of fresh-signature paths to generate in parallel.
                batch = []
                while len(batch) < max(_PARALLEL_WORKERS, 1) and attempts < max_attempts:
                    if idx >= len(paths):
                        random.shuffle(paths)
                        idx = 0
                    path_data = paths[idx]
                    idx += 1
                    sig = get_path_signature(path_data["path"])
                    if sig in seen:
                        continue
                    seen.add(sig)
                    batch.append(path_data)
                    attempts += 1
                if not batch:
                    break

                futures = {executor.submit(generator.generate_pair_from_path, pd): pd for pd in batch}
                for future in as_completed(futures):
                    try:
                        pair = future.result()
                    except Exception as e:
                        _backoff_on_rate_limit(e, attempts)
                        continue
                    if pair is not None:
                        cio.write_record(out, {"stage": cio.STAGE_PAIR, **pair})
                        generated += 1
                logger.info("pairs: %d/%d (attempts %d)", generated, pair_target, attempts)

    counts = cio.yield_counts(attempts, generated, target=pair_target, expected_yield=expected_yield)
    cio.write_stat(stats_path, "generate_qa_pair", counts)
    logger.info("pair stage done: %d pairs from %d attempts -> %s", generated, attempts, jsonl_path)


# --- Stage: validate_pair (non-Gemini pair check) ------------------------

def run_stage_validate_pair(args, jsonl_path: str, stats_path: str) -> None:
    from curriculum_generator.pair_check import check_pair  # lazy: needs openai
    chunk_size = int(get_phase_param('curriculum', 'pair_check_chunk_size', 64))
    stats = {"in": 0, "out": 0, "dropped": 0, "drop_reasons": {}}

    def process(chunk: List[Dict]) -> None:
        targets = [r for r in chunk if r.get("stage") == cio.STAGE_PAIR]
        if not targets:
            return
        with ThreadPoolExecutor(max_workers=max(_PARALLEL_WORKERS, 1)) as executor:
            futures = {
                executor.submit(check_pair, r.get("question", ""), r.get("answer", ""),
                                cio.format_path(r.get("paths", []))): r
                for r in targets
            }
            for future in as_completed(futures):
                r = futures[future]
                stats["in"] += 1
                try:
                    ok = bool(future.result())
                except Exception as e:
                    ok = False
                    r["pair_check_error"] = str(e)[:200]
                r["pair_verdict"] = ok
                if ok:
                    r["stage"] = cio.STAGE_VALIDATED_PAIR
                    stats["out"] += 1
                else:
                    r["stage"] = cio.STAGE_DROP
                    r["drop_reason"] = "pair_check"
                    stats["dropped"] += 1
                    stats["drop_reasons"]["pair_check"] = stats["drop_reasons"].get("pair_check", 0) + 1

    cio.transform_jsonl(jsonl_path, jsonl_path, process, chunk_size)
    counts = cio.yield_counts(stats["in"], stats["out"], drop_reasons=stats["drop_reasons"])
    cio.write_stat(stats_path, "validate_qa_pair", counts)
    logger.info("validate_pair done: %d/%d pairs kept -> %s", stats["out"], stats["in"], jsonl_path)


# --- Stage: item (add reasoning trace) -----------------------------------

def run_stage_item(args, generator: QAGenerator, jsonl_path: str, stats_path: str) -> None:
    chunk_size = int(get_phase_param('curriculum', 'item_chunk_size', 64))
    stats = {"in": 0, "out": 0, "dropped": 0, "drop_reasons": {}}

    def process(chunk: List[Dict]) -> None:
        targets = [r for r in chunk if r.get("stage") == cio.STAGE_VALIDATED_PAIR]
        if not targets:
            return
        with ThreadPoolExecutor(max_workers=max(_PARALLEL_WORKERS, 1)) as executor:
            futures = {executor.submit(generator.generate_item_from_pair, r): r for r in targets}
            for future in as_completed(futures):
                r = futures[future]
                stats["in"] += 1
                try:
                    item = future.result()
                except Exception as e:
                    item = None
                    _backoff_on_rate_limit(e, stats["in"])
                    r["item_error"] = str(e)[:200]
                if item is not None:
                    r["explanation"] = item["explanation"]
                    r["question_and_explanation"] = item["question_and_explanation"]
                    r["stage"] = cio.STAGE_ITEM
                    stats["out"] += 1
                else:
                    r["stage"] = cio.STAGE_DROP
                    r["drop_reason"] = "trace_gen"
                    stats["dropped"] += 1
                    stats["drop_reasons"]["trace_gen"] = stats["drop_reasons"].get("trace_gen", 0) + 1

    cio.transform_jsonl(jsonl_path, jsonl_path, process, chunk_size)
    counts = cio.yield_counts(stats["in"], stats["out"], drop_reasons=stats["drop_reasons"])
    cio.write_stat(stats_path, "generate_qa_item", counts)
    logger.info("item stage done: %d/%d traces added -> %s", stats["out"], stats["in"], jsonl_path)


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    jsonl_path = os.path.join(args.output_dir, "curriculum.jsonl")
    stats_path = args.stats_path or os.path.join(args.output_dir, "curriculum_stats.json")

    if args.stage == "all":
        if not args.api_key:
            raise ValueError("GOOGLE_API_KEY is required for --stage all (--api_key or env var)")
        run_stage_all(args, QAGenerator(api_key=args.api_key))
    elif args.stage == "pair":
        if not args.api_key:
            raise ValueError("GOOGLE_API_KEY is required for --stage pair")
        run_stage_pair(args, QAGenerator(api_key=args.api_key), jsonl_path, stats_path)
    elif args.stage == "validate_pair":
        run_stage_validate_pair(args, jsonl_path, stats_path)
    elif args.stage == "item":
        if not args.api_key:
            raise ValueError("GOOGLE_API_KEY is required for --stage item")
        run_stage_item(args, QAGenerator(api_key=args.api_key), jsonl_path, stats_path)


if __name__ == "__main__":
    main()
