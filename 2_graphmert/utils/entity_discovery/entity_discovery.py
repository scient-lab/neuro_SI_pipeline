#!/usr/bin/env python3
"""
entity_discovery.py  — GraphMERT Pipeline Step 2a

Identifies neuroscience entity mentions in each tokenized snippet using
a vLLM model. Processes all rows in a single job.

Usage:
  python utils/entity_discovery/entity_discovery.py \\
    --tokenized_dir  ${OUTPUT_BASE}/graphmert/tokenized_inputs/train_tokenized \\
    --output_dir     ${OUTPUT_BASE}/graphmert/dataset_with_heads/chunks \\
    --model_id       /path/to/qwen3-32b \\
    --tokenizer      ${OUTPUT_BASE}/graphmert/stable_tokenizer

Run via SLURM:
  sbatch slurm/entity_discovery.slurm
"""

import os
import re
import sys
import json
import time
import socket
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from vllm import LLM, SamplingParams
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer

from entity_discovery_prompts import SYSTEM_CONTEXT as SYSTEM_PROMPT

USER_TEMPLATE = "Input:\n{text}\n\nOutput:"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Entity discovery over tokenized text snippets")
    ap.add_argument("--tokenized_dir", required=True,
                    help="Path to tokenized train dataset (output of run_tokenization.py)")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for entity-annotated dataset chunks")
    ap.add_argument("--model_id", required=True,
                    help="Path to local vLLM model (e.g. Qwen3-32B)")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to stable tokenizer (output of run_tokenization.py)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--rows_target", type=int, default=10_000_000,
                    help="Max rows to process (default: all)")
    ap.add_argument("--debug_raw_limit", type=int, default=50)
    return ap.parse_args()


ARGS = parse_args()


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 200
    min_p: float = 0.0


def _extract_json_list(raw: str) -> List[str]:
    """Extract a JSON list of entity strings from model output."""
    raw = re.sub(r"</?think>", "", raw).strip()

    # Try direct JSON parse
    for match in re.finditer(r"\[.*?\]", raw, re.DOTALL):
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except Exception:
            pass

    # Fallback: extract quoted strings
    return [m.lower().strip() for m in re.findall(r'"([^"]+)"', raw) if m.strip()]


def main():
    args = ARGS
    logger.info("HOST=%s  FILE=%s", socket.gethostname(), os.path.abspath(__file__))
    logger.info("tokenized_dir=%s  output_dir=%s", args.tokenized_dir, args.output_dir)

    # Qwen3 thinking control. Pattern-match step; <think> trace is byproduct
    # waste here. Default True. Override via configs/default.yaml::graphmert.
    # entity_discovery_no_think.
    try:
        import os as _os, sys as _sys
        _repo_root = _os.environ.get("REPO_ROOT") or _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "..")
        )
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from pipeline_config import get_phase_param
        no_think = bool(get_phase_param('graphmert', 'entity_discovery_no_think', True))
    except Exception as e:
        logger.warning("could not read graphmert.entity_discovery_no_think (%s) — defaulting True", e)
        no_think = True
    think_suffix = " /no_think" if no_think else ""
    logger.info("Qwen3 thinking: %s", "OFF (/no_think)" if no_think else "ON")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Loading dataset: %s", args.tokenized_dir)
    dataset = load_from_disk(args.tokenized_dir)
    n = min(args.rows_target, len(dataset))
    if n < len(dataset):
        dataset = dataset.select(range(n))
    logger.info("Dataset rows to process: %d", n)

    logger.info("Initializing vLLM: %s", args.model_id)
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        temperature=SamplingConfig.temperature,
        top_p=SamplingConfig.top_p,
        top_k=SamplingConfig.top_k,
        max_tokens=SamplingConfig.max_tokens,
    )

    responses_all: List[str] = []
    t0 = time.time()

    for batch_start in range(0, n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        for ex in batch:
            text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=text) + think_suffix},
            ]
            prompts.append(messages)

        outputs = llm.chat(prompts, sampling_params=sampling)
        for out in outputs:
            responses_all.append(out.outputs[0].text if out.outputs else "")

        if (batch_end % 1000 == 0) or batch_end == n:
            elapsed = time.time() - t0
            logger.info("Processed %d/%d rows (%.1f rows/s)", batch_end, n, batch_end / max(elapsed, 1e-6))

    # Build annotated dataset
    def annotate(examples, indices):
        head_positions_list = []
        for i, idx in enumerate(indices):
            raw = responses_all[idx] if idx < len(responses_all) else ""
            entities = _extract_json_list(raw)
            # Map entity strings to approximate token position (first occurrence)
            input_ids = examples["input_ids"][i]
            text = tokenizer.decode(input_ids, skip_special_tokens=True).lower()
            positions = {}
            for ent in entities:
                pos = text.find(ent)
                if pos >= 0:
                    # Convert char position to approximate token index
                    tok_pos = min(len(tokenizer.encode(text[:pos], add_special_tokens=False)), 511)
                    positions[ent] = tok_pos
            head_positions_list.append(json.dumps(positions))
        examples["head_positions"] = head_positions_list
        return examples

    dataset_out = dataset.map(
        annotate, batched=True, with_indices=True, batch_size=args.batch_size,
        load_from_cache_file=False, desc="Annotating entities",
    )

    chunk_out = os.path.join(args.output_dir, "chunk_0")
    dataset_out.save_to_disk(chunk_out)
    logger.info("Saved annotated dataset to: %s (%d rows)", chunk_out, len(dataset_out))
    logger.info("Total time: %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
