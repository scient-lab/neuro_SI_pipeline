#!/usr/bin/env python3
# coding=utf-8
"""
add_llm_relations.py  — GraphMERT Pipeline Step 3a

For each tokenized snippet with known entity positions (from find_heads_positions.py),
assigns allowed knowledge-graph relations using a vLLM model.

Usage:
  python utils/relation_matching/add_llm_relations.py \\
    --dataset_path  ${OUTPUT_BASE}/graphmert/dataset_with_heads/neuro_heads_all_with_positions \\
    --output_root   ${OUTPUT_BASE}/graphmert/llm_relations \\
    --output_name   relations_all \\
    --model_id      /path/to/qwen3-14b \\
    --tokenizer     ${OUTPUT_BASE}/graphmert/stable_tokenizer

Run via SLURM:
  sbatch slurm/add_llm_relations.slurm
"""

import os
import re
import json
import ast
import sys
import time
import socket
import logging
import argparse
from typing import Any, Dict, List, Tuple, Optional

from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from relation_match_prompts import (
    SYSTEM_CONTEXT,
    example_user_1, example_assistant_1, example_explanation_1,
    example_user_2, example_assistant_2, example_explanation_2,
    example_user_3, example_assistant_3, example_explanation_3,
    expanded_kg_example_user_4, expanded_kg_assistant_4, expanded_kg_explanation_4,
    expanded_kg_example_user_5, expanded_kg_assistant_5, expanded_kg_explanation_5,
)

logger = logging.getLogger("add_llm_relations")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

POS_EXAMPLES: List[Tuple[str, str, str]] = [
    (example_user_1, example_assistant_1, example_explanation_1),
    (example_user_2, example_assistant_2, example_explanation_2),
    (example_user_3, example_assistant_3, example_explanation_3),
    (expanded_kg_example_user_4, expanded_kg_assistant_4, expanded_kg_explanation_4),
    (expanded_kg_example_user_5, expanded_kg_assistant_5, expanded_kg_explanation_5),
]


def parse_args():
    ap = argparse.ArgumentParser(description="Add LLM-predicted relations to entity dataset")
    ap.add_argument("--dataset_path", required=True,
                    help="Path to dataset with head_positions (from find_heads_positions.py)")
    ap.add_argument("--output_root", required=True,
                    help="Output root directory")
    ap.add_argument("--output_name", default="relations_all",
                    help="Output subdirectory name (default: relations_all)")
    ap.add_argument("--model_id", required=True,
                    help="Path to local vLLM model")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to stable tokenizer")
    ap.add_argument("--map_batch_size", type=int, default=96)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--subset", type=int, default=0,
                    help="If > 0, only process this many rows (for debugging)")
    ap.add_argument("--debug_raw_limit", type=int, default=50)
    return ap.parse_args()


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _safe_json_loads_any(x: Any) -> Any:
    if x is None: return None
    if isinstance(x, (dict, list)): return x
    if isinstance(x, str):
        s = x.strip()
        if not s: return None
        try: return json.loads(s)
        except Exception: return None
    return None


def extract_rightmost_json_object(response: str) -> str:
    if not response: return ""
    response = re.sub(r"(?m)^```.*\n?", "", response).strip()
    start, end = response.rfind("{"), response.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = response[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            pass
    matches = _JSON_OBJ_RE.findall(response)
    if matches:
        candidate = matches[-1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            pass
    return ""


def format_vllm_chat_messages(batch, tokenizer, pos_examples, no_think: bool = True):
    prompts: List[List[Dict[str, str]]] = []
    to_call_indices: List[int] = []
    input_ids_list = batch["input_ids"]
    head_positions_list = batch.get("head_positions", [None] * len(input_ids_list))

    # Qwen3 thinking control. Pattern-match relation extraction doesn't
    # benefit from <think>; suppress with /no_think to reclaim tokens.
    # Config knob: graphmert.add_llm_relations_no_think (defaults true).
    think_suffix = " /no_think" if no_think else ""

    for i in range(len(input_ids_list)):
        hp_raw = head_positions_list[i]
        hp_obj = _safe_json_loads_any(hp_raw)
        if not isinstance(hp_obj, dict) or not hp_obj:
            continue
        heads = list(hp_obj.keys())
        if not heads:
            continue
        to_call_indices.append(i)
        seq = tokenizer.decode(input_ids_list[i], skip_special_tokens=True)
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_CONTEXT}]
        for u, a, e in pos_examples:
            messages.extend([
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
                {"role": "user", "content": "Explanation:"},
                {"role": "assistant", "content": e},
            ])
        query = f"Input:\nsequence: {seq}\nheads: {heads}\n\nOutput:{think_suffix}"
        messages.append({"role": "user", "content": query})
        prompts.append(messages)

    return prompts, to_call_indices


def main() -> None:
    args = parse_args()
    logger.info("HOST=%s  model=%s", socket.gethostname(), args.model_id)

    # Read Qwen3 thinking control from configs/default.yaml. Same pattern as
    # graphrag_index.py uses for extract.no_think — pattern-match steps
    # suppress <think> to avoid wasting the token budget on output that the
    # downstream parser ignores. Override per-profile to flip if a new model
    # needs thinking on.
    try:
        # pipeline_config.py lives at repo root; sys.path setup mirrors
        # 1_seed_kg/graphrag_index.py's approach.
        import os as _os, sys as _sys
        _repo_root = _os.environ.get("REPO_ROOT") or _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), "..", "..", "..")
        )
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from pipeline_config import get_phase_param
        no_think = bool(get_phase_param('graphmert', 'add_llm_relations_no_think', True))
    except Exception as e:
        logger.warning("could not read graphmert.add_llm_relations_no_think (%s) — defaulting True", e)
        no_think = True
    logger.info("Qwen3 thinking: %s", "OFF (/no_think)" if no_think else "ON")

    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info("Loading dataset: %s", args.dataset_path)
    dataset: Dataset = load_from_disk(args.dataset_path)
    logger.info("dataset_len=%d  cols=%s", len(dataset), dataset.column_names)

    if args.subset > 0:
        n = min(args.subset, len(dataset))
        dataset = dataset.select(range(n))
        logger.info("Using subset of %d rows", n)

    logger.info("Initializing vLLM: %s", args.model_id)
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192)

    seen_debug_raw = 0

    def add_relations(batch: Dict[str, Any], indices: List[int]) -> Dict[str, Any]:
        nonlocal seen_debug_raw
        prompts, to_call_indices = format_vllm_chat_messages(batch, tokenizer, POS_EXAMPLES, no_think=no_think)
        relations_json: List[str] = [""] * len(batch["input_ids"])
        raw_model_out: List[str] = [""] * len(batch["input_ids"])

        if prompts:
            outputs = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=False)
            for out, local_i in zip(outputs, to_call_indices):
                raw = out.outputs[0].text if out.outputs else ""
                cleaned = extract_rightmost_json_object(raw)
                relations_json[local_i] = cleaned
                if seen_debug_raw < args.debug_raw_limit:
                    raw_model_out[local_i] = raw
                    seen_debug_raw += 1

        batch["relations_json"] = relations_json
        batch["relations_raw"] = raw_model_out
        return batch

    logger.info("Running Dataset.map() (batch_size=%d)...", args.map_batch_size)
    t0 = time.time()
    dataset_out = dataset.map(
        add_relations, batched=True, batch_size=args.map_batch_size,
        with_indices=True, load_from_cache_file=False,
        desc="LLM relation matching",
    )
    logger.info("map() done in %.1f sec", time.time() - t0)

    os.makedirs(args.output_root, exist_ok=True)
    out_dir = os.path.join(args.output_root, args.output_name)
    dataset_out.save_to_disk(out_dir)
    logger.info("Saved: %s (%d rows)", out_dir, len(dataset_out))


if __name__ == "__main__":
    main()
