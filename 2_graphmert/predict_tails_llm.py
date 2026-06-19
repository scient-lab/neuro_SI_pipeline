#!/usr/bin/env python3
# coding=utf-8
"""
predict_tails_llm.py  — GraphMERT Pipeline Step 6

Generates new tail entities for each (head, relation) pair using a vLLM model.
Supports SLURM array jobs for parallel sharding.

Usage:
  python predict_tails_llm.py \\
    --model_id     /path/to/qwen3-32b \\
    --tokenizer    ${OUTPUT_BASE}/graphmert/stable_tokenizer \\
    --dataset      ${OUTPUT_BASE}/graphmert/llm_relations/relations_cleaned_train \\
    --output_dir   ${OUTPUT_BASE}/graphmert/graphmert_kg/predictions \\
    --num_shards   4 \\
    --shard_id     0

SLURM array:
  #SBATCH --array=0-3
  python predict_tails_llm.py --num_shards 4  # shard_id auto-detected from SLURM_ARRAY_TASK_ID
"""

import os

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("DATASETS_DISABLE_CACHING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Pipeline config loader (repo root). Single source of truth for allowed
# relations and sampling/vllm settings — see also 1_seed_kg/prompts_kg.py and
# 3_si_curriculum/curriculum_generator/generate_questions.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_config import get_phase_param, get_relations  # noqa: E402


logger = logging.getLogger("predict_tails_llm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


ALLOWED_RELATIONS = set(get_relations())

# Sourced from configs/default.yaml::graphmert.predict_* (fallbacks preserved).
TENSOR_PARALLEL_SIZE   = get_phase_param('graphmert', 'predict_tensor_parallel_size', 1)
MAX_MODEL_LEN          = get_phase_param('graphmert', 'predict_max_model_len', 8192)
GPU_MEMORY_UTILIZATION = get_phase_param('graphmert', 'predict_gpu_memory_utilization', 0.90)
TEMPERATURE            = get_phase_param('graphmert', 'predict_temperature', 0.4)
TOP_P                  = get_phase_param('graphmert', 'predict_top_p', 0.95)
MAX_NEW_TOKENS         = get_phase_param('graphmert', 'predict_max_new_tokens', 512)
TAILS_MAX              = get_phase_param('graphmert', 'predict_tails_max', 50)
PROMPT_TEXT_MAX_CHARS  = get_phase_param('graphmert', 'predict_prompt_text_max_chars', 12000)

# Engine flags (not tunable per-run; kept as code constants).
ENFORCE_EAGER = False
TRUST_REMOTE_CODE = True


SYSTEM_PROMPT = """
You extract knowledge-graph triples from neuroscience text.

PRIMARY GOAL: MAXIMIZE RECALL.
Extract AS MANY valid tail entities as possible for the given HEAD and RELATION.

Guidelines:
- Use ONLY the provided RELATION.
- If a tail is plausibly supported by the TEXT (even indirectly), INCLUDE IT.
- Prefer short, canonical entity names.
- Do NOT invent facts not grounded in the TEXT.

Return ONLY a single JSON object with this schema:
{
  "head": string,
  "relation": string,
  "tails": [string, ...],
  "reason_if_none": string
}
""".strip()

ASSISTIVE_EXAMPLES: List[Dict[str, Any]] = [
    {"text": "The hippocampus is part of the limbic system.", "head": "hippocampus", "relation": "part_of",
     "json": {"head": "hippocampus", "relation": "part_of", "tails": ["limbic system"], "reason_if_none": ""}},
    {"text": "Dopamine binds to D1 and D2 receptors.", "head": "dopamine", "relation": "binds_to",
     "json": {"head": "dopamine", "relation": "binds_to", "tails": ["D1 receptors", "D2 receptors"], "reason_if_none": ""}},
    {"text": "Glutamate activates NMDA and AMPA receptors.", "head": "glutamate", "relation": "activates",
     "json": {"head": "glutamate", "relation": "activates", "tails": ["NMDA receptors", "AMPA receptors"], "reason_if_none": ""}},
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="Path to local vLLM model")
    ap.add_argument("--tokenizer", required=True, help="Path to stable tokenizer")
    ap.add_argument("--dataset", required=True, help="Path to cleaned LLM relations dataset")
    ap.add_argument("--output_dir", required=True, help="Output directory for prediction CSVs")
    ap.add_argument("--num_shards", type=int, default=4)
    ap.add_argument("--shard_id", type=int, default=None,
                    help="Override SLURM_ARRAY_TASK_ID if set")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=100)
    return ap.parse_args()


def build_system_prompt_with_examples() -> str:
    lines = [SYSTEM_PROMPT, "", "EXAMPLES:"]
    for ex in ASSISTIVE_EXAMPLES:
        lines += ["", "TEXT: " + ex["text"], "HEAD: " + ex["head"],
                  "RELATION: " + ex["relation"], "JSON: " + json.dumps(ex["json"])]
    return "\n".join(lines).strip()


def _safe_json_loads(s: Any) -> Any:
    if s is None: return {}
    if isinstance(s, (dict, list)): return s
    s = str(s).strip()
    if not s: return {}
    try: return json.loads(s)
    except Exception: return {}


def _clip_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars: return text
    return text[:max_chars // 2] + "\n...\n" + text[-(max_chars // 2):]


def _normalize_tail(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    if len(x) >= 2 and x[0] == x[-1] in ('"', "'"):
        x = x[1:-1].strip()
    return x


def _dedup_preserve_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _extract_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    if not isinstance(s, str): return None
    s = s.strip()
    for pattern in [r"```(?:json)?\s*({.*?})\s*```", r"({.*})"]:
        m = re.search(pattern, s, flags=re.DOTALL)
        if m:
            blob = m.group(1).strip()
            blob = blob[:blob.rfind("}") + 1] if blob.rfind("}") != -1 else blob
            try: return json.loads(blob)
            except Exception: pass
    return None


def build_user_prompt(head: str, relation: str, text: str) -> str:
    return (f"HEAD: {head}\nRELATION: {relation}\nTAILS_MAX: {TAILS_MAX}\n"
            f"TEXT:\n{_clip_text(text, PROMPT_TEXT_MAX_CHARS)}\n\nReturn ONLY the JSON object.")


def explode_queries(ds, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    out = []
    for i in range(len(ds)):
        ex = ds[i]
        # Same fallback pattern as dataset_preprocessing_utils.py:155 — the
        # upstream producer (clean_llm_relations.py) doesn't write an 'id'
        # column, so we use the row index as a deterministic substitute.
        # cid propagates as 'id' in query/triple output rows below for join
        # tracking; row index is unique-per-source-row, which is what matters.
        cid = ex.get("id", i)
        cleaned = _safe_json_loads(ex.get("cleaned_relations_json", "{}"))
        if not isinstance(cleaned, dict) or not cleaned:
            continue
        try:
            text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        except Exception:
            text = ""
        for head_lower, rels in cleaned.items():
            head_lower = str(head_lower).strip().lower()
            if not head_lower or not isinstance(rels, list):
                continue
            for rel in rels:
                rel = str(rel).strip()
                if rel in ALLOWED_RELATIONS:
                    out.append({"id": cid, "head": head_lower, "relation": rel, "text": text})
    return out


def infer_batch(llm, sys_prompt, batch, no_think: bool = False) -> List[Dict[str, Any]]:
    # Qwen3 thinking control. Default False (thinking ON) — empirical
    # regression on Purves showed quality collapse without thinking.
    # configs/default.yaml::graphmert.predict_tails_no_think.
    think_suffix = " /no_think" if no_think else ""
    conversations = [
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": build_user_prompt(q["head"], q["relation"], q["text"]) + think_suffix}]
        for q in batch
    ]
    sampling = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS, stop=["\n\n\n"])
    outs = llm.chat(conversations, sampling_params=sampling)

    results = []
    for q, out in zip(batch, outs):
        raw_text = out.outputs[0].text if out.outputs else ""
        parsed = _extract_first_json_object(raw_text) or {}
        tails = parsed.get("tails", [])
        if not isinstance(tails, list): tails = []
        tails = [_normalize_tail(t) for t in tails if _normalize_tail(t)]
        tails = _dedup_preserve_order(tails)[:TAILS_MAX]
        reason = parsed.get("reason_if_none", "") if not tails else ""
        results.append({"tails": tails, "reason_if_none": reason,
                        "relation_mismatch": str(parsed.get("relation", q["relation"])).strip() != q["relation"],
                        "raw_model_output": raw_text})
    return results


def main():
    args = parse_args()

    shard_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    shard_id = args.shard_id if args.shard_id is not None else (int(shard_env) if shard_env else 0)
    num_shards = args.num_shards

    os.makedirs(args.output_dir, exist_ok=True)
    out_exploded = os.path.join(args.output_dir, f"predictions_shard{shard_id}_of{num_shards}.csv")
    out_queries  = os.path.join(args.output_dir, f"queries_shard{shard_id}_of{num_shards}.csv")

    logger.info("SHARD %d/%d  model=%s", shard_id, num_shards, args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    ds = load_from_disk(args.dataset)
    logger.info("Loaded dataset: %d rows", len(ds))

    queries_all = explode_queries(ds, tokenizer)
    if not queries_all:
        raise RuntimeError("No queries produced after filtering. Check dataset and allowed relations.")

    queries = [q for idx, q in enumerate(queries_all) if (idx % num_shards) == shard_id]
    logger.info("Shard queries: %d / %d total", len(queries), len(queries_all))

    llm = LLM(model=args.model_id, tensor_parallel_size=TENSOR_PARALLEL_SIZE,
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
              enforce_eager=ENFORCE_EAGER, trust_remote_code=TRUST_REMOTE_CODE)
    sys_prompt = build_system_prompt_with_examples()

    # Qwen3 thinking control. configs/default.yaml::graphmert.predict_tails_no_think
    try:
        import os as _os, sys as _sys
        _repo_root = _os.environ.get("REPO_ROOT") or _os.path.abspath(
            _os.path.join(_os.path.dirname(__file__), "..")
        )
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from pipeline_config import get_phase_param
        no_think = bool(get_phase_param('graphmert', 'predict_tails_no_think', False))
    except Exception as e:
        logger.warning("could not read graphmert.predict_tails_no_think (%s) — defaulting False", e)
        no_think = False
    logger.info("Qwen3 thinking: %s", "OFF (/no_think)" if no_think else "ON")

    query_rows, triple_rows = [], []
    t0 = time.time()
    done = 0

    for start in range(0, len(queries), args.batch_size):
        batch = queries[start:start + args.batch_size]
        batch_res = infer_batch(llm, sys_prompt, batch, no_think=no_think)

        for q, r in zip(batch, batch_res):
            tails = r["tails"]
            query_rows.append({"id": q["id"], "head": q["head"], "relation": q["relation"],
                                "tails_json": json.dumps(tails), "reason_if_none": r["reason_if_none"],
                                "relation_mismatch": r["relation_mismatch"]})
            if not tails:
                triple_rows.append({"id": q["id"], "head": q["head"], "relation": q["relation"],
                                    "tail": "", "query_had_no_tails": True})
            else:
                for tail in tails:
                    triple_rows.append({"id": q["id"], "head": q["head"], "relation": q["relation"],
                                        "tail": tail, "query_had_no_tails": False})
            done += 1

        if done % args.progress_every == 0 or done == len(queries):
            elapsed = time.time() - t0
            logger.info("Progress %d/%d (%.2f q/s)", done, len(queries), done / max(elapsed, 1e-6))

    pd.DataFrame(triple_rows).to_csv(out_exploded, index=False)
    pd.DataFrame(query_rows).to_csv(out_queries, index=False)
    logger.info("Saved: %s (%d triples)", out_exploded, len(triple_rows))
    logger.info("Saved: %s (%d queries)", out_queries, len(query_rows))


if __name__ == "__main__":
    main()
