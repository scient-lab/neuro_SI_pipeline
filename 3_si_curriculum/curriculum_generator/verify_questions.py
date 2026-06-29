#!/usr/bin/env python3
"""
verify_questions.py — SI Pipeline Step 3

Validates generated Q&A questions using two LLM families simultaneously.
A question is kept only when BOTH models agree it is valid.

Usage:
  python curriculum_generator/verify_questions.py \\
    --input_json  ${OUTPUT_BASE}/SI/QA_items/curriculum_dataset.json \\
    --output_json ${OUTPUT_BASE}/SI/QA_items/verified/validated.json \\
    --model_ids   /path/to/mistral-nemo-12b /path/to/qwen3-14b

Run via SLURM:
  sbatch curriculum_generator/verify_questions.slurm
"""

import os
import sys
import gc
import torch
import json
import re
import math
import logging
import argparse
from pathlib import Path
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Pipeline config loader + Qwen3 tokenizer compat shim (repo root, 2 levels up).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _tokenizer_compat  # noqa: F401, E402  # side effect: vLLM 0.7.3 + Qwen3 fix
from pipeline_config import render_prompt, get_phase_param  # noqa: E402
# 3_si_curriculum on path so the shared curriculum_io package import resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curriculum_generator import curriculum_io as cio  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Sourced from prompts/curriculum_verify.yaml — byte-identical (post-.strip())
# to the prior in-file constant. {{domain}} is substituted from SI_DOMAIN.
# See docs/PROMPT_MIGRATION.md item #11.
SYSTEM_PROMPT_QA_VALIDATION = render_prompt("curriculum_verify")["system"].strip()


def parse_args():
    ap = argparse.ArgumentParser(description="Two-LLM validation of generated Q&A items")
    ap.add_argument("--input_json", default="",
                    help="[legacy mode] Input JSON array of generated Q&A items")
    ap.add_argument("--output_json", default="",
                    help="[legacy mode] Output JSON array of validated Q&A items")
    ap.add_argument("--curriculum_jsonl", default="",
                    help="[stage mode] curriculum.jsonl; grade stage:item records in place")
    ap.add_argument("--stats_path", default="",
                    help="[stage mode] curriculum_stats.json (default: alongside the jsonl)")
    ap.add_argument("--model_ids", nargs="+", required=True,
                    help="Exactly 2 model paths for two-LLM validation")
    # vLLM init defaults read from configs/default.yaml::curriculum.validate_qa_*.
    # In-code fallbacks (3rd arg of get_phase_param) preserve historical
    # values per [Preserve upstream defaults]. CLI flags still override YAML.
    ap.add_argument("--tensor_parallel_size", type=int,
                    default=get_phase_param('curriculum', 'validate_qa_tensor_parallel_size', 1))
    ap.add_argument("--gpu_memory_utilization", type=float,
                    default=get_phase_param('curriculum', 'validate_qa_gpu_memory_utilization', 0.70))
    # max_model_len: if 0/None, do NOT pass to vLLM — let it use the
    # model's nominal context. default.yaml ships `null` (upstream
    # behaviour); pilot/smoke profiles override with a concrete cap to
    # fit on 48 GB GPUs (~4096 is comfortable for tiny validation prompts).
    ap.add_argument("--max_model_len", type=int,
                    default=get_phase_param('curriculum', 'validate_qa_max_model_len', None),
                    help="vLLM KV cache cap; omit/0 to use model's nominal context")
    ap.add_argument("--batch_size", type=int,
                    default=get_phase_param('curriculum', 'validate_qa_batch_size', 64))
    # Generation cap. MUST fit Qwen3's <think> block + the final [yes]/[no];
    # the upstream hardcoded 256 truncated thinking mid-reason -> no verdict ->
    # 97% false-reject (smoke 2026-06-25). Bounded above by max_model_len.
    ap.add_argument("--max_tokens", type=int,
                    default=get_phase_param('curriculum', 'validate_qa_max_tokens', 1024),
                    help="generation cap; must fit Qwen3 <think> + the [yes]/[no] verdict")
    ap.add_argument("--subset", type=int, default=0,
                    help="If > 0, only validate this many items (for debugging)")
    return ap.parse_args()


def build_validation_prompt(item: Dict) -> str:
    context = item.get("path_string") or item.get("context_path") or cio.format_path(item.get("paths", []))
    question = item.get("question", "")
    answer = item.get("answer", "")
    explanation = item.get("explanation", item.get("thinking_trace", ""))
    return (
        f"Context Path: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Explanation: {explanation}"
    )


def _parse_verdict(text: str) -> bool:
    # Thinking models (Qwen3) emit "<think>...</think>" then the [yes]/[no]
    # verdict, and the reasoning itself may mention "[yes]"/"[no]" hypothetically.
    #   1. If <think> opened but never closed, the budget ran out before the
    #      verdict — no real answer was emitted, so reject (any [yes]/[no] in the
    #      raw reasoning is hypothetical). High drop here = raise
    #      curriculum.validate_qa_max_tokens.
    if "<think>" in text and "</think>" not in text:
        return False
    #   2. Read only the post-think answer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    #   3. Take the LAST [yes]/[no] — not re.search's FIRST, which could grab a
    #      reasoning mention before the final verdict.
    matches = re.findall(r"\[(yes|no)\]", text.lower())
    return matches[-1] == "yes" if matches else False


def validate_with_model(items: List[Dict], model_id: str,
                        tensor_parallel_size: int, gpu_memory_utilization: float,
                        batch_size: int, max_model_len = None,
                        max_tokens: int = 1024) -> List[bool]:
    # Build LLM kwargs conditionally — when max_model_len is None or 0, omit
    # it so vLLM uses the model's nominal config.max_position_embeddings.
    # This mirrors the original upstream behaviour (script never passed it).
    llm_kwargs = dict(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        # Skip redundant single-file weights (Mistral-Nemo ships both the HF
        # sharded format vLLM loads AND consolidated.safetensors + original/),
        # which otherwise ~doubles the cache to 46 GB (smoke 2026-06-25). No-op
        # for repos without them (Qwen3). TWIN: same list in
        # 2_graphmert/utils/llm_scores/fact_score.py — keep in sync.
        ignore_patterns=["original/**/*", "consolidated.safetensors"],
    )
    if max_model_len:  # treats None and 0 as "don't cap"
        llm_kwargs['max_model_len'] = int(max_model_len)
        logger.info("Loading model: %s (max_model_len=%d)", model_id, max_model_len)
    else:
        logger.info("Loading model: %s (max_model_len=model nominal)", model_id)
    llm = LLM(**llm_kwargs)
    # vLLM 0.7.3's `llm.chat()` triggers
    #   AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
    # on Qwen3 because the tokenizer wrapper doesn't proxy that property to
    # the underlying slow tokenizer. Workaround: load the tokenizer ourselves,
    # apply the chat template manually, and call `llm.generate()` (which
    # doesn't touch that codepath). When vLLM upgrades past 0.10.x revert
    # to llm.chat().
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    results = []

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        formatted_prompts = []
        for item in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_QA_VALIDATION},
                {"role": "user", "content": build_validation_prompt(item)},
            ]
            formatted_prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        outputs = llm.generate(formatted_prompts, sampling_params=sampling)
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            results.append(_parse_verdict(text))

        done = start + len(batch)
        if done % (batch_size * 10) == 0 or done == len(items):
            logger.info("Validated %d/%d [%s]", done, len(items), os.path.basename(model_id))

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_legacy(args) -> None:
    """Legacy array-in / array-out mode (curriculum.json -> curriculum_verified.json)."""
    logger.info("Loading input: %s", args.input_json)
    with open(args.input_json, "r") as f:
        items = json.load(f)
    logger.info("Total items: %d", len(items))

    if args.subset > 0:
        items = items[:args.subset]
        logger.info("Using subset of %d items", len(items))

    scores_1 = validate_with_model(
        items, args.model_ids[0],
        args.tensor_parallel_size, args.gpu_memory_utilization, args.batch_size,
        max_model_len=args.max_model_len, max_tokens=args.max_tokens
    )
    scores_2 = validate_with_model(
        items, args.model_ids[1],
        args.tensor_parallel_size, args.gpu_memory_utilization, args.batch_size,
        max_model_len=args.max_model_len, max_tokens=args.max_tokens
    )

    validated = [
        item for item, v1, v2 in zip(items, scores_1, scores_2) if v1 and v2
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(validated, f, indent=2)

    logger.info("Input:  %d items", len(items))
    logger.info("Valid (both agree): %d (%.1f%%)", len(validated),
                100 * len(validated) / max(len(items), 1))
    logger.info("Saved to: %s", args.output_json)


def run_stage_validate_item(args) -> None:
    """Stage mode (step validate_qa_item): grade stage:item records in curriculum.jsonl with
    BOTH models, stamp check_a_verdict / check_b_verdict + stage (verified|drop), and write
    per-grader yields (check_a_yield, check_b_yield, agreement_rate) to curriculum_stats.json.
    """
    stats_path = args.stats_path or os.path.join(
        os.path.dirname(os.path.abspath(args.curriculum_jsonl)), "curriculum_stats.json")
    records = list(cio.stream_records(args.curriculum_jsonl))
    items = [r for r in records if r.get("stage") == cio.STAGE_ITEM]
    logger.info("validate_qa_item: %d stage:item records (of %d total)", len(items), len(records))

    if not items:
        logger.warning("No stage:item records to validate in %s", args.curriculum_jsonl)
        cio.write_stat(stats_path, "validate_qa_item", cio.yield_counts(0, 0))
        return

    if args.subset > 0:
        items = items[:args.subset]
        logger.info("Using subset of %d items", len(items))

    scores_a = validate_with_model(
        items, args.model_ids[0],
        args.tensor_parallel_size, args.gpu_memory_utilization, args.batch_size,
        max_model_len=args.max_model_len, max_tokens=args.max_tokens
    )
    scores_b = validate_with_model(
        items, args.model_ids[1],
        args.tensor_parallel_size, args.gpu_memory_utilization, args.batch_size,
        max_model_len=args.max_model_len, max_tokens=args.max_tokens
    )

    n = len(items)
    kept = a_yes = b_yes = agree = 0
    for it, va, vb in zip(items, scores_a, scores_b):
        va, vb = bool(va), bool(vb)
        it["check_a_verdict"] = va
        it["check_b_verdict"] = vb
        a_yes += int(va)
        b_yes += int(vb)
        agree += int(va == vb)
        if va and vb:
            it["stage"] = cio.STAGE_VERIFIED
            kept += 1
        else:
            it["stage"] = cio.STAGE_DROP
            it["drop_reason"] = "consensus"

    cio.write_all_jsonl(args.curriculum_jsonl, records)
    counts = cio.yield_counts(
        n, kept,
        check_a_yield=round(a_yes / n, 4),
        check_b_yield=round(b_yes / n, 4),
        agreement_rate=round(agree / n, 4),
        drop_reasons={"consensus": n - kept},
    )
    cio.write_stat(stats_path, "validate_qa_item", counts)
    logger.info("validate_qa_item: kept %d/%d (A=%.1f%% B=%.1f%% agree=%.1f%%) -> %s",
                kept, n, 100 * a_yes / n, 100 * b_yes / n, 100 * agree / n, args.curriculum_jsonl)


def main():
    args = parse_args()

    if len(args.model_ids) != 2:
        raise ValueError("Exactly 2 model_ids required for two-LLM validation")

    if args.curriculum_jsonl:
        run_stage_validate_item(args)
    elif args.input_json:
        run_legacy(args)
    else:
        raise ValueError(
            "provide --curriculum_jsonl (stage mode) or --input_json + --output_json (legacy mode)")


if __name__ == "__main__":
    main()
