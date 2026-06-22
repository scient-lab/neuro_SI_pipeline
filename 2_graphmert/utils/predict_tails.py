#!/usr/bin/env python3
# coding=utf-8
"""
predict_tails.py — GraphMERT Pipeline Step 2.6b

Generates top-k tail predictions for each (head, relation) pair using the
trained GraphMERT MLM checkpoint (masked leaf-slot prediction).

Usage:
  python 2_graphmert/utils/predict_tails.py \\
    --model_dir   $OUTPUT_BASE/graphmert/checkpoints/best \\
    --tokenizer   $OUTPUT_BASE/graphmert/stable_tokenizer \\
    --relation_map $OUTPUT_BASE/graphmert/relation_map.json \\
    --dataset     $OUTPUT_BASE/graphmert/llm_relations/relations_clean_eval \\
    --output_dir  $OUTPUT_BASE/graphmert/predictions_graphmert \\
    --topk        20 \\
    --batch_size  8
"""

import argparse
import os
import shutil
import json
import logging
import random
import glob
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

os.environ["DATASETS_DISABLE_CACHING"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn import functional as F
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphmert_model import GraphMertConfig, GraphMertForMaskedLM
try:
    AutoConfig.register("graphmert", GraphMertConfig)
except ValueError:
    pass  # already registered

ROOT_NODES = 512
NUM_LEAVES = 3
MAX_NODES = ROOT_NODES * (1 + NUM_LEAVES)  # 2048

logger = logging.getLogger("predict_tails")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",    required=True, help="Path to GraphMERT checkpoint dir")
    ap.add_argument("--tokenizer",    required=True, help="Path to stable tokenizer")
    ap.add_argument("--relation_map", required=True, help="Path to relation_map.json")
    ap.add_argument("--dataset",      required=True, help="Path to cleaned LLM relations dataset")
    ap.add_argument("--output_dir",   required=True, help="Output directory for predictions")
    ap.add_argument("--topk",         type=int, default=20)
    ap.add_argument("--batch_size",   type=int, default=8)
    return ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_best_checkpoint(base_dir):
    if os.path.exists(os.path.join(base_dir, "config.json")):
        return base_dir
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not checkpoints:
        raise ValueError(f"No config.json or checkpoints found in {base_dir}")
    # Sort by checkpoint step number to get the latest
    latest = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    logger.info(f"Using latest checkpoint: {latest}")
    return latest

def write_preview_txt(ds: Dataset, path: str, stats: Dict[str, Any]):
    logger.info(f"Writing inspection preview to: {path}")
    rel_to_indices = defaultdict(list)
    for i in range(len(ds)):
        rel_to_indices[str(ds[i]["relation"])].append(i)
    
    lines = ["GraphMERT Prediction Preview", "=" * 60, ""]
    for rel, idxs in rel_to_indices.items():
        # Sample up to 2 examples per relation for quick check
        chosen = random.sample(idxs, min(2, len(idxs)))
        for i in chosen:
            ex = ds[i]
            lines.append(f"ID: {ex['id']} | Head: {ex['head']} | Rel: {ex['relation']}")
            lines.append("  Top-5 Predictions:")
            for t, lp in zip(ex['topk_tokens'][:5], ex['topk_logprobs'][:5]):
                lines.append(f"    - {t:<15} (logp={lp:.4f})")
            lines.append("-" * 30)
    with open(path, "w") as f:
        f.write("\n".join(lines))

def predict_topk_first_leaf_slot(examples, model, tokenizer, topk):
    device = next(model.parameters()).device
    
    # Convert lists to tensors and move to device
    input_nodes_ids = torch.tensor(examples["input_nodes"], dtype=torch.long, device=device).unsqueeze(-1)
    attention_in = torch.tensor(examples["attention_mask"], dtype=torch.long, device=device)
    head_pos = torch.tensor(examples["position"], dtype=torch.long, device=device)
    rel_num = torch.tensor(examples["relation_num"], dtype=torch.long, device=device)
    head_len = torch.tensor(examples["head_len"], dtype=torch.long, device=device)

    bsz = input_nodes_ids.size(0)
    idx = torch.arange(bsz, device=device)
    
    # Construct Leaf Relationships (Batch, RootNodes)
    leaf_relationships = torch.zeros((bsz, ROOT_NODES), dtype=torch.long, device=device)
    leaf_relationships[idx, head_pos] = rel_num
    
    # Construct Head Lengths (Batch, RootNodes)
    head_lengths = torch.zeros((bsz, ROOT_NODES), dtype=torch.long, device=device)
    head_lengths[idx, head_pos] = head_len

    # Identify the target slot (first leaf node of the head)
    slot = ROOT_NODES + NUM_LEAVES * head_pos
    
    # Prepare Inputs: Mask the target slot
    work_nodes = input_nodes_ids.clone()
    work_nodes[idx, slot, 0] = int(tokenizer.mask_token_id)

    # Attention Mask: Only attend to Roots + Target Leaf
    work_attn = torch.zeros((bsz, MAX_NODES), dtype=torch.long, device=device)
    work_attn[:, :ROOT_NODES] = attention_in[:, :ROOT_NODES] # Attend to valid roots
    work_attn[idx, slot] = 1 # Attend to the target leaf slot

    with torch.no_grad():
        # Use autocast for speed/memory efficiency
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(input_nodes=work_nodes, attention_mask=work_attn,
                        leaf_relationships=leaf_relationships, head_lengths=head_lengths)
    
    # Handle output format (dict vs object)
    if isinstance(out, dict):
        logits = out['logits']
    else:
        logits = out.logits
        
    # Extract logits for the target slot
    slot_logits = logits[idx, slot]
    
    # Mask out special tokens from predictions
    ban = [tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    slot_logits[:, [b for b in ban if b is not None]] = -1e9
    
    probs = F.softmax(slot_logits, dim=-1)
    topk_probs, topi = torch.topk(probs, k=topk, dim=-1)

    return {
        "topk_tokens": [tokenizer.convert_ids_to_tokens(ids) for ids in topi.cpu().tolist()],
        "topk_logprobs": torch.log(topk_probs).cpu().tolist(),
        "best_token": [tokenizer.convert_ids_to_tokens(ids)[0] for ids in topi.cpu().tolist()],
        "best_logprob": torch.log(topk_probs[:, 0]).cpu().tolist(),
    }

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    with open(args.relation_map, "r") as f:
        relation_map = json.load(f)

    actual_model_path = get_best_checkpoint(args.model_dir)
    config = AutoConfig.from_pretrained(actual_model_path)
    config.root_nodes = ROOT_NODES
    config.max_nodes = MAX_NODES

    model = GraphMertForMaskedLM.from_pretrained(actual_model_path, config=config).cuda()
    model.eval()

    raw_ds = load_from_disk(args.dataset)
    pad_id = tokenizer.pad_token_id
    out_rows = []

    for ex in raw_ds:
        cid = ex["id"]
        heads_rel = json.loads(ex["cleaned_relations_json"])
        head_positions = json.loads(ex["head_positions"])

        root_tokens = list(ex["input_ids"])[:ROOT_NODES]
        if len(root_tokens) < ROOT_NODES:
            root_tokens += [pad_id] * (ROOT_NODES - len(root_tokens))

        input_nodes = root_tokens + [pad_id] * (MAX_NODES - ROOT_NODES)
        attn = [1 if t != pad_id else 0 for t in input_nodes]

        for h, rels in heads_rel.items():
            pos = next((v for k, v in head_positions.items() if k.lower() == h.lower()), None)
            rel_name = rels[0] if isinstance(rels, list) else rels
            rel_num = relation_map.get(rel_name)

            if pos is not None and rel_num and pos < ROOT_NODES:
                out_rows.append({
                    "id": cid,
                    "input_nodes": input_nodes,
                    "attention_mask": attn,
                    "head": h,
                    "position": pos,
                    "relation": rel_name,
                    "relation_num": rel_num,
                    "head_len": len(tokenizer.encode(h, add_special_tokens=False))
                })

    logger.info(f"Generated {len(out_rows)} probe samples.")
    dataset = Dataset.from_list(out_rows)

    out = dataset.map(
        predict_topk_first_leaf_slot,
        batched=True,
        batch_size=args.batch_size,
        fn_kwargs=dict(model=model, tokenizer=tokenizer, topk=args.topk),
        load_from_cache_file=False
    )

    out.to_pandas().to_parquet(os.path.join(args.output_dir, "predictions.parquet"), index=False)
    write_preview_txt(out, os.path.join(args.output_dir, "inspection_preview.txt"), {})
    logger.info("Predictions complete.")

if __name__ == "__main__":
    main()