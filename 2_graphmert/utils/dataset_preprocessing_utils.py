#!/usr/bin/env python3
"""
dataset_preprocessing_utils.py — GraphMERT Pipeline Step 4

Performs co-occurrence grounding: for each triple in the seed KG, finds
text snippets where both the HEAD and TAIL entity are present, then builds
the GraphMERT-format training samples (input_nodes + leaf_relationships).

This step replaces the old graphmert_bridge.py — that script is no longer
needed. The bridge logic (tokenizer setup, relation map, grounding) is now
fully integrated here.

Inputs (all paths from YAML config or CLI overrides):
  - seed_kg_path:  CSV with columns [head, relation, tail]
  - train_src:     cleaned LLM relations dataset (from clean_llm_relations.py)
  - eval_src:      eval cleaned LLM relations dataset
  - tokenizer:     path to stable_tokenizer (from run_tokenization.py)
  - output_dir:    where to write preprocessed datasets and relation_map.json

Outputs (inside output_dir):
  - relation_map.json
  - ready_for_training_train/   <- HuggingFace Dataset
  - ready_for_training_eval/
"""

import logging
import os
import json
import sys
import shutil
from pathlib import Path
from collections import defaultdict

import yaml
import datasets
from datasets import load_from_disk, DatasetDict, Dataset, Features, Value, Sequence
import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(utils_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from graphmert_model import GraphMertDataCollatorForLanguageModeling, GraphMertConfig
except ImportError:
    from models import GraphMertDataCollatorForLanguageModeling, GraphMertConfig

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from .training_arguments import ModelArguments, DataTrainingArguments, PreprocessingArguments

try:
    AutoConfig.register("graphmert", GraphMertConfig)
except ValueError:
    pass

os.environ.setdefault("DATASETS_DISABLE_CACHING", "1")
logger = logging.getLogger(__name__)

# =========================================================
# ARCHITECTURE INVARIANTS
# =========================================================
ROOT_NODES = 512
NUM_LEAVES = 3
MAX_NODES = 2048


# =========================================================
# CO-OCCURRENCE GROUNDING (was graphmert_bridge.py)
# =========================================================

def _safe_json_loads(x):
    if x is None: return {}
    if isinstance(x, (dict, list)): return x
    try: return json.loads(str(x).strip())
    except: return {}


def build_relation_map(allowed_relations):
    return {r: i + 1 for i, r in enumerate(allowed_relations)}


def ground_triples_to_snippets(tok, rel_map, src, kg_df) -> Dataset:
    """
    For each (head, relation, tail) triple in kg_df, find snippets in src
    where BOTH head AND tail appear (co-occurrence filter), and build a
    GraphMERT training sample.
    """
    pad_id = int(tok.pad_token_id)
    rows = []

    logger.info("Building content index...")
    content_index = defaultdict(list)
    for i, ex in enumerate(src):
        hp = _safe_json_loads(ex.get("head_positions", {}))
        for head_str in hp.keys():
            content_index[head_str.lower()].append(i)

    stats = {"total_triples": len(kg_df), "success": 0, "no_cooccurrence": 0, "no_head_match": 0}

    for _, row in kg_df.iterrows():
        head = str(row["head"]).strip()
        rel  = str(row["relation"]).strip()
        tail = str(row["tail"]).strip()

        matched_indices = content_index.get(head.lower(), [])
        if not matched_indices:
            stats["no_head_match"] += 1
            continue

        rel_num = rel_map.get(rel)
        if not rel_num:
            continue

        tail_ids = tok.encode(tail, add_special_tokens=False)
        head_ids = tok.encode(head, add_special_tokens=False)

        for idx in matched_indices:
            ex = src[idx]
            snippet_entities = _safe_json_loads(ex.get("head_positions", {}))

            tail_present = any(k.lower() == tail.lower() for k in snippet_entities.keys())
            if not tail_present:
                stats["no_cooccurrence"] += 1
                continue

            roots = list(ex["input_ids"][:ROOT_NODES])
            if len(roots) < ROOT_NODES:
                roots += [pad_id] * (ROOT_NODES - len(roots))

            pos = None
            for k, v in snippet_entities.items():
                if k.lower() == head.lower():
                    pos = int(v)
                    break

            if pos is None or pos >= ROOT_NODES:
                continue

            leaves = [pad_id] * (MAX_NODES - ROOT_NODES)
            leaf_start = pos * NUM_LEAVES
            for j, tid in enumerate(tail_ids[:NUM_LEAVES]):
                leaves[leaf_start + j] = int(tid)

            leaf_special_mask = [1 if t == pad_id else 0 for t in leaves]

            rows.append({
                "id": str(ex["id"]),
                "input_nodes": roots + leaves,
                "attention_mask": [0 if t == pad_id else 1 for t in (roots + leaves)],
                "leaf_relationships": [rel_num if i == pos else 0 for i in range(ROOT_NODES)],
                "head_lengths": [min(len(head_ids), 32) if i == pos else 0 for i in range(ROOT_NODES)],
                "start_indices": [0] * ROOT_NODES,
                "special_tokens_mask": [
                    1 if t in {tok.pad_token_id, tok.cls_token_id} else 0 for t in roots
                ] + leaf_special_mask,
            })
            stats["success"] += 1

    logger.info("Grounding results: %s", json.dumps(stats))

    features = Features({
        "id": Value("string"),
        "input_nodes": Sequence(Value("int64"), length=MAX_NODES),
        "attention_mask": Sequence(Value("uint8"), length=MAX_NODES),
        "leaf_relationships": Sequence(Value("uint16"), length=ROOT_NODES),
        "head_lengths": Sequence(Value("uint16"), length=ROOT_NODES),
        "start_indices": Sequence(Value("uint8"), length=ROOT_NODES),
        "special_tokens_mask": Sequence(Value("uint8"), length=MAX_NODES),
    })
    return Dataset.from_list(rows, features=features)


# =========================================================
# MAIN
# =========================================================

def main(yaml_file: str, seed_kg_path: str = None, train_src: str = None,
         eval_src: str = None, tokenizer_path: str = None, output_dir: str = None):

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PreprocessingArguments))
    yaml_config = yaml.safe_load(Path(yaml_file).read_text())
    yaml_config["bf16"] = False
    yaml_config["fp16"] = False
    model_args, data_args, training_args, preprocessing_args = parser.parse_dict(yaml_config, allow_extra_keys=True)

    # CLI overrides take precedence over YAML
    seed_kg_path  = seed_kg_path  or getattr(data_args, "injections_train_path", None)
    train_src     = train_src     or getattr(preprocessing_args, "train_src", None)
    eval_src      = eval_src      or getattr(preprocessing_args, "eval_src", None)
    tokenizer_path = tokenizer_path or getattr(model_args, "tokenizer_name", None)
    output_dir    = output_dir    or getattr(preprocessing_args, "preprocessing_output_root", None)

    for name, val in [("seed_kg_path", seed_kg_path), ("train_src", train_src),
                      ("eval_src", eval_src), ("tokenizer_path", tokenizer_path),
                      ("output_dir", output_dir)]:
        if not val:
            raise ValueError(f"Required path '{name}' is not set. Pass via CLI or YAML.")

    os.makedirs(output_dir, exist_ok=True)

    # Load stable tokenizer (created by run_tokenization.py)
    logger.info("Loading stable tokenizer from: %s", tokenizer_path)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    # Build and save relation map. Source: the active domain's relations list
    # (e.g. domains/neuroscience.yaml::relations). This MUST match the relation
    # set graphrag used during extraction — otherwise any triple whose relation
    # isn't in this list is silently dropped during grounding, which (combined
    # with smoke-scale data sparsity) can produce 0 grounded samples and crash
    # Dataset.from_list with a "Keys mismatch" error.
    try:
        # pipeline_config sits at the repo root. Make sure it's importable
        # whether this is invoked from scripts/phases/graphmert.sh (which
        # cd's into 2_graphmert/) or from a different cwd.
        import sys as _sys
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from pipeline_config import get_relations  # noqa: WPS433
        allowed_relations = list(get_relations())
        if not allowed_relations:
            raise RuntimeError("domain config has no 'relations' list")
        logger.info("Loaded %d relations from active domain config", len(allowed_relations))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load relations from domain config (%s); "
                       "falling back to hardcoded biomed set.", exc)
        allowed_relations = [
            "part_of", "participates_in", "modulates", "contains", "located_in", "projects_to",
            "mediates_signal_for", "causes", "required_for", "controls", "impaired_in", "binds_to",
            "symptom_of", "associated_with", "connected_to", "receives_input_from", "results_in",
            "activates", "inhibits", "innervates", "encodes_representation_of", "responds_to",
            "expressed_in", "represents", "regulates", "transports", "originates_from",
            "forms_complex_with", "releases",
        ]
    rel_map = build_relation_map(allowed_relations)
    rel_map_path = os.path.join(output_dir, "relation_map.json")
    with open(rel_map_path, "w") as f:
        json.dump(rel_map, f, indent=2)
    logger.info("Relation map saved to: %s", rel_map_path)

    # Load seed KG
    logger.info("Loading seed KG from: %s", seed_kg_path)
    kg_df = pd.read_csv(seed_kg_path)
    logger.info("Seed KG: %d triples", len(kg_df))

    # Ground triples to snippets
    logger.info("Grounding train split...")
    train_ready = ground_triples_to_snippets(tok, rel_map, load_from_disk(train_src), kg_df)
    logger.info("Grounding eval split...")
    eval_ready  = ground_triples_to_snippets(tok, rel_map, load_from_disk(eval_src),  kg_df)

    train_out = os.path.join(output_dir, "ready_for_training_train")
    eval_out  = os.path.join(output_dir, "ready_for_training_eval")
    train_ready.save_to_disk(train_out)
    eval_ready.save_to_disk(eval_out)
    logger.info("Grounded train dataset: %s (%d rows)", train_out, len(train_ready))
    logger.info("Grounded eval  dataset: %s (%d rows)", eval_out,  len(eval_ready))

    # Initialize model config
    config = GraphMertConfig()
    if model_args.config_overrides:
        config.update_from_string(model_args.config_overrides)
    config.root_nodes = ROOT_NODES
    config.max_nodes  = MAX_NODES
    set_seed(training_args.seed)

    # Clear and populate model cache
    cache_dir = model_args.cache_dir
    if os.path.exists(cache_dir):
        logger.info("Purging training cache: %s", cache_dir)
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    for key, ds in [("train", train_ready), ("validation", eval_ready)]:
        out_path = os.path.join(cache_dir, key, "ready_for_training")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ds.save_to_disk(out_path)
        logger.info("Cached %s dataset at: %s", key, out_path)

    logger.info("Preprocessing complete. Run 'python run_mlm.py' next.")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
