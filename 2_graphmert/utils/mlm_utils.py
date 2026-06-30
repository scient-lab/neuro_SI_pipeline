#!/usr/bin/env python3
# coding=utf-8
"""
mlm_utils.py  — GraphMERT MLM training utilities

Called by run_mlm.py. Reads training configuration from a YAML file.
The stable tokenizer path must be set in args_mlm.yaml under `tokenizer_name`.
"""

import os
import sys
import math
import json
import shutil
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import datasets
from datasets import load_from_disk, DatasetDict

os.environ["DATASETS_DISABLE_CACHING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .training_arguments import (
    ModelArguments,
    DataTrainingArguments,
    PreprocessingArguments,
    unique_cache_filename,
)

# Register GraphMertConfig + GraphMertForMaskedLM with transformers' Auto
# registry. Two registrations are needed:
#   - AutoConfig("graphmert", GraphMertConfig)             so CONFIG_MAPPING["graphmert"] resolves
#   - AutoModelForMaskedLM(GraphMertConfig, ForMaskedLM)   so from_config() finds the model class
# dataset_preprocessing_utils.py registers AutoConfig for the preprocess
# step, but mlm_utils is imported by run_mlm.py independently of preprocess
# — without re-registering here, CONFIG_MAPPING raises KeyError('graphmert')
# at training time and the AutoModelForMaskedLM path errors with
# "Unrecognized configuration class".
_utils_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_utils_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
try:
    from graphmert_model import GraphMertConfig, GraphMertForMaskedLM  # noqa: E402
except ImportError:
    from models import GraphMertConfig, GraphMertForMaskedLM  # noqa: E402
try:
    AutoConfig.register("graphmert", GraphMertConfig)
    AutoModelForMaskedLM.register(GraphMertConfig, GraphMertForMaskedLM)
except ValueError:
    # Already registered by another module in this process — fine.
    pass

logger = logging.getLogger("graphmert_mlm")

# Architecture invariants — single source shared with
# dataset_preprocessing_utils and predict_tails. See architecture.py.
from .architecture import ROOT_NODES, NUM_LEAVES, MAX_NODES  # noqa: F401


def ensure_stable_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer missing PAD token. Run run_tokenization.py first.")
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer missing MASK token. Run run_tokenization.py first.")
    return tokenizer


def set_config_token_ids(config: AutoConfig, tokenizer: AutoTokenizer) -> None:
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    if bos_id is not None: config.bos_token_id = bos_id
    if eos_id is not None: config.eos_token_id = eos_id
    config.pad_token_id = tokenizer.pad_token_id


def _batchify(features: List[Dict[str, Any]], key: str, dtype: torch.dtype) -> torch.Tensor:
    vals = [f[key] for f in features]
    if isinstance(vals[0], torch.Tensor):
        return torch.stack(vals).to(dtype=dtype)
    return torch.tensor(vals, dtype=dtype)


# =========================================================
# TAIL-SLOT COLLATOR (Training Objective)
# =========================================================
@dataclass
class GraphMertTailSlotDataCollator:
    tokenizer: AutoTokenizer
    seed: int = 0

    def __post_init__(self):
        self._pad = int(self.tokenizer.pad_token_id)
        self._mask = int(self.tokenizer.mask_token_id)
        self._rng = random.Random(self.seed)

    def _pick_head_pos(self, leaf_relationships_row: torch.Tensor, input_nodes_row: torch.Tensor) -> Optional[Tuple[int, List[int]]]:
        cand = torch.nonzero(leaf_relationships_row != 0, as_tuple=False).view(-1).tolist()
        if not cand: return None
        self._rng.shuffle(cand)

        for hp in cand:
            block_start = ROOT_NODES + NUM_LEAVES * int(hp)
            block_end = block_start + NUM_LEAVES
            leaf_block = input_nodes_row[block_start:block_end]
            valid_slots = torch.nonzero(leaf_block != self._pad, as_tuple=False).view(-1).tolist()
            if valid_slots:
                abs_slots = [block_start + s for s in valid_slots]
                return int(hp), abs_slots
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_nodes = _batchify(features, "input_nodes", torch.long)
        leaf_relationships = _batchify(features, "leaf_relationships", torch.long)
        head_lengths = _batchify(features, "head_lengths", torch.long)

        bsz = input_nodes.size(0)
        labels = torch.full((bsz, MAX_NODES), -100, dtype=torch.long)
        attention_mask = torch.zeros((bsz, MAX_NODES), dtype=torch.long)
        attention_mask[:, :ROOT_NODES] = (input_nodes[:, :ROOT_NODES] != self._pad).to(torch.long)

        work_nodes = input_nodes.clone()
        active_labels = 0
        for i in range(bsz):
            res = self._pick_head_pos(leaf_relationships[i], work_nodes[i])
            if res:
                hp, valid_slots = res
                slot = self._rng.choice(valid_slots)
                labels[i, slot] = int(work_nodes[i, slot].item())
                work_nodes[i, slot] = self._mask
                for s in valid_slots:
                    attention_mask[i, s] = 1
                active_labels += 1

        return {
            "input_nodes": work_nodes.unsqueeze(-1),
            "attention_mask": attention_mask,
            "leaf_relationships": leaf_relationships,
            "head_lengths": head_lengths,
            "labels": labels,
        }


def resolve_num_relationships(num_relationships_cfg, relation_map_path):
    """Size the relation embeddings to cover every relation id the dataset uses.

    The relation embeddings (relation_encoder / relation_matrix_encoder =
    nn.Embedding(num_relationships + 1, ...)) are indexed by relation ids that
    come from relation_map.json — build_relation_map maps the domain's relations
    to ids 1..N. The embedding MUST cover the largest id there, so we DERIVE the
    size from that map (mirroring vocab_size = len(tokenizer)) rather than trust a
    hardcoded config value: a too-small value silently overflows the embedding and
    triggers a CUDA device-side assert at train step 0 (e.g. the space domain has
    67 relations vs a stale default of 43).

    The configured value is only a FLOOR — an explicit larger value keeps its
    headroom; a too-small one is corrected (with a warning), never used as-is.
    """
    cfg = num_relationships_cfg or 0
    if relation_map_path and os.path.exists(relation_map_path):
        with open(relation_map_path) as f:
            rel_map = json.load(f)
        # relation_map is {relation: id}; ids are 1..N, so the largest id is the
        # required upper bound (use max(id), robust to any future numbering).
        data_n = max((int(v) for v in rel_map.values()), default=0)
        if cfg and cfg < data_n:
            logger.warning(
                "num_relationships=%d (config) < %d relations in %s; using %d to "
                "avoid an out-of-range relation embedding (CUDA device-side assert).",
                cfg, data_n, relation_map_path, data_n)
        resolved = max(cfg, data_n)
        logger.info("num_relationships=%d (derived from %s; config floor=%d)",
                    resolved, relation_map_path, cfg)
        return resolved
    if cfg:
        logger.warning(
            "relation_map not found at %r; falling back to configured "
            "num_relationships=%d (verify it covers the KG's relations).",
            relation_map_path, cfg)
        return cfg
    raise ValueError(
        f"num_relationships unset and relation_map not found at {relation_map_path!r}; "
        "cannot size the relation embeddings — run dataset preprocessing first.")


def main(yaml_file: str):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PreprocessingArguments))
    model_args, data_args, training_args, preprocessing_args = parser.parse_yaml_file(yaml_file, allow_extra_keys=True)

    if os.path.exists(training_args.output_dir) and training_args.overwrite_output_dir:
        logger.info("Removing old output at %s", training_args.output_dir)
        shutil.rmtree(training_args.output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(training_args.seed)

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()

    config.num_relationships = resolve_num_relationships(
        model_args.num_relationships, preprocessing_args.relation_map_path)
    config.graph_types = model_args.graph_types
    config.mlm_sbo = model_args.mlm_sbo
    config.relation_emb_dropout = model_args.relation_emb_dropout

    if model_args.config_overrides:
        config.update_from_string(model_args.config_overrides)

    config.root_nodes = ROOT_NODES
    config.max_nodes = MAX_NODES

    # Use stable tokenizer from tokenizer_name in YAML
    stable_tokenizer_dir = model_args.tokenizer_name
    if not stable_tokenizer_dir or not os.path.isdir(stable_tokenizer_dir):
        raise ValueError(
            f"stable tokenizer not found at '{stable_tokenizer_dir}'.\n"
            "Set tokenizer_name in args_mlm.yaml to the output of run_tokenization.py."
        )
    logger.info("Loading stable tokenizer from: %s", stable_tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(stable_tokenizer_dir, cache_dir=model_args.cache_dir)
    tokenizer = ensure_stable_tokenizer(tokenizer)
    set_config_token_ids(config, tokenizer)
    config.vocab_size = len(tokenizer)

    if model_args.model_name_or_path:
        logger.info("Initializing with pretrained: %s", model_args.model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path, config=config,
            cache_dir=model_args.cache_dir, ignore_mismatched_sizes=True
        )
    else:
        logger.info("Initializing model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    train_path = os.path.join(model_args.cache_dir, "train", "ready_for_training")
    eval_path  = os.path.join(model_args.cache_dir, "validation", "ready_for_training")

    logger.info("Loading train dataset from: %s", train_path)
    logger.info("Loading eval  dataset from: %s", eval_path)

    if not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"Training cache not found at {train_path}.\n"
            "Run run_dataset_preprocessing.py first."
        )

    ds = DatasetDict({
        "train": load_from_disk(train_path),
        "validation": load_from_disk(eval_path),
    })

    required = ["input_nodes", "attention_mask", "leaf_relationships", "head_lengths"]
    for split in ds.keys():
        ds[split].set_format(type="torch", columns=required)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=GraphMertTailSlotDataCollator(tokenizer=tokenizer, seed=int(training_args.seed)),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint or False)
        trainer.save_model()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m utils.mlm_utils path/to/args_mlm.yaml")
    main(sys.argv[1])
