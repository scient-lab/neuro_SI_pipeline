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

# =========================================================
# ARCHITECTURE INVARIANTS (must match preprocessing)
# =========================================================
ROOT_NODES = 512
NUM_LEAVES = 3
MAX_NODES = ROOT_NODES * (1 + 3)  # 2048


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

    config.num_relationships = model_args.num_relationships
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
