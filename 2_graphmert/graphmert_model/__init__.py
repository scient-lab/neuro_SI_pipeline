import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from transformers import DataCollatorForLanguageModeling

from .configuration_graphmert import GraphMertConfig
from .modeling_graphmert import (
    GraphMertPreTrainedModel,
    GraphMertModel,
    GraphMertForMaskedLM,
)
from .collating_graphmert import GraphMertDataCollator


@dataclass
class GraphMertDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Lightweight HF-style collator used during preprocessing (does not require Cython)."""
    config: GraphMertConfig = None
    graph_types: List[str] = None
    process_arch_tensors: bool = True
    on_the_fly_processing: bool = False
    mlm_sbo: bool = False
    mlm_on_leaves_probability: float = 0.15
    subword_token_start: str = "##"

    def preprocess_items(self, items):
        if 'input_nodes' in items and isinstance(items['input_nodes'][0], (int, np.integer)):
            return items

        input_ids = items['input_ids']
        leaf_node_ids = items.get('leaf_node_ids')
        bsz = len(input_ids)
        flat_inputs = []

        for i in range(bsz):
            roots = list(input_ids[i])
            if len(roots) < self.config.root_nodes:
                roots += [self.tokenizer.pad_token_id] * (self.config.root_nodes - len(roots))
            else:
                roots = roots[:self.config.root_nodes]

            if leaf_node_ids is not None and len(leaf_node_ids[i]) > 0:
                leaves = leaf_node_ids[i]
                if isinstance(leaves[0], (list, np.ndarray)):
                    flat_leaves = [item for sublist in leaves for item in sublist]
                else:
                    flat_leaves = list(leaves)
            else:
                flat_leaves = []

            total_leaf_slots = self.config.max_nodes - self.config.root_nodes
            if len(flat_leaves) < total_leaf_slots:
                flat_leaves += [self.tokenizer.pad_token_id] * (total_leaf_slots - len(flat_leaves))
            else:
                flat_leaves = flat_leaves[:total_leaf_slots]

            flat_inputs.append(roots + flat_leaves)

        items['input_nodes'] = flat_inputs
        return items

    def get_start_indices(self, items):
        bsz = len(items['input_nodes'])
        items['start_indices'] = [[0] * self.config.root_nodes for _ in range(bsz)]
        return items


__all__ = [
    "GraphMertConfig",
    "GraphMertPreTrainedModel",
    "GraphMertModel",
    "GraphMertForMaskedLM",
    "GraphMertDataCollator",
    "GraphMertDataCollatorForLanguageModeling",
]
