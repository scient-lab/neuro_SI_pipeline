from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Pipeline config loader at the repo root. Sources vocabulary from the
# domain YAML configured via SI_DOMAIN (default: neuroscience).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_config import (  # noqa: E402
    get_entity_categories,
    get_focus_instructions,
    get_relations,
    render_prompt,
)


# Legacy filename/log label used by 1_seed_kg/graphrag_index.py to namespace
# output dirs (e.g. extracted_graph_responses_set2_0-1000.json). The ACTIVE
# relation list is sourced from the merged pipeline config below, NOT from
# this label. Kept only for filename backwards-compat with existing runs.
RELATION_SET_NAME = os.environ.get("KG_RELATION_SET", "set2").strip().lower()


def get_relation_types() -> List[str]:
    """Return the active relation id list from the merged pipeline config.

    Source of truth: domains/<SI_DOMAIN>.yaml::relations.
    """
    return get_relations()


def get_entity_types() -> List[str]:
    """Return the entity category id list from the merged pipeline config.

    Source of truth: domains/<SI_DOMAIN>.yaml::entity_categories.
    """
    return get_entity_categories()


def get_focus_instructions_text() -> str:
    """Return the free-text extractor focus instructions, or empty string."""
    return get_focus_instructions()


# ---------------------------------------------
# PROMPTS
# ---------------------------------------------
# Sourced from prompts/extract.yaml -- see docs/PROMPT_MIGRATION.md item #1.
# The 4 constants below preserve the original module-level API so existing
# consumers (1_seed_kg/graphrag_index.py + diagnose_llm_extraction.py) can
# keep their `.format(input_text=..., think_directive=..., tuple_delimiter=...,
# record_delimiter=..., completion_delimiter=..., relation_list=...)` calls
# completely unchanged. Each value is byte-identical (SHA-256 verified) to
# the prior hardcoded string on the neuroscience pipeline; cross-domain
# pipelines rewrite the underlying content in domains/<name>.yaml::extract_*.
#
# {input_text}, {think_directive}, {tuple_delimiter}, {record_delimiter},
# {completion_delimiter}, {relation_list} are Python-format placeholders
# that stay literal here and are filled by the consumer at call time.
# Qwen3 honors a bare "/no_think" control token to skip its <think>...</think>
# block; graphrag_index.py reads extract.extract_triples_no_think from
# configs/default.yaml and decides whether to inject " /no_think" or "".
_extract_prompts = render_prompt("extract")
PROMPT_TEMPLATE   = _extract_prompts["system"]
USER_EXAMPLE      = _extract_prompts["user_example"]
ASSISTANT_EXAMPLE = _extract_prompts["assistant_example"]
USER_PROMPT       = _extract_prompts["user"]
