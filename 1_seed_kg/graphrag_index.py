#!/usr/bin/env python3
"""
graphrag_index.py

Run the GraphRAG knowledge-graph extraction pipeline step-by-step.

Usage:
  python graphrag_index.py --root_dir /path/to/graphrag_dir --model_id /path/to/model --step 3

Steps:
  1  Create base text units (chunking, no GPU)
  2  Create final documents (no GPU)
  3  Run LLM entity/relation extraction (GPU required — submit via SLURM)
  4  Parse LLM responses into entity/relationship tables (no GPU)
  5  Clean, finalize, and write the seed KG parquet files

Set --root_dir to the directory containing settings.yaml and the input/ folder.
"""

import os
import json
import re
import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from collections.abc import Mapping
from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx

from graphrag.callbacks.workflow_callbacks_manager import WorkflowCallbacksManager
from graphrag.index.run.utils import create_run_context
from graphrag.config.load_config import load_config
from graphrag.storage.factory import StorageFactory
from graphrag.cache.factory import CacheFactory
from graphrag.index.context import PipelineRunStats
from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
from graphrag.index.utils.string import clean_str
from graphrag.index.operations.extract_graph.extract_graph import (
    _merge_entities,
    _merge_relationships,
)
from graphrag.index.operations.compute_edge_combined_degree import (
    compute_edge_combined_degree,
)

from vllm import LLM, SamplingParams

from prompts_kg import (
    PROMPT_TEMPLATE,
    USER_EXAMPLE,
    ASSISTANT_EXAMPLE,
    USER_PROMPT,
    get_relation_types,
    RELATION_SET_NAME,
)

# pipeline_config is on sys.path because prompts_kg (imported above) inserts it.
from pipeline_config import get_phase_param  # noqa: E402

# ------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="GraphRAG KG extraction pipeline")
    ap.add_argument("--root_dir", required=True,
                    help="Root directory containing settings.yaml and input/output folders")
    ap.add_argument("--model_id", default=None,
                    help="Path to local vLLM model (required for step 3)")
    ap.add_argument("--step", type=int, default=None,
                    help="Pipeline step to run: 1, 2, 3, 4, or 5. Omit to run all sequentially.")
    # Defaults come from extract.batch_size / extract.rows_per_job in the
    # merged config (profile > domain > default). CLI still wins when set.
    ap.add_argument("--batch_size", type=int,
                    default=get_phase_param('extract', 'batch_size', 100),
                    help="LLM batch size for step 3 (config: extract.batch_size)")
    ap.add_argument("--rows_per_job", type=int,
                    default=get_phase_param('extract', 'rows_per_job', 8000),
                    help="Rows per SLURM array job in step 3 (config: extract.rows_per_job)")
    return ap.parse_args()


ARGS = parse_args()
ROOT_DIR = Path(ARGS.root_dir)

# ------------------------------------------------
# GLOBAL CONFIG & CONTEXT
# ------------------------------------------------

callback_chain = WorkflowCallbacksManager()
cli_overrides: dict = {}

config = load_config(ROOT_DIR, None, cli_overrides)

storage_config = config.output.model_dump()
storage = StorageFactory().create_storage(
    storage_type=storage_config["type"],
    kwargs=storage_config,
)
cache_config = config.cache.model_dump()
cache = CacheFactory().create_cache(
    cache_type=cache_config["type"],
    root_dir=config.root_dir,
    kwargs=cache_config,
)
context = create_run_context(storage=storage, cache=cache, stats=None)


# ------------------------------------------------
# STATS DUMPING
# ------------------------------------------------

async def _dump_stats(stats: PipelineRunStats, storage: PipelineStorage) -> None:
    await storage.set(
        "stats.json", json.dumps(asdict(stats), indent=4, ensure_ascii=False)
    )


# ------------------------------------------------
# LLM EXTRACTION
# ------------------------------------------------

def extract_graph(text_units: pd.DataFrame, batch_size: int = 100):
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    model_id = ARGS.model_id
    if not model_id:
        raise ValueError("--model_id is required for step 3")

    # Sampling sourced from configs/default.yaml::extract.* (fallbacks preserved).
    temperature = get_phase_param('extract', 'temperature', 0.6)
    top_p       = get_phase_param('extract', 'top_p', 0.95)
    max_tokens  = get_phase_param('extract', 'max_tokens', 8192)
    # vLLM init knobs — every one configurable per profile so smoke / pilot /
    # paper can match their hardware. Pattern mirrors 2_graphmert/predict_tails_llm.py.
    max_model_len          = get_phase_param('extract', 'max_model_len', 4096)
    tensor_parallel_size   = get_phase_param('extract', 'tensor_parallel_size', 1)
    gpu_memory_utilization = get_phase_param('extract', 'gpu_memory_utilization', 0.90)
    top_k                  = get_phase_param('extract', 'top_k', 20)
    min_p                  = get_phase_param('extract', 'min_p', 0)

    relation_types = get_relation_types()
    relation_list_str = json.dumps(relation_types)

    logger.info(f"Using relation set: {RELATION_SET_NAME} (n={len(relation_types)})")
    logger.info(f"Loading LLM: {model_id}")
    logger.info(f"vLLM init: max_model_len={max_model_len}  "
                f"tp_size={tensor_parallel_size}  gpu_mem_util={gpu_memory_utilization}")

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        min_p=min_p,
    )
    print("LLM loaded")

    prompts_batches = []
    current_batch = []

    for _, row in text_units.iterrows():
        current_batch.append(
            [
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATE.format(
                        completion_delimiter="<|COMPLETE|>",
                        tuple_delimiter="<|>",
                        record_delimiter="##",
                        relation_list=relation_list_str,
                    ),
                },
                {"role": "user", "content": USER_EXAMPLE},
                {
                    "role": "assistant",
                    "content": ASSISTANT_EXAMPLE.format(
                        completion_delimiter="<|COMPLETE|>",
                        tuple_delimiter="<|>",
                        record_delimiter="##",
                    ),
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(input_text=row["text"]),
                },
            ]
        )
        if len(current_batch) == batch_size:
            prompts_batches.append(current_batch)
            current_batch = []

    if current_batch:
        prompts_batches.append(current_batch)

    all_responses: list[str] = []
    for pb in prompts_batches:
        outputs = llm.chat(pb, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        all_responses.extend(responses)

    return all_responses


# ------------------------------------------------
# PIPELINE 1: CREATE BASE TEXT UNITS
# ------------------------------------------------

async def pipeline_1():
    from graphrag.index.input.factory import create_input
    from graphrag.index.workflows.create_base_text_units import (
        run_workflow as run_create_base_text_units,
    )

    dataset = await create_input(config.input, None, config.root_dir)
    print("Final # of rows loaded: ", len(dataset))

    context.stats.num_documents = len(dataset)
    await _dump_stats(context.stats, context.storage)
    await write_table_to_storage(dataset, "documents", context.storage)
    result = await run_create_base_text_units(config, context, callback_chain)
    print(result)


# ------------------------------------------------
# PIPELINE 2: CREATE FINAL DOCUMENTS
# ------------------------------------------------

async def pipeline_2():
    from graphrag.index.workflows.create_final_documents import (
        run_workflow as run_create_final_documents,
    )
    result = await run_create_final_documents(config, context, callback_chain)
    print(result)


# ------------------------------------------------
# PIPELINE 3: RUN LLM AND SAVE RAW RESPONSES
# ------------------------------------------------

async def pipeline_3():
    text_units = await load_table_from_storage("text_units", context.storage)

    task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_id = 0 if task_id_env is None else int(task_id_env)

    rows_per_job = ARGS.rows_per_job
    start = task_id * rows_per_job
    end = min(start + rows_per_job, len(text_units))

    text_slice = text_units.iloc[start:end]
    all_responses = extract_graph(text_units=text_slice, batch_size=ARGS.batch_size)

    out_dir = ROOT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"extracted_graph_responses_{RELATION_SET_NAME}_{start}-{end}.json"

    with open(out_path, "w") as f:
        json.dump(all_responses, f, indent=4)
    print(f"Extracted responses saved to {out_path}.")


# ------------------------------------------------
# RESULT PARSING HELPERS
# ------------------------------------------------

def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")

def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")

async def _process_results_directed(
    results: dict[int, str],
    tuple_delimiter: str,
    record_delimiter: str,
    _join_descriptions: bool = True,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    for source_doc_id, extracted_data in results.items():
        records = [r.strip() for r in extracted_data.split(record_delimiter)]

        for record in records:
            if not record:
                continue
            record = re.sub(r"^\(|\)$", "", record.strip())
            record_attributes = record.split(tuple_delimiter)

            if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                entity_name = clean_str(record_attributes[1]).lower()
                entity_type = clean_str(record_attributes[2]).upper()
                entity_description = clean_str(record_attributes[3])

                if graph.has_node(entity_name):
                    node = graph.nodes[entity_name]
                    if _join_descriptions:
                        node["description"] = "\n".join(
                            {*_unpack_descriptions(node), entity_description}
                        )
                    elif len(entity_description) > len(node.get("description", "")):
                        node["description"] = entity_description
                    node["source_id"] = ", ".join(
                        {*_unpack_source_ids(node), str(source_doc_id)}
                    )
                    node["type"] = entity_type if entity_type != "" else node.get("type", "")
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_doc_id),
                    )

            if record_attributes[0] == '"relationship"' and len(record_attributes) >= 5:
                source = clean_str(record_attributes[1]).lower()
                target = clean_str(record_attributes[2]).lower()
                relation_str = clean_str(record_attributes[3]).lower()
                edge_source_id = clean_str(str(source_doc_id))
                try:
                    weight = float(record_attributes[-1])
                except ValueError:
                    weight = 1.0

                if not graph.has_node(source):
                    graph.add_node(source, type="", description="", source_id=edge_source_id)
                if not graph.has_node(target):
                    graph.add_node(target, type="", description="", source_id=edge_source_id)

                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target) or {}
                    prev = edge_data.get("relation_raw", "")
                    joined = "\n".join([x for x in [prev, relation_str] if x])
                    edge_data["relation_raw"] = joined
                    edge_data["weight"] = float(edge_data.get("weight", 0.0)) + weight
                    prev_src = edge_data.get("source_id", "")
                    edge_data["source_id"] = ", ".join([x for x in [prev_src, edge_source_id] if x])
                    graph.add_edge(source, target, **edge_data)
                else:
                    graph.add_edge(
                        source, target,
                        relation_raw=relation_str,
                        weight=weight,
                        source_id=edge_source_id,
                    )

    return graph


# ------------------------------------------------
# PIPELINE 4: PARSE RESPONSES
# ------------------------------------------------

async def pipeline_4():
    text_units = await load_table_from_storage("text_units", context.storage)
    think_pattern = re.compile(r"</think>(.*)", re.DOTALL)

    entity_dfs: list[pd.DataFrame] = []
    relationship_dfs: list[pd.DataFrame] = []

    out_dir = ROOT_DIR / "output"

    response_files = sorted(out_dir.glob(f"extracted_graph_responses_{RELATION_SET_NAME}_*.json"))
    if not response_files:
        raise FileNotFoundError(
            f"No response files found in {out_dir} for RELATION_SET_NAME='{RELATION_SET_NAME}'.\n"
            "Run step 3 first."
        )

    for json_path in response_files:
        print(f"Loading responses from {json_path}")
        with open(json_path, "r") as f:
            all_responses = json.load(f)

        # Recover start index from filename
        stem = json_path.stem  # e.g. extracted_graph_responses_neuro_0-8000
        parts = stem.rsplit("_", 1)
        try:
            start = int(parts[-1].split("-")[0])
        except (IndexError, ValueError):
            start = 0

        for i, result in enumerate(all_responses):
            think_match = think_pattern.search(result)
            result_clean = think_match.group(1).strip() if think_match else result.strip()
            graph = await _process_results_directed({i + start: result_clean}, "<|>", "##")

            for _, node in graph.nodes(data=True):
                if node is not None and "source_id" in node:
                    node["source_id"] = ",".join(
                        text_units["id"][int(idx)]
                        for idx in str(node["source_id"]).split(",")
                        if idx.strip() != ""
                    )
            for _, _, edge in graph.edges(data=True):
                if edge is not None and "source_id" in edge:
                    edge["source_id"] = ",".join(
                        text_units["id"][int(idx)]
                        for idx in str(edge["source_id"]).split(",")
                        if idx.strip() != ""
                    )

            entities = [
                {"title": item[0], **(item[1] or {})}
                for item in graph.nodes(data=True) if item is not None
            ]
            relationships = nx.to_pandas_edgelist(graph)

            if "description" not in relationships.columns and "relation_raw" in relationships.columns:
                relationships = relationships.rename(columns={"relation_raw": "description"})
            elif "description" not in relationships.columns:
                relationships["description"] = ""

            entity_dfs.append(pd.DataFrame(entities))
            relationship_dfs.append(pd.DataFrame(relationships))

    entities = _merge_entities(entity_dfs)
    relationships = _merge_relationships(relationship_dfs)

    if "relation_raw" not in relationships.columns and "description" in relationships.columns:
        relationships = relationships.rename(columns={"description": "relation_raw"})

    print(f"Extracted {len(entities)} entities and {len(relationships)} relationships.")
    await write_table_to_storage(entities, "entities", context.storage)
    await write_table_to_storage(relationships, "relationships", context.storage)
    print("Entities and relationships written to storage.")


def finalize_entities_relationships_directed(entities, relationships):
    graph = nx.from_pandas_edgelist(
        relationships,
        edge_attr=["relation", "weight", "text_unit_ids"],
        create_using=nx.DiGraph(),
    )
    entities_ = entities.copy()
    entities_.set_index("title", inplace=True)
    graph.add_nodes_from((n, dict(d)) for n, d in entities_.iterrows())

    degrees = pd.DataFrame(
        [{"title": node, "degree": int(degree)} for node, degree in graph.degree]
    )
    final_entities = entities.merge(degrees, on="title", how="left").drop_duplicates(subset="title")
    final_entities = final_entities.loc[entities["title"].notna()].reset_index(drop=True)
    final_entities["degree"] = final_entities["degree"].fillna(0).astype(int)
    final_entities.reset_index(inplace=True)
    final_entities["human_readable_id"] = final_entities.index
    final_entities["id"] = final_entities["human_readable_id"].apply(lambda _: str(uuid4()))
    final_entities = final_entities.loc[
        :, ["id", "human_readable_id", "title", "type", "description", "text_unit_ids", "degree"],
    ]

    final_relationships = relationships.drop_duplicates(subset=["source", "target", "relation"]).copy()
    final_relationships["combined_degree"] = compute_edge_combined_degree(
        final_relationships, degrees,
        node_name_column="title", node_degree_column="degree",
        edge_source_column="source", edge_target_column="target",
    )
    final_relationships.reset_index(inplace=True, drop=True)
    final_relationships["human_readable_id"] = final_relationships.index
    final_relationships["id"] = final_relationships["human_readable_id"].apply(lambda _: str(uuid4()))
    final_relationships = final_relationships.loc[
        :, ["id", "human_readable_id", "source", "target", "relation", "weight", "combined_degree", "text_unit_ids"],
    ].rename(columns={"relation": "description"})

    return final_entities, final_relationships


# ------------------------------------------------
# PIPELINE 5: CLEAN AND FINALIZE
# ------------------------------------------------

def _write_relation_counts_txt(df_rel, out_path, allowed_relations, relation_col="relation"):
    allowed = [r.strip().lower() for r in allowed_relations]
    allowed_set = set(allowed)
    rel_series = df_rel[relation_col].astype(str).str.strip().str.lower()
    rel_series = rel_series[rel_series.isin(allowed_set)]
    counts = rel_series.value_counts()
    rows = [(r, int(counts.get(r, 0))) for r in allowed]
    rows.sort(key=lambda x: (-x[1], x[0]))
    with open(out_path, "w") as f:
        for r, c in rows:
            f.write(f"{r}\t{c}\n")


async def pipeline_5():
    entities = await load_table_from_storage("entities", context.storage)
    relationships = await load_table_from_storage("relationships", context.storage)

    entity_types = [
        "ANATOMICAL STRUCTURE", "MOLECULAR ENTITY", "CELLULAR COMPONENT",
        "PROCESS", "CLINICAL ENTITY", "CONCEPTUAL ENTITY",
    ]
    if "type" in entities.columns:
        entities["type"] = entities["type"].astype(str).str.upper()

    entities = entities[
        entities["type"].isin(entity_types)
        & (entities["description"].astype(str).str.len() > 0)
        & (entities["title"].astype(str).str.len() > 0)
    ].copy()
    entities["title"] = entities["title"].astype(str).str.strip().str.lower()

    cleaned_entities = entities.groupby(["title"], sort=False).agg(
        description=("description", lambda x: np.concatenate(x.values)),
        text_unit_ids=("text_unit_ids", lambda x: np.concatenate(x.values)),
        type=("type", "first"),
    ).reset_index()

    if "relation_raw" in relationships.columns:
        raw_col = "relation_raw"
    elif "description" in relationships.columns:
        raw_col = "description"
    else:
        raise KeyError(f"relationships missing relation column; columns={relationships.columns.tolist()}")

    relationships = relationships.copy()
    relationships["source"] = relationships["source"].astype(str).str.strip().str.lower()
    relationships["target"] = relationships["target"].astype(str).str.strip().str.lower()

    if "weight" not in relationships.columns:
        relationships["weight"] = 1.0
    else:
        relationships["weight"] = pd.to_numeric(relationships["weight"], errors="coerce").fillna(1.0)

    if "text_unit_ids" not in relationships.columns:
        relationships["text_unit_ids"] = relationships.get("source_id", "")

    allowed_relations = [r.strip().lower() for r in get_relation_types()]
    allowed_set = set(allowed_relations)

    alias_map = {
        "projects to": "projects_to",
        "receives": "receives_input_from",
        "receives_input": "receives_input_from",
        "input_from": "receives_input_from",
        "contains_representation_of": "encodes_representation_of",
        "encodes": "encodes_representation_of",
        "leads_to": "results_in",
        "resulting_from": "results_in",
    }

    def _flatten_rel_cell(x) -> list[str]:
        if x is None:
            return []
        items = list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]
        out: list[str] = []
        for it in items:
            for part in str(it).split("\n"):
                rel = part.strip().lower()
                if rel:
                    out.append(rel)
        return out

    def _canonicalize_rel(rel: str) -> str:
        rel = rel.strip().lower()
        rel = alias_map.get(rel, rel)
        rel = rel.replace(" ", "_")
        return rel

    relationships["relation_list"] = relationships[raw_col].apply(_flatten_rel_cell)
    relationships["relation_list"] = relationships["relation_list"].apply(
        lambda rels: [_canonicalize_rel(r) for r in rels]
    )
    relationships["relation_list"] = relationships["relation_list"].apply(
        lambda rels: sorted({r for r in rels if r in allowed_set})
    )
    relationships = relationships[
        (relationships["relation_list"].str.len() > 0)
        & (relationships["source"].str.len() > 0)
        & (relationships["target"].str.len() > 0)
        & (relationships["source"] != relationships["target"])
    ].reset_index(drop=True)

    relationships = relationships.explode("relation_list").reset_index(drop=True)
    relationships = relationships.rename(columns={"relation_list": "relation"})

    metrics_dir = ROOT_DIR / "output"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = metrics_dir / f"relation_counts_{RELATION_SET_NAME}_{ts}.txt"
    _write_relation_counts_txt(relationships, str(metrics_path), allowed_relations)

    # Write final parquet outputs
    out_dir = ROOT_DIR / "output"
    relationships.to_parquet(out_dir / "final_relationships.parquet", index=False)
    cleaned_entities.to_parquet(out_dir / "final_entities.parquet", index=False)
    print(f"Final KG written to {out_dir}")
    print(f"Entities: {len(cleaned_entities)}  Relationships: {len(relationships)}")


# ------------------------------------------------
# MAIN
# ------------------------------------------------

async def run_all():
    print("Step 1: Creating base text units...")
    await pipeline_1()
    print("Step 2: Creating final documents...")
    await pipeline_2()
    print("Step 3: Running LLM extraction...")
    await pipeline_3()
    print("Step 4: Parsing responses...")
    await pipeline_4()
    print("Step 5: Cleaning and finalizing KG...")
    await pipeline_5()


async def run_step(step: int):
    steps = {
        1: pipeline_1,
        2: pipeline_2,
        3: pipeline_3,
        4: pipeline_4,
        5: pipeline_5,
    }
    if step not in steps:
        raise ValueError(f"Unknown step {step}. Choose 1-5.")
    print(f"Running step {step}...")
    await steps[step]()
    print(f"Step {step} complete.")


if __name__ == "__main__":
    if ARGS.step is not None:
        asyncio.run(run_step(ARGS.step))
    else:
        asyncio.run(run_all())
