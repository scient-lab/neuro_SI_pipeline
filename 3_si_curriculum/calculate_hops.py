#!/usr/bin/env python3
"""
calculate_hops.py

Computes the hop distance of each triple in the final KG relative to the seed KG.
Hop distance = minimum number of edges between the expanded triple and any seed triple.

Usage:
  python calculate_hops.py \\
    --kg_path      ${OUTPUT_BASE}/final_kg/validated_final_kg.csv \\
    --seed_kg_path ${OUTPUT_BASE}/final_seedkg/neuroscience_kg.csv \\
    --output_path  ${OUTPUT_BASE}/final_kg/all_hops_detailed.csv

Inputs:
  kg_path:      Full expanded KG CSV (head, relation, tail)
  seed_kg_path: Seed KG CSV (head, relation, tail)

Output:
  all_hops_detailed.csv: original columns + hop_distance column
"""

import argparse
import logging
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Compute hop distances from seed KG")
    ap.add_argument("--kg_path", required=True,
                    help="Full expanded KG (CSV or parquet; cols head/relation/tail)")
    ap.add_argument("--seed_kg_path", required=True,
                    help="Seed KG (CSV or parquet; cols head/relation/tail OR "
                         "graphrag-native source/target/description)")
    ap.add_argument("--output_path", required=True,
                    help="Output CSV path with hop_distance column added")
    ap.add_argument("--allow-seed-only", action="store_true",
                    help="if the expanded KG (kg_path) is empty, fall back to a "
                         "seed-only 1-hop curriculum instead of failing. Off by "
                         "default so an empty expansion fails loudly rather than "
                         "silently degrading (curriculum.allow_seed_only_fallback)")
    return ap.parse_args()


def _load_kg(path: str) -> pd.DataFrame:
    """Load a KG file as a DataFrame with normalized head/relation/tail columns.

    Auto-detects parquet vs CSV and renames graphrag's native
    source/target/description columns to head/tail/relation so downstream
    code can work off a single schema. Ported from upstream main 4d876bc.
    """
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if "source" in df.columns and "head" not in df.columns:
        df = df.rename(columns={"source": "head", "target": "tail"})
    if "description" in df.columns and "relation" not in df.columns:
        df = df.rename(columns={"description": "relation"})
    return df


def build_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        h = str(row["head"]).strip().lower()
        t = str(row["tail"]).strip().lower()
        if h and t:
            G.add_edge(h, t)
    return G


def compute_hop_distances(full_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each triple in full_df, compute the shortest path distance to any
    entity in the seed KG using a graph built from the full KG.
    """
    logger.info("Building full KG graph...")
    G = build_graph(full_df)

    # Seed entities
    seed_entities = set()
    for _, row in seed_df.iterrows():
        seed_entities.add(str(row["head"]).strip().lower())
        seed_entities.add(str(row["tail"]).strip().lower())

    logger.info("Seed entities: %d", len(seed_entities))
    logger.info("Full KG nodes: %d  edges: %d", G.number_of_nodes(), G.number_of_edges())

    # Find distances from all seed nodes (BFS from multiple sources)
    seed_nodes_in_graph = seed_entities & set(G.nodes())
    if not seed_nodes_in_graph:
        logger.warning("No seed entities found in full KG graph — all hops will be infinity")
        full_df["hop_distance"] = float("inf")
        return full_df

    logger.info("Computing BFS distances from %d seed nodes...", len(seed_nodes_in_graph))
    distances = {}
    for source in seed_nodes_in_graph:
        if source not in G:
            continue
        lengths = nx.single_source_shortest_path_length(G, source)
        for node, dist in lengths.items():
            if node not in distances or dist < distances[node]:
                distances[node] = dist

    # Assign hop distance to each triple (min of head, tail distances)
    hop_distances = []
    for _, row in full_df.iterrows():
        h = str(row["head"]).strip().lower()
        t = str(row["tail"]).strip().lower()
        d_h = distances.get(h, float("inf"))
        d_t = distances.get(t, float("inf"))
        hop_distances.append(min(d_h, d_t))

    full_df = full_df.copy()
    full_df["hop_distance"] = hop_distances
    return full_df


def main():
    args = parse_args()

    logger.info("Loading full KG: %s", args.kg_path)
    full_df = _load_kg(args.kg_path)
    logger.info("Full KG: %d triples", len(full_df))

    logger.info("Loading seed KG: %s", args.seed_kg_path)
    seed_df = _load_kg(args.seed_kg_path)
    logger.info("Seed KG: %d triples", len(seed_df))

    # Smoke / no-graphmert-expansion fallback.
    # At smoke scale (or any run where graphmert.validate_predictions
    # produces 0 validated triples), `full_df` is empty and the original
    # logic emits a manifest with 0 rows → generate_curriculum.py loads 0
    # paths → curriculum.generate_qa crashes with "No paths found".
    #
    # Treat that case as: use the seed KG itself as the path source, with
    # `hop_distance=1` so each seed triple counts as a 1-hop path under
    # `load_paths_from_manifest`'s `min_hops..max_hops` filter (default
    # [1, 5]). This lets curriculum generate questions directly off the
    # seed KG when graphmert has nothing useful to add.
    #
    # NOTE: the hop_distance=1 assignment is a labeling convenience, not
    # an actual hop-graph distance — seed entities are at distance 0 from
    # themselves. At pilot/paper scale, full_df will have real expanded
    # triples and the original `compute_hop_distances` path runs.
    if len(full_df) == 0:
        if not args.allow_seed_only:
            logger.error(
                "calculate_hops: kg_path (%s) has 0 rows — graphmert produced NO "
                "validated expansion triples. Refusing to silently fall back to a "
                "seed-only 1-hop curriculum (that would hide a failed graphmert / "
                "validate stage). Pass --allow-seed-only "
                "(curriculum.allow_seed_only_fallback=true) to permit it.",
                args.kg_path)
            sys.exit(1)
        logger.warning(
            "full_df (kg_path) has 0 rows — graphmert produced no validated "
            "expansion triples. Falling back to the seed KG as the path source "
            "with hop_distance=1 (--allow-seed-only set), so curriculum can still "
            "generate questions from seed triples."
        )
        result_df = seed_df.copy()
        result_df["hop_distance"] = 1
    else:
        result_df = compute_hop_distances(full_df, seed_df)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    logger.info("Hop distance distribution:")
    logger.info("%s", result_df["hop_distance"].value_counts().sort_index().to_string())
    logger.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
