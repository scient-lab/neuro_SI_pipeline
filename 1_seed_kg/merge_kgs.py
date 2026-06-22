#!/usr/bin/env python3
"""
merge_seed_kg_relationships.py

Merge:
  - final_relationships.parquet
  - final_relationships_old.parquet

Into:
  expanded_seed_kg/final_relationships.parquet

Deduping:
  A) fast normalization-key collapse (includes conservative plural fix)
  B) OPTIONAL fuzzy endpoint collapse: if BOTH source and target are >= SIM_THRESHOLD similar
     (bucketed by prefix+length so it stays conservative)

Then:
  - FILTER relations with < MIN_REL_COUNT unique triples (source,target,relation)
  - PRINT which relations were removed + their counts

Outputs:
  - expanded_seed_kg/final_relationships.parquet
  - expanded_seed_kg/relation_counts.txt
"""

import argparse
import re
from uuid import uuid4
from pathlib import Path
from typing import Iterable, Set, Dict, List, Tuple
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


# -----------------------------
# Normalization helpers
# -----------------------------
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

def _basic_norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _simple_singularize_last_token(s: str) -> str:
    toks = s.split()
    if not toks:
        return s
    last = toks[-1]
    if len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        toks[-1] = last[:-1]
    return " ".join(toks)

def _norm_entity_for_keys(s: str) -> str:
    return _simple_singularize_last_token(_basic_norm_text(s))

def _norm_relation(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------
# text_unit_ids union helper
# -----------------------------
def _split_ids(x) -> Iterable[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for it in x:
            out.extend(_split_ids(it))
        return out
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def _union_ids(series: pd.Series) -> str:
    ids: Set[str] = set()
    for v in series.tolist():
        for tid in _split_ids(v):
            ids.add(tid)
    return ",".join(sorted(ids))


# -----------------------------
# IO + schema normalization
# -----------------------------
def _load_parquet(path: Path) -> pd.DataFrame:
    # Sniff format from extension: fact_score writes validated_triples as
    # CSV, graphrag writes kg_final as parquet — both paths flow through
    # this loader. Name kept for git-history clarity.
    if not path.exists():
        raise FileNotFoundError(f"Missing KG table: {path}")
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)

def _ensure_cols(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()

    if "relation" in df.columns:
        rel_col = "relation"
    elif "description" in df.columns:
        rel_col = "description"
    else:
        raise KeyError(f"[{label}] expected 'relation' or 'description'; got {df.columns.tolist()}")

    # Accept either 'source'/'target' (graphrag KG convention used in
    # kg_final.parquet) OR 'head'/'tail' (KG triple convention used
    # downstream by predict_tails_llm → combine_tails → fact_score).
    # Normalize to source/target so the rest of this module is unchanged.
    if "source" in df.columns and "target" in df.columns:
        pass
    elif "head" in df.columns and "tail" in df.columns:
        df = df.rename(columns={"head": "source", "tail": "target"})
    else:
        raise KeyError(
            f"[{label}] missing 'source'/'target' (graphrag) or 'head'/'tail' "
            f"(graphmert) column pair; got {df.columns.tolist()}"
        )

    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "combined_degree" not in df.columns:
        df["combined_degree"] = np.nan
    if "text_unit_ids" not in df.columns:
        if "source_id" in df.columns:
            df["text_unit_ids"] = df["source_id"]
        else:
            df["text_unit_ids"] = ""

    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["target"] = df["target"].astype(str).str.strip().str.lower()
    df["relation"] = df[rel_col].astype(str).map(_norm_relation)

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    df["combined_degree"] = pd.to_numeric(df["combined_degree"], errors="coerce")

    df = df[
        (df["source"].str.len() > 0)
        & (df["target"].str.len() > 0)
        & (df["relation"].str.len() > 0)
        & (df["source"] != df["target"])
    ].copy()

    return df[["source", "target", "relation", "weight", "combined_degree", "text_unit_ids"]]


# -----------------------------
# Stage A: fast collapse
# -----------------------------
def _fast_collapse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_key"] = df["source"].map(_norm_entity_for_keys)
    df["target_key"] = df["target"].map(_norm_entity_for_keys)
    df["rel_key"] = df["relation"].map(_norm_relation)
    df["k_key"] = df["source_key"] + "||" + df["target_key"] + "||" + df["rel_key"]

    df["_combined_degree_fill"] = df["combined_degree"].fillna(-1.0)
    df = df.sort_values(by=["_combined_degree_fill", "weight"], ascending=[False, False], kind="mergesort")

    grouped = df.groupby("k_key", sort=False, as_index=False)
    out = grouped.agg(
        source=("source", "first"),
        target=("target", "first"),
        relation=("relation", "first"),
        weight=("weight", "max"),
        combined_degree=("combined_degree", "max"),
        text_unit_ids=("text_unit_ids", _union_ids),
        source_key=("source_key", "first"),
        target_key=("target_key", "first"),
        rel_key=("rel_key", "first"),
    )
    return out


# -----------------------------
# Stage B: fuzzy endpoint collapse
# -----------------------------
def _build_entity_canon_map(entities: List[str], threshold: float) -> Dict[str, str]:
    uniq = sorted(set(entities))

    buckets: Dict[Tuple[str, int], List[str]] = {}
    for e in uniq:
        e_norm = _norm_entity_for_keys(e)
        prefix = (e_norm[:3] if len(e_norm) >= 3 else e_norm)
        len_bin = len(e_norm) // 5
        buckets.setdefault((prefix, len_bin), []).append(e)

    canon: Dict[str, str] = {}
    for _, bucket_ents in buckets.items():
        reps: List[str] = []
        for e in bucket_ents:
            if e in canon:
                continue
            e_norm = _norm_entity_for_keys(e)

            matched_rep = None
            for r in reps:
                r_norm = _norm_entity_for_keys(r)

                if r_norm and e_norm:
                    if abs(len(r_norm) - len(e_norm)) / max(len(r_norm), len(e_norm)) > 0.25:
                        continue

                if _sim(e_norm, r_norm) >= threshold:
                    matched_rep = r
                    break

            if matched_rep is None:
                reps.append(e)
                canon[e] = e
            else:
                chosen = matched_rep if len(matched_rep) <= len(e) else e
                other = e if chosen == matched_rep else matched_rep
                canon[e] = chosen
                canon[other] = chosen
                for k, v in list(canon.items()):
                    if v == other:
                        canon[k] = chosen

    for e in uniq:
        canon.setdefault(e, e)
    return canon

def _fuzzy_collapse_triples(df: pd.DataFrame, sim_threshold: float) -> pd.DataFrame:
    df = df.copy()
    all_entities = df["source"].tolist() + df["target"].tolist()
    canon_map = _build_entity_canon_map(all_entities, threshold=sim_threshold)

    df["source_canon"] = df["source"].map(lambda x: canon_map.get(x, x))
    df["target_canon"] = df["target"].map(lambda x: canon_map.get(x, x))

    df["k2"] = df["source_canon"] + "||" + df["target_canon"] + "||" + df["relation"]

    df["_combined_degree_fill"] = df["combined_degree"].fillna(-1.0)
    df = df.sort_values(by=["_combined_degree_fill", "weight"], ascending=[False, False], kind="mergesort")

    grouped = df.groupby("k2", sort=False, as_index=False)
    out = grouped.agg(
        source=("source_canon", "first"),
        target=("target_canon", "first"),
        relation=("relation", "first"),
        weight=("weight", "max"),
        combined_degree=("combined_degree", "max"),
        text_unit_ids=("text_unit_ids", _union_ids),
    )
    return out


# -----------------------------
# Relation filtering + metrics
# -----------------------------
def _filter_low_count_relations(
    df: pd.DataFrame,
    min_rel_count: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    df columns: source, target, relation, weight, combined_degree, text_unit_ids
    Removes relations with < min_rel_count unique triples.
    Returns (filtered_df, removed_counts_series)
    """
    # unique triples per relation
    counts = df.groupby("relation")[["source", "target"]].apply(lambda g: g.drop_duplicates().shape[0])
    keep_rels = counts[counts >= min_rel_count].index
    removed = counts[counts < min_rel_count].sort_values(ascending=False)

    filtered = df[df["relation"].isin(keep_rels)].copy()
    return filtered, removed

def write_relation_counts_txt(df_final: pd.DataFrame, path_txt: Path) -> pd.DataFrame:
    rel = df_final["description"].astype(str).str.strip().str.lower()
    counts = rel.value_counts(dropna=False)

    with open(path_txt, "w") as f:
        for r, c in counts.items():
            f.write(f"{r}\t{int(c)}\n")

    return counts.rename_axis("relation").reset_index(name="count")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True,
                    help="Path to new final_relationships.parquet (from graphrag step 5)")
    ap.add_argument("--old", default=None,
                    help="Path to old final_relationships.parquet to merge with (optional)")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for merged seed KG")
    ap.add_argument("--sim_threshold", type=float, default=0.90)
    ap.add_argument("--disable_fuzzy", action="store_true")
    ap.add_argument("--min_rel_count", type=int, default=80, help="Drop relations with < this many unique triples")
    args = ap.parse_args()

    new_path = Path(args.new)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[load] new: {new_path}")
    df_new = _ensure_cols(_load_parquet(new_path), label="new")

    if args.old is not None:
        old_path = Path(args.old)
        print(f"[load] old: {old_path}")
        df_old = _ensure_cols(_load_parquet(old_path), label="old")
    else:
        df_old = df_new.iloc[:0].copy()  # empty dataframe with same schema

    print(f"[stats] new rows: {len(df_new)}")
    print(f"[stats] old rows: {len(df_old)}")

    df_all = pd.concat([df_old, df_new], ignore_index=True)

    # Stage A
    collapsed = _fast_collapse(df_all)
    print(f"[stage A] after key collapse: {len(collapsed)}")

    # Stage B
    if not args.disable_fuzzy:
        collapsed2 = _fuzzy_collapse_triples(collapsed, sim_threshold=args.sim_threshold)
        print(f"[stage B] after fuzzy collapse (thr={args.sim_threshold}): {len(collapsed2)}")
    else:
        collapsed2 = collapsed[["source", "target", "relation", "weight", "combined_degree", "text_unit_ids"]].copy()

    # Strict dedupe
    collapsed2 = collapsed2.drop_duplicates(subset=["source", "target", "relation"]).reset_index(drop=True)

    # Filter out low-support relations
    filtered, removed = _filter_low_count_relations(collapsed2, min_rel_count=args.min_rel_count)

    if len(removed) > 0:
        print("\n=== REMOVED LOW-COUNT RELATIONS ===")
        for rel, cnt in removed.items():
            print(f"{rel}\t{int(cnt)}")
    else:
        print("\n=== REMOVED LOW-COUNT RELATIONS ===")
        print("(none)")

    print(f"\n[filter] kept rows: {len(filtered)} (min_rel_count={args.min_rel_count})")

    # Final ids + schema
    out = filtered.copy()
    out["human_readable_id"] = np.arange(len(out), dtype=int)
    out["id"] = out["human_readable_id"].apply(lambda _: str(uuid4()))
    out = out.rename(columns={"relation": "description"})
    out = out[
        ["id", "human_readable_id", "source", "target", "description", "weight", "combined_degree", "text_unit_ids"]
    ]

    out_parquet = outdir / "final_relationships.parquet"
    out.to_parquet(out_parquet, index=False)
    print(f"[write] merged parquet: {out_parquet} (rows={len(out)})")

    counts_txt = outdir / "relation_counts.txt"
    counts_df = write_relation_counts_txt(out, counts_txt)
    print(f"[write] relation counts: {counts_txt}")

    print("\n=== FINAL MERGE METRICS ===")
    print(f"Total merged unique triples: {len(out)}")
    print(f"Unique relations: {out['description'].nunique()}")

    print("\nTop relations (high -> low):")
    for _, row in counts_df.iterrows():
        print(f"{row['relation']}\t{int(row['count'])}")


if __name__ == "__main__":
    main()
