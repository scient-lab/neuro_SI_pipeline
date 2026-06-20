#!/usr/bin/env python3
"""Quality analysis for the extract phase output (graphrag's kg_final.csv).

Sections:
  §1 Structure       header + column-count + malformation
  §2 Vocabulary      unique heads/tails/relations; coverage vs domain config
  §3 Concentration   relation histogram, top-K heads
  §4 Duplicates      near-duplicate heads/tails (entity-resolution opportunities)
  §5 Direction       asymmetric relations with likely direction errors
  §6 Inverse         redundant pairs like (A part_of B) AND (B contains A)
  §7 Verdict         summary + exit code (0 clean, 1 FAIL, 2 WARN-only)

Invoked by scripts/analysis.sh; can also be run directly for debugging.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Reporter — OK / WARN / FAIL output with --json + --quiet modes
# ---------------------------------------------------------------------------
class Reporter:
    def __init__(self, json_mode: bool = False, quiet: bool = False):
        self.json_mode = json_mode
        self.quiet = quiet
        self.results: list[dict] = []
        self.section_name = ""
        self.fails = 0
        self.warns = 0

    def section(self, name: str) -> None:
        self.section_name = name
        if not self.json_mode:
            print(f"\n [§{name}]")

    def ok(self, msg: str) -> None:
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "ok", "msg": msg})
        elif not self.quiet:
            print(f"  OK   {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "warn", "msg": msg})
        else:
            print(f"  WARN {msg}")

    def fail(self, msg: str) -> None:
        self.fails += 1
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "fail", "msg": msg})
        else:
            print(f"  FAIL {msg}")

    def note(self, msg: str) -> None:
        if self.json_mode or self.quiet:
            return
        print(f"       {msg}")

    def emit(self) -> int:
        if self.json_mode:
            print(json.dumps({
                "fails": self.fails, "warns": self.warns,
                "results": self.results,
            }, indent=2))
        else:
            verdict = "CLEAN" if self.fails == 0 and self.warns == 0 else \
                      "PASS WITH WARNINGS" if self.fails == 0 else "FAIL"
            print(f"\n  VERDICT: {self.fails} failures, {self.warns} warnings — {verdict}")
        return 1 if self.fails > 0 else (2 if self.warns > 0 else 0)


# ---------------------------------------------------------------------------
# Domain config loader — reads relations: from domains/<SI_DOMAIN>.yaml.
# Tries PyYAML first; falls back to a regex parse so this script still works
# without yaml installed (e.g. system python3 outside any venv).
# ---------------------------------------------------------------------------
def load_domain_relations(repo_root: Path) -> set[str]:
    domain = os.environ.get("SI_DOMAIN", "neuroscience").strip() or "neuroscience"
    yaml_path = repo_root / "domains" / f"{domain}.yaml"
    if not yaml_path.exists():
        return set()
    text = yaml_path.read_text()
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        rels: set[str] = set()
        for r in data.get("relations", []):
            if isinstance(r, dict) and "id" in r:
                rels.add(str(r["id"]))
            elif isinstance(r, str):
                rels.add(r)
        return rels
    except ImportError:
        pass
    # Regex fallback for inline-dict `{ id: foo, description: ... }` entries
    rels = set()
    m = re.search(r'^relations:\s*\n((?:\s*-.*\n)+)', text, re.M)
    if m:
        for line in m.group(1).split('\n'):
            m2 = re.search(r'id:\s*([a-z_]+)', line)
            if m2:
                rels.add(m2.group(1))
    return rels


# ---------------------------------------------------------------------------
# CSV discovery — try $OUTPUT_BASE/graphrag/output/, then logs/, then user
# override via --csv.
# ---------------------------------------------------------------------------
def find_kg_csv(repo_root: Path, run_prefix: Optional[str] = None) -> Optional[Path]:
    output_base = Path(os.environ.get("OUTPUT_BASE", repo_root / "outputs"))
    candidates: list[Path] = []
    candidates.extend(output_base.glob("graphrag/output/kg_final*.csv"))
    candidates.extend((repo_root / "logs").glob("kg_final*.csv"))
    if run_prefix:
        candidates = [p for p in candidates if run_prefix in str(p)]
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


# ---------------------------------------------------------------------------
# Near-duplicate entity detection — pairs where one is substring of the other
# AND lengths are within +5 chars (catches "glutamate" vs "glutamic acid").
# Bounded result set so noise doesn't drown the report.
# ---------------------------------------------------------------------------
def find_near_duplicates(entities: set[str], max_pairs: int = 30) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    ents = sorted({e for e in entities if len(e) >= 4})
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(ents):
        for b in ents[i + 1:]:
            if len(b) - len(a) > 5:
                break
            if abs(len(a) - len(b)) <= 5 and (a in b or b in a):
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


# ---------------------------------------------------------------------------
# Direction-error heuristics — for known asymmetric relations, flag triples
# where head/tail look swapped relative to the relation's typical usage.
# Conservative: only flags when head matches "good_tails" keywords AND tail
# matches "good_heads" keywords (the swapped pattern).
# ---------------------------------------------------------------------------
ASYMMETRIC_RELATIONS: dict[str, dict[str, list[str]]] = {
    "transports": {
        "good_heads": ["transporter", "channel", "carrier", "pump", "exchanger"],
        "good_tails": ["ion", "molecule", "neurotransmitter", "amino acid"],
    },
    "expressed_in": {
        "good_heads": ["gene", "protein", "receptor", "enzyme", "factor", "kinase"],
        "good_tails": ["cell", "neuron", "tissue", "region", "cortex", "hippocampus", "nucleus"],
    },
    "synthesized_in": {
        "good_heads": ["acid", "amine", "neurotransmitter", "molecule", "peptide"],
        "good_tails": ["cell", "neuron", "synapse", "axon", "soma", "cytoplasm", "terminal"],
    },
    "binds_to": {
        "good_heads": ["ligand", "neurotransmitter", "agonist", "antagonist", "drug"],
        "good_tails": ["receptor", "channel", "site", "binding"],
    },
    "projects_to": {
        "good_heads": ["nucleus", "neuron", "cell", "fiber", "tract", "pathway"],
        "good_tails": ["cortex", "region", "area", "nucleus"],
    },
}


def check_direction_heuristics(rows: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    issues: dict[str, list[tuple[str, str]]] = {}
    for h, t, r in rows:
        rules = ASYMMETRIC_RELATIONS.get(r)
        if not rules:
            continue
        h_lo, t_lo = h.lower(), t.lower()
        h_looks_like_tail = any(g in h_lo for g in rules["good_tails"])
        t_looks_like_head = any(g in t_lo for g in rules["good_heads"])
        h_correct = any(g in h_lo for g in rules["good_heads"])
        t_correct = any(g in t_lo for g in rules["good_tails"])
        # Flag only when both sides look swapped AND not also correct-looking
        if h_looks_like_tail and t_looks_like_head and not (h_correct and t_correct):
            issues.setdefault(r, []).append((h, t))
    return issues


# ---------------------------------------------------------------------------
# Inverse-relation redundancy — pairs of relations that should be one-way.
# Counts triples where (A rel_a B) AND (B rel_b A) coexist.
# ---------------------------------------------------------------------------
INVERSE_PAIRS = [
    ("part_of", "contains"),
    ("connected_to", "connected_to"),  # symmetric — both directions allowed
    ("projects_to", "receives_input_from"),
]


def count_inverse_redundancy(rows: list[tuple[str, str, str]], rel_a: str, rel_b: str) -> int:
    a_pairs = {(h, t) for h, t, r in rows if r == rel_a}
    b_pairs = {(t, h) for h, t, r in rows if r == rel_b}
    return len(a_pairs & b_pairs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run", default=None)
    p.add_argument("--step", default=None)  # accepted for compat; ignored
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    csv_path = Path(args.csv) if args.csv else find_kg_csv(repo_root, args.run)
    rpt = Reporter(json_mode=args.json, quiet=args.quiet)

    if not args.json:
        print(f"=== Extract phase quality report ===")

    if not csv_path or not csv_path.exists():
        rpt.section("0. CSV discovery")
        rpt.fail("No kg_final.csv found — checked $OUTPUT_BASE/graphrag/output/ and logs/")
        return rpt.emit()

    if not args.json:
        print(f"  source: {csv_path.relative_to(repo_root) if repo_root in csv_path.parents else csv_path}")

    # ----- Load CSV -----
    rows: list[tuple[str, str, str]] = []
    bad_rows: list[tuple[int, list[str]]] = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, [])
        for i, row in enumerate(reader, 2):
            if len(row) != 3:
                bad_rows.append((i, row))
            else:
                rows.append(tuple(s.strip() for s in row))  # type: ignore[arg-type]

    # ----- §1 Structure -----
    rpt.section("1. Structure")
    size_bytes = csv_path.stat().st_size
    size_kb = size_bytes / 1024
    density = len(rows) / size_kb if size_kb > 0 else 0
    rpt.ok(f"{len(rows)} triples in {size_kb:.1f} KB ({density:.2f} triples/KB)")
    if header == ["head", "tail", "relation"]:
        rpt.ok(f"Header: {','.join(header)}")
    else:
        rpt.warn(f"Header: {','.join(header)} (expected head,tail,relation)")
    if bad_rows:
        rpt.fail(f"{len(bad_rows)} malformed rows (column count != 3)")
        for ln, row in bad_rows[:3]:
            rpt.note(f"  line {ln}: {row}")
    else:
        rpt.ok(f"All {len(rows)} rows have 3 columns")

    if not rows:
        return rpt.emit()

    heads = [r[0] for r in rows]
    tails = [r[1] for r in rows]
    rels = [r[2] for r in rows]
    head_counts = Counter(heads)
    tail_counts = Counter(tails)
    rel_counts = Counter(rels)
    head_set = set(heads)
    tail_set = set(tails)
    rel_set = set(rels)

    # ----- §2 Vocabulary coverage -----
    rpt.section("2. Vocabulary coverage")
    rpt.ok(f"{len(head_set)} unique heads, {len(tail_set)} unique tails")
    allowed = load_domain_relations(repo_root)
    if not allowed:
        rpt.warn("Could not load domain relations from domains/<domain>.yaml — skipping vocab compliance check")
    else:
        out_of_vocab = rel_set - allowed
        unused = allowed - rel_set
        rpt.ok(f"{len(rel_set)} of {len(allowed)} domain-declared relations used")
        if out_of_vocab:
            rpt.fail(f"{len(out_of_vocab)} relations outside domain config: {sorted(out_of_vocab)[:10]}")
        else:
            rpt.ok("0 relations outside domain config")
        if unused:
            rpt.note(f"{len(unused)} declared but unused: {sorted(unused)[:10]}")

    # ----- §3 Concentration -----
    rpt.section("3. Concentration")
    top_rel, top_rel_n = rel_counts.most_common(1)[0]
    top_rel_pct = 100 * top_rel_n / len(rows)
    (rpt.warn if top_rel_pct > 20 else rpt.ok)(
        f"Top relation '{top_rel}' has {top_rel_n} ({top_rel_pct:.1f}%)"
    )
    top_head, top_head_n = head_counts.most_common(1)[0]
    top_head_pct = 100 * top_head_n / len(rows)
    (rpt.warn if top_head_pct > 25 else rpt.ok)(
        f"Top head '{top_head}' has {top_head_n} ({top_head_pct:.1f}%)"
    )
    if not args.json:
        rpt.note(f"Top-{args.top} relations:")
        for rel, cnt in rel_counts.most_common(args.top):
            rpt.note(f"  {cnt:5d}  {rel}")
        rpt.note(f"Top-{args.top} heads:")
        for head, cnt in head_counts.most_common(args.top):
            rpt.note(f"  {cnt:5d}  {head}")

    # ----- §4 Near-duplicate entities -----
    rpt.section("4. Near-duplicate entity candidates")
    near_dupes = find_near_duplicates(head_set | tail_set, max_pairs=30)
    if not near_dupes:
        rpt.ok("No obvious near-duplicate entity pairs detected")
    else:
        rpt.warn(f"{len(near_dupes)} near-duplicate entity pair(s) — entity-resolution opportunity")
        for a, b in near_dupes[:5]:
            ca = head_counts.get(a, 0) + tail_counts.get(a, 0)
            cb = head_counts.get(b, 0) + tail_counts.get(b, 0)
            rpt.note(f"  '{a}' ({ca}) <-> '{b}' ({cb})")
        if len(near_dupes) > 5:
            rpt.note(f"  ... and {len(near_dupes) - 5} more (use --json for full list)")

    # ----- §5 Direction-error heuristics -----
    rpt.section("5. Direction-error heuristics on asymmetric relations")
    direction_issues = check_direction_heuristics(rows)
    if not direction_issues:
        rpt.ok(f"No obvious direction errors in {len(ASYMMETRIC_RELATIONS)} checked relations")
    else:
        for rel, suspects in direction_issues.items():
            rpt.warn(f"'{rel}' has {len(suspects)} possible direction error(s):")
            for h, t in suspects[:3]:
                rpt.note(f"  {h} -> {rel} -> {t}")
            if len(suspects) > 3:
                rpt.note(f"  ... and {len(suspects) - 3} more")

    # ----- §6 Inverse-relation redundancy -----
    rpt.section("6. Inverse-relation redundancy")
    any_redundancy = False
    for rel_a, rel_b in INVERSE_PAIRS:
        if rel_a == rel_b:
            continue  # symmetric — skip
        if rel_a in rel_counts and rel_b in rel_counts:
            r = count_inverse_redundancy(rows, rel_a, rel_b)
            if r > 0:
                any_redundancy = True
                rpt.warn(
                    f"{rel_counts[rel_a]} '{rel_a}' + {rel_counts[rel_b]} '{rel_b}' "
                    f"coexist; {r} reciprocal pair(s) (A {rel_a} B AND B {rel_b} A)"
                )
                rpt.note("Recommend canonical direction in extraction prompts")
    if not any_redundancy:
        rpt.ok("No inverse-relation redundancy detected")

    return rpt.emit()


if __name__ == "__main__":
    sys.exit(main())
