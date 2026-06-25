#!/usr/bin/env python3
"""step_quality.py — deterministic per-step output-quality probes.

Answers, for each (phase, step): did it produce MEANINGFUL output, or empty /
garbage? The manifest already tracks *did it run* (status); this tracks *did it
produce something usable* — the orthogonal signal whose absence cost a full
debugging session (extract "completed" with 3 triples; validate "completed"
dropping 100%; graphmert then ground empty data for 10 min).

Verdict ∈ {pass, warn, fail, skip, unknown} + a one-line reason. The same
verdict is the future quality-gate's input (fail → halt), so every check is a
PURE, DETERMINISTIC function of (artifacts + logs + already-declared config) —
no LLM, no randomness, same inputs → same verdict.

Expectations are anchored, not hand-guessed:
  - scale-invariant bands/ratios → configs/default.yaml::expectations.<phase>.<step>
  - scale anchors (targets, sizes) → the step's OWN signals (input row count,
    input-doc count, curriculum.num_questions) — never duplicated as a constant.

Stdlib-only (csv, json, os) so it runs under ANY phase venv and can later be
called inline from run_step / written to the manifest. Config is read via
pipeline_config when importable (full profile-aware merge); if pyyaml is absent
in the active venv it degrades to the in-code fallback bands below.

Usage:
  python scripts/lib/step_quality.py                 # table for current run
  python scripts/lib/step_quality.py --phase validate
  python scripts/lib/step_quality.py --json          # machine-readable
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Outcome enum (single severity axis; also the gate verdict). None = N/A (the
# step hasn't reached a judgeable state yet — pending/running).
PASS, WARN, FAIL, SKIP, UNKNOWN = "pass", "warn", "fail", "skip", "unknown"


# --- config access (graceful: falls back to in-code bands without pyyaml) -----
_CFG = None


def _cfg() -> dict:
    global _CFG
    if _CFG is None:
        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from pipeline_config import load_config
            _CFG = load_config()
        except Exception:
            _CFG = {}
    return _CFG


def exp(phase: str, step: str, key: str, fallback):
    """expectations.<phase>.<step>.<key> from the merged config, else fallback.
    The fallback MIRRORS configs/default.yaml::expectations (defensive only —
    YAML is authoritative)."""
    try:
        block = (((_cfg().get("expectations") or {}).get(phase) or {}).get(step) or {})
        v = block.get(key, fallback)
        return fallback if v is None else v
    except Exception:
        return fallback


def phase_param(phase: str, key: str, fallback):
    """cfg[<phase>][<key>] — used to read a step's OWN scale anchor (e.g.
    curriculum.num_questions) rather than duplicating it as a threshold."""
    try:
        v = (_cfg().get(phase) or {}).get(key, fallback)
        return fallback if v is None else v
    except Exception:
        return fallback


# --- small stdlib helpers -----------------------------------------------------
def csv_rows(path: Path):
    """Data-row count (excluding header), or None if missing/unreadable."""
    if not path.exists():
        return None
    try:
        with path.open(newline="", errors="replace") as f:
            r = csv.reader(f)
            next(r, None)  # header
            return sum(1 for _ in r)
    except Exception:
        return None


def read_log(step_rec: dict) -> list:
    """Lines of the step's log (manifest log_file is repo-relative). [] if none."""
    lf = (step_rec or {}).get("log_file")
    if not lf:
        return []
    for c in (lf, str(REPO_ROOT / lf), os.path.join(os.getcwd(), lf)):
        try:
            with open(c, errors="replace") as f:
                return f.read().splitlines()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            continue
        except Exception:
            return []
    return []


class V:
    """A verdict: outcome + human reason + machine metrics."""
    __slots__ = ("outcome", "reason", "metrics")

    def __init__(self, outcome, reason, metrics=None):
        self.outcome = outcome
        self.reason = reason
        self.metrics = metrics or {}


class Ctx:
    """What a probe gets: output base + the manifest step record."""
    def __init__(self, ob: Path, step_rec: dict):
        self.ob = ob
        self.step = step_rec

    def log(self):
        return read_log(self.step)


# --- probes (one per high-value step; anchored + deterministic) ---------------
def probe_extract_triples(c: Ctx) -> V:
    """Triples ÷ input-doc count — a scale-invariant ratio anchored to the
    step's own corpus size, so smoke and paper judge correctly with one rule."""
    t = csv_rows(c.ob / "graphrag" / "output" / "kg_final.csv")
    if t is None:
        return V(FAIL, "no kg_final.csv produced")
    docs = len(list((c.ob / "graphrag" / "input").glob("*.txt")))
    if t == 0:
        return V(FAIL, "0 triples extracted", {"triples": 0, "docs": docs})
    if docs == 0:
        return V(UNKNOWN, f"{t} triples but no input docs to anchor ratio", {"triples": t})
    ratio = t / docs
    fail_r = exp("extract", "extract_triples", "triples_per_doc_fail", 1.0)
    warn_r = exp("extract", "extract_triples", "triples_per_doc_warn", 5.0)
    m = {"triples": t, "docs": docs, "per_doc": round(ratio, 2)}
    if ratio < fail_r:
        return V(FAIL, f"{t} triples / {docs} docs = {ratio:.1f}/doc (<{fail_r} → near-empty)", m)
    if ratio < warn_r:
        return V(WARN, f"{t} triples / {docs} docs = {ratio:.1f}/doc (<{warn_r} → thin)", m)
    return V(PASS, f"{t} triples / {docs} docs = {ratio:.1f}/doc", m)


def probe_seed_kg_consensus(c: Ctx) -> V:
    """Drop rate vs the step's OWN input — self-referential, profile-free.
    0% filtered is as suspect as ~100% dropped."""
    out = csv_rows(c.ob / "graphrag" / "output" / "kg_final_validated.csv")
    inp = csv_rows(c.ob / "graphrag" / "output" / "kg_final.csv")
    if out is None:
        return V(FAIL, "no kg_final_validated.csv produced")
    if not inp:
        return V(UNKNOWN, "no input seed KG to compare against", {"out": out})
    drop = (inp - out) / inp
    m = {"in": inp, "out": out, "drop_pct": round(drop * 100, 1)}
    if out == 0:
        return V(FAIL, f"100% dropped ({inp}->0): parser/model broken", m)
    fail_hi = exp("validate", "seed_kg_consensus", "drop_fail_high", 0.95)
    warn_hi = exp("validate", "seed_kg_consensus", "drop_warn_high", 0.60)
    warn_lo = exp("validate", "seed_kg_consensus", "drop_warn_low", 0.01)
    if drop >= fail_hi:
        return V(FAIL, f"{drop*100:.0f}% dropped ({inp}->{out}): near-total, likely broken", m)
    if drop <= warn_lo:
        return V(WARN, f"only {drop*100:.1f}% filtered ({inp}->{out}): consensus may be a no-op", m)
    if drop > warn_hi:
        return V(WARN, f"{drop*100:.0f}% dropped ({inp}->{out}): low yield", m)
    return V(PASS, f"{drop*100:.0f}% filtered, {out}/{inp} kept", m)


def probe_preprocess(c: Ctx) -> V:
    """Grounding success count from the step's own log ('Grounding results:
    {json}'). 0 grounded = no trainable samples = the success==0 dead end."""
    success = None
    for ln in c.log():
        i = ln.find("Grounding results:")
        if i != -1:
            try:
                stats = json.loads(ln[i + len("Grounding results:"):].strip())
                success = int(stats.get("success", 0))
            except Exception:
                pass
    dataset = c.ob / "graphmert" / "dataset"
    if success is None:
        if dataset.exists() and any(dataset.iterdir()):
            return V(UNKNOWN, "no 'Grounding results' line in log; dataset dir present")
        return V(FAIL, "no grounding stats and no dataset produced")
    min_ok = exp("graphmert", "preprocess", "min_grounding_success", 1)
    if success < min_ok:
        return V(FAIL, f"grounding success == {success} (<{min_ok}): no trainable samples", {"success": success})
    return V(PASS, f"grounding success == {success}", {"success": success})


def probe_generate_qa(c: Ctx) -> V:
    """Q&A produced vs the profile's OWN configured target (num_questions) —
    anchored to existing config, so no per-profile threshold needed."""
    path = c.ob / "curriculum" / "curriculum.json"
    if not path.exists():
        return V(FAIL, "no curriculum.json produced")
    try:
        data = json.loads(path.read_text())
        got = len(data) if isinstance(data, list) else 0
    except Exception:
        return V(FAIL, "curriculum.json present but unreadable/non-list")
    target = int(phase_param("curriculum", "num_questions", 0) or 0)
    if got == 0:
        return V(FAIL, "0 Q&A items generated", {"got": 0, "target": target})
    if target:
        frac_warn = exp("curriculum", "generate_qa", "frac_of_target_warn", 0.5)
        if got < frac_warn * target:
            return V(WARN, f"{got} Q&A < {int(frac_warn*100)}% of target {target}", {"got": got, "target": target})
        return V(PASS, f"{got} Q&A (target {target})", {"got": got, "target": target})
    return V(PASS, f"{got} Q&A items", {"got": got})


PROBES = {
    ("extract", "extract_triples"): probe_extract_triples,
    ("validate", "seed_kg_consensus"): probe_seed_kg_consensus,
    ("graphmert", "preprocess"): probe_preprocess,
    ("curriculum", "generate_qa"): probe_generate_qa,
}


def evaluate(phase: str, step_rec: dict, ob: Path) -> V:
    """Map a manifest step to a verdict. Status gates first (a step that
    hasn't completed can't have a quality outcome), then the bespoke probe,
    else UNKNOWN (never a silent pass for an un-probed completed step)."""
    status = (step_rec or {}).get("status", "pending")
    if status in ("pending", "running"):
        return V(None, status)                      # not judgeable yet → "—"
    if status == "skipped":
        return V(SKIP, "step skipped")
    if status == "failed":
        return V(FAIL, f"step failed (exit {step_rec.get('exit_code')})")
    probe = PROBES.get((phase, step_rec.get("name")))
    if probe is None:
        return V(UNKNOWN, "no probe defined")
    try:
        return probe(Ctx(ob, step_rec))
    except Exception as e:                            # a probe bug must not crash the report
        return V(UNKNOWN, f"probe error: {e}")


# --- CLI / rendering ----------------------------------------------------------
_GLYPH = {PASS: "● pass", WARN: "◐ warn", FAIL: "✗ fail",
          SKIP: "– skip", UNKNOWN: "? unkn", None: "—"}
_COLOR = {PASS: "32", WARN: "33", FAIL: "31", SKIP: "90", UNKNOWN: "90", None: "90"}


def _paint(outcome, tty):
    g = _GLYPH[outcome]
    if not tty:
        return g
    return f"\033[{_COLOR[outcome]}m{g}\033[0m"


def _print_table(run: dict, rows: list) -> None:
    tty = sys.stdout.isatty()
    print(f"Run     : {run.get('run_id')}")
    print(f"Profile : {run.get('profile')}    Domain: {run.get('domain')}\n")
    print(f"  {'PHASE/STEP':<34} {'STATUS':<11} {'OUTCOME':<8} REASON")
    print(f"  {'-'*34} {'-'*11} {'-'*8} {'-'*44}")
    for r in rows:
        name = f"{r['phase']}.{r['step']}"
        print(f"  {name:<34} {str(r['status']):<11} {_paint(r['outcome'], tty):<8} {r['reason']}")
    # headline: the dangerous combo — completed but not meaningful
    bad = [r for r in rows if r["status"] == "completed" and r["outcome"] in (FAIL, WARN)]
    if bad:
        print("\n  ⚠ completed but not meaningful:")
        for r in bad:
            print(f"      {r['phase']}.{r['step']}: {r['outcome']} — {r['reason']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None, help="run_manifest.json (default: $OUTPUT_BASE/run_manifest.json)")
    ap.add_argument("--output-base", default=None, help="outputs/ dir (default: $OUTPUT_BASE or repo outputs/)")
    ap.add_argument("--phase", default=None, help="limit to one phase")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--write", action="store_true",
                    help="persist outcome+reason into the manifest (for stats.sh OUTCOME column)")
    a = ap.parse_args()

    ob = Path(a.output_base or os.environ.get("OUTPUT_BASE") or (REPO_ROOT / "outputs"))
    manifest_path = Path(a.manifest) if a.manifest else ob / "run_manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    doc = json.loads(manifest_path.read_text())
    run = doc.get("run", {})
    # Align the config layer to THIS run's profile/domain so anchors +
    # expectations resolve at the right scale (num_questions etc.).
    if run.get("profile"):
        os.environ.setdefault("SI_PROFILE", run["profile"])
    if run.get("domain"):
        os.environ.setdefault("SI_DOMAIN", run["domain"])

    rows = []
    for p in run.get("phases", []):
        if a.phase and p.get("name") != a.phase:
            continue
        for s in p.get("steps", []):
            v = evaluate(p["name"], s, ob)
            rows.append({"phase": p["name"], "step": s.get("name"),
                         "status": s.get("status"), "outcome": v.outcome,
                         "reason": v.reason, "metrics": v.metrics})

    # Persist outcomes into the manifest so stats.sh can render the OUTCOME
    # column. Only steps with a real verdict (not pending/running → None).
    if a.write:
        outs = [{"phase": r["phase"], "step": r["step"],
                 "outcome": r["outcome"], "reason": r["reason"]}
                for r in rows if r["outcome"] is not None]
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import manifest as _m
            _m.apply_outcomes(str(manifest_path), outs)
            if not a.json:
                print(f"  wrote {len(outs)} outcomes -> {manifest_path}\n")
        except Exception as e:
            print(f"  failed to write outcomes: {e}", file=sys.stderr)

    if a.json:
        print(json.dumps({"run_id": run.get("run_id"), "steps": rows}, indent=2))
        return 0
    _print_table(run, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
