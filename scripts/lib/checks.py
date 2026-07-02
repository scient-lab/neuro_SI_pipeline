#!/usr/bin/env python3
"""checks.py — shared engine for the standardized run-inspection views.

Two lenses over one record (see docs/DIAGNOSE_ANALYSIS_STANDARDIZATION_PLAN_2026-06-29.md):

  - HEALTH  (diagnose): "can I proceed / WHERE is it broken?" — structural
    integrity + failure localization. Per step: the manifest status, the real
    exception parsed from the step's LOG (so a crash pinpoints file:line), and
    the I/O-contract state (declared inputs/outputs) so a silent failure
    pinpoints the empty/missing file.
  - QUALITY (analysis): "is the output GOOD?" — graded probes. Reuses
    step_quality.PROBES / evaluate() (already covers extract/validate/
    graphmert/curriculum); adds a per-step sample() preview.

Record = step_quality.V(outcome, reason, metrics). A Check ties a V to the file
it concerns. Stdlib-first; parquet row counts use pandas when importable, else
file presence (so HEALTH runs under any venv).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

import step_quality as sq  # noqa: E402  (the quality engine + primitives)
from step_quality import V, Ctx, PASS, WARN, FAIL, SKIP, UNKNOWN  # noqa: E402,F401

REPO_ROOT = Path(__file__).resolve().parents[2]


class Check:
    """A named check: a verdict (V) tied to the file path it concerns."""
    __slots__ = ("name", "v", "path")

    def __init__(self, name: str, v: V, path: str | None = None):
        self.name, self.v, self.path = name, v, path


_SEV = {FAIL: 3, WARN: 2, UNKNOWN: 1, PASS: 0, SKIP: 0, None: 0}


def worst(outcomes):
    """Worst outcome (for step/run roll-up); None if no judgeable checks."""
    o, best = None, -1
    for x in outcomes:
        s = _SEV.get(x, 0)
        if s > best:
            best, o = s, x
    return o


# --- I/O contract: per-(phase, step) declared inputs / outputs ---------------
class StepSpec:
    """Declares what a step reads/writes + how to sample/inspect it.

    inputs/outputs are paths or globs RELATIVE to the run's output base.
    A missing INPUT localizes upstream (WARN here); a missing/empty OUTPUT is
    this step's failure (FAIL). sample(ob, n) previews the primary output.
    """
    __slots__ = ("inputs", "outputs", "sample", "extra_health", "inspect")

    def __init__(self, inputs=None, outputs=None, sample=None,
                 extra_health=None, inspect=None):
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.sample = sample
        self.extra_health = extra_health
        self.inspect = inspect or []


SPECS: dict[tuple[str, str], StepSpec] = {}


def spec(phase: str, step: str, **kw) -> None:
    SPECS[(phase, step)] = StepSpec(**kw)


# --- path state -------------------------------------------------------------
def _is_glob(s: str) -> bool:
    return any(c in s for c in "*?[")


def path_state(ob: Path, pattern: str) -> dict:
    """Resolve <ob>/<pattern> (or glob) → {exists, rows?, cols?, size, paths}."""
    if _is_glob(pattern):
        matches = [m for m in sorted(ob.glob(pattern)) if m.exists()]
    else:
        p = ob / pattern
        matches = [p] if p.exists() else []
    if not matches:
        return {"pattern": pattern, "exists": False}
    st = {"pattern": pattern, "exists": True, "size": 0,
          "paths": [str(m.relative_to(ob)) for m in matches]}
    rows, have_rows, cols = 0, False, None
    for m in matches:
        if m.is_dir():
            st["size"] += sum(f.stat().st_size for f in m.rglob("*") if f.is_file())
            hf = _hf_rows(m)        # HuggingFace dataset dir → num_examples
            if hf is not None:
                rows += hf; have_rows = True       # catches "schema declared, 0 rows"
            else:
                n_txt = sum(1 for _ in m.glob("*.txt"))
                if n_txt:
                    rows += n_txt; have_rows = True  # dir-of-docs → file count as "rows"
            continue
        st["size"] += m.stat().st_size
        if m.suffix == ".csv":
            r = sq.csv_rows(m)
            if r is not None:
                rows += r; have_rows = True
        elif m.suffix == ".parquet":
            r, c = _parquet_rows(m)
            if r is not None:
                rows += r; have_rows = True
                cols = c or cols
    if have_rows:
        st["rows"] = rows
    if cols is not None:
        st["cols"] = cols
    return st


def _parquet_rows(p: Path):
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        return len(df), list(df.columns)
    except Exception:
        return None, None


def _hf_rows(d: Path):
    """A HuggingFace dataset dir's row count via dataset_info.json::splits
    (stdlib, no `datasets` import). None if not an HF dataset dir."""
    info = d / "dataset_info.json"
    if not info.is_file():
        return None
    try:
        splits = (json.loads(info.read_text()).get("splits") or {})
        if isinstance(splits, dict):
            return sum(int(s.get("num_examples", 0) or 0) for s in splits.values())
    except Exception:
        return None
    return None


# --- failure localizer (Class 1: thrown exceptions) -------------------------
_TB_START = "Traceback (most recent call last):"
_FILE_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')
_EXC_RE = re.compile(r'^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt))'
                     r'(?:\((?P<errno>[^)]*)\))?: ?(?P<msg>.*)$')
# Non-traceback fatal signals (OOM-kill, CUDA assert, segfault), scanned as fallback.
_FATAL_SIGNALS = [
    ("CUDA out of memory", "GPU OOM"),
    ("Killed", "process killed (likely host OOM)"),
    ("Segmentation fault", "segfault"),
    ("RuntimeError: CUDA", "CUDA runtime error"),
    ("torch.cuda.OutOfMemoryError", "GPU OOM"),
]


def extract_traceback(lines: list) -> dict | None:
    """Last Python traceback in the log → {type, msg, frames, line_no}, else None.

    Generic: pinpoints ANY exception from the real traceback (no per-error rule)."""
    starts = [i for i, ln in enumerate(lines) if ln.strip() == _TB_START]
    if not starts:
        return None
    i = starts[-1]
    frames, j = [], i + 1
    while j < len(lines):
        m = _FILE_RE.match(lines[j])
        if m:
            code = lines[j + 1].strip() if j + 1 < len(lines) else ""
            frames.append({"file": m["file"], "line": int(m["line"]),
                           "func": m["func"].strip(), "code": code})
            j += 2
            continue
        s = lines[j]
        if s.strip() and not s.startswith((" ", "\t")):
            em = _EXC_RE.match(s.strip())
            if em:
                return {"type": em["type"], "msg": em["msg"].strip(),
                        "frames": frames, "line_no": j + 1}
            return {"type": "Error", "msg": s.strip(), "frames": frames, "line_no": j + 1}
        j += 1
    return {"type": "Error", "msg": "(traceback without a parsed exception line)",
            "frames": frames, "line_no": i + 1}


def scan_fatal(lines: list) -> str | None:
    """Non-traceback fatal signal (OOM/segfault) for crashes that left no Python TB."""
    for ln in reversed(lines[-400:]):
        for needle, label in _FATAL_SIGNALS:
            if needle in ln:
                return label
    return None


# --- HEALTH: derive structural checks from the I/O contract -----------------
def health_checks(phase: str, step_rec: dict, ob: Path) -> list[Check]:
    """Structural integrity + failure localization for one step.

    1. If the step FAILED → the real exception (traceback) or a fatal signal,
       tied to the log file:line (Class 1).
    2. The I/O contract: each declared INPUT (missing → WARN, upstream) and
       OUTPUT (missing/empty → FAIL, this step) with its file (Class 2 — the
       silent green-but-empty failures a traceback can't see).
    """
    name = step_rec.get("name")
    status = step_rec.get("status", "pending")
    checks: list[Check] = []

    if status == "failed":
        lines = sq.read_log(step_rec)
        lf = (step_rec or {}).get("log_file") or "(no log_file)"
        tb = extract_traceback(lines)
        if tb:
            frame = tb["frames"][-1] if tb["frames"] else None
            where = f" at {frame['file']}:{frame['line']} in {frame['func']}" if frame else ""
            checks.append(Check("exception", V(FAIL, f"{tb['type']}: {tb['msg']}{where}",
                                               {"traceback": tb}), f"{lf}:{tb['line_no']}"))
        else:
            sig = scan_fatal(lines)
            reason = (f"exit {step_rec.get('exit_code')} — {sig}" if sig
                      else f"exit {step_rec.get('exit_code')} (no traceback in log)")
            checks.append(Check("exit", V(FAIL, reason), lf))

    sp = SPECS.get((phase, name))
    if sp:
        for inp in sp.inputs:
            st = path_state(ob, inp)
            if not st["exists"]:
                checks.append(Check(f"in {inp}", V(WARN, "input absent — upstream not done?"), inp))
            elif st.get("rows") == 0:
                checks.append(Check(f"in {inp}", V(WARN, "input present but empty (0 rows)",
                                                   {"rows": 0}), st["paths"][0]))
            else:
                r = st.get("rows")
                checks.append(Check(f"in {inp}", V(PASS, f"{r:,} rows" if r is not None else "present",
                                                   {"rows": r}), st["paths"][0]))
        for out in sp.outputs:
            st = path_state(ob, out)
            if not st["exists"]:
                checks.append(Check(f"out {out}", V(FAIL, "output not produced"), out))
            elif st.get("rows") == 0:
                checks.append(Check(f"out {out}", V(FAIL, "empty output (0 rows)", {"rows": 0}),
                                    st["paths"][0]))
            else:
                r = st.get("rows")
                checks.append(Check(f"out {out}", V(PASS, f"{r:,} rows" if r is not None else "present",
                                                    {"rows": r}), st["paths"][0]))
        if sp.extra_health:
            try:
                checks.extend(sp.extra_health(ob) or [])
            except Exception as e:
                checks.append(Check("extra_health", V(UNKNOWN, f"health probe error: {e}")))

    if not checks:
        # No I/O contract declared yet and the step didn't crash → can't assert health.
        checks.append(Check("io_contract", V(UNKNOWN, "no I/O contract declared for this step")))
    return checks


def inspect_hints(phase: str, step_rec: dict, ob: Path, checks: list[Check]) -> list[str]:
    """Concrete files/commands to open, for the diagnose card footer."""
    hints: list[str] = []
    for c in checks:
        if c.v.outcome in (FAIL, WARN) and c.path:
            hints.append(c.path)
    sp = SPECS.get((phase, step_rec.get("name")))
    if sp:
        hints.extend(sp.inspect)
    # de-dup, preserve order
    seen, out = set(), []
    for h in hints:
        if h not in seen:
            seen.add(h); out.append(h)
    return out


# --- QUALITY: reuse step_quality + the per-step sample preview ---------------
def quality_check(phase: str, step_rec: dict, ob: Path) -> Check:
    """One graded Check from the step_quality probe (the existing quality engine)."""
    v = sq.evaluate(phase, step_rec, ob)
    sp = SPECS.get((phase, step_rec.get("name")))
    primary = sp.outputs[0] if (sp and sp.outputs) else None
    return Check("quality", v, primary)


def sample_rows(phase: str, step_rec: dict, ob: Path, n: int) -> list[str]:
    sp = SPECS.get((phase, step_rec.get("name")))
    if not sp or not sp.sample:
        return []
    try:
        return sp.sample(ob, n) or []
    except Exception:
        return []


# =============================================================================
# Per-phase specs.  v1: extract (the reference). validate/graphmert/curriculum
# get the generic failure-localizer + quality reuse now; their I/O contracts +
# samples come next (plan §8 step 4).
# =============================================================================
def _sample_kg_final(ob: Path, n: int) -> list[str]:
    p = ob / "graphrag" / "output" / "kg_final.csv"
    if not p.exists():
        return []
    out = []
    with p.open(newline="", errors="replace") as f:
        r = csv.reader(f)
        next(r, None)  # header
        for i, row in enumerate(r):
            if i >= n:
                break
            out.append(" | ".join((row + ["", "", ""])[:3]))
    return out


spec("extract", "extract_triples",
     inputs=["graphrag/input"],
     outputs=["graphrag/output/*entities*.parquet", "graphrag/output/*relationships*.parquet"],
     inspect=["replay one chunk: ./scripts/diagnose_llm_extraction.sh"])

spec("extract", "finalize_seed_kg",
     inputs=["graphrag/output/*relationships*.parquet"],
     outputs=["graphrag/output/kg_final.csv", "graphrag/output/kg_final.parquet"],
     sample=_sample_kg_final)


def _sample_csv(rel: str, ncols: int = 3):
    """Sample factory: first N data rows of a CSV as 'c0 | c1 | c2'."""
    def _s(ob: Path, n: int) -> list[str]:
        p = ob / rel
        if not p.exists():
            return []
        out = []
        with p.open(newline="", errors="replace") as f:
            r = csv.reader(f)
            next(r, None)
            for i, row in enumerate(r):
                if i >= n:
                    break
                out.append(" | ".join((row + [""] * ncols)[:ncols]))
        return out
    return _s


# --- validate ---------------------------------------------------------------
spec("validate", "seed_kg_consensus",
     inputs=["graphrag/output/kg_final.csv"],
     outputs=["graphrag/output/kg_final_validated.csv"],
     sample=_sample_csv("graphrag/output/kg_final_validated.csv"))


# --- graphmert --------------------------------------------------------------
spec("graphmert", "tokenize",
     outputs=["graphmert/tokenized_inputs", "graphmert/stable_tokenizer"])

spec("graphmert", "preprocess",
     inputs=["graphrag/output/kg_final.csv"],
     outputs=["graphmert/dataset/preprocessed_train", "graphmert/dataset/preprocessed_eval",
              "graphmert/head_positions",
              "graphmert/llm_relations/relations_cleaned_train",
              "graphmert/llm_relations/relations_cleaned_eval"])

spec("graphmert", "train_mnm",
     inputs=["graphmert/dataset/preprocessed_train"],
     outputs=["graphmert/checkpoints"])

spec("graphmert", "validate_predictions",
     outputs=["graphmert/final_kg/validated_triples.csv"],
     sample=_sample_csv("graphmert/final_kg/validated_triples.csv"))

spec("graphmert", "expand_kg",
     inputs=["graphmert/final_kg/validated_triples.csv"],
     outputs=["graphmert/final_kg/final_relationships.parquet"])
# predict_tails / predict_tails_gm: generic localizer + quality probe (outputs vary).


# --- curriculum -------------------------------------------------------------
def _sample_qa(ob: Path, n: int) -> list[str]:
    p = ob / "curriculum_verified" / "curriculum_verified.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data[:n]:
        if isinstance(item, dict):
            q = (item.get("question") or item.get("Question")
                 or item.get("mcq") or item.get("prompt") or "")
        else:
            q = str(item)
        q = " ".join(str(q).split())
        out.append((q[:100] + "…") if len(q) > 100 else q)
    return out


spec("curriculum", "path_traversal",
     outputs=["curriculum/kg_manifest.json"])

spec("curriculum", "generate_qa_pair",
     inputs=["curriculum/kg_manifest.json"],
     outputs=["curriculum/curriculum.jsonl", "curriculum/curriculum_stats.json"])

spec("curriculum", "assemble_curriculum",
     inputs=["curriculum/curriculum.jsonl"],
     outputs=["curriculum_verified/curriculum_verified.json"],
     sample=_sample_qa)
# validate_qa_pair / generate_qa_item / validate_qa_item: quality probes (yields
# from curriculum_stats.json) + generic localizer; they mutate the shared jsonl.


# --- sft --------------------------------------------------------------------
spec("sft", "prepare_data",
     inputs=["curriculum_verified/curriculum_verified.json"],
     outputs=["sft_dataset"])

spec("sft", "train_lora",
     inputs=["sft_dataset"],
     outputs=["sft_checkpoints/checkpoint-*"])

spec("sft", "merge_lora",
     inputs=["sft_checkpoints/checkpoint-*"],
     outputs=["sft_checkpoints/checkpoint-*/merged_final_model"])
# eval_sft: no-op step (operator runs eval_models.py separately) → generic.


# --- rl ---------------------------------------------------------------------
spec("rl", "train_grpo",
     inputs=["sft_checkpoints/checkpoint-*/merged_final_model"],
     outputs=["rl_checkpoints/checkpoint-*"])

spec("rl", "merge_rl",
     inputs=["rl_checkpoints/checkpoint-*"],
     outputs=["rl_checkpoints/checkpoint-*/merged_final_model"])
# prepare_rl_dataset / eval_rl: generic localizer (data_prep output path varies; eval is no-op).
