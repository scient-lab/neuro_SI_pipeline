#!/usr/bin/env python3
"""checks_view.py — the standardized run-inspection renderer (both lenses).

  --lens health   → diagnose: status + exception(file:line) + I/O-contract state
                    + "inspect these" footer. Pinpoints the failing file/exception.
  --lens quality  → analysis: graded probe verdict + optional --sample preview.

One record (step_quality.V), one renderer, two check-sets — see
docs/DIAGNOSE_ANALYSIS_STANDARDIZATION_PLAN_2026-06-29.md. Exit: 0 clean,
1 if any FAIL, 2 if WARN-only (composable with CI), matching diagnose/analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

import checks as K  # noqa: E402
from checks import PASS, WARN, FAIL, SKIP, UNKNOWN  # noqa: E402

GLY = {PASS: "✓", WARN: "⚠", FAIL: "✗", SKIP: "–", UNKNOWN: "?", None: "·"}
_PHASE_ORDER = ["extract", "validate", "graphmert", "curriculum", "sft", "rl"]


def _phase_key(ph):
    return (_PHASE_ORDER.index(ph) if ph in _PHASE_ORDER else len(_PHASE_ORDER), ph)


def _verdict_bracket(status, worst):
    if status in ("pending", "running"):
        return status
    if worst == FAIL:
        return "FAIL"
    if worst == WARN:
        return "WARN"
    if worst == UNKNOWN:
        return "?"
    if status == "skipped":
        return "skip"
    return "OK"


# --- health lens ------------------------------------------------------------
def render_health_step(phase, srec, ob, out):
    status = srec.get("status", "pending")
    name = srec.get("name")
    if status in ("pending", "running"):
        out.append(f"  {phase}.{name:<28} [{status}]")
        return None
    checks = K.health_checks(phase, srec, ob)
    worst = K.worst(c.v.outcome for c in checks)
    detail = ""
    if status == "failed":
        detail = f"  (failed, exit {srec.get('exit_code')})"
    out.append(f"  {phase}.{name:<28} [{_verdict_bracket(status, worst)}]{detail}")
    for c in checks:
        if c.name in ("exception", "exit"):
            out.append(f"    {GLY[c.v.outcome]} {c.name}  {c.path or ''}")
            out.append(f"        {c.v.reason}")
            tb = c.v.metrics.get("traceback") if c.v.metrics else None
            if tb and tb.get("frames"):
                fr = tb["frames"][-1]
                if fr.get("code"):
                    out.append(f"          {fr['code']}")
        else:
            loc = f"  [{c.path}]" if c.path and c.v.outcome in (FAIL, WARN) else ""
            out.append(f"    {GLY[c.v.outcome]} {c.name:<42} {c.v.reason}{loc}")
    if worst in (FAIL, WARN):
        hints = K.inspect_hints(phase, srec, ob, checks)
        if hints:
            out.append("    → inspect  " + "\n               ".join(hints))
    return worst


# --- quality lens -----------------------------------------------------------
def render_quality_step(phase, srec, ob, out, sample_n):
    status = srec.get("status", "pending")
    name = srec.get("name")
    c = K.quality_check(phase, srec, ob)
    o = c.v.outcome
    if o is None:
        out.append(f"  {phase}.{name:<28} [{status}]")
        return None
    out.append(f"  {phase}.{name:<28} [{_verdict_bracket(status, o)}]")
    metr = f"   {c.v.metrics}" if c.v.metrics else ""
    out.append(f"    {GLY[o]} {c.v.reason}{metr}")
    if sample_n:
        rows = K.sample_rows(phase, srec, ob, sample_n)
        if rows:
            prim = (c.path or "output")
            out.append(f"    sample  {prim}  ({len(rows)}):")
            for r in rows:
                out.append(f"      {r}")
    return o


def _exit_for(worst):
    return 1 if worst == FAIL else (2 if worst == WARN else 0)


def _find_run_dir(base: Path, run_prefix):
    """Resolve the run directory under <base> (mirrors analysis_*.py / config_view):
    base itself if it holds run_manifest.json, else a --run prefix match, else newest."""
    if (base / "run_manifest.json").is_file():
        return base
    if not base.is_dir():
        return base
    if run_prefix:
        m = sorted((p for p in base.glob(f"{run_prefix}*")
                    if p.is_dir() and (p / "run_manifest.json").is_file()), key=lambda p: p.name)
        if m:
            return m[-1]
    cands = sorted((p for p in base.iterdir()
                    if p.is_dir() and (p / "run_manifest.json").is_file()),
                   key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", choices=["health", "quality"], required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--output-base", default=None)
    ap.add_argument("--run", default=None, help="RUN_ID prefix; default = $OUTPUT_BASE or newest")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--step", default=None)
    ap.add_argument("--sample", nargs="?", type=int, const=5, default=0,
                    help="quality lens: preview N rows of each step's output (default 5)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.manifest:
        manifest_path = Path(a.manifest)
        ob = Path(a.output_base or os.environ.get("OUTPUT_BASE") or manifest_path.parent)
    else:
        base = Path(a.output_base or os.environ.get("OUTPUT_BASE") or (K.REPO_ROOT / "outputs"))
        ob = _find_run_dir(base, a.run)
        manifest_path = ob / "run_manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    doc = json.loads(manifest_path.read_text())
    run = doc.get("run", {})
    # Align config scale to this run (step_quality.exp() / phase_param()).
    if run.get("profile"):
        os.environ.setdefault("SI_PROFILE", run["profile"])
    if run.get("domain"):
        os.environ.setdefault("SI_DOMAIN", run["domain"])

    steps = []
    for p in run.get("phases", []):
        if a.phase and p.get("name") != a.phase:
            continue
        for s in p.get("steps", []):
            if a.step and s.get("name") != a.step:
                continue
            steps.append((p["name"], s))

    if a.json:
        return _emit_json(run, steps, ob, a)

    lens = a.lens.upper()
    print(f"{lens} — run {run.get('run_id')}   ({run.get('profile')}/{run.get('domain')})\n")
    verdicts = []
    last_phase = None
    out: list[str] = []
    for phase, srec in sorted(steps, key=lambda t: (_phase_key(t[0]), t[1].get("name"))):
        if phase != last_phase:
            if out:
                out.append("")
            last_phase = phase
        if a.lens == "health":
            v = render_health_step(phase, srec, ob, out)
        else:
            v = render_quality_step(phase, srec, ob, out, a.sample)
        if v is not None:
            verdicts.append(v)
    print("\n".join(out))

    worst = K.worst(verdicts)
    bad = [v for v in verdicts if v in (FAIL, WARN)]
    print(f"\nVERDICT: {sum(v == FAIL for v in verdicts)} fail, "
          f"{sum(v == WARN for v in verdicts)} warn"
          + ("" if not bad else f" → exit {_exit_for(worst)}"))
    return _exit_for(worst)


def _emit_json(run, steps, ob, a):
    out = {"lens": a.lens, "run_id": run.get("run_id"), "phases": []}
    by_phase: dict = {}
    verdicts = []
    for phase, srec in steps:
        if a.lens == "health":
            checks = K.health_checks(phase, srec, ob)
            worst = K.worst(c.v.outcome for c in checks)
            entry = {"step": srec.get("name"), "status": srec.get("status"),
                     "verdict": worst,
                     "checks": [{"name": c.name, "outcome": c.v.outcome,
                                 "reason": c.v.reason, "path": c.path,
                                 "metrics": c.v.metrics} for c in checks],
                     "inspect": K.inspect_hints(phase, srec, ob, checks)}
        else:
            c = K.quality_check(phase, srec, ob)
            worst = c.v.outcome
            entry = {"step": srec.get("name"), "status": srec.get("status"),
                     "verdict": worst,
                     "check": {"name": c.name, "outcome": c.v.outcome,
                               "reason": c.v.reason, "path": c.path, "metrics": c.v.metrics}}
            if a.sample:
                entry["sample"] = K.sample_rows(phase, srec, ob, a.sample)
        if worst is not None:
            verdicts.append(worst)
        by_phase.setdefault(phase, []).append(entry)
    for ph in sorted(by_phase, key=_phase_key):
        out["phases"].append({"phase": ph, "steps": by_phase[ph]})
    worst = K.worst(verdicts)
    out["verdict"] = worst
    out["exit"] = _exit_for(worst)
    print(json.dumps(out, indent=2))
    return _exit_for(worst)


if __name__ == "__main__":
    sys.exit(main())
