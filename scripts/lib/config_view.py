#!/usr/bin/env python3
"""Effective-config view for a run — what config each phase/step ACTUALLY used.

Reads the per-step config ledger written at runtime by pipeline_config
(`<run>/config/<phase>.<step>.yaml`) and prints it nested:

    phase -> step -> section -> (key: value  (source-layer))

Sections: models / params / prompts. With no section flag, all are shown;
`--models` (or `--params` / `--prompts`) narrows it. This reflects what really
ran — post-merge + env overrides — not the input yaml. `source: fallback`
means no config layer set the key and the hard-coded default won (the
profile-key-name-trap tell). Run-dir resolution mirrors the analysis_*.py
modules (OUTPUT_BASE / --run prefix / newest).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

_PHASE_ORDER = ["extract", "validate", "graphmert", "curriculum", "sft", "rl"]
_SECTIONS = ["models", "params", "prompts"]


def _find_run_dir(output_base: Path, run_prefix: Optional[str]) -> Optional[Path]:
    if (output_base / "config").is_dir() or (output_base / "run_manifest.json").is_file():
        return output_base
    if not output_base.is_dir():
        return None
    if run_prefix:
        m = sorted((p for p in output_base.glob(f"{run_prefix}*") if p.is_dir()),
                   key=lambda p: p.name)
        if m:
            return m[-1]
    cands = sorted((p for p in output_base.iterdir() if p.is_dir() and (p / "config").is_dir()),
                   key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _load_ledgers(cfg_dir: Path):
    """[(phase, step, written_at, {section: {key: {value, source}}}), ...]."""
    rows = []
    for f in sorted(cfg_dir.glob("*.yaml")):
        try:
            d = yaml.safe_load(f.read_text()) or {}
        except Exception:
            continue
        meta = d.get("_meta", {}) or {}
        stem = f.stem
        phase = meta.get("phase") or stem.split(".")[0]
        step = meta.get("step") or (stem.split(".", 1)[1] if "." in stem else "")
        secs = {s: (d.get(s) or {}) for s in _SECTIONS}
        rows.append((phase, step, meta.get("written_at", ""), secs))
    return rows


def _phase_key(ph: str):
    return (_PHASE_ORDER.index(ph) if ph in _PHASE_ORDER else len(_PHASE_ORDER), ph)


def main() -> int:
    ap = argparse.ArgumentParser(description="effective config used per phase/step (from the run's config ledger)")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--run", default=None, help="RUN_ID prefix; default = $OUTPUT_BASE or newest")
    ap.add_argument("--phase", default=None, help="limit to one phase")
    ap.add_argument("--step", default=None, help="limit to one step")
    ap.add_argument("--models", action="store_true", help="show only the models section")
    ap.add_argument("--params", action="store_true", help="show only the params section")
    ap.add_argument("--prompts", action="store_true", help="show only the prompts section")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")   # parity, no-op
    a, _ = ap.parse_known_args()

    sel = [s for s, on in (("models", a.models), ("params", a.params), ("prompts", a.prompts)) if on]
    if not sel:
        sel = list(_SECTIONS)

    repo_root = Path(a.repo_root).resolve()
    output_base = Path(os.environ.get("OUTPUT_BASE", repo_root / "outputs"))
    run_dir = _find_run_dir(output_base, a.run)

    if not run_dir or not (run_dir / "config").is_dir():
        note = ("no config records yet — the per-step config ledger is written at "
                "runtime under <run>/config/. Run the pipeline (or a phase) first.")
        if a.json:
            print(json.dumps({"run_dir": str(run_dir) if run_dir else None,
                              "sections": sel, "phases": [], "note": note}, indent=2))
        else:
            print(f"CONFIG: {note}")
        return 0

    rows = _load_ledgers(run_dir / "config")
    if a.phase:
        rows = [r for r in rows if r[0] == a.phase]
    if a.step:
        rows = [r for r in rows if r[1] == a.step]

    # group: phase -> step -> (written_at, secs)
    phases: dict[str, dict[str, tuple]] = {}
    for phase, step, ts, secs in rows:
        # keep the step only if it has at least one selected, non-empty section
        if any(secs.get(s) for s in sel):
            phases.setdefault(phase, {})[step] = (ts, secs)

    if a.json:
        out = {"run_dir": str(run_dir), "sections": sel, "phases": []}
        for ph in sorted(phases, key=_phase_key):
            steps = []
            for st in sorted(phases[ph], key=lambda s: (phases[ph][s][0], s)):
                _ts, secs = phases[ph][st]
                entry = {"step": st}
                for s in sel:
                    if secs.get(s):
                        entry[s] = {k: {"value": v.get("value"), "source": v.get("source")}
                                    for k, v in secs[s].items()}
                steps.append(entry)
            out["phases"].append({"phase": ph, "steps": steps})
        print(json.dumps(out, indent=2))
        return 0

    print(f"CONFIG — run {run_dir.name}   (sections: {', '.join(sel)})\n")
    if not phases:
        scope = f" for phase '{a.phase}'" if a.phase else ""
        print(f"  (no config recorded{scope} in the selected section(s))")
        return 0
    for ph in sorted(phases, key=_phase_key):
        print(ph)
        for st in sorted(phases[ph], key=lambda s: (phases[ph][s][0], s)):
            _ts, secs = phases[ph][st]
            print(f"  {st or '·'}")
            for s in sel:
                items = secs.get(s) or {}
                if not items:
                    continue
                print(f"    {s}:")
                kw = max((len(k) for k in items), default=0)
                vw = max((len(str(v.get('value'))) for v in items.values()), default=0)
                for k in sorted(items):
                    v = items[k]
                    print(f"      {k.ljust(kw)}  {str(v.get('value')).ljust(vw)}  ({v.get('source')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
