#!/usr/bin/env python3
"""run_phase.py — the generic, catalog-driven phase runner (Phase B).

Drop-in for a phase's hardcoded `STEPS=(...)` loop + `step_<name>()` bash fns.
For the given --phase it:
  1. reads the ordered (phase, step) list from configs/pipeline_catalog.yaml
     (the single authored source of STRUCTURE),
  2. looks up each step's execution binding in configs/pipeline_execution.yaml
     (id -> {venv, entrypoint, kind}),
  3. applies the step filter (PIPELINE_STEP_FILTER, resolving renamed ids via the
     catalog id_aliases) and --resume skip-completed,
  4. execs  <repo>/.venvs/<venv>/bin/python <entrypoint>  with the fixed env
     contract, or RECORDS a kind:noop step without executing anything,
  5. updates run_manifest.json BY ID via the stdlib scripts/lib/manifest.py
     (atomic / lock-guarded), writing a per-step log — i.e. it mirrors
     common.sh::run_step at the phase level.

It reads YAML (needs pyyaml) so it is invoked under a pyyaml-capable interpreter
(pipeline.sh uses `uv run --with pyyaml`); it does NOT need the training deps —
each entrypoint runs in ITS OWN venv (the binding's venv) as a subprocess.

Env contract (exported by pipeline.sh; passed through to each entrypoint, plus
per-step SI_PHASE/SI_STEP so pipeline_config's ledger attributes correctly):
  REPO_ROOT, OUTPUT_BASE, RUN_ID, SI_DOMAIN, SI_PROFILE, SI_PLATFORM,
  PIPELINE_MANIFEST, PIPELINE_LOG_DIR, PIPELINE_STEP_FILTER, PIPELINE_RESUME

Modes:
  --list-migrated   print the phase ids that HAVE an execution binding (one per
                    line) and exit — lets pipeline.sh route only migrated phases
                    through the runner (every other phase keeps its bash path).
  --dry-run         resolve + PRINT the plan (order, venv, entrypoint, filter/
                    resume decisions, missing-file warnings) WITHOUT touching the
                    manifest or executing anything — proves the plumbing, no GPU.

Deferred parity (vs run_step; the monitor's periodic pass backfills these):
  * inline step_quality --write (OUTCOME column) and CloudWatch _cw_ship are not
    yet invoked here — see the plan's Phase B notes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _manifest_py() -> str:
    return os.path.join(_env("REPO_ROOT"), "scripts", "lib", "manifest.py")


def _manifest(*args: str) -> None:
    """Mutate run_manifest.json via the stdlib manifest.py (atomic/locked).
    Best-effort, exactly like common.sh::_manifest — a manifest write must never
    take the phase down."""
    path = _env("PIPELINE_MANIFEST")
    if not path:
        return
    try:
        subprocess.run([sys.executable, _manifest_py(), *args], check=False)
    except Exception as e:  # noqa: BLE001
        _log(f"[run_phase] manifest update failed ({args[0] if args else '?'}): {e}")


def _step_status(phase: str, step: str) -> str:
    """Manifest status of a step ('absent' on any problem) — for --resume."""
    path = _env("PIPELINE_MANIFEST")
    if not path:
        return "absent"
    try:
        out = subprocess.run(
            [sys.executable, _manifest_py(), "status",
             "--path", path, "--phase", phase, "--step", step],
            capture_output=True, text=True, check=False)
        return (out.stdout or "").strip() or "absent"
    except Exception:  # noqa: BLE001
        return "absent"


def _resolve_filter(catalog: dict):
    """(wanted_set | None, aliases). None => run every step (filter 'all')."""
    raw = (_env("PIPELINE_STEP_FILTER", "all").strip() or "all")
    aliases = catalog.get("id_aliases") or {}
    if raw == "all":
        return None, aliases
    wanted = set()
    for w in raw.split(","):
        w = w.strip()
        if not w:
            continue
        wr = aliases.get(w, w)
        if wr != w:
            _log(f"[run_phase] --step '{w}' was renamed to '{wr}' — resolving "
                 f"it (update your --step to '{wr}')")
        wanted.add(wr)
    return wanted, aliases


def _resolve_output(decl: str, out_base: str):
    """$OUTPUT_BASE-joined path for a RESOLVABLE declared output — non-empty, no
    glob `*`, no spaces — else None (run-varying / non-path: entrypoint resolves it).
    Injected as $STEP_OUTPUT so entrypoints read the path instead of hardcoding it."""
    decl = (decl or "").strip()
    if not decl or "*" in decl or " " in decl:
        return None
    return os.path.join(out_base, decl.rstrip("/"))


def _exec_tee(cmd: list[str], cwd: str, env: dict, logfile: Path) -> int:
    """Run cmd, teeing combined stdout+stderr to console AND the per-step log."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        return proc.wait()


def cmd_list_migrated(execution: dict) -> int:
    """Print phases where EVERY step is placeholder:false (fully migrated to runner
    entrypoints / handled noops). A phase with ANY placeholder:true step stays on
    the bash path — so a partially-migrated phase is never routed to the runner."""
    for pid, steps in (execution.get("phases") or {}).items():
        steps = steps or {}
        if steps and all(not (s or {}).get("placeholder", True) for s in steps.values()):
            print(pid)
    return 0


def run_phase(phase: str, catalog: dict, execution: dict, dry_run: bool) -> int:
    repo = _env("REPO_ROOT")
    if not repo:
        _log("[run_phase] REPO_ROOT not set (env contract)")
        return 2

    cat_phase = (catalog.get("phases") or {}).get(phase) or {}
    cat_steps = cat_phase.get("steps") or {}
    steps = list(cat_steps.keys())                           # catalog order = single source
    if not steps:
        _log(f"[run_phase] phase '{phase}' has no steps in the catalog")
        return 2

    bindings = ((execution.get("phases") or {}).get(phase)) or {}
    if not bindings:
        _log(f"[run_phase] phase '{phase}' has no execution binding "
             f"(configs/pipeline_execution.yaml) — not a migrated phase")
        return 2

    wanted, _aliases = _resolve_filter(catalog)
    manifest_path = _env("PIPELINE_MANIFEST")
    resume = _env("PIPELINE_RESUME", "0") == "1"
    out_base = _env("OUTPUT_BASE", os.path.join(repo, "outputs"))
    log_dir = _env("PIPELINE_LOG_DIR") or os.path.join(out_base, "logs", "adhoc")

    for step in steps:
        rellog = os.path.relpath(str(Path(log_dir) / phase / f"{step}.log"), repo)

        # 1. step filter
        if wanted is not None and step not in wanted:
            _log(f"{phase} :: {step} (skipped — not in step filter)")
            if not dry_run:
                _manifest("skip-step", "--path", manifest_path, "--phase", phase, "--step", step)
            continue

        # 2. --resume: skip already-completed
        if resume and _step_status(phase, step) == "completed":
            _log(f"{phase} :: {step} (skipped — already completed, --resume)")
            continue

        binding = bindings.get(step)
        if binding is None:
            _log(f"[run_phase] no execution binding for {phase}.{step} in "
                 f"pipeline_execution.yaml (catalog has it; add a binding)")
            return 2
        # kind lives ONCE in the catalog (single source) — read it there, not the binding.
        kind = (cat_steps.get(step) or {}).get("kind", "")

        # 3. noop — record, do not execute (plan §7.2)
        if kind == "noop":
            _log(f"{phase} :: {step} (noop — recorded, not executed)")
            if not dry_run:
                _manifest("start-step", "--path", manifest_path, "--phase", phase,
                          "--step", step, "--log-file", rellog)
                _manifest("end-step", "--path", manifest_path, "--phase", phase,
                          "--step", step, "--exit-code", "0", "--log-file", rellog)
            continue

        if binding.get("placeholder", True):
            _log(f"[run_phase] {phase}.{step}: placeholder:true (no runner entrypoint yet) — "
                 f"this phase should not have been routed to the runner. Fix pipeline_execution.yaml.")
            return 2
        venv = binding.get("venv", "")
        entrypoint = binding.get("entrypoint", "")
        if not venv or not entrypoint:
            _log(f"[run_phase] {phase}.{step}: binding needs venv + entrypoint "
                 f"(kind={kind or 'unset'})")
            return 2
        venv_py = os.path.join(repo, ".venvs", venv, "bin", "python")
        ep_path = os.path.join(repo, entrypoint)

        # 4. dry-run: print the resolved plan, execute nothing, touch no manifest
        if dry_run:
            so = _resolve_output(binding.get("output"), out_base)
            inj = f"  STEP_OUTPUT={so}" if so else ""
            _log(f"{phase} :: {step} [DRY-RUN] venv={venv} entrypoint={entrypoint} kind={kind}{inj}")
            missing = []
            if not os.path.exists(venv_py):
                missing.append(f"venv python {venv_py}")
            if not os.path.exists(ep_path):
                missing.append(f"entrypoint {ep_path}")
            if missing:
                _log("     WOULD FAIL — missing: " + "; ".join(missing))
            continue

        # 5. real exec under the binding's venv, with the env contract
        if not os.path.exists(venv_py):
            _log(f"[run_phase] {phase}.{step}: venv python not found: {venv_py}")
            return 2
        if not os.path.exists(ep_path):
            _log(f"[run_phase] {phase}.{step}: entrypoint not found: {ep_path}")
            return 2

        child = os.environ.copy()
        child["SI_PHASE"] = phase
        child["SI_STEP"] = step
        child["PYTHONPATH"] = repo + (os.pathsep + child["PYTHONPATH"] if child.get("PYTHONPATH") else "")
        # Inject the resolvable declared output so the entrypoint reads $STEP_OUTPUT
        # instead of hardcoding the path (run-varying/non-path outputs stay unset).
        step_output = _resolve_output(binding.get("output"), out_base)
        if step_output:
            child["STEP_OUTPUT"] = step_output

        _manifest("start-step", "--path", manifest_path, "--phase", phase,
                  "--step", step, "--log-file", rellog)
        _log(f"{phase} :: {step} (venv={venv}, entrypoint={entrypoint})")
        rc = _exec_tee([venv_py, ep_path], cwd=repo, env=child,
                       logfile=Path(log_dir) / phase / f"{step}.log")
        _manifest("end-step", "--path", manifest_path, "--phase", phase,
                  "--step", step, "--exit-code", str(rc), "--log-file", rellog)
        if rc != 0:
            _log(f"[run_phase] {phase}.{step} failed (exit {rc}) — aborting phase.")
            return rc

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="catalog-driven phase runner (Phase B)")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--catalog-yaml", default=None)
    ap.add_argument("--execution-yaml", default=None)
    ap.add_argument("--list-migrated", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    repo = _env("REPO_ROOT") or str(Path(__file__).resolve().parents[2])
    cat_path = a.catalog_yaml or os.path.join(repo, "configs", "pipeline_catalog.yaml")
    exec_path = a.execution_yaml or os.path.join(repo, "configs", "pipeline_execution.yaml")

    execution = _load_yaml(exec_path) if os.path.exists(exec_path) else {}

    if a.list_migrated:
        return cmd_list_migrated(execution)

    if not a.phase:
        _log("[run_phase] --phase is required (or use --list-migrated)")
        return 2
    catalog = _load_yaml(cat_path)
    return run_phase(a.phase, catalog, execution, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
