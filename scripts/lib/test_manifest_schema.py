#!/usr/bin/env python3
"""Drift guard: assert manifest.py's output matches docs/run_manifest.schema.json.

manifest.py is intentionally stdlib-only and does NOT validate against the schema
at runtime (that would add a dep on the hot, lock-guarded write path). So the
schema is a hand-maintained contract that can silently drift from the code. This
test closes that gap WITHOUT touching manifest.py: it drives manifest.py through a
realistic lifecycle — init, a COMPLETED run, and a FAILED run — and validates each
resulting manifest against the schema. The failed run is essential: it populates
phase/step `error`, the top-level `failure` summary, and `step_error`, which a
happy-path manifest never exercises.

If a manifest.py change adds/removes/retypes a field, this fails until the schema
(or the change) is fixed.

Run it directly — it re-execs under a venv that has `jsonschema` if the current
interpreter lacks it:

    python3 scripts/lib/test_manifest_schema.py
    # or explicitly:  ./.venvs/graphrag/bin/python scripts/lib/test_manifest_schema.py

Exit 0 = schema matches; 1 = drift; (skips with 0 only if no venv has jsonschema).
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MANIFEST_PY = os.path.join(HERE, "manifest.py")
SCHEMA_PATH = os.path.join(REPO, "docs", "run_manifest.schema.json")
PHASES_DIR = os.path.join(REPO, "scripts", "phases")


def _ensure_jsonschema():
    """Import jsonschema, re-execing under a project venv that has it if needed."""
    try:
        import jsonschema  # noqa: F401
        return True
    except ImportError:
        pass
    if os.environ.get("_MANIFEST_TEST_REEXEC") == "1":
        print("SKIP: jsonschema not available in any project venv "
              "(install it, or run under one that has it).")
        return False
    for venv in ("graphrag", "data_prep", "si_curriculum", "graphmert"):
        py = os.path.join(REPO, ".venvs", venv, "bin", "python")
        if os.path.exists(py):
            os.environ["_MANIFEST_TEST_REEXEC"] = "1"
            os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])
    print("SKIP: jsonschema not installed and no .venvs/* python found "
          "(try: ./.venvs/graphrag/bin/python scripts/lib/test_manifest_schema.py)")
    return False


def _mf(path, *args):
    """Run a manifest.py subcommand (stdlib-only -> current interpreter is fine)."""
    cmd = [sys.executable, MANIFEST_PY, args[0], "--path", path, *args[1:]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"manifest.py {args[0]} failed (rc={r.returncode}):\n{r.stderr}")
    return r


def _validate(validator, inst, label, results):
    errs = sorted(validator.iter_errors(inst), key=lambda e: list(e.path))
    if errs:
        print(f"  FAIL  {label}: {len(errs)} schema error(s)")
        for e in errs[:15]:
            loc = "/".join(map(str, e.path)) or "<root>"
            print(f"        - {loc}: {e.message}")
        results.append(False)
    else:
        print(f"  OK    {label}")
        results.append(True)


def _phases_in_order():
    files = sorted(f[:-3] for f in os.listdir(PHASES_DIR) if f.endswith(".sh"))
    if not files:
        raise RuntimeError(f"no phase scripts under {PHASES_DIR}")
    return files


def _pick_target(manifest):
    """Return (phase, step_ok, step_fail) from the live catalog so the test
    adapts to phase/step renames instead of hardcoding names."""
    for p in manifest["run"]["phases"]:
        steps = [s["name"] for s in p["steps"]]
        if len(steps) >= 2:
            return p["name"], steps[0], steps[1]
    for p in manifest["run"]["phases"]:
        steps = [s["name"] for s in p["steps"]]
        if steps:
            return p["name"], steps[0], steps[0]
    raise RuntimeError("no phase with any steps in the catalog")


def run() -> int:
    from jsonschema import Draft202012Validator
    schema = json.load(open(SCHEMA_PATH))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    print("schema is well-formed (Draft 2020-12)")

    phases = _phases_in_order()
    order = ",".join(phases)
    results: list = []

    with tempfile.TemporaryDirectory() as tmp:
        # A log file with content so error.log_tail is exercised (non-empty).
        log = os.path.join(tmp, "step.log")
        with open(log, "w") as fh:
            fh.write("starting step\nworking...\nTraceback (most recent call last):\nBoom\n")

        # --- 1. pristine init -------------------------------------------------
        a = os.path.join(tmp, "a.json")
        _mf(a, "init", "--phases-dir", PHASES_DIR, "--phase-order", order,
            "--selected", order, "--run-id", "20260101-000000-test-deadbee",
            "--domain", "space", "--profile", "smoke", "--platform", "runpod",
            "--git-sha", "deadbee", "--git-branch", "orchestration",
            "--step-filter", "all", "--pid", "1234", "--pgid", "1234",
            "--corpus-path", "corpus/space/smoke", "--runpod-pod-id", "pod-xyz")
        init_doc = json.load(open(a))
        _validate(validator, init_doc, "init (pristine, all-pending)", results)

        phase, s_ok, s_fail = _pick_target(init_doc)

        # --- 2. COMPLETED run (outcome present, no failure block) -------------
        b = os.path.join(tmp, "b.json")
        _mf(b, "init", "--phases-dir", PHASES_DIR, "--phase-order", order,
            "--selected", phase, "--run-id", "20260101-000001-test-deadbee",
            "--domain", "space", "--profile", "smoke", "--platform", "runpod")
        _mf(b, "start-phase", "--phase", phase, "--log-file", "logs/x/p.log")
        _mf(b, "start-step", "--phase", phase, "--step", s_ok,
            "--log-file", "logs/x/s.log", "--cw-stream", "cw-stream-123")
        _mf(b, "end-step", "--phase", phase, "--step", s_ok, "--exit-code", "0")
        _mf(b, "set-outcome", "--phase", phase, "--step", s_ok,
            "--outcome", "pass", "--reason", "128 triples / 12 docs = 10.7/doc")
        _mf(b, "end-phase", "--phase", phase, "--exit-code", "0")
        _mf(b, "finalize", "--status", "completed")
        _validate(validator, json.load(open(b)), "completed lifecycle (outcome set)", results)

        # --- 3. FAILED run (exercises error + failure + step_error) -----------
        c = os.path.join(tmp, "c.json")
        _mf(c, "init", "--phases-dir", PHASES_DIR, "--phase-order", order,
            "--selected", phase, "--run-id", "20260101-000002-test-deadbee",
            "--domain", "space", "--profile", "smoke", "--platform", "runpod")
        _mf(c, "start-phase", "--phase", phase, "--log-file", "logs/x/p.log")
        _mf(c, "start-step", "--phase", phase, "--step", s_fail, "--log-file", log)
        _mf(c, "end-step", "--phase", phase, "--step", s_fail, "--exit-code", "1",
            "--error-message", "synthetic boom", "--log-file", log)
        _mf(c, "end-phase", "--phase", phase, "--exit-code", "1", "--log-file", log)
        _mf(c, "finalize", "--status", "failed")
        failed_doc = json.load(open(c))
        _validate(validator, failed_doc, "failed lifecycle (error + failure)", results)

        # Structural asserts: schema alone can't prove the failure BRANCHES ran,
        # since failure/error/step_error are optional. Prove we actually populated
        # them, else the failed-path validation was vacuous.
        fail = failed_doc["run"].get("failure")
        struct_ok = (
            isinstance(fail, dict)
            and fail.get("phase") == phase
            and fail.get("step") == s_fail
            and isinstance(fail.get("error", {}).get("log_tail"), list)
            and fail["error"]["log_tail"]                       # non-empty tail
            and isinstance(fail.get("step_error"), dict)
        )
        if struct_ok:
            print("  OK    failure summary populated (phase/step/error.log_tail/step_error)")
        else:
            print(f"  FAIL  failure summary not populated as expected: {fail!r}")
        results.append(struct_ok)

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED — schema matches manifest.py output")
        return 0
    print(f"DRIFT DETECTED — {results.count(False)}/{len(results)} checks failed. "
          "Reconcile docs/run_manifest.schema.json with scripts/lib/manifest.py.")
    return 1


def test_manifest_matches_schema():
    """pytest entry point."""
    assert _ensure_jsonschema(), "jsonschema unavailable"
    assert run() == 0


if __name__ == "__main__":
    if not _ensure_jsonschema():
        sys.exit(0)
    sys.exit(run())
