#!/usr/bin/env python3
"""run_manifest.json reader/writer for pipeline.sh.

Stdlib-only (json, argparse, datetime, fcntl, os, re, sys) so it runs under
ANY venv the phase scripts activate — no pyyaml/boto3 dependency. Every
mutation is atomic (write temp + os.replace) and serialized with an flock on
a sidecar lockfile, so a best-effort S3 sync that reads the file mid-run can
never observe a half-written document.

The manifest has two halves:

  meta — STATIC catalog, identical for every run. Lets an API consumer learn
         the universe without hardcoding it: the status enum, the timestamp
         format, and the canonical ordered list of phases — each with its
         ordered steps + human descriptions. Parsed straight from
         scripts/phases/<phase>.sh (PHASE_DESC / STEPS / STEP_DESCS).

  run  — PER-RUN state: which phases/steps were selected, their status, start
         & end timestamps (RFC3339, timezone-aware), exit codes, and per-step
         log-file paths (relative to REPO_ROOT). Pre-populated as "pending" at
         init so a consumer sees the full plan upfront; flipped to running /
         completed / failed / skipped as execution proceeds.

Timestamps are produced by datetime.now(timezone.utc).isoformat(timespec=
"seconds") => e.g. "2026-06-17T14:15:23+00:00" (RFC3339 with offset).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

SCHEMA_VERSION = "1.0"
STATUS_ENUM = ["pending", "running", "completed", "failed", "skipped"]
TIMESTAMP_FORMAT = "RFC3339 / ISO-8601 with timezone offset (e.g. 2026-06-17T14:15:23+00:00)"

# Rough share of total wall-clock per phase (neuroscience pipeline). curriculum
# is Gemini-rate-limited and rl/GRPO is heavy, so they carry most weight — an
# ETA computed BEFORE those phases start will therefore read optimistically.
# Unknown phases (other domains) fall back to an equal-ish weight.
PHASE_WEIGHTS = {"extract": 0.12, "validate": 0.0, "graphmert": 0.25,
                 "curriculum": 0.30, "sft": 0.13, "rl": 0.20}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _update_progress_eta(run: dict) -> None:
    """Recompute run['progress'] (0..1) and run['estimated_completion_at']
    (absolute RFC3339) from weighted phase/step completion, and store them on
    the manifest so API/UI consumers — not just logs.sh — get the ETA.

    We store the ABSOLUTE completion timestamp, not a 'remaining' duration: a
    duration goes stale by the second, an absolute time does not (readers do
    `remaining = est - now`). NOTE: this is a snapshot at the last transition,
    so a long single step won't refresh it until the step ends — live readers
    may recompute against `now` for finer granularity.
    """
    phases = run.get("phases", []) or []
    total_w = done_w = 0.0
    for p in phases:
        st = p.get("status", "pending")
        if st == "skipped":              # won't run -> drop from denominator
            continue
        w = PHASE_WEIGHTS.get(p.get("name", ""), 0.17)
        total_w += w
        if st == "completed":
            done_w += w
        elif st == "running":            # credit partial via step progress
            steps = p.get("steps", []) or []
            tot = len(steps)
            ok = sum(1 for s in steps if s.get("status") == "completed")
            done_w += w * ((ok / tot) if tot else 0.0)

    progress = (done_w / total_w) if total_w > 0 else 0.0
    run["progress"] = round(progress, 4)

    status = run.get("status")
    if status in ("completed", "failed"):
        run["estimated_completion_at"] = run.get("finished_at")
        return

    # Prefer resumed_at over started_at for the elapsed/ETA calculation.
    # On a restart, started_at retains the audit-trail original (for history),
    # but progress is fresh — anchoring ETA to started_at produces absurd
    # numbers like "ETA 372h left" when in fact only 12 min have actually run.
    # Falls back to started_at when resumed_at is absent (single-invocation runs).
    start = run.get("resumed_at") or run.get("started_at")
    try:
        ts = datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
    except (ValueError, AttributeError):
        ts = None
    if status == "running" and ts and progress > 0:
        elapsed = (datetime.now(timezone.utc) - ts).total_seconds()
        est_total = elapsed / progress
        run["estimated_completion_at"] = (ts + timedelta(seconds=est_total)).isoformat(timespec="seconds")
    else:
        run["estimated_completion_at"] = None


def _read_log_tail(path: str, n: int = 30) -> list:
    """Last n non-empty lines of a log file. Safe — returns [] on any error
    (missing path, permission, binary garbage, etc.). The log file paths in
    the manifest are repo-relative; resolve against $REPO_ROOT or pwd."""
    if not path:
        return []
    candidates = [path]
    repo_root = os.environ.get("REPO_ROOT")
    if repo_root:
        candidates.append(os.path.join(repo_root, path))
    candidates.append(os.path.join(os.getcwd(), path))
    for c in candidates:
        try:
            with open(c, "r", errors="replace") as f:
                lines = [ln.rstrip() for ln in f.read().splitlines() if ln.strip()]
                return lines[-n:]
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            continue
        except Exception:
            return []
    return []


def _capture_error(exit_code: int, log_file: str, explicit_message: str,
                   tail_lines: int) -> dict:
    """Build an error dict for a failed phase/step. Always includes message
    and exit_code; log_tail is best-effort."""
    msg = explicit_message or f"exit code {exit_code}"
    return {
        "message": msg,
        "exit_code": exit_code,
        "log_tail": _read_log_tail(log_file, tail_lines),
    }


# --------------------------------------------------------------------------
# Atomic, lock-guarded read/modify/write
# --------------------------------------------------------------------------
@contextmanager
def _locked(path: str):
    """Hold an flock on <path>.lock for the duration of the block."""
    import fcntl

    lock_path = path + ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _load(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def _save(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _mutate(path: str, fn) -> None:
    """Lock, load, apply fn(data) in place, refresh progress/ETA, save atomically."""
    with _locked(path):
        data = _load(path)
        fn(data)
        if isinstance(data.get("run"), dict):
            _update_progress_eta(data["run"])
        _save(path, data)


def _find(seq, name):
    for item in seq:
        if item.get("name") == name:
            return item
    return None


# --------------------------------------------------------------------------
# Catalog parsing — read PHASE_DESC / STEPS / STEP_DESCS from a phase script
# --------------------------------------------------------------------------
def parse_phase_file(path: str):
    """Return (description, [step_names], [step_descs]) for one phases/*.sh."""
    try:
        text = open(path).read()
    except FileNotFoundError:
        return "", [], []

    m = re.search(r'^PHASE_DESC="(.*)"\s*$', text, re.M)
    desc = m.group(1) if m else ""

    m = re.search(r"^STEPS=\((.*?)\)", text, re.M | re.S)
    steps = m.group(1).split() if m else []

    descs: list[str] = []
    m = re.search(r"^STEP_DESCS=\((.*?)^\)", text, re.M | re.S)
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            q = re.match(r'^"(.*)"$', line)
            descs.append(q.group(1) if q else line)
    return desc, steps, descs


def build_catalog(phases_dir: str, phase_order: list[str]) -> list[dict]:
    catalog = []
    for phase in phase_order:
        desc, steps, descs = parse_phase_file(os.path.join(phases_dir, f"{phase}.sh"))
        catalog.append(
            {
                "name": phase,
                "description": desc,
                "steps": [
                    {"name": s, "description": descs[i] if i < len(descs) else ""}
                    for i, s in enumerate(steps)
                ],
            }
        )
    return catalog


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------
def _fresh_phase(cat_phase: dict) -> dict:
    """A run-phase record in the initial 'pending' state, from a catalog entry."""
    return {
        "name": cat_phase["name"],
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "log_file": None,
        "steps": [
            {
                "name": s["name"],
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "log_file": None,
                "cw_log_stream": None,
                # Output-quality verdict (pass/warn/fail/skip/unknown) + reason,
                # written post-hoc by scripts/lib/step_quality.py. Orthogonal to
                # status: status = did it RUN, outcome = did it PRODUCE meaningful
                # output. Null until a probe runs.
                "outcome": None,
                "outcome_reason": None,
            }
            for s in cat_phase["steps"]
        ],
    }


def cmd_init(a) -> None:
    phase_order = [p for p in a.phase_order.split(",") if p]
    selected = [p for p in a.selected.split(",") if p]
    catalog = build_catalog(a.phases_dir, phase_order)
    cat_by_name = {c["name"]: c for c in catalog}

    meta = {
        "status_enum": STATUS_ENUM,
        "timestamp_format": TIMESTAMP_FORMAT,
        "phases": catalog,
    }

    # MERGE path: an existing manifest for the SAME run_id means this is a
    # phase-wise invocation joining a logical run already in progress. Preserve
    # the records of phases that already ran; (re)set only the phases selected
    # NOW to pending; keep the original started_at; union the selected list.
    # A DIFFERENT run_id (or no/corrupt file) starts fresh, overwriting.
    existing = None
    if os.path.exists(a.path):
        try:
            prev = _load(a.path)
            if prev.get("run", {}).get("run_id") == a.run_id:
                existing = prev
        except (ValueError, OSError):
            existing = None

    os.makedirs(os.path.dirname(os.path.abspath(a.path)), exist_ok=True)

    if existing is not None:
        run = existing["run"]
        prior = {p["name"]: p for p in run.get("phases", [])}
        union = set(run.get("selected_phases", [])) | set(selected)
        new_phases = []
        for name in phase_order:
            if name not in union:
                continue
            if name in selected:
                new_phases.append(_fresh_phase(cat_by_name[name]))  # (re)run now
            else:
                new_phases.append(prior.get(name, _fresh_phase(cat_by_name[name])))
        run["phases"] = new_phases
        run["selected_phases"] = [p for p in phase_order if p in union]
        run["status"] = "running"
        run["finished_at"] = None
        run["current_phase"] = None
        run["step_filter"] = a.step_filter
        # Mark when this restart began. started_at stays as the original
        # audit-trail timestamp; resumed_at drives elapsed + ETA calc.
        # Set on EVERY merge invocation, even if the same phase is re-selected,
        # so each new pipeline.sh wave resets its own clock.
        run["resumed_at"] = now_iso()
        # Refresh PID/PGID — the previous invocation is dead; this one owns
        # the run from here. scripts/kill_pipeline.sh uses these to kill the
        # whole tree (orchestrator + bash phase scripts + python vLLM workers).
        if a.pid:
            run["pid"] = int(a.pid)
        if a.pgid:
            run["pgid"] = int(a.pgid)
        if a.corpus_path:
            run["corpus_path"] = a.corpus_path
        # Refresh the pod id on resume — a re-launched run may land on a new pod.
        if a.runpod_pod_id:
            run["runpod_pod_id"] = a.runpod_pod_id
        # If a previous top-level failure was set by cmd_finalize (e.g.
        # yesterday's failed graphmert), clear it when the SAME phase is
        # being re-run now. Stale failure blocks confuse the operator into
        # thinking the current attempt failed.
        prior_failure = run.get("failure")
        if prior_failure and prior_failure.get("phase") in selected:
            del run["failure"]
        existing["schema_version"] = SCHEMA_VERSION
        existing["meta"] = meta
        _update_progress_eta(run)
        with _locked(a.path):
            _save(a.path, existing)
        return

    run_phases = [
        _fresh_phase(cat_by_name[p]) for p in phase_order if p in selected
    ]
    doc = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "run": {
            "run_id": a.run_id,
            "status": "running",
            "domain": a.domain,
            "profile": a.profile,
            "platform": a.platform,
            "git_sha": a.git_sha,
            "git_branch": a.git_branch,
            "step_filter": a.step_filter,
            "started_at": now_iso(),
            "finished_at": None,
            "current_phase": None,
            # pipeline.sh's PID + process-group ID, for kill_pipeline.sh.
            "pid": int(a.pid) if a.pid else None,
            "pgid": int(a.pgid) if a.pgid else None,
            "corpus_path": a.corpus_path or None,
            "runpod_pod_id": a.runpod_pod_id or None,
            "progress": 0.0,
            "estimated_completion_at": None,
            "selected_phases": [p for p in phase_order if p in selected],
            "phases": run_phases,
        },
    }
    _update_progress_eta(doc["run"])
    with _locked(a.path):
        _save(a.path, doc)


def cmd_start_phase(a) -> None:
    def fn(d):
        d["run"]["current_phase"] = a.phase
        p = _find(d["run"]["phases"], a.phase)
        if p:
            p["status"] = "running"
            p["started_at"] = now_iso()
            if a.log_file:
                p["log_file"] = a.log_file
        # Clear any stale top-level failure that referred to THIS phase.
        # cmd_init's merge path covers the explicit-restart case, but this
        # also covers an interactive `pipeline.sh --phase X` invocation that
        # bypasses init merging (e.g. retrying a single failed phase).
        prior_failure = d["run"].get("failure")
        if prior_failure and prior_failure.get("phase") == a.phase:
            del d["run"]["failure"]

    _mutate(a.path, fn)


def cmd_end_phase(a) -> None:
    def fn(d):
        p = _find(d["run"]["phases"], a.phase)
        if p:
            p["finished_at"] = now_iso()
            p["exit_code"] = a.exit_code
            p["status"] = "completed" if a.exit_code == 0 else "failed"
            if a.exit_code != 0:
                p["error"] = _capture_error(
                    a.exit_code, a.log_file, a.error_message, a.tail_lines)
            elif "error" in p:
                # Successful re-run after a previous failure on this phase —
                # drop stale error so the manifest reflects current truth.
                del p["error"]
        d["run"]["current_phase"] = None

    _mutate(a.path, fn)


def _step(d, phase, step):
    p = _find(d["run"]["phases"], phase)
    return _find(p["steps"], step) if p else None


def cmd_start_step(a) -> None:
    def fn(d):
        s = _step(d, a.phase, a.step)
        if s:
            s["status"] = "running"
            s["started_at"] = now_iso()
            if a.log_file:
                s["log_file"] = a.log_file
            if a.cw_stream:
                s["cw_log_stream"] = a.cw_stream

    _mutate(a.path, fn)


def cmd_end_step(a) -> None:
    def fn(d):
        s = _step(d, a.phase, a.step)
        if s:
            s["finished_at"] = now_iso()
            s["exit_code"] = a.exit_code
            s["status"] = "completed" if a.exit_code == 0 else "failed"
            if a.log_file:
                s["log_file"] = a.log_file
            if a.exit_code != 0:
                s["error"] = _capture_error(
                    a.exit_code, a.log_file, a.error_message, a.tail_lines)
            elif "error" in s:
                del s["error"]

    _mutate(a.path, fn)


def cmd_skip_step(a) -> None:
    def fn(d):
        s = _step(d, a.phase, a.step)
        if s:
            # Preserve historical `completed` and `failed` states across
            # narrow re-runs. When pipeline.sh re-runs a subset of steps
            # via --step <list>, the other steps in the same phase used
            # to be unconditionally clobbered to "skipped" — losing the
            # record that they had succeeded earlier in the same RUN_ID.
            # Only mark "skipped" if the step is still in its default
            # (pending) state.
            if s.get("status") in ("completed", "failed"):
                return
            s["status"] = "skipped"

    _mutate(a.path, fn)


def apply_outcomes(path: str, outcomes: list) -> None:
    """Write per-step quality outcomes in ONE atomic mutation.
    outcomes: [{"phase","step","outcome","reason"}, ...]. Imported and called
    by scripts/lib/step_quality.py --write; also exposed as the `set-outcome`
    subcommand for shell callers."""
    def fn(d):
        for o in outcomes:
            s = _step(d, o.get("phase"), o.get("step"))
            if s:
                s["outcome"] = o.get("outcome")
                s["outcome_reason"] = o.get("reason", "")
    _mutate(path, fn)


def cmd_set_outcome(a) -> None:
    apply_outcomes(a.path, [{"phase": a.phase, "step": a.step,
                             "outcome": a.outcome, "reason": a.reason}])


def cmd_resume_info(a) -> int:
    """For pipeline.sh --resume. Validate the existing run is actually
    resumable, then print run_id (line 1) + a one-line summary (line 2) to
    stdout. On any non-resumable condition, print a clear reason to stderr and
    return non-zero so the caller refuses rather than silently adopting a stale
    or mismatched run."""
    try:
        data = _load(a.path)
    except Exception as e:
        print(f"cannot read manifest {a.path}: {e}", file=sys.stderr)
        return 1
    run = data.get("run", {}) or {}
    rid = run.get("run_id")
    if not rid:
        print("manifest has no run_id", file=sys.stderr)
        return 1
    status = run.get("status")
    if status == "completed":
        print(f"run {rid} already completed — nothing to resume (omit --resume "
              f"for a fresh run, or export RUN_ID={rid} to force).", file=sys.stderr)
        return 1
    if a.profile and run.get("profile") and a.profile != run.get("profile"):
        print(f"manifest is profile '{run.get('profile')}' but you requested "
              f"'{a.profile}'. Refusing — export RUN_ID={rid} to force, or omit "
              f"--resume for a fresh run.", file=sys.stderr)
        return 1
    if a.domain and run.get("domain") and a.domain != run.get("domain"):
        print(f"manifest is domain '{run.get('domain')}' but you requested "
              f"'{a.domain}'. Refusing.", file=sys.stderr)
        return 1
    phases = run.get("phases", []) or []
    total = len(phases)
    done = sum(1 for p in phases if p.get("status") == "completed")
    summary = f"status={status}, {done}/{total} phases done"
    failure = run.get("failure") or {}
    if failure:
        step = failure.get("step")
        summary += f", last failure at {failure.get('phase')}" + (f".{step}" if step else "")
    print(rid)
    print(summary)
    return 0


def cmd_finalize(a) -> None:
    def fn(d):
        d["run"]["status"] = a.status
        d["run"]["finished_at"] = now_iso()
        d["run"]["current_phase"] = None
        if a.status == "failed":
            # Surface the first failed phase (and within it, first failed step
            # if any) at the top level so consumers don't have to walk the tree.
            for p in d["run"]["phases"]:
                if p.get("status") == "failed":
                    failure = {
                        "phase": p["name"],
                        "exit_code": p.get("exit_code"),
                        # When this failure happened (phase finish time) — gives
                        # operators a date in case the failure is now stale from
                        # a previous invocation. Falls back to now if missing.
                        "at": p.get("finished_at") or now_iso(),
                        "error": p.get("error", {}),
                    }
                    for s in p.get("steps", []):
                        if s.get("status") == "failed":
                            failure["step"] = s["name"]
                            failure["step_error"] = s.get("error", {})
                            # Step-level finish time is more precise than phase-
                            # level when both exist.
                            if s.get("finished_at"):
                                failure["at"] = s["finished_at"]
                            break
                    d["run"]["failure"] = failure
                    break
        elif "failure" in d["run"]:
            # Successful run after a previous failure — clear stale summary.
            del d["run"]["failure"]

    _mutate(a.path, fn)

    # Drop a dirt-simple sentinel next to the manifest for existence-check
    # consumers (Hadoop _SUCCESS convention). _SUCCESS xor _FAILED.
    base = os.path.dirname(os.path.abspath(a.path))
    run_id = _load(a.path)["run"]["run_id"]
    success = os.path.join(base, "_SUCCESS")
    failed = os.path.join(base, "_FAILED")
    for f in (success, failed):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    marker = success if a.status == "completed" else failed
    with open(marker, "w") as fh:
        fh.write(f"{run_id}\n{now_iso()}\n")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--path", required=True)
    p.add_argument("--phases-dir", required=True)
    p.add_argument("--phase-order", required=True, help="comma-separated canonical order")
    p.add_argument("--selected", required=True, help="comma-separated selected phases")
    p.add_argument("--run-id", required=True)
    p.add_argument("--domain", default="")
    p.add_argument("--profile", default="")
    p.add_argument("--platform", default="")
    p.add_argument("--git-sha", default="")
    p.add_argument("--git-branch", default="")
    p.add_argument("--step-filter", default="all")
    # pipeline.sh's PID + process-group ID, for scripts/kill_pipeline.sh.
    # Defaults to "" so older pipeline.sh invocations don't break.
    p.add_argument("--pid", default="")
    p.add_argument("--pgid", default="")
    # Source corpus path (env CORPUS_PATH at invocation time). Surfaces in
    # logs.sh --summary so operator can quickly confirm which corpus is
    # being processed — especially useful when overriding pilot config with
    # a smoke fixture for debugging (CORPUS_PATH=corpus/<domain>/smoke).
    p.add_argument("--corpus-path", default="")
    # RunPod injects RUNPOD_POD_ID into every pod container. Recorded here so the
    # S3-synced manifest carries the pod id for an out-of-band watchdog
    # (scripts/monitor_pipeline.sh / a Lambda) to stop the pod on failure/stall.
    p.add_argument("--runpod-pod-id", default="")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("start-phase")
    p.add_argument("--path", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--log-file", default="")
    p.set_defaults(func=cmd_start_phase)

    p = sub.add_parser("end-phase")
    p.add_argument("--path", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.add_argument("--log-file", default="", help="path to phase log; tail captured on failure")
    p.add_argument("--error-message", default="", help="explicit error msg; default: 'exit code N'")
    p.add_argument("--tail-lines", type=int, default=30, help="N trailing log lines to capture (default 30)")
    p.set_defaults(func=cmd_end_phase)

    p = sub.add_parser("start-step")
    p.add_argument("--path", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--step", required=True)
    p.add_argument("--log-file", default="")
    p.add_argument("--cw-stream", default="")
    p.set_defaults(func=cmd_start_step)

    p = sub.add_parser("end-step")
    p.add_argument("--path", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--step", required=True)
    p.add_argument("--exit-code", type=int, required=True)
    p.add_argument("--log-file", default="")
    p.add_argument("--error-message", default="", help="explicit error msg; default: 'exit code N'")
    p.add_argument("--tail-lines", type=int, default=30)
    p.set_defaults(func=cmd_end_step)

    p = sub.add_parser("skip-step")
    p.add_argument("--path", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--step", required=True)
    p.set_defaults(func=cmd_skip_step)

    p = sub.add_parser("set-outcome")
    p.add_argument("--path", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--step", required=True)
    p.add_argument("--outcome", required=True,
                   choices=["pass", "warn", "fail", "skip", "unknown"])
    p.add_argument("--reason", default="")
    p.set_defaults(func=cmd_set_outcome)

    p = sub.add_parser("resume-info")
    p.add_argument("--path", required=True)
    p.add_argument("--profile", default="")
    p.add_argument("--domain", default="")
    p.set_defaults(func=cmd_resume_info)

    p = sub.add_parser("finalize")
    p.add_argument("--path", required=True)
    p.add_argument("--status", required=True, choices=["completed", "failed"])
    p.set_defaults(func=cmd_finalize)

    a = ap.parse_args()
    # Most subcommands mutate and return None (→ exit 0); resume-info returns an
    # int exit code that must propagate so pipeline.sh can refuse a bad resume.
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())
