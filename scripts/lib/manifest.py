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
from datetime import datetime, timezone

SCHEMA_VERSION = "1.0"
STATUS_ENUM = ["pending", "running", "completed", "failed", "skipped"]
TIMESTAMP_FORMAT = "RFC3339 / ISO-8601 with timezone offset (e.g. 2026-06-17T14:15:23+00:00)"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    """Lock, load, apply fn(data) in place, save atomically."""
    with _locked(path):
        data = _load(path)
        fn(data)
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
        existing["schema_version"] = SCHEMA_VERSION
        existing["meta"] = meta
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
            "selected_phases": [p for p in phase_order if p in selected],
            "phases": run_phases,
        },
    }
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

    _mutate(a.path, fn)


def cmd_end_phase(a) -> None:
    def fn(d):
        p = _find(d["run"]["phases"], a.phase)
        if p:
            p["finished_at"] = now_iso()
            p["exit_code"] = a.exit_code
            p["status"] = "completed" if a.exit_code == 0 else "failed"
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

    _mutate(a.path, fn)


def cmd_skip_step(a) -> None:
    def fn(d):
        s = _step(d, a.phase, a.step)
        if s:
            s["status"] = "skipped"

    _mutate(a.path, fn)


def cmd_finalize(a) -> None:
    def fn(d):
        d["run"]["status"] = a.status
        d["run"]["finished_at"] = now_iso()
        d["run"]["current_phase"] = None

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
    p.set_defaults(func=cmd_end_step)

    p = sub.add_parser("skip-step")
    p.add_argument("--path", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--step", required=True)
    p.set_defaults(func=cmd_skip_step)

    p = sub.add_parser("finalize")
    p.add_argument("--path", required=True)
    p.add_argument("--status", required=True, choices=["completed", "failed"])
    p.set_defaults(func=cmd_finalize)

    a = ap.parse_args()
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
