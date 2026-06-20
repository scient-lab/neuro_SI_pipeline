#!/usr/bin/env python3
"""Render the pipeline-run manifest as a status table.

Ports the summary block out of scripts/logs.sh into a standalone .py file
so scripts/stats.sh can drive it (including live-refresh and system-stats
modes) without the shell-embedded-python heredoc fragility documented in
the [Test shell-embedded python locally] memory.

Used by:
  scripts/stats.sh                  - main consumer
  scripts/logs.sh                   - keeps its own inline copy unchanged

CLI:
  python3 scripts/lib/stats_render.py --manifest PATH [--details] [--run-id ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


STATUS_MARK = {
    "completed": "✓", "failed": "✗", "running": "…",
    "skipped":  "-", "pending":  " ",
}

PHASE_W = 24
STEP_NAME_MAX = PHASE_W - 4 - 3   # 4 indent + 3 branch glyph + 1 space


def _parse(t):
    if not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    except Exception:
        return None


def _fmt_duration(seconds):
    if seconds is None:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    h, rem = divmod(seconds, 3600)
    mm = rem // 60
    return f"{h}h {mm:02d}m"


def _phase_duration(p):
    s, f = _parse(p.get("started_at")), _parse(p.get("finished_at"))
    if s is None:
        return None
    end = f if f else datetime.now(timezone.utc)
    return (end - s).total_seconds()


def _step_duration(s):
    a, b = _parse(s.get("started_at")), _parse(s.get("finished_at"))
    if a is None:
        return None
    end = b if b else datetime.now(timezone.utc)
    return (end - a).total_seconds()


def _hhmmss(iso):
    t = _parse(iso)
    return t.strftime("%H:%M:%S") if t else ""


def _truncate(s, max_len):
    return s if len(s) <= max_len else (s[:max_len - 1] + "…")


def render_header(m, this_run, requested_run):
    if requested_run and this_run != requested_run:
        print(f"Note: manifest is for {this_run}, requested {requested_run}")
        print("(summary is only available for the run whose manifest is current)")
        return False

    started_at = m.get("started_at", "")
    resumed_at = m.get("resumed_at", "")
    finished_at = m.get("finished_at", "")
    ts_anchor = _parse(resumed_at) or _parse(started_at)
    ts_end = _parse(finished_at) if finished_at else datetime.now(timezone.utc)
    total_dur = (ts_end - ts_anchor).total_seconds() if ts_anchor else None

    print(f"Run     : {this_run}")
    print(f'Status  : {m.get("status","?"):<11s}  ({_fmt_duration(total_dur)})')
    if m.get("status") == "running":
        eta_at = _parse(m.get("estimated_completion_at"))
        prog = m.get("progress")
        if eta_at:
            rem = (eta_at - datetime.now(timezone.utc)).total_seconds()
            pct = f"{prog * 100:.0f}% done · " if isinstance(prog, (int, float)) else ""
            print(f'ETA     : ~{_fmt_duration(max(0, rem))} left  '
                  f'({pct}~{eta_at.strftime("%H:%M")} UTC · rough)')
        else:
            print("ETA     : estimating…")
    print(f'Profile : {m.get("profile","")}    Domain: {m.get("domain","")}    '
          f'Platform: {m.get("platform","")}')
    cp = m.get("corpus_path") or "(default: corpus/[domain]/source_txt)"
    print(f"Corpus  : {cp}")
    print(f'Git     : {m.get("git_sha","")} ({m.get("git_branch","")})')
    print(f"Started : {started_at}")
    if resumed_at and resumed_at != started_at:
        print(f"Resumed : {resumed_at}")
    if finished_at:
        print(f"Finished: {finished_at}")
    print()
    return True


def render_phase_table(m, show_details):
    phases = m.get("phases", []) or []
    run_eta = _parse(m.get("estimated_completion_at"))

    print(f'  {"PHASE".ljust(PHASE_W)} {"STATUS".ljust(13)} '
          f'{"STARTED".ljust(10)} {"FINISHED".ljust(10)} '
          f'{"DURATION".ljust(11)} {"ETA".ljust(14)} {"STEPS".rjust(6)}')
    print(f'  {("-" * PHASE_W)} {("-" * 13)} {("-" * 10)} '
          f'{("-" * 10)} {("-" * 11)} {("-" * 14)} {("-" * 6)}')

    for p in phases:
        name = p.get("name", "?")
        st = p.get("status", "pending")
        mark = STATUS_MARK.get(st, "?")
        started = _hhmmss(p.get("started_at")) if st != "pending" else ""
        finished = _hhmmss(p.get("finished_at")) if st in ("completed", "failed", "skipped") else ""
        dur = _fmt_duration(_phase_duration(p)) if st != "pending" else ""
        eta = ""
        if st == "running" and run_eta:
            rem = (run_eta - datetime.now(timezone.utc)).total_seconds()
            eta = f"~{_fmt_duration(max(0, rem))} left"
        steps = p.get("steps", []) or []
        ok = sum(1 for s in steps if s.get("status") == "completed")
        total = len(steps)
        steps_str = f"{ok}/{total}" if total else ""
        print(f'  {name.ljust(PHASE_W)} {mark} {st.ljust(11)} '
              f'{started.ljust(10)} {finished.ljust(10)} {dur.ljust(11)} '
              f'{eta.ljust(14)} {steps_str.rjust(6)}')

        if show_details and st != "pending":
            for i, s in enumerate(steps):
                sname = _truncate(s.get("name", "?"), STEP_NAME_MAX)
                sst = s.get("status", "pending")
                smark = STATUS_MARK.get(sst, "?")
                sstart = _hhmmss(s.get("started_at")) if sst != "pending" else ""
                sfinish = _hhmmss(s.get("finished_at")) if sst in ("completed", "failed", "skipped") else ""
                sdur = _fmt_duration(_step_duration(s)) if sst != "pending" else ""
                branch = "└─" if i == len(steps) - 1 else "├─"
                nested = f"{branch} {sname}"
                print(f'    {nested.ljust(PHASE_W - 4)} {smark} {sst.ljust(11)} '
                      f'{sstart.ljust(10)} {sfinish.ljust(10)} {sdur.ljust(11)}')


def render_failure_block(m):
    f = m.get("failure")
    if not f:
        return
    phases = m.get("phases", []) or []
    failed_phase_name = f.get("phase")
    failed_phase_record = next(
        (pp for pp in phases if pp.get("name") == failed_phase_name), {})
    current_status_of_failed_phase = failed_phase_record.get("status", "")
    # Stale = a current/successful attempt under the same RUN_ID; the failure
    # record is from a prior invocation. Make that obvious so the operator
    # doesn't chase a ghost.
    is_stale = current_status_of_failed_phase in ("running", "completed", "pending")
    print()
    print(f"PREVIOUS-RUN FAILURE (phase now {current_status_of_failed_phase}):"
          if is_stale else "FAILURE:")
    print(f'  phase     : {f.get("phase")}')
    print(f'  step      : {f.get("step","(none)")}')
    print(f'  exit_code : {f.get("exit_code")}')
    f_at = f.get("at")
    if f_at:
        ts_f = _parse(f_at)
        ago = _fmt_duration((datetime.now(timezone.utc) - ts_f).total_seconds()) if ts_f else ""
        print(f"  at        : {f_at}  ({ago} ago)" if ago else f"  at        : {f_at}")
    err = f.get("error") or {}
    print(f'  message   : {err.get("message","")}')
    tail = err.get("log_tail", [])
    if tail:
        print(f"  log tail  : ({len(tail)} lines, last {min(5, len(tail))} shown)")
        for ln in tail[-5:]:
            print(f"    {ln}")
        print()
        this_run = m.get("run_id", "")
        print(f"  Full tail: ./scripts/logs.sh --run {this_run} --error")
        print(f'  Full log : ./scripts/logs.sh --run {this_run} --phase {f.get("phase")}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--run-id", default=None, help="Validate manifest is for this run")
    p.add_argument("--details", action="store_true")
    args = p.parse_args()

    try:
        with open(args.manifest) as f:
            m = json.load(f).get("run", {})
    except FileNotFoundError:
        print(f"No manifest at {args.manifest}", file=sys.stderr)
        return 1

    this_run = m.get("run_id", "")
    if not render_header(m, this_run, args.run_id):
        return 0

    render_phase_table(m, args.details)
    render_failure_block(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
