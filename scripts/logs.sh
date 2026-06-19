#!/usr/bin/env bash
# logs.sh — view per-phase / per-step logs from a pipeline run.
#
# Logs live at $OUTPUT_BASE/logs/<run_id>/:
#   <phase>.log              <- from pipeline.sh's tee  (phase-level)
#   <phase>/<step>.log       <- from lib/common.sh::run_step (step-level)
#
# Manifest at $OUTPUT_BASE/run_manifest.json gives status + failure info for
# the LATEST run (older runs are superseded; logs persist but status doesn't).
#
# Flag syntax mirrors pipeline.sh (--phase/--step). Usage examples:
#   ./scripts/logs.sh                                    # latest run, all phases
#   ./scripts/logs.sh --summary                          # health-check: per-phase status table
#   ./scripts/logs.sh --summary --details                # also expand each phase's steps inline
#   ./scripts/logs.sh --phase graphmert                  # one phase
#   ./scripts/logs.sh --phase graphmert --step tokenize  # one step
#   ./scripts/logs.sh --run 20260617 --error             # triage today's failure
#   ./scripts/logs.sh --tail                             # follow live
#   ./scripts/logs.sh --list                             # available runs
#   ./scripts/logs.sh --paths                            # just paths (pipe to vim/grep)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

RUN_ID=""
PHASE=""
STEP=""
TAIL=0
LIST_ONLY=0
ERROR_ONLY=0
PATHS_ONLY=0
SUMMARY_ONLY=0
SUMMARY_DETAILS=0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)      RUN_ID="$2"; shift 2 ;;
        --phase)    PHASE="$2"; shift 2 ;;
        --step)     STEP="$2"; shift 2 ;;
        --tail|-f)  TAIL=1; shift ;;
        --list|-l)  LIST_ONLY=1; shift ;;
        --error|-e) ERROR_ONLY=1; shift ;;
        --paths|-p) PATHS_ONLY=1; shift ;;
        --summary|-s) SUMMARY_ONLY=1; shift ;;
        --details|-d) SUMMARY_DETAILS=1; SUMMARY_ONLY=1; shift ;;
        --help|-h)  usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

LOGS_BASE="$OUTPUT_BASE/logs"
MANIFEST="$OUTPUT_BASE/run_manifest.json"

# --- --list -----------------------------------------------------------------
if [[ "$LIST_ONLY" -eq 1 ]]; then
    if [[ ! -d "$LOGS_BASE" ]]; then
        echo "(no logs dir: $LOGS_BASE)"; exit 0
    fi
    # The manifest only covers ONE run (the latest). For older runs we just
    # show their dir existence — no status is recoverable without per-run
    # manifests.
    current_run=""
    current_status=""
    if [[ -f "$MANIFEST" ]]; then
        read -r current_run current_status < <(python3 -c "
import json
m = json.load(open('$MANIFEST'))['run']
print(m.get('run_id',''), m.get('status',''))
")
    fi
    printf "%-44s %s\n" "RUN_ID" "STATUS"
    printf "%-44s %s\n" "------" "------"
    for rd in $(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -r); do
        if [[ "$rd" == "$current_run" ]]; then
            printf "%-44s %s\n" "$rd" "$current_status (current)"
        else
            printf "%-44s %s\n" "$rd" "(historical)"
        fi
    done
    exit 0
fi

# --- Resolve RUN_ID (default latest, or prefix match) -----------------------
if [[ -z "$RUN_ID" ]]; then
    RUN_ID=$(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -r | head -1 || true)
    [[ -z "$RUN_ID" ]] && { echo "No runs found in $LOGS_BASE/"; exit 1; }
elif [[ ! -d "$LOGS_BASE/$RUN_ID" ]]; then
    mapfile -t matches < <(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E "^${RUN_ID}" | sort -r)
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "No run matching: $RUN_ID" >&2; exit 1
    elif [[ ${#matches[@]} -gt 1 ]]; then
        echo "Multiple runs match '$RUN_ID':" >&2
        printf "  %s\n" "${matches[@]}" >&2
        echo "Specify a more specific prefix." >&2
        exit 1
    fi
    RUN_ID="${matches[0]}"
fi

LOG_DIR="$LOGS_BASE/$RUN_ID"

# --- --summary / -s (manifest health-check view) ----------------------------
# Compact run-health table: per-phase status + duration + step counts +
# top-level failure block if any. Reads the manifest only — no log dump.
if [[ "$SUMMARY_ONLY" -eq 1 ]]; then
    [[ -f "$MANIFEST" ]] || { echo "No manifest at $MANIFEST"; exit 1; }
    REQUESTED="$RUN_ID" python3 -c "
import json, os, sys
from datetime import datetime, timezone

m = json.load(open('$MANIFEST'))['run']
requested = os.environ['REQUESTED']
this_run = m.get('run_id', '')
if this_run != requested:
    print(f'Note: manifest is for {this_run}, requested {requested}')
    print('(summary is only available for the run whose manifest is current)')
    sys.exit(0)

def _parse(t):
    if not t: return None
    try: return datetime.fromisoformat(t.replace('Z','+00:00'))
    except Exception: return None

def _fmt_duration(seconds):
    if seconds is None: return ''
    seconds = int(seconds)
    if seconds < 60:  return f'{seconds}s'
    if seconds < 3600: return f'{seconds//60}m {seconds%60:02d}s'
    h, rem = divmod(seconds, 3600); mm = rem // 60
    return f'{h}h {mm:02d}m'

def _phase_duration(p):
    s, f = _parse(p.get('started_at')), _parse(p.get('finished_at'))
    if s is None: return None
    end = f if f else datetime.now(timezone.utc)
    return (end - s).total_seconds()

STATUS_MARK = {
    'completed': '✓', 'failed': '✗', 'running': '…',
    'skipped':  '-', 'pending':  ' ',
}

# --- header ---
started_at = m.get('started_at', '')
resumed_at = m.get('resumed_at', '')
finished_at = m.get('finished_at', '')
# Elapsed clock anchors on resumed_at when present (multi-invocation runs);
# falls back to started_at for single-invocation runs.
ts_anchor = _parse(resumed_at) or _parse(started_at)
ts_end = _parse(finished_at) if finished_at else datetime.now(timezone.utc)
total_dur = (ts_end - ts_anchor).total_seconds() if ts_anchor else None

print(f'Run     : {this_run}')
print(f'Status  : {m.get(\"status\",\"?\"):<11s}  ({_fmt_duration(total_dur)})')
# ETA is computed + stored by manifest.py (so the API/UI get it too); we just
# render it, deriving 'remaining' live from the absolute completion estimate.
if m.get('status') == 'running':
    eta_at = _parse(m.get('estimated_completion_at'))
    prog = m.get('progress')
    if eta_at:
        rem = (eta_at - datetime.now(timezone.utc)).total_seconds()
        pct = f'{prog*100:.0f}% done · ' if isinstance(prog, (int, float)) else ''
        print(f'ETA     : ~{_fmt_duration(max(0, rem))} left  ({pct}~{eta_at.strftime(\"%H:%M\")} UTC · rough)')
    else:
        print('ETA     : estimating…')
print(f'Profile : {m.get(\"profile\",\"\")}    Domain: {m.get(\"domain\",\"\")}    Platform: {m.get(\"platform\",\"\")}')
cp = m.get('corpus_path') or '(default: corpus/[domain]/source_txt)'
print(f'Corpus  : {cp}')
print(f'Git     : {m.get(\"git_sha\",\"\")} ({m.get(\"git_branch\",\"\")})')
print(f'Started : {started_at}')
if resumed_at and resumed_at != started_at:
    # Multi-invocation run: surface the most recent restart timestamp so the
    # operator can see the elapsed clock is anchored on it, not on the
    # original started_at from days/weeks ago.
    print(f'Resumed : {resumed_at}')
if finished_at:
    print(f'Finished: {finished_at}')
print()

# --- per-phase table ---
# Columns: PHASE | STATUS | STARTED | FINISHED | DURATION | ETA | STEPS
# Times shown as HH:MM:SS UTC (compact, sortable). Pending phases show
# blanks across the board. Running phases show their started_at + live
# duration; ETA is the run-level estimate, only on the currently-active
# phase. Completed phases show both started + finished + total duration.
phases = m.get('phases', []) or []
def _step_duration(s):
    a, b = _parse(s.get('started_at')), _parse(s.get('finished_at'))
    if a is None: return None
    end = b if b else datetime.now(timezone.utc)
    return (end - a).total_seconds()
def _hhmmss(iso):
    t = _parse(iso)
    return t.strftime('%H:%M:%S') if t else ''

show_details = '$SUMMARY_DETAILS' == '1'
# Run-level ETA (already computed up in the header). Reuse the absolute
# completion time to show on the currently-running phase row.
run_eta = _parse(m.get('estimated_completion_at'))
current_phase_name = m.get('current_phase')

# Column widths. PHASE column is 24 chars wide so nested step names
# rendered like '    branch step_name' (4-space indent + 3-char branch
# glyph + step name) fit in the same column as top-level phase names.
# Step names longer than 17 chars get truncated with an ellipsis so the
# STATUS column stays aligned across phase and step rows.
PHASE_W = 24
STEP_NAME_MAX = PHASE_W - 4 - 3   # 4 indent + 3 branch glyph + 1 space

def _truncate(s, max_len):
    return s if len(s) <= max_len else (s[:max_len - 1] + '…')

# Use .ljust()/.rjust() instead of variable-width f-string format specs.
# Reason: the enclosing bash heredoc parses certain shell metacharacters
# even inside Python comments — variable-width format specs use a pattern
# that bash interprets as process substitution, crashing with
# 'name: No such file or directory'. Method calls dodge that.
print(f'  {\"PHASE\".ljust(PHASE_W)} {\"STATUS\".ljust(13)} {\"STARTED\".ljust(10)} {\"FINISHED\".ljust(10)} {\"DURATION\".ljust(11)} {\"ETA\".ljust(14)} {\"STEPS\".rjust(6)}')
print(f'  {(\"-\"*PHASE_W)} {(\"-\"*13)} {(\"-\"*10)} {(\"-\"*10)} {(\"-\"*11)} {(\"-\"*14)} {(\"-\"*6)}')

for p in phases:
    name = p.get('name','?')
    st   = p.get('status', 'pending')
    mark = STATUS_MARK.get(st, '?')
    started  = _hhmmss(p.get('started_at')) if st != 'pending' else ''
    finished = _hhmmss(p.get('finished_at')) if st in ('completed', 'failed', 'skipped') else ''
    dur  = _fmt_duration(_phase_duration(p)) if st != 'pending' else ''
    eta  = ''
    if st == 'running' and run_eta:
        rem = (run_eta - datetime.now(timezone.utc)).total_seconds()
        eta = f'~{_fmt_duration(max(0, rem))} left'
    steps = p.get('steps', []) or []
    ok    = sum(1 for s in steps if s.get('status') == 'completed')
    total = len(steps)
    steps_str = f'{ok}/{total}' if total else ''
    print(f'  {name.ljust(PHASE_W)} {mark} {st.ljust(11)} {started.ljust(10)} {finished.ljust(10)} {dur.ljust(11)} {eta.ljust(14)} {steps_str.rjust(6)}')
    # --details: nest each step under the phase row (only for phases that
    # have started — pending phases would show all-pending steps and clutter
    # the view without adding info). Step rows mirror the phase columns
    # except ETA (no per-step progress info to compute one from).
    if show_details and st != 'pending':
        for i, s in enumerate(steps):
            sname = _truncate(s.get('name','?'), STEP_NAME_MAX)
            sst   = s.get('status', 'pending')
            smark = STATUS_MARK.get(sst, '?')
            sstart  = _hhmmss(s.get('started_at')) if sst != 'pending' else ''
            sfinish = _hhmmss(s.get('finished_at')) if sst in ('completed', 'failed', 'skipped') else ''
            sdur  = _fmt_duration(_step_duration(s)) if sst != 'pending' else ''
            branch = '└─' if i == len(steps) - 1 else '├─'
            # Build the nested branch+name prefix and pad to PHASE_W width so
            # the STATUS column lines up with phase rows above.
            nested = f'{branch} {sname}'
            print(f'    {nested.ljust(PHASE_W - 4)} {smark} {sst.ljust(11)} {sstart.ljust(10)} {sfinish.ljust(10)} {sdur.ljust(11)}')

# --- failure summary ---
f = m.get('failure')
if f:
    # If the failed phase is now running (or completed) in this invocation,
    # the failure block is from a PREVIOUS attempt of the same RUN_ID.
    # Mark it so the operator doesn't think the CURRENT attempt failed.
    failed_phase_name = f.get('phase')
    failed_phase_record = next(
        (pp for pp in phases if pp.get('name') == failed_phase_name), {})
    current_status_of_failed_phase = failed_phase_record.get('status', '')
    is_stale = current_status_of_failed_phase in ('running', 'completed', 'pending')
    print()
    if is_stale:
        print(f'PREVIOUS-RUN FAILURE (phase now {current_status_of_failed_phase}):')
    else:
        print('FAILURE:')
    print(f'  phase     : {f.get(\"phase\")}')
    print(f'  step      : {f.get(\"step\",\"(none)\")}')
    print(f'  exit_code : {f.get(\"exit_code\")}')
    f_at = f.get('at')
    if f_at:
        # How long ago — gives operator immediate orientation ('yesterday'
        # vs 'just now') without parsing the ISO timestamp.
        ts_f = _parse(f_at)
        ago = _fmt_duration((datetime.now(timezone.utc) - ts_f).total_seconds()) if ts_f else ''
        print(f'  at        : {f_at}  ({ago} ago)' if ago else f'  at        : {f_at}')
    err = f.get('error') or {}
    print(f'  message   : {err.get(\"message\",\"\")}')
    tail = err.get('log_tail', [])
    if tail:
        print(f'  log tail  : ({len(tail)} lines, last {min(5,len(tail))} shown)')
        for ln in tail[-5:]:
            print(f'    {ln}')
        print()
        print(f'  Full tail: ./scripts/logs.sh --run {this_run} --error')
        print(f'  Full log : ./scripts/logs.sh --run {this_run} --phase {f.get(\"phase\")}')
"
    exit 0
fi

# --- --error / -e (manifest triage view) ------------------------------------
if [[ "$ERROR_ONLY" -eq 1 ]]; then
    [[ -f "$MANIFEST" ]] || { echo "No manifest at $MANIFEST"; exit 1; }
    python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))['run']
this_run = m.get('run_id', '')
if this_run != '$RUN_ID':
    print(f'Note: manifest is for {this_run}, requested $RUN_ID')
    print('(failure info is only available for the run whose manifest is current)')
    sys.exit(0)
print(f'Run:    {this_run}')
print(f'Status: {m.get(\"status\")}')
print(f'Phases: {len(m.get(\"phases\",[]))}')
f = m.get('failure')
if not f:
    print()
    print('(no run.failure recorded)')
    sys.exit(0)
print()
print('FAILURE:')
print(f'  phase:     {f.get(\"phase\")}')
print(f'  step:      {f.get(\"step\", \"(none)\")}')
print(f'  exit_code: {f.get(\"exit_code\")}')
err = f.get('error') or {}
print(f'  message:   {err.get(\"message\")}')
print()
print('Log tail:')
for ln in err.get('log_tail', []):
    print(f'  {ln}')
"
    exit 0
fi

# --- Build the log-file list per filters -----------------------------------
# Explicit ()  init — `declare -a` alone leaves the var unset on some bash
# versions, which trips `set -u` when we read ${#LOG_FILES[@]} on an empty result.
LOG_FILES=()

if [[ -n "$STEP" ]]; then
    # Step-level: nested at <phase>/<step>.log (preferred), or grep the phase log.
    if [[ -n "$PHASE" ]]; then
        f="$LOG_DIR/$PHASE/$STEP.log"
        [[ -f "$f" ]] && LOG_FILES+=("$f")
    else
        # No phase: scan every phase dir for that step.
        for d in "$LOG_DIR"/*/; do
            [[ -d "$d" ]] || continue
            f="${d}${STEP}.log"
            [[ -f "$f" ]] && LOG_FILES+=("$f")
        done
    fi
elif [[ -n "$PHASE" ]]; then
    # Phase-level: phase.log + every step log under <phase>/.
    [[ -f "$LOG_DIR/$PHASE.log" ]] && LOG_FILES+=("$LOG_DIR/$PHASE.log")
    if [[ -d "$LOG_DIR/$PHASE" ]]; then
        while IFS= read -r sl; do LOG_FILES+=("$sl"); done < <(find "$LOG_DIR/$PHASE" -maxdepth 1 -name '*.log' -type f | sort)
    fi
else
    # Everything in the run.
    while IFS= read -r lf; do LOG_FILES+=("$lf"); done < <(find "$LOG_DIR" -name '*.log' -type f | sort)
fi

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
    echo "No log files matched in $LOG_DIR" >&2
    [[ -n "$PHASE" || -n "$STEP" ]] && echo "  (filters: phase=${PHASE:-*} step=${STEP:-*})" >&2
    exit 1
fi

# --- Render -----------------------------------------------------------------
# --paths: emit just the absolute paths, one per line. Useful for:
#   vim $(./scripts/logs.sh --paths --phase graphmert)
#   grep -l ERROR $(./scripts/logs.sh --paths)
if [[ "$PATHS_ONLY" -eq 1 ]]; then
    for lf in "${LOG_FILES[@]}"; do echo "$lf"; done
    exit 0
fi

echo "=== Run: $RUN_ID  (${#LOG_FILES[@]} file(s)) ==="
echo
echo "Log files:"
for lf in "${LOG_FILES[@]}"; do
    echo "  ${lf#"$REPO_ROOT"/}"
done

if [[ "$TAIL" -eq 1 ]]; then
    # tail -F handles multiple files with banners between them (built-in headers).
    tail -F "${LOG_FILES[@]}"
else
    for lf in "${LOG_FILES[@]}"; do
        echo
        echo "===== ${lf#"$REPO_ROOT"/} ====="
        cat "$lf"
    done
fi
