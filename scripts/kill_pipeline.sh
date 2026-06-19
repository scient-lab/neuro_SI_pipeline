#!/usr/bin/env bash
# kill_pipeline.sh — kill the running pipeline.sh process tree.
#
# WHY: `pkill -f pipeline.sh` only matches the orchestrator process. Its
# children (bash phase scripts → python phase entry points → vLLM workers)
# don't match the pattern and survive as orphans, holding GB of VRAM until
# they OOM or get reaped manually. This script kills the entire process
# group using the PID/PGID pipeline.sh recorded in the manifest at init.
#
# Order of attempts:
#   1. Process-group SIGTERM (catches every descendant by PGID)
#   2. Grace period (3 sec) so vLLM can release GPU memory cleanly
#   3. Process-group SIGKILL (anything that didn't exit)
#   4. Defensive sweep: SIGKILL any remaining known-pipeline python entry
#      points (covers PGID-loss edge cases — e.g. if a child re-set its PGID)
#   5. Cancel the periodic S3 sync loop too (separate process, not in PGID)
#
# Usage:
#   ./scripts/kill_pipeline.sh                # kills current pipeline + cleanup
#   ./scripts/kill_pipeline.sh --dry-run      # show what would be killed
#   ./scripts/kill_pipeline.sh --manifest <path>   # specific manifest
#   ./scripts/kill_pipeline.sh -h
#
# Exit code: 0 on clean kill (or nothing to kill), 1 on stuck processes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="${REPO_ROOT}/outputs/run_manifest.json"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --- Resolve PID/PGID from manifest ----------------------------------------
PID=""
PGID=""
if [[ -f "$MANIFEST" ]]; then
    read -r PID PGID < <(python3 -c "
import json
try:
    m = json.load(open('$MANIFEST'))['run']
    print(m.get('pid') or '', m.get('pgid') or '')
except Exception:
    print('', '')
")
fi

# --- The known-pipeline python entry points (for the defensive sweep) ------
# Each phase's primary python invocation. If a child orphans and outlives the
# PGID kill (e.g. because it re-grouped itself), this catches it.
SWEEP_PATTERN='python.*(graphrag_index|add_llm_relations|entity_discovery|find_heads_positions|clean_llm_relations|run_dataset_preprocessing|run_mlm|predict_tails_llm|combine_tails|fact_score|merge_kgs)\.py'

# --- Helpers ---------------------------------------------------------------
mark() { echo "  $*"; }

show_target() {
    local label="$1"
    local pid="$2"
    [[ -z "$pid" ]] && return
    if kill -0 "$pid" 2>/dev/null; then
        mark "$label PID=$pid alive  →  $(ps -p "$pid" -o cmd= 2>/dev/null | head -c 80)"
    else
        mark "$label PID=$pid not running"
    fi
}

# --- Dry-run inspection ----------------------------------------------------
echo "=== Targets ==="
mark "manifest      : ${MANIFEST#$REPO_ROOT/}"
mark "recorded PID  : ${PID:-(none)}"
mark "recorded PGID : ${PGID:-(none)}"
if [[ -n "$PID" ]]; then
    show_target "pipeline.sh" "$PID"
fi
echo
echo "Process tree under PGID $PGID:"
if [[ -n "$PGID" ]]; then
    ps -o pid,pgid,cmd -g "$PGID" 2>/dev/null | head -30 || mark "(none in PGID)"
fi
echo
echo "Sweep pattern matches:"
pgrep -af "$SWEEP_PATTERN" 2>/dev/null | head -20 || mark "(no matches)"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run only — no kills issued."
    exit 0
fi

# --- 1. PGID SIGTERM (graceful) --------------------------------------------
TOOK_ACTION=0
if [[ -n "$PGID" ]] && kill -0 "-$PGID" 2>/dev/null; then
    echo "[1/4] SIGTERM process group $PGID..."
    kill -TERM "-$PGID" 2>/dev/null || true
    TOOK_ACTION=1
fi

# --- 2. Grace period -------------------------------------------------------
if [[ "$TOOK_ACTION" -eq 1 ]]; then
    echo "[2/4] Grace period 3s for vLLM to release GPU memory..."
    sleep 3
fi

# --- 3. PGID SIGKILL (anything left) ---------------------------------------
if [[ -n "$PGID" ]] && kill -0 "-$PGID" 2>/dev/null; then
    echo "[3/4] SIGKILL process group $PGID..."
    kill -KILL "-$PGID" 2>/dev/null || true
    sleep 1
fi

# --- 4. Defensive sweep ----------------------------------------------------
SWEEP=$(pgrep -af "$SWEEP_PATTERN" 2>/dev/null || true)
if [[ -n "$SWEEP" ]]; then
    echo "[4/4] Sweep — orphaned python phase processes:"
    echo "$SWEEP" | sed 's/^/    /'
    pkill -KILL -f "$SWEEP_PATTERN" 2>/dev/null || true
    sleep 1
fi

# --- 5. Periodic S3 sync loop (separate process, not in PGID) --------------
if pgrep -f 'sync_outputs.sh --loop' >/dev/null 2>&1; then
    echo "[5/5] Stopping sync_outputs.sh --loop..."
    pkill -f 'sync_outputs.sh --loop' 2>/dev/null || true
fi

# --- Final state ----------------------------------------------------------
echo
echo "=== Final state ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU:"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
        | sed 's/^/    /'
fi
LEFT=$(pgrep -af "$SWEEP_PATTERN" 2>/dev/null || true)
if [[ -n "$LEFT" ]]; then
    echo "Still alive:"
    echo "$LEFT" | sed 's/^/    /'
    exit 1
else
    echo "All pipeline processes terminated."
fi

# Mark manifest as failed if it's still showing running (the trap in
# pipeline.sh handles this when killed via signal, but if the orchestrator
# was already dead and we just cleaned up orphans, the manifest can be left
# saying status=running indefinitely).
if [[ -f "$MANIFEST" ]] && [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then
    python3 -c "
import json, datetime, os
m = json.load(open('$MANIFEST'))
if m.get('run', {}).get('status') == 'running':
    m['run']['status'] = 'failed'
    m['run']['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    m['run']['current_phase'] = None
    tmp = '$MANIFEST' + '.tmp'
    json.dump(m, open(tmp, 'w'), indent=2)
    os.replace(tmp, '$MANIFEST')
    print('manifest status: running -> failed')
" 2>/dev/null || true
fi
