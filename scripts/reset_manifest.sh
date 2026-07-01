#!/usr/bin/env bash
# scripts/reset_manifest.sh — clear a failed pipeline run's terminal state
# so it can be resumed.
#
# What it does:
#   1. Removes the _FAILED / _SUCCESS Hadoop-style sentinels from the run dir
#   2. Resets run_manifest.json::run.status from "failed" → "running"
#   3. Clears run_manifest.json::run.failure (stale summary from the prior crash)
#   4. Optionally overrides phase.step statuses you specify via --mark
#
# What it doesn't touch:
#   - The actual data outputs (curriculum.json, checkpoints, etc.)
#   - Step-level history that wasn't explicitly --mark'd
#   - The pipeline.sh process itself (run separately after this script returns)
#
# Usage:
#   ./scripts/reset_manifest.sh                                  # latest run
#   ./scripts/reset_manifest.sh --run <prefix>                   # specific run
#   ./scripts/reset_manifest.sh --mark curriculum.generate_qa_pair=completed
#   ./scripts/reset_manifest.sh --mark sft.train_lora=pending --mark rl.train_grpo=pending
#   ./scripts/reset_manifest.sh --dry-run                        # print plan, no changes
#
# --mark <phase.step=status>:
#   Override a specific step's status. Common values: pending / completed /
#   failed / running. Multiple --mark flags are allowed. The 'finished_at'
#   timestamp is set to the artifact mtime when status=completed and a
#   conventional artifact exists; otherwise to now.
#
# Exit: 0 on success, 1 on error, 2 if --dry-run and would have changed something.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

# --- args ------------------------------------------------------------------
RUN_PREFIX=""
DRY_RUN=0
MARKS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)     RUN_PREFIX="$2"; shift 2 ;;
        --mark)    MARKS+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)         echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --- find the run dir -------------------------------------------------------
# Three layouts to handle:
#   (A) OUTPUT_BASE is already the RUN dir  — outputs/<RUN_ID>/
#       (pipeline.sh convention when operator exports OUTPUT_BASE)
#       Detected by: run_manifest.json present DIRECTLY at $OUTPUT_BASE/
#   (B) OUTPUT_BASE is the runs parent      — outputs/  (default when
#       OUTPUT_BASE unset; $REPO_ROOT/outputs)
#       Detected by: child dirs matching ^YYYYMMDD-HHMMSS-* WITH manifest
#   (C) Shared-logs layout (legacy)         — outputs/logs/<RUN_ID>/
#       Detected by: matching child under $OUTPUT_BASE/logs/
find_run_dir() {
    # (A) OUTPUT_BASE itself is the run dir — most common when the operator
    # follows the pipeline.sh-style export OUTPUT_BASE=outputs/<RUN_ID>.
    # The earlier bug here was iterating to $OUTPUT_BASE/logs first and
    # finding the nested LOG_DIR (which IS named outputs/<RUN_ID>/logs/<RUN_ID>/
    # per pipeline.sh:219), producing a doubled-RUN_ID path.
    if [[ -f "$OUTPUT_BASE/run_manifest.json" ]]; then
        echo "$OUTPUT_BASE"
        return 0
    fi
    # (B) and (C): scan for run-id-shaped child dirs that contain a manifest
    local layout
    for layout in "$OUTPUT_BASE" "$OUTPUT_BASE/logs"; do
        [[ -d "$layout" ]] || continue
        if [[ -z "$RUN_PREFIX" ]]; then
            local latest
            latest=$(find "$layout" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                | grep -E '^[0-9]{8}-[0-9]{6}' | sort -r | head -1 || true)
            [[ -n "$latest" && -f "$layout/$latest/run_manifest.json" ]] && \
                { echo "$layout/$latest"; return 0; }
        else
            local match
            match=$(find "$layout" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                | grep -E "^${RUN_PREFIX}" | sort -r | head -1 || true)
            [[ -n "$match" && -f "$layout/$match/run_manifest.json" ]] && \
                { echo "$layout/$match"; return 0; }
        fi
    done
    return 1
}

RUN_DIR="$(find_run_dir || true)"
if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: no run dir found." >&2
    echo "  searched: $OUTPUT_BASE/<RUN_ID>/ and $OUTPUT_BASE/logs/<RUN_ID>/" >&2
    echo "  available runs:" >&2
    for layout in "$OUTPUT_BASE" "$OUTPUT_BASE/logs"; do
        [[ -d "$layout" ]] || continue
        find "$layout" -mindepth 1 -maxdepth 1 -type d -printf '    %f\n' 2>/dev/null \
            | sort -r | head -10 >&2
    done
    exit 1
fi
RUN_ID=$(basename "$RUN_DIR")
MANIFEST="$RUN_DIR/run_manifest.json"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: no run_manifest.json at $MANIFEST" >&2
    exit 1
fi

echo "Run     : $RUN_ID"
echo "Run dir : ${RUN_DIR#$REPO_ROOT/}"
[[ "$DRY_RUN" -eq 1 ]] && echo "Mode    : DRY RUN (no changes will be written)"
echo

# --- step 1: remove _FAILED / _SUCCESS sentinels ---------------------------
echo "=== sentinel cleanup ==="
for f in _FAILED _SUCCESS; do
    if [[ -f "$RUN_DIR/$f" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "  would remove: $f"
        else
            rm -f "$RUN_DIR/$f"
            echo "  removed: $f"
        fi
    else
        echo "  not present: $f"
    fi
done

# --- step 2: manifest mutations (run-level + per-step --marks) ------------
echo
echo "=== manifest mutations ==="

# Pass marks as TSV via env to python; \\t between phase/step/status.
MARKS_ENV=""
for m in "${MARKS[@]:-}"; do
    [[ -z "$m" ]] && continue
    if [[ ! "$m" =~ ^[a-z_]+\.[a-z_]+=[a-z_]+$ ]]; then
        echo "ERROR: --mark '$m' must be phase.step=status (lowercase, underscores)" >&2
        exit 1
    fi
    MARKS_ENV+="${m}"$'\n'
done
export MARKS_ENV
export MANIFEST RUN_DIR DRY_RUN

python3 - <<'PY'
import json, os, sys
from datetime import datetime, timezone

manifest_path = os.environ['MANIFEST']
dry = os.environ.get('DRY_RUN', '0') == '1'
marks_raw = os.environ.get('MARKS_ENV', '').strip()

with open(manifest_path) as f:
    m = json.load(f)

changes = []

# --- run-level cleanups ---
if m.get('run', {}).get('status') == 'failed':
    changes.append("  run.status: failed -> running")
    if not dry:
        m['run']['status'] = 'running'

if 'failure' in m.get('run', {}):
    fail_summary = m['run']['failure']
    changes.append(f"  run.failure: cleared (was: {fail_summary.get('phase')}.{fail_summary.get('step')})")
    if not dry:
        del m['run']['failure']

# --- per-step --mark overrides ---
target_marks = {}
if marks_raw:
    for line in marks_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        phase_step, status = line.split('=', 1)
        phase, step = phase_step.split('.', 1)
        target_marks[(phase, step)] = status

if target_marks:
    now_iso = datetime.now(timezone.utc).isoformat()
    # Map of conventional artifact paths so finished_at gets a real timestamp.
    artifact_for = {
        # 4-step flow: pair/validate_pair/item/validate_item all stream curriculum.jsonl;
        # its mtime is a fine finished_at proxy (assemble emits the verified array).
        ('curriculum', 'generate_qa_pair'): 'curriculum/curriculum.jsonl',
        ('curriculum', 'validate_qa_pair'): 'curriculum/curriculum.jsonl',
        ('curriculum', 'generate_qa_item'): 'curriculum/curriculum.jsonl',
        ('curriculum', 'validate_qa_item'): 'curriculum/curriculum.jsonl',
        ('curriculum', 'assemble_curriculum'): 'curriculum_verified/curriculum_verified.json',
        ('extract', 'cache'): 'graphrag/output/kg_final.parquet',
    }

    # Manifest schema uses {'name': 'curriculum', 'steps': [{'name': 'generate_qa', ...}]}
    # NOT {'phase': '...', 'step': '...'} — the bug was the initial draft assumed
    # the latter, causing all --mark overrides to silently no-op.
    for phase in m.get('run', {}).get('phases', []):
        pname = phase.get('name')
        for step in phase.get('steps', []):
            sname = step.get('name')
            key = (pname, sname)
            if key not in target_marks:
                continue
            target_status = target_marks[key]
            old = step.get('status', 'pending')
            changes.append(f"  {pname}.{sname}: {old} -> {target_status}")
            if dry:
                continue
            step['status'] = target_status
            if target_status == 'completed':
                # Use artifact mtime if we know where it lives, else now.
                rel = artifact_for.get(key)
                fpath = os.path.join(os.environ['RUN_DIR'], rel) if rel else None
                if fpath and os.path.exists(fpath):
                    step['finished_at'] = datetime.fromtimestamp(
                        os.path.getmtime(fpath), tz=timezone.utc).isoformat()
                elif 'finished_at' not in step or not step['finished_at']:
                    step['finished_at'] = now_iso
                step['exit_code'] = 0
                step.pop('error', None)
            elif target_status == 'pending':
                # Wipe timestamps so re-run starts cleanly
                for f in ('started_at', 'finished_at', 'duration_seconds', 'exit_code', 'error'):
                    step.pop(f, None)

if not changes:
    print("  no changes needed (run already healthy)")
else:
    for c in changes:
        print(c)
    if not dry:
        with open(manifest_path, 'w') as f:
            json.dump(m, f, indent=2)
        print(f"\n  ✓ manifest updated: {manifest_path}")
    else:
        print(f"\n  (dry-run — manifest NOT written)")

sys.exit(2 if (dry and changes) else 0)
PY

echo
echo "=== done ==="
echo "Verify: ./scripts/stats.sh --run $RUN_ID"
