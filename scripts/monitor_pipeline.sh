#!/bin/bash
# scripts/monitor_pipeline.sh — Monitor pipeline and auto-kill pod on failure (nighttime safety net)
#
# Usage (background):
#   ./scripts/monitor_pipeline.sh &
#   # or with custom timeout:
#   MONITOR_TIMEOUT=600 ./scripts/monitor_pipeline.sh &
#
# Configuration:
#   MONITOR_TIMEOUT    — Seconds to wait after detecting failure before killing pod (default: 600 = 10 min)
#   MONITOR_CHECK_INTERVAL — Poll interval in seconds (default: 30)
#   MONITOR_ENABLED    — Set to "0" to disable monitoring (useful for daytime debugging)
#
# Behavior:
#   1. Polls run_manifest.json every CHECK_INTERVAL seconds
#   2. Detects phase failures (status = "failed")
#   3. Logs failure event to outputs/<RUN_ID>/monitor.log
#   4. Waits MONITOR_TIMEOUT seconds (grace period for manual intervention)
#   5. If phase still failed after timeout → stops pod via RunPod API (kills billing)
#   6. If phase succeeds/restarts → resets counter
#
# Note: designed for NIGHTTIME RUNS ONLY. Daytime: run pipeline directly without monitoring.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=./lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Configuration from env or .env
MONITOR_ENABLED="${MONITOR_ENABLED:-1}"
MONITOR_TIMEOUT="${MONITOR_TIMEOUT:-600}"  # 10 minutes
MONITOR_CHECK_INTERVAL="${MONITOR_CHECK_INTERVAL:-30}"  # Poll every 30 seconds

# Bail if monitoring is explicitly disabled
if [[ "$MONITOR_ENABLED" == "0" ]]; then
    log_info "monitor_pipeline :: monitoring disabled (MONITOR_ENABLED=0)"
    exit 0
fi

# Resolve output base and manifest
OUTPUT_BASE=$(resolve_output_base)
MANIFEST="$OUTPUT_BASE/run_manifest.json"
MONITOR_LOG="$OUTPUT_BASE/monitor.log"
RUNPOD_ID_FILE="$REPO_ROOT/.runpod_id"

# Initialize monitor log
mkdir -p "$OUTPUT_BASE"
{
    echo "=========================================="
    echo "Pipeline Monitor Started"
    echo "Started at: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "Monitor timeout: $MONITOR_TIMEOUT seconds"
    echo "Check interval: $MONITOR_CHECK_INTERVAL seconds"
    echo "=========================================="
} >> "$MONITOR_LOG"

log_monitor() {
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$MONITOR_LOG"
}

get_runpod_id() {
    """Read RunPod ID from .runpod_id file or environment."""
    if [[ -f "$RUNPOD_ID_FILE" ]]; then
        cat "$RUNPOD_ID_FILE"
    else
        echo "${RUNPOD_ID:-}"
    fi
}

kill_runpod() {
    """Kill the RunPod pod via API (stops billing)."""
    local pod_id="$1"

    if [[ -z "$pod_id" ]]; then
        log_monitor "ERROR: Could not determine RunPod ID. Pod NOT killed."
        return 1
    fi

    log_monitor "KILLING POD: $pod_id (stopping billing immediately)"

    # Try gh API call to RunPod (requires RUNPOD_API_KEY)
    if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
        local response
        response=$(curl -s -X POST "https://api.runpod.io/graphql" \
            -H "Content-Type: application/json" \
            -H "api_key: $RUNPOD_API_KEY" \
            -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$pod_id\\\"}) { id status } }\"}" 2>&1 || echo "{}")

        if echo "$response" | grep -q '"id"'; then
            log_monitor "SUCCESS: Pod $pod_id stopped via RunPod API"
            return 0
        else
            log_monitor "WARNING: RunPod API stop may have failed. Response: $response"
        fi
    fi

    # Fallback: kill local pipeline process and exit
    log_monitor "Fallback: killing local pipeline process"
    pkill -f "scripts/pipeline.sh" || true
    sleep 5

    # Final attempt: SSH to pod and stop it (if SSH key available)
    if command -v ssh >/dev/null 2>&1 && [[ -n "${RUNPOD_SSH_KEY:-}" ]]; then
        log_monitor "Final attempt: SSH stop"
        ssh -i "$RUNPOD_SSH_KEY" root@api.runpod.io "pkill -f pipeline.sh" || true
    fi

    log_monitor "Pod termination sequence complete"
    return 0
}

detect_failure() {
    """Check if any phase in manifest has status = 'failed'."""
    if [[ ! -f "$MANIFEST" ]]; then
        return 1  # No manifest yet
    fi

    # Use Python to parse JSON safely
    python3 -c "
import json, sys
try:
    with open('$MANIFEST') as f:
        data = json.load(f)
    for phase in data.get('run', {}).get('phases', []):
        if phase.get('status') == 'failed':
            print(phase.get('name', 'unknown'))
            sys.exit(0)
    sys.exit(1)
except Exception as e:
    print(f'error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
}

# --- Main monitoring loop ---
failure_start_time=""
pod_id=$(get_runpod_id)

log_monitor "Starting monitoring (pod_id=$pod_id, timeout=$MONITOR_TIMEOUT seconds)"

while true; do
    sleep "$MONITOR_CHECK_INTERVAL"

    if ! [[ -f "$MANIFEST" ]]; then
        # Manifest not created yet; pipeline still initializing
        continue
    fi

    failed_phase=$(detect_failure) || {
        # No failure detected
        if [[ -n "$failure_start_time" ]]; then
            log_monitor "Phase recovered or restarted. Resetting failure counter."
            failure_start_time=""
        fi
        continue
    }

    # Failure detected
    if [[ -z "$failure_start_time" ]]; then
        # First detection of this failure
        failure_start_time=$(date +%s)
        log_monitor "FAILURE DETECTED: phase '$failed_phase' (status=failed). Starting $MONITOR_TIMEOUT second grace period."
    else
        # Already in grace period; check if timeout expired
        current_time=$(date +%s)
        elapsed=$((current_time - failure_start_time))

        if [[ $elapsed -ge $MONITOR_TIMEOUT ]]; then
            log_monitor "TIMEOUT EXPIRED ($elapsed >= $MONITOR_TIMEOUT seconds). Killing pod."
            kill_runpod "$pod_id"
            log_monitor "Exiting monitor (pod killed)"
            exit 0
        else
            remaining=$((MONITOR_TIMEOUT - elapsed))
            log_monitor "Phase still failed. Grace period: $remaining seconds remaining."
        fi
    fi
done
