#!/bin/bash
# scripts/monitor.sh — pod babysitter: health telemetry + optional auto-kill.
#
# Supersedes monitor_pipeline.sh (which only watched manifest failures).
# Each tick it calls lib/health_sample.py (appends health_system.csv +
# health_gpu.csv under $OUTPUT_BASE/health/, rides the existing S3 sync) and
# then applies stateful kill decisions.
#
# Examples:
#   # health logging only (default — never kills); attended/daytime runs:
#   ./scripts/monitor.sh
#
#   # unattended/nighttime — kill on failure/disk/hang + hard 8h ceiling:
#   ./scripts/monitor.sh --kill-on-fail --max-runtime 8h --idle-min 15
#
#   # hard deadline only, but still debug failures by hand (no fail-kill):
#   ./scripts/monitor.sh --max-runtime 6h
#
#   # how bootstrap starts it — detached so it outlives the launching shell:
#   setsid nohup ./scripts/monitor.sh --kill-on-fail --max-runtime 8h \
#       >"$OUTPUT_BASE/health/monitor.out" 2>&1 &
#
#   # one-shot health sample, no loop (just the CSV writer):
#   python3 scripts/lib/health_sample.py --manifest outputs/run_manifest.json \
#       --system-csv outputs/health/health_system.csv \
#       --gpu-csv    outputs/health/health_gpu.csv
#
# Config — each knob takes a FLAG or its ENV var (flag wins). Defaults shown.
# Health-CSV logging is ALWAYS on, independent of every kill knob below.
#
#   flag            env var               default  meaning
#   --interval N    MONITOR_INTERVAL      60       sample/poll period (seconds)
#   --kill-on-fail  MONITOR_KILL_ON_FAIL  0 (off)  MASTER switch for the fail/disk/hang
#                                                  kills below; log-only when off
#   --kill-on-complete  MONITOR_KILL_ON_COMPLETE  0 (off)  pipeline reached 'completed'
#                                                  -> final sync + stop pod (cost control).
#                                                  Independent of --kill-on-fail
#   --fail-grace N  MONITOR_FAIL_GRACE    600      seconds a 'failed' phase persists
#                                                  before kill (MONITOR_TIMEOUT = alias)
#   --max-runtime D MONITOR_MAX_RUNTIME   0 (off)  hard deadline -> final sync + stop pod;
#                                                  8h|480m|28800|0. Fires INDEPENDENTLY of
#                                                  --kill-on-fail (its own opt-in)
#   --disk-crit PCT MONITOR_DISK_CRIT     95       disk% >= PCT -> kill (needs --kill-on-fail)
#   --idle-min N    MONITOR_IDLE_MIN      0 (off)  liveness-stall minutes -> hang kill (needs
#                                                  --kill-on-fail). GPU util for compute
#                                                  phases; net rx rate for curriculum
#   (none)          MONITOR_ENABLED       1        set 0 to disable the monitor entirely
#
# The kill toggles accept a bare flag (= on) OR an explicit 0|1 to mirror the
# env var, so all of these are equivalent: --kill-on-complete | --kill-on-complete 1
# | MONITOR_KILL_ON_COMPLETE=1
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# Make secrets (RUNPOD_API_KEY, S3_URI, …) available when launched outside
# pipeline.sh (bootstrap / manual) — without clobbering already-exported vars.
if [[ -z "${RUNPOD_API_KEY:-}" && -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

# --- config: each knob = env var (default), overridable by the matching flag ---
INTERVAL="${MONITOR_INTERVAL:-60}"
KILL_ON_FAIL="${MONITOR_KILL_ON_FAIL:-0}"
KILL_ON_COMPLETE="${MONITOR_KILL_ON_COMPLETE:-0}"
FAIL_GRACE="${MONITOR_FAIL_GRACE:-${MONITOR_TIMEOUT:-600}}"   # MONITOR_TIMEOUT = alias
MAX_RUNTIME_RAW="${MONITOR_MAX_RUNTIME:-0}"
DISK_CRIT="${MONITOR_DISK_CRIT:-95}"
IDLE_MIN="${MONITOR_IDLE_MIN:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)     INTERVAL="$2"; shift 2 ;;
        # boolean flags: bare = on, or take an explicit 0|1 to match the env var
        --kill-on-fail)     if [[ "${2:-}" =~ ^[01]$ ]]; then KILL_ON_FAIL="$2"; shift 2; else KILL_ON_FAIL=1; shift; fi ;;
        --kill-on-complete) if [[ "${2:-}" =~ ^[01]$ ]]; then KILL_ON_COMPLETE="$2"; shift 2; else KILL_ON_COMPLETE=1; shift; fi ;;
        --fail-grace)   FAIL_GRACE="$2"; shift 2 ;;
        --max-runtime)  MAX_RUNTIME_RAW="$2"; shift 2 ;;
        --disk-crit)    DISK_CRIT="$2"; shift 2 ;;
        --idle-min)     IDLE_MIN="$2"; shift 2 ;;
        *) echo "monitor.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

[[ "${MONITOR_ENABLED:-1}" == "0" ]] && { echo "monitor.sh: disabled (MONITOR_ENABLED=0)"; exit 0; }

# Parse a duration (8h | 480m | 28800s | 28800) into seconds.
parse_duration() {
    local d="$1"
    case "$d" in
        *h) echo $(( ${d%h} * 3600 )) ;;
        *m) echo $(( ${d%m} * 60 )) ;;
        *s) echo "${d%s}" ;;
        *)  echo "$d" ;;
    esac
}
MAX_RUNTIME=$(parse_duration "$MAX_RUNTIME_RAW")

OUTPUT_BASE=$(resolve_output_base)
HEALTH_DIR="$OUTPUT_BASE/health"
SYS_CSV="$HEALTH_DIR/health_system.csv"
GPU_CSV="$HEALTH_DIR/health_gpu.csv"
MANIFEST="$OUTPUT_BASE/run_manifest.json"
LOG="$HEALTH_DIR/monitor.log"
SAMPLER="$SCRIPT_DIR/lib/health_sample.py"
mkdir -p "$HEALTH_DIR"

log_m() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG"; }

resolve_pod_id() {
    if [[ -n "${RUNPOD_POD_ID:-}" ]]; then echo "$RUNPOD_POD_ID"
    elif [[ -f "$REPO_ROOT/.runpod_id" ]]; then cat "$REPO_ROOT/.runpod_id"
    else echo "${RUNPOD_ID:-}"; fi
}
POD_ID=$(resolve_pod_id)

final_sync() {
    [[ -x "$SCRIPT_DIR/s3_sync.sh" ]] && { log_m "final S3 sync before stop"; "$SCRIPT_DIR/s3_sync.sh" >>"$LOG" 2>&1 || log_m "WARN: final sync failed"; }
}

stop_pod() {
    local reason="$1"
    log_m "STOPPING POD ($reason) pod_id=${POD_ID:-<unknown>}"
    final_sync
    if [[ -n "$POD_ID" && -n "${RUNPOD_API_KEY:-}" ]]; then
        # RunPod REST API v1 (same Bearer auth as scripts/runpod/launch.sh).
        # POST /v1/pods/{podId}/stop -> HTTP 200 on success; preserves the volume.
        # (DELETE /v1/pods/{podId} would terminate instead — we want stop.)
        local resp code
        resp=$(curl -s -w '\n%{http_code}' -X POST \
            "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" 2>/dev/null || printf '\n000')
        code="${resp##*$'\n'}"; resp="${resp%$'\n'*}"
        if [[ "$code" == "200" ]]; then log_m "SUCCESS: pod $POD_ID stopped (HTTP 200)"; return 0; fi
        log_m "WARN: RunPod stop returned HTTP $code (pod may still be running): $resp"
    else
        log_m "WARN: no POD_ID / RUNPOD_API_KEY — cannot stop pod via API"
    fi
    pkill -f "scripts/pipeline.sh" 2>/dev/null || true
    return 0
}

# parse `key=value ...` summary from the sampler into assoc array S
declare -A S
read_summary() {
    local line="$1" kv k v
    S=()
    for kv in $line; do k="${kv%%=*}"; v="${kv#*=}"; S["$k"]="$v"; done
}

log_m "monitor start: interval=${INTERVAL}s kill_on_fail=$KILL_ON_FAIL kill_on_complete=$KILL_ON_COMPLETE fail_grace=${FAIL_GRACE}s max_runtime=${MAX_RUNTIME}s disk_crit=${DISK_CRIT}% idle_min=${IDLE_MIN}m pod=${POD_ID:-<none>}"

START=$(date +%s)
fail_since=""
idle_since=""

while true; do
    sleep "$INTERVAL"
    now=$(date +%s)

    # --- sample + append CSVs; capture the summary line ---
    summary=$(python3 "$SAMPLER" --manifest "$MANIFEST" \
        --system-csv "$SYS_CSV" --gpu-csv "$GPU_CSV" \
        --pod-id "$POD_ID" --disk-crit "$DISK_CRIT" 2>>"$LOG") || { log_m "WARN: sampler error"; continue; }
    read_summary "$summary"

    # --- max-runtime deadline (its own opt-in: only when set) ---
    if [[ "$MAX_RUNTIME" -gt 0 && $((now - START)) -ge "$MAX_RUNTIME" ]]; then
        stop_pod "max-runtime ${MAX_RUNTIME}s reached"; exit 0
    fi

    # --- pipeline completed successfully -> optional shutdown (cost control) ---
    # Independent of --kill-on-fail. finalize is gated to the last phase, so a
    # partial --phase run stays 'running' and won't trip this.
    if [[ "$KILL_ON_COMPLETE" -eq 1 && "${S[pipeline_status]:-}" == "completed" ]]; then
        stop_pod "pipeline completed"; exit 0
    fi

    # everything below only KILLS when --kill-on-fail; otherwise it just logs.
    # --- disk critical ---
    dmax="${S[disk_max_pct]:-}"
    if [[ -n "$dmax" && "$dmax" -ge "$DISK_CRIT" ]]; then
        log_m "CRITICAL: disk ${dmax}% >= ${DISK_CRIT}%"
        [[ "$KILL_ON_FAIL" -eq 1 ]] && { stop_pod "disk ${dmax}%"; exit 0; }
    fi

    # --- pipeline failure + grace ---
    if [[ "${S[pipeline_status]:-}" == "failed" ]]; then
        [[ -z "$fail_since" ]] && { fail_since=$now; log_m "FAILURE: phase '${S[failed_phase]:-?}' — grace ${FAIL_GRACE}s"; }
        if [[ $((now - fail_since)) -ge "$FAIL_GRACE" ]]; then
            log_m "failure persisted ${FAIL_GRACE}s"
            [[ "$KILL_ON_FAIL" -eq 1 ]] && { stop_pod "pipeline failed: ${S[failed_phase]:-?}"; exit 0; }
            fail_since=$now   # re-arm grace if not killing, avoid spamming
        fi
    elif [[ -n "$fail_since" ]]; then
        log_m "failure cleared"; fail_since=""
    fi

    # --- phase-aware liveness / hang detection ---
    if [[ "$IDLE_MIN" -gt 0 && "${S[pipeline_status]:-}" == "running" ]]; then
        phase="${S[phase]:-}"; idle=0
        if [[ "$phase" == "curriculum" ]]; then
            # API-bound: GPU idle is EXPECTED — liveness is network rx rate.
            r="${S[net_rx_mbps]:-}"
            [[ -n "$r" ]] && awk "BEGIN{exit !($r < 0.05)}" && idle=1
        elif [[ -n "$phase" ]]; then
            # compute phase: liveness is GPU util.
            u="${S[gpu_util_max]:-}"
            [[ -n "$u" ]] && awk "BEGIN{exit !($u < 1)}" && idle=1
        fi
        if [[ "$idle" -eq 1 ]]; then
            [[ -z "$idle_since" ]] && { idle_since=$now; log_m "liveness stall start (phase=$phase)"; }
            if [[ $(( (now - idle_since) / 60 )) -ge "$IDLE_MIN" ]]; then
                log_m "HANG: phase '$phase' idle ${IDLE_MIN}m"
                [[ "$KILL_ON_FAIL" -eq 1 ]] && { stop_pod "hang: $phase idle ${IDLE_MIN}m"; exit 0; }
                idle_since=$now
            fi
        elif [[ -n "$idle_since" ]]; then
            idle_since=""
        fi
    fi
done
