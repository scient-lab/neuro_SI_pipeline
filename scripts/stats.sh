#!/usr/bin/env bash
# scripts/stats.sh — pipeline run-status table with optional live-refresh
# and system-resource bars (btop-style minimal).
#
# Separate from scripts/logs.sh (which keeps its own --summary / --details
# unchanged for backward compat) so this can grow features without
# disturbing the log-viewer.
#
# Usage:
#   ./scripts/stats.sh                          # one-shot summary
#   ./scripts/stats.sh -d                       # with nested steps
#   ./scripts/stats.sh --live                   # refresh every 5s
#   ./scripts/stats.sh --live --interval 2      # custom refresh
#   ./scripts/stats.sh --live --system          # add CPU/RAM/GPU bars
#   ./scripts/stats.sh -d --live --system       # all-in-one operator view
#   ./scripts/stats.sh --run <prefix>           # historical run
#   ./scripts/stats.sh --no-color               # disable ANSI colors
#
# Exit codes match logs.sh (0 OK, 1 error).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

RUN_ID=""
SHOW_DETAILS=0
LIVE=0
INTERVAL=5
SHOW_SYSTEM=0
USE_COLOR=auto

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--details)   SHOW_DETAILS=1; shift ;;
        --live)         LIVE=1; shift ;;
        --interval)     INTERVAL="$2"; shift 2 ;;
        --system)       SHOW_SYSTEM=1; shift ;;
        --no-color)     USE_COLOR=no; shift ;;
        --run)          RUN_ID="$2"; shift 2 ;;
        -h|--help)      sed -n '/^#/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)              echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

LOGS_BASE="$OUTPUT_BASE/logs"
MANIFEST="$OUTPUT_BASE/run_manifest.json"

# Resolve RUN_ID to a specific prefix if needed, mirroring logs.sh:
# default = latest under logs/, otherwise prefix-match.
# Use `if` blocks rather than `[[ ]] && { ... }` short-circuits because
# `set -e` propagates the non-zero from a failed `[[ ]]` test out of the
# function, killing the script on the success path.
resolve_run_id() {
    if [[ -z "$RUN_ID" ]]; then
        if [[ ! -d "$LOGS_BASE" ]]; then
            echo "No runs found in $LOGS_BASE/" >&2
            exit 1
        fi
        RUN_ID="$(ls -t "$LOGS_BASE" 2>/dev/null | head -1)"
        if [[ -z "$RUN_ID" ]]; then
            echo "No runs found in $LOGS_BASE/" >&2
            exit 1
        fi
    elif [[ ! -d "$LOGS_BASE/$RUN_ID" ]]; then
        local matched
        matched="$(ls -1 "$LOGS_BASE" 2>/dev/null | grep -E "^${RUN_ID}" | head -1 || true)"
        if [[ -z "$matched" ]]; then
            echo "No run matching: $RUN_ID" >&2
            exit 1
        fi
        RUN_ID="$matched"
    fi
    return 0
}

resolve_run_id

# Color helpers — emit ANSI only when stdout is a tty and USE_COLOR != no.
_color_supported() { [[ "$USE_COLOR" == "no" ]] && return 1; [[ -t 1 ]]; }
c_red()    { _color_supported && printf '\033[31m' || true; }
c_yel()    { _color_supported && printf '\033[33m' || true; }
c_grn()    { _color_supported && printf '\033[32m' || true; }
c_dim()    { _color_supported && printf '\033[2m'  || true; }
c_reset()  { _color_supported && printf '\033[0m'  || true; }

# bar PCT WIDTH — render a horizontal bar of `width` chars, `pct`% filled.
# Uses full-block / light-shade. Colors by threshold (green/yellow/red).
bar() {
    local pct=${1:-0} width=${2:-30}
    [[ "$pct" -lt 0 ]] && pct=0
    [[ "$pct" -gt 100 ]] && pct=100
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    # Color by threshold
    if   (( pct >= 86 )); then c_red
    elif (( pct >= 61 )); then c_yel
    else                       c_grn
    fi
    [[ "$filled" -gt 0 ]] && printf '%0.s█' $(seq 1 $filled)
    c_dim
    [[ "$empty"  -gt 0 ]] && printf '%0.s░' $(seq 1 $empty)
    c_reset
}

# render_metric LABEL PCT DETAIL — one line of system stats
render_metric() {
    local label=$1 pct=$2 detail=${3:-}
    printf "  %-8s ▕" "$label"
    bar "$pct" 30
    printf "▏ %3d%%   %s\n" "$pct" "$detail"
}

# Cheap CPU% via /proc/stat (no busybox-vs-procps top quirks).
# Takes two samples 250ms apart for a meaningful instantaneous value.
sample_cpu_pct() {
    local s1 s2 idle1 idle2 total1 total2 di dt
    s1=$(grep '^cpu ' /proc/stat 2>/dev/null) || { echo 0; return; }
    sleep 0.25
    s2=$(grep '^cpu ' /proc/stat 2>/dev/null) || { echo 0; return; }
    # cpu user nice system idle iowait irq softirq steal …
    read -r _ u1 n1 sys1 idle1 io1 irq1 sirq1 _ <<< "$s1"
    read -r _ u2 n2 sys2 idle2 io2 irq2 sirq2 _ <<< "$s2"
    total1=$(( u1 + n1 + sys1 + idle1 + io1 + irq1 + sirq1 ))
    total2=$(( u2 + n2 + sys2 + idle2 + io2 + irq2 + sirq2 ))
    dt=$(( total2 - total1 ))
    di=$(( idle2 - idle1 ))
    [[ "$dt" -le 0 ]] && { echo 0; return; }
    echo $(( 100 * (dt - di) / dt ))
}

# MEM usage as percent + "used / total GB" detail.
sample_mem() {
    awk '/^Mem:/ {
            pct = 100*($3/$2)
            printf "%d %.1f %.1f\n", pct, $3/1024, $2/1024
         }' <(free -m 2>/dev/null) 2>/dev/null \
        || echo "0 0.0 0.0"
}

# GPU info (per-device): name util vram_used vram_total temp
# Outputs one line per GPU; empty if nvidia-smi missing or no device.
sample_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/, /,/g'
}

render_system() {
    echo "── system ─────────────────────────────────────────────────"
    local cpu_pct mem_line mem_pct mem_used mem_total
    cpu_pct=$(sample_cpu_pct)
    render_metric "CPU" "$cpu_pct"

    mem_line="$(sample_mem)"
    read -r mem_pct mem_used mem_total <<< "$mem_line"
    render_metric "RAM" "$mem_pct" "${mem_used} / ${mem_total} GB"

    local gpu_idx=0
    while IFS=, read -r name util used total temp; do
        # Trim possible leading space
        name="${name# }"; util="${util# }"; used="${used# }"; total="${total# }"; temp="${temp# }"
        [[ -z "$name" ]] && continue
        local vram_pct=0
        [[ "$total" -gt 0 ]] && vram_pct=$(( 100 * used / total ))
        local vram_gb_used vram_gb_total
        vram_gb_used=$(awk -v v="$used"  'BEGIN{printf "%.1f", v/1024}')
        vram_gb_total=$(awk -v v="$total" 'BEGIN{printf "%.1f", v/1024}')
        render_metric "GPU $gpu_idx" "$util" "$name"
        render_metric "VRAM"         "$vram_pct" "${vram_gb_used} / ${vram_gb_total} GB    ${temp}°C"
        gpu_idx=$(( gpu_idx + 1 ))
    done < <(sample_gpu)

    if [[ "$gpu_idx" -eq 0 ]]; then
        printf "  %s(no NVIDIA GPU detected via nvidia-smi)%s\n" "$(c_dim)" "$(c_reset)"
    fi
}

# Single render pass — used by both one-shot and --live modes.
render_one_frame() {
    if [[ ! -f "$MANIFEST" ]]; then
        echo "No manifest at $MANIFEST" >&2
        return 1
    fi
    local extra_args=( --manifest "$MANIFEST" --run-id "$RUN_ID" )
    [[ "$SHOW_DETAILS" -eq 1 ]] && extra_args+=( --details )
    python3 "$SCRIPT_DIR/lib/stats_render.py" "${extra_args[@]}"
    if [[ "$SHOW_SYSTEM" -eq 1 ]]; then
        echo
        render_system
    fi
}

if [[ "$LIVE" -eq 1 ]]; then
    # Modern command-style quit: 'q' or 'Q' from the keyboard, or Ctrl-C.
    # The non-blocking `read -t $INTERVAL -s -n 1` doubles as the inter-frame
    # delay so we don't sleep AND read separately (which would make the keystroke
    # response laggy on slow intervals).
    # If stdin isn't a TTY (piped / cron), skip keypress handling and just sleep.
    trap 'stty echo 2>/dev/null; echo; echo "(stopped)"; exit 0' INT TERM
    if [[ -t 0 ]]; then
        # Disable terminal echo so the q-press doesn't leak into the next frame.
        stty -echo 2>/dev/null || true
        while true; do
            clear
            printf "  refresh: %ds   press q to quit\n\n" "$INTERVAL"
            render_one_frame || { stty echo 2>/dev/null; exit 1; }
            key=""
            read -t "$INTERVAL" -s -n 1 key 2>/dev/null || true
            case "$key" in
                q|Q) stty echo 2>/dev/null; echo; echo "(stopped)"; exit 0 ;;
            esac
        done
    else
        while true; do
            clear
            printf "  refresh: %ds   (no tty — Ctrl-C to quit)\n\n" "$INTERVAL"
            render_one_frame || exit 1
            sleep "$INTERVAL"
        done
    fi
else
    render_one_frame
fi
