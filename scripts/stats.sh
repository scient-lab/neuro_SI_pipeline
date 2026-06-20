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

# Detect once: is stdout a tty AND colors enabled? Used by color codes AND
# by the erase-to-end-of-line code (\033[K) which would print literal "[K"
# on non-tty consumers (`stats.sh | head`).
IS_TTY=0
EOL=""
if [[ -t 1 ]]; then
    IS_TTY=1
    EOL=$'\033[K'
fi
USE_ANSI=0
if [[ "$IS_TTY" -eq 1 && "$USE_COLOR" != "no" ]]; then
    USE_ANSI=1
fi

_color_supported() { [[ "$USE_ANSI" -eq 1 ]]; }
c_bar()    { _color_supported && printf '\033[94m' || true; }   # bright blue
c_dim()    { _color_supported && printf '\033[2m'  || true; }
c_reset()  { _color_supported && printf '\033[0m'  || true; }

# bar PCT WIDTH — render a horizontal bar of `width` chars, `pct`% filled.
# Single neutral color (bright blue) for the filled portion. We dropped the
# old green/yellow/red thresholds: during training, 88% GPU and 91% VRAM
# are EXPECTED (you're using the resource you paid for), not warnings.
# Mirrors RunPod's telemetry-panel aesthetic.
bar() {
    local pct=${1:-0} width=${2:-30}
    [[ "$pct" -lt 0 ]] && pct=0
    [[ "$pct" -gt 100 ]] && pct=100
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    c_bar
    [[ "$filled" -gt 0 ]] && printf '%0.s█' $(seq 1 $filled)
    c_dim
    [[ "$empty"  -gt 0 ]] && printf '%0.s░' $(seq 1 $empty)
    c_reset
}

# Layout constants. Bar width + separator width + history-buffer length.
# Phase table in stats_render.py is 96 display columns wide:
#   2 indent + 24 PHASE + 1 + 13 STATUS + 1 + 10 STARTED + 1 + 10 FINISHED
#   + 1 + 11 DURATION + 1 + 14 ETA + 1 + 6 STEPS = 96
# Match that here so the system section visually extends to the same right
# edge as STEPS instead of trailing short like before.
TABLE_W=96
BAR_W=50         # was 20 — wider gauge gives finer visual resolution
HIST_W=20        # last 20 samples per metric in --live mode
declare -a HIST_cpu HIST_ram HIST_gpu HIST_vram

# push_hist <buffer-name> <value>
push_hist() {
    local name="HIST_$1" val=$2
    declare -n arr="$name"
    arr+=("$val")
    while [[ ${#arr[@]} -gt $HIST_W ]]; do
        arr=("${arr[@]:1}")
    done
}

# spark <buffer-name> — render a sparkline (last $HIST_W samples, right-aligned).
# Unfilled positions on the left padded with spaces so the chart width is
# stable across frames (avoids visual jump as the buffer fills).
spark() {
    local name="HIST_$1"
    declare -n arr="$name"
    local levels=( '▁' '▂' '▃' '▄' '▅' '▆' '▇' '█' )
    local n=${#arr[@]}
    local pad=$(( HIST_W - n ))
    c_dim
    [[ $pad -gt 0 ]] && printf ' %.0s' $(seq 1 $pad)
    c_bar
    local s lvl
    for s in "${arr[@]}"; do
        lvl=$(( s * 7 / 100 ))
        [[ $lvl -lt 0 ]] && lvl=0
        [[ $lvl -gt 7 ]] && lvl=7
        printf '%s' "${levels[$lvl]}"
    done
    c_reset
}

# render_metric LABEL PCT HIST_KEY DETAIL — one line of system stats with
# bar + sparkline. HIST_KEY can be empty to skip the sparkline (e.g. on
# the one-shot, non-live invocation where we have no history).
render_metric() {
    local label=$1 pct=$2 hist_key=${3:-} detail=${4:-}
    printf "  %-8s ▕" "$label"
    bar "$pct" "$BAR_W"
    printf "▏ %3d%%  " "$pct"
    if [[ -n "$hist_key" ]]; then
        spark "$hist_key"
        printf "  "
    fi
    # ${EOL} = erase-to-EOL (or empty on non-tty). Cleans up tail-garbage
    # from a longer previous frame in --live mode.
    printf "%s%s\n" "$detail" "$EOL"
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
    # Pass per-metric history-buffer keys only when in --live mode. The
    # one-shot invocation has empty buffers (single sample at most), which
    # would render as a single bar at the right edge — visually noisy and
    # not informative. render_metric treats an empty key as "skip sparkline".
    # Must initialize unconditionally for `set -u` compatibility.
    local hkey_cpu="" hkey_ram="" hkey_gpu="" hkey_vram=""
    if [[ "$LIVE" -eq 1 ]]; then
        hkey_cpu=cpu; hkey_ram=ram; hkey_gpu=gpu; hkey_vram=vram
    fi

    # Separator extends to TABLE_W so it visually reaches the same right
    # edge as the phase table's STEPS column above. "── system " is 10
    # display columns; pad to TABLE_W with box-drawing horizontal lines.
    local _sep_label="── system "
    local _sep_dashes=$(( TABLE_W - 10 ))
    printf "%s" "$_sep_label"
    [[ "$_sep_dashes" -gt 0 ]] && printf '─%.0s' $(seq 1 "$_sep_dashes")
    printf "%s\n" "$EOL"

    local cpu_pct mem_line mem_pct mem_used mem_total
    cpu_pct=$(sample_cpu_pct)
    push_hist cpu "$cpu_pct"
    render_metric "CPU" "$cpu_pct" "$hkey_cpu"

    mem_line="$(sample_mem)"
    read -r mem_pct mem_used mem_total <<< "$mem_line"
    push_hist ram "$mem_pct"
    render_metric "RAM" "$mem_pct" "$hkey_ram" "${mem_used} / ${mem_total} GB"

    # GPU sparkline buffers only track the FIRST GPU for now (multi-GPU
    # support adds buffer indexing; defer until needed).
    local gpu_idx=0
    while IFS=, read -r name util used total temp; do
        name="${name# }"; util="${util# }"; used="${used# }"; total="${total# }"; temp="${temp# }"
        [[ -z "$name" ]] && continue
        local vram_pct=0
        [[ "$total" -gt 0 ]] && vram_pct=$(( 100 * used / total ))
        local vram_gb_used vram_gb_total
        vram_gb_used=$(awk -v v="$used"  'BEGIN{printf "%.1f", v/1024}')
        vram_gb_total=$(awk -v v="$total" 'BEGIN{printf "%.1f", v/1024}')
        if [[ "$gpu_idx" -eq 0 ]]; then
            push_hist gpu  "$util"
            push_hist vram "$vram_pct"
        fi
        render_metric "GPU $gpu_idx" "$util"     "${gpu_idx:+}${hkey_gpu}"  "$name"
        render_metric "VRAM"         "$vram_pct" "${gpu_idx:+}${hkey_vram}" "${vram_gb_used} / ${vram_gb_total} GB    ${temp}°C"
        # If multi-GPU later, suppress sparkline on idx > 0 by passing "" as
        # hist_key. For now single-GPU case is the only path exercised.
        gpu_idx=$(( gpu_idx + 1 ))
    done < <(sample_gpu)

    if [[ "$gpu_idx" -eq 0 ]]; then
        printf "  %s(no NVIDIA GPU detected via nvidia-smi)%s\033[K\n" "$(c_dim)" "$(c_reset)"
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
    # Anti-flicker via in-place cursor positioning (btop-style). On entry:
    # one-time setup hides the cursor and clears the screen. Each subsequent
    # frame moves the cursor home (\033[H) WITHOUT clearing — the new content
    # overwrites the old in place. Each rendered line ends in \033[K (erase
    # to end-of-line) so a shorter line doesn't leave tail-garbage. After
    # the last line, \033[J erases anything below from a previous taller
    # frame. The `clear` call (full screen wipe each frame) is what caused
    # the 5s flicker the operator noticed; removing it.
    #
    # Modern command-style quit: 'q' / 'Q' or Ctrl-C. Non-blocking
    # `read -t $INTERVAL -s -n 1` doubles as the inter-frame delay so the
    # keystroke is responsive immediately (no waiting up to INTERVAL).
    # If stdin isn't a TTY (piped / cron), skip keypress + cursor tricks
    # and just sleep — terminal-control codes can confuse non-interactive
    # consumers.
    _restore_terminal() {
        stty echo 2>/dev/null || true
        # Show cursor, then clear screen and home cursor so the next prompt
        # lands cleanly without leftover stats content.
        printf '\033[?25h\033[2J\033[H'
    }

    trap '_restore_terminal; echo "(stopped)"; exit 0' INT TERM

    if [[ -t 0 ]] && [[ -t 1 ]]; then
        stty -echo 2>/dev/null || true
        # One-time setup: hide cursor, clear screen, home.
        printf '\033[?25l\033[2J\033[H'
        while true; do
            # Move cursor home WITHOUT clearing — new content overwrites old.
            printf '\033[H'
            printf "  refresh: %ds   press q to quit\033[K\n\033[K\n" "$INTERVAL"
            render_one_frame || { _restore_terminal; exit 1; }
            # Erase anything below current cursor (cleans up if previous
            # frame was taller, e.g. extra step rows appeared/disappeared).
            printf '\033[J'
            key=""
            read -t "$INTERVAL" -s -n 1 key 2>/dev/null || true
            case "$key" in
                q|Q) _restore_terminal; echo "(stopped)"; exit 0 ;;
            esac
        done
    else
        # No TTY — fall back to plain sleep-redraw without cursor tricks.
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
