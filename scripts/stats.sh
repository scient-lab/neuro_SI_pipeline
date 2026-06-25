#!/usr/bin/env bash
# scripts/stats.sh — pipeline run-status table with optional live-refresh
# and resource-utilization bars (btop-style minimal).
#
# Separate from scripts/logs.sh (which keeps its own --summary / --details
# unchanged for backward compat) so this can grow features without
# disturbing the log-viewer.
#
# Flags:
#   --steps      / -s    Nested step rows under each phase
#   --live       / -l    Auto-refresh every --interval seconds (default 5)
#   --resources  / -r    Add CPU/RAM/GPU/VRAM gauges
#   --interval N         Live refresh rate (seconds; min 1)
#   --run <prefix>       Specific historical run (default: latest)
#   --no-color           Disable ANSI colors
#
# Usage:
#   ./scripts/stats.sh                              # one-shot summary (phases only)
#   ./scripts/stats.sh --steps                      # nested step rows under each phase
#   ./scripts/stats.sh --live                       # refresh every 5s
#   ./scripts/stats.sh --live --interval 2          # custom refresh
#   ./scripts/stats.sh --live --resources           # add CPU/RAM/GPU/VRAM gauges
#   ./scripts/stats.sh --steps --live --resources   # all-in-one operator view
#   ./scripts/stats.sh -s -l -r                     # same, short-form
#   ./scripts/stats.sh --run <prefix>               # historical run
#   ./scripts/stats.sh --no-color                   # disable ANSI colors
#
# Exit codes match logs.sh (0 OK, 1 error).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

RUN_ID=""
SHOW_STEPS=0
LIVE=0
INTERVAL=5
SHOW_SYSTEM=0
USE_COLOR=auto

while [[ $# -gt 0 ]]; do
    case "$1" in
        # --steps / -s: nested per-step rows under each phase
        -s|--steps)        SHOW_STEPS=1; shift ;;
        # Legacy: -d / --details was renamed to --steps when the old short
        # flag (-d) became ambiguous with operators thinking it meant
        # "delete." Kept as a warning-only alias so prior automation
        # doesn't silently break.
        -d|--details)   echo "stats.sh: '-d' / '--details' renamed to '--steps' (the old short flag was ambiguous with 'delete')." >&2
                        SHOW_STEPS=1; shift ;;
        # --live / -l: auto-refresh every --interval seconds
        -l|--live)         LIVE=1; shift ;;
        --interval)     INTERVAL="$2"; shift 2 ;;
        # --resources / -r: CPU/RAM/GPU/VRAM gauges. Was --system (deprecated).
        -r|--resources) SHOW_SYSTEM=1; shift ;;
        --system)       echo "stats.sh: '--system' renamed to '--resources' (more specific: CPU/RAM/GPU/VRAM gauges, not e.g. hostname/kernel)." >&2
                        SHOW_SYSTEM=1; shift ;;
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
# Phase table in stats_render.py is 105 display columns wide:
#   2 indent + 24 PHASE + 1 + 13 STATUS + 1 + 8 OUTCOME + 1 + 10 STARTED
#   + 1 + 10 FINISHED + 1 + 11 DURATION + 1 + 14 ETA + 1 + 6 STEPS = 105
# Match that here so the system section visually extends to the same right
# edge as STEPS instead of trailing short like before.
TABLE_W=105
BAR_W=50         # was 20 — wider gauge gives finer visual resolution
HIST_W=80        # last 80 samples per metric in --live mode (~6.5 min at 5s refresh)
SPARK_CHARS=$(( HIST_W / 2 ))   # braille packs 2 samples per char → 40 chars wide
declare -a HIST_cpu HIST_ram HIST_gpu HIST_vram HIST_disk

# Disk usage changes slowly and `df` is comparatively expensive — sample at
# 1/DISK_INTERVAL_MULT the rate of other metrics. With --interval=5s and
# MULT=10, disk refreshes every 50s. Cached between samples.
DISK_INTERVAL_MULT=10
declare -i DISK_TICK=0
DISK_CACHED=""

# push_hist <buffer-name> <value>
push_hist() {
    local name="HIST_$1" val=$2
    declare -n arr="$name"
    arr+=("$val")
    while [[ ${#arr[@]} -gt $HIST_W ]]; do
        arr=("${arr[@]:1}")
    done
}

# spark <buffer-name> — render a sparkline (last $HIST_W samples).
# Uses braille characters (2 dot columns x 4 dot rows per char) so each
# char encodes 2 time samples × 5 levels (0-4 dots, bottom-up filled).
# Result is denser than single-row block sparkline (▁▂▃▄▅▆▇█) while
# matching btop's visual style. Wider HIST_W=60 = 30 braille chars wide.
#
# Braille dot bits per cell (Unicode U+2800 + bitmask):
#   left col:  dot 7=0x40 (bottom), 3=0x04, 2=0x02, 1=0x01 (top)
#   right col: dot 8=0x80 (bottom), 6=0x20, 5=0x10, 4=0x08 (top)
spark() {
    local name="HIST_$1"
    declare -n arr="$name"
    # Pad/truncate to exactly HIST_W samples (oldest on left, newest right).
    # Empty positions on the left get rendered as ⠀ (blank braille).
    local n=${#arr[@]}
    local pad=$(( HIST_W - n ))
    local -a samples=()
    if (( pad > 0 )); then
        local i
        for ((i=0; i<pad; i++)); do samples+=(0); done
    fi
    samples+=("${arr[@]}")
    [[ ${#samples[@]} -gt $HIST_W ]] && samples=("${samples[@]: -HIST_W}")
    # Delegate Unicode codepoint math to python (bash's printf '\u' is
    # unreliable across versions and `printf '\xNN'` would need UTF-8 byte
    # encoding per codepoint). One python call per metric per frame ≈ 50ms.
    c_bar
    python3 -c "
import sys
samples = [int(x) for x in sys.argv[1:]]
# Map % (0-100) to dot count (0-4). Boundaries: 0/13/38/63/88/100.
def d(p): return min(4, max(0, (p + 12) // 25))
LEFT  = [0, 0x40, 0x44, 0x46, 0x47]
RIGHT = [0, 0x80, 0xA0, 0xB0, 0xB8]
out = []
for i in range(0, len(samples), 2):
    l = d(samples[i])
    r = d(samples[i+1]) if i+1 < len(samples) else 0
    out.append(chr(0x2800 + (LEFT[l] | RIGHT[r])))
sys.stdout.write(''.join(out))
" "${samples[@]}"
    c_reset
}

# render_metric LABEL PCT HIST_KEY DETAIL — one line of system stats.
#
# Layout: LABEL  [SPARK]  DETAIL                              PCT
#   - LABEL is left-padded to 8 cols
#   - SPARK is only rendered in --live mode (hist_key non-empty); 40 cols
#   - DETAIL is left-aligned, padded to fill the remaining width
#   - PCT is right-aligned at TABLE_W (column 96) so single-digit (9%) and
#     triple-digit (100%) values share the same right edge.
#
# This reorder (pct from left to right) eliminates the per-row column drift
# caused by 1-vs-3-char pct values pushing the sparkline around. DETAIL
# becomes the visual anchor; PCT becomes a glanceable summary at row end.
#
# UTF-8 NOTE: bash `printf %-Ns` pads by BYTE count, not display columns.
# A detail like "RTX A6000 · 74°C" has 16 visible chars but 18 bytes (·
# and ° are 2 bytes each in UTF-8), causing pct to render 2 cols too far
# left.
#
# We use python for the count rather than awk because Debian/Ubuntu's
# default `awk` is mawk, which counts bytes regardless of locale — so
# `LC_ALL=C.UTF-8 awk length()` only works under gawk and silently
# regresses on RunPod-style Ubuntu images. python's `len()` always
# counts Unicode codepoints (= display columns for non-CJK strings),
# making the result deterministic across distros.
_pad_display() {
    local str="$1" w="$2"
    python3 -c "import sys
s = sys.argv[1]
w = int(sys.argv[2])
sys.stdout.write(s + ' ' * max(0, w - len(s)))" "$str" "$w"
}

render_metric() {
    local label=$1 pct=$2 hist_key=${3:-} detail=${4:-}
    # Track current column position so we can size the detail field to
    # right-align PCT at TABLE_W regardless of whether sparkline is shown.
    local prefix_w=$(( 2 + 8 + 1 ))   # "  LABEL(8) " = 11 cols
    printf "  %-8s " "$label"
    if [[ -n "$hist_key" ]]; then
        spark "$hist_key"
        printf "  "
        prefix_w=$(( prefix_w + SPARK_CHARS + 2 ))
    fi
    # Detail field width: fill up to (TABLE_W - 5), leaving " PCT%" suffix.
    local detail_w=$(( TABLE_W - prefix_w - 5 ))
    [[ "$detail_w" -lt 1 ]] && detail_w=1
    # ${EOL} = erase-to-EOL (or empty on non-tty). Cleans up tail-garbage
    # from a longer previous frame in --live mode.
    printf "%s %3d%%%s\n" "$(_pad_display "$detail" "$detail_w")" "$pct" "$EOL"
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

# Disk usage on /  (or $DISK_MOUNT if set). Outputs:
#   <pct> <used_gb> <total_gb> <mountpoint>
# `df -B 1G` reports in 1 GB blocks across both GNU and busybox df.
sample_disk() {
    local mnt="${DISK_MOUNT:-/}"
    df -B 1073741824 "$mnt" 2>/dev/null | awk -v m="$mnt" '
        NR==2 {
            gsub("%","",$5)
            printf "%d %d %d %s\n", $5, $3, $2, m
        }'
}

# maybe_sample_disk — invoke sample_disk only every DISK_INTERVAL_MULT
# frames; otherwise return the cached line. Counter persists across
# render_system calls (declared at top with `declare -i DISK_TICK=0`).
maybe_sample_disk() {
    if (( DISK_TICK == 0 )); then
        DISK_CACHED="$(sample_disk)"
    fi
    DISK_TICK=$(( (DISK_TICK + 1) % DISK_INTERVAL_MULT ))
    echo "$DISK_CACHED"
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
    local hkey_cpu="" hkey_ram="" hkey_gpu="" hkey_vram="" hkey_disk=""
    if [[ "$LIVE" -eq 1 ]]; then
        hkey_cpu=cpu; hkey_ram=ram; hkey_gpu=gpu; hkey_vram=vram; hkey_disk=disk
    fi

    # Separator extends to TABLE_W so it visually reaches the same right
    # edge as the phase table's STEPS column above. "── system " is 10
    # display columns; pad to TABLE_W with box-drawing horizontal lines.
    local _sep_label="── system "
    local _sep_dashes=$(( TABLE_W - 10 ))
    printf "%s" "$_sep_label"
    [[ "$_sep_dashes" -gt 0 ]] && printf '─%.0s' $(seq 1 "$_sep_dashes")
    printf "%s\n" "$EOL"

    local cpu_pct mem_line mem_pct mem_used mem_total cpu_cores
    cpu_pct=$(sample_cpu_pct)
    push_hist cpu "$cpu_pct"
    # CPU detail: core count (cheap one-shot read; doesn't change between samples).
    cpu_cores=$(nproc 2>/dev/null || echo "?")
    render_metric "CPU" "$cpu_pct" "$hkey_cpu" "${cpu_cores} cores"

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
        # Strip "NVIDIA " prefix — the row label already says GPU.
        # Move temp onto the GPU line (it's a GPU property, not VRAM) with
        # a middle-dot separator so the secondary fact reads as one unit.
        local short_name="${name#NVIDIA }"
        render_metric "GPU $gpu_idx" "$util"     "${gpu_idx:+}${hkey_gpu}"  "${short_name} · ${temp}°C"
        render_metric "VRAM"         "$vram_pct" "${gpu_idx:+}${hkey_vram}" "${vram_gb_used} / ${vram_gb_total} GB"
        # If multi-GPU later, suppress sparkline on idx > 0 by passing "" as
        # hist_key. For now single-GPU case is the only path exercised.
        gpu_idx=$(( gpu_idx + 1 ))
    done < <(sample_gpu)

    if [[ "$gpu_idx" -eq 0 ]]; then
        printf "  %s(no NVIDIA GPU detected via nvidia-smi)%s\033[K\n" "$(c_dim)" "$(c_reset)"
    fi

    # Disk (sub-sampled — see maybe_sample_disk + DISK_INTERVAL_MULT)
    local disk_line; disk_line="$(maybe_sample_disk)"
    if [[ -n "$disk_line" ]]; then
        local disk_pct disk_used disk_total disk_mnt
        read -r disk_pct disk_used disk_total disk_mnt <<< "$disk_line"
        push_hist disk "$disk_pct"
        # Suppress the mount label when it's just '/' (default root). The
        # lone '/' floated at the same column as VRAM's temperature, making
        # the alignment look broken. Show the mount only when it's been
        # explicitly overridden via $DISK_MOUNT.
        local disk_detail="${disk_used} / ${disk_total} GB"
        [[ "$disk_mnt" != "/" ]] && disk_detail+="    ${disk_mnt}"
        render_metric "DISK" "$disk_pct" "$hkey_disk" "$disk_detail"
    fi
}

# Single render pass — used by both one-shot and --live modes.
render_one_frame() {
    if [[ ! -f "$MANIFEST" ]]; then
        echo "No manifest at $MANIFEST" >&2
        return 1
    fi
    local extra_args=( --manifest "$MANIFEST" --run-id "$RUN_ID" )
    [[ "$SHOW_STEPS" -eq 1 ]] && extra_args+=( --details )
    # In live mode every line must end in \033[K so the in-place cursor-home
    # redraw doesn't ghost when a line shrinks (e.g. the variable-height
    # OUTPUT QUALITY block). render_system already does this; tell the table to.
    [[ "$LIVE" -eq 1 ]] && extra_args+=( --clear-eol )
    python3 "$SCRIPT_DIR/lib/stats_render.py" "${extra_args[@]}"
    if [[ "$SHOW_SYSTEM" -eq 1 ]]; then
        # Blank separator before the system panel — erase its tail in live mode
        # too, so it doesn't ghost when the frame above changes height.
        [[ "$LIVE" -eq 1 ]] && printf '\033[K\n' || echo
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
    # Save the COMPLETE terminal state before we mess with it. `stty -echo`
    # + `read -s -n 1` puts the terminal in cbreak (non-canonical) mode;
    # restoring only `stty echo` on exit leaves the terminal stuck in
    # cbreak — user sees an apparent SSH hang because line editing /
    # Enter / Ctrl-keys all silently break.
    _SAVED_STTY=""
    [[ -t 0 ]] && _SAVED_STTY=$(stty -g 2>/dev/null || true)

    _restore_terminal() {
        # Restore the saved terminal state if we have it; fall back to
        # 'sane' (POSIX-defined safe baseline) which fixes echo, canonical
        # mode, line wrap, signal handling, etc.
        if [[ -n "$_SAVED_STTY" ]]; then
            stty "$_SAVED_STTY" 2>/dev/null || stty sane 2>/dev/null || true
        else
            stty sane 2>/dev/null || true
        fi
        # Show cursor, then clear screen and home cursor so the next prompt
        # lands cleanly without leftover stats content.
        printf '\033[?25h\033[2J\033[H'
    }

    trap '_restore_terminal; echo "(stopped)"; exit 0' INT TERM
    # Restore on ANY exit path (uncaught errors, set -e abort, etc.), not
    # just signal — covers the case where render_one_frame fails mid-loop.
    trap '_restore_terminal' EXIT

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
