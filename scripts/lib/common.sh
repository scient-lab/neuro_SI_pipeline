#!/usr/bin/env bash
# common.sh - shared shell helpers. Sourced by pipeline.sh and phase scripts.

log_info()  { printf '[%(%H:%M:%S)T] INFO  %s\n'  -1 "$*" >&2; }
log_warn()  { printf '[%(%H:%M:%S)T] WARN  %s\n'  -1 "$*" >&2; }
log_error() { printf '[%(%H:%M:%S)T] ERROR %s\n'  -1 "$*" >&2; }

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        log_error "Required environment variable is unset: $name"
        exit 1
    fi
}

# step_enabled <step_name>
# Returns 0 if the step should run given PIPELINE_STEP_FILTER.
step_enabled() {
    local step="$1"
    local filter="${PIPELINE_STEP_FILTER:-all}"
    [[ "$filter" == "all" ]] && return 0
    IFS=',' read -ra wanted <<< "$filter"
    for w in "${wanted[@]}"; do
        [[ "$w" == "$step" ]] && return 0
    done
    return 1
}

# get_model_id <key> [default]
# Read models.<key> from the merged pipeline config. Returns the default
# (or empty string) if the key is missing.
get_model_id() {
    local key="$1" default="${2:-}"
    uv run --no-project --quiet --with pyyaml python3 -c \
        "import pipeline_config; print(pipeline_config.get_model_id('$key', '$default') or '')" \
        2>/dev/null || echo "$default"
}

# get_phase_param <phase> <key> [default]
# Read cfg[<phase>][<key>] from the merged pipeline config.
get_phase_param() {
    local phase="$1" key="$2" default="${3:-}"
    uv run --no-project --quiet --with pyyaml python3 -c \
        "import pipeline_config; v = pipeline_config.get_phase_param('$phase', '$key', '$default'); print(v if v is not None else '')" \
        2>/dev/null || echo "$default"
}

# resolve_output_base
# Resolve OUTPUT_BASE — operator-set env var wins; otherwise default to the
# repo's outputs/ dir. Phase scripts use this for all working directories.
resolve_output_base() {
    if [[ -n "${OUTPUT_BASE:-}" ]]; then
        echo "$OUTPUT_BASE"
    else
        echo "${REPO_ROOT:-$(pwd)}/outputs"
    fi
}

# require_python_step <step_name> <description>
# Helper that runs a python command inside a (...) subshell so a step failure
# doesn't kill the whole phase. Logs the outcome. Use like:
#   run_python_step "chunk" "graphrag chunk" \
#       "( cd \"$REPO_ROOT/1_seed_kg\" && python graphrag_index.py --root_dir $OUT/graphrag --step 1 )"
run_python_step() {
    local label="$1" desc="$2" cmd="$3"
    log_info "$label :: $desc"
    if ( eval "$cmd" ); then
        return 0
    else
        log_error "$label failed (exit $?)"
        return 1
    fi
}

# ==========================================================================
# Run manifest + per-step logging + optional CloudWatch
# --------------------------------------------------------------------------
# pipeline.sh exports PIPELINE_MANIFEST (run_manifest.json path), PIPELINE_LOG_DIR
# (logs/<run_id>/), and RUN_ID. All helpers below no-op gracefully when those
# are unset, so a phase script run standalone (`bash phases/extract.sh chunk`)
# still works — it just runs uninstrumented.
# ==========================================================================

# _manifest <subcommand...> — update run_manifest.json via the stdlib-only
# helper. No-op without PIPELINE_MANIFEST; never fatal (a manifest write must
# not take the pipeline down).
_manifest() {
    [[ -n "${PIPELINE_MANIFEST:-}" ]] || return 0
    python3 "${REPO_ROOT}/scripts/lib/manifest.py" "$@" \
        || log_warn "manifest update failed: $1"
}

# _cw_ship <phase> <step> <logfile> — push a finished step log to CloudWatch.
# No-op unless CW_LOG_GROUP is set; non-fatal (local file + S3 stay canonical).
_cw_ship() {
    local phase="$1" step="$2" logfile="$3"
    [[ -n "${CW_LOG_GROUP:-}" ]] || return 0
    [[ -f "$logfile" ]] || return 0
    command -v python3 >/dev/null 2>&1 || return 0
    python3 "${REPO_ROOT}/scripts/lib/cw_ship.py" \
        --group "$CW_LOG_GROUP" \
        --stream "${RUN_ID:-adhoc}/${phase}/${step}" \
        --file "$logfile" \
        || log_warn "CloudWatch ship failed for $phase/$step (non-fatal)"
}

# run_step <phase> <step> <fn> [args...]
# Instrumented step runner — the single chokepoint every phase step flows
# through. It:
#   1. honors PIPELINE_STEP_FILTER (records "skipped" + returns 0 when filtered)
#   2. stamps the step "running" in the manifest with its log-file path
#   3. runs <fn> [args...] with stdout+stderr tee'd to
#      logs/<run_id>/<phase>/<step>.log (path recorded in the manifest)
#   4. captures the real exit code via PIPESTATUS[0] (NOT tee's)
#   5. stamps "completed"/"failed" + exit code, then best-effort CloudWatch push
# <fn> must `return 1` (NOT `exit 1`) on failure so the manifest can be updated.
# Returns the step's exit code; callers do `run_step ... || exit $?`.
run_step() {
    local phase="$1" step="$2" fn="$3"
    shift 3  # remaining args ("$@") are forwarded to <fn>

    if ! step_enabled "$step"; then
        log_info "$phase :: $step (skipped — not in step filter)"
        _manifest skip-step --path "$PIPELINE_MANIFEST" --phase "$phase" --step "$step"
        return 0
    fi

    local logdir="${PIPELINE_LOG_DIR:-$(resolve_output_base)/logs/adhoc}/${phase}"
    mkdir -p "$logdir"
    local logfile="$logdir/${step}.log"
    local rellog="${logfile#"${REPO_ROOT}"/}"
    local cwstream=""
    [[ -n "${CW_LOG_GROUP:-}" ]] && cwstream="${RUN_ID:-adhoc}/${phase}/${step}"

    _manifest start-step --path "$PIPELINE_MANIFEST" --phase "$phase" --step "$step" \
        --log-file "$rellog" --cw-stream "$cwstream"

    # set +e around the pipeline so a failing step doesn't abort the phase
    # before we record its status; PIPESTATUS[0] is the step's code, not tee's.
    local rc=0
    set +e
    { "$fn" "$@"; } 2>&1 | tee "$logfile"
    rc=${PIPESTATUS[0]}
    set -e

    _manifest end-step --path "$PIPELINE_MANIFEST" --phase "$phase" --step "$step" \
        --exit-code "$rc" --log-file "$rellog"
    _cw_ship "$phase" "$step" "$logfile"

    if [[ "$rc" -ne 0 ]]; then
        log_error "$phase.$step failed (exit $rc)"
        return "$rc"
    fi
    return 0
}
