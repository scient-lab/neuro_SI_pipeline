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
