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
