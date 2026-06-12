#!/usr/bin/env bash
# Platform: runpod - bootstrap + exec on a RunPod pod.
# Sourced by pipeline.sh; must define exec_phase_on_platform.

exec_phase_on_platform() {
    local phase_script="$1"
    local step_filter="$2"

    # TODO: pod-side bootstrap (mounts, env, model downloads to /workspace, etc.)
    # For now, behave like local - extend this when the RunPod bootstrap is wired up.
    bash "$phase_script" "$step_filter"
}
