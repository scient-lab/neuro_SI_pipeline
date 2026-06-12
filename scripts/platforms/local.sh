#!/usr/bin/env bash
# Platform: local - workstation or on-prem (incl. Princeton on-prem).
# Sourced by pipeline.sh; must define exec_phase_on_platform.

exec_phase_on_platform() {
    local phase_script="$1"
    local step_filter="$2"
    bash "$phase_script" "$step_filter"
}
