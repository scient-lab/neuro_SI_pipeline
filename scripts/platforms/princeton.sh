#!/usr/bin/env bash
# Platform: princeton - on-prem Princeton cluster.
# Currently delegates to local.sh. When SLURM scheduling is needed, replace
# the body of exec_phase_on_platform with sbatch / srun calls.
# Sourced by pipeline.sh; must define exec_phase_on_platform.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=local.sh
source "$SCRIPT_DIR/local.sh"
