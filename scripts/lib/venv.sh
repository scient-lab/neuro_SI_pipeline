#!/usr/bin/env bash
# venv.sh - activate the appropriate uv-managed venv for a phase.
#
# Venvs are created by setup.sh at the repo root (see Palash's uv-impl):
#   .venvs/graphrag         - Python 3.11 for 1_seed_kg
#   .venvs/graphmert        - Python 3.10 for 2_graphmert
#   .venvs/si_curriculum    - Python 3.10 for 3_si_curriculum (curriculum + sft + rl)
#
# Phase scripts call source_venv <name> before invoking Python. When the
# requested venv is missing the function logs a warning and continues, so
# stub/smoke runs work on a fresh checkout without setup.sh having run.

source_venv() {
    local name="$1"
    local venv_path="${REPO_ROOT}/.venvs/${name}"
    if [[ ! -f "${venv_path}/bin/activate" ]]; then
        log_warn "Venv not found: ${venv_path}"
        log_warn "Run ./setup.sh ${name}  (or ./setup.sh for all)"
        return 0
    fi
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"
    log_info "Activated venv: ${name}"
}
