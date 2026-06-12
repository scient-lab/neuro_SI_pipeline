#!/usr/bin/env bash
# pipeline.sh - orchestrator for the specialized-SLM pipeline.
#
# Parses CLI flags, sets the env contract consumed by pipeline_config.py,
# and dispatches the requested phases via the selected platform wrapper.
#
# The config loader (pipeline_config.py) reads layers directly from
# this repo's configs/, domains/, and prompts/ at runtime; no pre-merge
# step is required.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# --- Defaults ---------------------------------------------------------------
DOMAIN="neuroscience"
PROFILE=""
PLATFORM="local"
PHASES="all"
STEPS="all"

ALL_PHASES=(extract validate graphmert curriculum sft rl)

# --- Arg parsing ------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: pipeline.sh [--domain <name>] [--profile <name>] [--platform <name>]
                   [--phase <list>] [--step <list>]

Options:
  --domain    Domain name (default: neuroscience). Must exist as domains/<name>.yaml.
  --profile   Scaling profile: smoke | pilot | paper. Default: no profile (use defaults).
  --platform  local (default) | runpod | aws | princeton.
  --phase     Comma-separated phase names, or "all" (default).
              Phases: extract, validate, graphmert, curriculum, sft, rl
  --step      Comma-separated step names within the phase, or "all" (default).
              Only meaningful when a single phase is specified.

Examples:
  pipeline.sh                                              # neuroscience + defaults
  pipeline.sh --profile smoke
  pipeline.sh --phase extract --step parse_pdf,chunk
  pipeline.sh --domain neuroscience --profile paper --platform runpod
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)   DOMAIN="$2"; shift 2 ;;
        --profile)  PROFILE="$2"; shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --phase)    PHASES="$2"; shift 2 ;;
        --step)     STEPS="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *)          log_error "Unknown flag: $1"; usage; exit 2 ;;
    esac
done

# --- Validation -------------------------------------------------------------
domain_file="$REPO_ROOT/domains/${DOMAIN}.yaml"
platform_script="$SCRIPT_DIR/platforms/${PLATFORM}.sh"

[[ -f "$domain_file" ]] || { log_error "Missing domain file: $domain_file"; exit 2; }
[[ -f "$platform_script" ]] || { log_error "Missing platform script: $platform_script"; exit 2; }

if [[ -n "$PROFILE" ]]; then
    profile_file="$REPO_ROOT/configs/profiles/${PROFILE}.yaml"
    [[ -f "$profile_file" ]] || { log_error "Missing profile file: $profile_file"; exit 2; }
fi

# Resolve phase list
if [[ "$PHASES" == "all" ]]; then
    selected_phases=("${ALL_PHASES[@]}")
else
    IFS=',' read -ra selected_phases <<< "$PHASES"
    for p in "${selected_phases[@]}"; do
        [[ -f "$SCRIPT_DIR/phases/${p}.sh" ]] || {
            log_error "Unknown phase: $p"; exit 2;
        }
    done
fi

# --- Dispatch ---------------------------------------------------------------
log_info "Domain   : $DOMAIN"
log_info "Profile  : ${PROFILE:-<none>}"
log_info "Platform : $PLATFORM"
log_info "Phases   : ${selected_phases[*]}"
log_info "Steps    : $STEPS"

# Env contract consumed by pipeline_config.py and phase scripts.
export SI_DOMAIN="$DOMAIN"
export SI_PROFILE="$PROFILE"
export SI_PLATFORM="$PLATFORM"
export PIPELINE_STEP_FILTER="$STEPS"
export REPO_ROOT

# shellcheck source=platforms/local.sh
source "$platform_script"

for phase in "${selected_phases[@]}"; do
    phase_script="$SCRIPT_DIR/phases/${phase}.sh"
    log_info "── Phase: $phase ─────────────────────────────────"
    exec_phase_on_platform "$phase_script" "$STEPS"
done

log_info "Pipeline complete."
