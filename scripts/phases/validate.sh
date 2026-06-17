#!/usr/bin/env bash
# Phase: validate - two-LLM consensus filter on candidate triples.
# Venv: graphrag.
#
# NOTE (2026-06-15): Princeton's reference implementation does NOT have a
# dedicated validate phase as a separate Python script. The "two-LLM
# consensus" check happens in two distinct places downstream:
#
#   1. 2_graphmert/utils/llm_scores/fact_score.py — scores graphmert-predicted
#      triples by two-LLM agreement (Stage 2.7 in the Princeton README).
#      Triggered by the GRAPHMERT phase, not here.
#
#   2. 3_si_curriculum/curriculum_generator/verify_questions.py — two-LLM
#      filter over generated Q&A items (Stage 3.2 in the Princeton README).
#      Triggered by the CURRICULUM phase.
#
# This phase is reserved for a future seed-KG-level two-LLM validation
# (after extract, before graphmert) that the Stephen & Jha 2026 paper
# describes but isn't implemented as a standalone script. For now it's a
# no-op so orchestration smoke still passes through the phase.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

PHASE_NAME=validate
STEPS=(prepare_candidates llm_check_a llm_check_b consensus dedupe_merge emit_seed_kg)
PHASE_DESC="(no-op) 2-LLM consensus checks happen inline in graphmert + curriculum"
STEP_DESCS=(
    "(no-op)"
    "(no-op)"
    "(no-op)"
    "(no-op)"
    "(no-op)"
    "(no-op)"
)

source_venv graphrag

# Every step is an intentional no-op (see header). They still flow through
# run_step so the manifest records them as completed/skipped with timestamps.
step_noop() {
    log_info "validate :: $1 (no-op — two-LLM checks happen in graphmert + curriculum phases)"
}
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" step_noop "$step" || exit $?
done
