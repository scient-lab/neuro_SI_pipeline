#!/usr/bin/env bash
# Phase: validate - Two-LLM consensus on extract's seed KG (paper §4.2).
# Delegates to 2_graphmert/utils/llm_scores/fact_score.py. Venv: graphmert.
#
# Implementation plan per Jake's clarification (2026-06-24):
#   - fact_score.py is generic: takes any CSV (head, relation, tail)
#   - Run it on extract's seed KG (kg_final.csv) BEFORE graphmert preprocess
#   - Output validated seed KG (kg_final_validated.csv)
#   - Graphmert preprocess consumes the validated version
#   - Downstream curriculum gets higher-quality triples, fewer errors
#
# Paper reference: Stephen & Jha 2026, §4.2 "Two-LLM Validation Strategy"
# - validate_a: Qwen/Qwen3-14B (consensus LLM #1)
# - validate_b: mistralai/Mistral-Nemo-Instruct-2407 (consensus LLM #2)
# - consensus_threshold: 0.6 (both must agree above confidence)
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
STEPS=(seed_kg_consensus)
PHASE_DESC="Two-LLM consensus on extract seed KG (paper §4.2 / Phase 2)"
STEP_DESCS=(
    "Two-LLM consensus filter on seed KG (fact_score.py)"
)

source_venv graphmert

OUTPUT_BASE=$(resolve_output_base)
SEED_KG="$OUTPUT_BASE/graphrag/output/kg_final.csv"
VALIDATED_SEED_KG="$OUTPUT_BASE/graphrag/output/kg_final_validated.csv"

# --- Steps ---
step_seed_kg_consensus() {
    log_info "validate :: seed_kg_consensus (fact_score.py on seed KG)"

    # Resolve the two consensus models from config
    local validate_a validate_b
    validate_a=$(get_model_id validate_a "")
    validate_b=$(get_model_id validate_b "")

    if [[ -z "$validate_a" || -z "$validate_b" ]]; then
        log_error "validate needs models.validate_a + models.validate_b from configs/default.yaml"
        return 1
    fi

    if [[ ! -f "$SEED_KG" ]]; then
        log_error "seed KG not found at $SEED_KG — extract phase must run first"
        return 1
    fi

    log_info "validate :: Running two-LLM consensus:"
    log_info "  Input: $SEED_KG"
    log_info "  Models: $validate_a (A) + $validate_b (B)"
    log_info "  Output: $VALIDATED_SEED_KG"

    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/llm_scores/fact_score.py \
          --input_csv   "$SEED_KG" \
          --output_csv  "$VALIDATED_SEED_KG" \
          --model_ids   "$validate_a" "$validate_b" \
          --batch_size  64 ) \
        || { log_error "seed_kg_consensus failed"; return 1; }
    # NOTE: --max_model_len intentionally omitted — fact_score.py reads
    # graphmert.fact_score_max_model_len from config (4096) so the cap is
    # profile-tunable in one place. See configs/default.yaml.

    # Report drop rate (guard before==0 so an empty seed KG gives a clear
    # message instead of a bash "division by 0" crash that masks the real cause).
    local before after drop pct
    before=$(($(wc -l < "$SEED_KG") - 1))
    after=$(($(wc -l < "$VALIDATED_SEED_KG") - 1))
    drop=$((before - after))
    if [[ "$before" -gt 0 ]]; then
        pct=$((drop * 100 / before))
        log_info "validate :: consensus filter: $before triples in → $after passed (dropped $drop, $pct%)"
    else
        log_warn "validate :: seed KG had 0 triples — nothing to validate (check the extract phase output)"
    fi
}

# --- Step dispatch ---
run_step "$PHASE_NAME" "seed_kg_consensus" "step_seed_kg_consensus" || exit $?
