#!/usr/bin/env bash
# Phase: curriculum - multi-hop Q&A generation with two-LLM check.
# Delegates to 3_si_curriculum. Venv: si_curriculum.
#
# Maps our STEPS onto the Princeton README:
#   path_traversal       calculate_hops.py — annotate KG triples with hop distance
#   prune_paths          (configured inside generate_curriculum; not a separate script)
#   generate_qa          generate_curriculum.py — Gemini-based Q&A items
#   validate_qa          verify_questions.py — two-LLM consensus filter
#   assemble_curriculum  (no-op — verify_questions writes the final JSON)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

STEPS=(path_traversal prune_paths generate_qa validate_qa assemble_curriculum)

source_venv si_curriculum

OUTPUT_BASE=$(resolve_output_base)
GRAPHRAG_DIR="$OUTPUT_BASE/graphrag"
GRAPHMERT_DIR="$OUTPUT_BASE/graphmert"
CURRICULUM_DIR="$OUTPUT_BASE/curriculum"
mkdir -p "$CURRICULUM_DIR"

CHECK_A=$(get_model_id curriculum_check_a "")
CHECK_B=$(get_model_id curriculum_check_b "")

# generate_curriculum.py needs Gemini API access.
require_env GEMINI_API_KEY 2>/dev/null || export GOOGLE_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"

NUM_QUESTIONS=$(get_phase_param curriculum num_questions 5000)

for step in "${STEPS[@]}"; do
    if ! step_enabled "$step"; then continue; fi

    case "$step" in
        path_traversal)
            log_info "curriculum :: path_traversal (calculate_hops.py)"
            KG_PATH="${KG_PATH:-$GRAPHMERT_DIR/final_kg/validated_triples.csv}"
            # calculate_hops.py does pd.read_csv(seed_kg_path); use the CSV
            # variant (extract.sh writes both .csv and .parquet from graphrag).
            SEED_KG="$GRAPHRAG_DIR/output/kg_final.csv"
            MANIFEST="$CURRICULUM_DIR/kg_manifest.json"
            ( cd "$REPO_ROOT/3_si_curriculum" && \
              python calculate_hops.py \
                  --kg_path      "$KG_PATH" \
                  --seed_kg_path "$SEED_KG" \
                  --output_path  "$MANIFEST" ) || { log_error "path_traversal failed"; exit 1; }
            log_info "Manifest written: $MANIFEST"
            ;;

        prune_paths)
            log_info "curriculum :: prune_paths (configured via hop_range + HUB_REMOVAL_PERCENTILE inside generate_curriculum)"
            ;;

        generate_qa)
            log_info "curriculum :: generate_qa (generate_curriculum.py — Gemini)"
            if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
                log_error "generate_qa needs GEMINI_API_KEY or GOOGLE_API_KEY in the environment"
                exit 1
            fi
            MANIFEST="$CURRICULUM_DIR/kg_manifest.json"
            ( cd "$REPO_ROOT/3_si_curriculum" && \
              python curriculum_generator/generate_curriculum.py \
                  --manifest_path "$MANIFEST" \
                  --output_dir    "$CURRICULUM_DIR" \
                  --target_count  "$NUM_QUESTIONS" \
                  --api_key       "$GOOGLE_API_KEY" \
                  --seed          42 ) || { log_error "generate_qa failed"; exit 1; }
            log_info "Generated curriculum: $CURRICULUM_DIR/curriculum.json"
            ;;

        validate_qa)
            log_info "curriculum :: validate_qa (verify_questions.py — two-LLM consensus)"
            if [[ -z "$CHECK_A" || -z "$CHECK_B" ]]; then
                log_error "validate_qa needs models.curriculum_check_a and curriculum_check_b"
                exit 1
            fi
            INPUT="$CURRICULUM_DIR/curriculum.json"
            VERIFIED="$OUTPUT_BASE/curriculum_verified"
            mkdir -p "$VERIFIED"
            ( cd "$REPO_ROOT/3_si_curriculum" && \
              python curriculum_generator/verify_questions.py \
                  --input_json  "$INPUT" \
                  --output_json "$VERIFIED/curriculum_verified.json" \
                  --model_ids   "$CHECK_A" "$CHECK_B" \
                  --batch_size  64 \
                  --tensor_parallel_size 1 ) || { log_error "validate_qa failed"; exit 1; }
            ;;

        assemble_curriculum)
            VERIFIED="$OUTPUT_BASE/curriculum_verified/curriculum_verified.json"
            if [[ -f "$VERIFIED" ]]; then
                log_info "curriculum :: assemble_curriculum (already written by verify_questions: $VERIFIED)"
            else
                log_warn "curriculum :: assemble_curriculum — verified curriculum missing"
            fi
            ;;
    esac
done
