#!/usr/bin/env bash
# Phase: curriculum - multi-hop Q&A generation with two-LLM check.
# Delegates to 3_si_curriculum. Venv: si_curriculum.
#
# Maps our STEPS onto the Princeton README:
#   path_traversal       calculate_hops.py — annotate KG triples with hop distance
#   prune_paths          (configured via hop_range inside generate_curriculum; no-op step)
#   generate_qa_pair     generate_curriculum.py --stage pair          — Gemini bare QA pairs
#   validate_qa_pair     generate_curriculum.py --stage validate_pair — 1 non-Gemini pair check
#   generate_qa_item     generate_curriculum.py --stage item          — Gemini Pro reasoning trace
#   validate_qa_item     verify_questions.py — 2 non-Gemini consensus (stamps per-grader verdicts)
#   assemble_curriculum  filter stage==verified -> curriculum_verified.json (+ finalize stats)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

PHASE_NAME=curriculum
STEPS=(path_traversal prune_paths generate_qa_pair validate_qa_pair generate_qa_item validate_qa_item assemble_curriculum)
PHASE_DESC="Generate Q&A curriculum from final KG via n-hop paths + Gemini (4-step: pair/check/item/check)"
STEP_DESCS=(
    "Find n-hop paths in the final KG (calculate_hops.py)"
    "(configured via hop_range inside generate_curriculum; no-op step)"
    "Generate bare QA pairs via Gemini (generate_curriculum.py --stage pair)"
    "Check QA pairs — 1 non-Gemini reasoning LLM (--stage validate_pair)"
    "Add reasoning trace via Gemini Pro (generate_curriculum.py --stage item)"
    "Check QA items — 2 non-Gemini consensus (verify_questions.py)"
    "Assemble stage==verified -> curriculum_verified.json (+ finalize stats)"
)

source_venv si_curriculum

OUTPUT_BASE=$(resolve_output_base)
GRAPHRAG_DIR="$OUTPUT_BASE/graphrag"
GRAPHMERT_DIR="$OUTPUT_BASE/graphmert"
CURRICULUM_DIR="$OUTPUT_BASE/curriculum"
mkdir -p "$CURRICULUM_DIR"

# curriculum_check_a/b (the 2-LLM consensus graders) are resolved INSIDE step_validate_qa_item —
# where SI_PHASE/SI_STEP are set — so pipeline_config records them to the per-step config ledger
# that config.sh --models reads (a top-level resolve here would go unrecorded).

# generate_curriculum.py needs Gemini API access. The Gemini SDK reads
# GOOGLE_API_KEY from env, but operators typically set GEMINI_API_KEY in
# .env (matches the env_file convention). Prefer caller-set GOOGLE_API_KEY
# if already exported (ad-hoc invocations); else mirror from GEMINI_API_KEY.
#
# Prior version `require_env GEMINI_API_KEY || export GOOGLE_API_KEY=...`
# was broken: require_env exits 1 on missing var (it doesn't return non-zero
# to the caller), so the `||` branch is unreachable in BOTH directions —
# success short-circuits the export, failure exits the script. Result:
# GOOGLE_API_KEY never got set even when GEMINI_API_KEY was present in .env.
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}"

NUM_QUESTIONS=$(get_phase_param curriculum num_questions 5000)

# KG_MANIFEST is the hop-annotated KG (calculate_hops.py output) — NOT the
# pipeline run_manifest.json. path_traversal writes it; generate_qa reads it.
KG_MANIFEST="$CURRICULUM_DIR/kg_manifest.json"
# The 4-step flow streams one working file (curriculum.jsonl) + a per-step stats file.
CURRICULUM_JSONL="$CURRICULUM_DIR/curriculum.jsonl"
CURRICULUM_STATS="$CURRICULUM_DIR/curriculum_stats.json"

# --- Steps ---------------------------------------------------------------
step_path_traversal() {
    log_info "curriculum :: path_traversal (calculate_hops.py)"
    # KG_PATH = the FULL expanded KG (seed ∪ graphmert-validated expansions) =
    # expand_kg's merge_kgs.py output (final_relationships.parquet). The earlier
    # default (validated_triples.csv) was the PRE-merge, expansion-ONLY set, so
    # the hop graph never saw the seed edges or the merge's dedup / relation-count
    # filtering — the merged KG was computed and discarded. _load_kg auto-detects
    # parquet, so pointing at the .parquet is fine.
    local KG_PATH="${KG_PATH:-$GRAPHMERT_DIR/final_kg/final_relationships.parquet}"
    local SEED_KG="$GRAPHRAG_DIR/output/kg_final.csv"
    # Seed-only fallback (empty expansion → seed-only 1-hop curriculum) is OFF by
    # default: an empty kg_path now FAILS loudly instead of silently degrading.
    # Profiles that expect a thin/empty expansion (smoke) opt in via
    # curriculum.allow_seed_only_fallback=true.
    local SEED_ONLY_ARGS=()
    # get_phase_param returns Python's "True"/"False" (capitalized) for YAML
    # bools, so match all truthy spellings — same guard as rl.sh:91. A bare
    # == "true" silently dropped smoke's allow_seed_only_fallback=true, so
    # calculate_hops failed loud on the all-hop-0 KG instead of falling back.
    local _allow_seed_only
    _allow_seed_only=$(get_phase_param curriculum allow_seed_only_fallback false)
    if [[ "$_allow_seed_only" == "true" || "$_allow_seed_only" == "True" || "$_allow_seed_only" == "1" ]]; then
        SEED_ONLY_ARGS=(--allow-seed-only)
    fi
    ( cd "$REPO_ROOT/3_si_curriculum" && \
      python calculate_hops.py \
          --kg_path      "$KG_PATH" \
          --seed_kg_path "$SEED_KG" \
          --output_path  "$KG_MANIFEST" \
          "${SEED_ONLY_ARGS[@]}" ) || { log_error "path_traversal failed"; return 1; }
    log_info "Hop manifest written: $KG_MANIFEST"
}

step_prune_paths() {
    log_info "curriculum :: prune_paths (configured via hop_range + HUB_REMOVAL_PERCENTILE inside generate_curriculum)"
}

step_generate_qa_pair() {
    log_info "curriculum :: generate_qa_pair (generate_curriculum.py --stage pair — Gemini)"
    if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        log_error "generate_qa_pair needs GEMINI_API_KEY or GOOGLE_API_KEY in the environment"
        return 1
    fi
    ( cd "$REPO_ROOT/3_si_curriculum" && \
      python curriculum_generator/generate_curriculum.py \
          --stage         pair \
          --manifest_path "$KG_MANIFEST" \
          --output_dir    "$CURRICULUM_DIR" \
          --target_count  "$NUM_QUESTIONS" \
          --api_key       "$GOOGLE_API_KEY" \
          --seed          42 ) || { log_error "generate_qa_pair failed"; return 1; }
    log_info "QA pairs written: $CURRICULUM_JSONL"
}

step_validate_qa_pair() {
    log_info "curriculum :: validate_qa_pair (pair_check — 1 non-Gemini reasoning LLM)"
    # pair_check.py uses the OpenAI SDK with a configurable base_url
    # (curriculum.pair_check_base_url). Hosted OpenAI needs the real key; a local
    # vLLM OpenAI server accepts any non-empty value. Key var name is configurable.
    local KEY_ENV
    KEY_ENV=$(get_phase_param curriculum pair_check_api_key_env "OPENAI_API_KEY")
    if [[ -z "${!KEY_ENV:-}" ]]; then
        log_error "validate_qa_pair needs \$$KEY_ENV (OpenAI key, or any non-empty value for a local vLLM base_url)"
        return 1
    fi
    ( cd "$REPO_ROOT/3_si_curriculum" && \
      python curriculum_generator/generate_curriculum.py \
          --stage      validate_pair \
          --output_dir "$CURRICULUM_DIR" ) || { log_error "validate_qa_pair failed"; return 1; }
}

step_generate_qa_item() {
    log_info "curriculum :: generate_qa_item (generate_curriculum.py --stage item — Gemini Pro trace)"
    if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        log_error "generate_qa_item needs GEMINI_API_KEY or GOOGLE_API_KEY in the environment"
        return 1
    fi
    ( cd "$REPO_ROOT/3_si_curriculum" && \
      python curriculum_generator/generate_curriculum.py \
          --stage      item \
          --output_dir "$CURRICULUM_DIR" \
          --api_key    "$GOOGLE_API_KEY" ) || { log_error "generate_qa_item failed"; return 1; }
}

step_validate_qa_item() {
    log_info "curriculum :: validate_qa_item (verify_questions.py — 2 non-Gemini consensus)"
    local CHECK_A CHECK_B
    CHECK_A=$(get_model_id curriculum_check_a "")
    CHECK_B=$(get_model_id curriculum_check_b "")
    if [[ -z "$CHECK_A" || -z "$CHECK_B" ]]; then
        log_error "validate_qa_item needs models.curriculum_check_a and curriculum_check_b"
        return 1
    fi
    # vLLM init knobs (batch_size, tensor_parallel_size, gpu_memory_utilization,
    # max_model_len) come from configs/default.yaml::curriculum.validate_qa_*
    # via get_phase_param inside verify_questions.py — YAML is the single source.
    ( cd "$REPO_ROOT/3_si_curriculum" && \
      python curriculum_generator/verify_questions.py \
          --curriculum_jsonl "$CURRICULUM_JSONL" \
          --model_ids        "$CHECK_A" "$CHECK_B" ) \
        || { log_error "validate_qa_item failed"; return 1; }
}

step_assemble_curriculum() {
    log_info "curriculum :: assemble_curriculum (stage==verified -> curriculum_verified.json)"
    local VERIFIED_DIR="$OUTPUT_BASE/curriculum_verified"
    mkdir -p "$VERIFIED_DIR"
    if [[ ! -f "$CURRICULUM_JSONL" ]]; then
        log_error "assemble_curriculum: $CURRICULUM_JSONL missing — earlier steps must run first"
        return 1
    fi
    ( cd "$REPO_ROOT/3_si_curriculum" && \
      python curriculum_generator/assemble_curriculum.py \
          --curriculum_jsonl "$CURRICULUM_JSONL" \
          --output_json      "$VERIFIED_DIR/curriculum_verified.json" \
          --stats_path       "$CURRICULUM_STATS" ) \
        || { log_error "assemble_curriculum failed"; return 1; }
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" "step_$step" || exit $?
done
