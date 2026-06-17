#!/usr/bin/env bash
# Phase: graphmert - masked-node-modeling KG expansion.
# Delegates to 2_graphmert/*. Venv: graphmert.
#
# Maps our STEPS onto the Princeton README's 7-step pipeline:
#   tokenize             step 1  (run_tokenization.py)
#   preprocess           step 2  (entity_discovery + find_heads_positions)
#                        step 3  (add_llm_relations + clean_llm_relations)
#                        step 4  (run_dataset_preprocessing — co-occurrence grounding)
#   train_mnm            step 5  (run_mlm.py)
#   predict_tails        step 6  (predict_tails_llm.py)
#   validate_predictions step 7  (combine_tails + fact_score, two-LLM consensus)
#   expand_kg            merge predictions into final KG (manual until merge_kgs.py is wired)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

PHASE_NAME=graphmert
STEPS=(tokenize preprocess train_mnm predict_tails validate_predictions expand_kg)
PHASE_DESC="Expand seed KG via masked-node-modeling training + LLM tail prediction"
STEP_DESCS=(
    "Build stable tokenizer + tokenized chunks (run_tokenization.py)"
    "Entity discovery + LLM relations + dataset grounding (substeps 2-4)"
    "Train GraphMERT MNM on grounded triples (run_mlm.py)"
    "vLLM predicts novel tails for held-out heads (predict_tails_llm.py)"
    "Combine prediction shards + 2-LLM fact scoring"
    "Merge seed KG + validated expansions into final KG (merge_kgs.py)"
)

source_venv graphmert

OUTPUT_BASE=$(resolve_output_base)
GRAPHRAG_DIR="$OUTPUT_BASE/graphrag"
GRAPHMERT_DIR="$OUTPUT_BASE/graphmert"
mkdir -p "$GRAPHMERT_DIR"

EXTRACT_MODEL_ID=$(get_model_id extract "")
VALIDATE_A=$(get_model_id validate_a "")
VALIDATE_B=$(get_model_id validate_b "")

# Princeton convention: stable tokenizer dir is reused across steps.
STABLE_TOKENIZER="$GRAPHMERT_DIR/stable_tokenizer"

# Seed KG CSV (the format dataset_preprocessing_utils.py expects).
SEED_KG_CSV="$GRAPHRAG_DIR/output/kg_final.csv"

# args_mlm.yaml is a TEMPLATE with ${VAR} placeholders. Resolve once at phase
# start via envsubst so all later steps (preprocess, train_mnm) read the same
# concrete paths. The resolved file lands in $GRAPHMERT_DIR so it's gitignored
# and survives across step invocations within a phase run.
ARGS_MLM_TMPL="${ARGS_MLM_TMPL:-$REPO_ROOT/2_graphmert/launch_configs/args_mlm.yaml}"
ARGS_MLM_YAML="$GRAPHMERT_DIR/args_mlm.resolved.yaml"
if [[ -f "$ARGS_MLM_TMPL" ]]; then
    if ! command -v envsubst >/dev/null 2>&1; then
        log_error "envsubst not found (apt install gettext-base) — cannot resolve $ARGS_MLM_TMPL"
        exit 1
    fi
    GRAPHMERT_DIR="$GRAPHMERT_DIR" \
        STABLE_TOKENIZER="$STABLE_TOKENIZER" \
        SEED_KG_CSV="$SEED_KG_CSV" \
        envsubst '${GRAPHMERT_DIR} ${STABLE_TOKENIZER} ${SEED_KG_CSV}' \
        < "$ARGS_MLM_TMPL" > "$ARGS_MLM_YAML"
fi

# --- Steps ---------------------------------------------------------------
# Each step is a function returning non-zero on failure (NOT exit) so run_step
# can record status/timing/exit-code in the manifest and tee a per-step log.
step_tokenize() {
    log_info "graphmert :: tokenize (step 1 — run_tokenization.py)"
    ( cd "$REPO_ROOT/2_graphmert" && \
      python run_tokenization.py \
          --input_dir   "$GRAPHRAG_DIR/input" \
          --output_dir  "$GRAPHMERT_DIR" \
          --tokenizer   "${TOKENIZER_BASE:-dmis-lab/biobert-base-cased-v1.2}" \
          --max_seq_length 128 \
          --validation_split_pct 5 \
          --num_workers 8 \
          --seed 0 ) || { log_error "graphmert.tokenize failed"; return 1; }
}

step_preprocess() {
            log_info "graphmert :: preprocess (steps 2+3+4)"
    # Step 2: entity_discovery + find_heads_positions
    TOK_TRAIN=$(ls -d "$GRAPHMERT_DIR/tokenized_inputs/train_"* 2>/dev/null | head -1)
    if [[ -z "$TOK_TRAIN" ]]; then
        log_error "tokenized train dir not found under $GRAPHMERT_DIR/tokenized_inputs/"
        return 1
    fi
    if [[ -z "$EXTRACT_MODEL_ID" ]]; then
        log_error "preprocess needs models.extract in configs/default.yaml"
        return 1
    fi
    log_info "  step 2a: entity_discovery"
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/entity_discovery/entity_discovery.py \
          --tokenized_dir "$TOK_TRAIN" \
          --output_dir    "$GRAPHMERT_DIR/entity_discovery" \
          --model_id      "$EXTRACT_MODEL_ID" \
          --tokenizer     "$STABLE_TOKENIZER" ) || { log_error "entity_discovery failed"; return 1; }
    log_info "  step 2b: find_heads_positions"
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/entity_discovery/find_heads_positions.py \
          --heads_chunks_dir "$GRAPHMERT_DIR/entity_discovery" \
          --output_dir       "$GRAPHMERT_DIR/head_positions" \
          --tokenizer        "$STABLE_TOKENIZER" ) || { log_error "find_heads_positions failed"; return 1; }
    # Step 3: add_llm_relations + clean_llm_relations
    log_info "  step 3a: add_llm_relations"
    # find_heads_positions.py now writes the Dataset directly to its
    # --output_dir (no "neuro_heads_all_with_positions" subdir).
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/relation_matching/add_llm_relations.py \
          --dataset_path "$GRAPHMERT_DIR/head_positions" \
          --output_root  "$GRAPHMERT_DIR/llm_relations" \
          --output_name  relations_all \
          --model_id     "$EXTRACT_MODEL_ID" \
          --tokenizer    "$STABLE_TOKENIZER" ) || { log_error "add_llm_relations failed"; return 1; }
    log_info "  step 3b: clean_llm_relations"
    # clean_llm_relations.py creates two Dataset dirs under --output_dir:
    #   <output_dir>/relations_cleaned_train/
    #   <output_dir>/relations_cleaned_eval/
    # We point --output_dir directly at llm_relations/ so the two end
    # up as siblings of relations_all/ (no redundant wrapper dir).
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/relation_matching/clean_llm_relations.py \
          --input_dir  "$GRAPHMERT_DIR/llm_relations/relations_all" \
          --output_dir "$GRAPHMERT_DIR/llm_relations" \
          --tokenizer  "$STABLE_TOKENIZER" ) || { log_error "clean_llm_relations failed"; return 1; }
    # Step 4: run_dataset_preprocessing (co-occurrence grounding)
    log_info "  step 4: run_dataset_preprocessing"
    # dataset_preprocessing_utils.py does pd.read_csv(seed_kg_path), so
    # we consume kg_final.csv (extract.sh's cache step writes both .csv
    # and .parquet from graphrag).
    ( cd "$REPO_ROOT/2_graphmert" && \
      python run_dataset_preprocessing.py \
          --yaml_file    "$ARGS_MLM_YAML" \
          --seed_kg_path "$GRAPHRAG_DIR/output/kg_final.csv" \
          --train_src    "$GRAPHMERT_DIR/llm_relations/relations_cleaned_train" \
          --eval_src     "$GRAPHMERT_DIR/llm_relations/relations_cleaned_eval" \
          --tokenizer    "$STABLE_TOKENIZER" \
          --output_dir   "$GRAPHMERT_DIR/dataset" ) || { log_error "run_dataset_preprocessing failed"; return 1; }
}

step_train_mnm() {
    log_info "graphmert :: train_mnm (step 5 — run_mlm.py)"
    ( cd "$REPO_ROOT/2_graphmert" && \
      python run_mlm.py "$ARGS_MLM_YAML" ) || { log_error "train_mnm failed"; return 1; }
}

step_predict_tails() {
    log_info "graphmert :: predict_tails (step 6 — predict_tails_llm.py)"
    local CHKPT="${MLM_CHECKPOINT:-$GRAPHMERT_DIR/checkpoints/best}"
    # --dataset is the eval split written by clean_llm_relations.py
    # (now flat under llm_relations/, no wrapper dir).
    ( cd "$REPO_ROOT/2_graphmert" && \
      python predict_tails_llm.py \
          --model_id   "$CHKPT" \
          --tokenizer  "$STABLE_TOKENIZER" \
          --dataset    "$GRAPHMERT_DIR/llm_relations/relations_cleaned_eval" \
          --output_dir "$GRAPHMERT_DIR/predictions" \
          --num_shards "${PRED_NUM_SHARDS:-1}" \
          --shard_id   "${PRED_SHARD_ID:-0}" ) || { log_error "predict_tails failed"; return 1; }
}

step_validate_predictions() {
    log_info "graphmert :: validate_predictions (step 7 — combine_tails + fact_score)"
    log_info "  combine_tails"
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/combine_tails/combine_tails.py \
          --pred_dir   "$GRAPHMERT_DIR/predictions" \
          --output_dir "$GRAPHMERT_DIR/combined" \
          --model_id   "$EXTRACT_MODEL_ID" \
          --tokenizer  "$STABLE_TOKENIZER" ) || { log_error "combine_tails failed"; return 1; }
    if [[ -z "$VALIDATE_A" || -z "$VALIDATE_B" ]]; then
        log_error "fact_score needs models.validate_a and validate_b in configs/default.yaml"
        return 1
    fi
    log_info "  fact_score (two-LLM agreement)"
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/llm_scores/fact_score.py \
          --input_csv   "$GRAPHMERT_DIR/combined/expanded_triples.csv" \
          --output_csv  "$GRAPHMERT_DIR/final_kg/validated_triples.csv" \
          --model_ids   "$VALIDATE_A" "$VALIDATE_B" \
          --batch_size  64 \
          --max_model_len 4096 \
          --tensor_parallel_size 1 ) || { log_error "fact_score failed"; return 1; }
}

step_expand_kg() {
    log_info "graphmert :: expand_kg (merge validated tails into KG)"
    local VALIDATED="$GRAPHMERT_DIR/final_kg/validated_triples.csv"
    local SEED_KG="$GRAPHRAG_DIR/output/kg_final.parquet"
    local FINAL="$GRAPHMERT_DIR/final_kg/expanded_kg.parquet"
    if [[ -f "$VALIDATED" && -f "$SEED_KG" ]]; then
        ( cd "$REPO_ROOT/1_seed_kg" && \
          python merge_kgs.py \
              --new "$VALIDATED" --old "$SEED_KG" \
              --outdir "$(dirname "$FINAL")" ) || { log_error "expand_kg merge failed"; return 1; }
    else
        log_warn "expand_kg: validated_triples.csv or seed kg missing; skipping merge"
    fi
    log_info "Final expanded KG: $FINAL"
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" "step_$step" || exit $?
done
