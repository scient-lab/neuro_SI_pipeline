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
#   predict_tails        step 6  (predict_tails_llm.py — LLM-based tail extraction)
#   predict_tails_gm     step 6b (utils/predict_tails.py — GraphMERT MLM-based tail prediction)
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
STEPS=(tokenize preprocess train_mnm predict_tails predict_tails_gm validate_predictions expand_kg)
PHASE_DESC="Expand seed KG via masked-node-modeling training + LLM tail prediction"
STEP_DESCS=(
    "Build stable tokenizer + tokenized chunks (run_tokenization.py)"
    "Entity discovery + LLM relations + dataset grounding (substeps 2-4)"
    "Train GraphMERT MNM on grounded triples (run_mlm.py)"
    "vLLM predicts novel tails for held-out heads (predict_tails_llm.py)"
    "GraphMERT MLM predicts tails from trained checkpoint (utils/predict_tails.py)"
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
# Tail-prediction LLM (NOT the trained GraphMERT checkpoint — see comment
# in configs/default.yaml::models.predict_tails). Loaded by vLLM in
# 2_graphmert/predict_tails_llm.py.
PREDICT_TAILS_MODEL_ID=$(get_model_id predict_tails "")

# Princeton convention: stable tokenizer dir is reused across steps.
STABLE_TOKENIZER="$GRAPHMERT_DIR/stable_tokenizer"

# Seed KG CSV (the format dataset_preprocessing_utils.py expects).
# Prefer the validated seed KG from the validate phase (paper §4.2 two-LLM
# consensus). The validated file must EXIST and be NON-EMPTY (header + >=1 row);
# an empty one (consensus dropped everything) is treated as missing. Using the
# UNVALIDATED seed KG silently bypasses the consensus, so it is OPT-IN only —
# otherwise fail loud rather than degrade quietly.
VALIDATED_SEED_KG="$GRAPHRAG_DIR/output/kg_final_validated.csv"
# get_phase_param yields "True"/"False" (capitalized) for YAML bools; match all
# truthy spellings so this opt-in isn't silently dropped (same trap as
# curriculum.sh / rl.sh — a bare == "true" fails on "True").
_allow_unvalidated=$(get_phase_param graphmert allow_unvalidated_seed_kg false)
if [[ -s "$VALIDATED_SEED_KG" ]] && [[ "$(wc -l < "$VALIDATED_SEED_KG")" -gt 1 ]]; then
    SEED_KG_CSV="$VALIDATED_SEED_KG"
    log_info "graphmert :: Using validated seed KG (from validate phase)"
elif [[ "$_allow_unvalidated" == "true" || "$_allow_unvalidated" == "True" || "$_allow_unvalidated" == "1" ]]; then
    SEED_KG_CSV="$GRAPHRAG_DIR/output/kg_final.csv"
    log_warn "graphmert :: validated seed KG missing/empty — using UNVALIDATED seed KG (graphmert.allow_unvalidated_seed_kg=true). Two-LLM consensus is BYPASSED."
else
    log_error "graphmert :: validated seed KG missing or empty at $VALIDATED_SEED_KG."
    log_error "  Run the validate phase (two-LLM consensus) first, or set"
    log_error "  graphmert.allow_unvalidated_seed_kg=true to proceed on the raw seed KG."
    exit 1
fi

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
    # we consume kg_final.csv (extract.sh's finalize_seed_kg step writes both
    # .csv and .parquet from graphrag).
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
    # run_mlm.py uses argparse: --yaml_file <path>. Positional invocation
    # produces "unrecognized arguments". Mirrors run_dataset_preprocessing.py
    # in step 4 which uses the same flag style.
    ( cd "$REPO_ROOT/2_graphmert" && \
      python run_mlm.py --yaml_file "$ARGS_MLM_YAML" ) || { log_error "train_mnm failed"; return 1; }
}

step_predict_tails() {
    log_info "graphmert :: predict_tails (step 6 — predict_tails_llm.py)"
    # predict_tails_llm.py loads the model via vLLM as a generative LM,
    # so the path must point to a Qwen-style causal LM (NOT the trained
    # GraphMERT checkpoint, which is a BERT-style MLM that vLLM cannot
    # load). Sourced from configs/default.yaml::models.predict_tails;
    # MLM_CHECKPOINT env var can still override for ad-hoc runs.
    if [[ -z "$PREDICT_TAILS_MODEL_ID" ]]; then
        log_error "predict_tails needs models.predict_tails in configs/default.yaml"
        return 1
    fi
    local PRED_MODEL="${MLM_CHECKPOINT:-$PREDICT_TAILS_MODEL_ID}"
    # --dataset is the eval split written by clean_llm_relations.py
    # (now flat under llm_relations/, no wrapper dir).
    ( cd "$REPO_ROOT/2_graphmert" && \
      python predict_tails_llm.py \
          --model_id   "$PRED_MODEL" \
          --tokenizer  "$STABLE_TOKENIZER" \
          --dataset    "$GRAPHMERT_DIR/llm_relations/relations_cleaned_eval" \
          --output_dir "$GRAPHMERT_DIR/predictions" \
          --num_shards "${PRED_NUM_SHARDS:-1}" \
          --shard_id   "${PRED_SHARD_ID:-0}" ) || { log_error "predict_tails failed"; return 1; }
}

step_predict_tails_gm() {
    log_info "graphmert :: predict_tails_gm (step 6b — utils/predict_tails.py)"
    # Step 6b complements step 6: where predict_tails_llm.py uses vLLM to ask a
    # generic causal LM to extract tails, this step runs the trained GraphMERT
    # checkpoint as a masked-node-modeling probe and reads top-k predictions off
    # the first leaf slot.
    #
    # CHECKPOINT PATH: train_mnm (step 5) writes to args_mlm.yaml::output_dir =
    # ${GRAPHMERT_DIR}/mlm_output. This previously read ${GRAPHMERT_DIR}/checkpoints
    # — a path that never exists — so the dir-missing branch ALWAYS hit and the
    # step silently no-op'd, meaning the GraphMERT model never produced any tails.
    local CKPT_ROOT="$GRAPHMERT_DIR/mlm_output"
    # A valid checkpoint root either IS a checkpoint (config.json) or CONTAINS
    # checkpoint-* subdirs (run_mlm writes checkpoint-NNN). get_best_checkpoint in
    # predict_tails.py handles either.
    local _have_ckpt=0
    if [[ -d "$CKPT_ROOT" ]] && { [[ -f "$CKPT_ROOT/config.json" ]] || ls -d "$CKPT_ROOT"/checkpoint-* >/dev/null 2>&1; }; then
        _have_ckpt=1
    fi
    if [[ "$_have_ckpt" -ne 1 ]]; then
        # A missing checkpoint after train_mnm is a REAL failure (training failed,
        # or the path is wrong) — FAIL LOUD by default, never silently skip. The
        # only legitimate skip is a run that deliberately bypasses train_mnm; opt
        # into that explicitly with GRAPHMERT_PREDICT_TAILS_GM_REQUIRED=0.
        if [[ "${GRAPHMERT_PREDICT_TAILS_GM_REQUIRED:-1}" == "0" ]]; then
            log_warn "predict_tails_gm: no GraphMERT checkpoint under $CKPT_ROOT — skipping by request (GRAPHMERT_PREDICT_TAILS_GM_REQUIRED=0); GraphMERT will NOT contribute tails."
            return 0
        fi
        log_error "predict_tails_gm: no trained GraphMERT checkpoint under $CKPT_ROOT."
        log_error "  train_mnm (step 5) writes it there (args_mlm.yaml::output_dir=\${GRAPHMERT_DIR}/mlm_output)."
        log_error "  Did train_mnm run and succeed? To deliberately bypass GraphMERT tails, set GRAPHMERT_PREDICT_TAILS_GM_REQUIRED=0."
        return 1
    fi
    # predict_tails.py expects a checkpoint dir with config.json, OR a
    # parent dir containing checkpoint-* — its get_best_checkpoint picks
    # the latest. Pass the parent so run_mlm naming variants (best/, last/,
    # checkpoint-NNN/) all resolve.
    local RELATION_MAP="$GRAPHMERT_DIR/dataset/relation_map.json"
    if [[ ! -f "$RELATION_MAP" ]]; then
        log_error "predict_tails_gm: relation_map missing at $RELATION_MAP (run preprocess first)"
        return 1
    fi
    ( cd "$REPO_ROOT/2_graphmert" && \
      python -m utils.predict_tails \
          --model_dir    "$CKPT_ROOT" \
          --tokenizer    "$STABLE_TOKENIZER" \
          --relation_map "$RELATION_MAP" \
          --dataset      "$GRAPHMERT_DIR/llm_relations/relations_cleaned_eval" \
          --output_dir   "$GRAPHMERT_DIR/predictions_graphmert" \
          --topk         "${GM_PRED_TOPK:-20}" \
          --batch_size   "${GM_PRED_BATCH_SIZE:-8}" ) \
        || { log_error "predict_tails_gm failed"; return 1; }
    # KNOWN INTEGRATION GAP — do NOT assume GraphMERT tails reach the KG yet:
    # this step writes predictions_graphmert/predictions.parquet, but combine_tails
    # (step 7) reads ONLY predictions/predictions_shard*.csv (the predict_tails_llm
    # output) — a different dir AND a different format. Until a parquet->shard
    # bridge / merge is added to combine_tails, these predictions are produced but
    # NOT merged into the validated KG. Warn loudly so a green run isn't misread.
    log_warn "predict_tails_gm: wrote predictions_graphmert/predictions.parquet, but combine_tails (step 7) does not yet read it (reads predictions/*.csv). GraphMERT tails are NOT merged into the KG until that bridge is built."
}

step_validate_predictions() {
    log_info "graphmert :: validate_predictions (step 7 — combine_tails + fact_score)"
    log_info "  combine_tails"
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/combine_tails/combine_tails.py \
          --pred_dir   "$GRAPHMERT_DIR/predictions" \
          --output_dir "$GRAPHMERT_DIR/combined" ) || { log_error "combine_tails failed"; return 1; }
    if [[ -z "$VALIDATE_A" || -z "$VALIDATE_B" ]]; then
        log_error "fact_score needs models.validate_a and validate_b in configs/default.yaml"
        return 1
    fi
    log_info "  fact_score (two-LLM agreement)"
    # combine_tails writes final_kg_combined.csv (head/relation/tail columns,
    # merge + deduplicate only — no LLM filter, per upstream dc5bb46). fact_score
    # below is the sole quality gate on the expanded KG (two-LLM consensus).
    ( cd "$REPO_ROOT/2_graphmert" && \
      python utils/llm_scores/fact_score.py \
          --input_csv   "$GRAPHMERT_DIR/combined/final_kg_combined.csv" \
          --output_csv  "$GRAPHMERT_DIR/final_kg/validated_triples.csv" \
          --model_ids   "$VALIDATE_A" "$VALIDATE_B" \
          --batch_size  64 \
          --tensor_parallel_size 1 ) || { log_error "fact_score failed"; return 1; }
    # NOTE: --max_model_len omitted — fact_score.py reads the cap from
    # graphmert.fact_score_max_model_len (config, 4096). Single source of truth.
}

step_expand_kg() {
    log_info "graphmert :: expand_kg (merge validated tails into KG)"
    local VALIDATED="$GRAPHMERT_DIR/final_kg/validated_triples.csv"
    local SEED_KG="$GRAPHRAG_DIR/output/kg_final.parquet"
    # merge_kgs.py ALWAYS writes final_relationships.parquet into --outdir (it does
    # not take an output name). This merged seed ∪ validated-expansion KG is what
    # curriculum.path_traversal now consumes. The var previously said
    # expanded_kg.parquet — a file merge_kgs never writes — so the success log
    # named a nonexistent file and the merge looked orphaned.
    local FINAL="$GRAPHMERT_DIR/final_kg/final_relationships.parquet"
    if [[ -f "$VALIDATED" && -f "$SEED_KG" ]]; then
        ( cd "$REPO_ROOT/1_seed_kg" && \
          python merge_kgs.py \
              --new "$VALIDATED" --old "$SEED_KG" \
              --outdir "$(dirname "$FINAL")" ) || { log_error "expand_kg merge failed"; return 1; }
        log_info "Final expanded KG: $FINAL"
    else
        # curriculum.path_traversal consumes $FINAL; without it that step fails.
        # Don't pretend success — this is a real missing dependency.
        log_error "expand_kg: validated_triples.csv or seed kg missing — cannot produce the merged KG ($FINAL), which curriculum.path_traversal requires."
        return 1
    fi
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" "step_$step" || exit $?
done
