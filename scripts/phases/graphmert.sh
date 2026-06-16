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

STEPS=(tokenize preprocess train_mnm predict_tails validate_predictions expand_kg)

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

# YAML config for run_dataset_preprocessing + run_mlm (HF-argparse YAML).
# Operator must point this at a valid file with paths substituted for the
# <YOUR_SCRATCH> placeholders.
ARGS_MLM_YAML="${ARGS_MLM_YAML:-$REPO_ROOT/2_graphmert/launch_configs/args_mlm.yaml}"

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    if ! step_enabled "$step"; then continue; fi

    case "$step" in
        tokenize)
            log_info "graphmert :: tokenize (step 1 — run_tokenization.py)"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python run_tokenization.py \
                  --input_dir   "$GRAPHRAG_DIR/input" \
                  --output_dir  "$GRAPHMERT_DIR" \
                  --tokenizer   "${TOKENIZER_BASE:-dmis-lab/biobert-base-cased-v1.2}" \
                  --max_seq_length 128 \
                  --validation_split_pct 5 \
                  --num_workers 8 \
                  --seed 0 ) || { log_error "graphmert.tokenize failed"; exit 1; }
            ;;

        preprocess)
            log_info "graphmert :: preprocess (steps 2+3+4)"
            # Step 2: entity_discovery + find_heads_positions
            TOK_TRAIN=$(ls -d "$GRAPHMERT_DIR/tokenized_inputs/train_"* 2>/dev/null | head -1)
            if [[ -z "$TOK_TRAIN" ]]; then
                log_error "tokenized train dir not found under $GRAPHMERT_DIR/tokenized_inputs/"
                exit 1
            fi
            if [[ -z "$EXTRACT_MODEL_ID" ]]; then
                log_error "preprocess needs models.extract in configs/default.yaml"
                exit 1
            fi
            log_info "  step 2a: entity_discovery"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python utils/entity_discovery/entity_discovery.py \
                  --tokenized_dir "$TOK_TRAIN" \
                  --output_dir    "$GRAPHMERT_DIR/entity_discovery" \
                  --model_id      "$EXTRACT_MODEL_ID" \
                  --tokenizer     "$STABLE_TOKENIZER" ) || { log_error "entity_discovery failed"; exit 1; }
            log_info "  step 2b: find_heads_positions"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python utils/entity_discovery/find_heads_positions.py \
                  --heads_chunks_dir "$GRAPHMERT_DIR/entity_discovery" \
                  --output_dir       "$GRAPHMERT_DIR/head_positions" \
                  --tokenizer        "$STABLE_TOKENIZER" ) || { log_error "find_heads_positions failed"; exit 1; }
            # Step 3: add_llm_relations + clean_llm_relations
            log_info "  step 3a: add_llm_relations"
            # find_heads_positions.py hardcodes the subdir name
            # "neuro_heads_all_with_positions" (Princeton-side biomed naming;
            # find_heads_positions.py:107). add_llm_relations expects that
            # exact Dataset directory, not its parent.
            ( cd "$REPO_ROOT/2_graphmert" && \
              python utils/relation_matching/add_llm_relations.py \
                  --dataset_path "$GRAPHMERT_DIR/head_positions/neuro_heads_all_with_positions" \
                  --output_root  "$GRAPHMERT_DIR/llm_relations" \
                  --output_name  relations_all \
                  --model_id     "$EXTRACT_MODEL_ID" \
                  --tokenizer    "$STABLE_TOKENIZER" ) || { log_error "add_llm_relations failed"; exit 1; }
            log_info "  step 3b: clean_llm_relations"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python utils/relation_matching/clean_llm_relations.py \
                  --input_dir  "$GRAPHMERT_DIR/llm_relations/relations_all" \
                  --output_dir "$GRAPHMERT_DIR/llm_relations/relations_clean" \
                  --tokenizer  "$STABLE_TOKENIZER" ) || { log_error "clean_llm_relations failed"; exit 1; }
            # Step 4: run_dataset_preprocessing (co-occurrence grounding)
            # clean_llm_relations.py hardcodes subdir names "relations_cleaned_train"
            # / "relations_cleaned_eval" under --output_dir (note "cleaned" not
            # "clean"; clean_llm_relations.py:~85). Step 4 reads those exact
            # dataset directories — must include the full subpath here.
            log_info "  step 4: run_dataset_preprocessing"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python run_dataset_preprocessing.py \
                  --yaml_file    "$ARGS_MLM_YAML" \
                  --seed_kg_path "$GRAPHRAG_DIR/output/kg_final.parquet" \
                  --train_src    "$GRAPHMERT_DIR/llm_relations/relations_clean/relations_cleaned_train" \
                  --eval_src     "$GRAPHMERT_DIR/llm_relations/relations_clean/relations_cleaned_eval" \
                  --tokenizer    "$STABLE_TOKENIZER" \
                  --output_dir   "$GRAPHMERT_DIR/dataset" ) || { log_error "run_dataset_preprocessing failed"; exit 1; }
            ;;

        train_mnm)
            log_info "graphmert :: train_mnm (step 5 — run_mlm.py)"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python run_mlm.py "$ARGS_MLM_YAML" ) || { log_error "train_mnm failed"; exit 1; }
            ;;

        predict_tails)
            log_info "graphmert :: predict_tails (step 6 — predict_tails_llm.py)"
            CHKPT="${MLM_CHECKPOINT:-$GRAPHMERT_DIR/checkpoints/best}"
            # --dataset is consumed by load_from_disk → must point at the
            # actual Dataset dir, which clean_llm_relations.py writes as
            # <output_dir>/relations_cleaned_eval (same naming as step 4).
            ( cd "$REPO_ROOT/2_graphmert" && \
              python predict_tails_llm.py \
                  --model_id   "$CHKPT" \
                  --tokenizer  "$STABLE_TOKENIZER" \
                  --dataset    "$GRAPHMERT_DIR/llm_relations/relations_clean/relations_cleaned_eval" \
                  --output_dir "$GRAPHMERT_DIR/predictions" \
                  --num_shards "${PRED_NUM_SHARDS:-1}" \
                  --shard_id   "${PRED_SHARD_ID:-0}" ) || { log_error "predict_tails failed"; exit 1; }
            ;;

        validate_predictions)
            log_info "graphmert :: validate_predictions (step 7 — combine_tails + fact_score)"
            log_info "  combine_tails"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python utils/combine_tails/combine_tails.py \
                  --pred_dir   "$GRAPHMERT_DIR/predictions" \
                  --output_dir "$GRAPHMERT_DIR/combined" \
                  --model_id   "$EXTRACT_MODEL_ID" \
                  --tokenizer  "$STABLE_TOKENIZER" ) || { log_error "combine_tails failed"; exit 1; }
            if [[ -z "$VALIDATE_A" || -z "$VALIDATE_B" ]]; then
                log_error "fact_score needs models.validate_a and validate_b in configs/default.yaml"
                exit 1
            fi
            log_info "  fact_score (two-LLM agreement)"
            ( cd "$REPO_ROOT/2_graphmert" && \
              python utils/llm_scores/fact_score.py \
                  --input_csv   "$GRAPHMERT_DIR/combined/expanded_triples.csv" \
                  --output_csv  "$GRAPHMERT_DIR/final_kg/validated_triples.csv" \
                  --model_ids   "$VALIDATE_A" "$VALIDATE_B" \
                  --batch_size  64 \
                  --max_model_len 4096 \
                  --tensor_parallel_size 1 ) || { log_error "fact_score failed"; exit 1; }
            ;;

        expand_kg)
            log_info "graphmert :: expand_kg (merge validated tails into KG)"
            VALIDATED="$GRAPHMERT_DIR/final_kg/validated_triples.csv"
            SEED_KG="$GRAPHRAG_DIR/output/kg_final.parquet"
            FINAL="$GRAPHMERT_DIR/final_kg/expanded_kg.parquet"
            if [[ -f "$VALIDATED" && -f "$SEED_KG" ]]; then
                ( cd "$REPO_ROOT/1_seed_kg" && \
                  python merge_kgs.py \
                      --new "$VALIDATED" --old "$SEED_KG" \
                      --outdir "$(dirname "$FINAL")" ) || { log_error "expand_kg merge failed"; exit 1; }
            else
                log_warn "expand_kg: validated_triples.csv or seed kg missing; skipping merge"
            fi
            log_info "Final expanded KG: $FINAL"
            ;;
    esac
done
