#!/usr/bin/env bash
# Phase: extract - single-LLM extraction with closed vocabulary.
# Delegates to 1_seed_kg/graphrag_index.py.
# Venv: graphrag.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

# Conceptual steps for this phase. The graphrag_index.py implementation has
# 5 sequential steps (1..5) that map onto these as follows:
#   parse_pdf         no-op (we feed pre-extracted .txt files in input_dir)
#   chunk             graphrag step 1 (chunking)
#   extract_triples   graphrag step 2 (documents) + step 3 (LLM extraction)
#   normalize         graphrag step 4 (parse responses)
#   cache             graphrag step 5 (clean + finalize seed KG)
STEPS=(parse_pdf chunk extract_triples normalize cache)

source_venv graphrag

# --- Stage input corpus at graphrag's expected location ------------------
# graphrag_index.py reads from $OUTPUT_BASE/graphrag/input/. The profile-
# resolved corpus may live elsewhere (e.g. corpus/<domain>/<scale>/
# committed as fixtures). cp -r (not symlink) so this works on RunPod and
# any FS that doesn't honor symlinks across mounts.
OUTPUT_BASE=$(resolve_output_base)
GRAPHRAG_DIR="$OUTPUT_BASE/graphrag"

# Input path resolution. Symmetric model:
#     local:  ${REPO_ROOT}/${CORPUS_PATH}
#     cloud:  ${S3_URI}/${CORPUS_PATH}
# Precedence (highest first):
#   1. profile YAML extract.input_dir   (smoke fixture override)
#   2. $CORPUS_PATH env (from .env)      (pilot/paper — operator's choice)
#   3. default: corpus/${SI_DOMAIN}/source_txt
# CORPUS_PATH can be a directory OR a single .txt file.
# REPO_ROOT is exported by pipeline.sh (computed from script location).
# On the pod it equals SI_HOME; on the workstation, SI_HOME may be unset.
INPUT_DIR_REPO=$(get_phase_param extract input_dir "")
if [[ -z "$INPUT_DIR_REPO" ]]; then
    INPUT_DIR_REPO="${CORPUS_PATH:-corpus/${SI_DOMAIN:-neuroscience}/source_txt}"
    log_info "Using CORPUS_PATH-derived input: $INPUT_DIR_REPO"
fi

ABS_INPUT="$REPO_ROOT/$INPUT_DIR_REPO"
ABS_INPUT="${ABS_INPUT%/}"

# Auto-pull from S3 when local is missing/empty AND we have both env vars
# set. Skip auto-pull for committed fixtures (paths containing /smoke/).
need_pull=0
if [[ "$INPUT_DIR_REPO" == *"/smoke/"* || "$INPUT_DIR_REPO" == *"/smoke" ]]; then
    : # committed fixture, never pull
elif [[ -n "${S3_URI:-}" ]]; then
    if [[ "$ABS_INPUT" == *.txt ]]; then
        [[ -f "$ABS_INPUT" ]] || need_pull=1
    else
        n_txt=$(find "$ABS_INPUT" -maxdepth 1 -name '*.txt' -type f 2>/dev/null | wc -l)
        [[ "$n_txt" -eq 0 ]] && need_pull=1
    fi
fi

if [[ "$need_pull" -eq 1 ]]; then
    log_info "Local $INPUT_DIR_REPO is missing/empty — pulling ${S3_URI%/}/$INPUT_DIR_REPO"
    CORPUS_PATH="$INPUT_DIR_REPO" \
        "$REPO_ROOT/scripts/data_prep/sync_corpus.sh" --pull \
        || { log_error "S3 corpus pull failed"; exit 1; }
fi

# Stage into graphrag's input dir. Handles both single-file and directory modes.
mkdir -p "$GRAPHRAG_DIR/input"
if [[ -f "$ABS_INPUT" ]]; then
    cp "$ABS_INPUT" "$GRAPHRAG_DIR/input/"
elif [[ -d "$ABS_INPUT" ]]; then
    find "$ABS_INPUT" -maxdepth 1 -name '*.txt' -type f \
        -exec cp -t "$GRAPHRAG_DIR/input" {} +
else
    log_error "Input path not found: $ABS_INPUT"
    log_error "  Set CORPUS_PATH in .env / .env.runpod, or drop files locally."
    exit 1
fi
n=$(find "$GRAPHRAG_DIR/input" -maxdepth 1 -name '*.txt' -type f | wc -l)
if [[ "$n" -eq 0 ]]; then
    log_error "No .txt files staged from $INPUT_DIR_REPO"
    exit 1
fi
log_info "Staged input: $GRAPHRAG_DIR/input (${n} .txt files from $INPUT_DIR_REPO)"

# graphrag_index.py expects settings.yaml at --root_dir; copy from the
# bundled 1_seed_kg/settings.yaml if not already present.
if [[ ! -f "$GRAPHRAG_DIR/settings.yaml" ]]; then
    mkdir -p "$GRAPHRAG_DIR"
    cp "$REPO_ROOT/1_seed_kg/settings.yaml" "$GRAPHRAG_DIR/settings.yaml"
    log_info "Staged settings.yaml at $GRAPHRAG_DIR/settings.yaml"
fi

# Model for step 3 (LLM extraction). vllm-loadable path (HF id or local).
MODEL_ID=$(get_model_id extract "")

# Helper that runs a graphrag step.
graphrag_step() {
    local n="$1" extra_args="${2:-}"
    ( cd "$REPO_ROOT/1_seed_kg" && \
      python graphrag_index.py --root_dir "$GRAPHRAG_DIR" --step "$n" $extra_args )
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    if ! step_enabled "$step"; then continue; fi

    case "$step" in
        parse_pdf)
            log_info "extract :: parse_pdf (no-op — corpus is .txt; see scripts/pdf_to_text.sh in stash for OCR option)"
            ;;
        chunk)
            log_info "extract :: chunk (graphrag step 1 — base text units)"
            graphrag_step 1 || { log_error "extract.chunk failed"; exit 1; }
            ;;
        extract_triples)
            log_info "extract :: extract_triples (graphrag step 2 + step 3)"
            graphrag_step 2 || { log_error "extract.extract_triples step 2 failed"; exit 1; }
            if [[ -z "$MODEL_ID" ]]; then
                log_error "extract.extract_triples step 3 needs models.extract in configs/default.yaml or domain override"
                exit 1
            fi
            graphrag_step 3 "--model_id $MODEL_ID" || { log_error "extract.extract_triples step 3 failed"; exit 1; }
            ;;
        normalize)
            log_info "extract :: normalize (graphrag step 4 — parse LLM responses)"
            graphrag_step 4 || { log_error "extract.normalize failed"; exit 1; }
            ;;
        cache)
            log_info "extract :: cache (graphrag step 5 — finalize seed KG)"
            graphrag_step 5 || { log_error "extract.cache failed"; exit 1; }
            # graphrag writes final_relationships.parquet (cols source/target/relation),
            # but downstream code expects:
            #   - kg_final.csv      (head, relation, tail) — for graphmert step 4 + curriculum calculate_hops
            #   - kg_final.parquet  (head, relation, tail) — for graphmert merge_kgs
            # Materialize both here so all consumers can use $GRAPHRAG_DIR/output/kg_final.*.
            log_info "extract :: write_seed_kg (convert graphrag → kg_final.{csv,parquet})"
            ( cd "$REPO_ROOT" && python3 -c "
import pandas as pd, sys
src = '$GRAPHRAG_DIR/output/final_relationships.parquet'
df = pd.read_parquet(src)
out = df[['source','target','relation']].rename(columns={'source':'head','target':'tail'})
out.to_csv('$GRAPHRAG_DIR/output/kg_final.csv', index=False)
out.to_parquet('$GRAPHRAG_DIR/output/kg_final.parquet', index=False)
print(f'wrote {len(out)} triples to kg_final.csv and kg_final.parquet')
" ) || { log_error "extract.cache write_seed_kg failed"; exit 1; }
            log_info "Seed KG written: $GRAPHRAG_DIR/output/kg_final.{csv,parquet}"
            ;;
    esac
done
