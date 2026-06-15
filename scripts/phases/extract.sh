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
INPUT_DIR_REPO=$(get_phase_param extract input_dir "")

if [[ -n "$INPUT_DIR_REPO" && -d "$REPO_ROOT/$INPUT_DIR_REPO" ]]; then
    mkdir -p "$GRAPHRAG_DIR/input"
    find "$REPO_ROOT/$INPUT_DIR_REPO" -maxdepth 1 -name '*.txt' -type f \
        -exec cp -t "$GRAPHRAG_DIR/input" {} +
    n=$(find "$GRAPHRAG_DIR/input" -maxdepth 1 -name '*.txt' -type f | wc -l)
    log_info "Staged input: $GRAPHRAG_DIR/input (${n} .txt files from $INPUT_DIR_REPO)"
fi

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
            log_info "Seed KG written: $GRAPHRAG_DIR/output/kg_final.parquet"
            ;;
    esac
done
