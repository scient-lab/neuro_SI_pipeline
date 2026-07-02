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
# shellcheck source=../lib/stage_corpus.sh
source "$SCRIPT_DIR/../lib/stage_corpus.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

# Conceptual steps for this phase. The graphrag_index.py implementation has
# 5 sequential steps (1..5) that map onto these as follows:
#   parse_pdf         no-op (we feed pre-extracted .txt files in input_dir)
#   chunk             graphrag step 1 (chunking)
#   extract_triples   graphrag step 2 (documents) + step 3 (LLM extraction)
#   build_graph       graphrag step 4 (parse responses)
#   finalize_seed_kg  graphrag step 5 (clean + finalize seed KG)
PHASE_NAME=extract
STEPS=(parse_pdf chunk extract_triples build_graph finalize_seed_kg)
PHASE_DESC="Build seed KG from text corpus (graphrag)"
STEP_DESCS=(
    "(no-op) corpus arrives pre-extracted as .txt"
    "Chunk text into base units (graphrag #1)"
    "vLLM extraction of head/relation/tail triples (graphrag #2+#3)"
    "Parse LLM responses into entity/relationship tables (graphrag #4)"
    "Clean + finalize seed KG; write kg_final.{csv,parquet} (graphrag #5)"
)

source_venv graphrag

# Stage the corpus + settings.yaml into graphrag's workspace. Factored into
# scripts/lib/stage_corpus.sh so the data-driven runner's parse_pdf entrypoint
# (scripts/entrypoints/extract_parse_pdf.py) runs the IDENTICAL logic — single
# source, zero divergence. OUTPUT_BASE/GRAPHRAG_DIR stay script-scope below for
# the step functions (graphrag_step / finalize_seed_kg).
OUTPUT_BASE=$(resolve_output_base)
GRAPHRAG_DIR="$OUTPUT_BASE/graphrag"
stage_corpus

# models.extract (the LLM-extraction model) is resolved INSIDE step_extract_triples — where
# run_step has set SI_PHASE/SI_STEP — so pipeline_config records it to the per-step config
# ledger that config.sh --models reads (a top-level resolve here would go unrecorded).

# Helper that runs a graphrag step.
graphrag_step() {
    local n="$1" extra_args="${2:-}"
    ( cd "$REPO_ROOT/1_seed_kg" && \
      python graphrag_index.py --root_dir "$GRAPHRAG_DIR" --step "$n" $extra_args )
}

# --- Steps ---------------------------------------------------------------
# Each step is a function returning non-zero on failure (NOT exit) so run_step
# can record status/timing/exit-code in the manifest and tee a per-step log.
step_parse_pdf() {
    log_info "extract :: parse_pdf (no-op — corpus is .txt; see scripts/pdf_to_text.sh in stash for OCR option)"
}

step_chunk() {
    log_info "extract :: chunk (graphrag step 1 — base text units)"
    graphrag_step 1 || { log_error "extract.chunk failed"; return 1; }
}

step_extract_triples() {
    log_info "extract :: extract_triples (graphrag step 2 + step 3)"
    graphrag_step 2 || { log_error "extract.extract_triples step 2 failed"; return 1; }
    local MODEL_ID
    MODEL_ID=$(get_model_id extract "")
    if [[ -z "$MODEL_ID" ]]; then
        log_error "extract.extract_triples step 3 needs models.extract in configs/default.yaml or domain override"
        return 1
    fi
    graphrag_step 3 "--model_id $MODEL_ID" || { log_error "extract.extract_triples step 3 failed"; return 1; }
}

step_build_graph() {
    log_info "extract :: build_graph (graphrag step 4 — parse LLM responses)"
    graphrag_step 4 || { log_error "extract.build_graph failed"; return 1; }
}

step_finalize_seed_kg() {
    log_info "extract :: finalize_seed_kg (graphrag step 5 — finalize seed KG)"
    graphrag_step 5 || { log_error "extract.finalize_seed_kg failed"; return 1; }
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
" ) || { log_error "extract.finalize_seed_kg write_seed_kg failed"; return 1; }
    log_info "Seed KG written: $GRAPHRAG_DIR/output/kg_final.{csv,parquet}"
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" "step_$step" || exit $?
done
