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
PHASE_NAME=extract
STEPS=(parse_pdf chunk extract_triples normalize cache)
PHASE_DESC="Build seed KG from text corpus (graphrag)"
STEP_DESCS=(
    "(no-op) corpus arrives pre-extracted as .txt"
    "Chunk text into base units (graphrag #1)"
    "vLLM extraction of head/relation/tail triples (graphrag #2+#3)"
    "Parse LLM responses into entity/relationship tables (graphrag #4)"
    "Clean + finalize seed KG; write kg_final.{csv,parquet} (graphrag #5)"
)

source_venv graphrag

# --- Stage input corpus at graphrag's expected location ------------------
# graphrag_index.py reads from $OUTPUT_BASE/graphrag/input/. The profile-
# resolved corpus may live elsewhere (e.g. corpus/<domain>/<scale>/
# committed as fixtures). cp -r (not symlink) so this works on RunPod and
# any FS that doesn't honor symlinks across mounts.
OUTPUT_BASE=$(resolve_output_base)
GRAPHRAG_DIR="$OUTPUT_BASE/graphrag"

# Corpus location — CORPUS_PATH is the SINGLE, REQUIRED source. There is no
# profile input_dir and no magic default: under orchestration (e.g. an AWS Step
# Functions workflow) the caller knows where the corpus is mounted / staged in
# S3 and MUST set CORPUS_PATH. Fail fast and loud if it's missing rather than
# silently reading the wrong or empty corpus.
#   relative: ${REPO_ROOT}/${CORPUS_PATH}   cloud: ${S3_URI}/${CORPUS_PATH}
#   absolute: ${CORPUS_PATH} used as-is (a local mount / external dir, local-only)
# CORPUS_PATH may be a directory of .txt files OR a single .txt file, and may use
# {SI_DOMAIN} / {SI_PROFILE} tokens (expanded at runtime against the effective
# domain/profile — e.g. corpus/{SI_DOMAIN}/smoke).
# REPO_ROOT is exported by pipeline.sh; on the pod it equals SI_HOME.
if [[ -z "${CORPUS_PATH:-}" ]]; then
    log_error "CORPUS_PATH is not set — it is REQUIRED (no input_dir, no default)."
    log_error "  point it at the corpus dir or .txt file, e.g.:"
    log_error "    local smoke : CORPUS_PATH=corpus/${SI_DOMAIN:-<domain>}/smoke"
    log_error "    domain-tmpl : CORPUS_PATH=corpus/{SI_DOMAIN}/smoke   (expanded at runtime)"
    log_error "    absolute    : CORPUS_PATH=/mnt/data/my_corpus        (local mount, used as-is)"
    log_error "    orchestrated: CORPUS_PATH=<mounted dir or S3 prefix under \$S3_URI>"
    exit 1
fi

# Defensive: if invoked standalone (not via pipeline.sh) the {SI_DOMAIN}/
# {SI_PROFILE} tokens won't have been expanded yet. Idempotent — a no-op once
# pipeline.sh has already expanded (no tokens remain).
CORPUS_PATH="$(expand_path_tokens "$CORPUS_PATH" "${SI_DOMAIN:-}" "${SI_PROFILE:-}")" \
    || { log_error "CORPUS_PATH token expansion failed"; exit 1; }
INPUT_DIR_REPO="$CORPUS_PATH"

# Absolute CORPUS_PATH (leading /) is used as-is; relative is anchored at
# REPO_ROOT (S3-mirror model). corpus_abs_path trims the trailing slash.
ABS_INPUT="$(corpus_abs_path "$INPUT_DIR_REPO" "$REPO_ROOT")"

# Auto-pull from S3 when local is missing/empty AND we have both env vars
# set. Skip auto-pull for committed fixtures (paths containing /smoke/).
need_pull=0
if [[ "$INPUT_DIR_REPO" == /* ]]; then
    : # absolute = local mount / external dir; not an S3-relative prefix, never pull
elif [[ "$INPUT_DIR_REPO" == *"/smoke/"* || "$INPUT_DIR_REPO" == *"/smoke" ]]; then
    : # committed fixture, never pull
elif [[ -n "${S3_URI:-}" ]]; then
    if [[ "$ABS_INPUT" == *.txt ]]; then
        [[ -f "$ABS_INPUT" ]] || need_pull=1
    else
        n_txt=$(find "$ABS_INPUT" -name '*.txt' -type f 2>/dev/null | wc -l)   # recursive
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
    # Recurse subdirs (CORPUS_PATH may hold nested corpus dirs). Flatten each
    # file's relative path into the staged name so files in different subdirs
    # don't collide (core/Sun.txt + Sun.txt -> core_Sun.txt + Sun.txt). graphrag
    # reads a FLAT input dir, hence the count below stays -maxdepth 1.
    while IFS= read -r -d '' f; do
        rel="${f#"$ABS_INPUT"/}"
        cp "$f" "$GRAPHRAG_DIR/input/${rel//\//_}"
    done < <(find "$ABS_INPUT" -name '*.txt' -type f -print0)
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

# Advisory scale check. extract.expected_input_docs (configs/profiles/<p>.yaml)
# documents the corpus size this profile is sized for. It is NOT enforced — there
# is no input cap in code, so extract processes ALL $n docs. Warn LOUDLY when the
# corpus exceeds the expectation so an oversized corpus on a paid pod doesn't
# silently blow up runtime/cost.
_expected_docs=$(get_phase_param extract expected_input_docs 0)
if [[ "${_expected_docs:-0}" -gt 0 && "$n" -gt "$_expected_docs" ]]; then
    log_warn "########################################################################"
    log_warn "# CORPUS LARGER THAN PROFILE EXPECTS"
    log_warn "#   staged $n .txt docs, but profile '${SI_PROFILE:-default}' is sized"
    log_warn "#   for ~${_expected_docs} (extract.expected_input_docs)."
    log_warn "#   This is ADVISORY, NOT a cap — extract will process ALL $n docs, so"
    log_warn "#   this run takes longer / costs more than the profile implies."
    log_warn "#   Trim CORPUS_PATH or use a larger profile if that's not intended."
    log_warn "########################################################################"
fi
log_info "Staged input: $GRAPHRAG_DIR/input (${n} .txt files from $INPUT_DIR_REPO)"

# graphrag_index.py expects settings.yaml at --root_dir; copy from the
# bundled 1_seed_kg/settings.yaml if not already present.
if [[ ! -f "$GRAPHRAG_DIR/settings.yaml" ]]; then
    mkdir -p "$GRAPHRAG_DIR"
    cp "$REPO_ROOT/1_seed_kg/settings.yaml" "$GRAPHRAG_DIR/settings.yaml"
    log_info "Staged settings.yaml at $GRAPHRAG_DIR/settings.yaml"
fi

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

step_normalize() {
    log_info "extract :: normalize (graphrag step 4 — parse LLM responses)"
    graphrag_step 4 || { log_error "extract.normalize failed"; return 1; }
}

step_cache() {
    log_info "extract :: cache (graphrag step 5 — finalize seed KG)"
    graphrag_step 5 || { log_error "extract.cache failed"; return 1; }
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
" ) || { log_error "extract.cache write_seed_kg failed"; return 1; }
    log_info "Seed KG written: $GRAPHRAG_DIR/output/kg_final.{csv,parquet}"
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" "step_$step" || exit $?
done
