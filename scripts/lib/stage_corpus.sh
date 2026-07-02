#!/usr/bin/env bash
# scripts/lib/stage_corpus.sh — corpus staging for the extract phase.
#
# Factored out of scripts/phases/extract.sh (its phase-level preamble) so that
# BOTH the bash phase path AND the data-driven runner's extract.parse_pdf
# entrypoint (scripts/entrypoints/extract_parse_pdf.py) run the IDENTICAL logic —
# single source, ZERO divergence. Chosen over re-deriving this in python
# (DATA_DRIVEN_PIPELINE_EXECUTOR_PLAN Phase B, "shared stage_corpus.sh" decision):
# corpus staging is inherently bash (find/cp/S3 sync + common.sh helpers), and a
# python re-implementation risks the parity drift that has burned a paid pilot.
#
# Resolves CORPUS_PATH (fail-loud), expands {SI_DOMAIN}/{SI_PROFILE} tokens,
# auto-pulls from S3 when local is missing/empty, stages + flattens .txt into
# $OUTPUT_BASE/graphrag/input/, warns (advisory) on an oversized corpus, and
# stages settings.yaml. Idempotent.
#
# Env in : REPO_ROOT, CORPUS_PATH (REQUIRED), OUTPUT_BASE (else repo/outputs),
#          SI_DOMAIN, SI_PROFILE, S3_URI (optional).
# Produces: $OUTPUT_BASE/graphrag/input/*.txt + $OUTPUT_BASE/graphrag/settings.yaml
#
# Two modes:
#   sourced    — extract.sh sources this and calls stage_corpus (deps already loaded).
#   executed   — `bash stage_corpus.sh` (the parse_pdf entrypoint): the guard at the
#                bottom bootstraps the SAME deps extract.sh loads, then stages.

# stage_corpus — the extract phase-level corpus preamble (verbatim), wrapped so
# it can be invoked once from either caller. Requires common.sh helpers
# (log_*, get_phase_param, resolve_output_base, expand_path_tokens,
# corpus_abs_path) to be sourced already.
stage_corpus() {
    # --- Stage input corpus at graphrag's expected location ------------------
    # graphrag_index.py reads from $OUTPUT_BASE/graphrag/input/. The profile-
    # resolved corpus may live elsewhere (e.g. corpus/<domain>/<scale>/
    # committed as fixtures). cp -r (not symlink) so this works on RunPod and
    # any FS that doesn't honor symlinks across mounts.
    # NOTE: OUTPUT_BASE is deliberately NOT `local` — resolve_output_base reads the
    # OUTPUT_BASE var by name, and bash dynamic scoping means a `local OUTPUT_BASE`
    # here (empty) would SHADOW the operator's env var and force the default dir.
    local GRAPHRAG_DIR INPUT_DIR_REPO ABS_INPUT need_pull n_txt rel f n _expected_docs
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
}

# --- Executable mode -------------------------------------------------------
# When run directly (the parse_pdf entrypoint does `bash stage_corpus.sh` under
# the graphrag venv), bootstrap the SAME deps extract.sh loads, then stage. When
# SOURCED (extract.sh), this block is skipped and only the function is defined.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    _SC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export REPO_ROOT="${REPO_ROOT:-$(cd "$_SC_DIR/../.." && pwd)}"
    # shellcheck source=./common.sh
    source "$_SC_DIR/common.sh"
    # shellcheck source=./venv.sh
    source "$_SC_DIR/venv.sh"
    source_venv graphrag
    stage_corpus
fi
