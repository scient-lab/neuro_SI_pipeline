#!/usr/bin/env bash
# Phase: extract - single-LLM extraction with closed vocabulary.
# Delegates to 1_seed_kg. Venv: graphrag.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

STEPS=(parse_pdf chunk extract_triples normalize cache)

source_venv graphrag

# --- Stage input corpus at graphrag's expected location ------------------
# graphrag_index.py reads from $REPO_ROOT/outputs/graphrag/input/. The
# profile-resolved corpus may live elsewhere (e.g. corpus/<domain>/<scale>/
# committed as fixtures). cp -r (not symlink) so this works on RunPod and
# any FS that doesn't honor symlinks across mounts.
input_dir=$(uv run --no-project --quiet --with pyyaml python3 -c \
    "import pipeline_config; print(pipeline_config.get_phase_param('extract','input_dir','') or '')" \
    2>/dev/null || true)

if [[ -n "$input_dir" && -d "$REPO_ROOT/$input_dir" ]]; then
    target="$REPO_ROOT/outputs/graphrag/input"
    mkdir -p "$target"
    # Copy .txt files only; explicitly skip READMEs and other metadata so
    # graphrag doesn't try to ingest them as documents.
    find "$REPO_ROOT/$input_dir" -maxdepth 1 -name '*.txt' -type f \
        -exec cp -t "$target" {} +
    n=$(find "$target" -maxdepth 1 -name '*.txt' -type f | wc -l)
    log_info "Staged input: $target (${n} .txt files from $input_dir)"
fi

for step in "${STEPS[@]}"; do
    if step_enabled "$step"; then
        log_info "extract :: $step (stub - wire to 1_seed_kg)"
        # TODO: dispatch to actual step here, e.g.:
        #   ( cd "$REPO_ROOT/1_seed_kg" && python graphrag_index.py --step "$step" )
    fi
done
