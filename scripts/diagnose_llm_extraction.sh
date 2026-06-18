#!/usr/bin/env bash
# scripts/diagnose_llm_extraction.sh — replay graphrag's entity+relationship
# extraction against any vLLM endpoint and tell you why it's producing 0
# relationships. Wraps 1_seed_kg/diagnose_llm_extraction.py with sane defaults.
#
# When extract phase produces 0 relationships (kg_final.csv empty), run this
# to localize: (a) prompt issue, (b) parser strictness, (c) model capability.
# All args passthrough to the Python tool. See its --help for full list.
#
# Quickest invocation (uses .env.runpod's VLLM_ENDPOINT_URL + VLLM_API_KEY,
# default text):
#   ./scripts/diagnose_llm_extraction.sh
#
# Test a real chunk from your corpus:
#   ./scripts/diagnose_llm_extraction.sh --file corpus/neuroscience/source_txt/purves_neuroscience_3e.txt
#   # ^ uses the whole file; trim it first if too long for max_tokens
#
# Save a JSON report (for paper appendix / regression tracking):
#   ./scripts/diagnose_llm_extraction.sh --out outputs/diagnose/extract_$(date +%Y%m%d).json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use any python that can resolve `import prompts_kg` from 1_seed_kg/. The
# tool itself is stdlib-only, so any python3 works. Prefer the graphrag venv
# (matches the indexer's env), then system python3.
PY=""
for cand in "$REPO_ROOT/.venvs/graphrag/bin/python" \
            "$(command -v python3 2>/dev/null || echo /nonexistent)"; do
    [[ -x "$cand" ]] && { PY="$cand"; break; }
done
[[ -z "$PY" ]] && { echo "no python3 found"; exit 1; }

exec "$PY" "$REPO_ROOT/1_seed_kg/diagnose_llm_extraction.py" "$@"
