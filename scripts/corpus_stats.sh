#!/usr/bin/env bash
# corpus_stats.sh — thin wrapper for scripts/corpus_stats.py that runs it under a
# venv python which has `transformers` (so --tokenizer gives exact counts). Falls
# back to system python3 for the dependency-free estimate path.
#
# Usage (args pass straight through to corpus_stats.py):
#   ./scripts/corpus_stats.sh corpus/space/smoke
#   ./scripts/corpus_stats.sh corpus/space/smoke --per-file
#   ./scripts/corpus_stats.sh corpus/space/smoke --tokenizer          # Qwen/Qwen3-8B exact
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY=""
for v in si_curriculum graphrag graphmert; do
    cand="$SCRIPT_DIR/../.venvs/$v/bin/python"
    if [[ -x "$cand" ]] && "$cand" -c 'import transformers' 2>/dev/null; then
        PY="$cand"; break
    fi
done
[[ -z "$PY" ]] && PY="$(command -v python3 || command -v python)"
[[ -z "$PY" ]] && { echo "no python found" >&2; exit 1; }

exec "$PY" "$SCRIPT_DIR/corpus_stats.py" "$@"
