#!/usr/bin/env bash
# conformance.sh — verify the code STEPS in scripts/phases/*.sh still map onto
# the catalog ids in configs/pipeline_catalog.yaml (the single authored source).
#
# The catalog YAML is the single source of pipeline structure. This gate checks
# the code conforms to it, absorbing the documented, temporary migration gaps
# declared in the YAML (id_aliases + split_pending). It exits 2 on UNEXPECTED
# drift — wire it into CI / pre-commit and any step-rename workflow.
#
# It reads the YAML (needs pyyaml — a dev/CI-time dep, NOT the pipeline hot
# path), so run it in a pipeline venv or override the interpreter with PYTHON=…
#
#   scripts/conformance.sh              # exit 2 on unexpected drift (CI gate)
#   scripts/conformance.sh --warn-only  # report drift, exit 0
#
# NOTE: there is no committed pipeline_catalog.json. It is a pure yaml->json
# derive, regenerated on demand (`manifest.py catalog --out …`) for stdlib
# readers / S3+UI publish, and is gitignored — never a second source.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"${PYTHON:-python3}" "$REPO_ROOT/scripts/lib/manifest.py" conformance \
    --catalog-yaml "$REPO_ROOT/configs/pipeline_catalog.yaml" \
    --phases-dir   "$REPO_ROOT/scripts/phases" \
    "$@"
