#!/bin/bash
# Create the three uv-managed virtual environments for the pipeline.
#
# Usage:
#   ./setup.sh              # create all three envs under ./.venvs/
#   ./setup.sh graphmert    # (re)create a single env: graphrag | graphmert | si_curriculum
#
# Each env is isolated (different Python versions + pinned cu121 wheels). uv
# resolves and installs these far faster than conda.

set -euo pipefail

# Repo root = directory containing this script (overridable via REPO_DIR).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_ROOT="${REPO_DIR}/.venvs"

# Sibling graphmert_umls repo — ships the vendored `graphrag` package that
# 1_seed_kg/graphrag_index.py imports. The bootstrap clones it next to
# REPO_DIR; users on a workstation can override via the env var.
GRAPHMERT_UMLS_ROOT="${GRAPHMERT_UMLS_ROOT:-$(cd "${REPO_DIR}/.." && pwd)/graphmert_umls}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH. Install it once with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "(on an HPC cluster, 'module load uv' may also work)" >&2
  exit 1
fi

# env name | python version | requirements file | post-install hook
create_env() {
  local name="$1" python="$2" reqs="$3" post="${4:-}"
  local venv="${VENV_ROOT}/${name}"

  echo "==> ${name} (Python ${python})"
  uv venv --python "${python}" "${venv}"
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
  # --index-strategy unsafe-best-match: required because our requirements
  # files use --extra-index-url for cu121 torch wheels. The cu121 index has
  # OLD versions of common packages (e.g. tqdm 4.66.5). uv's default
  # "first-index-wins" then refuses to use newer versions from PyPI even
  # when a transitive dep (e.g. graphrag's tqdm>=4.67) requires it. With
  # unsafe-best-match, uv considers all configured indexes and picks the
  # best version regardless of which index it came from. Safe in our
  # use case (all indexes are first-party / PyPI).
  uv pip install --index-strategy unsafe-best-match -r "${REPO_DIR}/${reqs}"
  if [[ -n "${post}" ]]; then
    eval "${post}"
  fi
  deactivate
  echo "    done: ${venv}"
  echo
}

target="${1:-all}"

case "${target}" in
  graphrag|graphmert|si_curriculum|all) ;;
  *)
    echo "Unknown env '${target}'. Choose: graphrag | graphmert | si_curriculum | all" >&2
    exit 1
    ;;
esac

if [[ "${target}" == "all" || "${target}" == "graphrag" ]]; then
  # Verify the vendored graphrag exists before creating the venv — we install
  # it editably as a post-step (no PyPI graphrag; matches the version Princeton
  # tested against). Missing = bootstrap didn't clone graphmert_umls; bail with
  # a clear message rather than a confusing pip error.
  if [[ ! -f "${GRAPHMERT_UMLS_ROOT}/graphrag/pyproject.toml" ]]; then
    echo "graphmert_umls/graphrag not found at: ${GRAPHMERT_UMLS_ROOT}/graphrag" >&2
    echo "Clone it as a sibling of this repo, e.g.:" >&2
    echo "  git clone -b dev git@github.com:scient-lab/graphmert_umls.git ${GRAPHMERT_UMLS_ROOT}" >&2
    echo "Or set GRAPHMERT_UMLS_ROOT to point at your existing checkout." >&2
    exit 1
  fi
  create_env graphrag 3.11 1_seed_kg/requirements.txt \
    "uv pip install --index-strategy unsafe-best-match -e '${GRAPHMERT_UMLS_ROOT}/graphrag'"
fi
if [[ "${target}" == "all" || "${target}" == "graphmert" ]]; then
  create_env graphmert     3.10 2_graphmert/requirements.txt "python -m spacy download en_core_web_sm"
fi
if [[ "${target}" == "all" || "${target}" == "si_curriculum" ]]; then
  create_env si_curriculum 3.10 3_si_curriculum/requirements.txt
fi

echo "All requested environments created under ${VENV_ROOT}/"
echo "Activate one with: source ${VENV_ROOT}/<name>/bin/activate"
