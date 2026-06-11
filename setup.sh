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
  uv pip install -r "${REPO_DIR}/${reqs}"
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
  create_env graphrag      3.11 1_seed_kg/requirements.txt
fi
if [[ "${target}" == "all" || "${target}" == "graphmert" ]]; then
  create_env graphmert     3.10 2_graphmert/requirements.txt "python -m spacy download en_core_web_sm"
fi
if [[ "${target}" == "all" || "${target}" == "si_curriculum" ]]; then
  create_env si_curriculum 3.10 3_si_curriculum/requirements.txt
fi

echo "All requested environments created under ${VENV_ROOT}/"
echo "Activate one with: source ${VENV_ROOT}/<name>/bin/activate"
