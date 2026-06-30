#!/usr/bin/env bash
# scripts/preflight.sh — run-time pre-flight for the pipeline. Phase-aware, fail-fast.
#
# Validates that the environment is correctly configured for the phases about to run, BEFORE
# any compute: config files, per-phase secrets, venv imports, the CUDA context, torch-vs-driver
# version compatibility, VRAM, and (with --deep) live API reachability. Exits 1 on any FAIL.
#
# Complements scripts/runpod/bootstrap.sh's launch-time GPU pre-flight: that asks "is this pod
# alive?"; this asks "is THIS run correctly configured?".
#
# Usage (invoked by pipeline.sh; also runnable standalone):
#   scripts/preflight.sh --phases extract,curriculum --profile pilot --domain neuroscience \
#                        --platform runpod [--deep]
set -uo pipefail   # deliberately NOT -e: run ALL checks, don't stop at the first failure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/venv.sh
source "$SCRIPT_DIR/lib/venv.sh"
PROBE="$SCRIPT_DIR/lib/preflight_probe.py"

PHASES="all"; PROFILE=""; DOMAIN="neuroscience"; PLATFORM="local"; DEEP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phases)   PHASES="$2";   shift 2 ;;
        --profile)  PROFILE="$2";  shift 2 ;;
        --domain)   DOMAIN="$2";   shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --deep)     DEEP=1;        shift ;;
        *)          shift ;;
    esac
done

# Export so pipeline_config (get_model_id / get_phase_param) resolves the right domain+profile
# layers, regardless of when the caller exports them. Self-contained for standalone runs.
export SI_DOMAIN="$DOMAIN"
[[ -n "$PROFILE" ]] && export SI_PROFILE="$PROFILE"

ALL_PHASES=(extract validate graphmert curriculum sft rl)
if [[ "$PHASES" == "all" ]]; then
    SELECTED=("${ALL_PHASES[@]}")
else
    IFS=',' read -ra SELECTED <<< "$PHASES"
fi

FAILS=0; WARNS=0
ok()   { echo "    ✓ $1"; }
warn() { echo "    ⚠ $1"; WARNS=$((WARNS + 1)); }
fail() { echo "    ✗ $1"; FAILS=$((FAILS + 1)); }
runs() { local p; for p in "${SELECTED[@]}"; do [[ "$p" == "$1" ]] && return 0; done; return 1; }

echo "── pre-flight: phases=[${SELECTED[*]}] profile=${PROFILE:-default} domain=$DOMAIN platform=$PLATFORM deep=$DEEP"

# --- Config -----------------------------------------------------------------
echo "  [config]"
[[ -f "$REPO_ROOT/domains/$DOMAIN.yaml" ]] && ok "domains/$DOMAIN.yaml" || fail "domains/$DOMAIN.yaml missing"
if [[ -n "$PROFILE" ]]; then
    [[ -f "$REPO_ROOT/configs/profiles/$PROFILE.yaml" ]] && ok "profile $PROFILE.yaml" \
        || fail "configs/profiles/$PROFILE.yaml missing"
fi
if runs curriculum; then
    for pr in curriculum_qa curriculum_verify curriculum_pair_check; do
        [[ -f "$REPO_ROOT/prompts/$pr.yaml" ]] && ok "prompt $pr.yaml" || fail "prompts/$pr.yaml missing"
    done
fi

# --- Secrets (phase-scoped) -------------------------------------------------
echo "  [secrets]"
if runs curriculum; then
    [[ -n "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" ]] && ok "GEMINI/GOOGLE_API_KEY set" \
        || fail "GEMINI_API_KEY/GOOGLE_API_KEY unset (curriculum generation)"
    key_env=$(get_phase_param curriculum pair_check_api_key_env OPENAI_API_KEY)
    [[ -n "${!key_env:-}" ]] && ok "pair-check key \$$key_env set" \
        || fail "pair-check key \$$key_env unset (validate_qa_pair — set it, or point pair_check_base_url at a local vLLM)"
    ca=$(get_model_id curriculum_check_a ""); cb=$(get_model_id curriculum_check_b "")
    [[ -n "$ca" && -n "$cb" ]] && ok "curriculum_check_a/b set ($ca + $cb)" \
        || fail "curriculum_check_a/b model ids unset"
fi
if runs validate || runs graphmert; then
    va=$(get_model_id validate_a ""); vb=$(get_model_id validate_b "")
    [[ -n "$va" && -n "$vb" ]] && ok "validate_a/b set ($va + $vb)" || fail "validate_a/b model ids unset"
fi
if runs validate || runs graphmert || runs curriculum || runs sft || runs rl; then
    [[ -n "${HF_TOKEN:-}" ]] && ok "HF_TOKEN set" || warn "HF_TOKEN unset (gated HF model downloads may 401)"
fi
if runs sft || runs rl; then
    [[ -n "${WANDB_API_KEY:-}" ]] && ok "WANDB_API_KEY set" || warn "WANDB_API_KEY unset (W&B logging disabled)"
fi

# --- Data (extract) ---------------------------------------------------------
if runs extract; then
    echo "  [data]"
    corpus="${CORPUS_PATH:-corpus/$DOMAIN/source_txt}"
    if [[ -e "$REPO_ROOT/$corpus" ]]; then
        ok "corpus present: $corpus"
    elif [[ -n "${S3_URI:-}" ]]; then
        ok "corpus via S3: $S3_URI/$corpus (extract pulls if local is empty)"
    else
        fail "no corpus: $REPO_ROOT/$corpus missing and S3_URI unset"
    fi
fi

# --- Venv + GPU probes (one per unique venv among the running phases) --------
echo "  [venv + gpu]"
vram_min=$(get_phase_param runpod vram_gb_min "")
oa_base=$(get_phase_param curriculum pair_check_base_url "")
oa_keyenv=$(get_phase_param curriculum pair_check_api_key_env OPENAI_API_KEY)
oa_model=$(get_model_id curriculum_pair_check "")

probe_venv() {  # $1=venv  $2=imports(csv)  $3=ping(csv)  $4=flash(1/0)
    local venv="$1" imports="$2" ping="$3" flash="$4"
    echo "    venv: $venv"
    local args=(--imports "$imports" --cuda --ping "$ping"
                --openai-base-url "$oa_base" --openai-key-env "$oa_keyenv" --openai-model "$oa_model")
    [[ -n "$vram_min" ]] && args+=(--vram-min "$vram_min")
    [[ "$flash" -eq 1 ]] && args+=(--flash-attn)
    [[ "$DEEP" -eq 1 ]] && args+=(--deep)
    (
        source_venv "$venv" >/dev/null 2>&1 || { echo "      ✗ venv '$venv' missing or activate failed"; exit 1; }
        python "$PROBE" "${args[@]}"
    )
    [[ $? -ne 0 ]] && FAILS=$((FAILS + 1))
}

want_graphrag=0; want_graphmert=0; want_sicur=0; want_flash=0
runs extract && want_graphrag=1
{ runs validate || runs graphmert; } && want_graphmert=1
{ runs curriculum || runs sft || runs rl; } && want_sicur=1
{ runs sft || runs rl; } && want_flash=1

[[ $want_graphrag -eq 1 ]]  && probe_venv graphrag  "torch,vllm"                          "hf" 0
[[ $want_graphmert -eq 1 ]] && probe_venv graphmert "torch,vllm,transformers"             "hf" 0
if [[ $want_sicur -eq 1 ]]; then
    pings="hf"
    runs curriculum && pings="gemini,openai,hf"
    probe_venv si_curriculum "torch,vllm,transformers,trl,peft,openai" "$pings" "$want_flash"
fi

# --- Summary ----------------------------------------------------------------
echo ""
if [[ $FAILS -gt 0 ]]; then
    log_error "pre-flight FAILED: $FAILS error(s), $WARNS warning(s). Fix the ✗ items above (or --skip-preflight to bypass)."
    exit 1
fi
[[ $WARNS -gt 0 ]] && log_warn "pre-flight passed with $WARNS warning(s)."
log_info "pre-flight OK — ${#SELECTED[@]} phase(s): ${SELECTED[*]}"
exit 0
