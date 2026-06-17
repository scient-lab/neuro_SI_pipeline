#!/usr/bin/env bash
# pipeline.sh - orchestrator for the specialized-SLM pipeline.
#
# Parses CLI flags, sets the env contract consumed by pipeline_config.py,
# and dispatches the requested phases via the selected platform wrapper.
#
# The config loader (pipeline_config.py) reads layers directly from
# this repo's configs/, domains/, and prompts/ at runtime; no pre-merge
# step is required.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Auto-source $REPO_ROOT/.env so secrets (GEMINI_API_KEY, HF_TOKEN,
# GRAPHRAG_API_KEY, …) reach phase subprocesses. `set -a` is critical —
# without it, plain `source .env` sets variables only in this shell and
# child Python processes hit KeyError on os.environ['…']. Idempotent: safe
# to source even if the user already sourced it manually.
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# --- Defaults ---------------------------------------------------------------
DOMAIN="neuroscience"
PROFILE=""
PLATFORM="local"
PHASES="all"
STEPS="all"
LIST_ONLY=0
FINAL=0

ALL_PHASES=(extract validate graphmert curriculum sft rl)

# --- Arg parsing ------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: pipeline.sh [--domain <name>] [--profile <name>] [--platform <name>]
                   [--phase <list>] [--step <list>] [--list]

Options:
  --domain    Domain name (default: neuroscience). Must exist as domains/<name>.yaml.
  --profile   Scaling profile: smoke | pilot | paper. Default: no profile (use defaults).
  --platform  local (default) | runpod | aws | princeton.
  --phase     Comma-separated phase names, or "all" (default).
              Phases: extract, validate, graphmert, curriculum, sft, rl
  --step      Comma-separated step names within the phase, or "all" (default).
              Only meaningful when a single phase is specified.
  --list      Print available phases and steps, then exit.
              Combine with --phase <name> to show steps for one phase only.
  --final     Force the run's terminal status to "completed" (+ _SUCCESS) at
              the end of this invocation, even if the last canonical phase
              (rl) wasn't selected. Use to close out a partial pipeline.

Run identity:
  RUN_ID is generated per invocation (<utc>-<profile>-<sha>). To run phases in
  SEPARATE invocations but have them belong to ONE logical run (one manifest,
  one logs/<run_id>/ dir, one S3 prefix), export RUN_ID once and reuse it:
    export RUN_ID=$(date -u +%Y%m%d-%H%M%S)-pilot-$(git rev-parse --short HEAD)
    ./scripts/pipeline.sh --phase extract
    ./scripts/pipeline.sh --phase graphmert
    ./scripts/pipeline.sh --phase curriculum,sft,rl   # rl last -> writes _SUCCESS
  manifest.py init merges new phases into the existing same-RUN_ID manifest
  instead of overwriting it.

Examples:
  pipeline.sh                                              # neuroscience + defaults
  pipeline.sh --profile smoke
  pipeline.sh --phase extract --step parse_pdf,chunk
  pipeline.sh --domain neuroscience --profile paper --platform runpod
  pipeline.sh --list                                       # all phases + steps
  pipeline.sh --list --phase graphmert                     # just graphmert's steps
EOF
}

# Extract phase metadata via grep/awk — safer than sourcing the script
# (which would activate venvs and run side effects).
_extract_steps_from_phase_file() {
    grep -E '^STEPS=\(' "$1" | head -1 | sed -E 's/^STEPS=\(//; s/\)[[:space:]]*$//'
}

_extract_phase_desc() {
    grep -E '^PHASE_DESC=' "$1" | head -1 | sed -E 's/^PHASE_DESC="//; s/"[[:space:]]*$//'
}

# Pull the array contents between `STEP_DESCS=(` and the matching closing `)`,
# yielding one description per line (still quoted) — caller strips quotes.
_extract_step_descs() {
    awk '
        /^STEP_DESCS=\(/ { inside=1; next }
        inside && /^\)/  { inside=0; next }
        inside           { print }
    ' "$1" | sed -E 's/^[[:space:]]*"//; s/"[[:space:]]*$//' | grep -v '^[[:space:]]*$'
}

list_phases_and_steps() {
    local target_phase="${1:-}"
    local first_phase=1
    for phase in "${ALL_PHASES[@]}"; do
        [[ -n "$target_phase" && "$phase" != "$target_phase" ]] && continue
        local pf="$SCRIPT_DIR/phases/${phase}.sh"
        [[ -f "$pf" ]] || continue

        local desc steps
        desc=$(_extract_phase_desc "$pf")
        steps=$(_extract_steps_from_phase_file "$pf")
        IFS=' ' read -ra step_arr <<< "$steps"
        mapfile -t desc_arr < <(_extract_step_descs "$pf")

        # Phase header (blank line between phases)
        [[ $first_phase -eq 0 ]] && echo
        first_phase=0
        printf "%s — %s\n" "$phase" "${desc:-<no description>}"

        # Indented step list (left col width: 22 chars)
        local i=0
        for s in "${step_arr[@]}"; do
            printf "    %-22s %s\n" "$s" "${desc_arr[$i]:-}"
            i=$((i + 1))
        done
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)   DOMAIN="$2"; shift 2 ;;
        --profile)  PROFILE="$2"; shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --phase)    PHASES="$2"; shift 2 ;;
        --step)     STEPS="$2"; shift 2 ;;
        --list)     LIST_ONLY=1; shift ;;
        --final)    FINAL=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          log_error "Unknown flag: $1"; usage; exit 2 ;;
    esac
done

# --- Listing mode (no execution) -------------------------------------------
if [[ "$LIST_ONLY" -eq 1 ]]; then
    if [[ "$PHASES" != "all" ]]; then
        list_phases_and_steps "$PHASES"
    else
        list_phases_and_steps
    fi
    exit 0
fi

# --- Validation -------------------------------------------------------------
domain_file="$REPO_ROOT/domains/${DOMAIN}.yaml"
platform_script="$SCRIPT_DIR/platforms/${PLATFORM}.sh"

[[ -f "$domain_file" ]] || { log_error "Missing domain file: $domain_file"; exit 2; }
[[ -f "$platform_script" ]] || { log_error "Missing platform script: $platform_script"; exit 2; }

if [[ -n "$PROFILE" ]]; then
    profile_file="$REPO_ROOT/configs/profiles/${PROFILE}.yaml"
    [[ -f "$profile_file" ]] || { log_error "Missing profile file: $profile_file"; exit 2; }
fi

# Resolve phase list
if [[ "$PHASES" == "all" ]]; then
    selected_phases=("${ALL_PHASES[@]}")
else
    IFS=',' read -ra selected_phases <<< "$PHASES"
    for p in "${selected_phases[@]}"; do
        [[ -f "$SCRIPT_DIR/phases/${p}.sh" ]] || {
            log_error "Unknown phase: $p"; exit 2;
        }
    done
fi

# --- Run identity + manifest -----------------------------------------------
# OUTPUT_BASE is the canonical anchor for all per-run artifacts.
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
mkdir -p "$OUTPUT_BASE"

# RUN_ID format: <UTC timestamp>-<profile>-<git-short-sha>. Sorts chronologically;
# embedded profile/sha makes it grep-friendly across many runs.
# If RUN_ID is already exported, REUSE it — that's how phase-wise invocations
# join one logical run (shared manifest / logs dir / S3 prefix); manifest.py
# init then merges rather than overwrites.
_git_sha=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
_git_branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo nogit)
_ts=$(date -u +%Y%m%d-%H%M%S)
RUN_ID="${RUN_ID:-${_ts}-${PROFILE:-default}-${_git_sha}}"
LOG_DIR="$OUTPUT_BASE/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# Reproducibility + live-status manifest. Built and mutated by the stdlib-only
# scripts/lib/manifest.py (atomic, lock-guarded) so it's safe to read mid-run
# (e.g. an API polling the S3 copy). See that file for the schema:
#   meta — static catalog (status enum, canonical phases + their steps)
#   run  — per-run status, tz-aware start/end timestamps per phase AND step,
#          exit codes, per-step log_file paths.
MANIFEST="$OUTPUT_BASE/run_manifest.json"
manifest_selected=$(IFS=,; echo "${selected_phases[*]}")
manifest_all=$(IFS=,; echo "${ALL_PHASES[*]}")
python3 "$SCRIPT_DIR/lib/manifest.py" init \
    --path "$MANIFEST" \
    --phases-dir "$SCRIPT_DIR/phases" \
    --phase-order "$manifest_all" \
    --selected "$manifest_selected" \
    --run-id "$RUN_ID" \
    --domain "$DOMAIN" \
    --profile "${PROFILE:-default}" \
    --platform "$PLATFORM" \
    --git-sha "$_git_sha" \
    --git-branch "$_git_branch" \
    --step-filter "$STEPS" \
    || log_warn "manifest init failed — run will proceed uninstrumented"

# Mark the run failed on any unexpected exit (set -e abort, signal, etc.)
# unless we already finalized successfully. Belt-and-suspenders around the
# explicit finalize at the end.
PIPELINE_FINALIZED=0
_on_exit() {
    local ec=$?
    if [[ "$PIPELINE_FINALIZED" -eq 0 ]]; then
        python3 "$SCRIPT_DIR/lib/manifest.py" finalize --path "$MANIFEST" --status failed 2>/dev/null || true
    fi
    exit "$ec"
}
trap _on_exit EXIT

# --- Dispatch ---------------------------------------------------------------
log_info "Run ID   : $RUN_ID"
log_info "Domain   : $DOMAIN"
log_info "Profile  : ${PROFILE:-<none>}"
log_info "Platform : $PLATFORM"
log_info "Phases   : ${selected_phases[*]}"
log_info "Steps    : $STEPS"
log_info "Logs     : $LOG_DIR"
log_info "Manifest : $MANIFEST"

# Env contract consumed by pipeline_config.py and phase scripts.
export SI_DOMAIN="$DOMAIN"
export SI_PROFILE="$PROFILE"
export SI_PLATFORM="$PLATFORM"
export PIPELINE_STEP_FILTER="$STEPS"
export REPO_ROOT OUTPUT_BASE RUN_ID
# Consumed by lib/common.sh::run_step for per-step manifest updates + logging.
export PIPELINE_MANIFEST="$MANIFEST"
export PIPELINE_LOG_DIR="$LOG_DIR"

# shellcheck source=platforms/local.sh
source "$platform_script"

_s3_sync_if_configured() {
    # Push outputs to s3://$S3_URI/runs/$RUN_ID/outputs/. No-op when S3_URI
    # isn't set (local workstation case). Non-fatal — sync failure prints a
    # warning but doesn't kill the pipeline.
    if [[ -n "${S3_URI:-}" && -x "$SCRIPT_DIR/data_prep/sync_outputs.sh" ]]; then
        "$SCRIPT_DIR/data_prep/sync_outputs.sh" \
            || log_warn "S3 output sync failed (non-fatal) — outputs still on local disk"
    fi
}

for phase in "${selected_phases[@]}"; do
    phase_script="$SCRIPT_DIR/phases/${phase}.sh"
    log_file="$LOG_DIR/${phase}.log"
    rel_log="${log_file#"$REPO_ROOT"/}"
    log_info "── Phase: $phase  (log: ${rel_log}) ─────────────"

    python3 "$SCRIPT_DIR/lib/manifest.py" start-phase \
        --path "$MANIFEST" --phase "$phase" --log-file "$rel_log" 2>/dev/null || true

    # Capture the phase exit code WITHOUT aborting (so we can record "failed"
    # and finalize). PIPESTATUS[0] is the phase's code, not tee's — pipefail
    # alone wouldn't let us run end-phase before set -e kills the script.
    set +e
    exec_phase_on_platform "$phase_script" "$STEPS" 2>&1 | tee "$log_file"
    phase_rc=${PIPESTATUS[0]}
    set -e

    python3 "$SCRIPT_DIR/lib/manifest.py" end-phase \
        --path "$MANIFEST" --phase "$phase" --exit-code "$phase_rc" \
        --log-file "$rel_log" 2>/dev/null || true

    _s3_sync_if_configured

    if [[ "$phase_rc" -ne 0 ]]; then
        log_error "Phase '$phase' failed (exit $phase_rc) — aborting pipeline."
        python3 "$SCRIPT_DIR/lib/manifest.py" finalize \
            --path "$MANIFEST" --status failed 2>/dev/null || true
        PIPELINE_FINALIZED=1
        _s3_sync_if_configured
        exit "$phase_rc"
    fi
done

# Decide whether THIS invocation closes the logical run. It does when the last
# canonical phase (rl) was part of it, or when --final was passed. Otherwise
# (phase-wise execution mid-pipeline) leave run.status = "running" so a later
# invocation — sharing this RUN_ID — can append more phases and close it then.
last_canonical="${ALL_PHASES[-1]}"
finalize_completed="$FINAL"
for p in "${selected_phases[@]}"; do
    [[ "$p" == "$last_canonical" ]] && finalize_completed=1
done

if [[ "$finalize_completed" -eq 1 ]]; then
    # Mark complete (writes _SUCCESS) before the final sync so terminal status
    # reaches S3 too.
    python3 "$SCRIPT_DIR/lib/manifest.py" finalize \
        --path "$MANIFEST" --status completed 2>/dev/null || true
    log_info "Pipeline complete. Run ID: $RUN_ID"
else
    log_info "Phase(s) done; run still 'running' (last phase '$last_canonical' not yet run)."
    log_info "  Reuse RUN_ID=$RUN_ID for the next phase, or pass --final to close the run."
fi
# Suppress the EXIT trap's failure-finalize — we exited cleanly either way.
PIPELINE_FINALIZED=1

# Belt-and-suspenders final sync (catches anything the per-phase missed).
_s3_sync_if_configured
