#!/usr/bin/env bash
# scripts/runpod/remote.sh — single entry point for running things on a RunPod
# pod remotely from the workstation. One SSH connection per invocation;
# output streams back to your terminal in real time.
#
# Saves the "SSH in, cd $SI_HOME, type the right command" dance for every
# routine pod operation. Subcommands map to existing pod-side scripts.
#
# SSH target resolution (in order):
#   1. --target / -t flag                            (highest priority)
#   2. $RUNPOD_SSH_TARGET (env or .env.runpod)
#   3. Error out with usage
#
# Subcommands:
#   bootstrap            — pull + run scripts/runpod/bootstrap.sh on the pod.
#                          Forwards GITHUB_TOKEN/REPO/BRANCH + the other env
#                          vars bootstrap.sh needs. Use after pod create or
#                          after `git pull` brings bootstrap changes.
#   pipeline [args]      — launch nohup ./scripts/pipeline.sh <args> on the pod.
#                          Background-detached so SSH disconnects don't kill it.
#                          Returns the PID + RUN_ID so you can monitor.
#   logs [args]          — run ./scripts/logs.sh <args> on the pod (synchronous).
#   diagnose [args]      — run ./scripts/diagnose.sh <args> on the pod.
#   diagnose-llm [args]  — run ./scripts/diagnose_llm_extraction.sh.
#   kill                 — run ./scripts/kill_pipeline.sh on the pod.
#   sync                 — run ./scripts/s3_sync.sh on the pod.
#   exec '<cmd>'         — run an arbitrary bash command in $SI_HOME on the pod.
#   ssh                  — open an interactive SSH session (cd $SI_HOME first).
#
# Options:
#   --target / -t <ssh-target>     e.g. root@abc.proxy.runpod.net
#   --port / -p <port>             default 22
#   -- <ssh args>                  raw ssh flags after `--`
#                                  (e.g. `-- -i ~/.ssh/runpod_key -o StrictHostKeyChecking=no`)
#
# Examples:
#   ./scripts/runpod/remote.sh -t root@abc.proxy.runpod.net bootstrap
#   ./scripts/runpod/remote.sh pipeline --profile pilot --platform runpod
#   ./scripts/runpod/remote.sh logs --summary
#   ./scripts/runpod/remote.sh diagnose --phase extract --deep
#   ./scripts/runpod/remote.sh kill
#   ./scripts/runpod/remote.sh exec 'nvidia-smi'
#   ./scripts/runpod/remote.sh ssh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- env --------------------------------------------------------------------
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env.runpod}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# --- arg parsing ------------------------------------------------------------
SSH_TARGET="${RUNPOD_SSH_TARGET:-}"
SSH_PORT="${RUNPOD_SSH_PORT:-22}"
SSH_EXTRA=()
SUBCMD=""
SUBARGS=()

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

# Parse leading flags + subcommand. Everything after the subcommand goes
# into SUBARGS verbatim; `--` switches us into SSH_EXTRA mode.
while [[ $# -gt 0 ]]; do
    if [[ -z "$SUBCMD" ]]; then
        case "$1" in
            -h|--help)         usage; exit 0 ;;
            -t|--target)       SSH_TARGET="$2"; shift 2 ;;
            -p|--port)         SSH_PORT="$2";   shift 2 ;;
            --)                shift; SSH_EXTRA+=("$@"); break ;;
            -*)                echo "unknown flag: $1" >&2; usage >&2; exit 1 ;;
            *)                 SUBCMD="$1"; shift ;;
        esac
    else
        # Subcommand seen — everything else is its args until --
        if [[ "$1" == "--" ]]; then shift; SSH_EXTRA+=("$@"); break; fi
        SUBARGS+=("$1"); shift
    fi
done

[[ -z "$SUBCMD" ]] && { echo "ERROR: subcommand required" >&2; usage >&2; exit 1; }
[[ -z "$SSH_TARGET" ]] && {
    echo "ERROR: SSH target required (pass --target or set RUNPOD_SSH_TARGET in $ENV_FILE)" >&2
    exit 2
}

# Reject RunPod's SSH proxy for command-mode subcommands. The proxy supports
# interactive sessions only — `ssh proxy 'some command'` silently drops the
# command and lands you in an interactive shell (the bug the user just hit).
# For command-mode operation, RunPod requires the direct TCP endpoint
# (root@<pod-public-ip> -p <high-port>), shown in the pod's Connect page
# under "SSH over exposed TCP".
# The `ssh` subcommand IS interactive, so allow proxy there.
case "$SSH_TARGET" in
    *@ssh.runpod.io)
        if [[ "$SUBCMD" != "ssh" ]]; then
            echo "ERROR: target '$SSH_TARGET' is RunPod's SSH PROXY." >&2
            echo "       The proxy is interactive-only — it cannot execute" >&2
            echo "       command-mode operations like '$SUBCMD'." >&2
            echo >&2
            echo "       Use the DIRECT TCP endpoint instead. In RunPod console:" >&2
            echo "         Pod → Connect → 'SSH over exposed TCP'" >&2
            echo "       It'll look like: root@<public-ip> -p <high-port>" >&2
            echo >&2
            echo "       Example:" >&2
            echo "         ./scripts/runpod/remote.sh --target root@1.2.3.4 -p 12345 $SUBCMD" >&2
            exit 3
        fi
        ;;
esac

# --- helpers ----------------------------------------------------------------
SI_HOME_REMOTE="${SI_HOME:-/workspace/neuro_SI_pipeline}"

# Quote args safely for embedding in the remote shell command.
# Each arg gets %q-escaped so spaces/quotes/special chars survive.
quote_args() {
    local out=""
    for a in "$@"; do
        out+=$(printf ' %q' "$a")
    done
    echo "${out# }"
}

# Run a command on the pod via SSH.
#
# Why -tt (force PTY allocation) for the "non-interactive" path:
#   RunPod's SSH proxy (ssh.runpod.io) REQUIRES a PTY even for command-mode
#   exec. Without -t/-tt it fails immediately with:
#     "Error: Your SSH client doesn't support PTY"
#   Direct-IP SSH to the pod accepts -tt without complaint (it'll print a
#   harmless "no terminal" line but still runs), so we use -tt unconditionally
#   to work across BOTH SSH topologies.
#
# Trade-off: PTY mode may inject control codes (colors, line discipline) into
# output. Acceptable for our use — the scripts we call (logs.sh, diagnose.sh,
# etc.) are designed for human-readable output anyway.
remote_run() {
    local remote_cmd="$1"
    ssh -tt -p "$SSH_PORT" "${SSH_EXTRA[@]}" "$SSH_TARGET" "bash -lc $(printf '%q' "$remote_cmd")"
}

# Run an interactive command on the pod (allocates TTY) — same as remote_run.
remote_run_tty() {
    remote_run "$1"
}

# --- subcommand dispatch ----------------------------------------------------
case "$SUBCMD" in

    bootstrap)
        # Re-run bootstrap.sh on the pod. Reuses GITHUB_TOKEN/REPO/BRANCH
        # already in $SI_HOME/.env on the pod (written by launch.sh's first
        # bootstrap). No env forwarding from workstation → no secrets in
        # the pod's `ps aux` lines.
        #
        # If $SI_HOME/.env doesn't exist (= the pod was never bootstrapped
        # in the first place), we ERROR rather than silently re-forwarding.
        # First-time setup must go through `./scripts/runpod/launch.sh`.
        #
        # GENERIC env-var pass-through: any SUBARG of the form `KEY=VALUE`
        # (uppercase + underscores in KEY) becomes an inline env var on the
        # remote bootstrap.sh invocation. Knowledge of WHICH env vars
        # bootstrap.sh honors (STAGES, etc.) stays in bootstrap.sh — this
        # forwarder doesn't care, it just passes things through.
        #
        # Example: remote.sh bootstrap STAGES=graphrag         # extract-only deps
        #          remote.sh bootstrap STAGES=graphrag,graphmert
        #          remote.sh bootstrap GITHUB_BRANCH=experimental    # alt branch
        # Build `export KEY=VALUE; ...` lines. `export` (vs bare assignment)
        # ensures the var is visible to bootstrap.sh, which runs in a child
        # process via `bash <(curl ...)`.
        ENV_PREFIX=""
        for a in "${SUBARGS[@]}"; do
            case "$a" in
                [A-Z_]*=*) ENV_PREFIX+="export $(printf '%q' "$a"); " ;;
                *) ;;
            esac
        done
        remote_run "
env_file=$SI_HOME_REMOTE/.env
if [[ ! -f \"\$env_file\" ]]; then
    echo 'ERROR: \$env_file not found — pod was never bootstrapped.' >&2
    echo '       Run ./scripts/runpod/launch.sh from workstation for first-time setup.' >&2
    exit 1
fi
${ENV_PREFIX}
set -a; . \"\$env_file\"; set +a
: \"\${GITHUB_TOKEN:?GITHUB_TOKEN missing from \$env_file}\"
: \"\${GITHUB_REPO:?GITHUB_REPO missing from \$env_file}\"
: \"\${GITHUB_BRANCH:?GITHUB_BRANCH missing from \$env_file}\"
bash <(curl -sH \"Authorization: token \$GITHUB_TOKEN\" \\
            -H \"Accept: application/vnd.github.v3.raw\" \\
            \"https://api.github.com/repos/\$GITHUB_REPO/contents/scripts/runpod/bootstrap.sh?ref=\$GITHUB_BRANCH\")
"
        ;;

    pipeline)
        # Detached start so SSH disconnect doesn't kill it. Returns the PID +
        # RUN_ID. Subsequent `remote.sh logs --summary` can monitor.
        args=$(quote_args "${SUBARGS[@]}")
        remote_run "
cd $SI_HOME_REMOTE
nohup ./scripts/pipeline.sh $args > nohup.out 2>&1 &
PID=\$!
sleep 1
echo \"pipeline.sh started — pid=\$PID  log=$SI_HOME_REMOTE/nohup.out\"
./scripts/logs.sh --summary 2>/dev/null | head -10 || true
"
        ;;

    logs)
        args=$(quote_args "${SUBARGS[@]}")
        remote_run "cd $SI_HOME_REMOTE && ./scripts/logs.sh $args"
        ;;

    diagnose)
        args=$(quote_args "${SUBARGS[@]}")
        remote_run "cd $SI_HOME_REMOTE && ./scripts/diagnose.sh $args"
        ;;

    diagnose-llm)
        args=$(quote_args "${SUBARGS[@]}")
        remote_run "cd $SI_HOME_REMOTE && ./scripts/diagnose_llm_extraction.sh $args"
        ;;

    kill)
        # Kill is short-lived; show the output.
        remote_run "cd $SI_HOME_REMOTE && ./scripts/kill_pipeline.sh"
        ;;

    sync)
        args=$(quote_args "${SUBARGS[@]}")
        remote_run "cd $SI_HOME_REMOTE && ./scripts/s3_sync.sh $args"
        ;;

    exec)
        # Run arbitrary command in $SI_HOME. User-supplied; trust the operator.
        if [[ ${#SUBARGS[@]} -eq 0 ]]; then
            echo "ERROR: exec needs a command, e.g. exec 'nvidia-smi'" >&2
            exit 1
        fi
        # Join args back into one command line — typical use is `exec '<cmd>'`
        # as a single quoted string, but also handle `exec ls -la`.
        joined=$(printf '%s ' "${SUBARGS[@]}")
        remote_run "cd $SI_HOME_REMOTE && ${joined% }"
        ;;

    ssh)
        # Open interactive SSH session in $SI_HOME.
        remote_run_tty "cd $SI_HOME_REMOTE && exec bash"
        ;;

    *)
        echo "ERROR: unknown subcommand: $SUBCMD" >&2
        usage >&2
        exit 1
        ;;
esac
