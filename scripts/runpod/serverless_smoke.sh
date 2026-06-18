#!/usr/bin/env bash
# scripts/runpod/serverless_smoke.sh — smoke-test a RunPod Serverless (vLLM worker) endpoint
# by asking N biomedical questions and saving the answers.
#
# Outputs land flat under $OUTPUT_BASE/runpod_serverless_smoke/ (overwritten each run):
#   answers.jsonl  — one line per question (question, status, latency, answer, raw)
#   run.log        — tee'd stdout/stderr from this script
#
# Loads gitignored .env.runpod for RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID
# (same file scripts/runpod/launch.sh uses). An exported env var or a CLI flag wins.
#
# Flags (all optional):
#   -n, --num-questions <N>    limit to first N (default: all)
#   --endpoint <id>            RunPod serverless endpoint id
#   --api-key <key>            RunPod API key
#   --env-file <path>          secrets file (default: <repo>/.env.runpod)
#   --questions-file <path>    one question per line (# comments ok)
#   --system <text>            system prompt
#   --max-tokens <int>         sampling max_tokens (default 512)
#   --temperature <float>      default 0.6
#   --top-p <float>            default 0.95
#   --out <path>               JSONL output (default runpod_answers.jsonl)
#   --poll-timeout <sec>       max wait per question (default 300)
#   --poll-interval <sec>      default 2.0
#   -h, --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Two levels up: scripts/runpod/serverless_smoke.sh -> repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Defaults ---------------------------------------------------------------
DEFAULT_ENDPOINT="8d8le1qwbz760l"
DEFAULT_SYSTEM="You are a biomedical expert specializing in neuroscience. Answer accurately and concisely, explaining the underlying mechanism where relevant."

NUM_QUESTIONS=0           # 0 = no limit
ENDPOINT=""
API_KEY=""
ENV_FILE="$REPO_ROOT/.env.runpod"
QUESTIONS_FILE=""
SYSTEM_PROMPT="$DEFAULT_SYSTEM"
MAX_TOKENS=512
TEMPERATURE=0.6
TOP_P=0.95
OUT=""                    # default computed below: $RUN_DIR/answers.jsonl
POLL_TIMEOUT=300
POLL_INTERVAL=2.0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

# --- Arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-questions)  NUM_QUESTIONS="$2"; shift 2 ;;
        --endpoint)          ENDPOINT="$2";       shift 2 ;;
        --api-key)           API_KEY="$2";        shift 2 ;;
        --env-file)          ENV_FILE="$2";       shift 2 ;;
        --questions-file)    QUESTIONS_FILE="$2"; shift 2 ;;
        --system)            SYSTEM_PROMPT="$2";  shift 2 ;;
        --max-tokens)        MAX_TOKENS="$2";     shift 2 ;;
        --temperature)       TEMPERATURE="$2";    shift 2 ;;
        --top-p)             TOP_P="$2";          shift 2 ;;
        --out)               OUT="$2";            shift 2 ;;
        --poll-timeout)      POLL_TIMEOUT="$2";   shift 2 ;;
        --poll-interval)     POLL_INTERVAL="$2";  shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# --- Secrets ----------------------------------------------------------------
# Load .env.runpod (non-overriding: exported env or --flag still wins).
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
API_KEY="${API_KEY:-${RUNPOD_API_KEY:-}}"
ENDPOINT="${ENDPOINT:-${RUNPOD_ENDPOINT_ID:-$DEFAULT_ENDPOINT}}"

if [[ -z "$API_KEY" ]]; then
    echo "ERROR: no RUNPOD_API_KEY. Add it to $ENV_FILE, export it, or pass --api-key." >&2
    exit 2
fi
command -v curl    >/dev/null || { echo "curl not found";    exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

BASE="https://api.runpod.ai/v2/$ENDPOINT"

# --- Output dir + log capture ----------------------------------------------
# Smoke run is single-shot — each invocation overwrites the previous results.
# If you want history, copy/rename the dir manually.
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
SMOKE_DIR="$OUTPUT_BASE/runpod_serverless_smoke"
mkdir -p "$SMOKE_DIR"

LOG_FILE="$SMOKE_DIR/run.log"
[[ -z "$OUT" ]] && OUT="$SMOKE_DIR/answers.jsonl"

# Capture stdout + stderr to run.log while still printing to the terminal.
# `>` (not `>>`) so each invocation starts with a clean log.
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Question set -----------------------------------------------------------
BUILTIN_QUESTIONS=(
    "What is long-term potentiation, and why is it important for memory?"
    "Explain the role of dopamine in the basal ganglia."
    "How does an action potential propagate along a myelinated axon?"
    "What distinguishes ionotropic from metabotropic receptors?"
    "Describe the blood-brain barrier and its primary function."
    "What is the role of the hippocampus in memory consolidation?"
    "Explain the pathophysiology of Parkinson's disease at the circuit level."
    "How do astrocytes support neuronal function?"
    "What is synaptic pruning, and when does it occur during development?"
    "Describe the mechanism of action of SSRIs."
    "How do NMDA receptors contribute to synaptic plasticity?"
    "What is the difference between gray matter and white matter?"
)

QUESTIONS=()
if [[ -n "$QUESTIONS_FILE" ]]; then
    [[ -f "$QUESTIONS_FILE" ]] || { echo "questions file not found: $QUESTIONS_FILE" >&2; exit 1; }
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
        QUESTIONS+=("$line")
    done < "$QUESTIONS_FILE"
else
    QUESTIONS=("${BUILTIN_QUESTIONS[@]}")
fi

# Apply --num-questions cap.
if [[ "$NUM_QUESTIONS" -gt 0 && "$NUM_QUESTIONS" -lt "${#QUESTIONS[@]}" ]]; then
    QUESTIONS=("${QUESTIONS[@]:0:$NUM_QUESTIONS}")
fi
TOTAL="${#QUESTIONS[@]}"

echo "Endpoint : $BASE"
echo "Questions: $TOTAL  ->  ${OUT#$REPO_ROOT/}"
echo "Log      : ${LOG_FILE#$REPO_ROOT/}"
echo

# --- Helpers (python for JSON-safe assembly + extraction) -------------------

# Build the API request body. Args: <prompt> <max_tokens> <temp> <top_p>
build_payload() {
    SYS_PROMPT="$1" MAX="$2" TEMP="$3" TOPP="$4" \
        python3 -c "
import json, os
sampling = {
    'max_tokens': int(os.environ['MAX']),
    'temperature': float(os.environ['TEMP']),
    'top_p': float(os.environ['TOPP']),
}
print(json.dumps({'input': {'prompt': os.environ['SYS_PROMPT'], 'sampling_params': sampling}}))
"
}

# Wrap a question in the Qwen3 chat template. Args: <system> <user>
build_prompt() {
    SYS="$1" USR="$2" python3 -c "
import os
print(f\"<|im_start|>system\n{os.environ['SYS']}<|im_end|>\n<|im_start|>user\n{os.environ['USR']}<|im_end|>\n<|im_start|>assistant\n\")
"
}

# Pull a top-level field from a JSON blob piped on stdin.
json_field() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('$1',''))"; }

# Best-effort generated-text extraction (mirrors the .py walker).
extract_text() {
    python3 -c "
import json, sys
resp = json.load(sys.stdin)
out = resp.get('output', resp)
def walk(o):
    if isinstance(o, str): return o
    if isinstance(o, list): return ''.join(walk(x) for x in o)
    if isinstance(o, dict):
        for k in ('text','content'):
            if isinstance(o.get(k), str): return o[k]
        for k in ('choices','tokens','message','delta'):
            if k in o: return walk(o[k])
    return ''
print(walk(out).strip() or json.dumps(out)[:2000])
"
}

# Append one JSONL record to $OUT. Args: <question> <status> <latency> <answer> <raw-json>
write_record() {
    Q="$1" ST="$2" DT="$3" ANS="$4" RAW="$5" OUTFILE="$OUT" python3 -c "
import json, os
raw_str = os.environ['RAW']
try:
    raw = json.loads(raw_str) if raw_str else {}
except Exception:
    raw = {'_unparseable': raw_str[:500]}
rec = {
    'question': os.environ['Q'],
    'status':   os.environ['ST'],
    'latency_s': float(os.environ['DT']),
    'answer':   os.environ['ANS'],
    'raw':      raw,
}
with open(os.environ['OUTFILE'], 'a') as f:
    f.write(json.dumps(rec) + '\n')
"
}

# --- Submit + poll one prompt; print STATUS\tRAW_JSON to stdout -------------
run_one() {
    local prompt="$1"
    local payload resp status job_id deadline now
    payload=$(build_payload "$prompt" "$MAX_TOKENS" "$TEMPERATURE" "$TOP_P")

    resp=$(curl -sS --max-time "$POLL_TIMEOUT" -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$payload" "$BASE/runsync" || echo '{"status":"ERROR","error":"curl failed"}')
    status=$(printf '%s' "$resp" | json_field status)
    job_id=$(printf '%s' "$resp" | json_field id)

    deadline=$(( $(date +%s) + ${POLL_TIMEOUT%.*} ))
    while [[ "$status" == "IN_QUEUE" || "$status" == "IN_PROGRESS" ]]; do
        now=$(date +%s)
        [[ $now -ge $deadline ]] && break
        sleep "$POLL_INTERVAL"
        resp=$(curl -sS --max-time "$POLL_TIMEOUT" \
            -H "Authorization: Bearer $API_KEY" \
            "$BASE/status/$job_id" || echo '{"status":"ERROR","error":"curl failed"}')
        status=$(printf '%s' "$resp" | json_field status)
    done
    printf '%s\t%s\n' "$status" "$resp"
}

# --- Main loop --------------------------------------------------------------
: > "$OUT"
ok=0
for i in "${!QUESTIONS[@]}"; do
    q="${QUESTIONS[$i]}"
    n=$((i + 1))
    prompt=$(build_prompt "$SYSTEM_PROMPT" "$q")

    t0=$(date +%s)
    output=$(run_one "$prompt")
    t1=$(date +%s)
    dt=$((t1 - t0))

    status="${output%%	*}"
    raw="${output#*	}"

    answer=""
    if [[ "$status" == "COMPLETED" ]]; then
        answer=$(printf '%s' "$raw" | extract_text)
        ok=$((ok + 1))
    fi

    printf '[%d/%d] %s (%ds)  %s\n' "$n" "$TOTAL" "$status" "$dt" "$q"
    if [[ -n "$answer" ]]; then
        printf '    -> %.200s%s\n\n' "$answer" "$([[ ${#answer} -gt 200 ]] && echo '…')"
    fi
    write_record "$q" "$status" "$dt" "$answer" "$raw"
done

echo
echo "Done: $ok/$TOTAL completed."
echo "  answers : ${OUT#$REPO_ROOT/}"
echo "  log     : ${LOG_FILE#$REPO_ROOT/}"
[[ "$ok" -gt 0 ]] && exit 0 || exit 1
