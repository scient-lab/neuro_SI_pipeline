#!/usr/bin/env bash
# scripts/runpod/vllm_smoke.sh — smoke-test a RunPod Pod running vLLM with its
# OpenAI-compatible API (chat completions). Counterpart to serverless_smoke.sh.
#
# Differences from serverless_smoke.sh:
#   - POSTs to <endpoint>/v1/chat/completions (OpenAI chat format)
#   - Synchronous response — no /status polling
#   - Server handles chat templating — no Qwen <|im_start|> wrapping client-side
#
# Outputs land flat under $OUTPUT_BASE/runpod_vllm_smoke/ (overwritten each run):
#   answers.jsonl  — one line per question (question, status, latency, answer, raw)
#   run.log        — tee'd stdout/stderr from this script
#
# Loads gitignored .env.runpod for VLLM_ENDPOINT_URL + VLLM_API_KEY
# (same file launch.sh + serverless_smoke.sh use). An exported env var or a
# CLI flag wins.
#
# Flags (all optional):
#   -n, --num-questions <N>    limit to first N (default: all)
#   --endpoint <url>           full base URL (e.g. https://abc-8000.proxy.runpod.net)
#   --api-key <key>            bearer token (sk-...)
#   --env-file <path>          secrets file (default: <repo>/.env.runpod)
#   --questions-file <path>    one question per line (# comments ok)
#   --system <text>            system prompt (default: built-in)
#   --model <name>             OPTIONAL model name (server picks default if absent)
#   --max-tokens <int>         (default 512)
#   --temperature <float>      (default 0.6)
#   --top-p <float>            (default 0.95)
#   --out <path>               JSONL output (default $OUTPUT_BASE/runpod_vllm_smoke/answers.jsonl)
#   --timeout <sec>            per-request timeout (default 120)
#   -h, --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Two levels up: scripts/runpod/vllm_smoke.sh -> repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Defaults ---------------------------------------------------------------
DEFAULT_SYSTEM="You are a biomedical expert specializing in neuroscience. Answer accurately and concisely, explaining the underlying mechanism where relevant."

NUM_QUESTIONS=0           # 0 = no limit
ENDPOINT=""
API_KEY=""
ENV_FILE="$REPO_ROOT/.env.runpod"
QUESTIONS_FILE=""
SYSTEM_PROMPT="$DEFAULT_SYSTEM"
MODEL=""                  # OpenAI calls require this; vLLM often accepts empty
MAX_TOKENS=512
TEMPERATURE=0.6
TOP_P=0.95
OUT=""                    # default computed below: $RUN_DIR/answers.jsonl
TIMEOUT=120

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
        --model)             MODEL="$2";          shift 2 ;;
        --max-tokens)        MAX_TOKENS="$2";     shift 2 ;;
        --temperature)       TEMPERATURE="$2";    shift 2 ;;
        --top-p)             TOP_P="$2";          shift 2 ;;
        --out)               OUT="$2";            shift 2 ;;
        --timeout)           TIMEOUT="$2";        shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# --- Secrets ----------------------------------------------------------------
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
API_KEY="${API_KEY:-${VLLM_API_KEY:-}}"
ENDPOINT="${ENDPOINT:-${VLLM_ENDPOINT_URL:-}}"

if [[ -z "$API_KEY" ]]; then
    echo "ERROR: no VLLM_API_KEY. Add it to $ENV_FILE, export it, or pass --api-key." >&2
    exit 2
fi
if [[ -z "$ENDPOINT" ]]; then
    echo "ERROR: no VLLM_ENDPOINT_URL. Add it to $ENV_FILE, export it, or pass --endpoint." >&2
    echo "       Example: https://abcd1234-8000.proxy.runpod.net" >&2
    exit 2
fi
command -v curl    >/dev/null || { echo "curl not found";    exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

# Strip trailing slash for clean URL composition.
ENDPOINT="${ENDPOINT%/}"
CHAT_URL="$ENDPOINT/v1/chat/completions"

# --- Output dir + log capture ----------------------------------------------
# Same convention as serverless_smoke.sh — outputs land under
# $OUTPUT_BASE/runpod_vllm_smoke/ (overwritten each run).
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
SMOKE_DIR="$OUTPUT_BASE/runpod_vllm_smoke"
mkdir -p "$SMOKE_DIR"

LOG_FILE="$SMOKE_DIR/run.log"
[[ -z "$OUT" ]] && OUT="$SMOKE_DIR/answers.jsonl"

# Fresh log each run, then tee everything.
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

echo "Endpoint : $CHAT_URL"
[[ -n "$MODEL" ]] && echo "Model    : $MODEL"
echo "Questions: $TOTAL  ->  ${OUT#$REPO_ROOT/}"
echo "Log      : ${LOG_FILE#$REPO_ROOT/}"
echo

# --- Helpers (python for JSON-safe assembly + extraction) -------------------

# Build the OpenAI chat-completions body. Args: <user_q>
build_payload() {
    SYS="$SYSTEM_PROMPT" USR="$1" MODEL="$MODEL" \
    MAX="$MAX_TOKENS" TEMP="$TEMPERATURE" TOPP="$TOP_P" \
        python3 -c "
import json, os
body = {
    'messages': [
        {'role': 'system', 'content': os.environ['SYS']},
        {'role': 'user',   'content': os.environ['USR']},
    ],
    'max_tokens':  int(os.environ['MAX']),
    'temperature': float(os.environ['TEMP']),
    'top_p':       float(os.environ['TOPP']),
}
if os.environ.get('MODEL'):
    body['model'] = os.environ['MODEL']
print(json.dumps(body))
"
}

# Extract assistant content from an OpenAI chat-completions response.
extract_text() {
    python3 -c "
import json, sys
try:
    resp = json.load(sys.stdin)
except Exception:
    print(''); sys.exit(0)
choices = resp.get('choices') or []
if choices:
    msg = choices[0].get('message') or {}
    text = msg.get('content') or ''
    print(text.strip())
else:
    print(json.dumps(resp)[:2000])
"
}

# Pull HTTP status string from response: 'COMPLETED', 'HTTP_400', 'ERROR', …
# Args: <raw_json> <curl_exit_code>
classify_status() {
    RAW="$1" RC="$2" python3 -c "
import json, os
rc = int(os.environ['RC'])
if rc != 0:
    print('ERROR'); raise SystemExit(0)
try:
    r = json.loads(os.environ['RAW'])
except Exception:
    print('ERROR'); raise SystemExit(0)
# vLLM/OpenAI: success has 'choices'; errors have 'error' or 'detail'
if isinstance(r, dict) and 'choices' in r and r['choices']:
    print('COMPLETED')
elif isinstance(r, dict) and ('error' in r or 'detail' in r):
    code = (r.get('error') or {}).get('code') if isinstance(r.get('error'), dict) else None
    print(f'HTTP_{code}' if code else 'ERROR')
else:
    print('ERROR')
"
}

# Append one JSONL record to $OUT. Args: <question> <status> <latency> <answer> <raw>
write_record() {
    Q="$1" ST="$2" DT="$3" ANS="$4" RAW="$5" OUTFILE="$OUT" python3 -c "
import json, os
raw_str = os.environ['RAW']
try:
    raw = json.loads(raw_str) if raw_str else {}
except Exception:
    raw = {'_unparseable': raw_str[:500]}
rec = {
    'question':  os.environ['Q'],
    'status':    os.environ['ST'],
    'latency_s': float(os.environ['DT']),
    'answer':    os.environ['ANS'],
    'raw':       raw,
}
with open(os.environ['OUTFILE'], 'a') as f:
    f.write(json.dumps(rec) + '\n')
"
}

# Submit one chat completion; print STATUS\tRAW_JSON to stdout.
run_one() {
    local question="$1"
    local payload resp rc
    payload=$(build_payload "$question")
    # Capture both response body AND curl exit code.
    set +e
    resp=$(curl -sS --max-time "$TIMEOUT" -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$payload" "$CHAT_URL")
    rc=$?
    set -e
    [[ -z "$resp" ]] && resp='{"error":"empty response"}'
    local status
    status=$(classify_status "$resp" "$rc")
    printf '%s\t%s\n' "$status" "$resp"
}

# --- Main loop --------------------------------------------------------------
: > "$OUT"
ok=0
for i in "${!QUESTIONS[@]}"; do
    q="${QUESTIONS[$i]}"
    n=$((i + 1))

    t0=$(date +%s)
    output=$(run_one "$q")
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
