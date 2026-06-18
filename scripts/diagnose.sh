#!/usr/bin/env bash
# scripts/diagnose.sh — post-mortem health check for a pipeline run.
#
# Run this when a phase fails with a confusing downstream error, OR after
# extract finishes (suspiciously fast/slow) to catch silent-success failures
# BEFORE they poison graphmert (or later phases).
#
# Today it covers extract → graphmert handoff:
#   1. Corpus presence/size sanity        — wrong-corpus-synced detection
#   2. Extract output (kg_final.csv)      — empty/header-only detection
#   3. GraphRAG artifacts                  — did indexing actually produce data?
#   4. Extract phase log scan              — silent warnings/errors
#   5. graphmert intermediate state        — stale/empty dataset detection
#   6. Verdict                             — pass/fail + recommended action
#
# Exit code: 0 if all checks pass, 1 if any FAIL, 2 if WARN-only.
# Designed to run in <5 sec — read-only, no side effects.
#
# Usage:
#   ./scripts/diagnose.sh                   # latest run from $OUTPUT_BASE/logs
#   ./scripts/diagnose.sh --run <prefix>    # specific (matches prefix like logs.sh)
#   ./scripts/diagnose.sh --quiet           # only print VERDICT + failed sections
#   ./scripts/diagnose.sh --tee <file>      # also write to file
#   ./scripts/diagnose.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

# Auto-source .env to pick up CORPUS_PATH for the wrong-corpus check.
_env_file="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$_env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$_env_file"
    set +a
fi

# --- args -------------------------------------------------------------------
RUN_ID=""
QUIET=0
TEE_FILE=""

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)   RUN_ID="$2"; shift 2 ;;
        --quiet) QUIET=1; shift ;;
        --tee)   TEE_FILE="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# --- output capture (optional tee) ------------------------------------------
if [[ -n "$TEE_FILE" ]]; then
    mkdir -p "$(dirname "$TEE_FILE")"
    : > "$TEE_FILE"
    exec > >(tee -a "$TEE_FILE") 2>&1
fi

# --- run id resolution ------------------------------------------------------
LOGS_BASE="$OUTPUT_BASE/logs"
if [[ -z "$RUN_ID" ]]; then
    RUN_ID=$(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                 | sort -r | head -1 || true)
    [[ -z "$RUN_ID" ]] && { echo "No runs found under $LOGS_BASE/"; exit 1; }
elif [[ ! -d "$LOGS_BASE/$RUN_ID" ]]; then
    match=$(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                | grep -E "^${RUN_ID}" | sort -r | head -1)
    [[ -z "$match" ]] && { echo "No run matching: $RUN_ID" >&2; exit 1; }
    RUN_ID="$match"
fi
LOG_DIR="$LOGS_BASE/$RUN_ID"
EXTRACT_LOG_DIR="$LOG_DIR/extract"

# --- counters + state -------------------------------------------------------
FAILS=0
WARNS=0
FINDINGS=()

mark_fail() { echo "  ✗ $*"; FAILS=$((FAILS + 1)); FINDINGS+=("FAIL: $*"); }
mark_warn() { echo "  ⚠ $*"; WARNS=$((WARNS + 1)); FINDINGS+=("WARN: $*"); }
mark_ok()   { echo "  ✓ $*"; }
section()   { [[ "$QUIET" -eq 1 ]] && return 0; echo; echo "=== $* ===" | head -c 70; echo; }
note()      { [[ "$QUIET" -eq 1 ]] && return 0; echo "  $*"; }
human_size() { numfmt --to=iec --suffix=B 2>/dev/null || cat; }

# --- header -----------------------------------------------------------------
echo "=== Pipeline diagnosis $(printf '%.0s=' {1..40})" | head -c 70; echo
echo "Run        : $RUN_ID"
echo "Run dir    : ${LOG_DIR#$REPO_ROOT/}"
echo "Repo       : $REPO_ROOT"
echo "Generated  : $(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# --- 1. Corpus check --------------------------------------------------------
section "1. Corpus"
CORPUS_RESOLVED="${CORPUS_PATH:-}"
if [[ -z "$CORPUS_RESOLVED" ]]; then
    mark_warn "CORPUS_PATH not set in .env — skipping corpus size check"
else
    # CORPUS_PATH is RELATIVE to REPO_ROOT in our symmetric s3-mirror model.
    CORPUS_DIR="$REPO_ROOT/$CORPUS_RESOLVED"
    if [[ ! -d "$CORPUS_DIR" ]]; then
        mark_fail "corpus dir missing: $CORPUS_DIR"
    else
        FCOUNT=$(find "$CORPUS_DIR" -type f \( -name '*.txt' -o -name '*.md' -o -name '*.json' \) 2>/dev/null | wc -l)
        BYTES=$(du -sb "$CORPUS_DIR" 2>/dev/null | awk '{print $1}')
        HSIZE=$(echo "$BYTES" | human_size)
        note "path  : ${CORPUS_DIR#$REPO_ROOT/}"
        note "files : $FCOUNT"
        note "size  : $HSIZE"
        # Smoke fixture is < 100 KB. Pilot/paper Kandel corpus is >5 MB.
        if [[ "$BYTES" -lt 100000 ]]; then
            mark_fail "corpus is tiny ($HSIZE) — looks like a smoke fixture, not a pilot/paper corpus"
        elif [[ "$BYTES" -lt 5000000 ]]; then
            mark_warn "corpus is small ($HSIZE) — verify this is the intended corpus"
        else
            mark_ok "corpus size plausible for pilot/paper"
        fi
    fi
fi

# --- 2. Extract output: kg_final.csv ----------------------------------------
section "2. Extract output (kg_final.csv)"
KG_FINAL="$OUTPUT_BASE/graphrag/output/kg_final.csv"
if [[ ! -f "$KG_FINAL" ]]; then
    mark_fail "kg_final.csv not produced — extract phase never reached the merge step"
    note "expected at: ${KG_FINAL#$REPO_ROOT/}"
else
    LINES=$(wc -l < "$KG_FINAL")
    BYTES=$(stat -c%s "$KG_FINAL" 2>/dev/null || stat -f%z "$KG_FINAL")
    DATA_ROWS=$((LINES - 1))           # subtract header
    [[ "$DATA_ROWS" -lt 0 ]] && DATA_ROWS=0
    note "path  : ${KG_FINAL#$REPO_ROOT/}"
    note "size  : $(echo "$BYTES" | human_size)"
    note "lines : $LINES total ($DATA_ROWS triples + 1 header)"
    if [[ "$QUIET" -eq 0 ]]; then
        note "header: $(head -1 "$KG_FINAL")"
    fi
    if [[ "$DATA_ROWS" -eq 0 ]]; then
        mark_fail "kg_final.csv has 0 triples — silent-success extract failure"
        note "  This is THE bug graphmert preprocess will hit:"
        note "    'Seed KG: 0 triples' → Dataset.from_list([], features=<schema>) → Keys mismatch"
    elif [[ "$DATA_ROWS" -lt 10 ]]; then
        mark_warn "only $DATA_ROWS triples — graphmert may train but with poor signal"
    else
        mark_ok "$DATA_ROWS triples"
    fi
fi

# --- 3. GraphRAG artifacts (did indexing produce data even if CSV merge failed?) ---
section "3. GraphRAG intermediate artifacts"
python3 - "$OUTPUT_BASE" "$REPO_ROOT" <<'PY' 2>&1 || true
import glob, os, sys
out_base, repo_root = sys.argv[1], sys.argv[2]
patterns = {
    "entities":      ["**/entities*.parquet", "**/create_final_entities*"],
    "relationships": ["**/relationships*.parquet", "**/create_final_relationships*"],
}
graphrag_root = os.path.join(out_base, "graphrag")
if not os.path.isdir(graphrag_root):
    print(f"  ⚠ no graphrag dir at {graphrag_root}")
    sys.exit(0)
findings = {}
for kind, pats in patterns.items():
    paths = []
    for p in pats:
        paths.extend(glob.glob(os.path.join(graphrag_root, p), recursive=True))
    # Dedup; prefer parquet over directory variants
    paths = sorted(set(paths))
    findings[kind] = paths

try:
    import pandas as pd
    have_pd = True
except ImportError:
    have_pd = False
    print("  ⚠ pandas unavailable — counting files only, not rows")

for kind, paths in findings.items():
    if not paths:
        print(f"  ⚠ {kind}: no artifacts found")
        continue
    total_rows = 0
    for p in paths:
        rel = p[len(repo_root) + 1:] if p.startswith(repo_root) else p
        if p.endswith(".parquet") and have_pd:
            try:
                df = pd.read_parquet(p)
                total_rows += len(df)
                print(f"  ✓ {rel}: {len(df):,} rows")
            except Exception as e:
                print(f"  ⚠ {rel}: read error ({e})")
        else:
            print(f"  - {rel}: (dir/no-pandas)")
    if have_pd and total_rows == 0 and paths:
        print(f"  ✗ {kind}: 0 rows across {len(paths)} file(s)")
PY

# --- 4. Extract phase log scan ----------------------------------------------
section "4. Extract phase log scan"
if [[ ! -d "$EXTRACT_LOG_DIR" ]]; then
    mark_warn "no extract step logs at ${EXTRACT_LOG_DIR#$REPO_ROOT/}"
else
    EXTRACT_LOGS=$(find "$EXTRACT_LOG_DIR" -name '*.log' -type f 2>/dev/null | sort)
    if [[ -z "$EXTRACT_LOGS" ]]; then
        mark_warn "no .log files under ${EXTRACT_LOG_DIR#$REPO_ROOT/}"
    else
        # shellcheck disable=SC2086
        TOTAL_BYTES=$(cat $EXTRACT_LOGS | wc -c)
        ERR=$(grep -hiE 'error|exception|traceback' $EXTRACT_LOGS 2>/dev/null | grep -vE 'INFO|^$' | wc -l)
        WRN=$(grep -hiE '\bwarn(ing)?\b' $EXTRACT_LOGS 2>/dev/null | wc -l)
        ZERO_HITS=$(grep -hiE '\b0 (entit|relat|trip|rows|results|chunks)' $EXTRACT_LOGS 2>/dev/null | wc -l)
        note "log size : $(echo "$TOTAL_BYTES" | human_size) across $(echo "$EXTRACT_LOGS" | wc -l) file(s)"
        note "errors   : $ERR matching error/exception/traceback"
        note "warnings : $WRN"
        note "0-counts : $ZERO_HITS log lines mentioning '0 entities/relations/triples/...'"
        if [[ "$ZERO_HITS" -gt 0 ]]; then
            mark_warn "extract log mentions zero-result counts ($ZERO_HITS hits) — likely the silent failure"
            note "  recent zero hits:"
            grep -hiE '\b0 (entit|relat|trip|rows|results|chunks)' $EXTRACT_LOGS 2>/dev/null | tail -3 | sed 's/^/    /'
        fi
        if [[ "$ERR" -gt 0 ]]; then
            mark_warn "$ERR error/exception lines in extract logs"
            note "  most recent error context:"
            grep -hniE 'error|exception|traceback' $EXTRACT_LOGS 2>/dev/null | tail -3 | sed 's/^/    /'
        fi
        [[ "$ERR" -eq 0 && "$WRN" -eq 0 && "$ZERO_HITS" -eq 0 ]] && mark_ok "no errors / warnings / zero-result hits in extract logs"
    fi
fi

# --- 5. graphmert intermediate state ----------------------------------------
section "5. graphmert intermediate state"
GRAPHMERT_DIR="$OUTPUT_BASE/graphmert"
if [[ ! -d "$GRAPHMERT_DIR" ]]; then
    note "no graphmert outputs yet (phase has not run on this RUN_ID)"
else
    # Check each known intermediate save_to_disk target: schema vs row count.
    python3 - "$GRAPHMERT_DIR" "$REPO_ROOT" <<'PY' 2>&1 || true
import json, os, sys
gm_dir, repo_root = sys.argv[1], sys.argv[2]
candidates = [
    "head_positions",
    "llm_relations/relations_all",
    "llm_relations/relations_cleaned_train",
    "llm_relations/relations_cleaned_eval",
    "dataset/preprocessed_train",
    "dataset/preprocessed_eval",
]
any_bad = False
for sub in candidates:
    d = os.path.join(gm_dir, sub)
    if not os.path.isdir(d):
        continue
    info = os.path.join(d, "dataset_info.json")
    arrows = [f for f in os.listdir(d) if f.endswith(".arrow") and not f.startswith("cache-")]
    rel = d[len(repo_root) + 1:] if d.startswith(repo_root) else d
    if not os.path.exists(info):
        print(f"  ⚠ {rel}: no dataset_info.json (incomplete save?)")
        any_bad = True; continue
    if not arrows:
        print(f"  ⚠ {rel}: no data-*.arrow (only cache files? incomplete save?)")
        any_bad = True; continue
    try:
        meta = json.load(open(info))
        feats = list((meta.get("features") or {}).keys())
        splits = meta.get("splits") or {}
        rows = sum(s.get("num_examples", 0) for s in splits.values()) if isinstance(splits, dict) else 0
        sizes = []
        for a in arrows:
            sz = os.path.getsize(os.path.join(d, a))
            sizes.append((a, sz))
        size_str = ", ".join(f"{a}={sz} B" for a, sz in sizes)
        print(f"  ✓ {rel}: {rows} rows, {len(feats)} features ({size_str})")
        if rows == 0 and feats:
            print(f"    ✗ schema declared ({len(feats)} features) but 0 rows — load_from_disk will Keys-mismatch")
            any_bad = True
        for _, sz in sizes:
            if sz < 1024:
                print(f"    ✗ arrow file < 1 KB — corrupt/partial write")
                any_bad = True
    except Exception as e:
        print(f"  ⚠ {rel}: cannot parse dataset_info.json ({e})")
        any_bad = True
if not any_bad:
    pass  # all good
PY
fi

# --- 6. Verdict -------------------------------------------------------------
echo
echo "=== VERDICT $(printf '%.0s=' {1..50})" | head -c 70; echo
if [[ "$FAILS" -eq 0 && "$WARNS" -eq 0 ]]; then
    echo "✓ All checks passed. Pipeline state is consistent."
    EXIT=0
elif [[ "$FAILS" -eq 0 ]]; then
    echo "⚠ $WARNS warning(s) — pipeline may still work but inspect before continuing."
    EXIT=2
else
    echo "✗ $FAILS failure(s), $WARNS warning(s) — DO NOT proceed without fixing root cause."
    EXIT=1
fi

if [[ ${#FINDINGS[@]} -gt 0 ]]; then
    echo
    echo "Findings:"
    for f in "${FINDINGS[@]}"; do
        echo "  - $f"
    done
fi

# --- Recommended action when extract is empty (the dominant failure mode) ---
if [[ -f "$KG_FINAL" ]]; then
    DATA_ROWS=$(($(wc -l < "$KG_FINAL") - 1))
    if [[ "$DATA_ROWS" -le 0 ]]; then
        echo
        echo "Recommended action — empty seed KG:"
        echo "  1. DON'T restart graphmert yet (it will fail the same way in 2+ hours)"
        echo "  2. Identify why kg_final.csv is empty:"
        echo "       - corpus too small?           (Section 1)"
        echo "       - graphrag indexing failed?   (Section 3)"
        echo "       - silent zero-row writes?     (Section 4)"
        echo "  3. Fix the root cause, then rerun extract phase only:"
        echo "       rm -rf $OUTPUT_BASE/graphrag $OUTPUT_BASE/graphmert"
        echo "       ./scripts/pipeline.sh --profile <p> --platform <pl> --phase extract"
    fi
fi

echo
exit "$EXIT"
