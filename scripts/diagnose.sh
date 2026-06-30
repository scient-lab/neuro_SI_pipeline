#!/usr/bin/env bash
# scripts/diagnose.sh — post-mortem health check for a pipeline run.
#
# Run this when a phase fails with a confusing downstream error, OR after
# extract finishes (suspiciously fast/slow) to catch silent-success failures
# BEFORE they poison graphmert (or later phases).
#
# Today it covers extract → graphmert → curriculum handoff. Each section
# is tagged with a (PHASE, STEP) — use --phase / --step to filter scope,
# --deep for depth.
#
#   §   Title                                Phase        Step
#   --  -----------------------------------  ----------   -----------
#   1   Corpus presence/size sanity          extract      -
#   2   Extract output (kg_final.csv)        extract      build_kg
#   3   GraphRAG artifacts                   extract      index
#   4   Extract phase log scan               extract      -
#   5   graphmert intermediate state         graphmert    preprocess
#   6   GraphRAG internals (--deep)          extract      -
#   7   graphmert internals (--deep)         graphmert    -
#   8   curriculum.generate_qa progress      curriculum   generate_qa
#   9   Verdict                              (always runs)
#
# Filters compose like pipeline.sh:
#   --phase extract              → §1, §2, §3, §4 (+ §6 with --deep)
#   --phase extract --step index → §1, §3        (+ §6 with --deep)
#   --phase graphmert            → §5             (+ §7 with --deep)
#   --phase curriculum           → §8
#   no filter                    → all
#
# Exit code: 0 if all checks pass, 1 if any FAIL, 2 if WARN-only.
# Fast mode (~5 sec) by default; --deep adds ~10-15 sec.
#
# Usage:
#   ./scripts/diagnose.sh                                       # all phases
#   ./scripts/diagnose.sh --phase extract                       # scope to extract
#   ./scripts/diagnose.sh --phase extract --step build_kg       # narrow further
#   ./scripts/diagnose.sh --phase graphmert --deep              # graphmert internals
#   ./scripts/diagnose.sh --phase curriculum                    # live generate_qa status
#   watch -n 10 ./scripts/diagnose.sh --phase curriculum        # live polling
#   ./scripts/diagnose.sh --run <prefix>                        # specific historical run
#   ./scripts/diagnose.sh --quiet                               # VERDICT + failures only
#   ./scripts/diagnose.sh --tee <file>                          # also write to file
#   ./scripts/diagnose.sh --std                                 # standardized view (all phases;
#                                                               #   status + exception(file:line) + I/O contract)
#   ./scripts/diagnose.sh --std --phase graphmert --json        # standardized, machine-readable
#   ./scripts/diagnose.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

# Find a python that has pandas (for parquet introspection) and json (always
# stdlib). Prefer venvs in order graphmert > graphrag > si_curriculum; fall
# back to system python3. We don't want to fail diagnose if pandas is absent
# — just degrade gracefully (parquet counts go from rows to file presence).
pick_py_with_pandas() {
    local p
    for p in "$REPO_ROOT/.venvs/graphmert/bin/python" \
             "$REPO_ROOT/.venvs/graphrag/bin/python" \
             "$REPO_ROOT/.venvs/si_curriculum/bin/python" \
             "$(command -v python3 2>/dev/null || echo /nonexistent)"; do
        [[ -x "$p" ]] || continue
        if "$p" -c 'import pandas' 2>/dev/null; then
            echo "$p"; return
        fi
    done
    # No venv has pandas — fall back to system python3 (parquet stats skipped).
    command -v python3 2>/dev/null || echo ""
}
PY="$(pick_py_with_pandas)"
[[ -z "$PY" ]] && { echo "no python3 found"; exit 1; }

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
DEEP=0
PHASE_FILTER=""
STEP_FILTER=""
STD=0
JSON_MODE=0

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)    RUN_ID="$2"; shift 2 ;;
        --phase)  PHASE_FILTER="$2"; shift 2 ;;
        --step)   STEP_FILTER="$2"; shift 2 ;;
        --quiet)  QUIET=1; shift ;;
        --tee)    TEE_FILE="$2"; shift 2 ;;
        --deep)   DEEP=1; shift ;;
        --std)    STD=1; shift ;;     # standardized view (scripts/lib/checks_view.py)
        --json)   JSON_MODE=1; shift ;;  # only with --std
        --help|-h) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# should_run <section_phase> [<section_step>]
# Decides whether a section runs given --phase / --step filters.
#  - No filters → always run
#  - --phase set → must match section's phase
#  - --step set → must also match section's step (sections with no step
#    are phase-wide and still match)
should_run() {
    local sec_phase="$1" sec_step="${2:-}"
    [[ -z "$PHASE_FILTER" ]] && return 0
    [[ "$PHASE_FILTER" != "$sec_phase" ]] && return 1
    [[ -z "$STEP_FILTER" ]] && return 0
    [[ -z "$sec_step" ]] && return 0
    [[ "$STEP_FILTER" != "$sec_step" ]] && return 1
    return 0
}

# --- output capture (optional tee) ------------------------------------------
if [[ -n "$TEE_FILE" ]]; then
    mkdir -p "$(dirname "$TEE_FILE")"
    : > "$TEE_FILE"
    exec > >(tee -a "$TEE_FILE") 2>&1
fi

# --- standardized view (--std): dispatch to the shared checks engine --------
# Opt-in for now: renders scripts/lib/checks_view.py (HEALTH lens — status +
# exception(file:line) + I/O-contract state). The legacy §-sections below stay
# the default until all phases are ported. See
# docs/DIAGNOSE_ANALYSIS_STANDARDIZATION_PLAN_2026-06-29.md.
if [[ "$STD" -eq 1 ]]; then
    sv=( "$PY" "$SCRIPT_DIR/lib/checks_view.py" --lens health --output-base "$OUTPUT_BASE" )
    [[ -n "$PHASE_FILTER" ]] && sv+=( --phase "$PHASE_FILTER" )
    [[ -n "$STEP_FILTER" ]]  && sv+=( --step "$STEP_FILTER" )
    [[ -n "$RUN_ID" ]]       && sv+=( --run "$RUN_ID" )
    [[ "$JSON_MODE" -eq 1 ]] && sv+=( --json )
    exec "${sv[@]}"
fi

# --- run id resolution ------------------------------------------------------
# Two layouts to support:
#   (A) per-run OUTPUT_BASE — outputs/<RUN_ID>/logs/adhoc/<phase>/    (ad-hoc / smoke)
#   (B) shared logs dir    — outputs/logs/<RUN_ID>/<phase>/           (pipeline.sh-driven)
# Prefer (A) if `outputs/<RUN_ID>/logs/` exists, else fall back to (B).
LOGS_BASE="$OUTPUT_BASE/logs"
RUN_PARENT="$OUTPUT_BASE"
if [[ ! -d "$LOGS_BASE" && -d "$RUN_PARENT" ]]; then
    LOGS_BASE="$RUN_PARENT"
fi
if [[ ! -d "$LOGS_BASE" ]]; then
    # Tell the operator WHERE we looked — silently exit is the worst UX.
    # Send to both stdout AND stderr so it survives both bare invocations
    # and `... 2>&1 | grep`-style filters.
    echo "ERROR: $LOGS_BASE does not exist. Run from repo root, or set OUTPUT_BASE." | tee /dev/stderr
    echo "  cwd:        $PWD" | tee /dev/stderr
    echo "  REPO_ROOT:  $REPO_ROOT" | tee /dev/stderr
    echo "  OUTPUT_BASE: $OUTPUT_BASE" | tee /dev/stderr
    exit 1
fi
if [[ -z "$RUN_ID" ]]; then
    RUN_ID=$(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                 | grep -E '^[0-9]{8}-[0-9]{6}' | sort -r | head -1 || true)
    if [[ -z "$RUN_ID" ]]; then
        echo "ERROR: no runs matching YYYYMMDD-HHMMSS under $LOGS_BASE/" | tee /dev/stderr
        exit 1
    fi
elif [[ ! -d "$LOGS_BASE/$RUN_ID" ]]; then
    match=$(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                | grep -E "^${RUN_ID}" | sort -r | head -1)
    if [[ -z "$match" ]]; then
        echo "ERROR: no run matching '$RUN_ID' under $LOGS_BASE/" | tee /dev/stderr
        echo "  available runs:" | tee /dev/stderr
        find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '    %f\n' 2>/dev/null \
            | sort -r | head -5 | tee /dev/stderr
        exit 1
    fi
    RUN_ID="$match"
fi
# LOG_DIR convention differs between the two layouts (see above). Resolve
# both candidates; §-sections pick whichever exists.
if [[ -d "$LOGS_BASE/$RUN_ID/logs/adhoc" ]]; then
    # Layout (A): outputs/<RUN_ID>/logs/adhoc/<phase>/
    LOG_DIR="$LOGS_BASE/$RUN_ID/logs/adhoc"
else
    # Layout (B): outputs/logs/<RUN_ID>/<phase>/
    LOG_DIR="$LOGS_BASE/$RUN_ID"
fi
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

# --- 1. Corpus check (phase=extract) ----------------------------------------
if should_run extract; then
section "1. Corpus"
CORPUS_RESOLVED="${CORPUS_PATH:-}"
if [[ -z "$CORPUS_RESOLVED" ]]; then
    mark_warn "CORPUS_PATH not set in .env — skipping corpus size check"
else
    # CORPUS_PATH is RELATIVE to REPO_ROOT in our symmetric s3-mirror model.
    # It can point to either a directory OR a single file — handle both.
    CORPUS_TGT="$REPO_ROOT/$CORPUS_RESOLVED"
    if [[ -f "$CORPUS_TGT" ]]; then
        # Single-file corpus (e.g. one textbook).
        BYTES=$(stat -c%s "$CORPUS_TGT" 2>/dev/null || stat -f%z "$CORPUS_TGT")
        HSIZE=$(echo "$BYTES" | human_size)
        note "path  : ${CORPUS_TGT#$REPO_ROOT/}  (single file)"
        note "size  : $HSIZE"
    elif [[ -d "$CORPUS_TGT" ]]; then
        FCOUNT=$(find "$CORPUS_TGT" -type f \( -name '*.txt' -o -name '*.md' -o -name '*.json' \) 2>/dev/null | wc -l)
        BYTES=$(du -sb "$CORPUS_TGT" 2>/dev/null | awk '{print $1}')
        HSIZE=$(echo "$BYTES" | human_size)
        note "path  : ${CORPUS_TGT#$REPO_ROOT/}  (directory)"
        note "files : $FCOUNT"
        note "size  : $HSIZE"
    else
        mark_fail "corpus path missing (neither file nor dir): $CORPUS_TGT"
        BYTES=0
    fi
    if [[ "${BYTES:-0}" -gt 0 ]]; then
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

fi  # end §1

# --- 2. Extract output: kg_final.csv (phase=extract step=build_kg) ----------
if should_run extract build_kg; then
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

fi  # end §2

# --- 3. GraphRAG artifacts (phase=extract step=index) -----------------------
if should_run extract index; then
section "3. GraphRAG intermediate artifacts"
"$PY" - "$OUTPUT_BASE" "$REPO_ROOT" <<'PY' 2>&1 || true
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

fi  # end §3

# --- 4. Extract phase log scan (phase=extract, all steps) -------------------
if should_run extract; then
section "4. Extract phase log scan"
if [[ ! -d "$EXTRACT_LOG_DIR" ]]; then
    mark_warn "no extract step logs at ${EXTRACT_LOG_DIR#$REPO_ROOT/}"
else
    EXTRACT_LOGS=$(find "$EXTRACT_LOG_DIR" -name '*.log' -type f 2>/dev/null | sort)
    if [[ -z "$EXTRACT_LOGS" ]]; then
        mark_warn "no .log files under ${EXTRACT_LOG_DIR#$REPO_ROOT/}"
    else
        # Disable pipefail in this block: a grep that finds 0 matches returns
        # exit 1, which combined with `set -e -o pipefail` will silently kill
        # the script mid-section. We WANT 0 matches to be a valid result.
        set +o pipefail
        # shellcheck disable=SC2086
        TOTAL_BYTES=$(cat $EXTRACT_LOGS | wc -c)
        ERR=$(grep -hiE 'error|exception|traceback' $EXTRACT_LOGS 2>/dev/null | grep -vE 'INFO|^$' | wc -l)
        WRN=$(grep -hiE '\bwarn(ing)?\b' $EXTRACT_LOGS 2>/dev/null | wc -l)
        ZERO_HITS=$(grep -hiE '\b0 (entit|relat|trip|rows|results|chunks)' $EXTRACT_LOGS 2>/dev/null | wc -l)
        # Also scan for graphrag-specific empty-output signals:
        REL_HITS=$(grep -hiE '(extract_relationships|relationship.*extract).*\b(0|empty|none|fail)' $EXTRACT_LOGS 2>/dev/null | wc -l)
        set -o pipefail
        note "log size : $(echo "$TOTAL_BYTES" | human_size) across $(echo "$EXTRACT_LOGS" | wc -l) file(s)"
        note "errors   : $ERR matching error/exception/traceback"
        note "warnings : $WRN"
        note "0-counts : $ZERO_HITS log lines mentioning '0 entities/relations/triples/...'"
        note "rel-hits : $REL_HITS log lines mentioning relationship extraction failures"
        set +o pipefail
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
        if [[ "$REL_HITS" -gt 0 ]]; then
            mark_warn "extract log mentions relationship-extraction failures ($REL_HITS hits)"
            note "  recent rel-hits:"
            grep -hiE '(extract_relationships|relationship.*extract).*\b(0|empty|none|fail)' $EXTRACT_LOGS 2>/dev/null | tail -3 | sed 's/^/    /'
        fi
        set -o pipefail
        [[ "$ERR" -eq 0 && "$WRN" -eq 0 && "$ZERO_HITS" -eq 0 && "$REL_HITS" -eq 0 ]] && mark_ok "no errors / warnings / zero-result hits in extract logs"
    fi
fi

fi  # end §4

# --- 5. graphmert intermediate state (phase=graphmert step=preprocess) ------
if should_run graphmert preprocess; then
section "5. graphmert intermediate state"
GRAPHMERT_DIR="$OUTPUT_BASE/graphmert"
if [[ ! -d "$GRAPHMERT_DIR" ]]; then
    note "no graphmert outputs yet (phase has not run on this RUN_ID)"
else
    # Check each known intermediate save_to_disk target:
    #   - schema vs row count (catches "0 rows but schema declared")
    #   - REQUIRED columns for downstream consumers (catches KeyError on
    #     missing fields — e.g. ground_triples_to_snippets needs 'id' in
    #     relations_cleaned_train; absence crashed the 2026-06-19 pilot)
    #   - arrow file sanity (corrupt / partial writes)
    "$PY" - "$GRAPHMERT_DIR" "$REPO_ROOT" <<'PY' 2>&1 || true
import json, os, sys
gm_dir, repo_root = sys.argv[1], sys.argv[2]

# Map: dataset path → columns the downstream consumer expects. When a
# required column is missing from features, the consumer will KeyError on
# every row. The "consumer" hint points at the file:function that fails.
candidates = [
    {
        "path": "head_positions",
        "required": ["input_ids", "head_positions"],
        "consumer": "dataset_preprocessing_utils.ground_triples_to_snippets",
    },
    {
        "path": "llm_relations/relations_all",
        "required": [],  # interim — only relations_cleaned_* is read downstream
        "consumer": "clean_llm_relations.py (interim)",
    },
    {
        "path": "llm_relations/relations_cleaned_train",
        "required": ["id", "input_ids", "head_positions"],
        "consumer": "dataset_preprocessing_utils.ground_triples_to_snippets",
    },
    {
        "path": "llm_relations/relations_cleaned_eval",
        "required": ["id", "input_ids", "head_positions"],
        "consumer": "dataset_preprocessing_utils.ground_triples_to_snippets",
    },
    {
        "path": "dataset/preprocessed_train",
        "required": ["id", "input_nodes", "attention_mask", "leaf_relationships",
                     "head_lengths", "start_indices", "special_tokens_mask"],
        "consumer": "run_mlm.py (GraphMERT MNM training)",
    },
    {
        "path": "dataset/preprocessed_eval",
        "required": ["id", "input_nodes", "attention_mask", "leaf_relationships",
                     "head_lengths", "start_indices", "special_tokens_mask"],
        "consumer": "run_mlm.py (GraphMERT MNM training)",
    },
]

any_bad = False
for cand in candidates:
    sub, required, consumer = cand["path"], cand["required"], cand["consumer"]
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
        sizes = [(a, os.path.getsize(os.path.join(d, a))) for a in arrows]
        size_str = ", ".join(f"{a}={sz} B" for a, sz in sizes)
        print(f"  ✓ {rel}: {rows} rows, {len(feats)} features ({size_str})")
        # Check required columns for downstream consumer
        missing = [c for c in required if c not in feats]
        if missing:
            print(f"    ✗ missing columns: {missing}  (consumer: {consumer})")
            print(f"       full feature list: {feats}")
            print(f"       → downstream will KeyError on these field accesses")
            any_bad = True
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

fi  # end §5

# --- 6. GraphRAG internals — deep (phase=extract) ---------------------------
# Drills into graphrag's own state to localize WHY relationship extraction
# produced 0 rows when entities did not. Covers:
#   - graphrag config + entry-point script
#   - LLM cache (was the relation-extraction LLM call ever made?)
#   - entity sample (does the type taxonomy look right for the domain?)
#   - relationships sample (when non-zero, show one or two)
#   - graphrag-side log files for the indexing pipeline
# Read-only. Adds ~5-10 sec — that's why it's behind --deep.
if [[ "$DEEP" -eq 1 ]] && should_run extract; then
    section "6. GraphRAG internals (--deep)"
    GRAPHRAG_DIR="$OUTPUT_BASE/graphrag"
    if [[ ! -d "$GRAPHRAG_DIR" ]]; then
        mark_warn "no graphrag output dir at ${GRAPHRAG_DIR#$REPO_ROOT/}"
    else
        # 6a. Config + entry-point
        note "config:"
        for f in "$REPO_ROOT/1_seed_kg/graphrag_index.py" \
                 "$REPO_ROOT/configs/default.yaml" \
                 "$GRAPHRAG_DIR/settings.yaml" \
                 "$REPO_ROOT/1_seed_kg/settings.yaml"; do
            if [[ -f "$f" ]]; then
                note "  ✓ ${f#$REPO_ROOT/}  ($(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") B)"
            fi
        done
        # graphrag prompt files (custom domain prompts if any)
        prompts=$(find "$REPO_ROOT/1_seed_kg" "$GRAPHRAG_DIR" -path '*prompt*' -name '*.txt' 2>/dev/null | head -5)
        if [[ -n "$prompts" ]]; then
            note "  prompts (custom):"
            while IFS= read -r p; do
                note "    ${p#$REPO_ROOT/}"
            done <<< "$prompts"
        else
            note "  prompts : using graphrag built-in defaults (no custom .txt files found)"
        fi

        # 6b. LLM call cache
        note "cache:"
        cache_root=""
        for c in "$GRAPHRAG_DIR/cache" "$GRAPHRAG_DIR/lancedb_cache" "$GRAPHRAG_DIR/output/cache"; do
            [[ -d "$c" ]] && { cache_root="$c"; break; }
        done
        if [[ -z "$cache_root" ]]; then
            note "  ⚠ no cache dir found — graphrag may not have cached LLM calls (rerun would re-pay)"
        else
            cache_files=$(find "$cache_root" -type f 2>/dev/null | wc -l)
            cache_bytes=$(du -sb "$cache_root" 2>/dev/null | awk '{print $1}')
            note "  root  : ${cache_root#$REPO_ROOT/}"
            note "  files : $cache_files  ($(echo "${cache_bytes:-0}" | human_size))"
            # Bucket cache files by step name to see relation-extraction call count.
            for step in entity_extraction extract_graph extract_relationships create_communities summarize_communities; do
                hits=$(find "$cache_root" -type f -name "*${step}*" 2>/dev/null | wc -l)
                [[ "$hits" -gt 0 ]] && note "  ${step}: $hits cached calls"
            done
            # Sample one cache entry to see what the LLM was asked + returned
            sample=$(find "$cache_root" -type f 2>/dev/null | head -1)
            if [[ -n "$sample" ]] && [[ "$QUIET" -eq 0 ]]; then
                note "  sample : ${sample#$REPO_ROOT/}"
                note "    (first 400 chars):"
                head -c 400 "$sample" 2>/dev/null | sed 's/^/      /'
                echo
            fi
        fi

        # 6c. Entity + relationship parquet samples
        "$PY" - "$GRAPHRAG_DIR" "$REPO_ROOT" <<'PY' 2>&1 || true
import glob, os, sys
try:
    import pandas as pd
except ImportError:
    print("  ⚠ pandas unavailable — skipping parquet samples")
    sys.exit(0)
gr_dir, repo_root = sys.argv[1], sys.argv[2]
for kind in ["entities", "relationships"]:
    paths = sorted(glob.glob(os.path.join(gr_dir, "**", f"{kind}*.parquet"), recursive=True))
    if not paths:
        print(f"  ⚠ no {kind} parquet found")
        continue
    p = paths[0]
    rel = p[len(repo_root) + 1:] if p.startswith(repo_root) else p
    try:
        df = pd.read_parquet(p)
        print(f"  {kind}: {len(df):,} rows  columns={list(df.columns)}")
        if len(df) == 0:
            print(f"    (empty — no sample to print)")
        else:
            # Pick the most informative columns if present.
            preferred = {
                "entities": ["title", "type", "description"],
                "relationships": ["source", "target", "description", "weight"],
            }
            cols = [c for c in preferred[kind] if c in df.columns] or list(df.columns)[:4]
            with pd.option_context("display.max_colwidth", 60, "display.width", 120):
                print("    sample (first 5 rows, key columns):")
                for line in df[cols].head(5).to_string(index=False).splitlines():
                    print(f"      {line}")
    except Exception as e:
        print(f"  ⚠ {rel}: read error ({e})")
PY

        # 6d. GraphRAG-side log files (separate from our phase logs)
        note "graphrag logs:"
        gr_logs=$(find "$GRAPHRAG_DIR" -maxdepth 3 -name '*.log' -o -name 'indexing-engine.log' -o -name 'logs.json' 2>/dev/null | head -10)
        if [[ -z "$gr_logs" ]]; then
            note "  ⚠ no .log files found under graphrag/ — indexing may not have run with file logging"
        else
            while IFS= read -r lg; do
                sz=$(stat -c%s "$lg" 2>/dev/null || stat -f%z "$lg")
                note "  ${lg#$REPO_ROOT/}  ($(echo "$sz" | human_size))"
                # Tail any "error" or "0 ..." lines from each graphrag log
                set +o pipefail
                rel_signal=$(grep -hiE 'relation|relationship' "$lg" 2>/dev/null | grep -iE 'error|fail|0\b|empty|none' | tail -3)
                set -o pipefail
                if [[ -n "$rel_signal" ]]; then
                    note "    relationship-related signal:"
                    echo "$rel_signal" | sed 's/^/      /'
                fi
            done <<< "$gr_logs"
        fi
    fi
fi

# --- 7. graphmert internals — deep (phase=graphmert) ------------------------
# Drills into graphmert's own state. Covers:
#   - resolved args_mlm.yaml (envsubst gaps, wrong paths)
#   - stable tokenizer state (vocab size, special tokens)
#   - per-substep dataset samples (relations_all, relations_cleaned_*)
#   - graphmert step-log grep for "0 relations" / OOM / skip patterns
#   - any partial MLM checkpoint state
if [[ "$DEEP" -eq 1 ]] && should_run graphmert; then
    section "7. graphmert internals (--deep)"
    GRAPHMERT_DIR="$OUTPUT_BASE/graphmert"
    if [[ ! -d "$GRAPHMERT_DIR" ]]; then
        mark_warn "no graphmert output dir at ${GRAPHMERT_DIR#$REPO_ROOT/}"
    else
        # 7a. Resolved args_mlm.yaml — every step downstream reads this
        note "args_mlm.resolved.yaml:"
        ARGS_RES="$GRAPHMERT_DIR/args_mlm.resolved.yaml"
        if [[ ! -f "$ARGS_RES" ]]; then
            note "  ⚠ not found at ${ARGS_RES#$REPO_ROOT/} — envsubst didn't run"
        else
            note "  path : ${ARGS_RES#$REPO_ROOT/}  ($(stat -c%s "$ARGS_RES" 2>/dev/null || stat -f%z "$ARGS_RES") B)"
            # Hunt unresolved ${VAR} placeholders (envsubst gaps)
            set +o pipefail
            unresolved=$(grep -cE '\${[A-Z_][A-Z0-9_]*}' "$ARGS_RES" 2>/dev/null || echo 0)
            set -o pipefail
            if [[ "$unresolved" -gt 0 ]]; then
                mark_fail "$unresolved unresolved \${VAR} placeholders in args_mlm.resolved.yaml — envsubst missed env vars"
                note "  unresolved patterns:"
                grep -hE '\${[A-Z_][A-Z0-9_]*}' "$ARGS_RES" | head -5 | sed 's/^/    /'
            else
                mark_ok "no unresolved \${VAR} placeholders"
            fi
            # Key paths
            for k in train_src eval_src injections_train_path injections_eval_path \
                     relation_map_path tokenizer_name preprocessing_output_root; do
                v=$(grep -E "^${k}:" "$ARGS_RES" 2>/dev/null | head -1 | sed "s/^${k}:[[:space:]]*//")
                [[ -n "$v" ]] && note "  $k = $v"
            done
        fi

        # 7b. Stable tokenizer state
        note "stable_tokenizer:"
        TOK_DIR="$GRAPHMERT_DIR/stable_tokenizer"
        if [[ ! -d "$TOK_DIR" ]]; then
            mark_warn "tokenizer dir missing at ${TOK_DIR#$REPO_ROOT/}"
        else
            tok_files=$(ls "$TOK_DIR" 2>/dev/null | wc -l)
            tok_size=$(du -sb "$TOK_DIR" 2>/dev/null | awk '{print $1}')
            note "  path  : ${TOK_DIR#$REPO_ROOT/}  ($tok_files files, $(echo "${tok_size:-0}" | human_size))"
            # Sanity-check vocab size — small vocab = broken tokenizer
            "$PY" - "$TOK_DIR" <<'PY' 2>&1 || true
import json, os, sys
tok_dir = sys.argv[1]
for fname in ("tokenizer.json", "vocab.json", "vocab.txt"):
    p = os.path.join(tok_dir, fname)
    if not os.path.exists(p):
        continue
    try:
        if fname.endswith(".json"):
            d = json.load(open(p))
            # tokenizer.json: model.vocab or added_tokens; vocab.json: dict
            vocab = d.get("model", {}).get("vocab") or d
            if isinstance(vocab, dict):
                print(f"  ✓ {fname}: {len(vocab):,} vocab entries")
            else:
                print(f"  ⚠ {fname}: unexpected format")
        else:
            with open(p) as f:
                n = sum(1 for _ in f)
            print(f"  ✓ {fname}: {n:,} lines")
    except Exception as e:
        print(f"  ⚠ {fname}: {e}")
PY
        fi

        # 7c. Sample rows from graphmert intermediate datasets
        "$PY" - "$GRAPHMERT_DIR" "$REPO_ROOT" <<'PY' 2>&1 || true
import os, sys
try:
    from datasets import load_from_disk
except ImportError:
    print("  ⚠ datasets unavailable — skipping graphmert dataset samples")
    sys.exit(0)
gm_dir, repo_root = sys.argv[1], sys.argv[2]
candidates = [
    ("head_positions",                        ["text_token_ids", "head_positions"]),
    ("llm_relations/relations_all",           ["chunk_id", "relations"]),
    ("llm_relations/relations_cleaned_train", ["chunk_id", "relation_id", "head", "tail"]),
    ("llm_relations/relations_cleaned_eval",  ["chunk_id", "relation_id", "head", "tail"]),
]
for sub, hint_cols in candidates:
    d = os.path.join(gm_dir, sub)
    if not os.path.isdir(d):
        continue
    rel = d[len(repo_root) + 1:] if d.startswith(repo_root) else d
    print(f"  {rel}:")
    try:
        ds = load_from_disk(d)
        print(f"    rows   : {len(ds):,}")
        print(f"    columns: {ds.column_names}")
        if len(ds) > 0:
            row0 = ds[0]
            # Trim long values for display
            shown = {}
            for k, v in row0.items():
                if isinstance(v, list):
                    shown[k] = f"[len={len(v)}, head={v[:5]}…]" if len(v) > 5 else v
                elif isinstance(v, str) and len(v) > 100:
                    shown[k] = v[:100] + "…"
                else:
                    shown[k] = v
            print(f"    row[0] : {shown}")
        else:
            print(f"    ⚠ 0 rows — downstream Dataset.from_list will Keys-mismatch")
    except Exception as e:
        print(f"    ⚠ load_from_disk error: {e}")
PY

        # 7d. Step-log grep for failure signals
        GM_LOG_DIR="$LOG_DIR/graphmert"
        note "step logs:"
        if [[ ! -d "$GM_LOG_DIR" ]]; then
            note "  ⚠ no graphmert step logs at ${GM_LOG_DIR#$REPO_ROOT/}"
        else
            gm_logs=$(find "$GM_LOG_DIR" -name '*.log' -type f 2>/dev/null | sort)
            if [[ -z "$gm_logs" ]]; then
                note "  (no .log files)"
            else
                set +o pipefail
                for lg in $gm_logs; do
                    sz=$(stat -c%s "$lg" 2>/dev/null || stat -f%z "$lg")
                    note "  ${lg#$REPO_ROOT/}  ($(echo "$sz" | human_size))"
                    # Grep for known failure modes
                    sig=$(grep -hiE '0 relations|no heads|empty dataset|OOM|CUDA out of memory|Keys mismatch|skip' "$lg" 2>/dev/null | tail -3)
                    if [[ -n "$sig" ]]; then
                        note "    failure signal:"
                        echo "$sig" | sed 's/^/      /'
                    fi
                done
                set -o pipefail
            fi
        fi

        # 7e. Partial MLM checkpoint state
        CKPT_DIR="$GRAPHMERT_DIR/checkpoints"
        if [[ -d "$CKPT_DIR" ]]; then
            note "checkpoints:"
            for c in "$CKPT_DIR"/*; do
                [[ -d "$c" ]] || continue
                cb=$(du -sb "$c" 2>/dev/null | awk '{print $1}')
                cf=$(ls "$c" 2>/dev/null | wc -l)
                note "  ${c#$REPO_ROOT/}  ($cf files, $(echo "${cb:-0}" | human_size))"
            done
        fi
    fi
fi

# --- 8. curriculum.generate_qa progress (phase=curriculum step=generate_qa) -
# Live status of the curriculum.generate_qa step: process state, in-flight
# vs saved question count, HTTP success/failure tallies, retry behavior.
# Useful both during an in-flight run (`watch -n 10 ./scripts/diagnose.sh
# --phase curriculum`) and post-run (no PID, but log + checkpoint counts
# stay readable).
if should_run curriculum generate_qa; then
section "8. curriculum.generate_qa progress"

CURRICULUM_LOG="$LOG_DIR/curriculum/generate_qa.log"
# Phase output dir convention varies between layouts (see RUN_ID resolution):
#   (A) outputs/<RUN_ID>/curriculum/curriculum.json   (ad-hoc / smoke)
#   (B) outputs/curriculum/curriculum.json            (no run subdir)
CURRICULUM_CHKPT="$OUTPUT_BASE/$RUN_ID/curriculum/curriculum.json"
[[ -f "$CURRICULUM_CHKPT" ]] || CURRICULUM_CHKPT="$OUTPUT_BASE/curriculum/curriculum.json"

# Resolve target N_QUESTIONS from active profile (parsed from RUN_ID, format
# YYYYMMDD-HHMMSS-<profile>-<sha>). Falls back to '?' if profile YAML or
# curriculum.num_questions key is missing.
CURRICULUM_PROFILE="${SI_PROFILE:-$(echo "$RUN_ID" | awk -F- '{print $3}')}"
CURRICULUM_TARGET="?"
CURRICULUM_PROFILE_YAML="$REPO_ROOT/configs/profiles/${CURRICULUM_PROFILE}.yaml"
if [[ -f "$CURRICULUM_PROFILE_YAML" ]]; then
    extracted=$(awk '
        /^curriculum:/ {in_s=1; next}
        in_s && /^[A-Za-z]/ {in_s=0}
        in_s && /^[[:space:]]*num_questions:/ {gsub(/[^0-9]/, ""); print; exit}
    ' "$CURRICULUM_PROFILE_YAML")
    [[ -n "$extracted" ]] && CURRICULUM_TARGET="$extracted"
fi

# Safe count helper — avoids the `grep -c | echo 0` double-output bug.
# awk index() is exit-code-safe and always prints the count.
_count_in_log() {
    local pattern="$1"
    [[ -f "$CURRICULUM_LOG" ]] || { echo 0; return; }
    awk -v pat="$pattern" 'index($0, pat) {n++} END {print n+0}' "$CURRICULUM_LOG"
}

# Process detection
CURRICULUM_PID=$(pgrep -f "generate_curriculum.py" | head -1 || true)
if [[ -n "$CURRICULUM_PID" ]]; then
    elapsed=$(ps -o etime= -p "$CURRICULUM_PID" 2>/dev/null | xargs)
    note "process:          running (PID $CURRICULUM_PID, elapsed $elapsed)"
else
    note "process:          NOT running"
fi
note "profile:          $CURRICULUM_PROFILE (target: $CURRICULUM_TARGET questions)"

# Checkpoint state + staleness detection
saved=0
stale_note=""
if [[ -f "$CURRICULUM_CHKPT" ]]; then
    if command -v jq >/dev/null 2>&1; then
        saved=$(jq length "$CURRICULUM_CHKPT" 2>/dev/null || echo 0)
    else
        saved=$(grep -c '^\s*"hop_count"' "$CURRICULUM_CHKPT" 2>/dev/null || echo 0)
    fi
    if [[ -n "$CURRICULUM_PID" ]]; then
        proc_start_epoch=$(ps -o lstart= -p "$CURRICULUM_PID" 2>/dev/null \
            | xargs -I {} date -d "{}" +%s 2>/dev/null || echo 0)
        chkpt_mtime_epoch=$(stat -c %Y "$CURRICULUM_CHKPT" 2>/dev/null || echo 0)
        if [[ "$chkpt_mtime_epoch" -gt 0 && "$proc_start_epoch" -gt 0 \
              && "$chkpt_mtime_epoch" -lt "$proc_start_epoch" ]]; then
            stale_note=" (STALE: from prior run, current hasn't checkpointed yet)"
        fi
    fi
fi

# HTTP / retry tallies
http_200=$(_count_in_log "HTTP/1.1 200")
http_503=$(_count_in_log "HTTP/1.1 503")
http_500=$(_count_in_log "HTTP/1.1 500")
http_504=$(_count_in_log "HTTP/1.1 504")
http_429=$(_count_in_log "HTTP/1.1 429")
retries=$(_count_in_log "Transient error")
gen_lines=$(_count_in_log "Generated ")

# Each accepted question costs ~6 successful 200s (the 6-step LLM pipeline).
in_flight_estimate=$(( http_200 / 6 ))

note "questions saved:  $saved / $CURRICULUM_TARGET$stale_note"
note "in-flight est:    ~$in_flight_estimate questions (HTTP 200s / 6 calls per Q)"

# Rate + ETA — based on SAVED questions vs. process elapsed time. Only useful
# when there's progress to extrapolate from. Skip if process isn't running,
# or saved count is from a stale checkpoint (prior run's data).
if [[ -n "$CURRICULUM_PID" && "$saved" -gt 0 && -z "$stale_note" ]]; then
    # `ps -o etimes=` returns elapsed seconds (no formatting). Some BusyBox
    # ps lacks etimes; fall back to parsing the etime "HH:MM:SS" format.
    elapsed_sec=$(ps -o etimes= -p "$CURRICULUM_PID" 2>/dev/null | xargs || true)
    if [[ -z "$elapsed_sec" || ! "$elapsed_sec" =~ ^[0-9]+$ ]]; then
        # Fallback: parse the formatted etime (dd-HH:MM:SS or HH:MM:SS or MM:SS)
        elapsed_sec=$(ps -o etime= -p "$CURRICULUM_PID" 2>/dev/null | xargs | \
            awk -F'[-:]' '{ n=NF; s=$n; if(n>=2)s+=$(n-1)*60; if(n>=3)s+=$(n-2)*3600;
                            if(n>=4)s+=$(n-3)*86400; print s }' || echo 0)
    fi
    if [[ "$elapsed_sec" -gt 0 ]]; then
        # Rate (q/min) and seconds-per-question, computed in awk to avoid bash float math.
        rate_per_min=$(awk -v s="$saved" -v t="$elapsed_sec" 'BEGIN{printf "%.2f", s / (t/60)}')
        sec_per_q=$(awk -v s="$saved" -v t="$elapsed_sec" 'BEGIN{printf "%.0f", t/s}')
        note "rate:             ${rate_per_min} q/min  (~${sec_per_q}s per accepted question)"
        # ETA — only meaningful when we know the target. Format shows the
        # math explicitly so the operator can sanity-check the projection:
        #   "time left: ~55m  (25 q × ~132s/q)"
        # means 25 questions remain at ~132s each → ~55 minutes total.
        if [[ "$CURRICULUM_TARGET" =~ ^[0-9]+$ ]]; then
            remaining=$((CURRICULUM_TARGET - saved))
            if [[ "$remaining" -gt 0 ]]; then
                eta_sec=$((remaining * sec_per_q))
                eta_hr=$((eta_sec / 3600))
                eta_min=$(( (eta_sec % 3600) / 60 ))
                if [[ "$eta_hr" -gt 0 ]]; then
                    note "time left:        ~${eta_hr}h ${eta_min}m  (${remaining} questions remaining × ~${sec_per_q}s/q)"
                else
                    note "time left:        ~${eta_min}m  (${remaining} questions remaining × ~${sec_per_q}s/q)"
                fi
            else
                note "time left:        target reached"
            fi
        fi
    fi
fi

note "checkpoint hits:  $gen_lines (one log line per _CHECKPOINT_EVERY successful Qs)"
note "HTTP 200 OK:      $http_200"
note "HTTP 503 Unavail: $http_503"
note "HTTP 500/504:     $((http_500 + http_504))"
note "HTTP 429 Rate:    $http_429"
note "transient retry: $retries"
if [[ -f "$CURRICULUM_LOG" ]]; then
    note "last log line:   $(tail -1 "$CURRICULUM_LOG" 2>/dev/null)"
fi

# Heuristic warnings — surface to verdict via mark_warn / mark_fail.
total_http=$((http_200 + http_503 + http_500 + http_504 + http_429))
if [[ "$total_http" -gt 20 ]]; then
    fail_pct=$(( (http_503 + http_500 + http_504) * 100 / total_http ))
    if [[ "$fail_pct" -gt 50 ]]; then
        mark_warn "Gemini 5xx rate ${fail_pct}% (${http_503} 503s + $((http_500 + http_504)) 5xx of ${total_http} calls) — upstream outage or our retry config too tight"
    fi
    # Retry inactivity check: if we have 5xx errors but zero retries logged,
    # the running process is using the pre-fix code path (only retried 429).
    if [[ "$http_503" -gt 0 && "$retries" -eq 0 ]]; then
        mark_warn "${http_503} 503 errors with 0 transient-retry log lines — running process predates the 503-retry fix (restart to apply)"
    fi
fi
if [[ -n "$CURRICULUM_PID" && -n "$stale_note" ]]; then
    mark_warn "curriculum.json count is from a prior run; current run hasn't reached its first _CHECKPOINT_EVERY boundary yet"
fi
fi

# --- 9. Verdict -------------------------------------------------------------
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
# KG_FINAL is only set when §2 (extract output) ran. With --phase graphmert
# or other narrow scopes, §2 is skipped, leaving KG_FINAL unbound. Guard so
# this block silently no-ops when §2 didn't define KG_FINAL.
if [[ -n "${KG_FINAL:-}" && -f "$KG_FINAL" ]]; then
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
        if [[ "$DEEP" -eq 0 ]]; then
            echo
            echo "For deeper triage (graphrag config, LLM cache, parquet samples):"
            echo "       ./scripts/diagnose.sh --phase extract --deep"
        fi
        echo
        echo "If --deep shows 0 relationships in entities.parquet vs 0 in"
        echo "relationships.parquet, isolate prompt vs parser vs model by"
        echo "replaying ONE chunk against the live vLLM endpoint:"
        echo "       ./scripts/diagnose_llm_extraction.sh"
        echo "       ./scripts/diagnose_llm_extraction.sh --file corpus/<domain>/source_txt/<file>"
    fi
fi

echo
exit "$EXIT"
