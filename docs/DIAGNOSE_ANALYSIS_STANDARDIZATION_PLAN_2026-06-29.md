# Diagnose / Analysis Standardization — Plan

**Date:** 2026-06-29 · **Status:** proposal (not yet implemented) · **Scope:** the
run-inspection tooling (`stats.sh`, `diagnose.sh`, `analysis.sh`) and the per-step
check engine (`step_quality.py`).

**Goal:** one standardized, all-phases view for **diagnosis** (is it broken / where
is the failure) and one for **analysis** (is the output good), built on a single
shared record + renderer, so both read identically across every phase and step —
and so diagnosis actually *pinpoints* the failing file / exception rather than
running a phase-specific checklist.

---

## 1. Problem — the tooling is a four-way reimplementation with half-coverage

Today the same "per-step OK/WARN/FAIL + reason + verdict" pattern is written **four
separate times**, in two languages, each with its own output format, and only **3 of
6 phases** are covered anywhere:

| Implementation | Lang | Concern | Phases covered | Per-step record |
|---|---|---|---|---|
| `scripts/diagnose.sh` (957 lines) | bash | health | extract, graphmert, curriculum | `mark_fail/warn/ok` + `FINDINGS[]` |
| `scripts/lib/analysis_extract.py` | python | quality | extract | own `Reporter` class |
| `scripts/lib/analysis_graphmert.py` | python | quality | graphmert | own `Reporter` class |
| `scripts/lib/analysis_curriculum.py` | python | quality | curriculum | own `Reporter` class |
| `scripts/lib/step_quality.py` | python | quality | all (generic) | **`V(outcome, reason, metrics)`** |

Consequences:
- **No shared scaffolding** — there is no `analysis_common`; each `analysis_*.py`
  re-rolls its OK/WARN/FAIL reporter, `--json`, `--quiet`, and the 0/1/2 exit code.
  `diagnose.sh` re-rolls the same in bash. Formats drift.
- **`step_quality.py` already overlaps both** — it emits `V(FAIL, "no kg_final.csv
  produced")` and `V(FAIL, "0 triples extracted", {...})` (health-shaped) *and* feeds
  the stats `OUTCOME` column (quality-shaped). It already has the cleanest schema —
  but nothing else reuses it.
- **Blurred boundaries** — `diagnose §8` (curriculum) computes rate / ETA / HTTP
  200/503 tallies / in-flight estimate. That is **live progress**, i.e. *stats*, not
  health.
- **Coverage gaps** — `validate`, `sft`, `rl` have **no** diagnose *or* analysis
  coverage at all.

### `stats.sh` — the third lens, and the one already doing it right

`scripts/stats.sh` + `scripts/lib/stats_render.py` is the **live status** tool: a
per-phase/step grid (`PHASE · STATUS · OUTCOME · STARTED · FINISHED · DURATION · ETA ·
STEPS`), with `--live` (re-render loop, ~5 s), `--steps` (one-shot), and `--resources`
(CPU/GPU/disk bars). Crucially, it is **not** a fourth reimplementation of the check
pattern — it already does it the right way:
- it **reads `run_manifest.json`** for `STATUS` (pending/running/completed/failed) +
  per-step timing and ETA, and
- it renders the **`OUTCOME` column directly from `step_quality.py`'s `V` verdicts**
  (wired 2026-06-29): `STATUS` = did it *run*; `OUTCOME` = is it *good*.

So stats is the existing proof that a single shared record works across tools, and it
sets two anchors for this plan:
- **stats already consumes the standardized record** (`step_quality`'s `V`). Diagnose
  and analysis should consume the *same* record the same way, instead of re-rolling
  their own reporters.
- **stats shares the manifest entry point** with diagnose's failure localizer (§4): the
  same `run_manifest.json` that gives stats `STATUS`/`ETA` gives diagnose the failed
  step + its `log_file`. **One source (the manifest), three lenses.**

What stats is missing (and should gain): the *progress detail* currently stranded in
`diagnose §8` (curriculum rate / ETA / HTTP 200-503 tallies / in-flight estimate) — that
is live status, and belongs in the stats lens, not the health lens (see §2 boundary
rule and §8 step 5).

---

## 2. The three lenses (the intended split)

Each lens answers one question and owns one verdict basis. Keep three entry points;
put them on one shared engine.

| Lens | Question | Verdict basis |
|---|---|---|
| **stats** (`stats.sh`) | *running? how far? ETA?* | live grid: manifest `STATUS` + timing/ETA + `step_quality` `OUTCOME` |
| **diagnose** (`diagnose.sh`) | *can I proceed? WHERE is it broken?* | structural integrity + failure localization |
| **analysis** (`analysis.sh`) | *is the output GOOD (given it's valid)?* | graded quality metrics + sample preview |

**Boundary rule (resolves the current overlaps):**
- *Missing / empty / malformed / crashed* → **diagnose FAIL** (you can't proceed).
- *Valid but weak* → **analysis WARN** (e.g. "12 triples, low diversity").
- *In-flight rate / ETA / HTTP tallies* → **stats** (move `diagnose §8` here).

Same input, different lens: `0 triples` is a diagnose FAIL; `12 triples, 4% direction
errors` is an analysis WARN.

---

## 3. The standardized unit — per-`(phase, step)` report

The standard unit is **not** "checks + verdict." It is a per-step **I/O contract +
checks-tied-to-files + a sample hook**, so diagnosis can localize the failing file and
analysis can preview the output.

```python
StepReport(phase, step):
  inputs   = [Path(...)]          # files it consumes (upstream outputs)  ─┐ I/O contract:
  outputs  = [Path(...)]          # files it produces                      ┘ localizes the break
  state(path) -> {exists, size, rows, cols}        # rendered per file
  health_checks()  -> [Check(name, V, path)]       # every check NAMES its file
  quality_checks() -> [Check(name, V, path)]
  sample(n) -> rows of the primary output          # analysis --sample preview
```

- **Record:** reuse `step_quality.py`'s `V(outcome, reason, metrics)` (outcomes:
  `pass / warn / fail / skip / unknown`). A `Check` is `(name, V, path)`.
- **Roll-up:** step verdict = worst check; run verdict = worst step → exit `0/1/2`.
- **Renderer:** one shared `checks_view.py` (human nested view + `--json`), the same
  pattern as the already-built `config_view.py`. Both diagnose and analysis render
  through it; they differ ONLY in which registry (`health_checks` vs `quality_checks`)
  they call.

The declared `inputs`/`outputs` across steps also form a per-run **data-lineage map**
(this step's output is that step's input) — which is what lets diagnosis say "the break
is upstream/downstream of here."

---

## 4. Diagnosis = a real debugger (two mechanisms, one view)

"Pinpoint any issue or exception" splits into two failure classes that need different
mechanisms. Diagnosis must do **both**.

### Class 1 — thrown exceptions (a step crashed). Localize generically from the log.
No per-exception check needed. The pipeline already provides the trail:
- `run_manifest.json` records each step's `status`, `exit_code`, and **`log_file`** →
  which step failed.
- `run_step` tees stdout+stderr to that per-step log → **the real traceback is in it**.

Diagnosis: manifest → failed step → **parse the last traceback** → surface
`ExceptionType: message` + `file:line` + the failing frame. Catches *any* crash.

```
DIAGNOSE  run <id>   → first failure: graphmert.preprocess

graphmert.preprocess                    [FAIL exit 1]      ← run_manifest.json
  exception  logs/<run>/graphmert/preprocess.log:412
    KeyError: 'id'
      2_graphmert/.../dataset_preprocessing_utils.py:88  in ground_triples_to_snippets
        row["id"]
  in   graphmert/llm_relations/relations_cleaned_train   ✓ 1,843 rows  ✗ no 'id' column (has chunk_id, head, tail)
  → ROOT CAUSE: consumer needs 'id'; producer clean_llm_relations.py didn't emit it.
    inspect: outputs/graphmert/llm_relations/relations_cleaned_train
             2_graphmert/.../clean_llm_relations.py
```

### Class 2 — silent failures (exit 0, wrong output). No traceback exists.
The dangerous class — the green-but-empty runs. The step *succeeded* (exit 0), so there
is **no exception to find**. Only a **structural check on the I/O contract** catches it.

```
DIAGNOSE  extract.extract_triples       [completed exit 0]   ← manifest says SUCCESS
  exception  none
  in   graphrag/output/documents.parquet       ✓ 1,204 rows
  out  graphrag/output/relationships.parquet   ✗ 0 rows
  out  graphrag/output/kg_final.csv            ✗ 0 triples
  → SILENT FAILURE: exited 0 but produced an empty KG. No traceback — only the
    I/O check catches this. inspect relationships.parquet + extract_triples.log
```

### Synthesis
| Failure | Mechanism | Catches |
|---|---|---|
| **thrown** | manifest → failed step → parse traceback from log | any exception (file:line + message) |
| **silent** | I/O contract + structural checks | green-but-empty / missing-column / 0-rows |

Diagnose card flow: **`status` + `exception(file:line)`** (manifest + log) first, then
I/O state + checks as supporting evidence, then **root-cause + "inspect these."** The
traceback parser is the **spine** (pinpoints any crash); the check registry is the
**safety net** (pinpoints the silent failures the traceback can't see).

---

## 5. Analysis = quality grades + sample preview

Analysis renders the `quality_checks()` + a per-step `sample()` (the "quick-preview
seed KG samples" need). `--sample [N]` dumps a few rows of the step's primary output.

```
ANALYSIS  extract.extract_triples --sample        [WARN]
  ✓ triple_count        6,157      (≥1000)
  ✓ relation_diversity  38/40 used
  ⚠ direction_errors    4.2%       (warn ≥3%; e.g. "helium fuses_into hydrogen")
  sample  kg_final.csv  (5 of 6,157):
    sun          | composed_of    | hydrogen
    hydrogen     | fuses_into     | helium
    earth        | orbits         | sun
    red giant    | collapses_into | white dwarf
    massive star | undergoes      | supernova
```

The sample hook already exists ad-hoc as `diagnose --deep`'s parquet sample (5 rows of
entities/relationships); this moves it into the standard view (preview = quality, not
health) and uniformizes it: curriculum samples Q&A, predict_tails samples head→tails,
sft/rl sample a metrics tail.

---

## 6. Target structure

```
scripts/lib/
  checks/__init__.py        # Check=(name,V,path); registries:
                            #   health_checks(phase, step)  -> [Check]
                            #   quality_checks(phase, step) -> [Check]
                            #   step_io(phase, step)        -> (inputs, outputs)
                            #   sample(phase, step, n)      -> rows
  checks/extract.py         # io + health_* + quality_* + sample for extract steps
  checks/validate.py        # NEW — fills the gap
  checks/graphmert.py
  checks/curriculum.py
  checks/sft.py             # NEW
  checks/rl.py              # NEW
  checks_view.py            # ONE renderer + verdict roll-up (nested + --json),
                            # plus the manifest→traceback failure localizer
scripts/diagnose.sh -> dispatch (failure-localizer + health_checks) through checks_view
scripts/analysis.sh -> dispatch (quality_checks + sample) through checks_view
scripts/stats.sh    -> ALREADY reads manifest (STATUS/timing/ETA) + renders step_quality OUTCOME;
                       gains the curriculum progress/ETA/HTTP detail moved out of diagnose §8
step_quality.py     -> becomes quality_checks + roll-up to the OUTCOME column (back-compat)
```

`step_quality.evaluate()` generalizes from "one `V` per step" to "list of checks,
verdict = worst" — the stats `OUTCOME` column keeps working, and analysis gains the
per-check breakdown for free.

---

## 7. Coverage matrix (current → target)

| Phase | step(s) | diagnose (health) | analysis (quality) |
|---|---|---|---|
| extract | chunk, extract_triples, normalize, cache | ✓ → port | ✓ → port |
| **validate** | seed_kg_consensus | ✗ **add** | ✗ **add** |
| graphmert | preprocess, train_mnm, predict_tails, validate_predictions | ✓ → port | ✓ → port |
| curriculum | generate_qa, validate_qa, assemble | ✓ (minus §8→stats) | ✓ → port |
| **sft** | train_lora | ✗ **add** | ✗ **add** |
| **rl** | train_grpo, merge_rl | ✗ **add** | ✗ **add** |

---

## 8. Build plan (incremental, reviewable)

1. **Shared spine.** `checks_view.py` (record + I/O-contract renderer + verdict +
   `--json`) and the **manifest→traceback failure localizer** (Class-1). This is the
   piece that turns diagnose from checklist into debugger.
2. **Port `extract`** to both lenses through the spine. Prove `diagnose.sh` and
   `analysis.sh` render identically, on a real empty-KG case (Class-2) and a forced
   crash (Class-1), plus `analysis --sample` seed-KG preview.
3. **Generalize `step_quality.evaluate()`** to return per-check lists; keep the OUTCOME
   roll-up back-compatible.
4. **Fill `validate` / `sft` / `rl`** (currently zero coverage).
5. **Move `diagnose §8`** (curriculum progress / ETA / HTTP) into `stats.sh`.
6. **Retire** the bash `§`-sections and the three `Reporter` classes once their checks
   are ported.

Each step is independently shippable; nothing is removed until its replacement renders.

---

## 9. Decisions / open questions

- **Three entry points or one?** Plan keeps `diagnose.sh` / `analysis.sh` / `stats.sh`
  as separate commands over one engine (matches current muscle memory). Could collapse
  to `inspect.sh --health|--quality|--status` later; not required.
- **Where do the I/O contracts live?** In `checks/<phase>.py` as declared knowledge of
  each step's reads/writes. (Alternative: have the pipeline *emit* the actual paths a
  step touched, like the config ledger emits config — more faithful but needs file-I/O
  instrumentation. Defer.)
- **Traceback parsing scope.** v1: last Python traceback in the failing step's log
  (type, message, file:line, frame tail). Non-Python failures (OOM-killed, segfault,
  vLLM CUDA assert) fall back to a log-tail + exit-code classification.
- **Relation to the config ledger** (`outputs/<run>/config/*.yaml`, built 2026-06-29):
  the diagnose card can cross-reference it — e.g. "this step ran model X / param Y
  (source: fallback)" — to catch config-caused failures (the profile-key-name-trap).

---

## References
- `scripts/diagnose.sh` — current health monolith (9 `§`-sections; extract/graphmert/curriculum).
- `scripts/analysis.sh` + `scripts/lib/analysis_{extract,graphmert,curriculum}.py` — current quality (3 `Reporter` classes).
- `scripts/lib/step_quality.py` — `V(outcome, reason, metrics)` engine + OUTCOME column (the schema to standardize on).
- `scripts/stats.sh` + `scripts/lib/stats_render.py` — the live status grid; already reads the manifest and renders `step_quality`'s `OUTCOME` (the existing shared-record consumer).
- `scripts/lib/manifest.py` / `run_step` (`scripts/lib/common.sh`) — per-step `status` / `exit_code` / `log_file` (the failure-localizer entry point, shared with stats).
- `scripts/lib/config_view.py` — already-built nested `phase → step → …` renderer pattern to mirror.
