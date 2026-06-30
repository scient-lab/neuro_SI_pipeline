# Run manifest — UI integration guide

`run_manifest.json` is the **single source of truth for pipeline run state** — to learn a
run's status, read this file (or its S3 mirror), not a DB, an API, or the logs. A UI renders
run/phase/step progress directly from it: a self-describing JSON document, updated live, no
server required. (That's source-of-truth for *state*; the *shape* is owned by the code that
writes it — see [Authority & validation](#authority--validation).)

- **Schema (derived contract):** [`run_manifest.schema.json`](run_manifest.schema.json) — JSON Schema (Draft 2020-12).
- **Produced by:** `scripts/lib/manifest.py` (stdlib-only; atomic, lock-guarded writes; authoritative for the shape).
- **Live sample:** `outputs/run_manifest.json` (latest run) and `outputs/<RUN_ID>/run_manifest.json`.

## Where it lives

| Location | Path |
|---|---|
| Local / on the pod | `outputs/run_manifest.json` (latest) and `outputs/<RUN_ID>/run_manifest.json` (per run) |
| S3 mirror (when `S3_URI` set) | `s3://<S3_URI>/runs/<RUN_ID>/outputs/run_manifest.json` |
| Success/fail sentinels | `_SUCCESS` xor `_FAILED` dropped next to the manifest (Hadoop convention) |

## Shape in one breath

```
{ schema_version, meta, run }
```
- **`meta`** = the STATIC catalog: every phase and step that *can* run, with descriptions.
  Identical across runs of the same code. Use it to lay out the full pipeline skeleton.
- **`run`** = the LIVE state: which phases/steps actually ran this time, with timestamps,
  exit codes, and quality outcomes. Use it to fill in progress against the skeleton.

## The one concept that trips people up: `status` ≠ `outcome`

Each step carries two independent signals:

| Field | Question it answers | Values |
|---|---|---|
| `status` | Did it **run**? | `pending` · `running` · `completed` · `failed` · `skipped` |
| `outcome` | Did it **produce meaningful output**? | `pass` · `warn` · `fail` · `skip` · `unknown` · `null` |

A step can be `status: completed` but `outcome: fail` — it ran to exit 0 but produced
empty/garbage output (e.g. 0 triples extracted). Render both. `outcome` is `null` until a
quality probe runs, and may be **absent** in older manifests — treat it as optional.

## Rendering hints

- **Top-line progress:** `run.progress` (0..1, weighted) and `run.estimated_completion_at`
  (absolute RFC3339 — compute `remaining = est - now`; it's a snapshot, so recompute against
  `now` for a smooth countdown). Both are rough; don't present to the second.
- **Per-phase / per-step:** join `run.phases[].steps[]` (live) against `meta.phases[].steps[]`
  (catalog) by `name` to show not-yet-started steps too.
- **Failure banner:** if `run.failure` is present, it's a pre-walked summary of the first failed
  `phase`(+`step`) with an `error.log_tail`. No need to traverse the tree yourself.
- **Logs:** `step.log_file` is a repo-relative path; `step.cw_log_stream` is the CloudWatch
  stream name when shipping is on.
- **Timestamps:** all RFC3339 with tz offset, or `null` when unset.

## Optional / conditional fields (handle defensively)

These are not always present — code against them as optional:
`run.resumed_at` (resumed runs only), `run.runpod_pod_id`, `step.outcome`, `step.outcome_reason`,
`run.failure`, and `phase.error` / `step.error` (failed nodes only).

## Authority & validation

There are two "sources of truth" here, on different axes — worth understanding so you trust
the right thing:

- **Run state** (what's a run's status?) → **`run_manifest.json`** is authoritative.
  `manifest.py` is its *sole writer*; everything else (`logs.sh`, `stats.sh`, the monitor, the
  S3 mirror, the `_SUCCESS`/`_FAILED` sentinels, this UI) only *reads* it. No competing store.
- **Shape** (what fields/types exist?) → **`manifest.py` (the code)** is authoritative. This
  schema is a *derived description* of what that code emits — not the other way round.
  `manifest.py` does **not** validate against the schema at runtime: it's stdlib-only and on a
  hot, lock-guarded path, and a status write must never fail. Drift is caught instead at
  dev/CI time by [`scripts/lib/test_manifest_schema.py`](../scripts/lib/test_manifest_schema.py),
  which drives `manifest.py` through real lifecycles and validates the output against this schema.

What that means for you: the schema is a faithful, **test-enforced** contract you can codegen
and validate against — but because it's derived, **validate on your side**. The model is
*producers test, consumers validate*: the pipeline guarantees shape via the drift test; you,
as the (untrusted) consumer, validate each manifest you ingest. The snippets below do exactly
that.

## Generate TypeScript types from the schema

```bash
# one-liner, no install
npx json-schema-to-typescript docs/run_manifest.schema.json -o src/runManifest.d.ts
# or, for a richer model (handles nullable unions nicely):
npx quicktype docs/run_manifest.schema.json -o src/runManifest.ts --lang ts --src-lang schema
```

## Validate a manifest against the schema

```bash
# Python (any env with the `jsonschema` package)
python - <<'PY'
import json; from jsonschema import Draft202012Validator
s=json.load(open("docs/run_manifest.schema.json")); inst=json.load(open("outputs/run_manifest.json"))
errs=list(Draft202012Validator(s).iter_errors(inst))
print("OK" if not errs else [ (list(e.path), e.message) for e in errs ])
PY

# JS
npx ajv-cli validate -s docs/run_manifest.schema.json -d outputs/run_manifest.json --spec=draft2020
```

## Polling safety

`manifest.py` writes via `flock` + temp-file `os.replace` (atomic rename), so a reader
**never sees a half-written file** — polling `run_manifest.json` on an interval is safe. For a
"is it done?" check without parsing, test for the `_SUCCESS` / `_FAILED` sentinel files.
