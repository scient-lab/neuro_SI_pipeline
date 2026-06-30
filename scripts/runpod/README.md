# RunPod runs & observability

How runs are produced on RunPod, where each run's data lands (on the pod **and**
in S3), and the four lenses for inspecting a run — **status / diagnosis /
analysis / health**. The last section is a **data contract** for anyone building
a UI/dashboard over the run fleet (e.g. a Lovable prototype): what to read, where
it lives, and whether it's a plain file read or needs a compute step.

> This doc is the **observability + UI-data** companion to
> [`../README.md`](../README.md). That file is the full operational reference
> (every flag, every phase's I/O, venvs, S3-sync internals). Read it for "how do
> I run X"; read this for "how do I *observe* a run and feed a dashboard".
>
> The four inspection tools (`stats.sh`, `diagnose.sh`, `analysis.sh`,
> `monitor.sh`) are **not RunPod-specific** — they work on any run, local or pod.
> They live here because the multi-run fleet a dashboard renders is the
> RunPod→S3 fleet.

---

## TL;DR — what a dashboard reads

| Lens | Question | Source artifact | S3-readable as-is? |
|---|---|---|---|
| **status** | running? how far? ETA? did each step pass? | `run_manifest.json` | ✅ plain JSON, no compute |
| **health** | CPU/GPU/VRAM/disk/net over time; throttling | `health/health_system.csv`, `health/health_gpu.csv` | ✅ plain CSV, no compute |
| **diagnosis** | *where* is it broken? exception file:line + I/O gaps | `diagnose.sh --json` (= `checks_view.py --lens health`) | ⚠️ needs compute (repo + venv) |
| **analysis** | is the output *good*? graded quality + sample | `analysis.sh --json` (= `checks_view.py --lens quality`) | ⚠️ needs compute (repo + venv) |

**Key fact for a remote UI:** the manifest already carries *both* per-step
`status` (did it run) **and** per-step `outcome` (did it produce good output —
`pass/warn/fail/skip`), written inline after every step. So **status + a
first-class quality verdict need zero compute** — just read the JSON. Only the
*deep per-check breakdown* (which file, which exception, which metric) needs the
Python tools run against a checked-out repo. See [§4](#4-data-contracts-for-a-uiapi).

---

## 1. Where the data comes from — the RunPod run lifecycle

Four steps; full detail in [`../README.md` Workflow 2](../README.md#workflow-2--run-on-runpod).

```
launch.sh (local)  ──POST──▶  RunPod API  ──spawns──▶  pod
   reads .env.runpod + configs/profiles/<profile>.yaml::runpod

bootstrap.sh (pod) ── clone repo, setup.sh venvs, write .env
                   └─ starts monitor.sh (detached) ── health/ telemetry + optional auto-kill

pipeline.sh (pod)  ── runs phases, writes outputs/ + run_manifest.json
                   └─ s3_sync.sh  ── pushes outputs/ ➜ s3://<bucket>/dss/runs/<run-id>/outputs/
                        · after each phase   · periodically (S3_SYNC_INTERVAL_SEC)   · at end
```

Net effect: while a run is live, a near-complete mirror of `outputs/` (manifest,
logs, health CSVs, partial artifacts) is continuously pushed to S3. A dashboard
never SSHes the pod — it reads S3.

---

## 2. One run's data layout (pod disk == S3 mirror)

`RUN_ID = <UTC-timestamp>-<profile>-<git-short-sha>` (e.g.
`20260630-102945-smoke-cc23b7f`). Sorts chronologically; profile + sha are
embedded for grep.

```
outputs/                                  s3://<bucket>/dss/runs/<run-id>/outputs/
├── run_manifest.json          ◀── STATUS + per-step OUTCOME (the spine)
├── _SUCCESS  | _FAILED        ◀── completion sentinel (exactly one, at end)
├── health/
│   ├── health_system.csv      ◀── HEALTH: one row per monitor tick
│   ├── health_gpu.csv         ◀── HEALTH: one row per (tick, gpu)
│   ├── monitor.log  monitor.out
├── logs/<run-id>/
│   ├── <phase>.log                       per-phase aggregate
│   └── <phase>/<step>.log                per-step (path recorded in manifest)
├── graphrag/output/kg_final.csv          extract artifact (seed KG)
├── graphmert/final_kg/expanded_kg.parquet
├── curriculum_verified/curriculum_verified.json
├── sft_checkpoints/ … rl_checkpoints/ …
```

`S3_URI` (in `.env.runpod`) sets the base, e.g. `s3://enlibra/dss/`. Excluded
from sync: `graphrag/cache`, `graphrag/input`, `__pycache__`, `*.pyc`. Logs and
health CSVs **are** synced.

**Enumerate the whole fleet** (what a dashboard's run-list reads):

```bash
aws s3 ls s3://<bucket>/dss/runs/          # one prefix per RUN_ID, newest sorts last
```

---

## 3. The four lenses (CLI)

One manifest, three+1 lenses. `stats` = did it *run*; `diagnose` = *where*
broken; `analysis` = is it *good*; `monitor` = the *health* telemetry feed.

| Tool | Lens | Answers | Reads |
|---|---|---|---|
| [`stats.sh`](../stats.sh) | status | running? how far? ETA? per-step OUTCOME | `run_manifest.json` + live `nvidia-smi`/`/proc` |
| [`diagnose.sh`](../diagnose.sh) | health/diagnosis | can I proceed? exception file:line + I/O gaps | manifest → failed step → its log + on-disk I/O contract |
| [`analysis.sh`](../analysis.sh) | quality | is the output good? graded + sample preview | each step's primary output artifact |
| [`monitor.sh`](../monitor.sh) | health feed | CPU/GPU/VRAM/disk/net over time; auto-kill | samples host + writes `health/*.csv` each tick |

### stats — live status grid
```bash
./scripts/stats.sh                       # phases-only summary (latest run)
./scripts/stats.sh --steps               # nested per-step rows
./scripts/stats.sh --steps --live --resources   # operator view: grid + CPU/GPU/disk gauges, ~5s refresh
./scripts/stats.sh --run 20260630        # a historical run (prefix match)
```
Columns: `PHASE · STATUS · OUTCOME · STARTED · FINISHED · DURATION · ETA ·
STEPS`. `STATUS` is read from the manifest; `OUTCOME` is the quality verdict
(also from the manifest). **This is the closest CLI analogue of the dashboard.**

### diagnose — *where* is it broken
```bash
./scripts/diagnose.sh                     # DEFAULT: standardized view, all phases
./scripts/diagnose.sh --phase graphmert --json    # machine-readable (see §4c)
./scripts/diagnose.sh --legacy --deep --phase extract   # old §-section deep dive
```
Two failure classes, one view: **thrown** (manifest → failed step → parse the
traceback → `ExceptionType: message` + `file:line`) and **silent** (exit 0 but
empty/missing output → caught by the I/O-contract structural checks). Ends with a
"inspect these files" footer.

### analysis — is the output *good*
```bash
./scripts/analysis.sh                     # DEFAULT: standardized quality view, all phases
./scripts/analysis.sh --phase extract --sample    # graded + a seed-KG row preview
./scripts/analysis.sh --json              # machine-readable (see §4c)
./scripts/analysis.sh --legacy --phase curriculum # old per-phase analyzer (richer metrics)
```
Grades the artifact (triple count, relation diversity, direction-error %, drop
rate, answer balance, training curves…). `0 triples` is a diagnose FAIL; `12
triples, 4% direction errors` is an analysis WARN. (Boundary rule per the
[standardization plan](../../docs/DIAGNOSE_ANALYSIS_STANDARDIZATION_PLAN_2026-06-29.md).)

> `diagnose.sh` and `analysis.sh` now share one engine,
> [`lib/checks_view.py`](../lib/checks_view.py) (`--lens health` vs
> `--lens quality`). `--legacy` falls back to the older per-phase
> implementations while coverage finishes porting (see the plan's coverage
> matrix — `validate`/`sft`/`rl` are still being filled in).

### monitor — the health telemetry feed
```bash
./scripts/monitor.sh                                    # log-only (never kills)
./scripts/monitor.sh --kill-on-fail --max-runtime 8h --idle-min 15   # unattended/nightly
```
bootstrap starts it detached. Each tick (default 60s) it calls
[`lib/health_sample.py`](../lib/health_sample.py), appending the two
`health/*.csv` files (which ride the normal S3 sync). Health-CSV logging is
**always on**, independent of every kill knob. Kill toggles (`--kill-on-fail`,
`--kill-on-complete`, `--max-runtime`, `--disk-crit`, `--idle-min`) are opt-in;
see the header of `monitor.sh` for the full flag/env matrix.

---

## 4. Data contracts for a UI/API

Everything a dashboard needs, with exact shapes. Versioned/authoritative schema
for the manifest: [`../../docs/run_manifest.schema.json`](../../docs/run_manifest.schema.json)
(+ the UI integration guide [`../../docs/run_manifest.md`](../../docs/run_manifest.md)).

### 4a. Run status + quality — `run_manifest.json` (file, no compute)

Two halves: `meta` (static catalog — the full phase/step shape, identical every
run; read once) and `run` (this run's live state). Per **step** record carries
both verdicts:

```jsonc
{
  "name": "extract_triples",
  "status": "completed",          // did it RUN: pending|running|completed|failed|skipped
  "outcome": "pass",              // did it PRODUCE good output: pass|warn|fail|skip|unknown|null
  "outcome_reason": "6,157 triples",
  "started_at": "2026-06-30T10:29:45+00:00",
  "finished_at": "2026-06-30T10:41:02+00:00",
  "exit_code": 0,
  "log_file": "outputs/logs/<run-id>/extract/extract_triples.log"
}
```

Run-level: `run.status`, `run.current_phase`, `run.selected_phases`,
`run.started_at/finished_at`, plus `run.failure` (failed phase/step/exit/message)
on a failed run. Rewritten atomically (flock + `os.replace`) at every transition,
so a mid-run reader never sees a half-written file. **`status` + `outcome` are
both already here — a status board with green/amber/red per step needs only this
file.**

### 4b. Health telemetry — `health/*.csv` (file, no compute)

`health_system.csv` — one row per monitor tick:
```
ts, phase, step, pipeline_status, failed_phase, pod_id, run_id, uptime_s,
cpu_pct, cores, load1, mem_used_gb, mem_total_gb, mem_pct,
disk_root_pct, disk_root_free_gb, disk_ws_pct, disk_ws_free_gb,
net_rx_mbps, net_tx_mbps, net_rx_mb, net_tx_mb,
gpu_count, gpu_util_avg, gpu_util_max, gpu_vram_pct_avg, gpu_vram_pct_max,
status, alerts
```
`health_gpu.csv` — one row per (tick, gpu):
```
gpu_index, ts, run_id, name, util_pct, vram_used_mb, vram_total_mb, vram_pct,
temp_c, power_w, throttle, top_proc_pid, top_proc_mem_mb
```
Append-only time series → trivial to plot. `throttle` is a human label
(`sw_thermal`, `hw_power_brake`, …); `alerts` carries any monitor warnings for
that tick.

### 4c. Diagnosis & analysis — `checks_view.py --json` (compute)

Not a static file — produced by running the tool against a checked-out repo +
venv (needs `pandas` for parquet introspection). Shape (both lenses):

```jsonc
{
  "lens": "health",                 // or "quality"
  "run_id": "20260630-102945-smoke-cc23b7f",
  "phases": [
    { "phase": "extract", "steps": [
      { "step": "extract_triples", "status": "completed", "verdict": "pass",
        // health lens: a LIST of checks, each naming its file + an "inspect" hint
        "checks": [ { "name": "out", "outcome": "pass",
                      "reason": "6,157 rows", "path": "graphrag/output/kg_final.csv",
                      "metrics": { "rows": 6157 } } ],
        "inspect": ["graphrag/output/kg_final.csv"]
        // quality lens instead: a single "check": {...} (+ "sample": [...] with --sample)
      }
    ]}
  ],
  "verdict": "pass",                 // worst across all steps
  "exit": 0                          // 0 clean · 1 any FAIL · 2 WARN-only
}
```

To surface diagnosis/analysis in a remote UI you have two options:
1. **Publish at run time** — have the pipeline/monitor run
   `checks_view.py --lens health --json` / `--lens quality --json` and write the
   output into `outputs/` so it syncs to S3 as a plain artifact (recommended for
   a pure-frontend dashboard; *not wired yet* — would be a small addition).
2. **Thin backend** — a small service with the repo checked out runs the tool
   on demand for the selected run.

(Status + health, §4a/§4b, need neither — they're already files in S3.)

### 4d. Completion sentinels

Exactly one of `_SUCCESS` / `_FAILED` is dropped at `outputs/` root when the run
ends (an EXIT trap guarantees `_FAILED` even on `set -e` abort or signal).
Cheapest possible "is it done, and did it pass" check:
```bash
aws s3 ls s3://<bucket>/dss/runs/<id>/outputs/_SUCCESS   # exists ⇒ done & ok
```
Neither present ⇒ still running (or the pod died without flushing — cross-check
the manifest's `finished_at` and the last `health_system.csv` `ts`).

---

## 5. Building the dashboard — recommended data flow

1. **Run list** — `aws s3 ls s3://<bucket>/dss/runs/`. Parse `RUN_ID` for
   timestamp / profile / sha. Sentinel presence ⇒ done/ok/failed; absence ⇒ live.
2. **Per-run header** — read `run_manifest.json`: `run.status`,
   `current_phase`, started/finished, `failure`.
3. **Per-step grid** — from the same manifest: each step's `status` (run state)
   × `outcome` (quality). This is the green/amber/red matrix; no compute.
4. **Health charts** — stream `health/health_system.csv` + `health_gpu.csv`;
   plot util/VRAM/disk/net over `ts`. Surface `throttle` / `alerts` as flags.
5. **Diagnosis / analysis detail** (drill-in) — either read a published
   `checks_view --json` artifact (§4c option 1) or call a thin backend (option 2).
   Use the manifest's per-step `log_file` to link straight to the failing log.

Two clocks to respect (so "stale" doesn't read as "broken"): the manifest's
**content** timestamps (how far the run got) vs the S3 object **LastModified**
(when the pod last synced). The `health_system.csv` last `ts` is the pod's
heartbeat — if it stopped advancing, the pod likely died.

---

## 6. See also

- [`../README.md`](../README.md) — full operational reference (flags, per-phase I/O, venvs, S3-sync internals, CloudWatch).
- [`../../docs/run_manifest.md`](../../docs/run_manifest.md) + [`run_manifest.schema.json`](../../docs/run_manifest.schema.json) — the manifest contract for consumers (authoritative shape).
- [`../../docs/DIAGNOSE_ANALYSIS_STANDARDIZATION_PLAN_2026-06-29.md`](../../docs/DIAGNOSE_ANALYSIS_STANDARDIZATION_PLAN_2026-06-29.md) — the three-lens design + coverage matrix.
- Tool headers (`-h`/`--help` on each) — `stats.sh`, `diagnose.sh`, `analysis.sh`, `monitor.sh`, `logs.sh` carry the canonical flag list.
