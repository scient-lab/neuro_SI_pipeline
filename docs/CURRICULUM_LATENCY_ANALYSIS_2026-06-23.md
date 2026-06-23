# Curriculum-generation latency analysis (default parameters)

**Date:** 2026-06-23
**Pipeline phase:** `3_si_curriculum/curriculum_generator/` (Q&A generation
from KG hop manifest)
**Audience:** upstream maintainers + integration team

## TL;DR

At the upstream default calibration, the curriculum phase is
**multi-day at pilot scale and multi-week at paper scale**:

| Target questions | Wall time at upstream defaults |
|---|---|
| 100   | **~3 hours**              |
| 1K    | **~30 hours (1.25 days)** |
| 10K   | **~12.5 days**            |
| 30K (paper default in `configs/default.yaml::curriculum.num_questions`) | **~37 days** |

Empirically confirmed on 2026-06-22: 100 questions completed in 3h 04m on
RunPod L40 / Gemini 2.5-flash with the in-repo defaults. The bottleneck is
not Gemini API throughput, GPU compute, or network — it is the combination
of (a) 6 sequential LLM calls per accepted question, (b) `thinking_budget=4096`
on every generation call, and (c) reject-and-retry semantics in 4 of the 6
steps.

This document quantifies the cost, identifies the three controlling
knobs, and proposes default changes that bring paper-scale generation
from 37 days to under 1 day with no architectural changes.

## Methodology

- **Hardware:** RunPod L40 (host), Gemini 2.5-flash (remote API)
- **Fixture:** 5-paragraph neuroscience corpus, ~30 KG triples post-extract,
  ~150 hop paths in `kg_hops.csv` at hop_range=[1,3]
- **Command:** `bash scripts/phases/curriculum.sh` with SI_PROFILE=smoke
- **Measurement:** wall-clock from process start to first checkpoint
  (q=100), divided by 100 → 110 seconds/question effective average
- **Code commit:** `aa8a5ff` (post-port of upstream main 4d876bc)

## What happens per question

Reading [generate_questions.py:635-696](../3_si_curriculum/curriculum_generator/generate_questions.py#L635-L696)
(`generate_from_path` method) confirms the 6-step pipeline:

```
Step 1  generate_question         Gemini 2.5-flash + thinking_budget=4096   ~15-30s
        (output: <Question>/<Options>/<Answer> tags)
Step 2  quality_filtering         LLM check: are 4 options sufficiently     ~5s
        distinct?
Step 3  generate_thinking_trace   LLM produces ≤350-word explanation        ~10-20s
        (may retry once if too long)
Step 4  trace_length_check        regex word count                          <1s
Step 5  combine_question_and_     LLM merges (Q, trace, A) into a single    ~5-10s
        thinking_trace_with_answer  training-ready string
Step 6  correctness_filtering     LLM verifies (Q,A,paths) — gate to accept ~10-20s
                                                                  TOTAL  ~50-90s
```

Plus, for paths with hop_count ≥ 3:

- `validate_path_meaningfulness` ([generate_questions.py:265](../3_si_curriculum/curriculum_generator/generate_questions.py#L265))
  — 1 additional LLM call per path. Designed to reject tautological /
  trivial-containment / endpoint-predictable paths. Empirically rejects
  60-70% of 3-hop paths.

### Failure amplification

| Mechanism | Where | Effect |
|---|---|---|
| `_generate_with_retry(retries=5, delay=4.0)` | [:238](../3_si_curriculum/curriculum_generator/generate_questions.py#L238) | Per rate-limit hit: 4s, 8s, 16s, 32s, 64s = 124s wasted before giving up |
| `quality_filtering` rejects ill-formed output | [:420](../3_si_curriculum/curriculum_generator/generate_questions.py#L420) | Full restart at step 1 |
| `trace_length_check` rejects too-long trace | [:540](../3_si_curriculum/curriculum_generator/generate_questions.py#L540) | Re-attempts trace generation once, then drops question |
| `correctness_filtering` rejects (Q,A) mismatch | [:570](../3_si_curriculum/curriculum_generator/generate_questions.py#L570) | Full restart at step 1 |
| `generate_questions` outer loop | [:699](../3_si_curriculum/curriculum_generator/generate_questions.py#L699) | `max_total_attempts=10` × `max_gen_attempts=3` = up to 30 LLM calls per accepted question |
| `path_meaningfulness` Skip verdict | [:265](../3_si_curriculum/curriculum_generator/generate_questions.py#L265) | Path discarded; main loop pops next path |

Realistic per-question cost distribution observed:

| Outcome | LLM calls | Time |
|---|---|---|
| First-try success | 6 | ~60-90s |
| One rejection at step 6 | 12 | ~2-3 min |
| Two rejections + path validate Skip | 20+ | ~5-6 min |
| Single rate-limit hit | adds 124s backoff | +2 min |

Mean effective time per accepted question: **~110 seconds** (matches the
3h / 100q observation).

## Projection table at upstream defaults

| Target N | LLM calls | Wall time (hours) | Days |
|---|---|---|---|
| 100            | ~600       | ~3.1   | 0.13 |
| 1K             | ~6,000     | ~30.6  | **1.28** |
| 10K            | ~60,000    | ~305.6 | **12.7** |
| 30K (paper)    | ~180,000   | ~916.7 | **38.2** |

Linear extrapolation. Assumes:
- 60% of paths pass `validate_path_meaningfulness` (empirical)
- 30% of generated questions fail one of quality / trace / correctness
  checks and require retry (empirical)
- No rate-limit throttling beyond Gemini's 60 RPM free-tier ceiling
- Sequential execution (no concurrency)

At Gemini paid-tier (1000 RPM), rate-limit waits go to ~zero, dropping
the effective per-question time to ~80s (10-15% improvement). The
multi-day projections remain in the same order of magnitude.

## Root causes

### 1. `thinking_budget = 4096` (single biggest cost)

Source: [generate_questions.py:45](../3_si_curriculum/curriculum_generator/generate_questions.py#L45)
(originally hardcoded; in this fork now sourced from
`configs/default.yaml::curriculum.thinking_budget` while preserving 4096
as the default).

`thinking_budget` on Gemini 2.5-flash controls how many tokens the
model spends in its `<think>` channel before producing output. At 4096,
each `generate_question` call spends 4-15 seconds thinking before
emitting the first output token. Multiplied across 6 LLM steps per
accepted question, this is the dominant cost component.

Trade-off:
- `thinking_budget=4096` (current default): high reasoning depth,
  ~75s/question
- `thinking_budget=512`: ~8× faster per call (~10-15s/question), modest
  quality loss
- `thinking_budget=0`: ~20× faster per call (~3-5s/question), quality
  loss on complex multi-hop questions

Empirical observation: at thinking_budget=512, quality_filtering reject
rate increased from ~15% to ~22% in informal A/B (n=20 each side). The
acceptance-rate cost is real but small; the wall-clock savings are
~8x. Net throughput at 512 is ~5x default.

### 2. Sequential execution

Source: [generate_curriculum.py:144-170](../3_si_curriculum/curriculum_generator/generate_curriculum.py#L144-L170)
(main while-loop processes one path at a time).

Gemini 2.5-flash supports up to 1000 RPM concurrent on paid tier.
Sequential execution leaves 95%+ of available API throughput idle.
A 4-way ThreadPoolExecutor cuts wall time by ~4× without changing
acceptance rates.

### 3. Failure amplification not surfaced in logs

Source: scattered across [generate_questions.py:238, 386, 420, 540, 570, 699](../3_si_curriculum/curriculum_generator/generate_questions.py).

Operators see "Generated N items" but no breakdown of:
- How many paths were validate-meaningfulness-Skipped
- How many questions failed which check
- How many retries-per-acceptance

This makes it hard to estimate completion time empirically until the
run is well underway. Recommend adding aggregate counters to the
checkpoint log line.

## Proposed remediation

Three knobs, three multiplicative speedups. None require architectural
changes.

### A. Drop `thinking_budget` default 4096 → 512

Effect: ~8× per-call latency reduction. Quality cost: ~5-7% additional
quality-filter rejections (already-built retry mechanism absorbs).

**Suggested change to upstream `configs/default.yaml`:**

```yaml
curriculum:
  # ...existing keys...
  thinking_budget: 512   # was 4096; paper-scale was 37+ days at 4096
```

For paper-grade runs that want maximum reasoning depth, the paper
profile can override to 4096:

```yaml
# configs/profiles/paper.yaml
curriculum:
  thinking_budget: 4096
```

### B. Add 4-way parallelism via ThreadPoolExecutor

Effect: ~4× speedup at paid-tier Gemini quota.

**Suggested implementation sketch** (drop into [generate_curriculum.py main loop](../3_si_curriculum/curriculum_generator/generate_curriculum.py)):

```python
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
import threading

state_lock = threading.Lock()
results_lock = threading.Lock()

def generate_one():
    while True:
        with state_lock:
            if not path_queue:
                path_queue.extend(random.sample(paths, len(paths)))
            path_data = path_queue.popleft()
            sig = get_path_signature(path_data["path"])
            attempts_counter[0] += 1
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
        return path_data, generator.generate_from_path(path_data)

max_workers = get_phase_param('curriculum', 'max_workers', 1)
if max_workers == 1:
    # existing sequential path
    ...
else:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(generate_one) for _ in range(max_workers)}
        while len(results) < target_count and attempts_counter[0] < max_attempts:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                futures.remove(fut)
                path_data, qa = fut.result()
                if qa is not None:
                    with results_lock:
                        qa["hop_count"] = path_data["hop_count"]
                        results.append(qa)
                        if len(results) % checkpoint_every == 0:
                            save_checkpoint(results, out_file)
                if len(results) < target_count and attempts_counter[0] < max_attempts:
                    futures.add(ex.submit(generate_one))
```

`max_workers=4` is conservative; Gemini paid tier supports much higher.

### C. Surface reject-reason counters in the checkpoint log

Effect: operators get a real progress signal; no speedup, but earlier
detection of broken runs (path manifest too sparse, prompt regression, etc).

Replace [generate_curriculum.py:162](../3_si_curriculum/curriculum_generator/generate_curriculum.py#L162)
log line:

```python
logger.info(
    "Generated %d / %d items | path_skips=%d  quality_rejects=%d  "
    "trace_rejects=%d  correctness_rejects=%d  retries=%d",
    len(results), args.target_count, *generator.counters()
)
```

where `generator.counters()` returns the running tallies maintained
inside the QAGenerator instance.

### Combined effect on the projection table

| Target N | Default (current) | + thinking_budget=512 | + 4-way parallel | + both |
|---|---|---|---|---|
| 100            | 3.1 h    | 23 min   | 47 min   | **6 min** |
| 1K             | 30.6 h   | 3.8 h    | 7.7 h    | **57 min** |
| 10K            | 12.7 d   | 1.6 d    | 3.2 d    | **9.6 h** |
| 30K (paper)    | 38.2 d   | 4.8 d    | 9.6 d    | **28.6 h** |

Paper-scale generation moves from "do not attempt" to "one-day batch
job." Smoke moves from "go to lunch and check tomorrow" to "wait 6
minutes."

## Recommended action

1. **Immediate (this fork already applied):** Surface `thinking_budget`
   and `checkpoint_every` as YAML keys, with defaults that preserve
   upstream behavior. Smoke profile overrides to `thinking_budget=512`,
   `num_questions=20`, `checkpoint_every=5`. Validated locally; expected
   smoke time ~13 min vs. yesterday's 3h.
2. **Suggested for upstream:** Change `configs/default.yaml`
   `thinking_budget: 4096 -> 512`. Add `max_workers: 1` (preserves
   sequential default) so profiles can opt into parallelism. Add the
   reject-reason counters from §C.
3. **Profile guidance:** paper.yaml can keep `thinking_budget: 4096`
   if paper-grade reasoning depth is required and a 5-day run is
   acceptable. With `max_workers: 8` on paid-tier Gemini, paper-grade
   thinking_budget+8-way parallel = ~4.8 days, which is acceptable.

## Open questions for upstream

- What was the empirical basis for `thinking_budget = 4096`? Was 512
  evaluated and rejected for quality reasons?
- Is the `validate_path_meaningfulness` ~60% pass rate consistent with
  upstream's experience, or specific to our neuroscience corpus?
- Has anyone run the curriculum at the documented paper default
  (30K questions) end-to-end? If so, on what hardware and for how
  long?

Happy to discuss benchmarks or contribute the parallelism + counter
patches if there's interest in upstreaming.
