#!/usr/bin/env python3
"""Quality analysis for the curriculum phase outputs.

§1 generation        generated count, schema sanity
§2 verification      drop rate via two-LLM consensus, paper §4.2 baseline check
§3 hops              hop_count distribution vs configured hop_range; detect
                     calculate_hops "fallback to seed_kg with hop_distance=1"
§4 answers           answer-letter (A/B/C/D) balance; positional bias detection
§5 trace             thinking-trace word-count distribution; TRACE_MAX_WORDS adherence
§6 diversity         unique source/target concepts; repeated-concept ratio
§sample              5 random verified questions with metadata

Reads OUTPUT_BASE/curriculum/curriculum.json (always required) and
OUTPUT_BASE/curriculum_verified/curriculum_verified.json (optional —
absent before validate_qa completes; analyzer skips §2 in that case).

Invoked by scripts/analysis.sh; can also be run directly for debugging.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Reporter — mirrors analysis_extract.py's pattern for consistent CLI output.
# ---------------------------------------------------------------------------
class Reporter:
    def __init__(self, json_mode: bool = False, quiet: bool = False):
        self.json_mode = json_mode
        self.quiet = quiet
        self.results: list[dict] = []
        self.section_name = ""
        self.fails = 0
        self.warns = 0

    def section(self, name: str) -> None:
        self.section_name = name
        if not self.json_mode:
            print(f"\n  [§{name}]")

    def ok(self, msg: str) -> None:
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "ok", "msg": msg})
        elif not self.quiet:
            print(f"    OK   {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "warn", "msg": msg})
        else:
            print(f"    WARN {msg}")

    def fail(self, msg: str) -> None:
        self.fails += 1
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "fail", "msg": msg})
        else:
            print(f"    FAIL {msg}")

    def note(self, msg: str) -> None:
        if self.json_mode or self.quiet:
            return
        print(f"         {msg}")

    def emit(self) -> int:
        if self.json_mode:
            print(json.dumps({
                "fails": self.fails, "warns": self.warns,
                "results": self.results,
            }, indent=2))
        else:
            verdict = "CLEAN" if self.fails == 0 and self.warns == 0 else \
                      "PASS WITH WARNINGS" if self.fails == 0 else "FAIL"
            print(f"\n  VERDICT: {self.fails} failures, {self.warns} warnings — {verdict}")
        return 1 if self.fails > 0 else (2 if self.warns > 0 else 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rel(path: Path, repo_root: Path) -> str:
    """Render a path as ./<relative> when under repo_root, else absolute."""
    try:
        return f"./{path.relative_to(repo_root)}"
    except ValueError:
        return str(path)


def _load_json(p: Path) -> Optional[list]:
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _trace_text(item: dict) -> str:
    """Extract the thinking trace from an item — schema varies."""
    return (item.get("explanation")
            or item.get("thinking_trace")
            or item.get("trace")
            or "")


def _word_count(s: str) -> int:
    return len(s.split())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    # Step filter scopes the analyzer to a subset of curriculum's pipeline
    # steps. Maps each step to the sections that are meaningful for it:
    #   path_traversal       § hops (kg_manifest.json input — not implemented)
    #   prune_paths          (no analyzable output)
    #   generate_qa          §1, §3 (gen-only), §4 (gen), §5 (gen), §6 (gen), §sample (gen)
    #   validate_qa          §1 (counts), §2 (drop rate)
    #   assemble_curriculum  §3 (verified), §4 (verified), §5 (verified), §6 (verified), §sample
    # Unset → all sections, using verified data when available, else generated.
    KNOWN_STEPS = ["path_traversal", "prune_paths", "generate_qa",
                   "validate_qa", "assemble_curriculum"]
    p.add_argument("--step", default=None, choices=KNOWN_STEPS + [None],
                   help="restrict analysis to one curriculum step")
    p.add_argument("--top", type=int, default=10, help="top-K concepts to surface in diversity check")
    p.add_argument("--sample", type=int, default=5, help="random verified questions to print")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run", default=None,
                   help="RUN_ID prefix; default = $OUTPUT_BASE or auto-detect latest")
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    output_base = Path(os.environ.get("OUTPUT_BASE", repo_root / "outputs"))

    # Run resolution: layered, matching diagnose.sh's pattern:
    #   1. --run <prefix> → look under output_base/<matching-run>/
    #   2. output_base itself has curriculum/ → use it as the run dir
    #   3. Auto-detect newest run under output_base/ with curriculum/
    def _find_run_dir() -> Optional[Path]:
        if (output_base / "curriculum" / "curriculum.json").exists():
            return output_base
        if args.run:
            match = sorted(
                (p for p in output_base.glob(f"{args.run}*")
                 if p.is_dir() and (p / "curriculum" / "curriculum.json").exists()),
                reverse=True
            )
            return match[0] if match else None
        # Auto-detect
        candidates = sorted(
            (p for p in output_base.iterdir() if p.is_dir()
             and (p / "curriculum" / "curriculum.json").exists()),
            reverse=True
        )
        return candidates[0] if candidates else None

    run_dir = _find_run_dir()
    if run_dir is None:
        rep = Reporter(json_mode=args.json, quiet=args.quiet)
        rep.section("init")
        rep.fail(f"no run found with curriculum/curriculum.json under {_rel(output_base, repo_root)}")
        if args.run:
            rep.note(f"--run '{args.run}' didn't match any dir")
        else:
            rep.note("Run extract → curriculum first, or pass --run <RUN_ID>")
        return rep.emit()

    # Two artifact paths to inspect. curriculum.json is the generate_qa output;
    # curriculum_verified.json is the validate_qa + assemble_curriculum output.
    gen_path = run_dir / "curriculum" / "curriculum.json"
    ver_path = run_dir / "curriculum_verified" / "curriculum_verified.json"

    rep = Reporter(json_mode=args.json, quiet=args.quiet)

    if not args.json:
        print(f"\n=== Curriculum phase quality report ===")
        print(f"  generated: {_rel(gen_path, repo_root)}")
        print(f"  verified:  {_rel(ver_path, repo_root)}")

    # Section-gating by --step. Each section declares which steps it
    # represents; sections fire only when the operator's filter overlaps.
    def _want(*steps: str) -> bool:
        return args.step is None or args.step in steps

    if not gen_path.exists():
        rep.section("init")
        rep.fail(f"no generated curriculum at {_rel(gen_path, repo_root)} (run generate_qa first)")
        return rep.emit()

    generated = _load_json(gen_path)
    if not isinstance(generated, list):
        rep.section("init")
        rep.fail(f"curriculum.json is not a JSON list (got {type(generated).__name__})")
        return rep.emit()

    verified = _load_json(ver_path) if ver_path.exists() else None

    # -----------------------------------------------------------------------
    # §1 generation — represents step `generate_qa` (also surfaced by
    # `validate_qa` since drop-rate math needs the generated count too).
    # -----------------------------------------------------------------------
    n_gen = len(generated)
    if _want("generate_qa", "validate_qa"):
        rep.section("1 generation")
        if n_gen == 0:
            rep.fail("curriculum.json is empty (zero questions)")
        elif n_gen < 10:
            rep.warn(f"only {n_gen} questions generated (smoke-scale)")
        else:
            rep.ok(f"{n_gen} questions in curriculum.json")

        # Schema sanity — required fields per generate_questions.py contract
        sample = generated[0] if n_gen > 0 else {}
        expected_keys = {"question", "answer", "hop_count"}
        missing_keys = expected_keys - set(sample.keys())
        if missing_keys:
            rep.warn(f"first item missing keys: {sorted(missing_keys)}")
        else:
            rep.ok(f"schema OK (question / answer / hop_count present)")

    # -----------------------------------------------------------------------
    # §2 verification (drop rate) — represents step `validate_qa`
    # -----------------------------------------------------------------------
    if _want("validate_qa"):
        rep.section("2 verification")
        if verified is None:
            rep.warn(f"no verified curriculum yet (validate_qa hasn't run) — skipping drop-rate check")
        else:
            n_ver = len(verified)
            if n_ver == 0:
                rep.fail("curriculum_verified.json is empty (zero questions passed two-LLM)")
            else:
                drop = n_gen - n_ver
                drop_pct = (drop / n_gen) * 100 if n_gen > 0 else 0
                msg = (f"generated {n_gen} → verified {n_ver} (dropped {drop}, {drop_pct:.1f}%)  "
                       f"({_rel(ver_path, repo_root)})")
                # Paper §4.2 baseline: 1,843 dropped of ~8,000 ≈ 23% drop.
                if 15 <= drop_pct <= 35:
                    rep.ok(msg)
                    rep.note(f"matches paper §4.2 baseline (~23% drop)")
                elif drop_pct < 15:
                    rep.warn(f"low drop rate (<15%): {msg}")
                    rep.note(f"consensus may be too lenient — paper baseline is ~23%")
                elif drop_pct > 50:
                    rep.fail(f"very high drop rate (>50%): {msg}")
                    rep.note(f"two-LLM models are over-rejecting — check prompts or model alignment")
                else:
                    rep.warn(f"high drop rate (35-50%): {msg}")

    # Quality sections (§3-§6, §sample) read from verified when available,
    # else fall back to generated. They represent the curriculum's OUTPUT
    # quality — meaningful for generate_qa (raw output) and
    # assemble_curriculum (post-verification output).
    source = verified if verified is not None else generated
    source_label = "verified" if verified is not None else "generated"

    # -----------------------------------------------------------------------
    # §3 hop distribution
    # -----------------------------------------------------------------------
    if _want("generate_qa", "assemble_curriculum"):
        rep.section("3 hop distribution")
        hops = [item.get("hop_count", 0) for item in source]
        hop_counter: dict[int, int] = {}
        for h in hops:
            hop_counter[h] = hop_counter.get(h, 0) + 1

        if not hop_counter:
            rep.warn("no hop_count field on any item")
        else:
            rep.ok(f"{source_label}: hop_counts seen → {dict(sorted(hop_counter.items()))}")
            all_one_hop = (len(hop_counter) == 1 and 1 in hop_counter)
            if all_one_hop:
                rep.warn("ALL items are 1-hop — calculate_hops likely fell back to seed_kg "
                         "(graphmert produced no real expansion)")
                rep.note("See calculate_hops.py:129-137 (empty-fallback). Real fix: "
                         "investigate why graphmert.validate_predictions yielded 0 triples.")
            elif max(hop_counter) > 5:
                rep.warn(f"hops > 5 present (max {max(hop_counter)}); paper uses [1..5]")

    # -----------------------------------------------------------------------
    # §4 answer letter distribution (A/B/C/D balance)
    # -----------------------------------------------------------------------
    if _want("generate_qa", "assemble_curriculum"):
        rep.section("4 answer balance")
        answers = [str(item.get("answer", "")).strip().upper() for item in source]
        # Normalise: items may store "A" or "Option A" etc.
        answer_letters = []
        for a in answers:
            for ch in a:
                if ch in "ABCD":
                    answer_letters.append(ch)
                    break

        if not answer_letters:
            rep.warn("no parseable A/B/C/D answers found")
        else:
            letter_counter = {"A": 0, "B": 0, "C": 0, "D": 0}
            for L in answer_letters:
                letter_counter[L] += 1
            total = sum(letter_counter.values())
            dist_str = "  ".join(f"{k}={v} ({v/total*100:.0f}%)" for k, v in letter_counter.items())
            max_pct = max(v / total for v in letter_counter.values()) * 100
            min_pct = min(v / total for v in letter_counter.values()) * 100

            if max_pct > 40:
                biased = max(letter_counter, key=letter_counter.get)
                rep.warn(f"positional bias: {biased} dominates "
                         f"({letter_counter[biased]/total*100:.0f}%)")
                rep.note(dist_str)
            elif min_pct < 10:
                scarce = min(letter_counter, key=letter_counter.get)
                rep.warn(f"answer letter {scarce} is rare "
                         f"({letter_counter[scarce]/total*100:.0f}%)")
                rep.note(dist_str)
            else:
                rep.ok(f"answers balanced — {dist_str}")

    # -----------------------------------------------------------------------
    # §5 thinking trace word-count distribution
    # -----------------------------------------------------------------------
    if _want("generate_qa", "assemble_curriculum"):
        rep.section("5 thinking trace")
        word_counts = [_word_count(_trace_text(item)) for item in source if _trace_text(item)]
        if not word_counts:
            rep.warn("no thinking traces found on any item")
        else:
            n = len(word_counts)
            wc_med = statistics.median(word_counts)
            rep.ok(f"n={n}  min={min(word_counts)}  median={wc_med:.0f}  "
                   f"mean={statistics.mean(word_counts):.0f}  max={max(word_counts)}")

            # TRACE_TARGET_WORDS=250, TRACE_MAX_WORDS=350 per generate_questions.py:38-40
            if wc_med < 100:
                rep.warn(f"median trace ({wc_med:.0f} words) below TRACE_MIN_WORDS (100) — too terse")
            elif wc_med > 350:
                rep.warn(f"median trace ({wc_med:.0f} words) above TRACE_MAX_WORDS (350) — "
                         "trace_length_check may not be firing")

            no_trace = sum(1 for item in source if not _trace_text(item))
            if no_trace > 0:
                rep.warn(f"{no_trace} items have empty / missing trace "
                         f"({no_trace/len(source)*100:.0f}%)")

    # -----------------------------------------------------------------------
    # §6 concept diversity
    # -----------------------------------------------------------------------
    if _want("generate_qa", "assemble_curriculum"):
        rep.section("6 concept diversity")
        sources = [str(item.get("source_concept", "")).strip() for item in source]
        targets = [str(item.get("target_concept", "")).strip() for item in source]
        sources = [s for s in sources if s]
        targets = [t for t in targets if t]

        if not sources and not targets:
            rep.warn("no source_concept / target_concept fields found")
        else:
            u_src = len(set(sources))
            u_tgt = len(set(targets))
            rep.ok(f"unique source concepts: {u_src} / {len(sources)} items "
                   f"({u_src/len(sources)*100:.0f}% diversity)")
            rep.ok(f"unique target concepts: {u_tgt} / {len(targets)} items "
                   f"({u_tgt/len(targets)*100:.0f}% diversity)")

            src_counter: dict[str, int] = {}
            for s in sources:
                src_counter[s] = src_counter.get(s, 0) + 1
            top_src = sorted(src_counter.items(), key=lambda x: -x[1])[:args.top]
            if top_src and (top_src[0][1] / len(sources)) > 0.20:
                rep.warn(f"hot source concept: '{top_src[0][0]}' appears "
                         f"{top_src[0][1]}/{len(sources)} times "
                         f"({top_src[0][1]/len(sources)*100:.0f}%)")
            rep.note(f"top-{args.top} source concepts:")
            for s, c in top_src:
                rep.note(f"  {c:>4}  {s}")

    # -----------------------------------------------------------------------
    # §sample — random questions from the strongest available output
    # -----------------------------------------------------------------------
    if _want("generate_qa", "assemble_curriculum"):
        rep.section("sample questions")
        if not source:
            rep.note("no items to sample")
        else:
            rng = random.Random(42)
            picks = rng.sample(source, min(args.sample, len(source)))
            for i, item in enumerate(picks, 1):
                hop = item.get("hop_count", "?")
                src = item.get("source_concept", "?")
                tgt = item.get("target_concept", "?")
                ans = item.get("answer", "?")
                q = item.get("question", "")[:200].replace("\n", " ")
                rep.note(f"  [{i}] hop={hop}  ({src} → {tgt})  answer={ans}")
                rep.note(f"      {q}{'...' if len(item.get('question', '')) > 200 else ''}")

    return rep.emit()


if __name__ == "__main__":
    sys.exit(main())
