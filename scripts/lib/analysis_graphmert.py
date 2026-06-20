#!/usr/bin/env python3
"""Quality analysis for the graphmert phase outputs.

Planned sections (built out after smoke graphmert produces real artifacts;
designing now against guessed schemas wastes effort):

  §1 tokenize         vocab size, special tokens, coverage on csv heads/tails
  §2 preprocess       grounded sample count, per-relation drop-out,
                      per-head coverage, head-tail overlap
  §3 train_mnm        eval_loss curve, best vs final checkpoint, overfit gap,
                      step-wise progression, lr schedule actually applied
  §4 predict_tails    prediction count, per-head coverage, confidence/
                      probability distribution, top-K diversity
  §5 validate_predictions
                      pass rate, two-LLM agreement, score distribution,
                      flagged-vs-accepted ratio
  §6 expand_kg        seed-vs-expanded triple count, growth ratio, new
                      entities introduced, relation-distribution shift

Currently this is a stub. It only walks $OUTPUT_BASE/graphmert/ and reports
what artifacts exist so the operator can confirm the phase ran end-to-end.
Sections above will be filled in once the smoke run completes and we can
inspect the actual artifact schemas (trainer_state.json structure, prediction
CSV columns, fact-score column names, etc.).

Invoked by scripts/analysis.sh; can also be run directly for debugging.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Step list mirrors scripts/phases/graphmert.sh::STEPS so --step values agree
# with the rest of the pipeline.
KNOWN_STEPS = ["tokenize", "preprocess", "train_mnm",
               "predict_tails", "validate_predictions", "expand_kg"]


def _emit(json_mode: bool, payload: dict) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--step", default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run", default=None)
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    output_base = Path(os.environ.get("OUTPUT_BASE", repo_root / "outputs"))
    gm_dir = output_base / "graphmert"

    if not args.json:
        print(f"\n=== Graphmert phase quality report ===")
        print(f"  source: {gm_dir.relative_to(repo_root) if repo_root in gm_dir.parents else gm_dir}")

    if not gm_dir.exists():
        if args.json:
            _emit(True, {"phase": "graphmert", "status": "not_run",
                         "expected_dir": str(gm_dir)})
        else:
            print(f"\n  STATUS: graphmert phase has not run yet (no {gm_dir})")
            print(f"  Run ./scripts/pipeline.sh --phase graphmert first.")
        return 0

    if args.step and args.step not in KNOWN_STEPS:
        print(f"\n  unknown --step '{args.step}'. Known: {KNOWN_STEPS}", file=sys.stderr)
        return 1

    # ----- Stub: report what's on disk so the operator can confirm phase ran -----
    if not args.json:
        print(f"\n  STUB — full quality analysis pending. Currently lists artifacts present:")

    artifacts: list[dict] = []
    for path in sorted(gm_dir.rglob("*")):
        if path.is_dir():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        rel = path.relative_to(gm_dir)
        artifacts.append({"path": str(rel), "size_bytes": size})

    if args.json:
        _emit(True, {
            "phase": "graphmert",
            "status": "stub",
            "step_filter": args.step,
            "artifacts": artifacts[:200],
            "artifact_count": len(artifacts),
        })
        return 0

    if not artifacts:
        print(f"  WARN  graphmert/ exists but is empty")
        return 2

    print(f"\n  Found {len(artifacts)} artifact(s):")
    for art in artifacts[:30]:
        size_h = _human(art["size_bytes"])
        print(f"    {size_h:>10}  {art['path']}")
    if len(artifacts) > 30:
        print(f"    ... and {len(artifacts) - 30} more")

    print()
    print(f"  Next: once smoke graphmert completes, full §1-§6 checks land here.")
    return 0


def _human(n: int) -> str:
    if n < 0:
        return "?"
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n //= 1024
    return f"{n:.0f}T"


if __name__ == "__main__":
    sys.exit(main())
