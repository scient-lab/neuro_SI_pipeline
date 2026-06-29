#!/usr/bin/env python3
"""assemble_curriculum.py — final curriculum step.

Filter stage==verified records out of curriculum.jsonl into curriculum_verified.json (the JSON
array consumed by SFT data_prep), and finalize curriculum_stats.json with an assemble summary
(verified count + per-hop distribution + answer-key balance).
"""
import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curriculum_generator import curriculum_io as cio  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Assemble verified curriculum + finalize stats")
    ap.add_argument("--curriculum_jsonl", required=True,
                    help="curriculum.jsonl produced by the 4-step flow")
    ap.add_argument("--output_json", required=True,
                    help="Output JSON array of stage==verified records (for SFT data_prep)")
    ap.add_argument("--stats_path", default="",
                    help="curriculum_stats.json (default: alongside the jsonl)")
    return ap.parse_args()


def main():
    args = parse_args()
    stats_path = args.stats_path or os.path.join(
        os.path.dirname(os.path.abspath(args.curriculum_jsonl)), "curriculum_stats.json")

    total = 0
    verified = []
    by_hop = Counter()
    by_answer = Counter()
    for rec in cio.stream_records(args.curriculum_jsonl):
        total += 1
        if rec.get("stage") == cio.STAGE_VERIFIED:
            verified.append(rec)
            by_hop[str(rec.get("hop_count"))] += 1
            by_answer[str(rec.get("answer"))] += 1

    Path(args.output_json).resolve().parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)

    cio.write_stat(stats_path, "assemble_curriculum", {
        "in": total,
        "out": len(verified),
        "dropped": total - len(verified),
        "yield": round(len(verified) / total, 4) if total else 0.0,
        "by_hop": dict(by_hop),
        "answer_key_balance": dict(by_answer),
    })
    print(f"assemble_curriculum: {len(verified)} verified / {total} records -> {args.output_json}")


if __name__ == "__main__":
    main()
