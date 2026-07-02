#!/usr/bin/env python3
"""corpus_stats.py — doc/word/size/token stats for a text corpus directory.

Recursively scans *.txt (configurable) under a corpus path and totals: number
of docs, size, lines, words, chars, and tokens.

Tokens default to a fast estimate (chars/4, no deps). Pass --tokenizer [HF_ID]
for an EXACT count via the model tokenizer (loads `transformers`; default id
Qwen/Qwen3-8B to match the pipeline base). Use scripts/corpus_stats.sh to run
under a venv that has transformers.

Usage:
  python scripts/corpus_stats.py corpus/space/smoke
  python scripts/corpus_stats.py corpus/space/smoke --per-file
  python scripts/corpus_stats.py corpus/space/smoke --tokenizer                 # Qwen/Qwen3-8B exact
  python scripts/corpus_stats.py corpus/space/smoke --tokenizer Qwen/Qwen3-14B
  python scripts/corpus_stats.py corpus/space/smoke --ext txt,md --json
"""
import argparse
import json
import sys
from pathlib import Path


def human(n: float) -> str:
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or u == "TiB":
            return f"{int(x)}B" if u == "B" else f"{x:.1f}{u}"
        x /= 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="corpus directory (e.g. corpus/space/smoke)")
    ap.add_argument("--ext", default="txt", help="comma-separated extensions (default: txt)")
    ap.add_argument("--tokenizer", nargs="?", const="Qwen/Qwen3-8B", default=None,
                    metavar="HF_ID", help="exact token count via this HF tokenizer "
                                          "(bare flag → Qwen/Qwen3-8B)")
    ap.add_argument("--per-file", action="store_true", help="per-doc table (sorted by tokens)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    root = Path(a.path)
    if not root.exists():
        print(f"path not found: {root}", file=sys.stderr)
        return 2
    exts = tuple("." + e.strip().lstrip(".").lower() for e in a.ext.split(",") if e.strip())
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not files:
        print(f"no {a.ext} files under {root}", file=sys.stderr)
        return 1

    # Optional exact tokenizer (content tokens only; no BOS/EOS).
    tok = None
    if a.tokenizer:
        try:
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()          # silence >max-length warnings
            tok = AutoTokenizer.from_pretrained(a.tokenizer)
            tok.model_max_length = int(1e12)          # we're counting, never truncating
        except Exception as e:
            print(f"could not load tokenizer '{a.tokenizer}': {e}\n"
                  f"  → falling back to the chars/4 estimate.", file=sys.stderr)
            tok = None
    exact = tok is not None

    per, tot = [], dict(bytes=0, lines=0, words=0, chars=0, tokens=0)
    for f in files:
        b = f.stat().st_size
        text = f.read_text(errors="replace")
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        words = len(text.split())
        chars = len(text)
        ntok = len(tok(text, add_special_tokens=False)["input_ids"]) if exact else round(chars / 4)
        per.append({"file": str(f.relative_to(root)), "bytes": b, "lines": lines,
                    "words": words, "chars": chars, "tokens": ntok})
        for k, v in (("bytes", b), ("lines", lines), ("words", words), ("chars", chars), ("tokens", ntok)):
            tot[k] += v

    n = len(files)
    tok_label = f"{a.tokenizer}, exact" if exact else "estimate ≈ chars/4 — pass --tokenizer for exact"

    if a.json:
        print(json.dumps({"corpus": str(root), "docs": n, "ext": exts,
                          "tokenizer": a.tokenizer if exact else None,
                          "tokens_exact": exact, "totals": tot,
                          "per_file": per if a.per_file else None}, indent=2))
        return 0

    print(f"Corpus  : {root}  ({n} {'/'.join(e.lstrip('.') for e in exts)} docs, recursive)")
    print(f"Size    : {human(tot['bytes'])}")
    print(f"Lines   : {tot['lines']:,}")
    print(f"Words   : {tot['words']:,}")
    print(f"Chars   : {tot['chars']:,}")
    print(f"Tokens  : {'' if exact else '~'}{tot['tokens']:,}  ({tok_label})")
    print(f"Per doc : {tot['words'] // n:,} words, {human(tot['bytes'] / n)}, "
          f"{'' if exact else '~'}{tot['tokens'] // n:,} tokens (avg)")

    if a.per_file:
        print(f"\n  {'TOKENS':>9}  {'WORDS':>8}  {'SIZE':>9}  FILE")
        print(f"  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*4}")
        for r in sorted(per, key=lambda x: x["tokens"], reverse=True):
            print(f"  {r['tokens']:>9,}  {r['words']:>8,}  {human(r['bytes']):>9}  {r['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
