#!/usr/bin/env python3
"""
diagnose_charset.py — Classify non-ASCII characters in a text/log/JSON file as
GENUINE foreign-script generation (language drift) vs DECODE CORRUPTION (mojibake).

Built for the RL Chinese-character question (2026-06-26): GRPO rollouts logged
Chinese characters, and we needed to know which problem we had —

  * the model genuinely drifting to Chinese  → fixable via prompt + a CJK reward
    penalty (Qwen3 code-switches into <think> a lot); or
  * the rollout decode path emitting mojibake → the known rl-rollout-decoding-open
    bug, where NO prompt/reward change helps and the fix is in the decode path.

Telling them apart by eye is unreliable; this does it deterministically. The file
is read with errors='replace', so invalid UTF-8 bytes surface as U+FFFD — a strong
corruption signal. Coherent foreign script (named CJK ideographs / kana / hangul /
Cyrillic ...) with ~0 U+FFFD points the other way: real language drift.

Works on any text file — training logs, completion dumps, curriculum JSON.

CONTEXT-SAFE BY DESIGN: pasting a raw log into a chat is what blows up an agent's
context window. Instead, share the file, run this, and paste back only its short
verdict — a capped table of distinct offending chars (with Unicode names) plus a
few truncated samples. It never echoes the whole file.

Usage:
  python scripts/analysis/diagnose_charset.py outputs/<RUN_ID>/logs/rl.log
  python scripts/analysis/diagnose_charset.py rl.log --top 12 --samples 3
  python scripts/analysis/diagnose_charset.py completions.json --context 60
  some_command | python scripts/analysis/diagnose_charset.py -        # read stdin

Exit code: 0 = clean, 1 = drift (foreign script), 2 = decode corruption (mojibake).
"""
import argparse
import collections
import re
import sys
import unicodedata

# Unicode ranges we care about. CJK ideographs + Japanese kana + Korean hangul
# cover the Qwen "drift to Chinese/Asian script" case; Cyrillic/Arabic/Hebrew/Greek
# are flagged too so this isn't China-specific.
_RANGES = {
    "CJK":      "㐀-鿿豈-﫿",
    "Kana":     "぀-ヿ",
    "Hangul":   "가-힯",
    "Cyrillic": "Ѐ-ӿ",
    "Arabic":   "؀-ۿ",
    "Hebrew":   "֐-׿",
    "Greek":    "Ͱ-Ͽ",
}
_SCRIPT_RE = {name: re.compile(f"[{rng}]") for name, rng in _RANGES.items()}
_FOREIGN_RE = re.compile("[" + "".join(_RANGES.values()) + "]")
_FFFD = "�"  # replacement char errors='replace' inserts for undecodable bytes


def script_of(ch):
    for name, rx in _SCRIPT_RE.items():
        if rx.match(ch):
            return name
    return "Other"


def samples(data, n, context):
    """Up to n non-overlapping windows of +/- `context` chars around foreign hits."""
    out, last_end = [], -1
    for m in _FOREIGN_RE.finditer(data):
        if m.start() < last_end:
            continue  # already inside a printed window
        a, b = max(0, m.start() - context), min(len(data), m.start() + context)
        out.append("  …" + data[a:b].replace("\n", "⏎").replace("\r", "") + "…")
        last_end = b
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="text/log/json file to scan ('-' for stdin)")
    ap.add_argument("--top", type=int, default=8,
                    help="distinct foreign chars to list (default 8)")
    ap.add_argument("--samples", type=int, default=2,
                    help="truncated context windows to show (default 2)")
    ap.add_argument("--context", type=int, default=50,
                    help="chars of context on each side of a sample (default 50)")
    a = ap.parse_args()

    if a.path == "-":
        data = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        label = "<stdin>"
    else:
        with open(a.path, "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
        label = a.path

    total = len(data)
    non_ascii = sum(1 for c in data if ord(c) > 0x7F)
    fffd = data.count(_FFFD)
    foreign = _FOREIGN_RE.findall(data)
    by_script = collections.Counter(script_of(c) for c in foreign)

    print(f"file: {label}")
    print(f"chars: {total} | non-ascii: {non_ascii} | foreign-script: {len(foreign)} "
          f"| U+FFFD (decode corruption): {fffd}")
    if by_script:
        print("by script: " + ", ".join(f"{k}={v}" for k, v in by_script.most_common()))

    for ch, n in collections.Counter(foreign).most_common(a.top):
        print(f"  {ch}  U+{ord(ch):04X}  x{n}  [{script_of(ch)}]  "
              f"{unicodedata.name(ch, '<unnamed>')}")

    snips = samples(data, a.samples, a.context)
    if snips:
        print("samples:")
        for s in snips:
            print(s)

    # Verdict. U+FFFD means bytes that could not be decoded at all — that is the
    # corruption signature, and it dominates: a decode bug can also splatter random
    # CJK, so the presence of real undecodable bytes is the stronger evidence.
    print("-" * 60)
    if fffd > 0:
        verdict, code = ("DECODE CORRUPTION (mojibake) — undecodable bytes present. "
                         "This is the decode path, NOT language drift: a prompt or "
                         "reward change will NOT help. Fix the rollout decode "
                         "(tokenizer / skip_special_tokens / byte handling)."), 2
    elif foreign:
        verdict, code = ("GENUINE foreign-script generation (language drift) — chars "
                         "are well-formed, named code points with no undecodable "
                         "bytes. The model really emits them. Address via the system "
                         "prompt + a foreign-script reward penalty (and/or sampling)."), 1
    else:
        verdict, code = ("CLEAN — no foreign-script chars and no decode corruption.", 0)
    print("VERDICT:", verdict)
    return code


if __name__ == "__main__":
    sys.exit(main())
