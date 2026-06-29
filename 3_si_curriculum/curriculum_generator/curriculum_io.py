"""Shared JSONL + stats I/O for the streamed 4-step curriculum flow.

The curriculum phase writes three files (see CURRICULUM_4STEP_REFACTOR_PLAN.md):
  - curriculum.jsonl        one record per line; each step adds fields + advances `stage`
                            (pair -> validated_pair -> item -> verified | drop). Drops are
                            RETAINED with a `drop_reason` so the yield chain + the "why" live
                            in one file.
  - curriculum_stats.json   {<step>: {in, out, dropped, yield, ...}} — each step writes its
                            own key during its streaming pass (cheap O(1) reads for analysis).
  - curriculum_verified.json  final JSON array of stage==verified records (assemble step).

Transform stages stream in constant memory: read a chunk, process it (the caller may run the
chunk's target-stage records through an LLM/API in parallel), write the chunk in order to a
temp file, then atomically rename over the output. input_path may equal output_path (in-place).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List

# Stage values (single source of truth for the field vocabulary).
STAGE_PAIR = "pair"
STAGE_VALIDATED_PAIR = "validated_pair"
STAGE_ITEM = "item"
STAGE_VERIFIED = "verified"
STAGE_DROP = "drop"


def stream_records(path: str) -> Iterator[Dict]:
    """Yield one record dict per non-blank line of a JSONL file (empty if absent)."""
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def format_path(paths: List[Dict]) -> str:
    """Render a [{start, relation, end}, ...] path as 'A -> rel -> B -> rel -> C'."""
    if not paths:
        return ""
    parts = [str(paths[0].get("start", ""))]
    for step in paths:
        parts.append(f"-> {step.get('relation', '')} ->")
        parts.append(str(step.get("end", "")))
    return " ".join(parts)


def write_stat(stats_path: str, step: str, counts: Dict) -> None:
    """Merge a per-step counts dict into curriculum_stats.json[step] (atomic)."""
    stats: Dict = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f) or {}
    stats[step] = counts
    _atomic_dump_json(stats_path, stats)


def yield_counts(in_n: int, out_n: int, **extra) -> Dict:
    """Build the canonical {in, out, dropped, yield, ...} counts dict."""
    counts = {
        "in": in_n,
        "out": out_n,
        "dropped": in_n - out_n,
        "yield": round(out_n / in_n, 4) if in_n else 0.0,
    }
    counts.update(extra)
    return counts


def transform_jsonl(input_path: str, output_path: str,
                    process_chunk: Callable[[List[Dict]], None],
                    chunk_size: int = 256) -> None:
    """Stream input_path in chunks; mutate each chunk in place via process_chunk; write the
    chunk (in input order) to output_path. Atomic (temp + rename), constant memory. The
    generator is fully drained before the rename, so input_path == output_path is safe.
    """
    out_dir = Path(output_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            chunk: List[Dict] = []
            for rec in stream_records(input_path):
                chunk.append(rec)
                if len(chunk) >= chunk_size:
                    process_chunk(chunk)
                    _write_lines(out, chunk)
                    chunk = []
            if chunk:
                process_chunk(chunk)
                _write_lines(out, chunk)
        os.replace(tmp, output_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def open_jsonl_writer(path: str):
    """Open a JSONL file for writing (creating parent dirs). Caller writes via write_record."""
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", encoding="utf-8")


def write_record(fh, record: Dict) -> None:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_all_jsonl(path: str, records: List[Dict]) -> None:
    """Write a list of records as JSONL via temp + atomic rename. Used by stages that must
    hold the working set in memory anyway (e.g. the vLLM consensus batches all items)."""
    out_dir = Path(path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".jsonl.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _write_lines(f, records)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_lines(fh, records: List[Dict]) -> None:
    for r in records:
        write_record(fh, r)


def _atomic_dump_json(path: str, obj) -> None:
    out_dir = Path(path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
