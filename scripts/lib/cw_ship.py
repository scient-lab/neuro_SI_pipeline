#!/usr/bin/env python3
"""Ship one completed step log to AWS CloudWatch Logs (optional, best-effort).

Invoked by common.sh::_cw_ship after a step finishes, ONLY when CW_LOG_GROUP
is set. One log stream per (run_id, phase, step) — so a single put_log_events
batch covers the whole step and we never juggle sequence tokens across calls.

Requires boto3 + AWS creds in the environment (the same creds the S3 sync
uses). If boto3 is missing this exits non-zero with a clear message and the
caller treats it as non-fatal — the local file + S3 copy remain the source of
truth. For LIVE streaming instead of per-step batches, install the CloudWatch
unified agent in runpod_bootstrap.sh and point it at logs/<run_id>/.

CloudWatch limits handled: <=10,000 events and <=1 MiB per PutLogEvents call;
oversized logs are truncated to the last N lines with a marker (the full log
is always in S3).
"""

from __future__ import annotations

import argparse
import sys
import time

MAX_EVENTS = 10_000
MAX_BYTES = 1_000_000  # leave headroom under the 1,048,576 hard cap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--file", required=True)
    a = ap.parse_args()

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("cw_ship: boto3 not installed; skipping CloudWatch push", file=sys.stderr)
        return 1

    try:
        with open(a.file, "r", errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
    except FileNotFoundError:
        return 0

    if not lines:
        return 0

    # Keep the tail if the log is huge; the complete copy lives in S3.
    if len(lines) > MAX_EVENTS:
        lines = ["[truncated — see full log in S3]"] + lines[-(MAX_EVENTS - 1):]

    ts = int(time.time() * 1000)
    events, size = [], 0
    for ln in lines:
        msg = ln or " "  # CloudWatch rejects empty messages
        ev_size = len(msg.encode("utf-8")) + 26  # 26B per-event overhead
        if size + ev_size > MAX_BYTES:
            break
        events.append({"timestamp": ts, "message": msg})
        size += ev_size

    logs = boto3.client("logs")
    for create in (logs.create_log_group, logs.create_log_stream):
        try:
            if create is logs.create_log_group:
                logs.create_log_group(logGroupName=a.group)
            else:
                logs.create_log_stream(logGroupName=a.group, logStreamName=a.stream)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                print(f"cw_ship: {e}", file=sys.stderr)

    try:
        logs.put_log_events(
            logGroupName=a.group, logStreamName=a.stream, logEvents=events
        )
    except ClientError as e:
        print(f"cw_ship: put_log_events failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
