#!/usr/bin/env python3
"""Emit one health sample as two CSV rows (system + per-GPU).

Stdlib-only (runs under any activated venv or bare python3). Appends to
two CSVs under a health/ dir; writes the header once. Designed to be
called once per tick by scripts/monitor.sh.

  health_system.csv : one row per tick  (ts, phase, step, cpu/mem/disk/net, gpu aggregates, status, alerts)
  health_gpu.csv    : one row per (tick, gpu)  (leads with gpu_index)

Net RATE (net_rx_mbps/net_tx_mbps) is derived from the delta vs the last
system-csv row, so the sampler is self-contained — no external state file.

On stdout it prints a compact `key=value` summary that monitor.sh parses
for its (stateful) kill decisions.
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone

SYS_COLS = [
    "ts", "phase", "step", "pipeline_status", "failed_phase", "pod_id", "run_id", "uptime_s",
    "cpu_pct", "cores", "load1", "mem_used_gb", "mem_total_gb", "mem_pct",
    "disk_root_pct", "disk_root_free_gb", "disk_ws_pct", "disk_ws_free_gb",
    "net_rx_mbps", "net_tx_mbps", "net_rx_mb", "net_tx_mb",
    "gpu_count", "gpu_util_avg", "gpu_util_max", "gpu_vram_pct_avg", "gpu_vram_pct_max",
    "status", "alerts",
]
GPU_COLS = [
    "gpu_index", "ts", "run_id", "name",
    "util_pct", "vram_used_mb", "vram_total_mb", "vram_pct", "temp_c", "power_w", "throttle",
    "top_proc_pid", "top_proc_mem_mb",
]

# nvidia-smi clocks_throttle_reasons.active bitmask -> human label.
# GpuIdle (0x1) / AppClocks (0x2) are benign and intentionally excluded.
THROTTLE_BITS = [
    (0x4, "sw_power_cap"), (0x8, "hw_slowdown"), (0x20, "sw_thermal"),
    (0x40, "hw_thermal"), (0x80, "hw_power_brake"),
]


def _round(v, n=1):
    try:
        return round(float(v), n)
    except Exception:
        return None


def read_cpu_pct():
    """Aggregate CPU utilisation % over a short delta."""
    def snap():
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = list(map(int, parts[1:]))
        idle = vals[3] + vals[4]            # idle + iowait
        return sum(vals), idle
    try:
        t0, i0 = snap(); time.sleep(0.2); t1, i1 = snap()
        dt, di = t1 - t0, i1 - i0
        return _round(100.0 * (dt - di) / dt) if dt > 0 else None
    except Exception:
        return None


def read_loadavg():
    try:
        with open("/proc/loadavg") as f:
            return _round(f.read().split()[0], 2)
    except Exception:
        return None


def read_mem():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k] = int(v.split()[0])      # kB
        total = info["MemTotal"] / 1024 / 1024
        avail = info.get("MemAvailable", info["MemFree"]) / 1024 / 1024
        used = total - avail
        pct = _round(100.0 * used / total) if total else None
        return _round(used), _round(total), pct
    except Exception:
        return None, None, None


def read_disk(mount):
    try:
        if not os.path.isdir(mount):
            return None, None
        u = shutil.disk_usage(mount)
        pct = _round(100.0 * u.used / u.total, 0)
        return (int(pct) if pct is not None else None), _round(u.free / 1e9)
    except Exception:
        return None, None


def read_net_cumulative():
    """Sum rx/tx bytes over non-loopback interfaces -> MB (cumulative)."""
    try:
        rx = tx = 0
        with open("/proc/net/dev") as f:
            for line in f.readlines()[2:]:
                iface, rest = line.split(":", 1)
                if iface.strip() == "lo":
                    continue
                cols = rest.split()
                rx += int(cols[0]); tx += int(cols[8])
        return _round(rx / 1e6), _round(tx / 1e6)
    except Exception:
        return None, None


def decode_throttle(raw):
    try:
        bits = int(str(raw).strip(), 16)
    except Exception:
        return "none"
    labels = [lbl for mask, lbl in THROTTLE_BITS if bits & mask]
    return "+".join(labels) if labels else "none"


def read_gpus():
    """Return list of per-GPU dicts via nvidia-smi; [] if unavailable."""
    q = "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        c = [x.strip() for x in line.split(",")]
        if len(c) < 9:
            continue
        used, total = _round(c[4], 0), _round(c[5], 0)
        vram_pct = _round(100.0 * used / total) if total else None
        gpus.append({
            "gpu_index": c[0], "uuid": c[1], "name": c[2],
            "util_pct": _round(c[3], 0), "vram_used_mb": int(used) if used is not None else None,
            "vram_total_mb": int(total) if total is not None else None, "vram_pct": vram_pct,
            "temp_c": _round(c[6], 0), "power_w": _round(c[7]),
            "throttle": decode_throttle(c[8]),
            "top_proc_pid": "", "top_proc_mem_mb": "",
        })
    # attach per-GPU top compute process (uuid -> index)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        top = {}   # uuid -> (pid, mem)
        for line in out.strip().splitlines():
            c = [x.strip() for x in line.split(",")]
            if len(c) < 3:
                continue
            pid, mem, uuid = c[0], _round(c[1], 0), c[2]
            if mem is not None and (uuid not in top or mem > top[uuid][1]):
                top[uuid] = (pid, mem)
        for g in gpus:
            if g["uuid"] in top:
                g["top_proc_pid"], g["top_proc_mem_mb"] = top[g["uuid"]][0], int(top[g["uuid"]][1])
    except Exception:
        pass
    return gpus


def read_manifest(path):
    """run_id, current phase/step, pipeline status, failed phase."""
    out = {"run_id": "", "phase": "", "step": "", "pipeline_status": "", "failed_phase": ""}
    try:
        with open(path) as f:
            run = (json.load(f) or {}).get("run", {})
        out["run_id"] = run.get("run_id", "") or ""
        out["pipeline_status"] = run.get("status", "") or ""
        out["phase"] = run.get("current_phase", "") or ""
        for ph in run.get("phases", []):
            if ph.get("status") == "failed":
                out["failed_phase"] = ph.get("name", "") or ""
            if ph.get("name") == out["phase"]:
                for st in ph.get("steps", []):
                    if st.get("status") == "running":
                        out["step"] = st.get("name", "") or ""
    except Exception:
        pass
    return out


def last_sys_row(path):
    """Last row of the system csv (for net-rate delta). None if absent."""
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def net_rate(prev, ts_now, rx_now, tx_now):
    """MB/s since the previous sample; None on first tick / counter reset."""
    if not prev:
        return None, None
    try:
        t0 = datetime.fromisoformat(prev["ts"]).timestamp()
        dt = ts_now.timestamp() - t0
        rx0, tx0 = float(prev["net_rx_mb"]), float(prev["net_tx_mb"])
        if dt <= 0 or rx_now < rx0 or tx_now < tx0:   # reset/reboot
            return None, None
        return _round((rx_now - rx0) / dt, 2), _round((tx_now - tx0) / dt, 2)
    except Exception:
        return None, None


def classify(disk_pcts, gpus, pipeline_status, failed_phase, a):
    """Stateless health verdict + alerts (idle-hang is stateful → monitor.sh)."""
    alerts, status = [], "ok"

    def bump(level):
        nonlocal status
        order = {"ok": 0, "warn": 1, "critical": 2}
        if order[level] > order[status]:
            status = level

    for mount, pct in disk_pcts:
        if pct is None:
            continue
        if pct >= a.disk_crit:
            alerts.append(f"disk:{mount} {pct}% (critical)"); bump("critical")
        elif pct >= a.disk_warn:
            alerts.append(f"disk:{mount} {pct}%"); bump("warn")
    for g in gpus:
        vp = g["vram_pct"]
        if vp is None:
            continue
        if vp >= a.vram_crit:
            alerts.append(f"gpu{g['gpu_index']} vram {vp}% (critical)"); bump("critical")
        elif vp >= a.vram_warn:
            alerts.append(f"gpu{g['gpu_index']} vram {vp}%"); bump("warn")
        if g["throttle"] != "none":
            alerts.append(f"gpu{g['gpu_index']} throttle {g['throttle']}"); bump("warn")
    if pipeline_status == "failed":
        alerts.append(f"pipeline failed: {failed_phase or '?'}"); bump("critical")
    return status, " | ".join(alerts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="")
    p.add_argument("--system-csv", required=True)
    p.add_argument("--gpu-csv", required=True)
    p.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""))
    p.add_argument("--disk-warn", type=int, default=85)
    p.add_argument("--disk-crit", type=int, default=95)
    p.add_argument("--vram-warn", type=int, default=92)
    p.add_argument("--vram-crit", type=int, default=98)
    a = p.parse_args()

    now = datetime.now(timezone.utc)
    ts = now.isoformat(timespec="seconds")
    man = read_manifest(a.manifest) if a.manifest else \
        {"run_id": os.environ.get("RUN_ID", ""), "phase": "", "step": "",
         "pipeline_status": "", "failed_phase": ""}

    try:
        uptime_s = int(float(open("/proc/uptime").read().split()[0]))
    except Exception:
        uptime_s = None
    cpu, cores, load1 = read_cpu_pct(), os.cpu_count(), read_loadavg()
    mem_used, mem_total, mem_pct = read_mem()
    dr_pct, dr_free = read_disk("/")
    dw_pct, dw_free = read_disk("/workspace")
    rx_mb, tx_mb = read_net_cumulative()
    gpus = read_gpus()

    prev = last_sys_row(a.system_csv)
    rx_bps, tx_bps = net_rate(prev, now, rx_mb or 0, tx_mb or 0)

    utils = [g["util_pct"] for g in gpus if g["util_pct"] is not None]
    vrams = [g["vram_pct"] for g in gpus if g["vram_pct"] is not None]
    gpu_util_avg = _round(sum(utils) / len(utils), 0) if utils else None
    gpu_util_max = max(utils) if utils else None
    gpu_vram_avg = _round(sum(vrams) / len(vrams)) if vrams else None
    gpu_vram_max = max(vrams) if vrams else None

    status, alerts = classify([("/", dr_pct), ("/workspace", dw_pct)], gpus,
                              man["pipeline_status"], man["failed_phase"], a)

    sys_row = {
        "ts": ts, "phase": man["phase"], "step": man["step"],
        "pipeline_status": man["pipeline_status"], "failed_phase": man["failed_phase"],
        "pod_id": a.pod_id, "run_id": man["run_id"], "uptime_s": uptime_s,
        "cpu_pct": cpu, "cores": cores, "load1": load1,
        "mem_used_gb": mem_used, "mem_total_gb": mem_total, "mem_pct": mem_pct,
        "disk_root_pct": dr_pct, "disk_root_free_gb": dr_free,
        "disk_ws_pct": dw_pct, "disk_ws_free_gb": dw_free,
        "net_rx_mbps": rx_bps, "net_tx_mbps": tx_bps, "net_rx_mb": rx_mb, "net_tx_mb": tx_mb,
        "gpu_count": len(gpus), "gpu_util_avg": gpu_util_avg, "gpu_util_max": gpu_util_max,
        "gpu_vram_pct_avg": gpu_vram_avg, "gpu_vram_pct_max": gpu_vram_max,
        "status": status, "alerts": alerts,
    }

    def append(path, cols, rows):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not (os.path.exists(path) and os.path.getsize(path) > 0)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)

    append(a.system_csv, SYS_COLS, [sys_row])
    gpu_rows = [{**g, "ts": ts, "run_id": man["run_id"]} for g in gpus]
    append(a.gpu_csv, GPU_COLS, gpu_rows)

    # compact summary for monitor.sh (stateful kill logic)
    print(f"status={status} phase={man['phase']} pipeline_status={man['pipeline_status']} "
          f"failed_phase={man['failed_phase']} gpu_util_max={gpu_util_max if gpu_util_max is not None else ''} "
          f"net_rx_mbps={rx_bps if rx_bps is not None else ''} "
          f"disk_max_pct={max([x for x in (dr_pct, dw_pct) if x is not None], default='')}")


if __name__ == "__main__":
    main()
