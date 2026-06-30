#!/usr/bin/env python3
"""In-venv pre-flight probe for scripts/preflight.sh.

Run INSIDE a specific venv (the orchestrator activates it first). Checks imports, the CUDA
context, torch-vs-driver version compatibility, VRAM, and flash_attn; with --deep it also does
live API reachability (Gemini, the OpenAI-compatible pair-check endpoint, Hugging Face).

Prints one indented line per check (✓/⚠/✗); exits 1 if any check FAILS (warnings don't fail).
"""
import argparse
import importlib
import importlib.util
import os
import re
import subprocess
import sys

_FAILS = 0


def ok(m):
    print(f"      ✓ {m}")


def warn(m):
    print(f"      ⚠ {m}")


def fail(m):
    global _FAILS
    _FAILS += 1
    print(f"      ✗ {m}")


def check_imports(mods):
    # find_spec checks INSTALLED-ness without executing the module — importing vllm/torch for
    # real is slow and would tax every pipeline start. torch is import-tested by check_cuda;
    # flash_attn (ABI-sensitive) is import-tested by check_flash_attn.
    for mod in mods:
        if not mod:
            continue
        try:
            if importlib.util.find_spec(mod) is not None:
                ok(f"{mod} installed")
            else:
                fail(f"{mod} NOT installed in this venv")
        except Exception as e:
            fail(f"{mod} import-check error: {type(e).__name__}: {str(e)[:80]}")


def _driver_max_cuda():
    """The driver's max supported CUDA, parsed from `nvidia-smi`'s 'CUDA Version: X.Y'."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def check_cuda(vram_min):
    try:
        import torch
    except Exception as e:
        fail(f"torch import failed: {e}")
        return
    tcuda = torch.version.cuda
    dmax = _driver_max_cuda()
    if torch.cuda.is_available():
        ok(f"torch.cuda available (torch CUDA {tcuda}, driver max {dmax or '?'})")
        if vram_min and vram_min > 0:
            try:
                gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gb + 1.0 >= vram_min:
                    ok(f"VRAM {gb:.0f} GB >= profile min {vram_min:g} GB")
                else:
                    warn(f"VRAM {gb:.0f} GB < profile min {vram_min:g} GB (may OOM)")
            except Exception as e:
                warn(f"VRAM check skipped: {e}")
    else:
        # Diagnose: a too-new venv torch (driver too old) vs a node mis-map (redeploy).
        diag = ""
        try:
            if tcuda and dmax:
                if tuple(map(int, tcuda.split("."))) > tuple(map(int, dmax.split("."))):
                    diag = f" — venv torch built for CUDA {tcuda} > driver max {dmax} (driver too old)"
                else:
                    diag = f" — torch CUDA {tcuda} <= driver {dmax}; likely GPU mis-map/node issue (redeploy)"
        except Exception:
            pass
        fail(f"torch.cuda NOT available (torch CUDA {tcuda}, driver max {dmax}){diag}")


def check_flash_attn():
    try:
        importlib.import_module("flash_attn")
        ok("import flash_attn")
    except Exception as e:
        fail(f"import flash_attn: {type(e).__name__}: {str(e)[:90]}")


def ping_gemini():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        warn("gemini ping skipped: no GOOGLE_API_KEY/GEMINI_API_KEY")
        return
    try:
        from google import genai
        client = genai.Client(api_key=key)
        next(iter(client.models.list()), None)
        ok("gemini API reachable")
    except Exception as e:
        fail(f"gemini ping: {type(e).__name__}: {str(e)[:90]}")


def ping_openai(base_url, key_env, model):
    key = os.environ.get(key_env or "OPENAI_API_KEY")
    if not key:
        warn(f"openai ping skipped: ${key_env or 'OPENAI_API_KEY'} unset")
        return
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url or None, api_key=key)
        models = {m.id for m in client.models.list().data}
        if model and model not in models:
            warn(f"openai reachable but model '{model}' not in the endpoint's model list")
        else:
            ok(f"openai endpoint reachable ({base_url or 'api.openai.com'}, model {model or 'any'})")
    except Exception as e:
        fail(f"openai ping: {type(e).__name__}: {str(e)[:90]}")


def ping_hf():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        warn("hf ping skipped: no HF_TOKEN")
        return
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami(token=token)
        ok(f"hf token valid (user {who.get('name', '?')})")
    except Exception as e:
        fail(f"hf ping: {type(e).__name__}: {str(e)[:90]}")


def main():
    ap = argparse.ArgumentParser(description="in-venv pre-flight probe")
    ap.add_argument("--imports", default="", help="comma-separated modules to import-check")
    ap.add_argument("--cuda", action="store_true", help="check torch.cuda + version compat")
    ap.add_argument("--vram-min", type=float, default=0, help="warn if VRAM below this (GB)")
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--deep", action="store_true", help="live API reachability pings")
    ap.add_argument("--ping", default="", help="comma list: gemini,openai,hf")
    ap.add_argument("--openai-base-url", default="")
    ap.add_argument("--openai-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--openai-model", default="")
    args = ap.parse_args()

    if args.imports:
        check_imports([m.strip() for m in args.imports.split(",")])
    if args.cuda:
        check_cuda(args.vram_min)
    if args.flash_attn:
        check_flash_attn()
    if args.deep:
        pings = {p.strip() for p in args.ping.split(",") if p.strip()}
        if "gemini" in pings:
            ping_gemini()
        if "openai" in pings:
            ping_openai(args.openai_base_url, args.openai_key_env, args.openai_model)
        if "hf" in pings:
            ping_hf()

    sys.exit(1 if _FAILS else 0)


if __name__ == "__main__":
    main()
