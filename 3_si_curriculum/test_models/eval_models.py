import os
import sys
import json
import glob
import re
import time
import argparse
import gc
import math
import random
from pathlib import Path
import torch
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pipeline config loader (repo root, 2 levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _tokenizer_compat  # noqa: F401, E402  # vLLM 0.7.3 + Qwen3 chat-template shim
from pipeline_config import render_prompt  # noqa: E402

# ==========================================
# CONFIGURATION - EDIT PATHS HERE
# ==========================================

# Set these via CLI --model_paths or MODEL_PATH_1..5 env vars
MODEL_PATH_1 = os.environ.get("MODEL_PATH_1", "none")
MODEL_PATH_2 = os.environ.get("MODEL_PATH_2", "none")
MODEL_PATH_3 = os.environ.get("MODEL_PATH_3", "none")
MODEL_PATH_4 = "none"
MODEL_PATH_5 = "none"



HF_MODELS = {}
for i, path in enumerate([MODEL_PATH_1, MODEL_PATH_2, MODEL_PATH_3, MODEL_PATH_4, MODEL_PATH_5], 1):
    if path and path.lower() != "none":
        name = os.path.basename(path.strip("/")) if "/" in path else f"Model_{i}"
        HF_MODELS[name] = path

# Sourced from prompts/eval_models.yaml. SYSTEM_PROMPT / GEMINI_SYSTEM_PROMPT
# inject {{domain_expert_role}} from domains/<SI_DOMAIN>.yaml (defaults to
# "expert neuroscientist", byte-identical to prior hardcoded constants).
# RECOVERY_PROMPT keeps `{question}` / `{reasoning}` placeholders intact so
# existing `.format(question=..., reasoning=...)` call sites work unchanged.
# See docs/PROMPT_MIGRATION.md item #12.
_eval_prompts = render_prompt("eval_models")
SYSTEM_PROMPT = _eval_prompts["system"]
GEMINI_SYSTEM_PROMPT = _eval_prompts["gemini_system"]
RECOVERY_PROMPT = _eval_prompts["recovery"]

DEFAULT_INPUT_DIR = os.environ.get("EVAL_INPUT_DIR", "")
DEFAULT_OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", "eval_results")

# ==========================================
# SUBSET MODE - Set to True to only run random questions
# ==========================================
SUBSET_MODE = False
SUBSET_SIZE = 3000
SUBSET_SEED = 6

MAX_NEW_TOKENS = 4096

RECOVERY_MAX_TOKENS = 512

# ==========================================
# GEMINI CONFIG
# ==========================================
GEMINI_MODEL = "gemini-3.1-pro-preview"

# Derive a clean tag for keys/filenames: "gemini-3.1-pro-preview" -> "gemini_3.1_pro"
GEMINI_MODEL_TAG = re.sub(r"-(preview|exp|experimental)$", "", GEMINI_MODEL).replace("-", "_")

GEMINI_THINKING_BUDGET = int(MAX_NEW_TOKENS * 3 / 4)

GEMINI_RECOVERY_THINKING_BUDGET = 256

def log(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

def log_separator(title: str = ""):
    log(f"\n{'='*60}")
    if title:
        log(f"  {title}")
        log(f"{'='*60}")

def clean_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def extract_ground_truth(text_block: str) -> str:
    match = re.search(r"<Answer>[:\s]*([A-D])[:\s]*</Answer>", text_block, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def get_text_outside_think(text: str) -> str:
    """Return only the text that lives OUTSIDE <think>...</think> blocks."""
    if not text:
        return ""
    outside = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    outside = re.sub(r"<think>.*", "", outside, flags=re.DOTALL)
    return outside.strip()


def extract_model_answer(response_text: str) -> str:
    if not response_text:
        return None

    visible = get_text_outside_think(response_text)
    matches = list(re.finditer(r"<Answer>\s*([A-D])\s*</Answer>", visible))
    if matches:
        return matches[-1].group(1).upper()

    return None


def extract_question_only(full_text: str) -> str:
    match = re.search(r"(<Question>.*?</Options>)", full_text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        raw = full_text
    raw = re.sub(r"</?Question>", "", raw)
    raw = re.sub(r"</?Options>", "", raw)
    raw = re.sub(r"^\s*\[", "", raw)
    raw = re.sub(r"\]\s*\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return raw

class GeminiHandler:
    def __init__(self, api_key, model_name=None, thinking_budget=None):
        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or GEMINI_MODEL
        self.thinking_budget = thinking_budget if thinking_budget is not None else GEMINI_THINKING_BUDGET

        self.last_call_time = 0
        self.min_interval = 1.0

        # Detect if SDK supports thinking_level (newer versions)
        self._supports_thinking_level = self._check_thinking_level_support()

        log(f"  Gemini SDK: google-genai (new)")
        log(f"  Model: {self.model_name}")
        log(f"  Total token budget: {MAX_NEW_TOKENS}")
        if self._supports_thinking_level:
            log(f"  Thinking: thinking_level=MEDIUM (SDK supports thinking_level)")
        else:
            log(f"  Thinking: thinking_budget={self.thinking_budget} (SDK fallback, upgrade google-genai for thinking_level)")
        log(f"  Recovery: {RECOVERY_MAX_TOKENS} max tokens, thinking=OFF")

    def _check_thinking_level_support(self):
        """Check if the installed SDK version supports thinking_level parameter."""
        types = self.types
        try:
            types.ThinkingConfig(thinking_level="MEDIUM")
            return True
        except (TypeError, Exception):
            return False

    def _make_thinking_config(self, level="MEDIUM"):
        """Create ThinkingConfig with thinking_level if supported, else fall back to thinking_budget."""
        types = self.types
        if level == "NONE" or level == "OFF":
            if self._supports_thinking_level:
                try:
                    return types.ThinkingConfig(thinking_level="LOW")
                except Exception:
                    return types.ThinkingConfig(thinking_budget=GEMINI_RECOVERY_THINKING_BUDGET)
            else:
                return types.ThinkingConfig(thinking_budget=GEMINI_RECOVERY_THINKING_BUDGET)
        else:
            if self._supports_thinking_level:
                return types.ThinkingConfig(thinking_level=level)
            else:
                # Map levels to budget approximations
                budget_map = {"LOW": MAX_NEW_TOKENS // 4, "MEDIUM": MAX_NEW_TOKENS // 2, "HIGH": int(MAX_NEW_TOKENS * 3 / 4)}
                budget = budget_map.get(level, self.thinking_budget)
                return types.ThinkingConfig(thinking_budget=budget)

    def _extract_text(self, response):
        text_parts = []
        candidates = getattr(response, 'candidates', None)
        if not candidates:
            return ""
        content = getattr(candidates[0], 'content', None)
        if not content:
            return ""
        parts = getattr(content, 'parts', None)
        if not parts:
            return ""
        for part in parts:
            if getattr(part, 'thought', False):
                continue
            if part.text:
                text_parts.append(part.text)
        return "\n".join(text_parts)

    def _extract_thinking(self, response):
        thinking_parts = []
        candidates = getattr(response, 'candidates', None)
        if not candidates:
            return ""
        content = getattr(candidates[0], 'content', None)
        if not content:
            return ""
        parts = getattr(content, 'parts', None)
        if not parts:
            return ""
        for part in parts:
            if getattr(part, 'thought', False) and part.text:
                thinking_parts.append(part.text)
        return "\n".join(thinking_parts)

    def _get_finish_reason(self, response):
        try:
            if response.candidates:
                return response.candidates[0].finish_reason
        except Exception:
            pass
        return None

    def _recovery_call(self, question, reasoning):
        types = self.types
        prompt = RECOVERY_PROMPT.format(question=question, reasoning=reasoning)

        thinking_cfg = self._make_thinking_config("NONE")

        config = types.GenerateContentConfig(
            temperature=0.6,
            max_output_tokens=RECOVERY_MAX_TOKENS,
            thinking_config=thinking_cfg,
        )

        time.sleep(1.0)
        self.last_call_time = time.time()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return self._extract_text(response)

    def generate(self, prompt, question_text=None):
        consecutive_429s = 0
        MAX_CONSECUTIVE_429 = 10
        DEEP_SLEEP_DURATION = 43200
        types = self.types

        thinking_cfg = self._make_thinking_config("MEDIUM")

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.6,
            max_output_tokens=MAX_NEW_TOKENS,
            thinking_config=thinking_cfg,
        )

        while True:
            time_since_last = time.time() - self.last_call_time
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            try:
                self.last_call_time = time.time()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                visible = self._extract_text(response)
                thinking = self._extract_thinking(response)
                finish_reason = self._get_finish_reason(response)

                needs_recovery = False
                if not visible.strip() and str(finish_reason) == "FinishReason.MAX_TOKENS":
                    needs_recovery = True
                elif visible.strip() and extract_model_answer(visible) is None:
                    needs_recovery = True

                if needs_recovery and question_text:
                    # Use visible text first (same as HF), fall back to thinking trace only if empty
                    reasoning_context = visible.strip() if visible.strip() else thinking
                    if not reasoning_context.strip():
                        reasoning_context = "(no reasoning available)"

                    log(f"  [Gemini] Recovery call (finish_reason={finish_reason}, "
                        f"thinking_len={len(thinking)}, visible_len={len(visible)})")
                    try:
                        recovery_resp = self._recovery_call(question_text, reasoning_context)
                        recovery_answer = extract_model_answer(recovery_resp)
                        if recovery_answer:
                            log(f"  [Gemini] Recovery succeeded -> {recovery_answer}")
                            return f"[RECOVERED from thinking overflow]\n{recovery_resp}", thinking
                        else:
                            log(f"  [Gemini] Recovery failed to parse: {recovery_resp[:100]}")
                    except Exception as e:
                        log(f"  [Gemini] Recovery call error: {e}")

                return visible, thinking

            except Exception as e:
                error_str = str(e).lower()
                if "resource_exhausted" in error_str or "quota" in error_str:
                    log(f"\n[!!!] DAILY LIMIT HIT. Deep Sleep 12h...")
                    time.sleep(DEEP_SLEEP_DURATION)
                    consecutive_429s = 0
                    continue
                elif "429" in error_str:
                    consecutive_429s += 1
                    log(f"  [Rate Limit]: Streak {consecutive_429s}")
                    if consecutive_429s >= MAX_CONSECUTIVE_429:
                        log(f"\n[!!!] Soft Ban. Deep Sleep 12h...")
                        time.sleep(DEEP_SLEEP_DURATION)
                        consecutive_429s = 0
                    else:
                        time.sleep(60.0 * consecutive_429s)
                    continue
                else:
                    log(f"  [Gemini Error]: {e}")
                    time.sleep(5)
                    return "ERROR_API_FAIL", ""

class HFHandler:
    def __init__(self, model_path):
        BASE_TOKENIZER_PATH = os.environ.get("BASE_TOKENIZER_PATH", model_path)

        self.model_path = model_path
        self._is_deepseek = "deepseek" in model_path.lower()

        log(f"  Loading Tokenizer from: {model_path}")
        log(f"  DeepSeek detected: {self._is_deepseek}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except (AttributeError, TypeError, OSError, ValueError) as e:
            log(f"    [WARNING] Failed to load tokenizer from {model_path}")
            log(f"    [ERROR DETAIL] {e}")
            log(f"    [FALLBACK] Loading tokenizer from base: {BASE_TOKENIZER_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_PATH, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log(f"  Loading Model to GPU (bf16): {model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        log(f"  Model loaded successfully: {model_path}")

    def _build_chat_input(self, messages, enable_thinking=True):
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            ).to(self.model.device)
            return input_ids
        except TypeError:
            pass

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            return input_ids
        except Exception:
            pass

        if messages and messages[0].get("role") == "system":
            text = messages[0]["content"] + "\n\n"
            text += "\n".join(m["content"] for m in messages[1:])
        else:
            text = "\n".join(m["content"] for m in messages)
        text += "\nAnswer:"
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    def _get_terminators(self):
        terminators = []
        if self.tokenizer.eos_token_id is not None:
            terminators.append(self.tokenizer.eos_token_id)
        for token in ["<|eot_id|>", "<|im_end|>"]:
            t_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(t_id, int) and t_id != self.tokenizer.unk_token_id:
                terminators.append(t_id)
        return terminators

    def _recovery_call(self, question, reasoning):
        prompt = RECOVERY_PROMPT.format(question=question, reasoning=reasoning)
        messages = [
            {"role": "user", "content": prompt}
        ]

        input_ids = self._build_chat_input(messages, enable_thinking=False)
        attention_mask = torch.ones_like(input_ids)
        terminators = self._get_terminators()

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=RECOVERY_MAX_TOKENS,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def generate(self, prompt, question_text=None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        input_ids = self._build_chat_input(messages, enable_thinking=True)
        attention_mask = torch.ones_like(input_ids)
        terminators = self._get_terminators()

        try:
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )
            response = outputs[0][input_ids.shape[-1]:]
            decoded = self.tokenizer.decode(response, skip_special_tokens=True)

            parsed_ans = extract_model_answer(decoded)

            # Extract thinking for HF models (inside <think> tags)
            think_match = re.search(r"<think>(.*?)</think>", decoded, re.DOTALL)
            hf_thinking = think_match.group(1).strip() if think_match else ""

            if parsed_ans is None and question_text:
                visible_reasoning = get_text_outside_think(decoded).strip()
                reasoning_context = visible_reasoning if visible_reasoning else "(no reasoning available)"
                log(f"  [HF] Recovery call (no answer tag outside <think>, "
                    f"response_len={len(decoded)}, visible_len={len(visible_reasoning)}, "
                    f"is_deepseek={self._is_deepseek})")
                try:
                    recovery_resp = self._recovery_call(question_text, reasoning_context)
                    recovery_answer = extract_model_answer(recovery_resp)
                    if recovery_answer:
                        log(f"  [HF] Recovery succeeded -> {recovery_answer}")
                        return f"[RECOVERED from parse failure]\n{recovery_resp}", hf_thinking
                    else:
                        log(f"  [HF] Recovery failed to parse: {recovery_resp[:100]}")
                except Exception as e:
                    log(f"  [HF] Recovery call error: {e}")

            return decoded, hf_thinking
        except Exception as e:
            log(f"      [Generation error]: {e}")
            clean_gpu_memory()
            return "ERROR_GEN_FAIL", ""

# ==========================================
# MAIN EXECUTION
# ==========================================

def get_file_chunk(all_files):
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    task_count = os.getenv("SLURM_ARRAY_TASK_COUNT")

    if not task_id or not task_count:
        log(f"Running in Standard Mode (Processing all {len(all_files)} files).")
        return all_files

    task_id = int(task_id)
    task_count = int(task_count)

    total_files = len(all_files)
    chunk_size = math.ceil(total_files / task_count)

    start_idx = task_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)

    log(f"Running in Array Mode (Job {task_id}/{task_count}).")
    log(f"Processing slice {start_idx} to {end_idx} (Total: {end_idx - start_idx} files).")

    if start_idx >= total_files:
        return []

    return all_files[start_idx:end_idx]

def process_pipeline(args):
    log_separator("EVAL PIPELINE START")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Python: {sys.executable}")
    log(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log("[WARNING] No GPU detected!")

    run_type = "mixed"
    if args.no_gemini and not args.no_hf:
        run_type = "hf"
    elif args.no_hf and not args.no_gemini:
        run_type = "gemini"

    file_tag = GEMINI_MODEL_TAG if run_type == "gemini" else run_type

    log(f"Run Mode: {run_type.upper()}")
    log(f"File Tag: {file_tag}")
    log(f"MAX_NEW_TOKENS (total budget): {MAX_NEW_TOKENS}")
    log(f"RECOVERY_MAX_TOKENS: {RECOVERY_MAX_TOKENS}")
    log(f"GEMINI_MODEL: {GEMINI_MODEL}")
    log(f"GEMINI_THINKING_LEVEL: MEDIUM")
    log(f"SUBSET_MODE: {SUBSET_MODE} (size={SUBSET_SIZE}, seed={SUBSET_SEED})")
    log(f"Input: {args.input_dir}")
    log(f"Output: {args.output_dir}")
    log(f"Models to evaluate: {list(HF_MODELS.keys()) if args.use_hf else []}")

    all_files = []
    if os.path.isfile(args.input_dir):
        all_files = [args.input_dir]
    elif os.path.isdir(args.input_dir):
        all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    else:
        all_files = sorted(glob.glob(args.input_dir))

    if not all_files:
        log(f"[ERROR] No files found at {args.input_dir}")
        return

    log(f"Found {len(all_files)} input file(s).")

    files_to_process = get_file_chunk(all_files)
    if not files_to_process:
        log("No files assigned to this job slice.")
        return

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_data = {}
    for fpath in files_to_process:
        fname = os.path.basename(fpath)

        # Explicit checkpoint path — no glob guessing
        task_id = os.getenv("SLURM_ARRAY_TASK_ID", "single")
        ckpt_name = f"ckpt_{task_id}_{file_tag}_{fname}"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        if os.path.isfile(ckpt_path):
            # Checkpoint exists — load it directly (it IS the full data with responses baked in)
            log(f"  [CHECKPOINT FOUND] {ckpt_path}")
            try:
                with open(ckpt_path, 'r') as f:
                    batch_data[fpath] = json.load(f)
                # Count how many already have responses for each model we'll run
                for mname in [GEMINI_MODEL_TAG] + list(HF_MODELS.keys()):
                    answered = sum(1 for item in batch_data[fpath] if f"response_{mname}" in item)
                    if answered > 0:
                        log(f"    Found {answered}/{len(batch_data[fpath])} items with response_{mname}")
            except Exception as e:
                log(f"    [Error] Corrupt checkpoint: {e}")
                log(f"    [FALLBACK] Loading original file: {fpath}")
                with open(fpath, 'r') as f:
                    batch_data[fpath] = json.load(f)
        else:
            # No checkpoint — load original file
            log(f"  [NO CHECKPOINT] Loading original: {fpath}")
            try:
                with open(fpath, 'r') as f:
                    batch_data[fpath] = json.load(f)
            except Exception as e:
                log(f"[ERROR] Reading {fpath}: {e}")
                continue

    if SUBSET_MODE:
        log_separator(f"SUBSET MODE: Sampling {SUBSET_SIZE} questions per file (seed={SUBSET_SEED})")
        for fpath in batch_data:
            items = batch_data[fpath]
            original_count = len(items)
            if len(items) > SUBSET_SIZE:
                rng = random.Random(SUBSET_SEED)
                batch_data[fpath] = rng.sample(items, SUBSET_SIZE)
                log(f"  {os.path.basename(fpath)}: {original_count} -> {len(batch_data[fpath])} questions")
            else:
                log(f"  {os.path.basename(fpath)}: {original_count} questions (less than subset size, using all)")

    models = []
    if args.use_gemini:
        models.append((GEMINI_MODEL_TAG, "Gemini-API"))
    if args.use_hf:
        for name, path in HF_MODELS.items():
            models.append((name, path))

    log(f"\nModels to run: {[m[0] for m in models]}")

    for model_name, model_path in models:
        log_separator(f"Loading Model: {model_name}")
        log(f"  Path: {model_path}")

        handler = None
        try:
            if model_name.startswith("gemini"):
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    log("  [SKIP] Gemini: GEMINI_API_KEY not set.")
                    continue
                handler = GeminiHandler(
                    api_key=api_key,
                    model_name=args.gemini_model if hasattr(args, 'gemini_model') and args.gemini_model else None,
                    thinking_budget=args.thinking_budget if hasattr(args, 'thinking_budget') and args.thinking_budget is not None else None,
                )
            else:
                handler = HFHandler(model_path)
            log(f"  Handler initialized successfully for {model_name}")
        except Exception as e:
            log(f"  [FATAL] Failed to init {model_name}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            continue

        for fpath, items in batch_data.items():
            fname = os.path.basename(fpath)
            total_items = len(items)
            log_separator(f"Evaluating: {model_name} on {fname} ({total_items} questions)")

            processed_count = 0
            correct_count = 0
            error_count = 0
            skipped_count = 0
            recovered_count = 0
            start_time = time.time()

            # --- Resume skip phase: count previously answered items ---
            for idx, item in enumerate(items):
                if f"response_{model_name}" in item:
                    processed_count += 1
                    skipped_count += 1
                    if item.get(f"correctness_{model_name}") == "yes":
                        correct_count += 1

            if skipped_count > 0:
                skip_acc = (correct_count / skipped_count * 100) if skipped_count > 0 else 0
                log(f"  [RESUMED] Skipping {skipped_count} already-answered items "
                    f"(prior acc: {correct_count}/{skipped_count} = {skip_acc:.1f}%)")
            else:
                log(f"  [FRESH] Starting from question 0")

            # --- Main eval loop: process remaining items ---
            for idx, item in enumerate(items):
                # Skip already answered
                if f"response_{model_name}" in item:
                    continue

                # Process new item
                full_text = item.get("question_and_explanation", "")
                q_text = extract_question_only(full_text)
                ground_truth = extract_ground_truth(full_text)

                if not ground_truth:
                    continue

                raw_resp, thinking = handler.generate(q_text, question_text=q_text)
                parsed_ans = extract_model_answer(raw_resp)

                was_recovered = raw_resp.startswith("[RECOVERED")
                if was_recovered:
                    recovered_count += 1

                if parsed_ans is None:
                    verdict = "error"
                    error_count += 1
                elif parsed_ans == ground_truth:
                    verdict = "yes"
                    correct_count += 1
                else:
                    verdict = "no"

                item[f"response_{model_name}"] = raw_resp
                item[f"parsed_{model_name}"] = parsed_ans
                item[f"correctness_{model_name}"] = verdict
                if thinking:
                    item[f"thinking_{model_name}"] = thinking

                processed_count += 1
                new_count = processed_count - skipped_count

                # Save checkpoint after EVERY new item (each Gemini call is expensive)
                task_id = os.getenv("SLURM_ARRAY_TASK_ID", "single")
                ckpt_name = f"ckpt_{task_id}_{file_tag}_{fname}"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

                with open(ckpt_path, 'w') as f:
                    json.dump(items, f, indent=4)

                # Log stats every 5 new items
                if new_count % 5 == 0:
                    elapsed = time.time() - start_time
                    acc = (correct_count / processed_count * 100) if processed_count > 0 else 0

                    log(f"    --- [{processed_count}/{total_items}] Total Acc: {correct_count}/{processed_count} ({acc:.1f}%) | "
                        f"New: {new_count} | Errors: {error_count} | Recovered: {recovered_count} | {elapsed:.0f}s | Checkpoint saved ---")

            # Final Results Logging
            elapsed = time.time() - start_time
            answered_this_run = processed_count - skipped_count
            acc = (correct_count / processed_count * 100) if processed_count > 0 else 0
            
            log_separator(f"RESULTS: {model_name} on {fname}")
            log(f"  Total questions: {total_items}")
            log(f"  Answered (this run): {answered_this_run}")
            log(f"  Skipped (already done): {skipped_count}")
            log(f"  Total Correct: {correct_count}/{processed_count} ({acc:.1f}%)")
            log(f"  Recovered (this run): {recovered_count}")
            log(f"  Parse errors (this run): {error_count}")
            log(f"  Time: {elapsed:.1f}s")

        log(f"\nUnloading model: {model_name}")
        del handler
        clean_gpu_memory()

    os.makedirs(args.output_dir, exist_ok=True)

    for fpath, items in batch_data.items():
        fname = os.path.basename(fpath)
        task_id = os.getenv("SLURM_ARRAY_TASK_ID", "single")

        subset_tag = f"_subset{SUBSET_SIZE}" if SUBSET_MODE else ""
        out_name = f"eval_{task_id}_{file_tag}{subset_tag}_{fname}"
        out_path = os.path.join(args.output_dir, out_name)

        with open(out_path, 'w') as f:
            json.dump(items, f, indent=4)
        log(f"Saved Final: {out_path}")

    log_separator("PIPELINE COMPLETE")
    log(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no_gemini", action="store_true")
    parser.add_argument("--no_hf", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--gemini_model", type=str, default=None,
                        help="Override Gemini model (e.g. gemini-2.5-flash, gemini-3.1-pro-preview)")
    parser.add_argument("--thinking_budget", type=int, default=None,
                        help="Thinking token budget for Gemini 2.5 models (0=off, -1=dynamic, 1024-8192 recommended)")

    args = parser.parse_args()
    args.use_gemini = not args.no_gemini
    args.use_hf = not args.no_hf

    if args.max_tokens is not None:
        MAX_NEW_TOKENS = args.max_tokens
        if args.thinking_budget is None:
            GEMINI_THINKING_BUDGET = int(MAX_NEW_TOKENS * 3 / 4)

    if args.thinking_budget is not None:
        GEMINI_THINKING_BUDGET = args.thinking_budget

    log(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Arguments: {args}")

    process_pipeline(args)