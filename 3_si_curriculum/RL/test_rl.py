"""
V3 smoke test — validates the real rl_training.py V3 config end-to-end.

Mirrors rl_training.py V3 exactly, but runs only 20 steps (~50 minutes on
4x A100-80GB, fits in a 1-hour allocation). Designed to be runnable on
easier-to-get GPUs before committing to the full 2-week run.

What this smoke test validates:
  - Reward functions compute without error (gated path alignment, smooth length penalty)
  - log_completions=True writes completions to TRL's logger
  - GenerationDumpCallback writes JSONL successfully
  - HeldOutEvalCallback runs at least twice without crashing
  - save_strategy="steps" fires at least twice (step 10, step 20)
  - DeepSpeed ZeRO-3 checkpoint gather works correctly
  - No OOM at the smaller max_completion_length=1280

Cadences are compressed vs the real run so everything fires in 20 steps:
  - eval_steps=10   (fires at step 10, 20)
  - save_steps=10   (fires at step 10, 20)
  - generation_dump_every=5  (fires at step 5, 10, 15, 20)

Usage:
    torchrun --nproc_per_node=4 test_rl_v2.py
"""

import os
import re
import json
import logging
import datetime
from collections import Counter
from typing import List

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from data_prep import preprocess_grpo_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SYSTEM_PROMPT = """A conversation between user and assistant. The user asks a single-choice Multiple Choice Question, and the assistant solves it using step-by-step reasoning. Please answer the multiple choice question by selecting only one from option A, option B, option C, option D. 

The assistant first thinks through the problem systematically, then provides the explanation and final answer. Use <think>...</think> tags for internal reasoning, then provide the explanation process and answer enclosed within <explanation> </explanation> and <answer> </answer> tags, respectively."""

TASK_SPECIFIC_INSTRUCTIONS = "Please provide complete and accurate answers with clear reasoning. The answer must only be a single letter from A, B, C, D."


# =====================================================================
#  Reward helpers — identical to rl_training.py V3
# =====================================================================

STOP_WORDS = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'can', 'this', 'that', 'these', 'those'
}


def normalize_tokens(text: str) -> List[str]:
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t and t not in STOP_WORDS]


def repetition_penalty_factor(tokens: List[str], threshold: float = 0.35) -> float:
    if not tokens:
        return 1.0
    counts = Counter(tokens)
    most_common = counts.most_common(1)[0][1]
    ratio = most_common / max(1, len(tokens))
    base = max(0.0, 1.0 - max(0.0, ratio - threshold) * 3.0)
    max_run = 1
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    run_penalty = 1.0 - max(0.0, (max_run - 3)) * 0.05
    return max(0.0, base * run_penalty)


def smooth_length_penalty(num_tokens: int, soft_start: int = 600, hard_cap: int = 1000, max_penalty: float = 1.0) -> float:
    if num_tokens <= soft_start:
        return 0.0
    if num_tokens >= hard_cap:
        return max_penalty
    return max_penalty * (num_tokens - soft_start) / (hard_cap - soft_start)


def extract_answer(text: str) -> str:
    try:
        text_clean = re.sub(r'\*+', '', text)
        match = re.search(r'Final Answer\s*[:\-]\s*([A-D])', text_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        answer_match = re.search(r'<answer>\s*([A-D])\s*</answer>', text_clean, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        if '</think>' in text_clean:
            after_think = text_clean.split('</think>')[-1]
            letters = re.findall(r'\b[A-D]\b', after_think)
            if letters:
                return letters[0].upper()
        return ""
    except Exception:
        return ""


def extract_thinking(text: str) -> str:
    try:
        return text.split("<think>")[-1].split("</think>")[0].strip()
    except IndexError:
        return ""


# =====================================================================
#  V3 Reward functions — identical to rl_training.py
# =====================================================================

_TOKENIZER_REF = None


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """V3: +1/-1 correctness + smooth token-based length penalty (600 -> 1000)."""
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_answer(r) for r in responses]

    gt_answers = []
    for ans in answer:
        letters = re.findall(r'\b[A-D]\b', ans)
        gt_answers.append(letters[-1] if letters else "")

    rewards = []
    for i, (pred, gt) in enumerate(zip(extracted, gt_answers)):
        if pred == gt and pred != "":
            reward = 1.0
        else:
            reward = -1.0

        if _TOKENIZER_REF is not None:
            num_tokens = len(_TOKENIZER_REF.encode(responses[i], add_special_tokens=False))
        else:
            num_tokens = len(responses[i].split()) * 2

        reward -= smooth_length_penalty(num_tokens, soft_start=600, hard_cap=1000, max_penalty=1.0)
        rewards.append(reward)
    return rewards


def path_alignment_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """V3: GATED on correctness. Only pays when answer is right."""
    responses = [c[0]["content"] for c in completions]
    thinkings = [extract_thinking(r) for r in responses]
    extracted = [extract_answer(r) for r in responses]

    gt_answers = []
    for ans in answer:
        letters = re.findall(r'\b[A-D]\b', ans)
        gt_answers.append(letters[-1] if letters else "")

    paths_batch = kwargs.get("paths", [None] * len(responses))

    rewards = []
    for thinking, kg_path, pred, gt in zip(thinkings, paths_batch, extracted, gt_answers):
        if pred != gt or pred == "":
            rewards.append(0.0)
            continue
        if kg_path is None:
            rewards.append(0.0)
            continue

        path_tokens = set(normalize_tokens(str(kg_path)))
        thinking_tokens_list = normalize_tokens(thinking)
        thinking_tokens_set = set(thinking_tokens_list)

        if not path_tokens:
            rewards.append(0.0)
            continue

        hits = thinking_tokens_set & path_tokens
        coverage = len(hits) / max(1, len(path_tokens))
        min_unique_hit = 1.0 if len(hits) >= 2 else 0.0
        rep_factor = repetition_penalty_factor(thinking_tokens_list)
        base_reward = (0.8 * coverage + 0.3 * min_unique_hit)
        rewards.append(min(base_reward * rep_factor, 0.8))
    return rewards


# =====================================================================
#  Callbacks — identical to rl_training.py V3
# =====================================================================

class GenerationDumpCallback(transformers.TrainerCallback):
    def __init__(self, output_dir: str, dump_every: int = 5):
        self.output_dir = output_dir
        self.dump_every = dump_every
        self.jsonl_path = os.path.join(output_dir, "training_generations.jsonl")
        self.trainer = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.dump_every != 0:
            return control
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return control
        if self.trainer is None:
            return control

        completions_data = None
        for attr in ("_last_logged_completions", "_last_completions", "_buffered_inputs"):
            if hasattr(self.trainer, attr):
                completions_data = getattr(self.trainer, attr)
                if completions_data:
                    break

        record = {
            "step": state.global_step,
            "epoch": round(float(state.epoch), 4),
            "metrics": {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float, str))},
        }

        if completions_data is not None:
            try:
                if isinstance(completions_data, dict):
                    prompts = completions_data.get("prompts") or completions_data.get("prompt")
                    completions = completions_data.get("completions") or completions_data.get("completion")
                    if prompts and completions:
                        record["sample_prompt"] = str(prompts[0])[:2000]
                        sample_comp = completions[0]
                        if isinstance(sample_comp, list) and sample_comp:
                            sample_comp = sample_comp[0].get("content", str(sample_comp[0]))
                        record["sample_completion"] = str(sample_comp)[:4000]
            except Exception as e:
                record["completion_dump_error"] = str(e)

        try:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logging.warning(f"GenerationDumpCallback write failed: {e}")

        return control


class HeldOutEvalCallback(transformers.TrainerCallback):
    def __init__(self, eval_dataset, eval_every: int = 10, max_new_tokens: int = 1280):
        self.eval_dataset = eval_dataset
        self.eval_every = eval_every
        self.max_new_tokens = max_new_tokens
        self.trainer = None
        self.history = []
        self.history_path = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every != 0:
            return control
        if self.trainer is None:
            return control

        rank = int(os.environ.get("LOCAL_RANK", 0))
        logging.info(f"[Rank {rank}] SMOKE TEST EVAL at step {state.global_step}")

        model = self.trainer.model
        tokenizer = self.trainer.processing_class
        model.eval()

        correct = 0
        total = 0
        malformed = 0

        try:
            with torch.no_grad():
                for example in self.eval_dataset:
                    prompt = example.get("prompt")
                    gt_answer = example.get("answer", "")
                    if not prompt:
                        continue

                    if isinstance(prompt, list):
                        prompt_text = tokenizer.apply_chat_template(
                            prompt, tokenize=False, add_generation_prompt=True
                        )
                    else:
                        prompt_text = str(prompt)

                    inputs = tokenizer(
                        prompt_text, return_tensors="pt", truncation=True, max_length=4096
                    ).to(model.device)

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    generated = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )

                    pred = extract_answer(generated)
                    gt_letters = re.findall(r'\b[A-D]\b', gt_answer)
                    gt = gt_letters[-1] if gt_letters else ""

                    if pred == "":
                        malformed += 1
                    elif pred == gt and gt != "":
                        correct += 1
                    total += 1

            accuracy = correct / max(1, total)
            logging.info(
                f"[SMOKE EVAL step={state.global_step}] accuracy={accuracy:.4f} "
                f"({correct}/{total}) malformed={malformed}"
            )
            if rank == 0:
                self.history.append({
                    "step": state.global_step,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "malformed": malformed,
                })
                if self.history_path:
                    try:
                        with open(self.history_path, "w") as f:
                            json.dump(self.history, f, indent=2)
                    except Exception as e:
                        logging.warning(f"Failed to write eval history: {e}")
        except Exception as e:
            logging.error(f"[Rank {rank}] Smoke test eval failed: {e}")
        finally:
            model.train()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return control


# =====================================================================
#  Main
# =====================================================================

def main():
    global _TOKENIZER_REF

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=600))

    # ---- Paths (set via env vars) ----
    model_path = os.environ.get("MODEL_NAME", "")
    dataset_path = os.environ.get("DATASET_PATH", "")
    _output_base = os.environ.get("OUTPUT_BASE", "")
    if not model_path or not dataset_path or not _output_base:
        raise ValueError("Set MODEL_NAME, DATASET_PATH, and OUTPUT_BASE env vars before running the smoke test.")
    output_dir = os.path.join(_output_base, "_v3_smoke_test")
    ds_config = os.path.abspath("deepspeed_config.json")

    # ---- Dataset with held-out split (smaller eval set for smoke test) ----
    logging.info("Loading full dataset...")
    full_dataset = preprocess_grpo_dataset(
        dataset_path=dataset_path,
        split="train",
        chunk_size=1000,
        enable_thinking=True,
        system_prompt=SYSTEM_PROMPT,
        task_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    )
    logging.info(f"Full dataset: {len(full_dataset)} examples")

    # Smaller eval split for smoke test: 20 examples instead of 100
    split = full_dataset.train_test_split(test_size=20, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logging.info(f"Train: {len(train_dataset)}, Eval (held-out): {len(eval_dataset)}")

    # ---- Model ----
    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    _TOKENIZER_REF = tokenizer

    # ---- Callbacks (compressed cadences so everything fires in 20 steps) ----
    os.makedirs(output_dir, exist_ok=True)
    gen_dump_cb = GenerationDumpCallback(output_dir=output_dir, dump_every=5)
    eval_cb = HeldOutEvalCallback(eval_dataset=eval_dataset, eval_every=10, max_new_tokens=1280)
    eval_cb.history_path = os.path.join(output_dir, "eval_history.json")

    # ---- GRPO config: IDENTICAL to V3 rl_training.py except max_steps=20 and compressed cadences ----
    training_args = GRPOConfig(
        deepspeed=ds_config,

        # V3 values
        learning_rate=2e-6,
        beta=0.08,                                   # V3: was 0.04
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,

        bf16=True,

        num_generations=4,
        max_completion_length=1280,                  # V3: was 1792
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15,

        optim="adamw_torch",

        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        num_train_epochs=3,

        # THE ONLY DIFFERENCE: cap at 20 steps
        max_steps=20,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # V3: step-based saving. Compressed to every 10 steps for smoke test.
        save_strategy="steps",
        save_steps=10,
        save_total_limit=4,

        # V3: TRL built-in completion logging
        log_completions=True,
        num_completions_to_print=2,

        logging_steps=1,
        max_grad_norm=1.0,
        output_dir=output_dir,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func, path_alignment_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[gen_dump_cb, eval_cb],
    )

    gen_dump_cb.trainer = trainer
    eval_cb.trainer = trainer

    logging.info("=" * 70)
    logging.info("V3 SMOKE TEST — 20 steps with full V3 config")
    logging.info("  beta=0.08, lr=2e-6, num_gen=4, max_completion=1280")
    logging.info("  gated path_alignment, smooth length penalty (600->1000)")
    logging.info("  save_steps=10, eval_steps=10, dump_every=5")
    logging.info("")
    logging.info("Expected events during smoke test:")
    logging.info("  - generation dump at steps 5, 10, 15, 20")
    logging.info("  - held-out eval at steps 10, 20")
    logging.info("  - checkpoint save at steps 10, 20 (step 20 via final save)")
    logging.info("")
    logging.info("What to look for in output:")
    logging.info("  1. No crashes in reward functions, callbacks, or eval")
    logging.info("  2. training_generations.jsonl exists and has entries")
    logging.info("  3. eval_history.json exists with 2 entries")
    logging.info("  4. checkpoint-10 directory exists with model weights")
    logging.info("  5. Reward values reasonable (not NaN, not all -1, not all +1)")
    logging.info("  6. Completion lengths stay under ~700 tokens")
    logging.info("=" * 70)

    trainer.train()

    # ---- Peak memory report ----
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(local_rank) / 1e9
        reserved = torch.cuda.memory_reserved(local_rank) / 1e9
        peak = torch.cuda.max_memory_allocated(local_rank) / 1e9
        total = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        logging.info("=" * 70)
        logging.info(f"[GPU {local_rank}] Final: alloc={alloc:.1f} GB, reserved={reserved:.1f} GB")
        logging.info(f"[GPU {local_rank}] PEAK: {peak:.1f} / {total:.0f} GB ({peak/total*100:.0f}%)")
        logging.info("=" * 70)

    # ---- Final save check ----
    logging.info("Testing final checkpoint save (ZeRO-3 gather)...")
    trainer.save_model(output_dir)
    trainer.accelerator.wait_for_everyone()

    if local_rank == 0:
        files = os.listdir(output_dir)
        has_weights = any(f.endswith(('.safetensors', '.bin')) for f in files)
        has_gen_dump = os.path.exists(os.path.join(output_dir, "training_generations.jsonl"))
        has_eval_history = os.path.exists(os.path.join(output_dir, "eval_history.json"))
        step10_ckpt = os.path.exists(os.path.join(output_dir, "checkpoint-10"))

        logging.info("=" * 70)
        logging.info("V3 SMOKE TEST RESULTS:")
        logging.info(f"  Final weights saved:        {has_weights}")
        logging.info(f"  training_generations.jsonl: {has_gen_dump}")
        logging.info(f"  eval_history.json:          {has_eval_history}")
        logging.info(f"  checkpoint-10 exists:       {step10_ckpt}")
        if has_weights and has_gen_dump and has_eval_history and step10_ckpt:
            logging.info("  STATUS: PASS — ready for full run")
        else:
            logging.info("  STATUS: FAIL — investigate missing artifacts before full run")
        logging.info("=" * 70)


if __name__ == "__main__":
    main()