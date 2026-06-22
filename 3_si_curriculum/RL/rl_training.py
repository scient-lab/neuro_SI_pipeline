"""
Reinforcement Learning training using GRPO (Group Relative Policy Optimization).
V4 changes vs V3:
  (1) path_alignment coverage computed only over first 400 tokens of thinking
  (2) path_alignment uses F1 (precision + recall) instead of recall-only coverage
  (6) length penalty soft_start lowered 600 -> 400 (hard_cap unchanged at 1000)
  (7) new format_reward_func enforcing clean <think>/<explanation>/<answer> structure
  (9) path_alignment now gated on BOTH correct answer AND valid format
  (13) generation repetition_penalty raised 1.15 -> 1.22
- outcome, model stayed same/got a tiny bit worse after 400 more steps, and mean comp length dropped to 350, no more redundancy/weird behavoir but less reasoning 

V5 changes vs V4:
  (1) path_alignment coverage computed only over first 550 tokens of thinking (was 400)
  (2) length penalty soft_start raised 450 -> 550, hard_cap raised 1000 -> 1500
  (3) generation repetition_penalty reverted 1.22 -> 1.15
  (4) Retained strict formatting gates and strict F1 math to isolate variables.
- outcome: 
"""

import os
import re
import sys
import json
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from collections import Counter
import warnings
import logging

# Pipeline config loader (repo root, 2 levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import get_model_id, get_phase_param, render_prompt  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datasets import load_from_disk, Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from data_prep import preprocess_grpo_dataset

# =====================================================================
#                       MANUAL OVERRIDES (env-var driven)
# RL_DO_EVAL=1 to run eval (default off — slow); RL_NUM_EVAL_EXAMPLES
# tunes how many items get evaluated. Originally hardcoded; promoted to
# env vars so a paid run can switch without source edits.
DO_EVAL = os.environ.get("RL_DO_EVAL", "").lower() in ("1", "true", "yes", "on")
NUM_EVAL_EXAMPLES = int(os.environ.get("RL_NUM_EVAL_EXAMPLES", "20"))
# =====================================================================


def log_gpu_memory(tag: str = ""):
    """Log GPU memory usage for all visible devices."""
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device(f"cuda:{local_rank}")
    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved = torch.cuda.memory_reserved(dev) / 1024**3
    total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    free = total - reserved
    logging.info(
        f"[GPU {local_rank}] {tag} | "
        f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | "
        f"Free (est): {free:.2f} GB | Total: {total:.2f} GB"
    )


# =====================================================================
#                       System Prompt
# =====================================================================
# Sourced from prompts/rl_mcq.yaml (shared with test_rl.py). Byte-identical
# to the prior in-file SYSTEM_PROMPT / TASK_SPECIFIC_INSTRUCTIONS constants
# — see docs/PROMPT_MIGRATION.md item #13.
_rl_mcq = render_prompt("rl_mcq")
SYSTEM_PROMPT = _rl_mcq["system"]
TASK_SPECIFIC_INSTRUCTIONS = _rl_mcq["task_instructions"]


@dataclass
class TrainingConfig:
    """Configuration for GRPO (RL) training — full fine-tuning."""

    model_name: str = field(
        default_factory=lambda: get_model_id('base_sft', 'Qwen/Qwen3-14B'),
        metadata={"help": "Base model name (sourced from configs/default.yaml::models.base_sft)"},
    )
    sft_checkpoint_path: str = field(default="", metadata={"help": "Path to pre-merged SFT model"})
    cache_dir: str = field(default="~/.cache/huggingface/hub")

    dataset_path: str = field(default="/path/to/your/rl_dataset")
    output_dir: str = field(default="./rl_models/model-grpo")

    # GRPO params sourced from configs/default.yaml::rl.* (with hardcoded fallbacks).
    learning_rate: float = field(default_factory=lambda: get_phase_param('rl', 'learning_rate', 8e-7))
    beta: float = field(default_factory=lambda: get_phase_param('rl', 'beta', 0.12), metadata={"help": "KL penalty"})

    deepspeed: Optional[str] = field(default=None)

    num_generations: int = field(default_factory=lambda: get_phase_param('rl', 'num_generations', 4))
    max_completion_length: int = field(default_factory=lambda: get_phase_param('rl', 'max_completion_length', 1280))

    per_device_train_batch_size: int = field(default_factory=lambda: get_phase_param('rl', 'per_device_train_batch_size', 1))
    gradient_accumulation_steps: int = field(default_factory=lambda: get_phase_param('rl', 'gradient_accumulation_steps', 16))
    num_train_epochs: int = field(default_factory=lambda: get_phase_param('rl', 'num_train_epochs', 10))
    max_grad_norm: float = field(default_factory=lambda: get_phase_param('rl', 'max_grad_norm', 1.0))

    eval_size: int = field(default_factory=lambda: get_phase_param('rl', 'eval_size', 100), metadata={"help": "Number of held-out examples for eval"})
    eval_steps: int = field(default_factory=lambda: get_phase_param('rl', 'eval_steps', 50))
    save_steps: int = field(default_factory=lambda: get_phase_param('rl', 'save_steps', 100))
    generation_dump_every: int = field(default_factory=lambda: get_phase_param('rl', 'generation_dump_every', 25))

    wandb_project: Optional[str] = field(default=None)


# =====================================================================
#                       Utility Functions
# =====================================================================

def extract_answer(text: str) -> str:
    """Extract answer letter (A-D) from model output."""
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
    """Extract content within <think> tags."""
    try:
        return text.split("<think>")[-1].split("</think>")[0].strip()
    except IndexError:
        return ""


# ---------------------------------------------------------------------
# V4/V5: format validation, used by both format reward and path-align gate
# ---------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)


def validate_format(text: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Returns (fully_valid, component_flags). Component flags expose which
    pieces of the format are correct so the format reward can give partial
    credit, while gating logic uses the boolean.
    """
    flags = {
        "think_once": False,
        "expl_once": False,
        "answer_once": False,
        "no_trailing": False,
        "ordered": False,
    }

    think_matches = _THINK_RE.findall(text)
    expl_matches = _EXPL_RE.findall(text)
    answer_matches = _ANSWER_RE.findall(text)

    flags["think_once"] = len(think_matches) == 1
    flags["expl_once"] = len(expl_matches) == 1
    flags["answer_once"] = len(answer_matches) == 1

    answer_close_idx = text.lower().rfind("</answer>")
    if answer_close_idx != -1:
        trailing = text[answer_close_idx + len("</answer>"):]
        flags["no_trailing"] = (trailing.strip() == "")
    else:
        flags["no_trailing"] = False

    if flags["think_once"] and flags["expl_once"] and flags["answer_once"]:
        try:
            t_close = text.index("</think>")
            e_open = text.index("<explanation>")
            e_close = text.index("</explanation>")
            a_open = text.index("<answer>") if "<answer>" in text else text.lower().index("<answer>")
            a_close = text.lower().index("</answer>")
            flags["ordered"] = (t_close < e_open < e_close < a_open < a_close)
        except ValueError:
            flags["ordered"] = False

    fully_valid = all(flags.values())
    return fully_valid, flags


# =====================================================================
#                       Reward Helpers
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


# Defaults read once at module load — config-tunable per profile.
LENGTH_PENALTY_SOFT_START = get_phase_param('rl', 'length_penalty_soft_start', 550)
LENGTH_PENALTY_HARD_CAP   = get_phase_param('rl', 'length_penalty_hard_cap',   1500)
PATH_ALIGNMENT_THINKING_TOKENS = get_phase_param('rl', 'path_alignment_thinking_tokens', 550)


def smooth_length_penalty(num_tokens: int,
                          soft_start: int = LENGTH_PENALTY_SOFT_START,
                          hard_cap: int = LENGTH_PENALTY_HARD_CAP,
                          max_penalty: float = 1.0) -> float:
    """
    Smooth length penalty. Defaults loaded from config at module init
    (rl.length_penalty_soft_start / rl.length_penalty_hard_cap).
    """
    if num_tokens <= soft_start:
        return 0.0
    if num_tokens >= hard_cap:
        return max_penalty
    return max_penalty * (num_tokens - soft_start) / (hard_cap - soft_start)


def truncate_thinking_for_coverage(thinking: str, max_tokens: int = PATH_ALIGNMENT_THINKING_TOKENS) -> str:
    """Cap the thinking string at its first N tokens (config: rl.path_alignment_thinking_tokens)."""
    if _TOKENIZER_REF is not None:
        ids = _TOKENIZER_REF.encode(thinking, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return thinking
        truncated_ids = ids[:max_tokens]
        return _TOKENIZER_REF.decode(truncated_ids, skip_special_tokens=True)
    else:
        words = thinking.split()
        approx_word_cap = int(max_tokens / 1.3)
        if len(words) <= approx_word_cap:
            return thinking
        return " ".join(words[:approx_word_cap])


# =====================================================================
#                       Reward Functions
# =====================================================================

_TOKENIZER_REF = None


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Correctness reward.
    V5: smooth token-based length penalty (soft_start=550, hard_cap=1500, max=-1.0)
    """
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

        # V5: soft_start=550, hard_cap=1500
        reward -= smooth_length_penalty(num_tokens, soft_start=550, hard_cap=1500, max_penalty=1.0)
        rewards.append(reward)

    return rewards


def format_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """Format reward (unchanged in V5)."""
    responses = [c[0]["content"] for c in completions]
    rewards = []
    per_component = 0.04  # 5 components × 0.04 = 0.20 max
    for r in responses:
        _, flags = validate_format(r)
        score = sum(per_component for v in flags.values() if v)
        rewards.append(score)
    return rewards


def path_alignment_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    V5 path alignment reward:
      - Coverage computed over the FIRST 550 TOKENS of <think> only.
      - Maintains strict format gating.
      - Maintains strict F1 score.
    """
    responses = [c[0]["content"] for c in completions]
    thinkings = [extract_thinking(r) for r in responses]
    extracted = [extract_answer(r) for r in responses]

    gt_answers = []
    for ans in answer:
        letters = re.findall(r'\b[A-D]\b', ans)
        gt_answers.append(letters[-1] if letters else "")

    paths_batch = kwargs.get("paths", [None] * len(responses))

    rewards = []
    for response, thinking, kg_path, pred, gt in zip(
        responses, thinkings, paths_batch, extracted, gt_answers
    ):
        # Gate 1: must be correct
        if pred != gt or pred == "":
            rewards.append(0.0)
            continue

        # Gate 2: must have valid format
        format_valid, _ = validate_format(response)
        if not format_valid:
            rewards.append(0.0)
            continue

        if kg_path is None:
            rewards.append(0.0)
            continue

        # V5: truncate thinking to first 550 tokens
        truncated_thinking = truncate_thinking_for_coverage(thinking, max_tokens=550)

        path_tokens = set(normalize_tokens(str(kg_path)))
        thinking_tokens_list = normalize_tokens(truncated_thinking)
        thinking_tokens_set = set(thinking_tokens_list)

        if not path_tokens or not thinking_tokens_set:
            rewards.append(0.0)
            continue

        hits = thinking_tokens_set & path_tokens

        recall = len(hits) / max(1, len(path_tokens))
        precision = len(hits) / max(1, len(thinking_tokens_set))
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

        min_unique_hit = 1.0 if len(hits) >= 2 else 0.0

        full_thinking_tokens = normalize_tokens(thinking)
        rep_factor = repetition_penalty_factor(full_thinking_tokens)

        base_reward = 0.7 * f1 + 0.2 * min_unique_hit
        rewards.append(min(base_reward * rep_factor, 0.8))

    return rewards


# =====================================================================
#                       Generation Dump Callback
# =====================================================================

class GenerationDumpCallback(transformers.TrainerCallback):
    def __init__(self, output_dir: str, dump_every: int = 25):
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


# =====================================================================
#                       Held-out Eval Callback
# =====================================================================

class HeldOutEvalCallback(transformers.TrainerCallback):
    def __init__(self, eval_dataset, eval_every: int = 100, max_new_tokens: int = 1280):
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
        logging.info(f"[Rank {rank}] Starting held-out eval at step {state.global_step}")

        model = self.trainer.model
        tokenizer = self.trainer.processing_class
        model.eval()

        correct = 0
        total = 0
        malformed = 0

        try:
            with torch.no_grad():
                for idx, example in enumerate(self.eval_dataset):
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
            malformed_rate = malformed / max(1, total)

            logging.info(
                f"[EVAL step={state.global_step}] accuracy={accuracy:.4f} "
                f"({correct}/{total}) malformed={malformed_rate:.4f}"
            )

            if rank == 0:
                self.history.append({
                    "step": state.global_step,
                    "epoch": round(float(state.epoch), 4),
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "malformed": malformed,
                })
                if self.history_path is not None:
                    try:
                        with open(self.history_path, "w") as f:
                            json.dump(self.history, f, indent=2)
                    except Exception as e:
                        logging.warning(f"Failed to write eval history: {e}")

        except Exception as e:
            logging.error(f"[Rank {rank}] Eval failed at step {state.global_step}: {e}")
        finally:
            model.train()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return control


# =====================================================================
#                       Main Training Function
# =====================================================================

def train():
    """Main RL training function — full fine-tuning (no LoRA)."""
    global _TOKENIZER_REF

    parser = transformers.HfArgumentParser(TrainingConfig)
    config = parser.parse_args_into_dataclasses()[0]    

    # ---- Apply env-var overrides (eval scope) ----
    # `num_train_epochs` is NO LONGER silently overridden — it now flows from
    # CLI (--num_train_epochs) or rl_training's own config defaults. This used
    # to be `config.num_train_epochs = 10` which silently ignored CLI/config.
    if not DO_EVAL:
        config.eval_size = 0
    else:
        config.eval_size = NUM_EVAL_EXAMPLES
    if not config.output_dir:
        raise ValueError("output_dir is required. Pass --output_dir to the script.")

    if config.deepspeed and not os.path.isabs(config.deepspeed):
        config.deepspeed = os.path.abspath(config.deepspeed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    merged_path = config.sft_checkpoint_path
    logging.info(f"Using pre-merged model at {merged_path}")
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            import datetime
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600))
        dist.barrier()
    logging.info(f"Rank {local_rank}/{world_size} initialized")
    log_gpu_memory("After dist init")

    # ---- Dataset ----
    logging.info(f"Preprocessing dataset from {config.dataset_path}")
    full_dataset = preprocess_grpo_dataset(
        dataset_path=config.dataset_path,
        split="train",
        chunk_size=1000,
        enable_thinking=True,
        system_prompt=SYSTEM_PROMPT,
        task_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    )
    logging.info(f"Full dataset: {len(full_dataset)} examples")

    if DO_EVAL and config.eval_size > 0:
        split = full_dataset.train_test_split(test_size=config.eval_size, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logging.info(f"Train: {len(train_dataset)}, Eval (held-out): {len(eval_dataset)}")
    else:
        train_dataset = full_dataset
        eval_dataset = None
        logging.info(f"Train: {len(train_dataset)}, Eval (held-out): SKIPPED (DO_EVAL=False)")

    # ---- Model ----
    logging.info(f"Loading model from {merged_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        use_cache=False,
    )
    log_gpu_memory("After model load")

    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    _TOKENIZER_REF = tokenizer

    # ---- Callbacks ----
    os.makedirs(config.output_dir, exist_ok=True)
    gen_dump_cb = GenerationDumpCallback(
        output_dir=config.output_dir,
        dump_every=config.generation_dump_every,
    )

    callbacks_list = [gen_dump_cb]
    if DO_EVAL and eval_dataset is not None:
        eval_cb = HeldOutEvalCallback(
            eval_dataset=eval_dataset,
            eval_every=config.eval_steps,
            max_new_tokens=config.max_completion_length,
        )
        eval_cb.history_path = os.path.join(config.output_dir, "eval_history.json")
        callbacks_list.append(eval_cb)

    # ---- GRPO config ----
    training_args = GRPOConfig(
        deepspeed=config.deepspeed,

        learning_rate=config.learning_rate,
        beta=config.beta,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,

        bf16=True,

        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        # Sampling for GRPO rollouts — config-tunable.
        temperature=get_phase_param('rl', 'generation_temperature', 0.6),
        top_p=get_phase_param('rl', 'generation_top_p', 0.9),
        repetition_penalty=get_phase_param('rl', 'generation_repetition_penalty', 1.15),

        optim="adamw_torch",

        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=20,

        log_completions=True,
        num_completions_to_print=1,

        logging_steps=1,
        max_grad_norm=config.max_grad_norm,
        output_dir=config.output_dir,
        report_to=[] if config.wandb_project is None else ["wandb"],
    )

    log_gpu_memory("After GRPOConfig")

    reward_funcs = [
        correctness_reward_func,
        format_reward_func,
        path_alignment_reward_func,
    ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks_list,
    )

    gen_dump_cb.trainer = trainer
    if DO_EVAL and eval_dataset is not None:
        eval_cb.trainer = trainer

    if DO_EVAL:
        logging.info(f"  eval_steps={config.eval_steps} | eval_size={config.eval_size}")
    else:
        logging.info("  eval loop completely skipped")

    logging.info(f"  generation dump every {config.generation_dump_every} steps")
    logging.info(f"  V5: length penalty soft_start=550, hard_cap=1500, max=-1.0")
    logging.info(f"  V5: path_alignment uses F1 over first-550-token thinking, gated on correct+format")
    logging.info(f"  V5: format_reward_func added (max +0.2)")
    logging.info(f"  V5: generation repetition_penalty=1.15")
    logging.info("=" * 70)

    # --- RESUME FROM CHECKPOINT LOGIC ---
    resume_path = os.environ.get("RESUME_CHECKPOINT", "")

    if resume_path and os.path.exists(resume_path):
        logging.info(f"Resuming training from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        if resume_path:
            logging.warning(f"Checkpoint {resume_path} not found! Starting from scratch.")
        trainer.train()
    # ------------------------------------
    logging.info("Saving final model...")
    trainer.save_model(config.output_dir)
    trainer.accelerator.wait_for_everyone()

    logging.info("Training complete!")


if __name__ == "__main__":
    train()