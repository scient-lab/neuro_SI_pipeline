'''
Copyright (c) 2025 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.

Adapted from https://github.com/simplescaling/s1/blob/main/train/sft.py
'''

import os
import sys
import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

# Pipeline config loader (repo root, 2 levels up).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import get_model_id, get_phase_param  # noqa: E402

# MONKEY PATCH: transformers internally calls dist.fsdp.register_fsdp_forward_method,
# which does not exist in older PyTorch builds. Must be patched BEFORE transformers import.
import torch.distributed.fsdp as _fsdp_module
if not hasattr(_fsdp_module, 'register_fsdp_forward_method'):
    _fsdp_module.register_fsdp_forward_method = lambda *a, **kw: None

import transformers
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import LoraConfig, TaskType

# Enforce NCCL timeout in Python as well (belt-and-suspenders with slurm env vars)
os.environ.setdefault("NCCL_TIMEOUT", "3600")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str = field(
        default_factory=lambda: os.environ.get("MODEL_NAME", "") or get_model_id('base_sft', ''),
        metadata={"help": "Path to base model. Set via --model_name, MODEL_NAME env var, or configs/default.yaml::models.base_sft."},
    )
    block_size: int = field(default_factory=lambda: get_phase_param('sft', 'block_size', 32768))
    wandb_project: str = field(default="sft_neuro_kg")
    wandb_dir: str = field(default=os.environ.get("WANDB_DIR", "./wandb_logs"))
    train_dataset_path: str = field(default=os.environ.get("DATASET_PATH", ""),
                                    metadata={"help": "Path to tokenized training dataset. Set via --train_dataset_path or DATASET_PATH env var."})
    use_lora: bool = field(default=True)
    # LoRA params sourced from configs/default.yaml::sft.* (with hardcoded fallbacks).
    lora_r: int = field(default_factory=lambda: get_phase_param('sft', 'lora_r', 32))
    lora_alpha: int = field(default_factory=lambda: get_phase_param('sft', 'lora_alpha', 64))
    lora_dropout: float = field(default_factory=lambda: get_phase_param('sft', 'lora_dropout', 0.05))
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    # ------------------------------------------------------------------ #
    # SUBSET TRAINING: set > 0 to use a fixed-size subset for quick tests.
    # Pass via CLI: --train_subset_size=256
    # 0 means use the full dataset.
    # ------------------------------------------------------------------ #
    train_subset_size: int = field(default=0)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_DIR'] = self.wandb_dir


class FastCompletionOnlyCollator(transformers.DataCollatorForLanguageModeling):
    """
    Masks all tokens before (and including) the response template so the
    model only trains on the assistant completion.

    Accepts pre-computed token IDs directly to avoid encoding issues with
    special tokens that contain fullwidth Unicode characters.
    """
    def __init__(self, response_template_ids, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, *args, **kwargs)
        self.response_token_ids = response_template_ids

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            labels = batch["labels"][i]
            seq_len = len(self.response_token_ids)

            unfolded = labels.unfold(0, seq_len, 1)
            template_tensor = torch.tensor(self.response_token_ids, device=labels.device)

            matches = (unfolded == template_tensor).all(dim=1)
            match_indices = matches.nonzero(as_tuple=True)[0]

            if len(match_indices) > 0:
                first_match_idx = match_indices[0]
                labels[: first_match_idx + seq_len] = -100

        return batch


def save_model_fsdp_safe(trainer, output_dir: str, tokenizer):
    """
    Safely extract the full PEFT adapter state dict from an FSDP-wrapped model
    and save it only from rank 0. This avoids the Trainer's internal allgather
    path that deadlocks when modules_to_save wrappers desync ranks.

    All ranks must call this function together (it contains collective ops),
    but only rank 0 writes to disk.
    """
    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    if trainer.is_fsdp_enabled:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        logger.info(f"[Rank {rank}] Entering FSDP FULL_STATE_DICT gather...")

        # All ranks must participate in this context manager
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            trainer.model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            state_dict = trainer.model.state_dict()

        if is_main:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"[Rank 0] Saving adapter weights to {output_dir}")
            trainer.model.save_pretrained(output_dir, state_dict=state_dict)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"[Rank 0] Adapter and tokenizer saved successfully.")
    else:
        # Non-FSDP path (DDP or single GPU)
        if is_main:
            os.makedirs(output_dir, exist_ok=True)
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"[Rank 0] Model saved to {output_dir}")

    # Barrier so no rank exits before rank 0 finishes writing
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class EpochCheckpointCallback(transformers.TrainerCallback):
    """
    Saves a full FSDP-safe checkpoint at the end of every epoch.
    All ranks call save_model_fsdp_safe together (required for FSDP collectives);
    only rank 0 writes to disk.

    The trainer reference is set after trainer construction to avoid a
    circular dependency at init time.
    """
    def __init__(self, base_output_dir: str, tokenizer):
        self.base_output_dir = base_output_dir
        self.tokenizer = tokenizer
        self.trainer = None  # set externally after SFTTrainer is constructed

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            logger.warning("EpochCheckpointCallback: trainer not set, skipping checkpoint.")
            return control

        epoch = int(state.epoch)
        checkpoint_dir = os.path.join(self.base_output_dir, f"checkpoint-epoch-{epoch}")
        rank = int(os.environ.get("RANK", 0))
        logger.info(f"[Rank {rank}] End of epoch {epoch} — saving checkpoint to {checkpoint_dir}")
        save_model_fsdp_safe(self.trainer, checkpoint_dir, self.tokenizer)
        return control


def _resolve_response_template_ids(tokenizer):
    """
    Find the token ID(s) for the DeepSeek assistant turn delimiter.

    DeepSeek-R1-0528-Qwen3-8B uses <｜Assistant｜> (fullwidth Unicode pipes)
    as a single special token. We look it up by name in the vocabulary to
    avoid encoding issues.

    Returns a list of token IDs.
    """
    # The exact special token name used by DeepSeek's tokenizer
    # Fullwidth pipe: U+FF5C (｜) — NOT regular pipe U+007C (|)
    ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"  # <｜Assistant｜>

    # Method 1: Direct vocabulary lookup (most reliable for special tokens)
    token_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
    if token_id != tokenizer.unk_token_id and token_id is not None:
        logger.info(f"Found assistant token via vocab lookup: "
                    f"'{ASSISTANT_TOKEN}' -> id {token_id}")
        return [token_id]

    # Method 2: Search added_tokens for partial match
    for tok_str, tok_id in tokenizer.get_added_vocab().items():
        if "Assistant" in tok_str:
            logger.info(f"Found assistant token via added_vocab search: "
                        f"'{tok_str}' -> id {tok_id}")
            return [tok_id]

    # Method 3: Fallback — encode the token string
    encoded = tokenizer.encode(ASSISTANT_TOKEN, add_special_tokens=False)
    if encoded:
        logger.warning(f"Using encode fallback for assistant token: "
                       f"'{ASSISTANT_TOKEN}' -> ids {encoded}")
        return encoded

    raise ValueError(
        f"Could not find the assistant turn delimiter token in the tokenizer. "
        f"Tried: '{ASSISTANT_TOKEN}'. This tokenizer may not be DeepSeek-R1-0528."
    )


def train():
    # ------------------------------------------------------------------ #
    # FIX: Explicitly bind each rank to its own GPU BEFORE any CUDA or
    # distributed call. This is the root cause of "Duplicate GPU detected"
    # on Della nodes where SLURM's GPU binding is broken.
    # torchrun sets LOCAL_RANK for each subprocess; we use it to pin the
    # CUDA device immediately so NCCL never sees two ranks on the same GPU.
    # ------------------------------------------------------------------ #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    rank = int(os.environ.get("RANK", 0))
    print(f"[Rank {rank}] Pinned to CUDA device {local_rank} "
          f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')} "
          f"| Host={os.uname().nodename}")

    print(f"Transformers Version: {transformers.__version__}")
    print(f"PyTorch Version: {torch.__version__}")

    is_main = rank == 0

    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()

    # ------------------------------------------------------------------ #
    # 1. Load Model
    # ------------------------------------------------------------------ #
    if is_main:
        print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    # ------------------------------------------------------------------ #
    # 2. Load Tokenizer & Resize Embeddings
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True, trust_remote_code=True
    )
    tokenizer.add_special_tokens({'pad_token': '<|fim_pad|>'})

    original_vocab_size = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = len(tokenizer)

    # Freeze original embedding rows via gradient hook (slice-level
    # requires_grad assignment is a no-op in PyTorch; hooks work correctly).
    if new_vocab_size > original_vocab_size:
        def _freeze_original_embed_rows(grad):
            grad[:original_vocab_size] = 0
            return grad

        embed = model.get_input_embeddings()
        embed.weight.register_hook(_freeze_original_embed_rows)

        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head.weight.data_ptr() != embed.weight.data_ptr():
            def _freeze_original_lm_head_rows(grad):
                grad[:original_vocab_size] = 0
                return grad
            lm_head.weight.register_hook(_freeze_original_lm_head_rows)

    # Required for gradient checkpointing with LoRA
    model.enable_input_require_grads()

    # ------------------------------------------------------------------ #
    # 3. Configure LoRA
    # NOTE: modules_to_save is intentionally omitted.
    # Using modules_to_save=["embed_tokens","lm_head"] causes PEFT to wrap
    # those layers in ModulesToSaveWrapper. Under FSDP this creates a
    # rank-asymmetric allgather path -> 30-min hang -> SIGABRT.
    # The gradient hooks above handle this safely instead.
    # ------------------------------------------------------------------ #
    lora_config = None
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if is_main:
            print(f"LoRA config: r={config.lora_r}, alpha={config.lora_alpha}, "
                  f"scale={config.lora_alpha / config.lora_r:.2f}")

    # ------------------------------------------------------------------ #
    # 4. Load Data
    # ------------------------------------------------------------------ #
    dataset = load_from_disk(config.train_dataset_path)

    # ------------------------------------------------------------------ #
    # SUBSET MODE: triggered by --train_subset_size > 0
    # Useful for quick smoke-tests before a full run.
    # ------------------------------------------------------------------ #
    if config.train_subset_size > 0:
        if is_main:
            print("\n" + "=" * 60)
            print(f"SUBSET MODE: using {config.train_subset_size} train examples")
            print("=" * 60 + "\n")

        train_n = min(config.train_subset_size, len(dataset['train']))
        dataset['train'] = dataset['train'].select(range(train_n))

        if 'test' in dataset:
            test_n = max(32, train_n // 8)
            test_n = min(test_n, len(dataset['test']))
            dataset['test'] = dataset['test'].select(range(test_n))

    if is_main:
        print(f"Dataset sizes — train: {len(dataset['train'])}, "
              f"test: {len(dataset['test']) if 'test' in dataset else 'N/A'}")

    # ------------------------------------------------------------------ #
    # 5. Collator — find DeepSeek's assistant token for completion masking
    # ------------------------------------------------------------------ #
    # DeepSeek-R1-0528-Qwen3-8B uses <｜Assistant｜> as the assistant turn
    # delimiter (fullwidth Unicode pipes, single special token).
    # We look up the token ID directly to avoid encoding issues.
    response_template_ids = _resolve_response_template_ids(tokenizer)
    if is_main:
        print(f"Response template token IDs: {response_template_ids}")

    collator = FastCompletionOnlyCollator(
        response_template_ids=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    # ------------------------------------------------------------------ #
    # 6. Training arguments — safety overrides
    # ------------------------------------------------------------------ #
    # save_strategy="no": prevents Trainer's internal mid-training FSDP
    # allgather for checkpoints. We save per-epoch via EpochCheckpointCallback
    # using our own FSDP-safe path instead.
    args.save_strategy = "no"
    # drop_last: ensures every rank sees the same number of batches
    # (prevents FSDP hang on uneven tail)
    args.dataloader_drop_last = True
    # ddp_timeout: gives collectives more room under heavy I/O load
    args.ddp_timeout = 7200
    # Avoid safetensors serialization issues with FSDP-gathered state dicts
    args.save_safetensors = False
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    # Limit dataloader workers to avoid IO stalls on /scratch that cascade
    # into NCCL timeouts. Config-tunable per profile because RunPod single-node,
    # Princeton SLURM cluster, and paper-scale multi-node have very different
    # I/O profiles and CPU budgets.
    args.dataloader_num_workers = get_phase_param('sft', 'dataloader_num_workers', 2)
    args.dataloader_prefetch_factor = get_phase_param('sft', 'dataloader_prefetch_factor', 2)

    if is_main:
        print("--- Training Safety Config ---")
        print("  save_strategy         : no (save per-epoch via EpochCheckpointCallback)")
        print("  dataloader_drop_last  : True")
        print("  ddp_timeout           : 7200s")
        print(f"  dataloader_num_workers: {args.dataloader_num_workers}")
        print(f"  dataloader_prefetch  : {args.dataloader_prefetch_factor}")
        print(f"  learning_rate         : {args.learning_rate}")

    # ------------------------------------------------------------------ #
    # 7. Build epoch checkpoint callback (trainer ref injected after init)
    # ------------------------------------------------------------------ #
    epoch_ckpt_callback = EpochCheckpointCallback(
        base_output_dir=args.output_dir,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------ #
    # 8. SFTTrainer
    # ------------------------------------------------------------------ #
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else None,
        args=args,
        peft_config=lora_config,
        data_collator=collator,
        callbacks=[epoch_ckpt_callback],
    )

    # Inject trainer reference into callback now that it exists
    epoch_ckpt_callback.trainer = trainer

    # ------------------------------------------------------------------ #
    # 9. Train
    # ------------------------------------------------------------------ #
    print(f"[Rank {rank}] Starting training...")
    trainer.train()

    # ------------------------------------------------------------------ #
    # 10. Final save — FSDP-safe, identical to epoch checkpoints
    # ------------------------------------------------------------------ #
    print(f"[Rank {rank}] Training complete. Saving final model...")
    save_model_fsdp_safe(trainer, args.output_dir, tokenizer)

    if is_main:
        print(f"Final checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    train()