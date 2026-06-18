"""
Dataset preprocessing and splitting for SFT and RL training.

This script handles:
1. Converting raw data to messages format
2. Creating train/test splits
3. Preprocessing for GRPO training (with KG path integration)

Configure all parameters in the section below, then run:
    python data_prep.py
"""

import os
import re
from typing import List, Dict, Optional, Any
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import logging

# ============================================================
# CONFIGURE THESE PARAMETERS (replaces command-line arguments)
# ============================================================

INPUT_PATH = os.environ.get("INPUT_PATH", "")   # Path to input dataset (set via env var or edit here)
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "")  # Path to save processed dataset
MODE = os.environ.get("RL_DATA_PREP_MODE", "rl")          # "sft" or "rl"

# Take only the last N items from the dataset; "0" or "all" → use all items.
# Smoke/pilot can shrink this for fast iteration; paper sets it to all.
_last_n_raw = os.environ.get("RL_LAST_N", "5000").lower()
LAST_N = None if _last_n_raw in ("0", "all", "none", "") else int(_last_n_raw)

ENABLE_THINKING = os.environ.get("RL_ENABLE_THINKING", "1").lower() in ("1", "true", "yes", "on")
EVAL_SPLIT_RATIO = float(os.environ.get("RL_EVAL_SPLIT_RATIO", "0.02"))   # SFT mode only
MIN_EVAL_SIZE   = int(os.environ.get("RL_MIN_EVAL_SIZE", "200"))         # SFT mode only

# ============================================================
# END CONFIGURATION
# ============================================================


def extract_answer(text: str) -> str:
    """
    Extract answer from text (supports multiple formats).
    
    Handles:
    - 'Final Answer: X' format
    - '<answer>X</answer>' tags
    - Markdown formatting (e.g., **Final Answer:** C)
    
    Args:
        text: Text containing the answer
    
    Returns:
        Extracted answer letter (A/B/C/D) or empty string
    """
    try:
        # Strip markdown formatting
        text_clean = re.sub(r'\*+', '', text)
        
        # Look for "Final Answer: X" pattern
        match = re.search(r'Final Answer\s*[:\-]\s*([A-D])', text_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback 1: Check for <answer>X</answer> tags
        answer_match = re.search(r'<answer>\s*([A-D])\s*</answer>', text_clean, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Fallback 2: Check for just <answer>X (no closing tag)
        answer_match2 = re.search(r'<answer>\s*([A-D])', text_clean, re.IGNORECASE)
        if answer_match2:
            return answer_match2.group(1).upper()
        
        # Fallback 3: Extract any A-D letter that appears after </think>
        if '</think>' in text_clean:
            after_think = text_clean.split('</think>')[-1]
            letters = re.findall(r'\b[A-D]\b', after_think)
            if letters:
                return letters[0].upper()
        
        return ""
    except Exception:
        return ""


def to_messages_format(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dataset example to messages format for TRL training.
    
    Expected input format (in 'question_and_explanation' field):
        <Question>: {question text}
        <Options>: {options A-D}
        <Explanation>: {chain of thought}
        <Answer>: {answer letter}
    
    Output format:
        {
            "messages": [
                {"role": "user", "content": "{question}\nOptions:{options}"},
                {"role": "assistant", "content": "<think>\n{explanation}\n</think>\nFinal Answer: {answer}"}
            ]
        }
    """
    qae = example.get('question_and_explanation', '')
    try:
        # Extract question
        if '<Question>:' in qae:
            question = qae.split('<Question>:')[1].split('</Question>')[0]
        elif '<Question>' in qae:
            question = qae.split('<Question>')[1].split('</Question>')[0]
        else:
            question = ''

        # Extract options
        if '<Options>' in qae:
            options = qae.split('<Options>')[1].split('</Options>')[0]
        elif '<Options>:' in qae:
            options = qae.split('<Options>:')[1].split('</Options>')[0]
        else:
            options = ''

        # Extract explanation (chain of thought)
        cot = qae.split('<Explanation>')[1].split('</Explanation>')[0]
        
        # Extract answer
        answer = qae.split('<Answer>:')[1].split('</Answer>')[0]

        # Create user prompt (question + options)
        user_content = question.strip() + '\nOptions:' + options.strip()
        
        # Create assistant response (thinking + final answer)
        assistant_content = "<think>\n" + cot.strip() + "\n</think>\nFinal Answer: " + answer.strip()
        
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
    except Exception as e:
        # Fallback to text field if parsing fails
        fallback_text = example.get('text', '')
        return {
            "messages": [
                {"role": "user", "content": "Answer the following question:"},
                {"role": "assistant", "content": fallback_text}
            ]
        }


def prepare_sft_dataset(
    dataset: Dataset,
    eval_split_ratio: float = 0.02,
    min_eval_size: int = 500
) -> DatasetDict:
    """
    Prepare dataset for SFT training.
    
    Args:
        dataset: Input dataset with raw examples
        eval_split_ratio: Ratio of data to use for evaluation
        min_eval_size: Minimum number of examples in eval split
    
    Returns:
        DatasetDict with 'train' and 'test' splits in messages format
    """
    # Convert to messages format
    dataset = dataset.map(to_messages_format, batched=False)
    
    # Create eval split if not present
    holdout = min(min_eval_size, max(1, int(eval_split_ratio * len(dataset))))
    eval_ds = dataset.select(range(0, holdout))
    train_ds = dataset.select(range(holdout, len(dataset)))
    
    print(f"SFT dataset prepared - Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return DatasetDict({'train': train_ds, 'test': eval_ds})


def preprocess_grpo_dataset(
    dataset_path: str,
    split: str = "train",
    chunk_size: int = 1000,
    enable_thinking: bool = True,
    system_prompt: str = "A conversation between user and assistant. The user asks a single-choice Multiple Choice Question, and the assistant solves it using step-by-step reasoning. Please answer the multiple choice question by selecting only one from option A, option B, option C, option D. \n\nThe assistant first thinks through the problem systematically, then provides the explanation process and final answer. Use <think>...</think> tags for internal reasoning, then provide the answer enclosed within <answer> </answer> tags.",
    task_instructions: str = "Please provide complete and accurate answers with clear reasoning. The answer must only be a single letter from A, B, C, D."
) -> Dataset:
    """
    Preprocess dataset for GRPO (RL) training.
    
    Converts dataset to prompt/answer format with optional thinking mode.
    Includes knowledge graph path information if available.
    """
    loaded = load_from_disk(dataset_path)
    if isinstance(loaded, DatasetDict):
        dataset = loaded[split]
    else:
        dataset = loaded
    
    def process_batch(batch):
        prompts = []
        answers = []
        paths_list = []
        
        has_paths = "paths" in batch
        for idx, qae in enumerate(batch["question_and_explanation"]):
            # Parse question
            if '<Question>' in qae:
                question = qae.split('<Question>')[1].split('</Question>')[0].strip()
            else:
                question = ''
            
            # Parse options
            if '<Options>' in qae:
                options = qae.split('<Options>')[1].split('</Options>')[0].strip()
            else:
                options = ''
            
            # Parse answer
            answer = ''
            ans_match = re.search(r'<Answer>\s*:?\s*([A-D])', qae, re.IGNORECASE)
            if ans_match:
                answer = ans_match.group(1).upper()
            
            user_content = question + "\n" + options
            if enable_thinking:
                user_content += "\n/think"
            else:
                user_content += "\n/no_think"
            
            prompt = [
                {"role": "system", "content": system_prompt + "\n" + task_instructions},
                {"role": "user", "content": user_content}
            ]
            
            prompts.append(prompt)
            answers.append(answer)
            
            if has_paths:
                paths_list.append(batch["paths"][idx])
        
        result = {"prompt": prompts, "answer": answers}
        if has_paths:
            result["paths"] = paths_list
        return result
    
    return dataset.map(process_batch, batched=True, batch_size=chunk_size)


def main():
    """Run preprocessing with parameters configured at top of file."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Loading dataset from {INPUT_PATH}")
    logging.info(f"Mode: {MODE}")
    if LAST_N is not None:
        logging.info(f"Taking last {LAST_N} items from dataset")
    
    if MODE == "sft":
        dataset = load_from_disk(INPUT_PATH)
        if isinstance(dataset, DatasetDict):
            if 'train' in dataset:
                dataset = dataset['train']
            else:
                # Take first available split
                dataset = dataset[list(dataset.keys())[0]]
        
        # Slice to last N items if specified
        if LAST_N is not None:
            total = len(dataset)
            start_idx = max(0, total - LAST_N)
            dataset = dataset.select(range(start_idx, total))
            logging.info(f"Selected last {LAST_N} items (indices {start_idx}..{total-1}, got {len(dataset)} items)")
        
        processed = prepare_sft_dataset(
            dataset,
            eval_split_ratio=EVAL_SPLIT_RATIO,
            min_eval_size=MIN_EVAL_SIZE
        )
        processed.save_to_disk(OUTPUT_PATH)
        logging.info(f"SFT dataset saved - Train: {len(processed['train'])}, Test: {len(processed['test'])}")
        
    elif MODE == "rl":
        # Load from JSON file
        full_ds = load_dataset("json", data_files=INPUT_PATH, split="train")
        
        if LAST_N is not None:
            total = len(full_ds)
            start_idx = max(0, total - LAST_N)
            full_ds = full_ds.select(range(start_idx, total))
            logging.info(f"Selected last {LAST_N} items (indices {start_idx}..{total-1}, got {len(full_ds)} items)")
        
        # Save as HF dataset so preprocess_grpo_dataset can load it
        tmp_path = OUTPUT_PATH + "_tmp_sliced"
        DatasetDict({"train": full_ds}).save_to_disk(tmp_path)
        
        processed = preprocess_grpo_dataset(
            tmp_path,
            split="train",
            enable_thinking=ENABLE_THINKING
        )
        
        # Clean up temp
        import shutil
        shutil.rmtree(tmp_path, ignore_errors=True)
        
        processed.save_to_disk(OUTPUT_PATH)
        logging.info(f"RL dataset saved - {len(processed)} examples")
    
    logging.info(f"Processed dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()