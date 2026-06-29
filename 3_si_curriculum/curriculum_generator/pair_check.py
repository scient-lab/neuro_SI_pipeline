"""Single-LLM, non-Gemini grounding check of a generated QA *pair* (pre-trace).

Phase 4 (curriculum), step ``validate_qa_pair``. Per Prof Jha (2026-06-29):

    "Follow the bottom-up paper (Dedhia's) paradigm: 1 LLM to check QA pairs and 2 LLMs to
     check QA items. Note that these three LLMs should be from another family (not the same
     family of LLMs that generated the QA pair/item)."

The generator is Gemini, so this grader runs on a NON-Gemini *reasoning* model.

Transport: the OpenAI SDK with a configurable ``base_url`` -- the SAME client talks to either
hosted OpenAI (default) or a local vLLM OpenAI-compatible server (e.g. gpt-oss-20b on
vLLM>=0.10). Flip ``curriculum.pair_check_base_url``; no code change.

Everything is config/prompt driven (no hardcoding):
  - model:    models.curriculum_pair_check         (default o3-mini; MUST be a reasoning model)
  - base_url: curriculum.pair_check_base_url        (default https://api.openai.com/v1)
  - api key:  env var named by curriculum.pair_check_api_key_env (default OPENAI_API_KEY)
  - prompt:   prompts/curriculum_pair_check.yaml    (no-trace grounding rubric)

Requires ``openai>=1.40`` in the venv (pure-Python, no torch conflict).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

# pipeline_config lives at the repo root (curriculum_generator/ -> 3_si_curriculum/ -> root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import render_prompt, get_model_id, get_phase_param  # noqa: E402

_DEFAULT_MODEL = "o3-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    base_url = get_phase_param("curriculum", "pair_check_base_url", _DEFAULT_BASE_URL)
    key_env = get_phase_param("curriculum", "pair_check_api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(key_env)
    if not api_key:
        # For a local vLLM OpenAI server any non-empty string works; for hosted OpenAI the
        # real key must be present. Fail loudly rather than silently degrade.
        raise RuntimeError(
            f"pair_check: no API key in ${key_env}. Export it, or point "
            f"curriculum.pair_check_base_url at a local vLLM server and set ${key_env} to any "
            f"non-empty value."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def check_pair(question: str, answer: str, context_path: str) -> bool:
    """True iff the bare QA pair passes the no-trace grounding rubric ([yes]).

    Args mirror the prompt slots in prompts/curriculum_pair_check.yaml. ``question`` is the
    full question text (the A./B./C./D. options are embedded in it, as stored on the curriculum
    record); ``context_path`` is the KG path string the pair was derived from.
    """
    p = render_prompt(
        "curriculum_pair_check",
        question=question,
        answer=answer,
        path=context_path,
    )
    resp = _client().chat.completions.create(
        model=get_model_id("curriculum_pair_check", _DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ],
        # o-series uses max_completion_tokens (not max_tokens) and counts hidden reasoning
        # tokens against it; do NOT pass temperature (o-series only supports the default).
        max_completion_tokens=get_phase_param("curriculum", "pair_check_max_tokens", 2048),
    )
    verdict = (resp.choices[0].message.content or "").lower()
    # Robust to both hosted o-series (internal reasoning -> "[yes]"/"[no]") and local <think>
    # models: a [no] anywhere, or the absence of [yes], fails the pair.
    return "[yes]" in verdict and "[no]" not in verdict


if __name__ == "__main__":
    # Manual smoke (needs the configured API key / endpoint). Not a domain fixture.
    ok = check_pair(
        question="Which neurotransmitter is primarily released by motor neurons at the "
        "neuromuscular junction?\nA. Dopamine\nB. Acetylcholine\nC. Serotonin\nD. GABA",
        answer="B",
        context_path="motor neuron -> releases -> acetylcholine -> binds -> nicotinic receptor",
    )
    print(f"pair check verdict: {'PASS' if ok else 'FAIL'}")
