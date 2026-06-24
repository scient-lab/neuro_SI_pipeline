"""Tokenizer compat shim for vLLM 0.7.3 + Qwen3.

Some `transformers` versions ship `Qwen2Tokenizer` (slow variant used for
Qwen3 in our si_curriculum venv) without `all_special_tokens_extended` —
a property normally defined on SpecialTokensMixin. Both vLLM 0.7.3's
internal tokenizer wrapper AND `transformers.apply_chat_template` may
access it. When missing, callers crash with:

  AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended

This module monkey-patches a backport: a property that wraps the
existing `all_special_tokens` list as `AddedToken` instances. Good
enough for chat-template rendering and vLLM's tokenizer probing —
both only read the string forms of special tokens, not their
`AddedToken` metadata fields.

Usage:
    # In ANY file that may load Qwen3 (directly or via vLLM):
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # to repo root
    import _tokenizer_compat  # noqa: F401  # SIDE EFFECT: monkey-patches Qwen2Tokenizer

Affected callers (any vLLM 0.7.3 + Qwen3 user, currently in
.venvs/si_curriculum/):
  - 3_si_curriculum/curriculum_generator/verify_questions.py
  - 3_si_curriculum/training/data_prep.py
  - 3_si_curriculum/training/trainer.py
  - 3_si_curriculum/RL/rl_training.py
  - 3_si_curriculum/test_models/eval_models.py

Remove this shim when:
  - vLLM upgrades past 0.10.x (chat-template handling fixed upstream), AND
  - All venvs resolve to a `transformers` version where Qwen2Tokenizer
    natively exposes `all_special_tokens_extended`.
"""
from __future__ import annotations


def _patch_class(cls) -> None:
    """Add `all_special_tokens_extended` to `cls` if missing.

    The property returns AddedToken-wrapped versions of `all_special_tokens`
    strings, matching the contract that callers expect (list of token-like
    objects with `.content` attribute on each).
    """
    existing = getattr(cls, 'all_special_tokens_extended', None)
    if isinstance(existing, property):
        return  # already provided natively

    try:
        from transformers.tokenization_utils import AddedToken
    except ImportError:
        return  # transformers not available; nothing to patch

    @property
    def _patched(self):
        return [tok if not isinstance(tok, str) else AddedToken(tok)
                for tok in self.all_special_tokens]

    cls.all_special_tokens_extended = _patched


# Patch the slow variant (used by Qwen3 in some configs).
try:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    _patch_class(Qwen2Tokenizer)
except ImportError:
    pass

# Patch the fast variant too — defense in depth, in case a future model
# config flips to the fast tokenizer and hits the same bug.
try:
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    _patch_class(Qwen2TokenizerFast)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# vLLM 0.7.3 + Mistral fix: MistralCommonBackend strict-validates kwargs.
# ---------------------------------------------------------------------------
# vLLM passes internal kwargs (max_loras, _from_auto, ...) to
# AutoTokenizer.from_pretrained. For Mistral models, transformers' newer
# `MistralCommonBackend.from_pretrained` rejects ANY kwarg it doesn't
# recognize with:
#   ValueError: Some kwargs in ['max_loras', '_from_auto'] are not
#   supported by `MistralCommonBackend.from_pretrained`.
#
# Workaround: wrap from_pretrained to silently drop the vLLM-internal
# kwargs the backend doesn't understand. The dropped kwargs aren't
# load-time configuration the tokenizer needs anyway (max_loras is a
# vLLM LoRA-tracking shim; _from_auto is just AutoTokenizer's internal
# routing marker).
try:
    from transformers.tokenization_mistral_common import MistralCommonBackend

    _orig_mcb_from_pretrained = MistralCommonBackend.from_pretrained.__func__

    # Kwargs vLLM 0.7.3 passes that newer MistralCommonBackend rejects.
    # Update this set if a future vLLM or transformers version adds more.
    _VLLM_INTERNAL_KWARGS = ('max_loras', '_from_auto', 'trust_remote_code',
                              'gpu_memory_utilization', 'tensor_parallel_size')

    @classmethod
    def _patched_mcb_from_pretrained(cls, *args, **kwargs):
        for k in _VLLM_INTERNAL_KWARGS:
            kwargs.pop(k, None)
        return _orig_mcb_from_pretrained(cls, *args, **kwargs)

    MistralCommonBackend.from_pretrained = _patched_mcb_from_pretrained
except ImportError:
    pass  # transformers too old to have MistralCommonBackend; nothing to patch
