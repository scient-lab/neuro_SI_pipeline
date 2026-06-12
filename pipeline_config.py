"""pipeline_config - config loader for the neuro_SI orchestration pipeline.

All configuration lives in this repo:

    configs/default.yaml                  - operational defaults
    configs/profiles/<profile>.yaml       - scaling profile overrides
    configs/platforms/<platform>.yaml     - platform-specific overrides
    domains/<domain>.yaml                  - domain vocabulary + few-shot + focus
    prompts/<phase>.yaml                   - LLM prompt templates

The loader walks the layers at runtime and returns a merged dict.

Environment variables (all optional):

    SI_DOMAIN     - which domain (default: 'neuroscience').
    SI_PROFILE    - scaling profile (smoke / pilot / paper). Optional; if unset,
                     no profile layer is applied.
    SI_PLATFORM   - platform (local / runpod / aws / princeton). Optional.

API:

    load_config()                                 -> dict   (merged)
    get_relations()                               -> list[str]
    get_relation_descriptions()                   -> dict[str, str]
    get_entity_categories()                       -> list[str]
    get_entity_category_descriptions()            -> dict[str, str]
    get_few_shot_examples()                       -> list[dict]
    get_focus_instructions()                      -> str
    get_domain()                                  -> str
    get_model_id(key, default=None)               -> str | None
    get_phase_param(phase, key, default=None)     -> Any
    get_platform_param(key, default=None)         -> Any
    get_prompt(name)                              -> dict
    describe(prefix='  ')                          -> str   (diagnostic)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml


_REPO_ROOT = Path(__file__).resolve().parent


# --- Env resolution ------------------------------------------------------

def get_si_home() -> Path:
    return _REPO_ROOT


def get_domain_name() -> str:
    return os.environ.get("SI_DOMAIN", "neuroscience").strip()


def get_profile_name() -> Optional[str]:
    name = os.environ.get("SI_PROFILE", "").strip()
    return name or None


def get_platform_name() -> Optional[str]:
    name = os.environ.get("SI_PLATFORM", "").strip()
    return name or None


# --- Merge primitives ----------------------------------------------------

def _read(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must parse to a dict; got {type(data).__name__}")
    return data


def _merge(base: Any, overlay: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for k, v in overlay.items():
        out[k] = _merge(base.get(k), v) if k in base else v
    return out


# --- Layered config load --------------------------------------------------

@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Merge the config layers in order (later wins on key conflict):

      1. configs/default.yaml
      2. domains/<SI_DOMAIN>.yaml
      3. configs/profiles/<SI_PROFILE>.yaml      (if SI_PROFILE set)
      4. configs/platforms/<SI_PLATFORM>.yaml    (if SI_PLATFORM set)
    """
    home = _REPO_ROOT
    domain = get_domain_name()

    cfg: dict[str, Any] = {}
    cfg = _merge(cfg, _read(home / "configs" / "default.yaml"))
    cfg = _merge(cfg, _read(home / "domains" / f"{domain}.yaml"))

    profile = get_profile_name()
    if profile:
        cfg = _merge(cfg, _read(home / "configs" / "profiles" / f"{profile}.yaml"))

    platform = get_platform_name()
    if platform:
        cfg = _merge(cfg, _read(home / "configs" / "platforms" / f"{platform}.yaml"))

    return cfg


# --- Vocabulary helpers ---------------------------------------------------

def get_domain(default: Optional[str] = None) -> Optional[str]:
    return load_config().get("domain", default)


def get_relations() -> list[str]:
    return [r["id"] if isinstance(r, dict) else r for r in load_config().get("relations", [])]


def get_relation_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    for r in load_config().get("relations", []):
        if isinstance(r, dict):
            out[r["id"]] = r.get("description", "")
    return out


def get_entity_categories() -> list[str]:
    return [c["id"] if isinstance(c, dict) else c for c in load_config().get("entity_categories", [])]


def get_entity_category_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    for c in load_config().get("entity_categories", []):
        if isinstance(c, dict):
            out[c["id"]] = c.get("description", "")
    return out


def get_few_shot_examples() -> list[dict[str, Any]]:
    return load_config().get("few_shot_examples", []) or []


def get_focus_instructions() -> str:
    return load_config().get("focus_instructions", "") or ""


# --- Operational helpers --------------------------------------------------

def get_model_id(key: str, default: Optional[str] = None) -> Optional[str]:
    """cfg['models'][key] or default. e.g. get_model_id('extract')."""
    return (load_config().get("models") or {}).get(key, default)


def get_phase_param(phase: str, key: str, default: Any = None) -> Any:
    """cfg[<phase>][<key>] or default."""
    return (load_config().get(phase) or {}).get(key, default)


def get_platform_param(key: str, default: Any = None) -> Any:
    """cfg['platform'][<key>] or default."""
    return (load_config().get("platform") or {}).get(key, default)


# --- Prompt loader --------------------------------------------------------

def get_prompt(name: str) -> dict[str, Any]:
    """Return the prompt template for a phase by name.

    Lookup order (first hit wins):
      1. prompts/overrides/<SI_DOMAIN>/<name>.yaml   (per-domain override)
      2. prompts/<name>.yaml                          (canonical)
    """
    domain = get_domain_name()
    override = _REPO_ROOT / "prompts" / "overrides" / domain / f"{name}.yaml"
    if override.is_file():
        return _read(override)
    return _read(_REPO_ROOT / "prompts" / f"{name}.yaml")


# --- Diagnostic ----------------------------------------------------------

def describe(prefix: str = "  ") -> str:
    cfg = load_config()
    lines = [
        f"{prefix}SI_HOME         : {_REPO_ROOT}",
        f"{prefix}SI_DOMAIN       : {get_domain_name()}",
        f"{prefix}SI_PROFILE      : {get_profile_name() or '(unset)'}",
        f"{prefix}SI_PLATFORM     : {get_platform_name() or '(unset)'}",
        f"{prefix}categories      : {len(cfg.get('entity_categories') or [])}",
        f"{prefix}relations       : {len(cfg.get('relations') or [])}",
        f"{prefix}few_shot        : {len(cfg.get('few_shot_examples') or [])}",
        f"{prefix}models          : {list((cfg.get('models') or {}).keys())}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(prefix=""))
