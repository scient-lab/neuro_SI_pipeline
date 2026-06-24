#!/usr/bin/env python3
"""
derive_vocab.py — Derive canonical vocabulary from open-extraction Pass 1.

Two-pass extraction methodology (Prof Jha 2026-06-15 guidance):

  Pass 1: extract with OPEN vocabulary (no relation/category constraints)
          outputs: raw_triples.csv (free-text relations, bare nouns)
                           ↓
          [THIS SCRIPT] derive_vocab.py
          - normalize formatting (underscore → space, lowercase)
          - use Qwen3-14B to find semantic synonyms
          - rank by frequency (on canonical forms)
          - apply conservative threshold (50-100+ occurrences)
          - output derived vocabulary for Pass 2
                           ↓
  Pass 2: extract with CLOSED vocabulary (using derived vocab)
          outputs: seed_kg.csv (high-quality, vocabulary-filtered)

Usage:
  python scripts/analysis/derive_vocab.py \\
    --run-id 20260624-074435-pilot-f94c515 \\
    --raw-triples outputs/20260624-074435-pilot-f94c515/graphrag/output/raw_triples.csv \\
    --output-dir outputs/20260624-074435-pilot-f94c515/vocabulary_derivation \\
    --relation-threshold 50 \\
    --category-threshold 10

Outputs:
  derived_relations.yaml      — YAML for domains/<domain>_derived.yaml::relations
  derived_entity_categories.yaml — YAML for domains/<domain>_derived.yaml::entity_categories
  derivation_analysis.json    — machine-readable full analysis (S3-synced)
  derivation_report.md        — human-readable summary for Niraj review (S3-synced)
"""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Tuple, Set
import pandas as pd

# Try to import vLLM for LLM inference
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("WARNING: vLLM not found. Install with: pip install vllm", file=sys.stderr)

# Try to import pipeline config loader
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline_config import get_phase_param
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("WARNING: pipeline_config not found. Will use CLI args only.", file=sys.stderr)

def normalize_text(text: str) -> str:
    """Normalize text: replace underscores with spaces, lowercase, strip."""
    return text.replace('_', ' ').lower().strip()

def group_synonyms_with_llm(unique_items: List[str], item_type: str = "relation", llm_model: str = None) -> Dict[str, List[str]]:
    """
    Use LLM to find semantic synonyms among unique items.
    Groups items that mean the same thing semantically.

    Returns: {canonical_form: [item1, item2, ...], ...}
    """
    # Default to models.extract from config if not provided
    if llm_model is None:
        if HAS_CONFIG:
            llm_model = get_phase_param('models', 'extract', 'Qwen/Qwen3-14B')
        else:
            llm_model = 'Qwen/Qwen3-14B'

    if not HAS_VLLM:
        print(f"WARNING: Cannot use LLM canonicalization without vLLM. Falling back to format normalization only.",
              file=sys.stderr)
        # Fallback: group by normalized form only
        groups = {}
        for item in unique_items:
            normalized = normalize_text(item)
            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(item)
        return groups

    if not unique_items or len(unique_items) <= 1:
        return {normalize_text(unique_items[0]): unique_items} if unique_items else {}

    print(f"\nUsing {llm_model} to find {item_type} synonyms ({len(unique_items)} unique items)...")

    # Initialize vLLM with configured model
    try:
        llm = LLM(model=llm_model, dtype="float16", gpu_memory_utilization=0.7)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2000, top_p=0.95)
    except Exception as e:
        print(f"ERROR: Could not load {llm_model} via vLLM: {e}", file=sys.stderr)
        # Fallback to format normalization
        groups = {}
        for item in unique_items:
            normalized = normalize_text(item)
            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(item)
        return groups

    # Batch items for LLM processing (e.g., 10 items at a time)
    batch_size = 10
    groups = {}
    processed = set()

    for i in range(0, len(unique_items), batch_size):
        batch = unique_items[i:i+batch_size]
        unprocessed_batch = [item for item in batch if item not in processed]

        if not unprocessed_batch:
            continue

        # Create a prompt to ask the LLM to find synonyms
        prompt = f"""You are analyzing {item_type} names from a knowledge graph extraction.
Your task: identify which items in this list are synonyms (mean the same thing semantically).

Items to analyze:
{chr(10).join(f"- {item}" for item in unprocessed_batch)}

For each item, output its canonical form (pick the clearest/most standard name from the list if it's a synonym group, or keep it as-is if unique).

Output ONLY JSON (no other text):
{{
  "canonical_forms": {{
    "item1": "canonical_form_1",
    "item2": "canonical_form_1",  # same group as item1
    "item3": "canonical_form_3"   # unique
  }}
}}"""

        try:
            outputs = llm.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text.strip()

            # Parse JSON response
            try:
                response_json = json.loads(response_text)
                canonical_forms = response_json.get('canonical_forms', {})

                for item, canonical in canonical_forms.items():
                    canonical_norm = normalize_text(canonical)
                    if canonical_norm not in groups:
                        groups[canonical_norm] = []
                    if item not in groups[canonical_norm]:
                        groups[canonical_norm].append(item)
                    processed.add(item)
            except json.JSONDecodeError:
                print(f"WARNING: Could not parse LLM response for batch {i//batch_size}. Response: {response_text[:200]}",
                      file=sys.stderr)
                # Fallback for this batch
                for item in unprocessed_batch:
                    normalized = normalize_text(item)
                    if normalized not in groups:
                        groups[normalized] = []
                    groups[normalized].append(item)
                    processed.add(item)
        except Exception as e:
            print(f"WARNING: LLM batch processing failed: {e}. Using format normalization for this batch.",
                  file=sys.stderr)
            for item in unprocessed_batch:
                normalized = normalize_text(item)
                if normalized not in groups:
                    groups[normalized] = []
                groups[normalized].append(item)
                processed.add(item)

    return groups

def load_raw_triples(raw_triples_path: Path) -> pd.DataFrame:
    """Load raw triples from Pass 1 open extraction."""
    if not raw_triples_path.exists():
        raise FileNotFoundError(f"Raw triples not found: {raw_triples_path}")

    df = pd.read_csv(raw_triples_path)
    required_cols = {'subject', 'predicate', 'object'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Raw triples must have columns: {required_cols}. Found: {df.columns.tolist()}")

    return df

def derive_relations(df: pd.DataFrame, threshold: int = 50, llm_model: str = None) -> Dict[str, Any]:
    """Derive canonical relations using LLM-based synonym grouping."""
    unique_relations = df['predicate'].dropna().unique().tolist()
    print(f"Found {len(unique_relations)} unique relations in Pass 1 extraction")

    # Use LLM to group synonyms
    relation_groups = group_synonyms_with_llm(unique_relations, item_type="relation", llm_model=llm_model)

    # Map original relations to canonical forms
    canonical_mapping = {}
    for canonical, variants in relation_groups.items():
        for variant in variants:
            canonical_mapping[variant] = canonical

    # Recount with canonical forms
    df['predicate_canonical'] = df['predicate'].map(canonical_mapping)
    relation_counts = df['predicate_canonical'].value_counts()

    # Apply threshold
    relations_above_threshold = [
        {
            'relation': rel,
            'count': int(count),
            'percentage': round(100.0 * count / len(df), 2)
        }
        for rel, count in relation_counts.items()
        if count >= threshold
    ]

    relations_above_threshold = sorted(relations_above_threshold, key=lambda x: x['count'], reverse=True)

    return {
        'total_triples': len(df),
        'unique_relations_raw': len(unique_relations),
        'unique_relations_canonical': len(relation_counts),
        'canonical_mapping': {k: v for k, v in sorted(canonical_mapping.items())},
        'threshold': threshold,
        'relations_above_threshold': relations_above_threshold,
        'all_relations_ranked': sorted(
            [{'relation': r, 'count': int(c), 'percentage': round(100.0*c/len(df), 2)}
             for r, c in relation_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )
    }

def derive_categories(df: pd.DataFrame, threshold: int = 10, llm_model: str = None) -> Dict[str, Any]:
    """Derive canonical entity categories using LLM-based synonym grouping."""
    categories = []

    if 'subject_type' in df.columns:
        categories.extend(df['subject_type'].dropna().unique().tolist())
    if 'object_type' in df.columns:
        categories.extend(df['object_type'].dropna().unique().tolist())

    if not categories:
        return {
            'total_triples': len(df),
            'unique_categories_raw': 0,
            'unique_categories_canonical': 0,
            'threshold': threshold,
            'categories_above_threshold': [],
            'note': 'No entity type columns in raw triples. Pass 1 may not have typed extraction.'
        }

    unique_categories = list(set(categories))
    print(f"Found {len(unique_categories)} unique entity categories in Pass 1 extraction")

    # Use LLM to group synonyms
    category_groups = group_synonyms_with_llm(unique_categories, item_type="entity category", llm_model=llm_model)

    # Map original categories to canonical forms
    canonical_mapping = {}
    for canonical, variants in category_groups.items():
        for variant in variants:
            canonical_mapping[variant] = canonical

    # Recount with canonical forms
    categories_canonical = [canonical_mapping[c] for c in categories]
    category_counts = Counter(categories_canonical)

    # Apply threshold
    categories_above_threshold = [
        {
            'category': cat,
            'count': int(count),
            'percentage': round(100.0 * count / len(categories_canonical), 2)
        }
        for cat, count in category_counts.items()
        if count >= threshold
    ]

    categories_above_threshold = sorted(categories_above_threshold, key=lambda x: x['count'], reverse=True)

    return {
        'total_entities': len(categories_canonical),
        'unique_categories_raw': len(unique_categories),
        'unique_categories_canonical': len(set(categories_canonical)),
        'canonical_mapping': {k: v for k, v in sorted(canonical_mapping.items())},
        'threshold': threshold,
        'categories_above_threshold': categories_above_threshold,
        'all_categories_ranked': sorted(
            [{'category': c, 'count': int(cnt), 'percentage': round(100.0*cnt/len(categories_canonical), 2)}
             for c, cnt in category_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )
    }

def generate_yaml_relations(relations_data: Dict[str, Any]) -> str:
    """Generate YAML snippet for domains/<domain>_derived.yaml::relations."""
    yaml_lines = ["relations:"]

    for item in relations_data['relations_above_threshold']:
        rel = item['relation']
        desc = f"Frequency: {item['count']} occurrences ({item['percentage']}%)"
        yaml_lines.append(f"  - {{ id: {rel}, description: \"{desc}\" }}")

    return '\n'.join(yaml_lines)

def generate_yaml_categories(categories_data: Dict[str, Any]) -> str:
    """Generate YAML snippet for domains/<domain>_derived.yaml::entity_categories."""
    yaml_lines = ["entity_categories:"]

    for item in categories_data['categories_above_threshold']:
        cat = item['category']
        desc = f"Frequency: {item['count']} occurrences ({item['percentage']}%)"
        yaml_lines.append(f"  - id: {cat}")
        yaml_lines.append(f"    description: \"{desc}\"")

    return '\n'.join(yaml_lines)

def generate_markdown_report(relations_data: Dict[str, Any], categories_data: Dict[str, Any]) -> str:
    """Generate human-readable derivation report."""
    report = f"""# Vocabulary Derivation Report (Qwen3-14B Canonicalization)

**Analysis Date:** {datetime.utcnow().isoformat()}Z
**Source:** Open-vocabulary extraction (Pass 1)
**Canonicalization Method:** Qwen3-14B semantic synonym grouping

## Summary

### Relations
- **Total triples:** {relations_data['total_triples']:,}
- **Unique relations (raw):** {relations_data['unique_relations_raw']}
- **Unique relations (canonical):** {relations_data['unique_relations_canonical']}
- **Threshold:** {relations_data['threshold']} occurrences
- **Derived relations:** {len(relations_data['relations_above_threshold'])}

### Entity Categories
- **Total entities:** {categories_data['total_entities']}
- **Unique categories (raw):** {categories_data['unique_categories_raw']}
- **Unique categories (canonical):** {categories_data['unique_categories_canonical']}
- **Threshold:** {categories_data['threshold']} occurrences
- **Derived categories:** {len(categories_data['categories_above_threshold'])}

## Top Relations by Frequency

| Rank | Relation | Count | % |
|------|----------|-------|---|
"""

    for idx, item in enumerate(relations_data['relations_above_threshold'][:20], 1):
        report += f"| {idx} | `{item['relation']}` | {item['count']} | {item['percentage']}% |\n"

    if len(relations_data['relations_above_threshold']) > 20:
        report += f"| ... | ... and {len(relations_data['relations_above_threshold']) - 20} more | | |\n"

    report += f"\n## Top Entity Categories by Frequency\n\n"
    report += f"| Rank | Category | Count | % |\n"
    report += f"|------|----------|-------|---|\n"

    for idx, item in enumerate(categories_data['categories_above_threshold'][:15], 1):
        report += f"| {idx} | `{item['category']}` | {item['count']} | {item['percentage']}% |\n"

    if len(categories_data['categories_above_threshold']) > 15:
        report += f"| ... | ... and {len(categories_data['categories_above_threshold']) - 15} more | | |\n"

    report += f"""
## Canonicalization Mapping

Qwen3-14B grouped semantically similar items into canonical forms:

### Relations Grouped
```
{json.dumps(relations_data['canonical_mapping'], indent=2)}
```

### Categories Grouped
```
{json.dumps(categories_data['canonical_mapping'], indent=2)}
```

## Next Steps

1. **Review** this report with Prof. Jha
2. **Validate** canonicalization (Qwen3 groupings make sense?)
3. **Create** `domains/<domain>_derived.yaml` using YAML snippets below
4. **Run Pass 2** extraction with derived vocabulary
5. **Compare** Pass 2 seed KG quality vs Pass 1

## Derived Relations (YAML for domains/<domain>_derived.yaml)

```yaml
{generate_yaml_relations(relations_data)}
```

## Derived Entity Categories (YAML for domains/<domain>_derived.yaml)

```yaml
{generate_yaml_categories(categories_data)}
```
"""
    return report

def main():
    # Default LLM model from config
    default_llm_model = None
    if HAS_CONFIG:
        try:
            default_llm_model = get_phase_param('models', 'extract', 'Qwen/Qwen3-14B')
        except:
            default_llm_model = 'Qwen/Qwen3-14B'
    else:
        default_llm_model = 'Qwen/Qwen3-14B'

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--run-id', required=True, help='Run ID for outputs/')
    parser.add_argument('--raw-triples', required=True, help='Path to raw_triples.csv from Pass 1')
    parser.add_argument('--output-dir', required=True, help='Output directory for derived vocabulary')
    parser.add_argument('--relation-threshold', type=int, default=50,
                       help='Min occurrences to include relation (default: 50)')
    parser.add_argument('--category-threshold', type=int, default=10,
                       help='Min occurrences to include category (default: 10)')
    parser.add_argument('--llm-model', default=default_llm_model,
                       help=f'LLM model for semantic canonicalization (default: {default_llm_model} from profile config)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_triples_path = Path(args.raw_triples)

    print(f"Loading raw triples from Pass 1: {raw_triples_path}")
    df = load_raw_triples(raw_triples_path)
    print(f"Loaded {len(df):,} triples")

    print(f"\nDeriving canonical relations (threshold: {args.relation_threshold})")
    relations_data = derive_relations(df, threshold=args.relation_threshold, llm_model=args.llm_model)
    print(f"Derived {len(relations_data['relations_above_threshold'])} canonical relations")

    print(f"\nDeriving canonical categories (threshold: {args.category_threshold})")
    categories_data = derive_categories(df, threshold=args.category_threshold, llm_model=args.llm_model)
    print(f"Derived {len(categories_data['categories_above_threshold'])} canonical categories")

    # Write JSON analysis
    analysis = {
        'metadata': {
            'run_id': args.run_id,
            'raw_triples_path': str(raw_triples_path),
            'analysis_date': datetime.utcnow().isoformat() + 'Z',
            'canonicalization_method': 'Qwen3-14B semantic synonym grouping',
            'thresholds': {
                'relations': args.relation_threshold,
                'categories': args.category_threshold
            }
        },
        'relations': relations_data,
        'categories': categories_data
    }

    json_path = output_dir / "derivation_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✓ Wrote analysis: {json_path}")

    # Write Markdown report
    md_path = output_dir / "derivation_report.md"
    report = generate_markdown_report(relations_data, categories_data)
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"✓ Wrote report: {md_path}")

    # Write derived relations YAML
    yaml_rel_path = output_dir / "derived_relations.yaml"
    with open(yaml_rel_path, 'w') as f:
        f.write(generate_yaml_relations(relations_data))
    print(f"✓ Wrote derived relations: {yaml_rel_path}")

    # Write derived categories YAML
    yaml_cat_path = output_dir / "derived_entity_categories.yaml"
    with open(yaml_cat_path, 'w') as f:
        f.write(generate_yaml_categories(categories_data))
    print(f"✓ Wrote derived categories: {yaml_cat_path}")

    print(f"\n✅ Vocabulary derivation complete!")
    print(f"\nOutputs in: {output_dir}")
    print(f"\nNext: create domains/<domain>_derived.yaml using YAML snippets")
    print(f"Then: SI_DOMAIN=<domain>_derived ./scripts/pipeline.sh --phase extract")

if __name__ == '__main__':
    main()
