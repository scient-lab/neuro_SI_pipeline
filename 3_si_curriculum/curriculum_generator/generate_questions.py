'''
Copyright (c) 2024 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha
Modified by: Jake Stephen 

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import networkx as nx
import json
import random
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from google import genai
from typing import Optional
import os
import re
import time

# Pipeline config loader (repo root, 2 levels up from this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_config import get_model_id, get_phase_param, get_relations, get_exception_config  # noqa: E402

# ================= MODEL CONFIGURATION =================
# Sourced from configs/default.yaml::models (overridable via domain/profile).
# Defaults match the Stephen & Jha 2026 paper's curriculum-generation stack.
MODEL_GENERATION  = get_model_id('curriculum_generation',  'gemini-2.5-flash')   # 1. Initial Question Generation
MODEL_QUALITY     = get_model_id('curriculum_quality',     'gemini-2.5-flash')   # 2. Quality/Formatting Check
MODEL_TRACE       = get_model_id('curriculum_trace',       'gemini-2.5-flash')   # 3. Thinking Trace/Explanation
MODEL_CORRECTNESS = get_model_id('curriculum_correctness', 'gemini-2.5-flash')   # 4. Final Verification/Correctness
# =======================================================

# ================= CONCISENESS CONFIG ==================
# Target word count for thinking traces
TRACE_TARGET_WORDS = 250      # Aim for this many words in the explanation
TRACE_MAX_WORDS = 350       # Soft limit: checker rejects above this
TRACE_HARD_MAX_WORDS = 350  # Hard limit: truncated/removed in post-processing
TRACE_MIN_WORDS = 100        # Too short = not useful for training
# =======================================================

# ================= THINKING BUDGET =====================
# Sourced from configs/default.yaml::curriculum.thinking_budget (default 4096).
# Smoke profile drops to 512 for ~8x faster per-call latency. Per Jha
# 2026-06-04 mandate ("reasoning LLMs mandatory at every stage"), do NOT
# set to 0 in production; trim aggressively only when smoke-validating.
THINKING_BUDGET = get_phase_param('curriculum', 'thinking_budget', 4096)
# =======================================================

# ================= RETRY SEMANTICS =====================
# Sourced from configs/exceptions.yaml::gemini. Single source of
# truth for which Gemini SDK error messages count as transient (retry
# with exponential backoff) vs. permanent (fail fast). Fallback defaults
# preserve the pre-YAML hardcoded behavior so the file going missing
# doesn't crash the run, only widens the retry envelope slightly.
_GEMINI_RETRY_CFG       = get_exception_config('gemini')
_GEMINI_TRANSIENT       = tuple(str(m).lower()
                                 for m in _GEMINI_RETRY_CFG.get('transient_markers',
                                                                ['429', 'resource']))
_GEMINI_INITIAL_DELAY_S = float(_GEMINI_RETRY_CFG.get('initial_delay_seconds', 4.0))
_GEMINI_MAX_RETRIES     = int(_GEMINI_RETRY_CFG.get('max_retries', 5))
# =======================================================


def _extract_model_text(response) -> Optional[str]:
    """
    Extract the actual model output text from a Gemini response,
    skipping any 'thought' parts (internal reasoning from thinking models).
    
    Gemini 2.5 Flash with thinking enabled returns multiple parts:
      - parts with thought=True are internal reasoning (NOT the answer)
      - parts without thought=True (or thought=False) are the actual output
    
    We want the last non-thought text part.
    """
    if not response or not hasattr(response, 'candidates') or not response.candidates:
        return None
    
    candidate = response.candidates[0]
    if not hasattr(candidate, 'content') or not candidate.content or not candidate.content.parts:
        return None
    
    # Collect all non-thought text parts
    text_parts = []
    for part in candidate.content.parts:
        # Skip thinking/reasoning parts
        if hasattr(part, 'thought') and part.thought:
            continue
        if hasattr(part, 'text') and part.text:
            text_parts.append(part.text)
    
    if text_parts:
        # Return the last non-thought text part (the actual output)
        return text_parts[-1]
    
    # Fallback: if no non-thought parts found, try the last part anyway
    # (this handles models that don't use the thought attribute)
    last_part = candidate.content.parts[-1]
    if hasattr(last_part, 'text') and last_part.text:
        return last_part.text
    
    return None


class PathGenerator:
    def __init__(self, vocab_path: str, graph_path: str, icd10_categories_path: Optional[str], vocab_freq_path: str = None):
        self.vocab = self._load_vocab(vocab_path)
        self.concept2id = {w: i for i, w in enumerate(self.vocab)}
        self.graph = self._load_graph(graph_path)
        self.icd10_categories = self._load_icd10_categories(icd10_categories_path)
        self.vocab_freq = {vocab: 0 for vocab in self.vocab}
        if vocab_freq_path is not None and os.path.exists(vocab_freq_path):
            self.vocab_freq = self.__update_vocab_freq(vocab_freq_path)
            print("Updated vocab freq")
        # Active relation list from merged pipeline config (single source of
        # truth, shared with 2_graphmert/predict_tails_llm.py::ALLOWED_RELATIONS
        # and 1_seed_kg/prompts_kg.py::get_relation_types()).
        self.merged_relations = get_relations()

        self.HUB_REMOVAL_PERCENTILE = 0.01
        self.PRUNE_TRANSITIVE = True
        self.EXCLUDED_RELATIONS = {
            'represents'
        }
        self.top_hubs = self._identify_hubs()

    def _identify_hubs(self) -> set:
        degrees = dict(self.graph.degree())
        num_hubs = int(len(degrees) * self.HUB_REMOVAL_PERCENTILE)
        if num_hubs < 1:
            num_hubs = 1
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return set(n for n, d in sorted_nodes[:num_hubs])

    def __update_vocab_freq(self, path: str) -> Dict:
        with open(path, 'r') as f:
            vocab_freq = json.load(f)
        for vocab in vocab_freq:
            self.vocab_freq[vocab] = vocab_freq[vocab]
        return self.vocab_freq

    def _load_vocab(self, path: str) -> List[str]:
        with open(path, 'r') as f:
            return f.read().splitlines()

    def _load_graph(self, path: str) -> nx.Graph:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_icd10_categories(self, path: Optional[str]) -> Dict:
        if path is None or not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _get_k_hop_path_dfs(self, start_node: int, k: int) -> Tuple[List[Tuple[int, int, str]], bool]:
        start_neighbors = set()
        if self.PRUNE_TRANSITIVE:
            for neighbor in self.graph.neighbors(start_node):
                start_neighbors.add(neighbor)
        init_hub_count = 1 if start_node in self.top_hubs else 0
        return self._dfs_recursive(start_node, [], {start_node}, start_neighbors, 0, init_hub_count, k)

    def _dfs_recursive(self, current_node, path, visited, start_neighbors, current_exc_count, current_hub_count, target_depth):
        path_len = len(path)
        if path_len == target_depth:
            return path, True
        if self.PRUNE_TRANSITIVE and path_len > 1 and current_node in start_neighbors:
            return [], False

        neighbors = list(self.graph.neighbors(current_node))
        random.shuffle(neighbors)

        for neighbor in neighbors:
            rel_idx = self.graph[current_node][neighbor][0]['rel']
            if rel_idx >= len(self.merged_relations):
                rel_idx = rel_idx - len(self.merged_relations)
            relation_str = self.merged_relations[rel_idx]

            is_excluded = 1 if relation_str in self.EXCLUDED_RELATIONS else 0
            is_hub = 1 if neighbor in self.top_hubs else 0
            new_exc_count = current_exc_count + is_excluded
            new_hub_count = current_hub_count + is_hub

            if (new_exc_count + new_hub_count) > 1:
                continue
            if neighbor in visited:
                continue
            if self.PRUNE_TRANSITIVE and path_len >= 1 and neighbor in start_neighbors:
                continue

            visited.add(neighbor)
            path.append((current_node, neighbor, relation_str))
            result_path, found = self._dfs_recursive(
                neighbor, path, visited, start_neighbors, new_exc_count, new_hub_count, target_depth
            )
            if found:
                return result_path, True
            path.pop()
            visited.remove(neighbor)

        return [], False

    def generate_paths(self, category: Optional[str], k_hops: int = 1) -> Dict:
        concepts_in_category = None
        if category is not None and self.icd10_categories:
            concepts_in_category = self.icd10_categories.get(category)

        max_attempts = 20
        for _ in range(max_attempts):
            if concepts_in_category:
                sampled_concept = random.sample(concepts_in_category, 1)[0]
            else:
                concepts = list(self.vocab_freq.keys())
                inverse_freqs = [1.0 / (freq + 1e-10) for freq in self.vocab_freq.values()]
                total = sum(inverse_freqs)
                probs = [freq / total for freq in inverse_freqs]
                sampled_concept = random.choices(concepts, weights=probs, k=1)[0]

            if sampled_concept not in self.concept2id:
                continue

            concept_id = self.concept2id[sampled_concept]
            if not self.graph.has_node(concept_id):
                print(f"Node {concept_id} ({sampled_concept}) found in vocab but missing from graph.")
                continue

            paths, success = self._get_k_hop_path_dfs(concept_id, k_hops)
            if success:
                return {
                    "source_concept": sampled_concept,
                    "paths": [
                        {"start": self.vocab[start], "end": self.vocab[end], "relation": relation}
                        for start, end, relation in paths
                    ],
                    'target_concept': self.vocab[paths[-1][1]],
                }

        raise ValueError(f"Could not find valid path after {max_attempts} attempts")


class GeminiLLMBackend:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=self.api_key)
        self.model_generation = MODEL_GENERATION
        self.model_quality = MODEL_QUALITY
        self.model_trace = MODEL_TRACE
        self.model_correctness = MODEL_CORRECTNESS

    def _generate_with_retry(self, model: str, contents: str, config: Optional[Dict] = None, retries: Optional[int] = None):
        # Retry parameters sourced from configs/exceptions.yaml::gemini
        # via module-level constants. CLI `retries=` override still wins so
        # callers can short-circuit during smoke tests.
        if retries is None:
            retries = _GEMINI_MAX_RETRIES
        delay = _GEMINI_INITIAL_DELAY_S
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                if response is None or not hasattr(response, 'candidates') or not response.candidates:
                    raise ValueError("Empty response from API")
                return response
            except Exception as e:
                error_str = str(e).lower()
                is_transient = any(m in error_str for m in _GEMINI_TRANSIENT)
                if is_transient:
                    if attempt < retries - 1:
                        print(f"Transient error ({e.__class__.__name__}). "
                              f"Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        print("Max retries reached.")
                        raise e
                else:
                    print(f"Generation error (non-transient): {e}")
                    raise e
        return None

    def validate_path_meaningfulness(self, paths: List[Dict]) -> Tuple[str, str]:
        """
        Quick LLM check to determine if a knowledge graph path is meaningful
        enough to generate a good multi-hop question from.
        
        For paths with ≤2 hops, validation is skipped (auto-valid).
        
        Returns: (verdict, reason)
            verdict: "Valid" or "Skip"
            reason: one-sentence explanation
        
        Defaults to Skip on any parse failure or error (conservative).
        """
        # Skip validation entirely for short paths (1-2 hops)
        num_hops = len(paths)
        if num_hops <= 2:
            return "Valid", ""

        paths_str = ', '.join([f"({p['start']} -> {p['relation']} -> {p['end']})" for p in paths])
        prompt = f"""You are a strict evaluator deciding whether a knowledge graph path can produce a genuinely challenging multi-hop neuroscience exam question. Be AGGRESSIVE about skipping — it is better to skip a borderline path than to waste an expensive generation call on it.

Path: {paths_str}

SKIP the path if the hops are:
1. TAUTOLOGICAL/DEFINITIONAL: Any hop where the end concept is essentially a restatement or direct definition of the start concept (e.g., "corpus callosum -> required_for -> interhemispheric communication" — the structure IS the function)
2. TRIVIAL CONTAINMENT: Any hop that is purely location, containment, or "part_of" with no mechanistic insight (e.g., "hippocampus -> located_in -> temporal lobe")
3. PREDICTABLE FROM ENDPOINTS: A student could  easily guess the correct answer knowing ONLY the source and target concepts without reasoning through intermediate hops at all (be hesitant to skip bc of this)

Respond with EXACTLY one line in this format:
Verdict: Valid|Skip — [one sentence reason]"""

        try:
            response = self._generate_with_retry(
                model=self.model_quality,  # cheap flash call
                contents=prompt
            )
            if response and response.candidates:
                raw = response.candidates[0].content.parts[0].text.strip()
                content = raw
                
                # Strip "Verdict:" prefix if present
                if content.startswith("Verdict:"):
                    content = content[len("Verdict:"):].strip()
                
                # Try to parse the dash-separated reason
                def extract_reason(text, fallback):
                    for sep in ["—", "-", "–"]:
                        if sep in text:
                            return text.split(sep, 1)[1].strip()
                    return fallback
                
                if content.lower().startswith("skip"):
                    reason = extract_reason(content, "Path deemed trivial")
                    #print(f"  [VALIDATOR] SKIP: {reason} | Path: {paths_str[:120]}")
                    return "Skip", reason
                elif content.lower().startswith("valid"):
                    reason = extract_reason(content, "Path is meaningful")
                    #print(f"  [VALIDATOR] VALID: {reason} | Path: {paths_str[:120]}")
                    return "Valid", reason
                else:
                    # Parse failure — default to SKIP (conservative)
                    #print(f"  [VALIDATOR] PARSE FAIL -> SKIP | Raw: {raw[:120]} | Path: {paths_str[:120]}")
                    return "Skip", f"Could not parse validator response, defaulting to skip: {raw[:80]}"
            
            # Empty response — default to SKIP
            #print(f"  [VALIDATOR] EMPTY RESPONSE -> SKIP | Path: {paths_str[:120]}")
            return "Skip", "Empty response from validator, defaulting to skip"
        except Exception as e:
            # Error — default to SKIP
            #print(f"  [VALIDATOR] ERROR -> SKIP: {e} | Path: {paths_str[:120]}")
            return "Skip", f"Validator error, defaulting to skip: {e}"

    def generate_question(self, source_concept: str, target_concept: str, paths: List[Dict]) -> Optional[str]:
        paths_str = ','.join([f"({path['start']} , {path['relation']}, {path['end']})" for path in paths])
        prompt = f"""
        You are designing an extremely high-difficulty neuroscience board exam question. Your goal is to write a question that CANNOT be answered correctly without explicitly reasoning through each step of a multi-hop knowledge chain.

        The knowledge chain is: {paths_str}
        Source concept: {source_concept}
        Target concept: {target_concept}

        COMPOSITIONAL REASONING REQUIREMENTS:
        - The question must require the student to mentally traverse EVERY hop in the chain above to reach the answer.
        - A student who only knows the source concept, or only knows the target concept, must fail.
        - some wrong options must be plausible given partial knowledge of the chain 

        VIGNETTE DESIGN:
        - Write a vignette that encodes the STARTING POINT of the chain in disguised form (describe the phenomenon without naming the concept directly).
        - Do NOT name {target_concept} directly anywhere in the question stem or options.
        - Some examples of vignettes are given below, but feel free to be creative as long as the above requirements are met. 
            - CLINICAL: A patient presents with symptoms in a hospital/clinic setting. Describe findings without naming the concept.
            - EXPERIMENTAL: A researcher observes results. Frame the chain as a finding or anomaly.
            - PHARMACOLOGICAL: A drug with a described mechanism (not named) produces an unexpected downstream effect. The student must trace the mechanism.
            - COMPARATIVE/EVOLUTIONARY: A non-human model organism (e.g., zebrafish, C. elegans, Drosophila) exhibits a described behavior or phenotype tied to the chain.
            - FORENSIC/PATHOLOGICAL: A post-mortem or biopsy finding initiates the chain. Used for structural or molecular concepts.


        OPTIONS:
        - Write 4 answer options (A, B, C, D). Exactly one must be correct; the other three must be wrong.
        - The correct answer requires full traversal of the knowledge chain.
        - Wrong options should be plausible to a student with partial knowledge 
        - The correct answer must correspond to the TARGET CONCEPT of the chain (or a direct description of it), not a related-but-different concept in the same neighborhood.
        - Options may be in the same conceptual neighborhood, this is expected and desirable. They should be distinct enough that only one is defensibly correct.
                
        Strict Output Format (use exactly these tags, no deviations, especially the tags):
        <Question>
        Write the vignette and question stem here. Do not label sections. Do not name source/target concepts directly.
        </Question>
        <Options>
        A. Option text
        B. Option text
        C. Option text
        D. Option text
        </Options>
        <Answer>
        Correct Option Letter (e.g., A)
        </Answer>
        """
        generation_config = {
            'thinking_config': {
                'include_thoughts': True,
                'thinking_budget': THINKING_BUDGET
            }
        }

        try:
            response = self._generate_with_retry(
                model=self.model_generation,
                contents=prompt,
                config=generation_config
            )
            if response and response.candidates:
                # Use helper to skip thinking parts and get actual output
                return _extract_model_text(response)
            return None
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

    def separate_question_and_answer(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        if not question:
            return None, None
        try:
            if '<Answer>' not in question or '</Answer>' not in question:
                print("Missing Answer tags in generated output.")
                return None, None

            question_extracted = question.split('<Answer>')[0].strip()
            answer_part = question.split('<Answer>')[1]
            answer = answer_part.split('</Answer>')[0].strip()
            return question_extracted, answer
        except Exception as e:
            print(f"Error parsing question/answer: {e}")
            return None, None

    def quality_filtering(self, question: str) -> bool:
        if not question:
            return False

        required_tags = ['<Question>', '</Question>', '<Options>', '</Options>']
        if not all(tag in question for tag in required_tags):
            print("Required tags not present")
            return False

        options = ['A.', 'B.', 'C.', 'D.']
        if not all(opt in question for opt in options):
            print("Options not present")
            return False

        for line in question.splitlines():
            line_strip = line.strip()
            for opt in ['A.', 'B.', 'C.', 'D.']:
                if line_strip.startswith(opt):
                    content = line_strip[len(opt):].strip()
                    if not content or (all(ord(c) < 128 for c in content) and len(content) < 3):
                        print(f"Option {opt} is too short or empty: '{content}'")
                        return False

        prompt = f"""
        You will be given a multiple choice neuroscience question. Evaluate whether the answer options are sufficiently distinct to be a fair question.
        Only respond with: 'Yes' or 'No', nothing else. "Yes' if the options are sufficiently different from each other, 'No' otherwise.
        Check the question: {question}
        """
        try:
            response = self._generate_with_retry(
                model=self.model_quality,
                contents=prompt
            )
            if response and response.candidates:
                content = response.candidates[0].content.parts[0].text
                content = content.strip().lower()
                if content == 'no':
                    print("Options are near duplicates")
                    return False
            else:
                return False
        except Exception as e:
            print(f"Error checking question quality: {e}")
            return False

        return True

    def generate_thinking_trace(self, question: str, paths: List[Dict], correct_answer: str) -> Optional[str]:
        """
        Generate a thinking trace that explains the correct answer.
        The correct answer letter is provided to ensure the trace matches.
        """
        paths_str = ','.join([f"({path['start']} , {path['relation']}, {path['end']})" for path in paths])
        prompt = f"""Generate a thinking trace for the following neuroscience question. The correct answer is {correct_answer}. Your job is to explain WHY {correct_answer} is correct and briefly why each other option is wrong.

Your explanation must be approximately {TRACE_TARGET_WORDS} words (no more than {TRACE_MAX_WORDS} words).

STRICT RULES:
- Be direct and decisive. State the reasoning, eliminate wrong options briefly, and commit to the answer.
- Do NOT hedge, repeat yourself, restate the question, or use filler phrases like "Let's break this down" or "This is interesting".
- Do NOT use headers, bullet points with bold labels, or numbered sections.
- Write in flowing prose, as if a knowledgeable neuroscientist is explaining their reasoning to a colleague.
- Cover each wrong option in 1 sentence max. Spend most words on WHY {correct_answer} is correct.
- End with a clear final statement identifying {correct_answer} as the correct answer.
- Do NOT arrive at a different answer. The correct answer is {correct_answer}.
Structure your explanation by walking through the chain step by step: first identify the starting concept from the vignette, then reason through each hop explicitly (concept A → relationship → concept B → ...), then explain why the final hop leads to the correct answer. Then briefly dismiss each distractor.
Use this context to inform your reasoning: {paths_str}

Question: {question}"""

        try:
            response = self._generate_with_retry(
                model=self.model_trace,
                contents=prompt,
            )
            if response and response.candidates:
                trace = response.candidates[0].content.parts[0].text

                # Quick word count check - retry once if way too long
                word_count = len(trace.split())
                if word_count > TRACE_MAX_WORDS:
                    print(f"  Trace too long ({word_count} words), retrying with stricter prompt...")
                    stricter_prompt = f"""You previously generated a {word_count}-word explanation. That is too long. Rewrite it in EXACTLY {TRACE_TARGET_WORDS} words or fewer. Be ruthlessly concise. No bullet points, no headers, no filler. The correct answer is {correct_answer}. Make sure your explanation concludes that {correct_answer} is correct.

Question: {question}
Context (do not mention): {paths_str}

Previous explanation to condense:
{trace}"""
                    response2 = self._generate_with_retry(
                        model=self.model_trace,
                        contents=stricter_prompt,
                    )
                    if response2 and response2.candidates:
                        trace2 = response2.candidates[0].content.parts[0].text
                        word_count2 = len(trace2.split())
                        # Use the shorter one
                        if word_count2 < word_count:
                            trace = trace2
                            word_count = word_count2
                            print(f"  Retry produced {word_count} words (using this)")
                        else:
                            print(f"  Retry was {word_count2} words (keeping original {word_count})")

                return trace
            return None
        except Exception as e:
            print(f"Error generating COT explanation: {e}")
            return None

    def trace_length_check(self, trace: str) -> bool:
        """
        Check if a thinking trace meets length requirements.
        Returns True if acceptable, False if too long or too short.
        """
        if not trace:
            return False
        word_count = len(trace.split())
        if word_count > TRACE_MAX_WORDS:
            print(f"  Trace REJECTED: {word_count} words exceeds soft max of {TRACE_MAX_WORDS}")
            return False
        if word_count < TRACE_MIN_WORDS:
            print(f"  Trace REJECTED: {word_count} words below minimum of {TRACE_MIN_WORDS}")
            return False
        return True

    def combine_question_and_thinking_trace_with_answer(self, question: str, explanation: str, answer: str) -> str:
        return f"{question}\n<Explanation>\n{explanation}\n</Explanation>\n<Answer>:\n{answer}\n</Answer>"

    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> bool:
        paths_str = ','.join([f"({path['start']} , {path['relation']}, {path['end']})" for path in paths])
        prompt = f"""You are a neuroscience examiner. You are given a neuroscience question and an explanation with an answer. The question and answer are formatted as follows:
        <Question>: [Explanatory Vignette] </Question>
        <Options>
        A. [Option]
        B. [Option]
        C. [Option]
        D. [Option]
        </Options>
        <Explanation>: [Explanation] </Explanation>
        <Answer>: [Correct Option Letter] </Answer>

        1. Judge whether the question and answer are logically correct and scientifically accurate, and follow the source. If there is an explanation, also judge the explanation along with the answer and just evaluate the correctness of the answer otherwise.
        2. Respond with only "Yes" or "No".
        Format your response exactly like this:
        Correct: [Yes/No]

        Question and Answer: {question_answer_explanation}
        Source: {paths_str}
        """

        try:
            response = self._generate_with_retry(
                model=self.model_correctness,
                contents=prompt
            )
            if response and response.candidates:
                content = response.candidates[0].content.parts[0].text
            else:
                return "error"
        except Exception as e:
            print(f"Error correcting: {e}")
            return None

        correct_match = re.search(r'Correct:\s*(Yes|No)', content, re.IGNORECASE)
        if correct_match is None:
            correct_str = "error"
        elif correct_match.group(1).lower() == "yes":
            correct_str = "Yes"
        elif correct_match.group(1).lower() == "no":
            correct_str = "No"
        else:
            correct_str = "error"

        return correct_str


class QAGenerator:
    def __init__(self, api_key: str = None, kg_dir: str = None):
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY is required")
        os.environ['GOOGLE_API_KEY'] = api_key
        # kg_dir is only required for auto-path mode (generate_questions()
        # uses PathGenerator to pick paths from vocab.txt + neuro_graph.pickle
        # + vocab_freq.json). The manifest-driven path mode used by
        # generate_curriculum.py supplies paths directly and never touches
        # PathGenerator — so we lazy-init it to avoid forcing callers that
        # don't need those files to provide them.
        self._kg_dir = kg_dir if kg_dir is not None else os.environ.get("KG_DIR", "")
        self._path_generator = None  # lazy, see self.generator @property
        self.llm = GeminiLLMBackend()

    @property
    def generator(self) -> "PathGenerator":
        # Lazy. Only constructed when auto-path mode (generate_questions)
        # is actually called. If kg_dir isn't set and the upstream csv→KG
        # producer hasn't run, the error surfaces here instead of at
        # construction — so generate_from_path callers proceed unaffected.
        if self._path_generator is None:
            if not self._kg_dir:
                raise ValueError(
                    "kg_dir (path to final_kg directory containing "
                    "vocab.txt + neuro_graph.pickle + vocab_freq.json) "
                    "is required for auto-path mode. Pass --kg_dir or "
                    "set KG_DIR env var, OR call generate_from_path("
                    "path_data) with paths from a hop-manifest CSV."
                )
            self._path_generator = PathGenerator(
                vocab_path=os.path.join(self._kg_dir, 'vocab.txt'),
                graph_path=os.path.join(self._kg_dir, 'neuro_graph.pickle'),
                icd10_categories_path=None,
                vocab_freq_path=os.path.join(self._kg_dir, 'vocab_freq.json')
            )
        return self._path_generator

    def generate_from_path(self, path_data: Dict) -> Optional[Dict]:
        """Generate a single Q&A item from a pre-loaded hop path (e.g. from
        calculate_hops.py's manifest). Bypasses PathGenerator entirely —
        runs the 6-step LLM pipeline on the supplied path.

        path_data shape (from generate_curriculum.load_paths_from_manifest):
            {"hop_count": int, "path": [{"start", "relation", "end"}, ...]}
        Returns the full QA dict (question/answer/explanation/paths/source/
        target/hop_count/question_and_explanation), or None on any pipeline
        step's failure.
        """
        path_steps = path_data.get("path") or []
        if not path_steps:
            return None
        source_concept = str(path_steps[0]["start"])
        target_concept = str(path_steps[-1]["end"])
        paths = path_steps  # already in {start, relation, end} schema

        # Step 1: generate question (LLM)
        question_full = self.llm.generate_question(
            source_concept=source_concept,
            target_concept=target_concept,
            paths=paths,
        )
        if not question_full:
            return None
        question, answer = self.llm.separate_question_and_answer(question_full)
        if not question or not answer:
            return None

        # Step 2: quality filter
        if not self.llm.quality_filtering(question):
            return None

        # Step 3: thinking trace
        explanation = self.llm.generate_thinking_trace(question, paths, answer)
        if not explanation:
            return None

        # Step 4: length check
        if not self.llm.trace_length_check(explanation):
            return None

        # Step 5: combine
        combined = self.llm.combine_question_and_thinking_trace_with_answer(
            question, explanation, answer,
        )

        # Step 6: correctness filter
        if not self.llm.correctness_filtering(combined, paths):
            return None

        return {
            "source_concept": source_concept,
            "target_concept": target_concept,
            "paths": paths,
            "question": question,
            "answer": answer,
            "explanation": explanation,
            "question_and_explanation": combined,
            "hop_count": path_data.get("hop_count"),
        }

    def generate_questions(self, k_hops: int = 1, category: str = None) -> Optional[Dict]:
        max_total_attempts = 10

        for i in range(max_total_attempts):
            try:
                path = self.generator.generate_paths(
                    category=None,
                    k_hops=k_hops
                )

                max_gen_attempts = 3
                for j in range(max_gen_attempts):
                    question_full = self.llm.generate_question(
                        source_concept=path['source_concept'],
                        target_concept=path['target_concept'],
                        paths=path['paths']
                    )

                    if not question_full:
                        continue

                    question_extracted, answer = self.llm.separate_question_and_answer(question_full)

                    if question_extracted and answer:
                        path['question'] = question_extracted
                        path['answer'] = answer
                        return path
                    else:
                        print(f"Attempt {j+1}/{max_gen_attempts}: Generated question format invalid. Retrying LLM gen...")

            except ValueError as ve:
                print(f"Path generation failed on attempt {i}: {ve}")
                continue
            except Exception as e:
                print(f"Unexpected error in generation loop: {e}")
                continue

        print("Failed to generate a valid question after max attempts.")
        return None

    def quality_filtering(self, question: str) -> bool:
        return self.llm.quality_filtering(question)

    def generate_thinking_trace(self, question: str, paths: List[Dict], correct_answer: str) -> Optional[str]:
        return self.llm.generate_thinking_trace(question, paths, correct_answer)

    def trace_length_check(self, trace: str) -> bool:
        """Check if trace meets length requirements."""
        return self.llm.trace_length_check(trace)

    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> bool:
        return self.llm.correctness_filtering(question_answer_explanation, paths)

    def combine_question_and_thinking_trace_with_answer(self, question: str, explanation: str, answer: str) -> str:
        return self.llm.combine_question_and_thinking_trace_with_answer(question, explanation, answer)


def post_process_length_filter(items: List[Dict], field: str = "question_and_explanation") -> Tuple[List[Dict], int]:
    """
    Hard post-processing pass: remove any items where the explanation exceeds TRACE_HARD_MAX_WORDS.
    This is a safety net in case the generation + checker still let something through.
    Returns (filtered_items, num_removed).
    """
    filtered = []
    removed = 0
    for item in items:
        content = item.get(field, "")
        # Extract explanation block
        expl_match = re.search(r"<Explanation>\s*(.*?)\s*</Explanation>", content, re.DOTALL)
        if expl_match:
            explanation = expl_match.group(1)
            word_count = len(explanation.split())
            if word_count > TRACE_HARD_MAX_WORDS:
                print(f"  [POST-FILTER] Removing item {item.get('id', '?')}: explanation is {word_count} words (hard max: {TRACE_HARD_MAX_WORDS})")
                removed += 1
                continue
            if word_count < TRACE_MIN_WORDS:
                print(f"  [POST-FILTER] Removing item {item.get('id', '?')}: explanation is {word_count} words (min: {TRACE_MIN_WORDS})")
                removed += 1
                continue
        filtered.append(item)
    return filtered, removed


def main():
    gemini_gym = QAGenerator()
    print("Generating with Gemini...")
    question = gemini_gym.generate_questions(
        category=None,
        k_hops=2
    )

    if not question:
        print("Failed to generate question object.")
        return

    print(question['question'])
    quality = gemini_gym.quality_filtering(question['question'])
    if not quality:
        print("Quality filtering failed")
        return
    print("Quality filtering passed")

    # Generate concise thinking trace (with internal retry if too long)
    explanation = gemini_gym.generate_thinking_trace(
        question=question['question'],
        paths=question['paths'],
        correct_answer=question['answer']
    )

    if not explanation:
        print("Failed to generate thinking trace")
        return

    # Check trace length
    word_count = len(explanation.split())
    print(f"Thinking trace: {word_count} words")

    if not gemini_gym.trace_length_check(explanation):
        print(f"Trace length check failed ({word_count} words). Skipping this item.")
        return

    combined = gemini_gym.combine_question_and_thinking_trace_with_answer(
        question=question['question'],
        explanation=explanation,
        answer=question['answer']
    )
    correctness = gemini_gym.correctness_filtering(combined, question['paths'])
    if not correctness:
        print("Correctness filtering failed")
        return
    print("Correctness filtering passed")
    print(combined)


if __name__ == "__main__":
    main()