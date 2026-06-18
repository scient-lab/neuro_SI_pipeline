from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Pipeline config loader at the repo root. Sources vocabulary from the
# domain YAML configured via SI_DOMAIN (default: neuroscience).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_config import (  # noqa: E402
    get_entity_categories,
    get_focus_instructions,
    get_relations,
)


# Legacy filename/log label used by 1_seed_kg/graphrag_index.py to namespace
# output dirs (e.g. extracted_graph_responses_set2_0-1000.json). The ACTIVE
# relation list is sourced from the merged pipeline config below, NOT from
# this label. Kept only for filename backwards-compat with existing runs.
RELATION_SET_NAME = os.environ.get("KG_RELATION_SET", "set2").strip().lower()


def get_relation_types() -> List[str]:
    """Return the active relation id list from the merged pipeline config.

    Source of truth: domains/<SI_DOMAIN>.yaml::relations.
    """
    return get_relations()


def get_entity_types() -> List[str]:
    """Return the entity category id list from the merged pipeline config.

    Source of truth: domains/<SI_DOMAIN>.yaml::entity_categories.
    """
    return get_entity_categories()


def get_focus_instructions_text() -> str:
    """Return the free-text extractor focus instructions, or empty string."""
    return get_focus_instructions()


# ---------------------------------------------
# PROMPTS
# ---------------------------------------------
# Prompt template strings used by 1_seed_kg/graphrag_index.py. The
# `{relation_list}` slot is filled at render time with the value of
# get_relation_types(). Kept inline here for now; a future phase may move
# them to prompts/extract.yaml (see ORCHESTRATION_PLAN.md).

PROMPT_TEMPLATE = """-Role-
You are an AI assistant specialized in extracting structured information from neuroscience textbook content to build a knowledge graph about the nervous system, brain function, and neural mechanisms.

-Goal-
Given neuroscience textbook content, a predefined list of entity types, and a predefined list of relations, identify EVERY SINGLE entity of those types and the scientifically meaningful relationships explicitly described among the identified entities within the text.
Extract only information that is directly stated in the text—do not infer, generalize, or use external scientific knowledge.
Try to extract as many entities and relationships as possible while maintaining quality.

-Entity Types-
You should extract entities from the following 7 entity types:
Anatomical Structure, Molecular Entity, Cellular Component, Process, Clinical Entity, Conceptual Entity, Physical Entity.

Use the subcategories listed below SOLELY as guidance to help you determine the correct main entity type. Only use the 7 main entity types in your output.

1. Anatomical Structure: Brain Region; Neural Pathway; Tract; Nucleus; Ganglion; Cortical Area; Subcortical Structure; Spinal Region; Peripheral Nerve; Sensory Organ; Motor Structure
2. Molecular Entity: Neurotransmitter; Neuromodulator; Receptor; Ion Channel; Protein; Enzyme; Gene; Hormone; Signaling Molecule; Pharmaceutical Agent
3. Cellular Component: Neuron; Glial Cell; Astrocyte; Oligodendrocyte; Microglia; Synapse; Axon; Dendrite; Cell Body; Myelin; Membrane; Organelle; Postsynaptic Density; Synaptic Vesicle
4. Process: Synaptic Transmission; Action Potential; Neural Signaling; Neurotransmitter Release; Signal Transduction; Neural Development; Neuroplasticity; Sensory Processing; Motor Control; Depolarization; Repolarization; Exocytosis
5. Clinical Entity: Neurological Disorder; Neurodegenerative Disease; Psychiatric Disorder; Developmental Disorder; Brain Injury; Seizure; Symptom; Syndrome
6. Conceptual Entity: Cognitive Function; Behavior; Learning; Memory; Perception; Motor Planning; Reflex; Neural Circuit; Central Pattern Generator; Topographic Map; Somatotopic Organization; Computational Principle; Gain Modulation
7. Physical Entity: Ion; Electrical Quantity; Current; Voltage; Membrane Potential; Concentration; Gradient; Frequency; Rate

-Relation Types-
Only use ONE relation per relationship tuple, and it MUST be one of these:
{relation_list}

-Steps-
1. Identify all entities corresponding to one of the 7 main entity types and relevant to neuroscience.
For each entity, output:
- entity_name: lowercase, specific and canonical
- entity_type: one of the 7 entity types (exactly)
- entity_description: concise, from the text only
Format:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Identify all explicitly stated relationships between identified entities.
For each relationship, output:
- source_entity: lowercase name from step 1
- target_entity: lowercase name from step 1
- relationship: one relation from the list above (exact string)
- relationship_strength: numeric confidence (7 = central, 5 = supporting, 3 = brief mention)
Format:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship>{tuple_delimiter}<relationship_strength>)

3. Return output as a single flat list, delimited by {record_delimiter}.
Output ONLY tuples. No prose.

4. End with {completion_delimiter}
"""


USER_EXAMPLE = """######################
Entity_types: Anatomical Structure, Molecular Entity, Cellular Component, Process, Clinical Entity, Conceptual Entity
Text:
Voltage-gated sodium channels are responsible for the rising phase of the action potential. The channel protein contains four homologous domains, each with six transmembrane segments. The S4 segment acts as the voltage sensor, moving outward in response to membrane depolarization. This conformational change opens the channel pore, allowing sodium ions to flow into the cell. The resulting inward sodium current causes rapid depolarization of the membrane. Within milliseconds, the channel enters an inactivated state when the inactivation gate occludes the pore. Mutations in the SCN1A gene, which encodes the Nav1.1 sodium channel, cause Dravet syndrome, a severe form of epilepsy characterized by frequent seizures beginning in infancy.
######################
Output:
"""

ASSISTANT_EXAMPLE = """("entity"{tuple_delimiter}voltage-gated sodium channels{tuple_delimiter}Molecular Entity{tuple_delimiter}ion channels responsible for the rising phase of the action potential, contain four homologous domains with six transmembrane segments each.)
{record_delimiter}
("entity"{tuple_delimiter}action potential{tuple_delimiter}Process{tuple_delimiter}electrical signal with a rising phase attributed to voltage-gated sodium channels.)
{record_delimiter}
("entity"{tuple_delimiter}channel protein{tuple_delimiter}Molecular Entity{tuple_delimiter}protein component of the voltage-gated sodium channel described as containing four homologous domains.)
{record_delimiter}
("entity"{tuple_delimiter}four homologous domains{tuple_delimiter}Cellular Component{tuple_delimiter}domains contained within the channel protein, each described as having six transmembrane segments.)
{record_delimiter}
("entity"{tuple_delimiter}six transmembrane segments{tuple_delimiter}Cellular Component{tuple_delimiter}transmembrane segments described as present in each homologous domain of the channel protein.)
{record_delimiter}
("entity"{tuple_delimiter}s4 segment{tuple_delimiter}Cellular Component{tuple_delimiter}segment that acts as the voltage sensor and moves outward in response to membrane depolarization.)
{record_delimiter}
("entity"{tuple_delimiter}membrane depolarization{tuple_delimiter}Process{tuple_delimiter}process that the s4 segment responds to by moving outward.)
{record_delimiter}
("entity"{tuple_delimiter}conformational change{tuple_delimiter}Process{tuple_delimiter}change described as opening the channel pore.)
{record_delimiter}
("entity"{tuple_delimiter}channel pore{tuple_delimiter}Cellular Component{tuple_delimiter}pore that opens due to the described conformational change, allowing sodium ions to flow into the cell.)
{record_delimiter}
("entity"{tuple_delimiter}sodium ions{tuple_delimiter}Molecular Entity{tuple_delimiter}ions that flow into the cell through the open channel pore.)
{record_delimiter}
("entity"{tuple_delimiter}cell{tuple_delimiter}Cellular Component{tuple_delimiter}cell into which sodium ions are described as flowing.)
{record_delimiter}
("entity"{tuple_delimiter}inward sodium current{tuple_delimiter}Process{tuple_delimiter}current resulting from sodium ions flowing into the cell, described as causing rapid depolarization of the membrane.)
{record_delimiter}
("entity"{tuple_delimiter}membrane{tuple_delimiter}Cellular Component{tuple_delimiter}structure described as undergoing rapid depolarization due to inward sodium current.)
{record_delimiter}
("entity"{tuple_delimiter}inactivated state{tuple_delimiter}Process{tuple_delimiter}state the channel enters within milliseconds when the inactivation gate occludes the pore.)
{record_delimiter}
("entity"{tuple_delimiter}inactivation gate{tuple_delimiter}Cellular Component{tuple_delimiter}gate that occludes the channel pore, associated with the channel entering an inactivated state.)
{record_delimiter}
("entity"{tuple_delimiter}scn1a gene{tuple_delimiter}Molecular Entity{tuple_delimiter}gene described as encoding the nav1.1 sodium channel.)
{record_delimiter}
("entity"{tuple_delimiter}nav1.1 sodium channel{tuple_delimiter}Molecular Entity{tuple_delimiter}sodium channel encoded by the scn1a gene; mutations in scn1a are described as causing dravet syndrome.)
{record_delimiter}
("entity"{tuple_delimiter}dravet syndrome{tuple_delimiter}Clinical Entity{tuple_delimiter}severe form of epilepsy described as caused by mutations in scn1a and characterized by frequent seizures beginning in infancy.)
{record_delimiter}
("entity"{tuple_delimiter}epilepsy{tuple_delimiter}Clinical Entity{tuple_delimiter}clinical condition described as the category for dravet syndrome.)
{record_delimiter}
("entity"{tuple_delimiter}seizures{tuple_delimiter}Clinical Entity{tuple_delimiter}frequent events described as characterizing dravet syndrome and beginning in infancy.)
{record_delimiter}
("entity"{tuple_delimiter}infancy{tuple_delimiter}Conceptual Entity{tuple_delimiter}life stage during which seizures are described as beginning.)
{record_delimiter}
("relationship"{tuple_delimiter}voltage-gated sodium channels{tuple_delimiter}action potential{tuple_delimiter}mediates_signal_for{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}channel protein{tuple_delimiter}voltage-gated sodium channels{tuple_delimiter}part_of{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}channel protein{tuple_delimiter}four homologous domains{tuple_delimiter}contains{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}four homologous domains{tuple_delimiter}six transmembrane segments{tuple_delimiter}contains{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}s4 segment{tuple_delimiter}four homologous domains{tuple_delimiter}part_of{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}s4 segment{tuple_delimiter}membrane depolarization{tuple_delimiter}responds_to{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}conformational change{tuple_delimiter}channel pore{tuple_delimiter}results_in{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}channel pore{tuple_delimiter}cell{tuple_delimiter}located_in{tuple_delimiter}3)
{record_delimiter}
("relationship"{tuple_delimiter}voltage-gated sodium channels{tuple_delimiter}sodium ions{tuple_delimiter}transports{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}inward sodium current{tuple_delimiter}membrane depolarization{tuple_delimiter}causes{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}inactivation gate{tuple_delimiter}channel pore{tuple_delimiter}modulates{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}inactivation gate{tuple_delimiter}inactivated state{tuple_delimiter}required_for{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}scn1a gene{tuple_delimiter}nav1.1 sodium channel{tuple_delimiter}encodes_representation_of{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}scn1a gene{tuple_delimiter}dravet syndrome{tuple_delimiter}causes{tuple_delimiter}7)
{record_delimiter}
("relationship"{tuple_delimiter}dravet syndrome{tuple_delimiter}epilepsy{tuple_delimiter}part_of{tuple_delimiter}5)
{record_delimiter}
("relationship"{tuple_delimiter}dravet syndrome{tuple_delimiter}seizures{tuple_delimiter}symptom_of{tuple_delimiter}7)
{completion_delimiter}"""

USER_PROMPT = """######################
Entity_types: Anatomical Structure, Molecular Entity, Cellular Component, Process, Clinical Entity, Conceptual Entity
Text:
{input_text}
######################
Output:{think_directive}
"""
# {think_directive} is filled at format time with either " /no_think" or "".
# Qwen3 honors the bare "/no_think" control token in the prompt to skip its
# <think>...</think> block. graphrag_index.py reads extract.no_think from
# configs/default.yaml and decides which to substitute.
