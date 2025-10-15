#!/usr/bin/env python3

import json
import re
import os
from collections import defaultdict

TAXONOMY = "docs/reduced_mission_planning_taxonomy.json"
MAPPING = "docs/mapping_requirements_taxonomy.json"
OUTPUT = "docs/heat_tree_data.json"
ID_MAP = "docs/id_mapping.json"

# ---------- Configurable Weights ----------
PRIORITY_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}
TRACEABILITY_WEIGHTS = {"Direct": 1.0, "Indirect": 0.5}


def _reuse_existing_id(name: str, existing_mapping: dict, used_ids: set) -> str | None:
    """
    Retrieve and register a pre-existing ID for the given name.

    :param name: The name to look up
    :type name: str
    :param existing_mapping: Dictionary-mapping names to existing IDs
    :type existing_mapping: dict
    :param used_ids: Set of IDs currently in use
    :type used_ids: set
    :return: The existing ID if found, None otherwise
    :rtype: str or None
    """
    if existing_mapping and name in existing_mapping:
        existing_id = existing_mapping[name]
        used_ids.add(existing_id)
        return existing_id
    return None


def _generate_base_id(name: str) -> str:
    """
    Generate a base ID from the name using uppercase letters.

    Extracts uppercase letters from CamelCase names, or falls back to
    the first 3 characters if no uppercase pattern is found.

    :param name: The input name
    :type name: str
    :return: Base identifier string
    :rtype: str
    """
    parts = re.findall('[A-Z][a-z]*', name)
    if not parts:
        # Fallback: use the first 3 characters
        return name[:3].upper()

    # Extract the first letter of each part
    return ''.join([part[0].upper() for part in parts])


def _ensure_unique_id(base_id: str, used_ids: set) -> str:
    """
    Ensure the ID is unique by appending a numeric suffix if needed.

    :param base_id: The base identifier to make unique
    :type base_id: str
    :param used_ids: Set of IDs already in use
    :type used_ids: set
    :return: A unique identifier
    :rtype: str
    """
    if base_id not in used_ids:
        return base_id

    counter = 2
    unique_id = f"{base_id}{counter}"
    while unique_id in used_ids:
        counter += 1
        unique_id = f"{base_id}{counter}"

    return unique_id


def make_id(name: str, used_ids: set, existing_mapping=None) -> str:
    """
    Generate a unique identifier for a given input name. It tries to create a unique
    ID based on uppercase letters from the name. If there are conflicts with existing IDs,
    it appends a numeric suffix to ensure uniqueness. If an existing mapping is provided
    and the name already has a mapped ID, it reuses that ID.

    :param name: The input name from which to generate the ID.
    :type name: str
    :param used_ids: A set of IDs that have already been used to prevent conflicts.
    :type used_ids: set
    :param existing_mapping: Optional dictionary mapping names to pre-existing IDs.
        If provided and the name exists in this mapping, its associated ID will be
        reused.
    :type existing_mapping: dict, optional
    :return: A unique identifier for the name.
    :rtype: str
    """
    # Step 1: Check for existing mapping
    existing_id = _reuse_existing_id(name, existing_mapping, used_ids)
    if existing_id:
        return existing_id

    # Step 2: Generate base ID from name
    base_id = _generate_base_id(name)

    # Step 3: Ensure uniqueness
    final_id = _ensure_unique_id(base_id, used_ids)

    # Step 4: Register the new ID
    used_ids.add(final_id)

    return final_id


def _compute_weighted_scores(req_mapping: dict[str, list[dict[str, str]]]) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """
    Computes weighted scores for domains and concepts based on priority and traceability.

    :param req_mapping: Mapping of requirements with priorities and traceability levels
    :return: Tuple of (domain_scores, concept_scores, domain_co_refs)
    """
    domain_scores = defaultdict(float)
    concept_scores = defaultdict(float)
    domain_co_refs = defaultdict(float)

    for req in req_mapping["RequirementMapping"]:
        priority_weight = PRIORITY_WEIGHTS.get(req.get("Priority", "Medium"), 1)
        traceability_weight = TRACEABILITY_WEIGHTS.get(req.get("TraceabilityLevel", "Direct"), 1.0)
        combined_weight = priority_weight * traceability_weight

        seen_domains = set()
        for ref in req.get("RelatedTaxonomy", []):
            domain = ref["Domain"]
            domain_scores[domain] += combined_weight

            for concept in ref.get("Concepts", []):
                concept_scores[(domain, concept)] += combined_weight

            seen_domains.add(domain)

        # Track co-occurrence for domain pairs
        for domain1 in seen_domains:
            for domain2 in seen_domains:
                if domain1 < domain2:
                    domain_pair = (domain1, domain2)
                    domain_co_refs[domain_pair] += combined_weight

    return domain_scores, concept_scores, domain_co_refs


def _compute_normalization_factors(domain_scores: dict[str, float],
                                   concept_scores: dict[tuple[str, str], float]) -> tuple[float, float]:
    """
    Computes maximum values for normalising domain and concept scores.

    :param domain_scores: Dictionary of domain scores
    :param concept_scores: Dictionary of concept scores
    :return: Tuple of (max_domain_score, max_concept_score)
    """
    max_domain_score = max(domain_scores.values()) if domain_scores else 1
    max_concept_score = max(concept_scores.values()) if concept_scores else 1
    return max_domain_score, max_concept_score


def _build_concept_node(concept_name: str,
                        domain_name: str,
                        concept_scores: dict[tuple[str, str], float],
                        max_concept_score: float,
                        used_ids: set,
                        existing_mapping: dict[str, str],
                        id_mapping: dict[str, str]) -> dict:
    """
    Builds a single concept node with normalised heat value.

    :param concept_name: Name of the concept
    :param domain_name: Parent domain name
    :param concept_scores: Dictionary of concept scores
    :param max_concept_score: Maximum concept score for normalisation
    :param used_ids: Set of already used IDs
    :param existing_mapping: Optional mapping of names to stable IDs
    :param id_mapping: Dictionary to track name-to-ID mappings
    :return: Dictionary representing concept node
    """
    concept_id = make_id(concept_name, used_ids, existing_mapping)
    id_mapping[concept_name] = concept_id

    concept_key = (domain_name, concept_name)
    raw_score = concept_scores.get(concept_key, 0)
    normalized_heat = raw_score / max_concept_score if max_concept_score else 0

    return {
        "id": concept_id,
        "name": concept_name,
        "score": raw_score,
        "heat": normalized_heat,
        "children": []
    }


def _build_domain_node(domain_name: str,
                       domain_data: dict,
                       domain_scores: dict,
                       concept_scores: dict[tuple[str, str], float],
                       max_domain_score: float,
                        max_concept_score: float,
                       used_ids: set,
                       existing_mapping: dict[str, str],
                       id_mapping: dict[str, str],
                       domain_id_map: dict[str, str]) -> dict:
    """
    Builds a single domain node with its concept children.

    :param domain_name: Name of the domain
    :param domain_data: Domain data from taxonomy
    :param domain_scores: Dictionary of domain scores
    :param concept_scores: Dictionary of concept scores
    :param max_domain_score: Maximum domain score for normalisation
    :param max_concept_score: Maximum concept score for normalisation
    :param used_ids: Set of already used IDs
    :param existing_mapping: Optional mapping of names to stable IDs
    :param id_mapping: Dictionary to track name-to-ID mappings
    :param domain_id_map: Dictionary to track domain name-to-ID mappings
    :return: Dictionary representing domain node
    """
    domain_id = make_id(domain_name, used_ids, existing_mapping)
    id_mapping[domain_name] = domain_id
    domain_id_map[domain_name] = domain_id

    raw_domain_score = domain_scores.get(domain_name, 0)
    normalized_domain_heat = raw_domain_score / max_domain_score if max_domain_score else 0

    concept_children = [
        _build_concept_node(concept_name, domain_name, concept_scores, max_concept_score,
                            used_ids, existing_mapping, id_mapping)
        for concept_name in domain_data.get("Concepts", {}).keys()
    ]

    return {
        "id": domain_id,
        "name": domain_name,
        "score": raw_domain_score,
        "heat": normalized_domain_heat,
        "children": concept_children
    }


def _build_heat_tree_structure(taxonomy: dict,
                               domain_scores: dict,
                               concept_scores: dict[tuple[str, str], float],
                               max_domain_score: float,
                               max_concept_score: float,
                               used_ids: set,
                               existing_mapping: dict[str, str],
                               id_mapping: dict[str, str]) -> tuple[dict, dict]:
    """
    Builds the complete hierarchical heat tree structure.

    :param taxonomy: Taxonomy structure with domains and concepts
    :param domain_scores: Dictionary of domain scores
    :param concept_scores: Dictionary of concept scores
    :param max_domain_score: Maximum domain score for normalisation
    :param max_concept_score: Maximum concept score for normalisation
    :param used_ids: Set of already used IDs
    :param existing_mapping: Optional mapping of names to stable IDs
    :param id_mapping: Dictionary to track name-to-ID mappings
    :return: Tuple of (heat_tree, domain_id_map)
    """
    heat_tree = []
    domain_id_map = {}

    for domain_name, domain_data in taxonomy["MissionPlanningTaxonomy"].items():
        domain_node = _build_domain_node(
            domain_name, domain_data, domain_scores, concept_scores,
            max_domain_score, max_concept_score, used_ids, existing_mapping,
            id_mapping, domain_id_map
        )
        heat_tree.append(domain_node)

    return heat_tree, domain_id_map


def _build_domain_links(taxonomy: dict,
                        domain_id_map: dict,
                        domain_co_refs: dict) -> list[dict[str, int | float]]:
    """
    Builds links between related domains based on co-occurrence weights.

    :param taxonomy: Taxonomy structure with domain relationships
    :param domain_id_map: Mapping of domain names to IDs
    :param domain_co_refs: Dictionary of domain pair co-occurrence weights
    :return: List of link dictionaries
    """
    links = []

    for domain_name, domain_data in taxonomy["MissionPlanningTaxonomy"].items():
        source_id = domain_id_map[domain_name]

        for related_domain in domain_data.get("RelatedTo", []):
            target_id = domain_id_map.get(related_domain)
            if target_id:
                domain_pair = tuple(sorted([domain_name, related_domain]))
                co_occurrence_weight = domain_co_refs.get(domain_pair, 0)

                links.append({
                    "source": source_id,
                    "target": target_id,
                    "weight": co_occurrence_weight
                })

    return links


def build_heat_tree(taxonomy: dict,
                    req_mapping: dict,
                    existing_mapping=None) -> tuple[dict, dict]:
    """
    Builds a hierarchical "heat tree" structure based on taxonomy and mapping data,
    including weighted scores for domains and concepts, normalised over their maximum
    values. Additionally, establishes links between related domains based on co-occurrence
    weights derived from mappings.

    The heat tree represents hierarchical relationships between domains and concepts,
    with scores and heat values reflecting their respective importance or relevance.
    Links establish co-occurrence relationships between domains.

    :param taxonomy: A dictionary representing the taxonomy structure, including domains
                     and their associated concepts and relationships.
    :type taxonomy: dict
    :param req_mapping: A mapping of requirements including priorities, traceability
                        levels, and references to related taxonomy entities.
    :type req_mapping: dict
    :param existing_mapping: An optional parameter for an externally provided mapping of
                             names to stable IDs. Defaults to None.
    :type existing_mapping: dict, optional
    :return: A tuple containing the generated heat tree structure and an updated mapping
             between names and IDs.
    :rtype: tuple
    """
    used_ids = set()
    id_mapping = {}

    # Compute weighted scores
    domain_scores, concept_scores, domain_co_refs = _compute_weighted_scores(req_mapping)

    # Compute normalisation factors
    max_domain_score, max_concept_score = _compute_normalization_factors(domain_scores, concept_scores)

    # Build the tree structure
    heat_tree, domain_id_map = _build_heat_tree_structure(
        taxonomy, domain_scores, concept_scores, max_domain_score, max_concept_score,
        used_ids, existing_mapping, id_mapping
    )

    # Build links
    links = _build_domain_links(taxonomy, domain_id_map, domain_co_refs)

    return {"heat_tree": heat_tree, "links": links}, id_mapping


def _load_json_file(filepath: str, description=None) -> None:
    """
    Loads JSON data from a file.

    :param filepath: Path to the JSON file
    :type filepath: str
    :param description: Optional description for user feedback
    :type description: str, optional
    :return: Parsed JSON data, or None if file doesn't exist
    :rtype: dict or None
    """
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    if description:
        print(f"Loaded {description} from {filepath}...")

    return data


def _save_json_file(filepath: str, data: dict) -> None:
    """
    Saves the given data to a JSON file at the specified file path. The function writes
    the data in a human-readable format with an indentation level of 2.

    :param filepath: The file path where the JSON file will be saved.
    :type filepath: str
    :param data: The dictionary data to be saved into the JSON file.
    :type data: dict
    :return: This function does not return a value.
    :rtype: None
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Load input files
    taxonomy = _load_json_file(TAXONOMY)
    req_mapping = _load_json_file(MAPPING)

    # Load existing id_mapping.json if it exists
    existing_mapping = _load_json_file(ID_MAP, "existing ID mapping")

    # Build heat tree structure
    heat_tree_data, id_mapping = build_heat_tree(taxonomy, req_mapping, existing_mapping)

    # Save output files
    _save_json_file(OUTPUT, heat_tree_data)
    _save_json_file(ID_MAP, id_mapping)

    print(f"{OUTPUT} and {ID_MAP} generated successfully.")
