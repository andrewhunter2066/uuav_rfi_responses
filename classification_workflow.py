#!/usr/bin/env python3

import pandas as pd
import random
import re
import nltk
import os
import json
from nltk.corpus import stopwords, wordnet
from datetime import datetime
from typing import List
from collections import defaultdict

# Download required corpora
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Domain-specific acronym expansions for text preprocessing
ACRONYM_EXPANSIONS = {
    r'\bao\b': 'area of operations',
    r'\buuv\b': 'unmanned underwater vehicle',
    r'\bgps\b': 'global positioning system',
    r'\bapi\b': 'application programming interface',
    r'\brfi\b': 'request for information',
    r'\bmbes\b': 'multibeam echo sounder',
    r'\bvs\b': 'versus',
    r'\bephemerals\b': 'ephemeral',
    r'\bmin\b': 'minimum',
    r'\bmax\b': 'maximum',
    r'\bcomms\b': 'communications',
    r'\bhipap\b': 'high precision acoustic positioning',
    r'\be.g.\b': '',
    r'\b%\b': 'percent',
}

# Regex pattern to keep only word characters, spaces, and hyphens
PUNCTUATION_PATTERN = r'[^\w\s\-]'

# Which method to use, domain-specific (True) or general NLP (False)
DOMAIN_SPECIFIC = True

# Suppression Log
SUPPRESSION_LOG = []


def build_category_keywords(taxonomy: dict, lowercase: bool = True, deduplicate: bool = True) -> dict[str, list[str]]:
    """
    Build flat keyword lists per taxonomy category for classification.
    Traverses the taxonomy hierarchy and collects all terms under 'Concepts'.

    Args:
        taxonomy (dict): The full taxonomy dict.
        lowercase (bool): Whether to normalise keywords to lowercase.
        deduplicate (bool): Whether to deduplicate keyword lists.

    Returns:
        dict[str, list[str]]: Mapping of category → keyword list.
    """
    category_keywords = {}

    # Expect the taxonomy to be unwrapped at this point
    # Each top-level key should be a category with a "Concepts" sub-dictionary
    for category, content in taxonomy.items():
        keywords = []
        if isinstance(content, dict) and "Concepts" in content:
            for group_terms in content["Concepts"].values():
                if isinstance(group_terms, list):
                    keywords.extend(group_terms)

        # Optional normalization
        if lowercase:
            keywords = [kw.lower().strip() for kw in keywords]

        # Optional deduplication while preserving order
        if deduplicate:
            seen = set()
            keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]

        category_keywords[category] = keywords

    return category_keywords


def tokenize_text(text: str) -> List[str]:
    """
    Split text into candidate tokens/phrases for validation.
    Keeps multi-word domain terms intact if possible.
    """
    # Normalise and keep multi-word phrases (basic approach)
    tokens = re.findall(r"[A-Za-z0-9\-\s]+", text)
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    return tokens


def document_validate(text: str,
                      context: str,
                      CONFLICT_RULES: dict,
                      PROTECTED_TERMS: set,
                      DOMAIN_SYNONYMS: dict,
                      strict_mode: bool = False,
                      export_csv: str = None,
                      export_json: str = None) -> dict:
    """
    Validate a whole text document against conflict rules, with composite context support.

    Args:
        text (str): Raw document text.
        context (str): Context (single or composite like 'mission_ops+vehicle').
        CONFLICT_RULES (dict): Context conflict rules.
        PROTECTED_TERMS (set): Global protected terms.
        DOMAIN_SYNONYMS (dict): Domain synonym mappings.
        strict_mode (bool): If True, cross-context validation is enabled.
        export_csv (str): Path to export CSV results.
        export_json (str): Path to export JSON results.

    Returns:
        dict: Structured validation report.
    """

    tokens = tokenize_text(text)
    report = batch_validate(tokens, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode)

    # Export if requested
    df = pd.DataFrame(report["results"])
    if export_csv:
        df.to_csv(export_csv, index=False)
    if export_json:
        df.to_json(export_json, orient="records", indent=2)

    return report


def validate_responses_csv(input_csv,
                           output_dir,
                           CONFLICT_RULES,
                           PROTECTED_TERMS,
                           DOMAIN_SYNONYMS,
                           strict_mode=False):
    """
    Validate all responses in a normalised SME CSV (e.g. normalised_all_responses.csv).
    Generates per-scenario, per-question conflict reports.

    Args:
        input_csv (str): Path to the normalised responses CSV.
        output_dir (str): Directory to store validation reports.
        CONFLICT_RULES (dict): Context-specific conflict rules.
        PROTECTED_TERMS (set): Global protected terms.
        DOMAIN_SYNONYMS (dict): Domain-level synonym mappings.
        strict_mode (bool): If True, check for cross-context conflicts.

    Returns:
        dict: Summary statistics across all scenarios/questions.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    all_summary = {}

    for (scenario, question), group in df.groupby(["ScenarioNumber", "Question"]):
        context = map_question_to_context(question)  # You need a mapping fn
        text_block = " ".join(group["ResponseText"].dropna().astype(str))

        # Validate document-level
        report = document_validate(
            text=text_block,
            context=context,
            CONFLICT_RULES=CONFLICT_RULES,
            PROTECTED_TERMS=PROTECTED_TERMS,
            DOMAIN_SYNONYMS=DOMAIN_SYNONYMS,
            strict_mode=strict_mode,
            export_csv=os.path.join(output_dir, f"S{scenario}_{question}_conflict_report.csv"),
            export_json=os.path.join(output_dir, f"S{scenario}_{question}_conflict_report.json")
        )

        # Store summary
        all_summary[(scenario, question)] = report["summary"]

    return all_summary


def expand_contexts(taxonomy: dict, seeds: list[str], depth: int = 1, normalise: bool = True) -> list[str]:
    """
    Expand seed taxonomy domains via 'RelatedTo' links, with optional normalisation.
    """
    expanded = set(seeds)
    frontier = set(seeds)

    for _ in range(depth):
        new_frontier = set()
        for domain in frontier:
            related = taxonomy.get(domain, {}).get("RelatedTo", [])
            for rel in related:
                if rel not in expanded:
                    expanded.add(rel)
                    new_frontier.add(rel)
        frontier = new_frontier

    if normalise:
        return sorted([camel_to_title(x) for x in expanded])
    else:
        return sorted(expanded)


def map_question_to_context(question: str, taxonomy: dict) -> str:
    """
    Map RFI question to taxonomy-aligned context by auto-expanding seeds.
    Returns normalised (human-readable) categories.
    """
    q = question.lower().strip()

    seed_map = {
        "q1": ["TerrainAndBathymetry", "EnvironmentalAndOceanographicConditions"],
        "q2": ["VehicleCapabilitiesAndConstraints", "MissionParametersAndObjectives"],
        "q3": ["EstimationAndUncertaintyModeling", "DataProductsAndRequirements"],
        "q4": ["ThreatsAndRiskManagement"],
        "q5": ["ThreatsAndRiskManagement", "MissionParametersAndObjectives"],
        "q6": ["HistoricalAndContextualData", "DataProductsAndRequirements"],
    }

    for prefix, seeds in seed_map.items():
        if q.startswith(prefix):
            expanded = expand_contexts(taxonomy["MissionPlanningTaxonomy"], seeds, depth=1, normalise=True)
            return "+".join(expanded)

    return "General"


def camel_to_title(name: str) -> str:
    """
    Convert CamelCase (e.g. 'MissionParametersAndObjectives')
    into a human-readable title (e.g. 'Mission Parameters And Objectives').
    """
    # Split on capital letters
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return s.strip()



def log_suppression(word, protected, context_snippet):
    SUPPRESSION_LOG.append({
        "word": word,
        "protected_trigger": protected,
        "context_snippet": context_snippet[:120]
    })


def should_skip_synonym(text, synonym_group, conflict_rules):
    """Check if a synonym group should be skipped based on protected term co-occurrence."""
    text_lower = text.lower()
    for rule in conflict_rules.values():
        if synonym_group in rule["synonyms"]:
            for protected in rule["protected"]:
                if protected in text_lower:
                    return True
    return False


def augment_text_minimal(text: str, aug_prob=0.1) -> str:
    """
    Augments text with minimal character-level variations that preserve meaning.
    Suitable for technical/jargon-dense text where synonym replacement is risky.

    Techniques:
    - Random word order swap (adjacent words only, rarely)
    - Punctuation variation
    - Article insertion/removal (a/an/the)

    :param text: The input text to augment.
    :param aug_prob: Probability of applying augmentation.
    :return: Lightly augmented text.
    """
    words = text.split()
    word_count = len(words)

    if word_count < 3 or random.random() > aug_prob:
        return text

    augmented_words = words.copy()

    # Occasionally swap adjacent non-technical words
    if word_count > 3:
        idx = random.randint(0, word_count - 2)
        # Only swap if neither word is in protected terms
        if (augmented_words[idx].lower() not in PROTECTED_TERMS and
                augmented_words[idx + 1].lower() not in PROTECTED_TERMS):
            augmented_words[idx], augmented_words[idx + 1] = \
                augmented_words[idx + 1], augmented_words[idx]

    return ' '.join(augmented_words)


def augment_text_structural(text: str) -> str:
    """
    Creates structural variations while preserving technical content.

    Examples:
    - "X is needed for Y" → "Y requires X"
    - "System must support X" → "X support is required"

    :param text: The input text to augment.
    :return: Structurally varied text.
    """
    # Simple pattern-based transformations
    transformations = [
        (r'\b(\w+) is needed for (\w+)', r'\2 requires \1'),
        (r'\bsystem must support (\w+)', r'\1 support is required'),
        (r'\b(\w+) capability', r'capability to \1'),
        (r'\brequirement for (\w+)', r'\1 requirement'),
    ]

    augmented = text
    # Apply one random transformation if a pattern matches
    random.shuffle(transformations)
    for pattern, replacement in transformations:
        if re.search(pattern, augmented, re.IGNORECASE):
            augmented = re.sub(pattern, replacement, augmented, count=1, flags=re.IGNORECASE)
            break

    return augmented


def check_conflict(word, context, conflict_rules):
    """
    Validate whether a given word conflicts with the protected or synonym sets
    of a specific semantic context.

    Parameters
    ----------
    word : str
        The candidate word or phrase to test.
    context : str
        The domain context (e.g. 'temporal', 'survey', 'mission_ops').
    conflict_rules : dict
        The CONFLICT_RULES structure defining protected/synonym sets.

    Returns
    -------
    dict
        A structured validation result:
        {
            "status": "protected" | "synonym_conflict" | "allowed" | "unknown_context",
            "context": context,
            "reason": str
        }

    Examples
    --------
    >>> check_conflict("bathymetry", "terrain_bathymetry", CONFLICT_RULES)
    {'status': 'protected', 'context': 'terrain_bathymetry', 'reason': 'Exact match in protected terms.'}

    >>> check_conflict("time", "temporal", CONFLICT_RULES)
    {'status': 'synonym_conflict', 'context': 'temporal', 'reason': 'Matches synonym list; may conflict semantically.'}

    >>> check_conflict("fish", "environmental", CONFLICT_RULES)
    {'status': 'allowed', 'context': 'environmental', 'reason': 'No match found in protected or synonym lists.'}
    """
    # Normalise inputs
    word = word.strip().lower()
    context = context.strip().lower()

    # Validate context
    if context not in conflict_rules:
        return {
            "status": "unknown_context",
            "context": context,
            "reason": f"Context '{context}' not defined in conflict rules."
        }

    rules = conflict_rules[context]
    protected = set(w.lower() for w in rules.get("protected", []))
    synonyms = set(w.lower() for w in rules.get("synonyms", []))

    # Exact match checks
    if word in protected:
        return {
            "status": "protected",
            "context": context,
            "reason": "Exact match in protected terms."
        }

    if word in synonyms:
        return {
            "status": "synonym_conflict",
            "context": context,
            "reason": "Matches synonym list; may conflict semantically."
        }

    # Partial match heuristic (e.g., "mission phase" contains "mission")
    for p in protected:
        if re.search(rf"\b{re.escape(p)}\b", word):
            return {
                "status": "protected",
                "context": context,
                "reason": f"Contains protected token '{p}'."
            }

    for s in synonyms:
        if re.search(rf"\b{re.escape(s)}\b", word):
            return {
                "status": "synonym_conflict",
                "context": context,
                "reason": f"Contains synonym token '{s}'."
            }

    # Otherwise safe
    return {
        "status": "allowed",
        "context": context,
        "reason": "No match found in protected or synonym lists."
    }


def check_conflict_all(word, conflict_rules):
    """
    Run check_conflict across all defined contexts and return all conflicts found.
    """
    results = []
    for ctx in conflict_rules.keys():
        result = check_conflict(word, ctx, conflict_rules)
        if result["status"] in ("protected", "synonym_conflict"):
            results.append(result)
    return results or [{
        "status": "allowed",
        "context": None,
        "reason": "No conflicts found in any context."
    }]


def validate_term(term, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode=False):
    """
    Validate a term against protected terms and conflict rules.
    Supports composite contexts separated by '+' (e.g. "mission_ops+vehicle").
    """

    term_lower = term.lower().strip()
    contexts = [c.strip().lower() for c in context.split("+")]

    all_results = []
    final_status = "valid"
    final_details = []

    # --- Global protection check
    if term_lower in PROTECTED_TERMS:
        return {
            "term": term,
            "context": context,
            "status": "protected",
            "details": f"Globally protected technical term – must not be replaced."
        }

    # --- Check each context individually
    for ctx in contexts:
        if ctx not in CONFLICT_RULES:
            continue

        rules = CONFLICT_RULES[ctx]
        protected = [p.lower() for p in rules.get("protected", [])]
        synonyms = [s.lower() for s in rules.get("synonyms", [])]

        if term_lower in protected:
            all_results.append(("protected", f"Protected within context '{ctx}'."))
        elif term_lower in synonyms:
            all_results.append(("conflict", f"Potential synonym conflict in context '{ctx}'."))

        # Strict mode: check cross-context conflicts
        if strict_mode:
            cross_conflicts = []
            for other_ctx, other_rules in CONFLICT_RULES.items():
                if other_ctx == ctx:
                    continue
                if term_lower in [p.lower() for p in other_rules.get("protected", [])]:
                    cross_conflicts.append(other_ctx)
            if cross_conflicts:
                all_results.append(("cross_conflict", f"Protected in other context(s): {', '.join(cross_conflicts)}"))

    # --- Domain-level synonym conflict
    for key, syns in DOMAIN_SYNONYMS.items():
        all_syns = [s.lower() for s in syns] + [key.lower()]
        if term_lower in all_syns and key.lower() in PROTECTED_TERMS:
            all_results.append(("conflict", f"'{term}' overlaps with protected domain term '{key}'."))

    # --- Merge results
    if all_results:
        # Priority: protected > cross_conflict > conflict > valid
        priorities = {"protected": 3, "cross_conflict": 2, "conflict": 1, "valid": 0}
        final_status = max(all_results, key=lambda x: priorities[x[0]])[0]
        final_details = [msg for _, msg in all_results]
    else:
        final_status = "valid"
        final_details = ["No conflict detected."]

    return {
        "term": term,
        "context": context,
        "status": final_status,
        "details": "; ".join(final_details)
    }


def batch_validate(terms, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode=False):
    """
    Batch-validate a list of terms against protected and conflict rules.
    Handles composite contexts (split by '+').

    Args:
        terms (list[str]): Words or phrases to validate.
        context (str): Active domain context (or composite, e.g. 'mission_ops+vehicle').
        CONFLICT_RULES (dict): Context rules containing 'protected' and 'synonyms' lists.
        PROTECTED_TERMS (set): Globally protected terms.
        DOMAIN_SYNONYMS (dict): Domain-level synonym mappings.
        strict_mode (bool): If True, check for cross-context conflicts.

    Returns:
        dict: {
            "context": str,
            "results": [ {"term": str, "status": str, "details": str} ],
            "summary": {"valid": int, "protected": int, "conflict": int, "cross_conflict": int}
        }
    """

    results = []
    summary = {"valid": 0, "protected": 0, "conflict": 0, "cross_conflict": 0}

    for term in terms:
        result = validate_term(term, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode)
        results.append({
            "term": result["term"],
            "status": result["status"],
            "details": result["details"]
        })
        summary[result["status"]] = summary.get(result["status"], 0) + 1

    return {
        "context": context,
        "results": results,
        "summary": summary
    }


def get_domain_synonyms(word: str, domain_synonyms: dict[str, list[str]], protected_terms: set[str]) -> list[str]:
    """
    Retrieve domain-appropriate synonyms from a controlled dictionary with protection logic.

    This function:
      - Checks for protected terms (and submatches of multi-word protected terms)
      - Normalizes input for consistent matching
      - Handles plural/singular equivalence
      - Ensures synonyms are safe and not recursive
      - Optionally supports fuzzy matching for near-misses (e.g. 'depths' → 'depth')

    :param word: The word or phrase to find synonyms for.
    :param domain_synonyms: Controlled mapping of domain terms → synonym lists.
    :param protected_terms: Set of protected technical or domain-specific terms.
    :return: List of domain-safe synonyms, or [] if none is found, or the term is protected.
    """

    # Normalize input
    word_norm = word.strip().lower()

    # --- Protected term check (exact and substring) ---
    if any(re.fullmatch(rf"\b{re.escape(term)}\b", word_norm) or term in word_norm for term in protected_terms):
        return []

    # --- Direct lookup ---
    if word_norm in domain_synonyms:
        return domain_synonyms[word_norm]

    # --- Reverse lookup (catch when a synonym is used instead of its key) ---
    for key, syns in domain_synonyms.items():
        if word_norm in [s.lower() for s in syns]:
            return [key] + [s for s in syns if s.lower() != word_norm]

    # --- Singular/plural normalization ---
    # Handle simple plural/singular equivalence (e.g., "routes" → "route")
    if word_norm.endswith('s') and word_norm[:-1] in domain_synonyms:
        return domain_synonyms[word_norm[:-1]]
    elif f"{word_norm}s" in domain_synonyms:
        return domain_synonyms[f"{word_norm}s"]

    # --- Fuzzy fallback (minor typo tolerance) ---
    # For example, "acqusition" → "acquisition"
    for key in domain_synonyms.keys():
        if abs(len(key) - len(word_norm)) <= 2 and sum(a != b for a, b in zip(key, word_norm)) <= 2:
            return domain_synonyms[key]

    # --- No match found ---
    return []


def preprocess(text: str) -> str:
    """
    Processes a given text by performing several preprocessing steps such as
    lowercasing, expanding acronyms, removing specific punctuation, and
    filtering out stop words. The method is designed to standardise the text
    and prepare it for further natural language processing tasks.

    :param text: The input text to be preprocessed.
    :type text: str
    :return: The processed text after applying the preprocessing steps.
    :rtype: str
    """
    # Convert to lowercase for case-insensitive processing
    normalized_text = text.lower()

    # Expand domain-specific acronyms before removing punctuation
    for acronym_pattern, expansion in ACRONYM_EXPANSIONS.items():
        normalized_text = re.sub(acronym_pattern, expansion, normalized_text)

    # Remove punctuation while preserving hyphens
    cleaned_text = re.sub(PUNCTUATION_PATTERN, '', normalized_text)

    # Filter out English stop words
    stop_words = set(stopwords.words('english'))
    tokens = cleaned_text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(filtered_tokens)


# Custom synonym augmentation using NLTK WordNet
def _extract_lemma_synonyms(synset, original_word: str) -> set[str]:
    """
    Extract and process lemma synonyms from a synset, excluding the original word.

    :param synset: WordNet synset containing lemmas
    :param original_word: The original word to exclude from results
    :return: Set of processed synonym strings
    :rtype: set
    """
    lemma_synonyms = set()
    for lemma in synset.lemmas():
        processed_synonym = lemma.name().replace('_', ' ')
        if processed_synonym.lower() != original_word.lower():
            lemma_synonyms.add(processed_synonym)
    return lemma_synonyms


def get_synonyms(word: str) -> list[str]:
    """
    Retrieve a list of synonyms for a given word using the WordNet lexical database.
    The function uses WordNet synsets to extract synonyms. Each synonym is
    processed to replace underscores with spaces and is included in the result if
    it differs from the input word (case-insensitive comparison). The returned
    list contains unique synonyms in alphabetical order.
    :param word: The word for which synonyms are to be retrieved.
    :type word: str
    :return: A list of synonyms related to the given word.
    :rtype: list
    """
    all_synonyms = set()
    synsets = wordnet.synsets(word)

    for synset in synsets:
        lemma_synonyms = _extract_lemma_synonyms(synset, word)
        all_synonyms.update(lemma_synonyms)

    return sorted(list(all_synonyms))


def _calculate_replacement_count(word_count: int, aug_prob: float, max_synonyms: int) -> int:
    """
    Calculate the number of words to replace during text augmentation.

    :param word_count: Total number of words in the text.
    :param aug_prob: Probability of augmentation (proportion of words to replace).
    :param max_synonyms: Maximum number of synonyms to use for replacement.
    :return: Number of words to replace.
    """
    words_by_probability = max(1, int(word_count * aug_prob))
    return min(max_synonyms, words_by_probability)


def _select_random_indices(word_count: int, num_to_replace: int) -> list[int]:
    """
    Select random word indices for replacement.

    :param word_count: Total number of words available.
    :param num_to_replace: Desired number of indices to select.
    :return: List of randomly selected indices.
    """
    actual_replacements = min(num_to_replace, word_count)
    return random.sample(range(word_count), actual_replacements)


def _replace_words_with_synonyms(words: list[str], indices: list[int]) -> list[str]:
    """
    Replace words at specified indices with their synonyms.

    :param words: Original list of words.
    :param indices: Indices of words to replace.
    :return: List of words with replacements applied.
    """
    augmented_words = words.copy()
    for idx in indices:
        if DOMAIN_SPECIFIC:
            synonyms = get_domain_synonyms(words[idx], DOMAIN_SYNONYMS, PROTECTED_TERMS)
        else:
            synonyms = get_synonyms(words[idx])
        if synonyms:
            augmented_words[idx] = random.choice(synonyms)
    return augmented_words


def augment_text(text: str, aug_prob=0.3, max_synonyms=2) -> str:
    """
    Augments a given text by replacing some words with their synonyms based on specified
    augmentation probability and maximum synonyms to use. The function randomly selects words
    from the input text and replaces them with their synonyms, if available.

    :param text: The input text to be augmented.
    :type text: str
    :param aug_prob: The probability of augmentation, determining the proportion of words to replace.
    :type aug_prob: float
    :param max_synonyms: The maximum number of synonyms to use for word replacement.
    :type max_synonyms: int
    :return: The augmented text with some words replaced by their synonyms.
    :rtype: str
    """
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return text

    num_to_replace = _calculate_replacement_count(word_count, aug_prob, max_synonyms)
    indices_to_replace = _select_random_indices(word_count, num_to_replace)
    augmented_words = _replace_words_with_synonyms(words, indices_to_replace)

    augmented_text = ' '.join(augmented_words)

    # Optionally, apply minimal structural variation
    if random.random() < 0.3:  # 30% chance of structural change
        augmented_text = augment_text_structural(augmented_text)

    return augmented_text


# --- Simple keyword-based classification ---
# Since transformers may also have dependency issues, implementing a keyword-based classifier
def _calculate_category_scores(text_lower: str, category_keywords: dict[str, list[str]]) -> dict[str, int]:
    """
    Calculates category scores based on keyword matches in text.

    Args:
        text_lower (str): Input text (lowercased).
        category_keywords (dict): Mapping of category -> keyword list.

    Returns:
        dict[str, int]: Scores per category.
    """
    scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[category] = score
    return scores


def classify_response(text: str, taxonomy: dict) -> tuple[str, dict[str, float]]:
    """
    Classify a response into a taxonomy category using keywords from taxonomy.

    Args:
        text (str): Response text.
        taxonomy (dict): Full taxonomy dict.

    Returns:
        tuple[str, dict[str, float]]:
            - predicted category
            - normalized scores
    """
    text_lower = text.lower()

    # Build category keywords dynamically from taxonomy
    category_keywords = build_category_keywords(taxonomy, lowercase=True, deduplicate=True)

    # Score categories
    scores = _calculate_category_scores(text_lower, category_keywords)

    # Pick best-scoring category
    predicted_category = max(scores, key=scores.get, default='uncategorised')
    if scores.get(predicted_category, 0) == 0:
        predicted_category = 'uncategorised'

    # Normalize scores
    total_score = sum(scores.values()) if sum(scores.values()) > 0 else 1
    normalized_scores = {k: v / total_score for k, v in scores.items()}

    return predicted_category, normalized_scores


def load_and_filter_responses(file_path: str, scenario_number: int, question_prefix: str) -> pd.DataFrame:
    """
    Loads response data from CSV and filters by scenario and question criteria.

    :param file_path: Path to the input CSV file containing responses.
    :param scenario_number: Scenario number to filter responses.
    :param question_prefix: Question prefix to filter responses.
    :return: Series of filtered response texts.
    :raises FileNotFoundError: If the input CSV file is not found.
    """
    df = pd.read_csv(file_path)
    # df = pd.DataFrame(response_set)
    df['ResponseText'] = df['ResponseText'].str.replace(r'[/]', ' ', regex=True)

    filtered_df = df[
        (df['ScenarioNumber'] == scenario_number) &
        (df['Question'].str.startswith(question_prefix))
        ][['ResponseID','ResponseText']]

    return filtered_df


def preprocess_responses(response_series: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and deduplicates response texts.

    :param response_series: Series of raw response texts.
    :return: List of preprocessed unique responses.
    """
    processed_rows = []
    for _, row in response_series.iterrows():
        if isinstance(row['ResponseText'], str) and row['ResponseText'].strip() != '':
            cleaned_text = preprocess(row['ResponseText'])
            processed_rows.append({'ResponseID': row['ResponseID'], 'ResponseText': cleaned_text})
    return pd.DataFrame(processed_rows).drop_duplicates(subset=['ResponseText'], keep='first')


def augment_response_dataset(responses: pd.DataFrame, aug_prob: float = 0.3,
                             max_synonyms: int = 2, sample_size: int = 500) -> pd.DataFrame:
    """
    Augments response dataset using synonym replacement and samples the result.
    Maintains ResponseID throughout the augmentation process.

    :param responses: DataFrame with ResponseID and ResponseText columns.
    :param aug_prob: Probability of word augmentation.
    :param max_synonyms: Maximum number of synonyms to use per word.
    :param sample_size: Maximum number of responses to the sample.
    :return: DataFrame of augmented and sampled responses with ResponseIDs preserved.
    """
    augmented_rows = []
    # Create both original and augmented versions for each response
    for _, row in responses.iterrows():
        response_id = row['ResponseID']
        response_text = row['ResponseText']
        # Add original response
        augmented_rows.append({
            'ResponseID': response_id,
            'ResponseText': response_text
        })
        # Add an augmented version
        augmented_text = augment_text(response_text, aug_prob=aug_prob, max_synonyms=max_synonyms)
        augmented_rows.append({
            'ResponseID': response_id,
            'ResponseText': augmented_text
        })

    # Convert to DataFrame
    augmented_df = pd.DataFrame(augmented_rows)

    # Sample if needed (maintaining ResponseID association)
    if len(augmented_df) > sample_size:
        augmented_df = augmented_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return augmented_df


def classify_with_taxonomy(text: str, child_keywords: dict, parent_keywords: dict):
    """
    Classify text against taxonomy at both child and parent levels.

    Args:
        text (str): Input text.
        child_keywords (dict): Mapping of child categories to keywords + parent reference.
        parent_keywords (dict): Mapping of parent categories to keyword lists.

    Returns:
        dict: {
            "child_scores": {child_category: score, ...},
            "parent_scores": {parent_category: score, ...},
            "predicted_child": str,
            "predicted_parent": str
        }
    """
    text_lower = text.lower()

    # --- Child-level scores
    child_scores = {}
    parent_scores = defaultdict(int)

    for child, data in child_keywords.items():
        keywords = [kw.lower() for kw in data["keywords"]]
        parent = data["parent"]
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            child_scores[child] = score
            parent_scores[parent] += score

    # --- Parent-only scan (fallback if no child hits)
    if not child_scores:
        for parent, keywords in parent_keywords.items():
            score = sum(1 for kw in [kw.lower() for kw in keywords] if kw in text_lower)
            if score > 0:
                parent_scores[parent] += score

    # --- Pick best matches
    predicted_child = max(child_scores, key=child_scores.get) if child_scores else "uncategorised"
    predicted_parent = max(parent_scores, key=parent_scores.get) if parent_scores else "uncategorised"

    return {
        "child_scores": dict(child_scores),
        "parent_scores": dict(parent_scores),
        "predicted_child": predicted_child,
        "predicted_parent": predicted_parent
    }


def classify_responses(responses: pd.DataFrame, child_keywords: dict, parent_keywords: dict) -> list[dict]:
    """
    Classifies a list of responses against taxonomy (child + parent).
    Adds both predicted child and predicted parent categories.
    """

    results = []
    for _, resp in responses.iterrows():
        resp_id = resp['ResponseID']
        text = resp['ResponseText']

        # Run hybrid taxonomy classifier
        classification = classify_with_taxonomy(text, child_keywords, parent_keywords)

        results.append({
            'ResponseID': resp_id,
            'response': text,
            'PredictedChild': classification["predicted_child"],
            'PredictedParent': classification["predicted_parent"],
            'ChildScores': str(classification["child_scores"]),
            'ParentScores': str(classification["parent_scores"])
        })

    return results


def summarise_classifications(results: list[dict]) -> dict:
    """
    Summarises classification results at both child and parent levels.

    Args:
        results (list[dict]): Classification results with 'PredictedChild' and 'PredictedParent'.

    Returns:
        dict: {
            "child_summary": {category: count},
            "parent_summary": {category: count}
        }
    """
    child_counts = {}
    parent_counts = {}

    for row in results:
        child = row.get("PredictedChild", "uncategorised")
        parent = row.get("PredictedParent", "uncategorised")

        child_counts[child] = child_counts.get(child, 0) + 1
        parent_counts[parent] = parent_counts.get(parent, 0) + 1

    return {
        "child_summary": child_counts,
        "parent_summary": parent_counts
    }


def save_and_display_results(results: list[dict], output_path: str) -> None:
    """
    Saves classification results to CSV and displays summary statistics.

    :param results: List of classification result dictionaries.
    :param output_path: Path to save the output CSV file.
    :raises IOError: If there are issues writing to the output CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(df.head())
    print(f"\nTotal responses classified: {len(df)}")
    print(f"\nCategory distribution:")
    print("\nPredicted Child distribution:")
    print(df['PredictedChild'].value_counts())

    print("\nPredicted Parent distribution:")
    print(df['PredictedParent'].value_counts())


def main(question: str, scenario: int, taxonomy: dict, merge_to_source: bool = False, strict_mode: bool = True):
    """
    Extended main pipeline:
      1. Load and filter responses
      2. Preprocess responses
      3. Augment dataset
      4. Validate responses (protected/conflict terms)
      5. Classify validated responses
      6. Save classification results
      7. Optionally merge results to source
    """

    # Extract the inner taxonomy for classification if wrapped
    if "MissionPlanningTaxonomy" in taxonomy:
        category_keywords = {humanise_key(k): v for k, v in taxonomy["MissionPlanningTaxonomy"].items()}
    else:
        category_keywords = {humanise_key(k): v for k, v in taxonomy.items()}

    # Load and filter data
    filtered_responses = load_and_filter_responses(
        file_path='./input/normalised_all_responses.csv',
        scenario_number=scenario,
        question_prefix=f"{question} - "
    )

    # Preprocess responses
    responses = preprocess_responses(filtered_responses)

    # Augment dataset with synonyms
    responses_aug = augment_response_dataset(
        responses=responses,
        aug_prob=0.3,
        max_synonyms=2,
        sample_size=500
    )

    # --- Run validation before classification ---
    context = map_question_to_context(question, taxonomy)
    validation_reports = {}

    enriched_responses = []
    for _, row in responses_aug.iterrows():
        text = row['ResponseText']
        report = document_validate(
            text=text,
            context=context,
            CONFLICT_RULES=CONFLICT_RULES,
            PROTECTED_TERMS=PROTECTED_TERMS,
            DOMAIN_SYNONYMS=DOMAIN_SYNONYMS,
            strict_mode=strict_mode
        )
        flags = extract_validation_flags(report)

        enriched_responses.append({
            "ResponseID": row["ResponseID"],
            "ResponseText": text,
            **flags
        })

    validated_df = pd.DataFrame(enriched_responses)

    # Save validation logs per scenario/question
    validation_output_path = f"./output/S{scenario}_{question}_validation_summary.json"
    pd.DataFrame(validation_reports).to_json(validation_output_path, orient="records", indent=2)
    print(f"Validation report saved: {validation_output_path}")

    # Load taxonomy
    parent_keywords, child_keywords = load_taxonomy("./docs/mission_planning_taxonomy.json")

    # Run classification for each scenario
    results = []
    for _, row in validated_df.iterrows():
        text = row['ResponseText']
        classification = classify_with_taxonomy(text, child_keywords, parent_keywords)

        results.append({
            "ResponseID": row["ResponseID"],
            "ResponseText": text,
            "PredictedChild": classification["predicted_child"],
            "PredictedParent": classification["predicted_parent"],
            "ChildScores": str(classification["child_scores"]),
            "ParentScores": str(classification["parent_scores"]),
            # keep validation flags
            "has_protected": row["has_protected"],
            "has_conflict": row["has_conflict"],
            "has_cross_conflict": row["has_cross_conflict"],
        })

    # --- Classify only validated responses ---
    #results = classify_responses(validated_df, category_keywords)

    classification_output_path = f"./output/S{scenario}_{question}_classification_results.csv"
    save_and_display_results(results, classification_output_path)

    # Create a summary report
    summary_output_path = f"./output/S{scenario}_{question}_classification_summary.csv"
    save_summary_report(results, summary_output_path)

    # Summarise both child and parent distributions
    summary = summarise_classifications(results)

    # Save summaries to JSON for reuse
    summary_path = f"./output/S{scenario}_{question}_classification_summary.json"
    with open(summary_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"\nSummary for Scenario {scenario}, {question}:")
    print("Child categories:", summary["child_summary"])
    print("Parent categories:", summary["parent_summary"])

    # Optionally merge classifications to source
    if merge_to_source:
        print("\n" + "=" * 60)
        print("Merging ALL classifications back to source data...")
        print("=" * 60)
        merge_all_classifications()


def extract_validation_flags(report: dict) -> dict:
    """
    Extracts simple boolean flags from a validation report summary.
    """
    summary = report.get("summary", {})
    return {
        "has_protected": summary.get("protected", 0) > 0,
        "has_conflict": summary.get("conflict", 0) > 0,
        "has_cross_conflict": summary.get("cross_conflict", 0) > 0,
        "validation_summary": summary
    }


def merge_classifications_to_source(
        classification_results_path: str,
        original_data_path: str,
        output_path: str
) -> None:
    """
    Merges classification results back into the original normalised responses dataset.
    Handles multiple classifications per response without duplicates.

    :param classification_results_path: Path to classification results CSV.
    :param original_data_path: Path to original normalised responses CSV.
    :param output_path: Path to save merged output CSV.
    """
    # Load classification results
    classifications_df = pd.read_csv(classification_results_path)

    # Load original normalized data
    original_df = pd.read_csv(original_data_path)

    # Group by ResponseID and aggregate unique categories
    # This handles the case where augmentation created multiple entries per ResponseID
    aggregated_classifications = classifications_df.groupby('ResponseID').agg({
        'predicted_category': lambda x: '|'.join(sorted(set(x))),  # Unique categories separated by |
        'all_scores': 'first'  # Take first scores (they should be similar for the same ResponseID)
    }).reset_index()

    # Rename columns for clarity
    aggregated_classifications.rename(columns={
        'predicted_category': 'PredictedCategories'
    }, inplace=True)

    # Merge back to original data
    merged_df = original_df.merge(
        aggregated_classifications[['ResponseID', 'PredictedCategories']],
        on='ResponseID',
        how='left'
    )

    # Fill NaN values for responses that weren't classified
    merged_df['PredictedCategories'].fillna('not_classified', inplace=True)

    # Save to a new file
    merged_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Merged classifications saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(merged_df)}")
    print(f"Classified responses: {len(merged_df[merged_df['PredictedCategories'] != 'not_classified'])}")
    print(f"Unclassified responses: {len(merged_df[merged_df['PredictedCategories'] == 'not_classified'])}")

    # Show the distribution of classification patterns
    print(f"\nClassification patterns:")
    category_counts = merged_df['PredictedCategories'].value_counts().head(10)
    print(category_counts)

    # Count responses with multiple categories
    multi_category = merged_df[merged_df['PredictedCategories'].str.contains('\|', na=False)]
    print(f"\nResponses with multiple categories: {len(multi_category)}")


def batch_merge_all_scenarios(question: str, scenarios: list[int]) -> None:
    """
    Merges classification results for multiple scenarios into the source data.

    :param question: Question identifier (e.g. "Q1")
    :param scenarios: List of scenario numbers to process
    """
    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Processing Scenario {scenario}, Question {question}")
        print(f"{'=' * 60}")

        classification_results_path = f"./output/S{scenario}_{question}_classification_results.csv"
        original_data_path = './input/normalised_all_responses.csv'
        output_path = f"./output/S{scenario}_{question}_normalised_classified_responses.csv"

        try:
            merge_classifications_to_source(
                classification_results_path=classification_results_path,
                original_data_path=original_data_path,
                output_path=output_path
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not process Scenario {scenario} - {e}")
            continue


def create_master_classified_file(question: str, scenarios: list[int]) -> None:
    """
    Creates a single master file with classifications from all scenarios.
    Each scenario's classifications are added as separate columns.

    :param question: Question identifier (e.g. "Q1")
    :param scenarios: List of scenario numbers to include
    """
    # Load original data
    master_df = pd.read_csv('./input/normalised_all_responses.csv')

    # Merge classifications from each scenario
    for scenario in scenarios:
        classification_path = f"./output/S{scenario}_{question}_classification_results.csv"

        try:
            # Load and aggregate classifications for this scenario
            classifications_df = pd.read_csv(classification_path)

            # Aggregate unique categories per ResponseID
            aggregated = classifications_df.groupby('ResponseID').agg({
                'predicted_category': lambda x: '|'.join(sorted(set(x)))
            }).reset_index()

            # Rename column with scenario number
            column_name = f'S{scenario}_{question}_Categories'
            aggregated.rename(columns={'predicted_category': column_name}, inplace=True)

            # Merge to master
            master_df = master_df.merge(
                aggregated,
                on='ResponseID',
                how='left'
            )

            # Fill NaN
            master_df[column_name].fillna('', inplace=True)

            print(f"Added classifications for Scenario {scenario}")

        except FileNotFoundError:
            print(f"Warning: Classification file not found for Scenario {scenario}")
            continue

    # Save the master file
    output_path = './output/normalised_all_classified_responses.csv'
    master_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Master classified file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nColumns added:")
    for scenario in scenarios:
        col_name = f'S{scenario}_{question}_Categories'
        if col_name in master_df.columns:
            print(f"  - {col_name}")

    print(f"\nTotal responses: {len(master_df)}")


def merge_all_classifications_single_column(
        classification_files: list[tuple[int, str]],
        original_data_path: str,
        output_path: str,
        combine_child_parent: bool = True
) -> None:
    """
    Merges classification results into the original dataset.
    Uses PredictedChild / PredictedParent instead of predicted_category.

    :param classification_files: List of (scenario, question) tuples.
    :param original_data_path: Path to normalised responses CSV.
    :param output_path: Path to save merged output.
    :param combine_child_parent: If True, combines Child and Parent into a single string (Child|Parent).
    """

    original_df = pd.read_csv(original_data_path)
    current_date = datetime.now().strftime('%Y-%m-%d')

    if 'Classification' not in original_df.columns:
        original_df['Classification'] = 'not_classified'
    if 'Version' not in original_df.columns:
        original_df['Version'] = 'v0.1'
    if 'ChangeNote' not in original_df.columns:
        original_df['ChangeNote'] = 'Initial merge'
    if 'ChangeDate' not in original_df.columns:
        original_df['ChangeDate'] = ''

    for scenario, question in classification_files:
        classification_path = f"./output/S{scenario}_{question}_classification_results.csv"

        try:
            file_mod_timestamp = os.path.getmtime(classification_path)
            file_mod_date = datetime.fromtimestamp(file_mod_timestamp).strftime("%Y-%m-%d")

            classifications_df = pd.read_csv(classification_path)

            # Ensure required cols exist
            if not {"PredictedChild", "PredictedParent"}.issubset(classifications_df.columns):
                print(f"Warning: Expected classification columns not found in {classification_path}")
                print(f"Available columns: {classifications_df.columns.tolist()}")
                continue

            # Create a unified column
            if combine_child_parent:
                classifications_df['ClassificationCombined'] = (
                    classifications_df['PredictedChild'].astype(str) + "|" +
                    classifications_df['PredictedParent'].astype(str)
                )
            else:
                classifications_df['ClassificationCombined'] = classifications_df.apply(
                    lambda r: r['PredictedChild'] if r['PredictedChild'] != 'uncategorised' else r['PredictedParent'],
                    axis=1
                )

            aggregated = classifications_df.groupby('ResponseID').agg({
                'ClassificationCombined': lambda x: '|'.join(sorted(set(x)))
            }).reset_index()

            for _, row in aggregated.iterrows():
                response_id = row['ResponseID']
                category = row['ClassificationCombined']
                mask = original_df['ResponseID'] == response_id

                original_df.loc[mask, 'Classification'] = category
                if category not in ['uncategorised', 'not_classified', '']:
                    original_df.loc[mask, 'Version'] = 'v0.2'
                    original_df.loc[mask, 'ChangeNote'] = 'Initial classification'
                    original_df.loc[mask, 'ChangeDate'] = file_mod_date

            print(f"Merged classifications for Scenario {scenario}, {question} (file date: {file_mod_date})")

        except FileNotFoundError:
            print(f"Classification file not found for Scenario {scenario}, {question}")
            continue
        except Exception as e:
            print(f"Error processing Scenario {scenario}, {question}: {str(e)}")
            continue

    original_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Merged file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(original_df)}")
    classified = original_df[
        (original_df['Classification'] != 'not_classified') &
        (original_df['Classification'] != 'uncategorised') &
        (original_df['Classification'] != '')
    ]
    print(f"Classified responses: {len(classified)}")
    print(f"Unclassified responses: {len(original_df) - len(classified)}")
    print(f"\nVersion distribution:\n{original_df['Version'].value_counts()}")
    print(f"\nTop 10 classification patterns:\n{original_df['Classification'].value_counts().head(10)}")
    multi_category = original_df[original_df['Classification'].str.contains('\|', na=False)]
    print(f"\nResponses with multiple categories: {len(multi_category)}")
    updated_records = original_df[original_df['Version'] == 'v0.2']
    print(f"\nRecords updated to v0.2: {len(updated_records)}")
    print(f"Classification date: {current_date}")


def merge_all_classifications_multi_column(
        classification_files: list[tuple[int, str]],
        original_data_path: str,
        output_path: str
) -> None:
    """
    Merges classification results into the original data with TWO columns:
    - PredictedChilds: all child categories (deduplicated, joined with "|")
    - PredictedParents: all parent categories (deduplicated, joined with "|")

    Handles multiple scenario/question classification files and preserves version control fields.
    """
    # Load original data
    original_df = pd.read_csv(original_data_path)

    current_date = datetime.now().strftime('%Y-%m-%d')

    # Ensure columns exist
    if 'PredictedChilds' not in original_df.columns:
        original_df['PredictedChilds'] = 'not_classified'
    if 'PredictedParents' not in original_df.columns:
        original_df['PredictedParents'] = 'not_classified'
    if 'Version' not in original_df.columns:
        original_df['Version'] = original_df.get('Version', 'v0.1')
    if 'ChangeNote' not in original_df.columns:
        original_df['ChangeNote'] = original_df.get('ChangeNote', 'Initial merge')
    if 'ChangeDate' not in original_df.columns:
        original_df['ChangeDate'] = ''

    # Process each classification file
    for scenario, question in classification_files:
        classification_path = f"./output/S{scenario}_{question}_classification_results.csv"

        try:
            file_mod_timestamp = os.path.getmtime(classification_path)
            file_mod_date = datetime.fromtimestamp(file_mod_timestamp).strftime("%Y-%m-%d")

            classifications_df = pd.read_csv(classification_path)

            # Check columns exist
            if not {"PredictedChild", "PredictedParent"}.issubset(classifications_df.columns):
                print(f"Warning: required columns not found in {classification_path}")
                print(f"Available columns: {classifications_df.columns.tolist()}")
                continue

            # Aggregate all unique child and parent classifications per ResponseID
            aggregated = classifications_df.groupby("ResponseID").agg({
                "PredictedChild": lambda x: "|".join(sorted(set(x))),
                "PredictedParent": lambda x: "|".join(sorted(set(x)))
            }).reset_index()

            # Update original_df with merged classifications
            for _, row in aggregated.iterrows():
                response_id = row["ResponseID"]
                child_cats = row["PredictedChild"]
                parent_cats = row["PredictedParent"]

                mask = original_df['ResponseID'] == response_id

                original_df.loc[mask, 'PredictedChilds'] = child_cats
                original_df.loc[mask, 'PredictedParents'] = parent_cats

                if (child_cats not in ['uncategorised', 'not_classified', '']) or \
                   (parent_cats not in ['uncategorised', 'not_classified', '']):
                    original_df.loc[mask, 'Version'] = 'v0.2'
                    original_df.loc[mask, 'ChangeNote'] = 'Initial classification'
                    original_df.loc[mask, 'ChangeDate'] = file_mod_date

            print(f"Merged classifications for Scenario {scenario}, {question} (file date: {file_mod_date})")

        except FileNotFoundError:
            print(f"Classification file not found for Scenario {scenario}, {question}")
            continue
        except Exception as e:
            print(f"Error processing Scenario {scenario}, {question}: {str(e)}")
            continue

    # Save merged file
    original_df.to_csv(output_path, index=False)

    # Print stats
    print(f"\n{'=' * 60}")
    print(f"Merged file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(original_df)}")

    classified = original_df[
        (original_df['PredictedChilds'] != 'not_classified') |
        (original_df['PredictedParents'] != 'not_classified')
    ]
    print(f"Classified responses: {len(classified)}")
    print(f"Unclassified responses: {len(original_df) - len(classified)}")

    print(f"\nVersion distribution:")
    print(original_df['Version'].value_counts())

    print(f"\nTop 10 child classification patterns:")
    print(original_df['PredictedChilds'].value_counts().head(10))

    print(f"\nTop 10 parent classification patterns:")
    print(original_df['PredictedParents'].value_counts().head(10))

    updated_records = original_df[original_df['Version'] == 'v0.2']
    print(f"\nRecords updated to v0.2: {len(updated_records)}")
    print(f"Classification date: {current_date}")


def merge_all_classifications():
    """
    Convenience function to merge multiple question classifications from multiple scenarios
    into the source dataset, outputting *two* classification columns:
    - PredictedChilds
    - PredictedParents
    """
    classification_files = [
        (1, 'Q1'),
        (2, 'Q1'),
        (3, 'Q1'),
        (1, 'Q2'),
        (2, 'Q2'),
        (3, 'Q2'),
        (1, 'Q3'),
        (2, 'Q3'),
        (3, 'Q3')
    ]

    merge_all_classifications_multi_column(
        classification_files=classification_files,
        original_data_path='./input/normalised_all_responses.csv',
        output_path='./output/normalised_all_classified_responses.csv'
    )



def save_summary_report(results: list[dict], summary_path: str) -> None:
    """
    Creates a summary CSV report of classification results with validation flags.

    Args:
        results (list[dict]): Per-response classification and validation results.
        summary_path (str): Path to save the summary CSV.
    """
    df = pd.DataFrame(results)

    # Group by category and flag states
    summary_child = df.groupby(
        ["PredictedChild", "has_protected", "has_conflict", "has_cross_conflict"]
    ).size().reset_index(name="count")

    summary_parent = df.groupby(
        ["PredictedParent", "has_protected", "has_conflict", "has_cross_conflict"]
    ).size().reset_index(name="count")

    # Save child summary
    child_path = summary_path.replace('.csv', '_child.csv')
    summary_child.to_csv(child_path, index=False)
    print(f"\nChild summary report saved to: {child_path}")
    print(summary_child.head(10))

    # Save parent summary
    parent_path = summary_path.replace('.csv', '_parent.csv')
    summary_parent.to_csv(parent_path, index=False)
    print(f"\nParent summary report saved to: {parent_path}")
    print(summary_parent.head(10))


def validate_domain_synonyms(file_path: str) -> dict:
    """
    Validate the structure of the DOMAIN_SYNONYMS JSON file.

    Expected structure:
    {
        "term": ["synonym1", "synonym2", ...],
        ...
    }

    Rules:
    - Keys must be strings.
    - Values must be lists.
    - Each list item must be a string.
    """

    errors = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("DOMAIN_SYNONYMS file must contain a dictionary at the top level.")

    for key, value in data.items():
        if not isinstance(key, str):
            errors.append(f"Key {key} is not a string.")

        if not isinstance(value, list):
            errors.append(f"Value for key '{key}' must be a list, got {type(value).__name__}.")
            continue

        for i, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(
                    f"Value at {key}[{i}] must be a string, got {type(item).__name__}."
                )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "count_terms": len(data),
        "count_synonyms": sum(len(v) for v in data.values() if isinstance(v, list))
    }


def build_category_key_map(category_keywords: dict) -> dict:
    """
    Build a normalisation map from many possible variants (camelCase, snake_case,
    no-spaces, etc.) to the canonical CATEGORY_KEYWORDS labels.

    Args:
        category_keywords (dict): The canonical category keywords dict where keys are
                                  human-readable category labels (e.g. "Terrain and Bathymetry").

    Returns:
        dict: Mapping of normalised variants → canonical labels.
    """
    key_map = {}

    for canonical in category_keywords.keys():
        base = canonical.lower()

        # Normalise by removing spaces, hyphens, underscores
        normalized = re.sub(r'[^a-z0-9]', '', base)
        key_map[normalized] = canonical

        # Variants: camelCase style, snake_case, kebab-case
        camel = ''.join(word.capitalize() for word in base.split())
        key_map[camel.lower()] = canonical  # terrainandbathymetry
        key_map[camel] = canonical          # TerrainAndBathymetry

        snake = '_'.join(base.split())
        key_map[snake] = canonical          # terrain_and_bathymetry

        kebab = '-'.join(base.split())
        key_map[kebab] = canonical          # terrain-and-bathymetry

    return key_map


def normalise_category_keys(imported_dict: dict, category_keywords: dict) -> dict:
    """
    Remap imported category keys (in arbitrary formatting) to the canonical keys
    from CATEGORY_KEYWORDS.

    Args:
        imported_dict (dict): Imported dictionary with potentially inconsistent keys.
        category_keywords (dict): Canonical CATEGORY_KEYWORDS dict.

    Returns:
        dict: Dictionary with keys remapped to canonical labels.
    """
    key_map = build_category_key_map(category_keywords)

    def normalise_key(key: str) -> str:
        return re.sub(r'[^a-z0-9]', '', key.lower())

    normalized_dict = {}
    for k, v in imported_dict.items():
        norm_k = normalise_key(k)
        mapped_key = key_map.get(norm_k, k)  # fallback to original if not found
        normalized_dict[mapped_key] = v

    return normalized_dict


def humanise_key(key: str) -> str:
    """
    Convert taxonomy keys like 'MissionPlanningTaxonomy'
    into 'Mission Planning Taxonomy'.
    Handles CamelCase, PascalCase, snake_case, and kebab-case.
    """
    # Replace underscores and hyphens with spaces
    key = key.replace("_", " ").replace("-", " ")
    # Add spaces before capital letters (but not at start)
    key = re.sub(r'(?<!^)(?=[A-Z])', ' ', key)
    # Normalise whitespace and title case
    key = re.sub(r'\s+', ' ', key).strip()
    return key


def normalise_key(name: str) -> str:
    """
    Convert CamelCase or mixed keys to a human-readable form.
    Example: "TerrainAndBathymetry" -> "Terrain and Bathymetry"
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).strip()


def load_taxonomy(json_path: str):
    """
    Load mission planning taxonomy JSON and build hybrid CATEGORY_KEYWORDS.

    Returns:
        parent_keywords (dict): Parent categories with union of child keywords.
        child_keywords (dict): Child categories with their specific keywords + parent reference.
    """
    with open(json_path, "r") as f:
        taxonomy = json.load(f)

    taxonomy = taxonomy.get("MissionPlanningTaxonomy", taxonomy)

    parent_keywords = {}
    child_keywords = {}

    for parent_key, parent_val in taxonomy.items():
        parent_name = normalise_key(parent_key)
        parent_set = set()

        for child_key, keywords in parent_val.get("Concepts", {}).items():
            child_name = normalise_key(child_key)
            child_keywords[child_name] = {
                "keywords": keywords,
                "parent": parent_name
            }
            parent_set.update(keywords)

        parent_keywords[parent_name] = sorted(list(parent_set))

    return parent_keywords, child_keywords


def load_protected_terms(path: str, flatten: bool = True) -> set[str] | dict[str, set[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = {k: {t.lower().strip() for t in v} for k, v in data.items()}

    if flatten:
        # Merge into one global set
        return set().union(*grouped.values())
    return grouped


def load_conflict_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # normalise to lowercase for consistent matching
    normalized = {}
    for ctx, rules in data.items():
        normalized[ctx.lower()] = {
            "protected": [p.lower().strip() for p in rules.get("protected", [])],
            "synonyms": [s.lower().strip() for s in rules.get("synonyms", [])]
        }
    return normalized


if __name__ == "__main__":
    # Define keywords for each category as a module constant
    # Load the JSON file
    with open('./docs/mission_planning_taxonomy.json', 'r') as f:
        marine_planning_taxonomy_raw = json.load(f)

    # Extract the actual taxonomy data from the wrapper
    if "MissionPlanningTaxonomy" in marine_planning_taxonomy_raw:
        marine_planning_taxonomy = marine_planning_taxonomy_raw["MissionPlanningTaxonomy"]
    else:
        marine_planning_taxonomy = marine_planning_taxonomy_raw

    CATEGORY_KEYWORDS = {humanise_key(k): v for k, v in marine_planning_taxonomy.items()}
    # Debug print
    print("\nCategory Keyword Summary")
    print("==========================")
    for cat, kws in CATEGORY_KEYWORDS.items():
        print(f"{cat}: {len(kws)} keywords")

    # Load DOMAIN_SYNONYMS from JSON file
    report = validate_domain_synonyms("./docs/domain_synonyms.json")
    print("\nDomain Synonym Validation Report:")
    print("====================================")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Terms: {report['count_terms']}")
    print(f"  Synonyms: {report['count_synonyms']}")
    if not report["is_valid"]:
        print("\nErrors found:")
        for err in report["errors"]:
            print(f"  - {err}")

    with open('./docs/domain_synonyms.json', 'r', encoding='utf-8') as f:
        DOMAIN_SYNONYMS = {k.lower(): [s.lower() for s in v] for k, v in json.load(f).items()}
    print("Loaded terms:", len(DOMAIN_SYNONYMS))
    print("Sample mission synonyms:", DOMAIN_SYNONYMS.get("mission"))

    # Load protected terms from the JSON file
    PROTECTED_TERMS = load_protected_terms("./docs/protected_terms.json", flatten=False)
    # Load conflict rules from a JSON file
    CONFLICT_RULES = load_conflict_rules("./docs/conflict_rules.json")

    # Run classification for all scenarios
    for scenario_num in [1, 2, 3]:
        print(f"\n{'#' * 60}")
        for question_num in ["Q1", "Q2", "Q3"]:
            print(f"Running classification for Scenario {scenario_num}, Question {question_num}")
            main(question_num, scenario_num, marine_planning_taxonomy_raw, merge_to_source=False)

    # After all scenarios are classified, merge into a single file
    print(f"\n{'#' * 60}")
    print(f"# Merging all classifications into master file")
    print(f"{'#' * 60}\n")
    merge_all_classifications()

    # Save a suppression report
    if len(SUPPRESSION_LOG) > 0:
        suppression_log_path = './output/suppression_log.csv'
        with open(suppression_log_path, 'w') as f:
            f.write('\n'.join(SUPPRESSION_LOG))
        print(f"\nSuppression log saved to: {suppression_log_path}")

