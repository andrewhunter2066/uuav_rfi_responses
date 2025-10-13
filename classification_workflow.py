#!/usr/bin/env python3

import pandas as pd
import random
import re
import nltk
import os
import json
from nltk.corpus import stopwords, wordnet
from datetime import datetime
from typing import List, Optional, Any, NamedTuple
from collections import defaultdict, Counter

from openpyxl.pivot.fields import Boolean

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

# Question prefix to taxonomy seed context mapping
QUESTION_CONTEXT_SEEDS = {
    "q1": ["TerrainAndBathymetry", "EnvironmentalAndOceanographicConditions", "NavigationAndPositioning"],
    "q2": ["VehicleCapabilitiesAndConstraints", "MissionParametersAndObjectives", "OperationalLogistics"],
    "q3": ["EstimationAndUncertaintyModeling", "DataProductsAndRequirements", "HistoricalAndContextualData"],
    "q4": ["ThreatsAndRiskManagement", "NavigationAndPositioning", "TestAndEvaluationScenarios"],
    "q5": ["ThreatsAndRiskManagement", "AuditAndAccountability", "DataProductsAndRequirements"],
    "q6": ["HistoricalAndContextualData", "DataProductsAndRequirements", "AuditAndAccountability"],
}

# Regex pattern to insert space between lowercase and uppercase letters
CAMEL_CASE_SPLIT_PATTERN = r'([a-z])([A-Z])'

# Magic number for log suppression
MAX_CONTEXT_LENGTH = 120

# Magic number for word augmentation
MIN_WORDS_FOR_AUGMENTATION = 4

# Which method to use, domain-specific (True) or general NLP (False)
DOMAIN_SPECIFIC = True

# Pattern-based text transformation rules for requirement augmentation
_REGEX_FLAGS = re.IGNORECASE

_TRANSFORMATION_NEEDED_FOR = (r'\b(\w+) is needed for (\w+)', r'\2 requires \1')
_TRANSFORMATION_SYSTEM_SUPPORT = (r'\bsystem must support (\w+)', r'\1 support is required')
_TRANSFORMATION_CAPABILITY = (r'\b(\w+) capability', r'capability to \1')
_TRANSFORMATION_REQUIREMENT_FOR = (r'\brequirement for (\w+)', r'\1 requirement')
_TRANSFORMATION_CONFIDENCE_IN = (r'\bconfidence in (\w+)', r'\1 confidence')
_TRANSFORMATION_RISK_OF = (r'\brisk of (\w+)', r'\1 risk')
_TRANSFORMATION_UNDERSTANDING_OF = (r'\bunderstanding (?:of )?(\w+(?:\s+\w+)*)', r'\1 understanding')
_TRANSFORMATION_BASED_ON = (r'\bbased on (\w+)', r'using \1')
_TRANSFORMATION_PRIOR_TO = (r'\bprior to (\w+)', r'before \1')
_TRANSFORMATION_DUE_TO = (r'\bdue to (\w+)', r'caused by \1')
_TRANSFORMATION_FAILURE_OF = (r'\bfailure of (\w+)', r'\1 failure')
_TRANSFORMATION_LOSS_OF = (r'\bloss of (\w+)', r'\1 loss')
_TRANSFORMATION_ASSESSMENT_OF = (r'\bassessment of (\w+)', r'\1 assessment')
_TRANSFORMATION_MODELING_FOR = (r'\bmodeling for (\w+)', r'\1 modeling')
_TRANSFORMATION_PERFORMANCE_OF = (r'\bperformance of (\w+)', r'\1 performance')

_TEXT_TRANSFORMATIONS = [
    _TRANSFORMATION_NEEDED_FOR,
    _TRANSFORMATION_SYSTEM_SUPPORT,
    _TRANSFORMATION_CAPABILITY,
    _TRANSFORMATION_REQUIREMENT_FOR,
    _TRANSFORMATION_CONFIDENCE_IN,
    _TRANSFORMATION_RISK_OF,
    _TRANSFORMATION_UNDERSTANDING_OF,
    _TRANSFORMATION_BASED_ON,
    _TRANSFORMATION_PRIOR_TO,
    _TRANSFORMATION_DUE_TO,
    _TRANSFORMATION_FAILURE_OF,
    _TRANSFORMATION_LOSS_OF,
    _TRANSFORMATION_ASSESSMENT_OF,
    _TRANSFORMATION_MODELING_FOR,
    _TRANSFORMATION_PERFORMANCE_OF,
]

# Constants for conflict statuses
_STATUS_PROTECTED = "protected"
_STATUS_SYNONYM_CONFLICT = "synonym_conflict"
_STATUS_ALLOWED = "allowed"
_STATUS_UNKNOWN_CONTEXT = "unknown_context"
_STATUS_VALID = "valid"
_STATUS_CONFLICT = "conflict"
_STATUS_CROSS_CONFLICT = "cross_conflict"

_CONFLICT_STATUSES = (_STATUS_PROTECTED, _STATUS_SYNONYM_CONFLICT)

# Default result when no conflicts are found
_NO_CONFLICT_RESULT = {
    "status": _STATUS_ALLOWED,
    "context": None,
    "reason": "No conflicts found in any context."
}

# Status priority mapping for determining final validation status
_STATUS_PRIORITIES = {
    _STATUS_PROTECTED: 3,
    _STATUS_CROSS_CONFLICT: 2,
    _STATUS_CONFLICT: 1,
    _STATUS_VALID: 0
}

# Text preprocessing constants
FORWARD_SLASH_REPLACEMENT_PATTERN = r'[/]'
FORWARD_SLASH_REPLACEMENT = ' '

# Constants for classification values
UNCLASSIFIED_VALUES = {'uncategorised', 'not_classified', ''}
DEFAULT_VERSION = 'v0.1'
UPDATED_VERSION = 'v0.2'
INITIAL_MERGE_NOTE = 'Initial merge'
INITIAL_CLASSIFICATION_NOTE = 'Initial classification'

# Configuration constants for classification merging
CLASSIFICATION_FILE_CONFIGS = [
    (1, 'Q1'),
    (2, 'Q1'),
    (3, 'Q1'),
    (1, 'Q2'),
    (2, 'Q2'),
    (3, 'Q2'),
    (1, 'Q3'),
    (2, 'Q3'),
    (3, 'Q3'),
    (1, 'Q4'),
    (2, 'Q4'),
    (3, 'Q4'),
    (1, 'Q5'),
    (2, 'Q5'),
    (3, 'Q5'),
    (1, 'Q6'),
    (2, 'Q6'),
    (3, 'Q6'),
]

ORIGINAL_DATA_PATH = './input/normalised_all_responses.csv'
CLASSIFIED_OUTPUT_PATH = './output/normalised_all_classified_responses.csv'

# Regex pattern to insert space before capital letters (except at the start of string)
_CAMEL_CASE_PATTERN = r'(?<!^)(?=[A-Z])'
# Regex pattern to normalize multiple whitespace characters into single space
_WHITESPACE_NORMALIZATION_PATTERN = r'\s+'
# Regex pattern to insert space before uppercase letter followed by lowercase (e.g., "camelCase" -> "camel Case")
_UPPERCASE_LOWERCASE_PATTERN = r'(.)([A-Z][a-z]+)'
# Regex pattern to insert space before uppercase letter after lowercase/digit (e.g., "camel2Case" -> "camel2 Case")
_LOWERCASE_DIGIT_UPPERCASE_PATTERN = r'([a-z0-9])([A-Z])'

# Suppression Log
SUPPRESSION_LOG = []


class ValidationRules:
    """
    Encapsulates validation rules for document validation including conflict rules,
    protected terms, and domain synonyms.
    """

    def __init__(self, conflict_rules: dict, protected_terms: set, domain_synonyms: dict):
        """
        Initialise validation rules.

        :param conflict_rules: A dictionary defining the conflict rules to be applied.
        :param protected_terms: A set of terms that should not be altered or flagged
                                during validation.
        :param domain_synonyms: A dictionary mapping synonyms to a domain-specific
                                canonical form.
        """
        self.conflict_rules = conflict_rules
        self.protected_terms = protected_terms
        self.domain_synonyms = domain_synonyms


class ScenarioPaths(NamedTuple):
    """Paths for classification processing of a single scenario."""
    classification_results: str
    original_data: str
    output: str


def _process_keywords(keywords: list[str], lowercase: bool, deduplicate: bool) -> list[str]:
    """
    Process a list of keywords by optionally normalising to lowercase and removing duplicates.

    :param keywords: List of keywords to process.
    :param lowercase: Whether to convert keywords to lowercase and strip whitespace.
    :param deduplicate: Whether to remove duplicate keywords while preserving order.
    :return: Processed list of keywords.
    :rtype: list[str]
    """
    processed = keywords

    if lowercase:
        processed = [kw.lower().strip() for kw in processed]

    if deduplicate:
        seen = set()
        processed = [kw for kw in processed if not (kw in seen or seen.add(kw))]

    return processed


def build_category_keywords(taxonomy: dict, lowercase: bool = True, deduplicate: bool = True) -> dict[str, list[str]]:
    """
    Builds a dictionary mapping categories to a list of associated keywords.

    The function processes the provided taxonomy to extract keywords from its "Concepts"
    dictionaries. It allows optional normalisation of keywords to lowercase and
    elimination of duplicate keywords while preserving their order.

    :param taxonomy: A dictionary where each top-level key represents a category, and
        its value is expected to contain a "Concepts" sub-dictionary with keyword groupings.
    :param lowercase: A boolean flag to indicate whether to convert keywords to lowercase.
        Defaults to True.
    :param deduplicate: A boolean flag to indicate whether to remove duplicate keywords
        while preserving their order. Defaults to True.
    :return: A dictionary with top-level categories as keys and lists of processed keywords
        as values.
    :rtype: dict[str, list[str]]
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

        # Process keywords with optional normalisation and deduplication
        category_keywords[category] = _process_keywords(keywords, lowercase, deduplicate)

    return category_keywords


def tokenise_text(text: str) -> List[str]:
    """
    Tokenises the given text into a list of normalised tokens. This function extracts
    alphanumeric text content in a basic manner, preserving multi-word phrases.

    :param text: A string to be tokenised.
    :type text: str
    :return: A list of normalised tokens extracted from the input text.
    :rtype: List[str]
    """
    # Normalise and keep multi-word phrases (basic approach)
    tokens = re.findall(r"[A-Za-z0-9\-\s]+", text)
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    return tokens


def document_validate(text: str,
                      context: str,
                      validation_rules: ValidationRules,
                      strict_mode: bool = False,
                      export_csv: str = None,
                      export_json: str = None) -> dict:
    """
    Validates a given text against specified validation rules, optionally
    generating reports in CSV or JSON format. This function processes tokens
    from the text and evaluates them for adherence to conflict rules, checks
    for protected terms, and applies domain synonyms to ensure consistency
    according to the context provided. It returns a structured report containing
    validation results.

    :param text: The input text to be validated.
    :param context: The context against which the validation is performed.
    :param validation_rules: A ValidationRules object containing conflict rules,
                             protected terms, and domain synonyms.
    :param strict_mode: A boolean flag indicating whether strict rule enforcement
                        is enabled. Defaults to False.
    :param export_csv: An optional file path to export the validation results
                       in CSV format. Defaults to None.
    :param export_json: An optional file path to export the validation results
                        in JSON format. Defaults to None.
    :return: A dictionary containing the validation results, including tokens
             processed and any conflicts or issues identified.
    """
    tokens = tokenise_text(text)
    report = batch_validate(
        tokens,
        context,
        validation_rules,
        strict_mode
    )

    # Export if requested
    df = pd.DataFrame(report["results"])
    if export_csv:
        df.to_csv(export_csv, index=False)
    if export_json:
        df.to_json(export_json, orient="records", indent=2)

    return report


def _build_report_path(output_dir: str, scenario: int, question: str, file_format: str) -> str:
    """
    Build a standardised report file path.

    :param output_dir: The directory where reports will be saved.
    :param scenario: The scenario number.
    :param question: The question identifier.
    :param file_format: The file format extension (e.g. 'csv', 'json').
    :return: Complete file path for the report.
    """
    filename = f"S{scenario}_{question}_conflict_report.{file_format}"
    return os.path.join(output_dir, filename)


def validate_responses_csv(input_csv: str,
                           output_dir: str,
                           validation_rules: ValidationRules,
                           strict_mode=False) -> dict:
    """
    Validates responses from a CSV file using specified validation rules, processes the responses grouped
    by scenario and question, and generates conflict reports in both CSV and JSON format. A summary of
    the validation for each group is returned.
    :param input_csv: The input CSV file path containing responses for validation.
    :type input_csv: str
    :param output_dir: The directory where the conflict reports will be saved.
    :type output_dir: str
    :param validation_rules: An object representing the validation rules to be used for the process.
    :type validation_rules: ValidationRules
    :param strict_mode: If True, applies stricter validation rules, otherwise applies less strict rules.
    :type strict_mode: bool
    :return: A summary of the validation for each scenario and question as a dictionary, with
            keys as tuples (scenario, question) and values as their respective summaries.
    :rtype: dict
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    all_summary = {}

    for (scenario, question), group in df.groupby(["ScenarioNumber", "Question"]):
        context = map_question_to_context(question)
        text_block = " ".join(group["ResponseText"].dropna().astype(str))

        # Validate document-level
        report = document_validate(
            text=text_block,
            context=context,
            validation_rules=validation_rules,
            strict_mode=strict_mode,
            export_csv=_build_report_path(output_dir, scenario, question, "csv"),
            export_json=_build_report_path(output_dir, scenario, question, "json")
        )

        # Store summary
        all_summary[(scenario, question)] = report["summary"]

    return all_summary


def _expand_frontier(taxonomy: dict, frontier: set[str], expanded: set[str]) -> set[str]:
    """
    Expand the frontier by one level using RelatedTo relationships.

    :param taxonomy: The taxonomy dictionary containing domain relationships
    :param frontier: Current set of domains to expand from
    :param expanded: Set of already expanded domains to avoid duplicates
    :return: New frontier set containing newly discovered domains
    """
    new_frontier = set()
    for domain in frontier:
        related = taxonomy.get(domain, {}).get("RelatedTo", [])
        for rel in related:
            if rel not in expanded:
                expanded.add(rel)
                new_frontier.add(rel)
    return new_frontier


def expand_contexts(taxonomy: dict, seeds: list[str], depth: int = 1, normalise: bool = True) -> list[str]:
    """
    Expand a set of seed concepts based on their relationships in a taxonomy.
    This function iteratively expands the provided seed concepts by traversing
    their "RelatedTo" relationships in the given taxonomy up to a specified depth.
    The expanded set of concepts can optionally be normalised into a more human-readable
    format (e.g. converting a camel case to a title case).
    :param taxonomy: A dictionary representing the hierarchical structure and relationships
        between different domains. Each key represents a domain, and its value is expected
        to contain a dictionary with a "RelatedTo" key, which lists related domains.
    :param seeds: A list of seed concept strings that serve as the starting point for
        taxonomy traversal.
    :param depth: The maximum depth to traverse the "RelatedTo" relationships
        for expanding the seed concepts. Defaults to 1.
    :param normalise: A boolean indicating whether to normalise the expanded
        concepts into a human-readable format by converting camel case
        strings to a title case. Defaults to True.
    :return: A sorted list of expanded concepts as strings, optionally normalised
        based on the `normalise` parameter.
    """
    expanded = set(seeds)
    frontier = set(seeds)

    for _ in range(depth):
        frontier = _expand_frontier(taxonomy, frontier, expanded)

    result = [camel_to_title(x) for x in expanded] if normalise else list(expanded)
    return sorted(result)


def map_question_to_context(question: str, taxonomy: dict) -> str:
    """
    Maps a given question to a corresponding context based on provided taxonomy
    and specified mapping rules.

    This function processes the input question, determines its prefix, and identifies which
    contexts are associated with the prefix from a predefined map (QUESTION_CONTEXT_SEEDS). It leverages
    the expand_contexts function to fetch a list of additional related contexts from the
    taxonomy, normalises them, and combines them into a single output string. If no suitable
    prefix match is found, "General" is returned as the default context.

    :param question: The question string to be mapped, typically starting with a specific
                     prefix (e.g. "q1", "q2").
    :type question: str
    :param taxonomy: A dictionary representing the taxonomy, which includes details about
                     mission planning contexts and hierarchical categorisation.
    :type taxonomy: dict
    :return: A string indicating the expanded and concatenated contexts related to the
             question prefix or a default value ("General") if no match is found.
    :rtype: str
    """
    q = question.lower().strip()

    for prefix, seeds in QUESTION_CONTEXT_SEEDS.items():
        if q.startswith(prefix):
            expanded = expand_contexts(taxonomy["MissionPlanningTaxonomy"], seeds, depth=1, normalise=True)
            return "+".join(expanded)

    return "General"


def camel_to_title(name: str) -> str:
    """
    Converts a camel case string to a title case string.

    This function splits a camel case string into separate words based on capital
    letters and returns the resulting string in title case format. It ensures that
    the words are separated by a single space.

    :param name: The camel case string to be converted.
    :type name: str
    :return: The converted title case string.
    :rtype: str
    """
    words_separated = re.sub(CAMEL_CASE_SPLIT_PATTERN, r'\1 \2', name)
    return words_separated.strip()


def log_suppression(word: str, is_protected_trigger: bool, context_snippet: str) -> None:
    """
    Logs suppression activity by appending details of the suppressed word, the triggering context,
    and a snippet of the surrounding context to the suppression log.

    Args:
        word: The suppressed word that is being logged.
        is_protected_trigger: Indicates whether the suppression was triggered by a protected term.
        context_snippet: A relevant snippet of the surrounding context of the suppressed word.
    """
    SUPPRESSION_LOG.append({
        "word": word,
        "protected_trigger": is_protected_trigger,
        "context_snippet": context_snippet[:MAX_CONTEXT_LENGTH]
    })


def _is_synonym_in_rule(synonym_group: str, rule: dict) -> bool:
    """
    Checks if a synonym group is present in the given rule.

    :param synonym_group: The group of synonyms to check
    :type synonym_group: str
    :param rule: A conflict rule containing synonyms and the protected terms
    :type rule: dict
    :return: True if synonym_group is in rule's synonyms, False otherwise
    :rtype: bool
    """
    return synonym_group in rule["synonyms"]


def _has_protected_term_in_text(text_lower: str, rule: dict) -> bool:
    """
    Checks if any protected term from the rule appears in the text.

    :param text_lower: The lowercased input text to search within
    :type text_lower: str
    :param rule: A conflict rule containing the protected terms
    :type rule: dict
    :return: True if any protected term is found in text, False otherwise
    :rtype: bool
    """
    return any(protected in text_lower for protected in rule["protected"])


def should_skip_synonym(text: str, synonym_group: str, conflict_rules: dict) -> bool:
    """
    Determines whether a synonym should be skipped based on conflict rules and
    protected terms. This function evaluates if the provided synonym group
    matches any of the rules and if any of the protected terms appear in the
    given text. If a match is found, the synonym will be skipped.

    :param text: The input string in which the synonym will be evaluated
        against the conflict rules.
    :type text: str
    :param synonym_group: The group of synonyms to check against the conflict
        rules.
    :type synonym_group: str
    :param conflict_rules: A dictionary containing conflict rules. Each rule
        includes a list of synonyms that relate to the conflict group and a
        list of protected terms that must not appear in the text for the synonym
        group to be valid.
    :type conflict_rules: dict
    :return: A boolean indicating whether the synonym group should be skipped
        (True) or not (False).
    :rtype: bool
    """
    text_lower = text.lower()

    for rule in conflict_rules.values():
        if _is_synonym_in_rule(synonym_group, rule):
            if _has_protected_term_in_text(text_lower, rule):
                return True

    return False


def _swap_adjacent_words(words: list[str], protected_terms: set[str]) -> list[str]:
    """
    Swaps two adjacent non-protected words at a random position.

    :param words: List of words to potentially swap.
    :param protected_terms: Set of terms that should not be swapped.
    :return: List of words with a potential swap applied.
    """
    augmented_words = words.copy()
    idx = random.randint(0, len(words) - 2)

    # Only swap if neither word is in protected terms
    if (augmented_words[idx].lower() not in protected_terms and
            augmented_words[idx + 1].lower() not in protected_terms):
        augmented_words[idx], augmented_words[idx + 1] = \
            augmented_words[idx + 1], augmented_words[idx]

    return augmented_words


def augment_text_minimal(text: str, aug_prob=0.1) -> str:
    """
    Augments the input text by occasionally swapping adjacent, non-technical words
    based on a specified augmentation probability. The augmentation excludes
    certain "protected" terms and operates only when the text contains enough words.

    :param text: The original text to augment.
    :type text: str
    :param aug_prob: The probability of applying text augmentation. Defaults to 0.1.
    :type aug_prob: float
    :return: The modified text with possible word swaps, or the same text if
             conditions for augmentation are not met.
    :rtype: str
    """
    words = text.split()
    word_count = len(words)

    if word_count < MIN_WORDS_FOR_AUGMENTATION or random.random() > aug_prob:
        return text

    augmented_words = _swap_adjacent_words(words, PROTECTED_TERMS)
    return ' '.join(augmented_words)


def augment_text_structural(text: str) -> str:
    """
    Transforms an input text by applying one of several predefined pattern-based transformations
    randomly. The transformation modifies specific phrases or structures within the text to
    produce variations.

    :param text: The input string to be augmented.
    :type text: str
    :return: The augmented string with one transformation applied, if a matching pattern is found.
    :rtype: str
    """
    augmented = text
    # Apply one random transformation if a pattern matches
    transformations = _TEXT_TRANSFORMATIONS.copy()
    random.shuffle(transformations)

    for pattern, replacement in transformations:
        if re.search(pattern, augmented, _REGEX_FLAGS):
            augmented = re.sub(pattern, replacement, augmented, count=1, flags=_REGEX_FLAGS)
            break

    return augmented


def _normalize_inputs(word: str, context: str) -> tuple[str, str]:
    """
    Normalise the input word and context strings by stripping any leading or trailing
    whitespace and converting them to lowercase.

    :param word: The input word to be normalised (str).
    :param context: The input context associated with the word to be normalised (str).
    :return: A tuple containing the normalised word and context (tuple[str, str]).
    """
    return word.strip().lower(), context.strip().lower()


def _validate_context(context: str, conflict_rules: dict) -> tuple[bool, Optional[dict]]:
    """
    Validates the given context against the provided conflict rules.

    This function checks if the specified context exists within the conflict rules.
    If the context is not found, it returns a validation failure status and a
    detailed description of the reason.

    :param context: The context to validate.
    :type context: str
    :param conflict_rules: Set or iterable containing valid context definitions.
    :type conflict_rules: set or iterable
    :return: A tuple containing a boolean indicating whether the context is valid
        and, if invalid, a dictionary containing failure details.
    :rtype: tuple[bool, Optional[dict]]
    """
    if context not in conflict_rules:
        return False, {
            "status": _STATUS_UNKNOWN_CONTEXT,
            "context": context,
            "reason": f"Context '{context}' not defined in conflict rules."
        }
    return True, None


def _extract_term_sets(rules: dict) -> tuple[set, set]:
    """
    Extracts and processes term sets from a dictionary of rules.

    This function retrieves "protected" and "synonyms" terms from the input rules.
    The terms are converted to lowercase and made into sets, which are then returned.

    :param rules: dict
        A dictionary of rules containing "protected" and "synonyms" keys.
    :return: tuple
        A tuple containing two sets: the first for "protected" terms and the second
        for "synonyms" terms.
    """
    protected = set(w.lower() for w in rules.get("protected", []))
    synonyms = set(w.lower() for w in rules.get("synonyms", []))
    return protected, synonyms


def _create_result(status: Any, context: Any, reason: Any) -> dict:
    """
    Creates a result dictionary using the provided status, context, and reason.

    This function generates a dictionary structure to encapsulate a status, the
    associated context, and the reason for the given status. The dictionary is
    useful in various scenarios, such as generating standardised API responses
    or structured data for further processing.

    :param status: The status value to include in the result.
    :type status: Any
    :param context: The context relevant to the status.
    :type context: Any
    :param reason: The rationale or explanation corresponding to the status.
    :type reason: Any
    :return: A dictionary containing the status, context, and reason.
    :rtype: dict
    """
    return {
        "status": status,
        "context": context,
        "reason": reason
    }


def _check_exact_matches(word: str, protected: set[str], synonyms: set[str], context: Any) -> dict | None:
    """
    Checks if a word has exact matches in protected terms or synonyms.

    This function examines a given word to determine if it exists in the list
    of protected terms or synonyms. If an exact match is found in protected terms,
    it identifies it as a protected term. If an exact match is found in the
    synonym list, it points out potential semantic conflicts. If no matches
    are found, it returns `None`.

    :param word: The word to check for exact matches.
    :type word: str
    :param protected: A set of strings representing protected terms.
    :type protected: set[str]
    :param synonyms: A set of strings representing synonyms.
    :type synonyms: set[str]
    :param context: Contextual information used in the result creation.
    :type context: Any
    :return: A result object indicating the match type and context, or `None`
        if no match is found.
    :rtype: dict | None
    """
    if word in protected:
        return _create_result(_STATUS_PROTECTED, context, "Exact match in protected terms.")

    if word in synonyms:
        return _create_result(
            _STATUS_SYNONYM_CONFLICT,
            context,
            "Matches synonym list; may conflict semantically."
        )

    return None


def _check_partial_matches(word: str, protected: set[str], synonyms: set[str], context: Any) -> dict | None:
    """
    Checks for partial matches between the given word and lists of protected and
    synonym tokens. Returns a result if any matches are found, with details about
    the type of conflict detected.

    :param word: The word to check for conflicts.
    :param protected: A list of tokens that are protected and should not appear in
                      the word.
    :param synonyms: A list of tokens considered as synonyms that should not
                     appear in the word.
    :param context: The context associated with this check (e.g. additional
                    metadata or data relevant to the specific operation being
                    performed).
    :return: A result indicating the type of match
             ('protected' or 'synonym_conflict') and the associated context
             if a conflict is found, or None if no conflicts exist.
    """
    for p in protected:
        if re.search(rf"\b{re.escape(p)}\b", word):
            return _create_result(
                _STATUS_PROTECTED,
                context,
                f"Contains protected token '{p}'."
            )

    for s in synonyms:
        if re.search(rf"\b{re.escape(s)}\b", word):
            return _create_result(
                _STATUS_SYNONYM_CONFLICT,
                context,
                f"Contains synonym token '{s}'."
            )

    return None


def check_conflict(word: str, context: str, conflict_rules: dict) -> dict:
    """
    Checks if a given word conflicts with a set of protected or synonym terms
    defined in conflict rules for a specific context. The function performs
    exact match and partial match checks for both protected and synonym terms,
    identifying conflicts or certifying the word as allowed.
    :param word:
        The word to be checked for conflicts.
    :type word: str
    :param context:
        The specific context in which the conflict rules are defined.
    :type context: str
    :param conflict_rules:
        A dictionary of conflict rules, where the key is the context and
        the value contains lists of 'protected' terms and 'synonyms'.
    :type conflict_rules: dict
    :return:
        A dictionary containing the status of the word in the given context,
        the evaluated context, and the reason for the status.
    :rtype: dict
    """
    # Normalise inputs
    word, context = _normalize_inputs(word, context)

    # Validate context
    is_valid, error_result = _validate_context(context, conflict_rules)
    if not is_valid:
        return error_result

    # Extract term sets
    rules = conflict_rules[context]
    protected, synonyms = _extract_term_sets(rules)

    # Check exact matches
    exact_match_result = _check_exact_matches(word, protected, synonyms, context)
    if exact_match_result:
        return exact_match_result

    # Check partial matches
    partial_match_result = _check_partial_matches(word, protected, synonyms, context)
    if partial_match_result:
        return partial_match_result

    # No conflicts found
    return _create_result(
        _STATUS_ALLOWED,
        context,
        "No match found in protected or synonym lists."
    )


def _collect_conflicts(word: str, conflict_rules: dict) -> list:
    """
    Collects all conflicts for a word across all contexts defined in conflict rules.

    :param word: str
        The word to check for conflicts.
    :param conflict_rules: dict
        A dictionary where keys represent contexts, and values define the conflict rules.
    :return: list
        A list of conflict results where status indicates a conflict.
    """
    conflicts = []
    for context in conflict_rules:
        result = check_conflict(word, context, conflict_rules)
        if result["status"] in _CONFLICT_STATUSES:
            conflicts.append(result)
    return conflicts


def check_conflict_all(word: str, conflict_rules: dict) -> list:
    """
    Checks whether the provided word conflicts with any given conflict rules in all contexts.
    This function iterates through all the contexts defined in the conflict rules and checks the word for
    potential conflicts by invoking the `check_conflict` function for each context. If the word conflicts
    in one or more contexts, the corresponding results are accumulated. If no conflicts are found in any
    context, the function returns a single result indicating that the word is allowed.
    :param word: str
        The word to check for conflicts.
    :param conflict_rules: dict
        A dictionary where keys represent contexts, and values define the conflict rules for each context.
    :return: list
        A list of dictionaries containing the conflict check results. Each dictionary describes the conflict
        `status`, the associated `context`, and the `reason`. If no conflicts are found, a default result
        indicating "allowed" is returned.
    """
    conflicts = _collect_conflicts(word, conflict_rules)
    return conflicts if conflicts else [_NO_CONFLICT_RESULT]


def _check_global_protection(term: str, term_lower: str, context: str,
                             validation_rules: ValidationRules) -> Optional[dict]:
    """
    Checks if a term is globally protected based on the provided validation rules
    and return protection details if applicable.

    A globally protected term is a term that must not be replaced under any
    circumstances. This method evaluates whether the given term is globally
    protected, logs appropriate information if it is, and provides details of the
    protection status.

    :param term: The original term being checked.
    :param term_lower: Lowercase version of the term for case-insensitive matching.
    :param context: Contextual information where the term appears.
    :param validation_rules: Validation rules contain the list of globally
                             protected terms.
    :return: A dictionary containing details of the protection status for the term
             if it is globally protected, or None if the term is not protected.
    :rtype: Optional[dict]
    """
    if term_lower in validation_rules.protected_terms:
        log_suppression(
            word=term,
            is_protected_trigger=True,
            context_snippet=f"Globally protected term in context '{context}'"
        )
        return {
            "term": term,
            "context": context,
            "status": _STATUS_PROTECTED,
            "details": "Globally protected technical term – must not be replaced."
        }
    return None


def _check_context_rules(term: str, term_lower: str, ctx: str,
                         rules: dict) -> Optional[tuple[str, str]]:
    """
    Checks the given term against specific context-based rules to determine if it is
    protected or its usage could lead to a synonym conflict.

    The function evaluates the term within the context by comparing it to the lists
    of protected terms or synonyms specified in the rules. If matched with a
    protected term, it logs this trigger and returns the protected status alongside
    an explanatory message. Similarly, if the term conflicts with a synonym, it logs
    this information and returns a conflict status.

    :param term: The original term being checked in the context.
    :param term_lower: The lowercased version of the term for comparison purposes.
    :param ctx: The context in which the term is being evaluated.
    :param rules: A dictionary of context rules with possible keys such as "protected"
        (a list of terms protected in the context) and "synonyms" (a list of terms
        that could conflict in the same context).
    :return: A tuple containing a status flag and explanatory message if the term matches
        a rule (protected or synonym conflict), or None if no rules apply.
    """
    protected = [p.lower() for p in rules.get("protected", [])]
    synonyms = [s.lower() for s in rules.get("synonyms", [])]

    if term_lower in protected:
        log_suppression(
            word=term,
            is_protected_trigger=True,
            context_snippet=f"Context-protected term in '{ctx}'"
        )
        return _STATUS_PROTECTED, f"Protected within context '{ctx}'."
    elif term_lower in synonyms:
        log_suppression(
            word=term,
            is_protected_trigger=False,
            context_snippet=f"Synonym conflict detected in context '{ctx}'"
        )
        return _STATUS_CONFLICT, f"Potential synonym conflict in context '{ctx}'."
    return None


def _check_cross_context_conflicts(term: str, term_lower: str, current_ctx: str,
                                   validation_rules: ValidationRules) -> Optional[tuple[str, str]]:
    """
    Checks for cross-context conflicts between the current context and other contexts
    within the provided validation rules. A term is considered in conflict if it exists
    as a protected word in other contexts.

    :param term: The term being evaluated for cross-context conflicts.
    :type term: str
    :param term_lower: The lowercase version of the term, used for case-insensitive comparisons.
    :type term_lower: str
    :param current_ctx: The current context in which the term is being validated.
    :type current_ctx: str
    :param validation_rules: The validation rules contain potential conflict definitions.
    :type validation_rules: ValidationRules
    :return: A tuple containing the conflict status and a description of the conflict,
        or None if no conflicts are detected.
    :rtype: Optional[tuple[str, str]]
    """
    cross_conflicts = []
    for other_ctx, other_rules in validation_rules.conflict_rules.items():
        if other_ctx == current_ctx:
            continue
        if term_lower in [p.lower() for p in other_rules.get("protected", [])]:
            cross_conflicts.append(other_ctx)

    if cross_conflicts:
        log_suppression(
            word=term,
            is_protected_trigger=True,
            context_snippet=f"Cross-context conflict: protected in {', '.join(cross_conflicts)}"
        )
        return (_STATUS_CROSS_CONFLICT,
                f"Protected in other context(s): {', '.join(cross_conflicts)}")
    return None


def _check_domain_synonyms(term: str, term_lower: str,
                           validation_rules: ValidationRules) -> list[tuple[str, str]]:
    """
    Checks if the given term overlaps with protected domain terms based on the
    provided validation rules and returns a list of conflicts if found. This
    function evaluates domain synonyms and protected terms to identify any
    potential conflicts.

    :param term: The original term that needs to be validated.
    :param term_lower: Lowercase version of the term, used for case-insensitive
        comparison.
    :param validation_rules: A set of validation rules containing domain
        synonyms and protected terms.
    :return: A list of tuples where each tuple contains the conflict status and
        a descriptive message regarding the conflict.
    """
    results = []
    for key, syns in validation_rules.domain_synonyms.items():
        all_syns = [s.lower() for s in syns] + [key.lower()]
        if term_lower in all_syns and key.lower() in validation_rules.protected_terms:
            results.append((_STATUS_CONFLICT,
                            f"'{term}' overlaps with protected domain term '{key}'."))
            log_suppression(
                word=term,
                is_protected_trigger=True,
                context_snippet=f"Domain synonym overlap with protected term '{key}'"
            )
    return results


def _determine_final_status(all_results: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """
    Determines the final status and details based on the provided results.

    This function evaluates a collection of status-message pairs, selects the highest
    priority status according to predefined priorities, and collects all the associated
    messages for the final details. If no results are provided, it defaults to a base
    status with a specific message.

    :param all_results: A list of tuples where each tuple consists of a status as a
        string and a corresponding message as a string.
    :return: A tuple containing the final status as a string and a list of strings
        representing the aggregated details.
    """
    if not all_results:
        return _STATUS_VALID, ["No conflict detected."]

    final_status = max(all_results, key=lambda x: _STATUS_PRIORITIES[x[0]])[0]
    final_details = [msg for _, msg in all_results]
    return final_status, final_details


def validate_term(term: str,
                  context: str,
                  validation_rules: ValidationRules,
                  strict_mode: bool = False) -> dict:
    """
    Validates a term against a set of contextual and domain-specific validation rules.

    This function processes a term through a series of checks, such as global
    validations, context-specific rules, cross-context conflicts (if strict mode
    is enabled), and domain-level synonym conflicts. It aggregates the results
    of these checks to generate a final status and additional details about the
    validation process.

    :param term: The term to be validated.
    :type term: str
    :param context: The context(s) in which the term should be validated.
    :type context: str
    :param validation_rules: An instance of ValidationRules containing the
        rules and configurations for validations.
    :type validation_rules: ValidationRules
    :param strict_mode: A flag indicating if strict mode checks such as cross-context conflicts should be applied. Defaults to False.
    :type strict_mode: bool
    :return: A dictionary containing the validated term, its context(s), the
        final validation status, and additional details.
    :rtype: dict
    """
    term_lower = term.lower().strip()
    contexts = [c.strip().lower() for c in context.split("+")]

    # Check global protection
    global_result = _check_global_protection(term, term_lower, context, validation_rules)
    if global_result:
        return global_result

    all_results = []

    # Check each context individually
    for ctx in contexts:
        if ctx not in validation_rules.conflict_rules:
            continue

        rules = validation_rules.conflict_rules[ctx]

        # Check context-specific rules
        context_result = _check_context_rules(term, term_lower, ctx, rules)
        if context_result:
            all_results.append(context_result)

        # Check cross-context conflicts if strict mode enabled
        if strict_mode:
            cross_result = _check_cross_context_conflicts(term, term_lower, ctx, validation_rules)
            if cross_result:
                all_results.append(cross_result)

    # Check domain-level synonym conflicts
    domain_results = _check_domain_synonyms(term, term_lower, validation_rules)
    all_results.extend(domain_results)

    # Determine final status
    final_status, final_details = _determine_final_status(all_results)

    return {
        "term": term,
        "context": context,
        "status": final_status,
        "details": "; ".join(final_details)
    }


def _create_result_entry(validation_result: dict) -> dict:
    """
    Creates a standardised result entry from a validation result.

    :param validation_result: The raw validation results from validate_term.
    :type validation_result: dict
    :return: A dictionary containing term, status, and details.
    :rtype: dict
    """
    return {
        "term": validation_result["term"],
        "status": validation_result["status"],
        "details": validation_result["details"]
    }


def _update_summary(summary: dict, status: str) -> None:
    """
    Updates the validation summary with the given status count.

    :param summary: The summary dictionary to update.
    :type summary: dict
    :param status: The validation status to increment.
    :type status: str
    """
    summary[status] = summary.get(status, 0) + 1


def batch_validate(terms: list,
                   context: Any,
                   validation_rules: ValidationRules,
                   strict_mode: Boolean = False) -> dict:
    """
    Validates a batch of terms against the provided validation rules within a given context.

    This function processes a list of terms, validates each term using the provided context and
    validation rules, and returns the results of the validation along with a summary of statuses.
    The validation can be strict or lenient depending on the strict_mode parameter.

    :param terms: A list of terms to be validated.
    :type terms: list
    :param context: The contextual information required for validation.
    :type context: Any
    :param validation_rules: The set of rules used for validating terms.
    :type validation_rules: ValidationRules
    :param strict_mode: Flag indicating whether the validation should be strict. Defaults to False.
    :type strict_mode: bool
    :return: A dictionary containing the context, validation results for each term, and a summary of statuses.
    :rtype: dict
    """
    results = []
    summary = {_STATUS_VALID: 0, _STATUS_PROTECTED: 0, _STATUS_CONFLICT: 0, _STATUS_CROSS_CONFLICT: 0}

    for term in terms:
        validation_result = validate_term(term, context, validation_rules, strict_mode)
        result_entry = _create_result_entry(validation_result)
        results.append(result_entry)
        _update_summary(summary, validation_result["status"])

    return {
        "context": context,
        "results": results,
        "summary": summary
    }


def _is_protected_term(word_norm: str, protected_terms: set[str]) -> bool:
    """
    Determines whether a given word matches any of the protected terms.

    The function checks if the normalised word matches exactly or contains any
    of the provided protected terms. Protected terms are specific words or
    phrases that must be identified or safeguarded in some context.

    :param word_norm: A string representing the normalised version of the word
        to be checked.
    :param protected_terms: A set of strings representing the protected terms
        against which the `word_norm` is checked.
    :return: A boolean indicating whether the word matches or contains any of
        the protected terms.
    """
    return any(
        re.fullmatch(rf"\b{re.escape(term)}\b", word_norm) or term in word_norm
        for term in protected_terms
    )


def _try_direct_lookup(word_norm: str, domain_synonyms: dict[str, list[str]]) -> list[str]:
    """
    Attempts to directly look up synonyms for a normalised word in the provided domain-specific
    synonym dictionary. The function retrieves the list of synonyms corresponding to the input-normalised word if it
    exists.

    :param word_norm: The normalised word for which synonyms need to be found.
    :type word_norm: str
    :param domain_synonyms: A dictionary mapping normalised words to lists of their synonyms.
    :type domain_synonyms: dict[str, list[str]]
    :return: A list of synonyms found for the given normalised word, or an empty list if the word
        does not exist in the domain synonym dictionary.
    :rtype: list[str]
    """
    return domain_synonyms.get(word_norm, [])


def _try_reverse_lookup(word_norm: str, domain_synonyms: dict[str, list[str]]) -> list[str]:
    """
    Attempts to perform a reverse lookup for a given normalised word within a
    dictionary of domain-specific synonyms. For each entry in the dictionary,
    if the normalised word is found within the normalised synonyms of a domain,
    returns a list with the domain key as the first element followed by all
    matching synonyms except the one that corresponds to the input word.

    :param word_norm: Normalised word to search for within the synonyms.
    :param domain_synonyms: Dictionary mapping domain keys to a list of associated
        synonyms.
    :return: A list containing the domain key as the first element and all
        synonyms, excluding the one matching the input-normalised word. If no
        match is found, returns an empty list.
    """
    for key, syns in domain_synonyms.items():
        normalized_synonyms = [s.lower() for s in syns]
        if word_norm in normalized_synonyms:
            return [key] + [s for s in syns if s.lower() != word_norm]
    return []


def _try_plural_singular_lookup(word_norm: str, domain_synonyms: dict[str, list[str]]) -> list[str]:
    """
    Attempts to find synonyms for the given word by either removing or adding 's'
    to handle singular and plural forms. If the word ends with 's', it checks for
    a singular form by removing the 's'. If this singular form exists in the
    provided domain synonyms, the corresponding synonyms are returned. If not, it
    appends an 's' to the word to generate a plural form and checks for synonyms.

    :param word_norm: The normalised word for which synonyms need to be
                      identified by accounting for singular and plural forms.
    :param domain_synonyms: A dictionary where keys are words and values are
                            lists of synonyms for those words.
    :return: A list of synonyms for the processed version (singular or plural)
             of the input word. An empty list is returned if no synonyms are
             found for either transformation.
    """
    # Try removing 's' for plural -> singular
    if word_norm.endswith('s'):
        singular_form = word_norm[:-1]
        if singular_form in domain_synonyms:
            return domain_synonyms[singular_form]

    # Try adding 's' for singular -> plural
    plural_form = f"{word_norm}s"
    if plural_form in domain_synonyms:
        return domain_synonyms[plural_form]

    return []


def _try_fuzzy_lookup(word_norm: str, domain_synonyms: dict[str, list[str]]) -> list[str]:
    """
    Attempts to perform a fuzzy lookup of a normalised word within a dictionary
    of domain-specific synonyms. The function compares the given word to each
    key in the dictionary and checks for similarity based on predetermined
    constraints on character and length differences. If a suitable match is found,
    the associated synonyms are returned.

    :param word_norm: A normalised string word to compare against domain-specific
                      synonym keys.
    :param domain_synonyms: A dictionary where keys are strings representing the
                            domain-specific terms and values are lists of strings
                            containing synonyms for those terms.
    :return: A list of strings containing synonyms for the matching domain-specific
             term. If no match is found, an empty list is returned.
    """
    max_length_diff = 2
    max_char_diff = 2

    for key in domain_synonyms.keys():
        length_diff = abs(len(key) - len(word_norm))
        if length_diff > max_length_diff:
            continue

        char_differences = sum(a != b for a, b in zip(key, word_norm))
        if char_differences <= max_char_diff:
            return domain_synonyms[key]

    return []


def get_domain_synonyms(word: str, domain_synonyms: dict[str, list[str]], protected_terms: set[str]) -> list[str]:
    """
    Retrieves domain-specific synonyms for a given word using a variety
    of lookup methods, including direct and reverse mappings, plural/
    singular normalisation, and fuzzy matching. Additionally, ensures
    that certain protected terms are not processed for synonym lookups.

    :param word: The input word for which domain synonyms are
        to be retrieved.
    :param domain_synonyms: A dictionary where keys are domain-specific
        terms and values are lists of their respective synonyms.
    :param protected_terms: A set of terms that are protected and
        should not trigger synonym lookups.
    :return: A list of synonyms for the input word, or an empty list
        if no matches are found or the term is considered protected.
    """
    word_norm = word.strip().lower()

    # Check if the word is protected
    if _is_protected_term(word_norm, protected_terms):
        log_suppression(
            word=word,
            is_protected_trigger=True,
            context_snippet=f"Protected term blocked synonym lookup for '{word}'"
        )
        return []

    # Try direct lookup
    result = _try_direct_lookup(word_norm, domain_synonyms)
    if result:
        return result

    # Try reverse lookup
    result = _try_reverse_lookup(word_norm, domain_synonyms)
    if result:
        return result

    # Try plural/singular normalization
    result = _try_plural_singular_lookup(word_norm, domain_synonyms)
    if result:
        return result

    # Try fuzzy matching as a last resort
    result = _try_fuzzy_lookup(word_norm, domain_synonyms)
    if result:
        return result

    # No match found
    return []


def _normalize_and_expand_acronyms(text: str) -> str:
    """
    Normalizes the input text by converting it to lowercase and replacing acronyms
    based on predefined patterns and expansions. The method uses regex to identify
    acronyms and substitutes them with their corresponding expansions.

    :param text: The input string to be normalized and expanded.
    :type text: str
    :return: The normalized and expanded text.
    :rtype: str
    """
    normalized = text.lower()
    for acronym_pattern, expansion in ACRONYM_EXPANSIONS.items():
        normalized = re.sub(acronym_pattern, expansion, normalized)
    return normalized


def _remove_punctuation(text: str) -> str:
    """
    Removes all punctuation from the input text.

    This function takes a string as input and returns a new string with all
    punctuation characters removed. Punctuation characters are matched using
    a predefined regular expression pattern.

    :param text: A string from which punctuation will be removed.
    :type text: str
    :return: A string with all punctuation characters removed.
    :rtype: str
    """
    return re.sub(PUNCTUATION_PATTERN, '', text)


def _filter_stopwords(text: str) -> str:
    """
    Filters out stopwords from the given text using a predefined list of English
    stopwords. The function splits the input text into tokens, removes the
    stopwords, and then joins the filtered tokens back into a single string.

    :param text:
        A string containing the input text from which stopwords are to be removed.
    :return:
        A string containing the input text with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)


def preprocess(text: str) -> str:
    """
    Processes input text by normalizing, expanding acronyms, removing punctuation,
    and filtering out stopwords.

    The function applies multiple preprocessing steps on the provided text
    to prepare it for further analysis or processing tasks. Each step ensures the
    input text is cleaned and standardized.

    :param text: The input text to preprocess
    :type text: str
    :return: The processed text after normalization, punctuation removal, and
        stopword filtering
    :rtype: str
    """
    text = _normalize_and_expand_acronyms(text)
    text = _remove_punctuation(text)
    text = _filter_stopwords(text)
    return text


# Custom synonym augmentation using NLTK WordNet
def _normalize_lemma_name(lemma_name: str) -> str:
    """
    Normalizes the input lemma name by replacing underscores with spaces.

    :param lemma_name: The lemma name to be normalized.
    :type lemma_name: str
    :return: A normalized lemma name with underscores replaced by spaces.
    :rtype: str
    """
    return lemma_name.replace('_', ' ')


def _should_exclude_synonym(synonym: str, original_word: str) -> bool:
    """
    Determines if a synonym should be excluded based on its equivalence to the
    original word, ignoring case.

    :param synonym: The synonym to be evaluated.
    :type synonym: str
    :param original_word: The original word to compare against.
    :type original_word: str
    :return: True if the synonym matches the original word (case insensitive),
        otherwise False.
    :rtype: bool
    """
    return synonym.lower() == original_word.lower()


def _extract_lemma_synonyms(synset, original_word: str) -> set[str]:
    """
    Extracts synonyms (lemmas) from a given synset while excluding those that match
    certain criteria. The synonyms are normalized before being included in the
    returned set.

    :param synset: The synset from which to extract and normalize synonyms.
    :type synset: Synset
    :param original_word: The word used as a reference to exclude certain synonyms.
    :type original_word: str
    :return: A set of normalized synonyms from the synset, excluding inappropriate
        ones based on the original word.
    :rtype: set[str]
    """
    return {
        normalized_synonym
        for lemma in synset.lemmas()
        if not _should_exclude_synonym(
            normalized_synonym := _normalize_lemma_name(lemma.name()),
            original_word
        )
    }


def get_synonyms(word: str) -> list[str]:
    """
    Get a list of synonyms for the specified word.
    This function fetches all synonyms for a given word by querying its synsets
    using the WordNet lexical database. It processes lemmas from the synsets and
    returns a unique, sorted list of synonyms.
    :param word: The input word for which synonyms need to be generated.
    :type word: str
    :return: A sorted list of unique synonyms for the given word.
    :rtype: list[str]
    """
    all_synonyms = {
        synonym
        for synset in wordnet.synsets(word)
        for synonym in _extract_lemma_synonyms(synset, word)
    }
    return sorted(all_synonyms)


def _calculate_replacement_count(word_count: int, aug_prob: float, max_synonyms: int) -> int:
    """
    Calculate the number of word replacements based on total word count, augmentation
    probability, and the maximum allowable synonyms.
    :param word_count: The total number of words in the input text.
    :type word_count: int
    :param aug_prob: The probability of augmentation used to determine the proportion
        of words to be replaced.
    :type aug_prob: float
    :param max_synonyms: The maximum number of synonyms available for replacement.
    :type max_synonyms: int
    :return: The calculated number of replacements to perform.
    :rtype: int
    """
    desired_replacements = max(1, int(word_count * aug_prob))
    capped_replacements = min(max_synonyms, desired_replacements)
    return capped_replacements


def _select_random_indices(word_count: int, num_to_replace: int) -> list[int]:
    """
    Select a random set of indices based on the provided word count and number of replacements.
    This function ensures that the number of replaced indices will not exceed
    either the given word count or the specified number of replacements, whichever is smaller.
    :param word_count: Total number of words to choose indices from.
    :type word_count: int
    :param num_to_replace: Number of indices to select for replacement.
    :type num_to_replace: int
    :return: A list containing randomly selected unique indices for replacement.
    :rtype: list[int]
    """
    max_possible_replacements = min(num_to_replace, word_count)
    return random.sample(range(word_count), max_possible_replacements)


def _get_synonym_for_word(word: str) -> str | None:
    """
    Retrieve a random synonym for the given word.

    Retrieves synonyms based on whether domain-specific mode is enabled.
    If no synonyms are found, logs a suppression event and returns None.

    :param word: The word to find a synonym for.
    :return: A randomly selected synonym, or None if no synonyms are available.
    """
    if DOMAIN_SPECIFIC:
        synonyms = get_domain_synonyms(word, DOMAIN_SYNONYMS, PROTECTED_TERMS)
    else:
        synonyms = get_synonyms(word)

    if synonyms:
        return random.choice(synonyms)
    else:
        log_suppression(
            word=word,
            is_protected_trigger=False,
            context_snippet=f"No synonyms available for augmentation of '{word}'"
        )
        return None


def _replace_words_with_synonyms(words: list[str], indices: list[int]) -> list[str]:
    """
    Replace specified words in a list with their synonyms.

    This function performs word augmentation by replacing specific words in a
    provided list with random synonyms. The words to be replaced are specified
    by their indices in the list. If domain-specific synonyms are enabled, the
    function will prioritize retrieving synonyms from the predefined domain-specific
    synonyms set; otherwise, general synonyms will be retrieved. When no synonyms
    are found, the original word is retained, and an event with relevant context
    is logged.

    :param words: A list of words to process.
    :param indices: A list of indices specifying which words in the list to replace.
    :return: A new list of words with specified words replaced by synonyms.
    """
    augmented_words = words.copy()
    for idx in indices:
        synonym = _get_synonym_for_word(words[idx])
        if synonym:
            augmented_words[idx] = synonym
    return augmented_words


def augment_text(text: str, aug_prob=0.3, max_synonyms=2, structural_aug_prob=0.3) -> str:
    """
    Augments the given text by replacing words with synonyms based on specified
    augmentation probability and maximum synonyms. Optionally applies minimal
    structural variations to the augmented text.

    :param text: The input text to be augmented.
    :type text: str
    :param aug_prob: The probability of replacing a word with its synonym. Defaults to 0.3.
    :type aug_prob: float
    :param max_synonyms: The maximum number of synonyms to consider for replacement. Defaults to 2.
    :type max_synonyms: int
    :param structural_aug_prob: The probability of applying structural variations. Defaults to 0.3.
    :type structural_aug_prob: float
    :return: The augmented text with selected words replaced by synonyms
        and optional structural variations applied.
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
    if random.random() < structural_aug_prob:
        augmented_text = augment_text_structural(augmented_text)

    return augmented_text


# --- Simple keyword-based classification ---
# Since transformers may also have dependency issues, implementing a keyword-based classifier
def _count_keyword_matches(text_lower: str, keywords: list[str]) -> int:
    """
    Count the number of keywords that appear in the given text.

    This function performs simple substring matching for each keyword in the text.
    Each keyword is counted at most once, regardless of how many times it appears
    in the text.

    :param text_lower: The text to search for keywords, expected in lowercase
        for case-insensitive matching.
    :type text_lower: str
    :param keywords: A list of keywords to search for in the text.
    :type keywords: list[str]
    :return: The count of keywords found in the text.
    :rtype: int
    """
    return sum(1 for keyword in keywords if keyword in text_lower)


def _calculate_category_scores(text_lower: str, category_keywords: dict[str, list[str]]) -> dict[str, int]:
    """
    Calculate scores for each category based on the occurrences of keywords
    in the given text.

    This function analyzes the provided text and calculates a score for each
    category based on the number of times the associated keywords for that
    category are present in the text.

    :param text_lower: The text to analyze, expected in lowercase for
        case-insensitive keyword matching.
    :type text_lower: str
    :param category_keywords: A dictionary where keys represent categories
        (as strings) and values are lists of keywords for each category.
    :type category_keywords: dict[str, list[str]]
    :return: A dictionary where keys are category names and values are the
        calculated scores, representing keyword occurrences in the text.
    :rtype: dict[str, int]
    """
    scores = {}
    for category, keywords in category_keywords.items():
        scores[category] = _count_keyword_matches(text_lower, keywords)
    return scores


def _select_best_category(scores: dict[str, int]) -> str:
    """
    Select the best-scoring category from the scores dictionary.

    Returns 'uncategorised' if no category has a positive score,
    otherwise returns the category with the highest score.

    :param scores: Dictionary mapping category names to their scores
    :type scores: dict[str, int]
    :return: The name of the best-scoring category or 'uncategorised'
    :rtype: str
    """
    predicted_category = max(scores, key=scores.get, default='uncategorised')
    if scores.get(predicted_category, 0) == 0:
        predicted_category = 'uncategorised'
    return predicted_category


def _normalize_scores(scores: dict[str, int]) -> dict[str, float]:
    """
    Normalize scores so they sum to 1.0.

    If all scores are zero, they remain zero (dividing by 1).

    :param scores: Dictionary mapping category names to their raw scores
    :type scores: dict[str, int]
    :return: Dictionary with normalized scores (values sum to 1.0)
    :rtype: dict[str, float]
    """
    total_score = sum(scores.values()) if sum(scores.values()) > 0 else 1
    return {k: v / total_score for k, v in scores.items()}


def classify_response(text: str, taxonomy: dict) -> tuple[str, dict[str, float]]:
    """
    [DEPRECATED - Use classify_with_taxonomy for production code]

    Classify a given text into a category based on a taxonomy and return the
    predicted category along with the normalized scores for each category.

    This function provides a simpler, flat classification interface without
    hierarchical parent/child relationships. It is retained for:
    - Backward compatibility
    - Simple testing and prototyping
    - Baseline comparison with hierarchical classification

    For production use, prefer classify_with_taxonomy() which supports
    hierarchical classification with both child and parent categories.

    The function takes a text input and a taxonomy dictionary, builds category
    keywords from the taxonomy, calculates scores for how well the text matches
    each category, and finally selects the best matching category. If no
    categories score above zero, the function assigns 'uncategorised' as the
    default category. It also normalizes the scores for each category to sum to 1.

    :param text: The text input to be classified
    :type text: str
    :param taxonomy: A dictionary where keys represent category names,
        and the values are lists of keywords associated with each category
    :type taxonomy: dict
    :return: A tuple with the predicted category as a string and a dictionary
        of normalized scores for each category
    :rtype: tuple[str, dict[str, float]]

    Example:
        >>> taxonomy = {"Category1": {...}, "Category2": {...}}
        >>> category, scores = classify_response("sample text", taxonomy)
        >>> print(category)
        'Category1'
    """
    text_lower = text.lower()

    # Build category keywords dynamically from taxonomy
    category_keywords = build_category_keywords(taxonomy, lowercase=True, deduplicate=True)

    # Score categories
    scores = _calculate_category_scores(text_lower, category_keywords)

    # Pick best-scoring category
    predicted_category = _select_best_category(scores)

    # Normalize scores
    normalized_scores = _normalize_scores(scores)

    return predicted_category, normalized_scores


def load_preprocess_and_filter_responses(
        file_path: str,
        scenario_number: int,
        question_prefix: str
) -> pd.DataFrame:
    """
    Load, preprocess, and filter responses based on scenario and question criteria.

    This function performs three operations:
    1. Loads response data from a CSV file
    2. Preprocesses text by normalizing forward slashes to spaces
    3. Filters responses matching the specified scenario number and question prefix

    :param file_path: The path to the CSV file containing the response data.
    :type file_path: str
    :param scenario_number: The scenario number to filter responses by.
    :type scenario_number: int
    :param question_prefix: The prefix of the question to match when filtering.
    :type question_prefix: str
    :return: A pandas DataFrame containing filtered responses with columns
             `ResponseID` and `ResponseText`.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(file_path)

    # Normalize forward slashes in response text for consistent processing
    df['ResponseText'] = df['ResponseText'].str.replace(
        FORWARD_SLASH_REPLACEMENT_PATTERN,
        FORWARD_SLASH_REPLACEMENT,
        regex=True
    )

    # Filter by scenario and question criteria
    filtered_df = df[
        (df['ScenarioNumber'] == scenario_number) &
        (df['Question'].str.startswith(question_prefix))
        ][['ResponseID', 'ResponseText']]

    return filtered_df


def _is_valid_response(response_text) -> bool:
    """
    Validates if a response text is a non-empty string.

    :param response_text: The response text to validate
    :return: True if the response is a valid non-empty string, False otherwise
    :rtype: bool
    """
    return isinstance(response_text, str) and response_text.strip() != ''


def preprocess_responses(response_series: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a DataFrame of response data by cleaning the 'ResponseText' column, retaining only
    valid, non-empty strings, and removing duplicate entries based on the cleaned text.

    :param response_series: A panda DataFrame containing the columns 'ResponseID' and
        'ResponseText', where 'ResponseText' represents the raw unprocessed text entries of responses.
    :type response_series: pd.DataFrame

    :return: A panda DataFrame containing two columns, 'ResponseID' and 'ResponseText', with
        cleaned response texts and duplicates removed based on the 'ResponseText' column.
    :rtype: pd.DataFrame
    """
    # Filter valid responses using a vectorized boolean mask
    valid_mask = response_series['ResponseText'].apply(_is_valid_response)
    filtered_df = response_series[valid_mask].copy()

    # Apply preprocessing to ResponseText column vectorized
    filtered_df['ResponseText'] = filtered_df['ResponseText'].apply(preprocess)

    # Remove duplicates and select required columns
    return filtered_df[['ResponseID', 'ResponseText']].drop_duplicates(
        subset=['ResponseText'],
        keep='first'
    )


def _create_augmented_responses(response_id: str, response_text: str,
                                aug_prob: float, max_synonyms: int,
                                use_minimal_augmentation: bool,
                                minimal_aug_prob: float) -> list:
    """
    Creates augmented versions of a single response.

    :param response_id: The identifier for the response.
    :type response_id: str
    :param response_text: The text content of the response.
    :type response_text: str
    :param aug_prob: Probability for synonym-based augmentation.
    :type aug_prob: float
    :param max_synonyms: Maximum number of synonyms for augmentation.
    :type max_synonyms: int
    :param use_minimal_augmentation: Whether to apply minimal word swapping.
    :type use_minimal_augmentation: bool
    :param minimal_aug_prob: Probability for minimal augmentation.
    :type minimal_aug_prob: float
    :return: List of response dictionaries (original and augmented).
    :rtype: list
    """
    responses = []

    # Add original response
    responses.append({
        'ResponseID': response_id,
        'ResponseText': response_text
    })

    # Add a synonym-based augmented version
    augmented_text = augment_text(response_text, aug_prob=aug_prob, max_synonyms=max_synonyms)
    responses.append({
        'ResponseID': response_id,
        'ResponseText': augmented_text
    })

    # Optionally add minimal augmentation version
    if use_minimal_augmentation:
        minimal_augmented_text = augment_text_minimal(response_text, aug_prob=minimal_aug_prob)
        if minimal_augmented_text != response_text:
            responses.append({
                'ResponseID': response_id,
                'ResponseText': minimal_augmented_text
            })

    return responses


def augment_response_dataset(responses: pd.DataFrame, aug_prob: float = 0.3,
                             max_synonyms: int = 2, sample_size: int = 500,
                             use_minimal_augmentation: bool = False,
                             minimal_aug_prob: float = 0.1) -> pd.DataFrame:
    """
    Augments a dataset of text responses by applying various text augmentation techniques
    such as synonym-based replacements and optional minimal word swapping augmentation.
    Both the original and augmented responses are included in the output. Additionally,
    the resultant dataset can be subsampled to a specified size.

    :param responses: Input DataFrame containing the responses to augment. It must include
        at least two columns: 'ResponseID' (identifiers for the responses) and 'ResponseText'
        (the actual response text to augment).
    :type responses: pandas.DataFrame
    :param aug_prob: Probability for applying synonym-based augmentation to each word in the
        responses.
    :type aug_prob: float
    :param max_synonyms: Maximum number of synonyms to consider during synonym-based augmentation.
    :type max_synonyms: int
    :param sample_size: Maximum size of the returned dataset after augmentation. If the augmented
        dataset is larger, a random sample of the specified size will be returned.
    :type sample_size: int
    :param use_minimal_augmentation: Flag that determines whether to apply a minimal word swapping
        augmentation technique in addition to synonym-based augmentation.
    :type use_minimal_augmentation: bool
    :param minimal_aug_prob: Probability for applying the minimal word swapping augmentation
        to each word in the responses. Only used if `use_minimal_augmentation` is True.
    :type minimal_aug_prob: float
    :return: A DataFrame containing the original and augmented responses. Each response is paired
        with its original 'ResponseID' for consistent association.
    :rtype: pandas.DataFrame
    """
    augmented_rows = []

    for _, row in responses.iterrows():
        response_entries = _create_augmented_responses(
            response_id=str(row['ResponseID']),
            response_text=str(row['ResponseText']),
            aug_prob=aug_prob,
            max_synonyms=max_synonyms,
            use_minimal_augmentation=use_minimal_augmentation,
            minimal_aug_prob=minimal_aug_prob
        )
        augmented_rows.extend(response_entries)

    augmented_df = pd.DataFrame(augmented_rows)

    if len(augmented_df) > sample_size:
        augmented_df = augmented_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return augmented_df


def _normalize_keywords(keywords: list[str]) -> list[str]:
    """
    Normalizes keywords to lowercase for case-insensitive matching.

    :param keywords: List of keyword strings.
    :type keywords: list[str]
    :return: List of lowercase keyword strings.
    :rtype: list[str]
    """
    return [kw.lower() for kw in keywords]


def _calculate_keyword_score(text_lower: str, keywords: list[str]) -> int:
    """
    Calculates the number of keyword matches in the given text.

    :param text_lower: The lowercase text to search for keywords.
    :type text_lower: str
    :param keywords: List of keywords to match (should be lowercase).
    :type keywords: list[str]
    :return: Count of keyword matches found in the text.
    :rtype: int
    """
    return sum(1 for kw in keywords if kw in text_lower)


def classify_with_taxonomy(text: str, child_keywords: dict, parent_keywords: dict):
    """
    Analyzes a given text to classify it into predefined child and parent taxonomy categories based on the
    presence of keywords. It calculates scores for child-level and parent-level classifications and
    returns the results.

    :param text: The input text to be classified.
    :type text: str
    :param child_keywords: A dictionary where keys are child category names and values are dictionaries
        containing "keywords" (list of associated keywords) and "parent" (name of the parent category).
    :type child_keywords: dict
    :param parent_keywords: A dictionary where keys are parent category names and values are lists of
        associated keywords for the parent-level categories.
    :type parent_keywords: dict
    :return: A dictionary containing:
        - "child_scores": Scores for each child category based on keyword matches.
        - "parent_scores": Scores for each parent category based on keyword matches.
        - "predicted_child": The child category with the highest score or "uncategorised" if no match.
        - "predicted_parent": The parent category with the highest score or "uncategorised" if no match.
    :rtype: dict
    """
    text_lower = text.lower()

    # --- Child-level scores
    child_scores = {}
    parent_scores = defaultdict(int)

    for child, data in child_keywords.items():
        keywords = _normalize_keywords(data["keywords"])
        parent = data["parent"]
        score = _calculate_keyword_score(text_lower, keywords)
        if score > 0:
            child_scores[child] = score
            parent_scores[parent] += score

    # --- Parent-only scan (fallback if no child hits)
    if not child_scores:
        for parent, keywords in parent_keywords.items():
            normalized_keywords = _normalize_keywords(keywords)
            score = _calculate_keyword_score(text_lower, normalized_keywords)
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


def _create_classification_result(resp_id: str, text: str, classification: dict) -> dict:
    """
    Creates a structured dictionary containing classification results for a single response.

    :param resp_id: Identifier of the response
    :type resp_id: str
    :param text: Original response text
    :type text: str
    :param classification: Dictionary containing classification data with keys:
        'predicted_child', 'predicted_parent', 'child_scores', 'parent_scores'
    :type classification: dict
    :return: Formatted classification result dictionary
    :rtype: dict
    """
    return {
        'ResponseID': resp_id,
        'response': text,
        'PredictedChild': classification["predicted_child"],
        'PredictedParent': classification["predicted_parent"],
        'ChildScores': str(classification["child_scores"]),
        'ParentScores': str(classification["parent_scores"])
    }


def classify_responses(responses: pd.DataFrame, child_keywords: dict, parent_keywords: dict) -> list[dict]:
    """
    [DEPRECATED - No longer used in production workflow]

    This function has been superseded by direct iteration with classify_with_taxonomy()
    in the main() workflow. It is retained for:
    - Backward compatibility with legacy scripts
    - Reference implementation
    - Potential standalone usage scenarios

    For production use, iterate through your DataFrame and call classify_with_taxonomy()
    directly for each response, as shown in the main() function.

    Original documentation:
    ------------------------
    Classifies responses by applying a hybrid taxonomy-based classifier to determine
    child and parent categories for each response text.

    The function processes a given DataFrame of responses, where each response consists
    of an ID and text. Using the provided child and parent keywords dictionaries, it
    classifies each response under a predicted child and parent category. Additionally,
    it assigns scores for both child and parent classifications and outputs structured
    results.

    :param responses: A DataFrame containing the responses to classify. Each response
        should include a 'ResponseID' (identifier) and 'ResponseText' (text of the
        response).
    :type responses: pd.DataFrame
    :param child_keywords: A dictionary mapping child category keywords for classification.
    :type child_keywords: dict
    :param parent_keywords: A dictionary mapping parent category keywords for classification.
    :type parent_keywords: dict
    :return: A list of dictionaries where each dictionary contains the following keys:
        - 'ResponseID': Identifier of the response
        - 'response': Original response text
        - 'PredictedChild': Name of the predicted child category
        - 'PredictedParent': Name of the predicted parent category
        - 'ChildScores': String representation of child category scores
        - 'ParentScores': String representation of parent category scores
    :rtype: list[dict]
    """
    results = []
    for _, resp in responses.iterrows():
        resp_id = resp['ResponseID']
        text = resp['ResponseText']

        # Run hybrid taxonomy classifier
        classification = classify_with_taxonomy(text, child_keywords, parent_keywords)

        result = _create_classification_result(resp_id, text, classification)
        results.append(result)

    return results


def summarise_classifications(results: list[dict]) -> dict:
    """
    Summarizes classification results by counting occurrences of predicted child
    and parent categories. It takes a list of dictionaries where each dictionary
    represents a classification result with potential `PredictedChild` and
    `PredictedParent` keys. If these keys are missing in a classification result,
    the categories will default to "uncategorised". The function returns a summary
    containing counts for each unique child and parent category.

    :param results: A list of dictionaries where each dictionary represents a
        classification result. Each dictionary may contain keys `PredictedChild`
        and `PredictedParent` for predicted child and parent categories.
    :type results: list[dict]
    :return: A dictionary containing two summaries: `child_summary` which is a
        dictionary with counts of each unique predicted child category, and
        `parent_summary` which is a dictionary with counts of each unique predicted
        parent category.
    :rtype: dict
    """

    def _extract_category(row: dict, key: str, default: str = "uncategorised") -> str:
        """Extract category value from row with default fallback."""
        return row.get(key, default)

    child_categories = [_extract_category(row, "PredictedChild") for row in results]
    parent_categories = [_extract_category(row, "PredictedParent") for row in results]

    return {
        "child_summary": dict(Counter(child_categories)),
        "parent_summary": dict(Counter(parent_categories))
    }


def save_results(results: list[dict], output_path: str) -> pd.DataFrame:
    """
    Saves the classification results to a CSV file.

    :param results: The list of dictionaries containing classification results.
    :param output_path: The file path where the CSV file will be saved.
    :return: DataFrame containing the results.
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    return df


def display_results_summary(df: pd.DataFrame) -> None:
    """
    Displays a summary of the classification results, including the first few rows,
    total number of responses, and distributions of predicted child and parent categories.

    :param df: DataFrame containing classification results.
    :return: None
    """
    print(df.head())
    print(f"\nTotal responses classified: {len(df)}")
    print(f"\nCategory distribution:")
    print("\nPredicted Child distribution:")
    print(df['PredictedChild'].value_counts())
    print("\nPredicted Parent distribution:")
    print(df['PredictedParent'].value_counts())


def save_and_display_results(results: list[dict], output_path: str) -> None:
    """
    Saves the classification results to a CSV file and displays a summary of the
     data, including the first few rows, total number of responses, and distributions
    of predicted child and parent categories.

    :param results: The list of dictionaries containing classification results.
    :param output_path: The file path where the CSV file will be saved.
    :return: None
    """
    df = save_results(results, output_path)
    display_results_summary(df)


def prepare_taxonomy(taxonomy: dict) -> dict:
    """
    Prepares taxonomy by extracting inner taxonomy and humanising keys.

    :param taxonomy: Raw taxonomy dictionary
    :return:  with humanised category keywords
    """
    if "MissionPlanningTaxonomy" in taxonomy:
        return {humanise_key(k): v for k, v in taxonomy["MissionPlanningTaxonomy"].items()}
    return {humanise_key(k): v for k, v in taxonomy.items()}


def validate_responses(responses_aug: pd.DataFrame, question: str, taxonomy: dict,
                       validation_rules: ValidationRules, strict_mode: bool) -> pd.DataFrame:
    """
    Validates responses and enriches them with validation flags.

    :param responses_aug: Augmented responses DataFrame
    :param question: Target question string
    :param taxonomy: Taxonomy dictionary for context mapping
    :param validation_rules: Validation rules instance
    :param strict_mode: Whether to apply strict validation
    :return: DataFrame with validation flags
    """
    context = map_question_to_context(question, taxonomy)
    enriched_responses = []

    for _, row in responses_aug.iterrows():
        text = row['ResponseText']
        report = document_validate(
            text=str(text),
            context=context,
            validation_rules=validation_rules,
            strict_mode=strict_mode
        )
        flags = extract_validation_flags(report)
        enriched_responses.append({
            "ResponseID": row["ResponseID"],
            "ResponseText": text,
            **flags
        })

    return pd.DataFrame(enriched_responses)


def classify_validated_responses(validated_df: pd.DataFrame) -> list[dict]:
    """
    Classifies validated responses using taxonomy.

    :param validated_df: DataFrame with validated responses and flags
    :return: List of classification results with validation flags
    """
    parent_keywords, child_keywords = load_taxonomy("./docs/mission_planning_taxonomy.json")
    results = []

    for _, row in validated_df.iterrows():
        text = row['ResponseText']
        classification = classify_with_taxonomy(str(text), child_keywords, parent_keywords)
        results.append({
            "ResponseID": row["ResponseID"],
            "ResponseText": text,
            "PredictedChild": classification["predicted_child"],
            "PredictedParent": classification["predicted_parent"],
            "ChildScores": str(classification["child_scores"]),
            "ParentScores": str(classification["parent_scores"]),
            "has_protected": row["has_protected"],
            "has_conflict": row["has_conflict"],
            "has_cross_conflict": row["has_cross_conflict"],
        })

    return results


def save_classification_outputs(results: list[dict], scenario: int, question: str):
    """
    Saves all classification outputs including results, summaries, and JSON reports.

    :param results: List of classification results
    :param scenario: Scenario identifier
    :param question: Question identifier
    """
    # Save classification results
    classification_output_path = f"./output/S{scenario}_{question}_classification_results.csv"
    save_and_display_results(results, classification_output_path)

    # Save a summary report
    summary_output_path = f"./output/S{scenario}_{question}_classification_summary.csv"
    save_summary_report(results, summary_output_path)

    # Generate and save JSON summary
    summary = summarise_classifications(results)
    summary_path = f"./output/S{scenario}_{question}_classification_summary.json"
    with open(summary_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"\nSummary for Scenario {scenario}, {question}:")
    print("Child categories:", summary["child_summary"])
    print("Parent categories:", summary["parent_summary"])


def main(question: str, scenario: int, taxonomy: dict, validation_rules: ValidationRules,
         merge_to_source: bool = False, strict_mode: bool = True):
    """
    Performs classification workflow: loads data, validates responses, classifies them,
    and generates reports for a given question and scenario.

    :param question: A string representing the target question.
    :param scenario: An integer specifying the scenario identifier.
    :param taxonomy: A dictionary containing taxonomy definitions for classification.
    :param validation_rules: Instance of ValidationRules specifying the rules to validate responses.
    :param merge_to_source: A boolean flag indicating if the classifications should be saved back to the source data.
    :param strict_mode: A boolean flag indicating if strict validation rules should be applied.
    :return: None
    """
    # Prepare taxonomy
    category_keywords = prepare_taxonomy(taxonomy)

    # Load and filter data
    filtered_responses = load_preprocess_and_filter_responses(
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

    # Validate responses
    validated_df = validate_responses(responses_aug, question, taxonomy, validation_rules, strict_mode)

    # Save validation logs
    validation_output_path = f"./output/S{scenario}_{question}_validation_summary.json"
    validated_df.to_json(validation_output_path, orient="records", indent=2)
    print(f"Validation report saved: {validation_output_path}")

    # Classify validated responses
    results = classify_validated_responses(validated_df)

    # Save all outputs
    save_classification_outputs(results, scenario, question)

    # Optionally merge classifications to source
    if merge_to_source:
        print("\n" + "=" * 60)
        print("Merging ALL classifications back to source data...")
        print("=" * 60)
        merge_all_classifications()


def _has_validation_issue(summary: dict, key: str) -> bool:
    """
    Check if a validation issue exists for the given key.

    :param summary: Validation summary dictionary
    :param key: The validation key to check
    :return: True if the count for the key is greater than 0
    """
    return summary.get(key, 0) > 0


def extract_validation_flags(report: dict) -> dict:
    """
    Extract validation flags from the provided report.
    The function analyzes the `summary` sub-dictionary within the given report
    and determines the presence of specific validation flags such as
    `has_protected`, `has_conflict`, and `has_cross_conflict`. Additionally, it
    includes the full validation summary in the returned dictionary.
    :param report: The input dictionary contains a "summary" key with
        validation details.
    :type report: dict
    :return: A dictionary containing:
        - `has_protected` (bool): True if the "protected" count in the summary is
          greater than 0, otherwise False.
        - `has_conflict` (bool): True if the "conflict" count in the summary is
          greater than 0, otherwise False.
        - `has_cross_conflict` (bool): True if the "cross_conflict" count in
          the summary is greater than 0, otherwise False.
        - `validation_summary` (dict): The entire "summary" sub-dictionary from
          the input report, containing detailed validation metrics.
    :rtype: dict
    """
    summary = report.get("summary", {})
    return {
        "has_protected": _has_validation_issue(summary, _STATUS_PROTECTED),
        "has_conflict": _has_validation_issue(summary, _STATUS_CONFLICT),
        "has_cross_conflict": _has_validation_issue(summary, _STATUS_CROSS_CONFLICT),
        "validation_summary": summary
    }


def _aggregate_classifications_legacy(classifications_df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEPRECATED - Legacy single-column workflow]
    Aggregates classifications for each ResponseID by combining predicted categories
    and keeping only the first set of all_scores. The function processes the input
    DataFrame by grouping its rows by ResponseID, consolidating predicted categories
    into a single string with distinct values separated by '|', and retaining the
    first available all_scores for the group. The resulting DataFrame is reset to
    default indexing.

    :param classifications_df: A DataFrame containing classification results where
        'ResponseID' is the group identifier, 'predicted_category' contains predicted
        categories for each entry, and 'all_scores' holds the associated scores.
    :type classifications_df: pd.DataFrame

    :return: A DataFrame with aggregated classifications, containing columns
        'ResponseID', 'PredictedCategories', and 'all_scores', where categories are
        consolidated and scores represent the first available entry within each group.
    :rtype: pd.DataFrame
    """
    aggregated = classifications_df.groupby('ResponseID').agg({
        'predicted_category': lambda x: '|'.join(sorted(set(x))),
        'all_scores': 'first'
    }).reset_index()

    aggregated.rename(columns={'predicted_category': 'PredictedCategories'}, inplace=True)
    return aggregated


def _print_merge_statistics_legacy(merged_df: pd.DataFrame, output_path: str) -> None:
    """
    [DEPRECATED - Legacy single-column workflow]
    Prints statistics about the merge of predictions and their categories and displays
    relevant high-level details about the classified and unclassified responses based on
    the given DataFrame.

    :param merged_df: DataFrame containing the merged classification results. It is
        expected to have a column named 'PredictedCategories'.
    :type merged_df: pd.DataFrame
    :param output_path: File path where the merged classifications are saved.
    :type output_path: str
    :return: This function does not return any value.
    :rtype: None
    """
    classified_mask = merged_df['PredictedCategories'] != 'not_classified'
    multi_category_mask = merged_df['PredictedCategories'].str.contains('\|', na=False)

    print(f"\n{'=' * 60}")
    print(f"Merged classifications saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(merged_df)}")
    print(f"Classified responses: {classified_mask.sum()}")
    print(f"Unclassified responses: {(~classified_mask).sum()}")

    print(f"\nClassification patterns:")
    category_counts = merged_df['PredictedCategories'].value_counts().head(10)
    print(category_counts)

    print(f"\nResponses with multiple categories: {multi_category_mask.sum()}")


def merge_classifications_to_source(
        classification_results_path: str,
        original_data_path: str,
        output_path: str
) -> None:
    """
    [DEPRECATED - Use merge_all_classifications_multi_column() instead]
    Merges classification results with the original dataset to create a combined output that includes
    newly classified categories for responses. Handles multiple entries per response caused by
    augmentation and ensures only unique categories are aggregated. Outputs a file with the merged
    results and provides a summary of the classification outcomes.

    :param classification_results_path: Path to a CSV file containing the classification results.
                                         Must include 'ResponseID' and 'predicted_category' columns.
    :type classification_results_path: str
    :param original_data_path: Path to a CSV file containing the original responses dataset.
                               Must include a 'ResponseID' column.
    :type original_data_path: str
    :param output_path: Path where the merged output CSV file will be saved.
    :type output_path: str
    :return: None
    """
    # Load classification results
    classifications_df = pd.read_csv(classification_results_path)

    # Load original normalized data
    original_df = pd.read_csv(original_data_path)

    # Aggregate classifications to handle augmentation duplicates
    aggregated_classifications = _aggregate_classifications(classifications_df)

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

    # Print comprehensive statistics
    _print_merge_statistics(merged_df, output_path)


def _generate_scenario_paths(scenario: int, question: str) -> ScenarioPaths:
    """
    Generates file paths for classification processing of a scenario.

    :param scenario: The scenario identifier.
    :param question: The question identifier.
    :return: A ScenarioPaths named tuple containing all required paths.
    """
    return ScenarioPaths(
        classification_results=f"./output/S{scenario}_{question}_classification_results.csv",
        original_data='./input/normalised_all_responses.csv',
        output=f"./output/S{scenario}_{question}_normalised_classified_responses.csv"
    )


def _print_processing_header(scenario: int, question: str, separator_length: int = 60) -> None:
    """
    Prints a formatted header for scenario processing.

    :param scenario: The scenario identifier.
    :param question: The question identifier.
    :param separator_length: Length of the separator line.
    :return: None
    """
    separator = '=' * separator_length
    print(f"\n{separator}")
    print(f"Processing Scenario {scenario}, Question {question}")
    print(f"{separator}")


def batch_merge_all_scenarios(question: str, scenarios: list[int]) -> None:
    """
    [DEPRECATED - Use merge_all_classifications() instead]

    This function has been superseded by merge_all_classifications() which provides:
    - Multi-column classification output (PredictedChilds, PredictedParents)
    - Enhanced version control and change tracking
    - Single consolidated output file for all scenarios and questions

    This function is retained for backward compatibility only and may be removed
    in a future version.

    Original documentation:
    ------------------------
    Processes multiple scenarios by merging classification results with original
    data for each scenario and question.

    This function iterates through a list of integer scenario identifiers, and for
    each provided scenario, it prints a header message indicating the processing
    status. It attempts to merge corresponding classification results with original
    data using the `merge_classifications_to_source` function. If the classification
    file for a given scenario is not found, it logs a warning and continues to the
    next scenario.

    :param question: The identifier for the question being processed.
    :param scenarios: A list of integer scenario identifiers to process.
    :return: None
    """
    import warnings
    warnings.warn(
        "batch_merge_all_scenarios() is deprecated. Use merge_all_classifications() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    for scenario in scenarios:
        _print_processing_header(scenario, question)

        paths = _generate_scenario_paths(scenario, question)

        try:
            merge_classifications_to_source(
                classification_results_path=paths.classification_results,
                original_data_path=paths.original_data,
                output_path=paths.output
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not process Scenario {scenario} - {e}")
            continue


def _process_scenario_classification(
        master_df: pd.DataFrame,
        scenario: int,
        question: str
) -> tuple[pd.DataFrame, bool]:
    """
    Process and merge classification results for a single scenario.

    :param master_df: The master dataframe to merge classifications into
    :param scenario: The scenario number to process
    :param question: The question identifier
    :return: Tuple of (updated master dataframe, success flag)
    """
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
        updated_df = master_df.merge(
            aggregated,
            on='ResponseID',
            how='left'
        )

        # Fill NaN
        updated_df[column_name].fillna('', inplace=True)
        print(f"Added classifications for Scenario {scenario}")

        return updated_df, True

    except FileNotFoundError:
        print(f"Warning: Classification file not found for Scenario {scenario}")
        return master_df, False


def create_master_classified_file(question: str, scenarios: list[int]) -> None:
    """
    Creates and saves a master classified file by merging and aggregating
    classification results for different scenarios into a single dataset.
    This function processes responses classified for each scenario, aggregates
    unique predicted categories per response, and appends them to a master dataset.
    The output file contains all classified responses merged with their respective
    categories for each scenario.
    :param question: The name of the question or topic for which the classifications are performed.
    :type question: str
    :param scenarios: A list of scenario numbers. Each scenario corresponds to a
                      classification file to be loaded and merged with the master file.
    :type scenarios: list[int]
    :return: This function does not return any value. The resulting dataset is written
             to a CSV file.
    :rtype: None
    """
    # Load original data
    master_df = pd.read_csv('./input/normalised_all_responses.csv')

    # Merge classifications from each scenario
    successfully_processed_scenarios = []
    for scenario in scenarios:
        master_df, success = _process_scenario_classification(master_df, scenario, question)
        if success:
            successfully_processed_scenarios.append(scenario)

    # Save the master file
    output_path = './output/normalised_all_classified_responses.csv'
    master_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Master classified file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nColumns added:")
    for scenario in successfully_processed_scenarios:
        col_name = f'S{scenario}_{question}_Categories'
        if col_name in master_df.columns:
            print(f"  - {col_name}")
    print(f"\nTotal responses: {len(master_df)}")


def _initialize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initializes missing columns in a pandas DataFrame with default values.

    This function ensures that the given DataFrame has specific required columns
    ('Classification', 'Version', 'ChangeNote', 'ChangeDate'). If any of these
    columns are missing, they are added to the DataFrame with their respective
    default values.

    :param df: The pandas DataFrame to initialize with default columns.
    :type df: pd.DataFrame
    :return: The updated DataFrame with all required columns ensured.
    :rtype: pd.DataFrame
    """
    if 'Classification' not in df.columns:
        df['Classification'] = 'not_classified'
    if 'Version' not in df.columns:
        df['Version'] = DEFAULT_VERSION
    if 'ChangeNote' not in df.columns:
        df['ChangeNote'] = INITIAL_MERGE_NOTE
    if 'ChangeDate' not in df.columns:
        df['ChangeDate'] = ''
    return df


def _combine_classifications(
        classifications_df: pd.DataFrame,
        combine_child_parent: bool
) -> pd.DataFrame:
    """
    Combines classification labels derived from predicted child and parent categories. This function
    either concatenates the child and parent classifications into a single string, or selectively assigns
    the child classification unless it is 'uncategorised', in which case the parent classification is used.

    :param classifications_df: A pandas DataFrame containing columns `PredictedChild` and
        `PredictedParent` that represent the predicted classification labels for child and parent
        categories respectively.
    :type classifications_df: pd.DataFrame
    :param combine_child_parent: A boolean flag that indicates whether to concatenate the child and
        parent classifications into a single combined label. If False, selectively combines either
        `PredictedChild` or `PredictedParent`.
    :type combine_child_parent: bool
    :return: A pandas DataFrame with an additional column `ClassificationCombined` containing
        the combined classification labels.
    :rtype: pd.DataFrame
    """
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
    return classifications_df


def _aggregate_classifications(classifications_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates classification data by grouping on unique response IDs and combining
    classifications into a single string separated by a pipe ('|') character. Ensures
    duplicate classifications are removed, and the aggregated classifications are
    sorted alphabetically.

    :param classifications_df: A Pandas DataFrame containing classification data. It must
        have at least the following columns:
        - 'ResponseID': Unique identifiers for responses.
        - 'ClassificationCombined': Classifications associated with each response.

    :return: A Pandas DataFrame grouped by 'ResponseID', with classification data
        combined and deduplicated.
    """
    return classifications_df.groupby('ResponseID').agg({
        'ClassificationCombined': lambda x: '|'.join(sorted(set(x)))
    }).reset_index()


def _update_original_dataframe(
        original_df: pd.DataFrame,
        aggregated: pd.DataFrame,
        file_mod_date: str
) -> None:
    """
    Updates the original dataframe with classification and related metadata from
    the aggregated dataframe. This process involves identifying matching response
    entries in the original dataframe and updating specific columns for those
    entries.

    :param original_df: The original pandas DataFrame containing response data
                        that will be updated based on the aggregated DataFrame.
    :param aggregated: The aggregated pandas DataFrame containing classification
                       data and metadata that will be used to update the original
                       DataFrame.
    :param file_mod_date: A string indicating the modification date that will be
                          applied to the 'ChangeDate' column during updates.
    :return: None. The original_df is updated in place.
    """
    for _, row in aggregated.iterrows():
        response_id = row['ResponseID']
        category = row['ClassificationCombined']
        mask = original_df['ResponseID'] == response_id

        original_df.loc[mask, 'Classification'] = category
        if category not in UNCLASSIFIED_VALUES:
            original_df.loc[mask, 'Version'] = UPDATED_VERSION
            original_df.loc[mask, 'ChangeNote'] = INITIAL_CLASSIFICATION_NOTE
            original_df.loc[mask, 'ChangeDate'] = file_mod_date


def _process_classification_file(
        scenario: int,
        question: str,
        original_df: pd.DataFrame,
        combine_child_parent: bool
) -> bool:
    """
    Processes a classification results file for a specific scenario and question. This function reads
    a file containing classifications, verifies required columns, combines child and parent
    classifications if specified, aggregates the classifications, and updates the original dataframe
    with the processed classifications.

    :param scenario: The scenario identifier as an integer.
    :param question: A string specifying the question identifier associated with the scenario.
    :param original_df: A pandas DataFrame that will be updated with aggregated classification
        information.
    :param combine_child_parent: A boolean indicating whether child and parent classifications
        should be combined.

    :return: A boolean indicating the success or failure of the processing.
    """
    classification_path = f"./output/S{scenario}_{question}_classification_results.csv"

    try:
        file_mod_timestamp = os.path.getmtime(classification_path)
        file_mod_date = datetime.fromtimestamp(file_mod_timestamp).strftime("%Y-%m-%d")

        classifications_df = pd.read_csv(classification_path)

        if not {"PredictedChild", "PredictedParent"}.issubset(classifications_df.columns):
            print(f"Warning: Expected classification columns not found in {classification_path}")
            print(f"Available columns: {classifications_df.columns.tolist()}")
            return False

        classifications_df = _combine_classifications(classifications_df, combine_child_parent)
        aggregated = _aggregate_classifications(classifications_df)
        _update_original_dataframe(original_df, aggregated, file_mod_date)

        print(f"Merged classifications for Scenario {scenario}, {question} (file date: {file_mod_date})")
        return True

    except FileNotFoundError:
        print(f"Classification file not found for Scenario {scenario}, {question}")
        return False
    except Exception as e:
        print(f"Error processing Scenario {scenario}, {question}: {str(e)}")
        return False


def _print_merge_statistics(original_df: pd.DataFrame, current_date: str, output_path: str) -> None:
    """
    Prints merge statistics for a given DataFrame, including summaries of classified
    and unclassified responses, version distributions, top classification patterns,
    records updated to a specific version, and multi-category responses. Additionally,
    outputs the merged file's save location and the current classification date.

    :param original_df: The original DataFrame containing classification data and
        related metadata.
    :type original_df: pd.DataFrame
    :param current_date: The current date represented as a string, used to mark
        the classification date in the statistics.
    :type current_date: str
    :param output_path: The file path where the merged file has been saved,
        used for display purposes in the output.
    :type output_path: str
    :return: This function does not return a value; it outputs statistics to the console.
    :rtype: None
    """
    print(f"\n{'=' * 60}")
    print(f"Merged file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(original_df)}")

    classified = original_df[
        ~original_df['Classification'].isin(UNCLASSIFIED_VALUES)
    ]
    print(f"Classified responses: {len(classified)}")
    print(f"Unclassified responses: {len(original_df) - len(classified)}")
    print(f"\nVersion distribution:\n{original_df['Version'].value_counts()}")
    print(f"\nTop 10 classification patterns:\n{original_df['Classification'].value_counts().head(10)}")

    multi_category = original_df[original_df['Classification'].str.contains(r'\|', na=False)]
    print(f"\nResponses with multiple categories: {len(multi_category)}")

    updated_records = original_df[original_df['Version'] == UPDATED_VERSION]
    print(f"\nRecords updated to {UPDATED_VERSION}: {len(updated_records)}")
    print(f"Classification date: {current_date}")


def merge_all_classifications_single_column(
        classification_files: list[tuple[int, str]],
        original_data_path: str,
        output_path: str,
        combine_child_parent: bool = True
) -> None:
    """
    Merges multiple classification results files into a single column of an original dataset. The function iterates
    over provided classification files, reads their contents, aggregates classifications for each `ResponseID`,
    and updates the original dataset accordingly. It can optionally combine classifications from child and
    parent categories into a unified column.

    The merged data is saved to a specified output path, and summary information about the merging process,
    such as the total number of responses, classified responses, unclassified ones, and version distribution,
    is displayed.

    :param classification_files: List of tuples where each tuple contains a scenario ID (int) and a question ID (str).
    :param original_data_path: Path to the original dataset in CSV format.
    :param output_path: Path where the updated dataset will be saved.
    :param combine_child_parent: If True, combines child and parent classifications into a single classification.
                                 Defaults to True.
    :return: None
    """
    original_df = pd.read_csv(original_data_path)
    current_date = datetime.now().strftime('%Y-%m-%d')

    original_df = _initialize_dataframe_columns(original_df)

    for scenario, question in classification_files:
        _process_classification_file(scenario, question, original_df, combine_child_parent)

    original_df.to_csv(output_path, index=False)
    _print_merge_statistics(original_df, current_date, output_path)


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that the input DataFrame contains specific required columns. If any of the specified
    columns are missing, they are added to the DataFrame with default values.

    :param df: Input DataFrame to be processed.
    :type df: pd.DataFrame
    :return: The updated DataFrame with all required columns ensured.
    :rtype: pd.DataFrame
    """
    if 'PredictedChilds' not in df.columns:
        df['PredictedChilds'] = 'not_classified'
    if 'PredictedParents' not in df.columns:
        df['PredictedParents'] = 'not_classified'
    if 'Version' not in df.columns:
        df['Version'] = df.get('Version', 'v0.1')
    if 'ChangeNote' not in df.columns:
        df['ChangeNote'] = df.get('ChangeNote', 'Initial merge')
    if 'ChangeDate' not in df.columns:
        df['ChangeDate'] = ''
    return df


def _aggregate_and_merge_classifications(
        original_df: pd.DataFrame,
        classifications_df: pd.DataFrame,
        file_mod_date: str
) -> None:
    """
    Aggregates classifications by grouping the data by 'ResponseID', collecting unique sorted
    values of 'PredictedChild' and 'PredictedParent', and then updates the original dataframe
    with the aggregated results. Additionally, it updates specific classifications and
    metadata fields in the original dataframe based on given conditions.

    :param original_df: The original DataFrame to be updated with aggregated and merged values.
    :type original_df: pd.DataFrame
    :param classifications_df: The DataFrame containing predictions to be aggregated and merged.
    :type classifications_df: pd.DataFrame
    :param file_mod_date: A string representing the modification date to be applied to updated records.
    :type file_mod_date: str
    :return: None
    """
    aggregated = classifications_df.groupby("ResponseID").agg({
        "PredictedChild": lambda x: "|".join(sorted(set(x))),
        "PredictedParent": lambda x: "|".join(sorted(set(x)))
    }).reset_index()

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


def _process_single_classification_file(
        scenario: int,
        question: str,
        original_df: pd.DataFrame
) -> None:
    """
    Processes a single classification result file for the given scenario and question,
    merging it with the original dataframe if the required columns are present. If the
    classification file does not exist or an error occurs during processing, appropriate
    messages are printed, and processing for that specific scenario and question halts.

    :param scenario: Scenario identifier
    :type scenario: int
    :param question: Question identifier
    :type question: str
    :param original_df: The original dataframe to merge classification results into
    :type original_df: pd.DataFrame
    :return: None
    """
    classification_path = f"./output/S{scenario}_{question}_classification_results.csv"

    try:
        file_mod_timestamp = os.path.getmtime(classification_path)
        file_mod_date = datetime.fromtimestamp(file_mod_timestamp).strftime("%Y-%m-%d")

        classifications_df = pd.read_csv(classification_path)

        if not {"PredictedChild", "PredictedParent"}.issubset(classifications_df.columns):
            print(f"Warning: required columns not found in {classification_path}")
            print(f"Available columns: {classifications_df.columns.tolist()}")
            return

        _aggregate_and_merge_classifications(original_df, classifications_df, file_mod_date)
        print(f"Merged classifications for Scenario {scenario}, {question} (file date: {file_mod_date})")

    except FileNotFoundError:
        print(f"Classification file not found for Scenario {scenario}, {question}")
    except Exception as e:
        print(f"Error processing Scenario {scenario}, {question}: {str(e)}")


def _print_classification_statistics(
        df: pd.DataFrame,
        output_path: str,
        current_date: str
) -> None:
    """
    Prints classification statistics of a given DataFrame and saves
    related information to the output path. This includes detailed reporting
    on classified and unclassified responses, distribution of versions, and
    frequent classification patterns. Additionally, the number of records
    updated to a specific version is displayed.

    :param df: The DataFrame containing classification data.
    :param output_path: The directory path where the merged file is saved.
    :param current_date: The date of classification as a string.
    :return: This function does not return a value.
    """
    print(f"\n{'=' * 60}")
    print(f"Merged file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(df)}")

    classified = df[
        (df['PredictedChilds'] != 'not_classified') |
        (df['PredictedParents'] != 'not_classified')
        ]
    print(f"Classified responses: {len(classified)}")
    print(f"Unclassified responses: {len(df) - len(classified)}")

    print(f"\nVersion distribution:")
    print(df['Version'].value_counts())

    print(f"\nTop 10 child classification patterns:")
    print(df['PredictedChilds'].value_counts().head(10))

    print(f"\nTop 10 parent classification patterns:")
    print(df['PredictedParents'].value_counts().head(10))

    updated_records = df[df['Version'] == 'v0.2']
    print(f"\nRecords updated to v0.2: {len(updated_records)}")
    print(f"Classification date: {current_date}")


def merge_all_classifications_multi_column(
        classification_files: list[tuple[int, str]],
        original_data_path: str,
        output_path: str
) -> None:
    """
    Merges multiple classification files into a single output file by incorporating their data into the
    original dataset. Processes each classification file to ensure correct formatting and combines the
    results into a multi-column dataset. Adds classification statistics and outputs the final dataset
    to a specified file.

    :param classification_files: List of tuples where each tuple contains an integer representing the
        scenario ID and a string representing the file path to a classification file.
    :param original_data_path: Path to the input file containing the original data in CSV format.
    :param output_path: Path to the CSV file where the merged dataset will be saved.
    :return: None
    """
    original_df = pd.read_csv(original_data_path)
    current_date = datetime.now().strftime('%Y-%m-%d')

    original_df = _ensure_required_columns(original_df)

    for scenario, question in classification_files:
        _process_single_classification_file(scenario, question, original_df)

    original_df.to_csv(output_path, index=False)
    _print_classification_statistics(original_df, output_path, current_date)


def merge_all_classifications():
    """
    Merge classifications from multiple sources into a single unified dataset.

    This function processes multiple classification files, each identified by a
    tuple containing a scenario number and a corresponding question identifier.
    It utilizes the `merge_all_classifications_multi_column` function to merge
    data from these classification sources. The combined data is read from the
    original data path and written to the output path.

    The classification files are configured via the CLASSIFICATION_FILE_CONFIGS
    constant, which defines all scenario and question combinations to process.

    :raises FileNotFoundError: If the original data file path does not exist.
    :return: None
    """
    merge_all_classifications_multi_column(
        classification_files=CLASSIFICATION_FILE_CONFIGS,
        original_data_path=ORIGINAL_DATA_PATH,
        output_path=CLASSIFIED_OUTPUT_PATH
    )


def _create_grouped_summary(df: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """
    Creates a grouped summary from the dataframe for a specific prediction column.

    :param df: DataFrame containing the results data
    :param prediction_column: Column name to a group by (e.g. 'PredictedChild' or 'PredictedParent')
    :return: Grouped summary DataFrame with counts
    """
    grouping_columns = [prediction_column, "has_protected", "has_conflict", "has_cross_conflict"]
    return df.groupby(grouping_columns).size().reset_index(name="count")


def _save_and_display_summary(summary_df: pd.DataFrame, output_path: str, summary_type: str) -> None:
    """
    Saves the summary DataFrame to CSV and displays a preview.

    :param summary_df: Summary DataFrame to save
    :param output_path: Path where the CSV file will be saved
    :param summary_type: Type of summary for display purposes (e.g. 'Child', 'Parent')
    """
    summary_df.to_csv(output_path, index=False)
    print(f"\n{summary_type} summary report saved to: {output_path}")
    print(summary_df.head(10))


def save_summary_report(results: list[dict], summary_path: str) -> None:
    """
    Generates and saves summary reports based on the provided results data. The function
    uses the input data to create two separate grouped summaries, one for 'child' and one
    for 'parent' predictions, which are then saved as CSV files.
    :param results: List of dictionaries containing prediction results and associated
        flags to be summarized.
    :type results: list[dict]
    :param summary_path: Path where the summary reports will be saved. The function
        generates two separate CSV files, appending '_child' and '_parent' to the base
        filename before saving them.
    :type summary_path: str
    :return: None
    """
    df = pd.DataFrame(results)

    # Generate child summary
    summary_child = _create_grouped_summary(df, "PredictedChild")
    child_path = summary_path.replace('.csv', '_child.csv')
    _save_and_display_summary(summary_child, child_path, "Child")

    # Generate parent summary
    summary_parent = _create_grouped_summary(df, "PredictedParent")
    parent_path = summary_path.replace('.csv', '_parent.csv')
    _save_and_display_summary(summary_parent, parent_path, "Parent")


def _validate_key(key, errors: list) -> None:
    """
    Validates the given key to ensure it is a string.

    If the key is not a string, an error message will be appended to
    the provided errors list.

    :param key: The key to validate.
    :type key: str
    :param errors: A list to which error messages will be appended
        if the validation fails.
    :type errors: list[str]
    :return: None
    """
    if not isinstance(key, str):
        errors.append(f"Key {key} is not a string.")


def _validate_value_is_list(key: str, value, errors: list) -> bool:
    """
    Validates that the provided value is a list. If the value is not a list, an error message is appended to the
    errors list, and validation fails.

    :param key: The key associated with the value being validated.
    :type key: str
    :param value: The value to validate.
    :type value: Any
    :param errors: A list to which error messages are appended if validation fails.
    :type errors: list
    :return: True if the value is a list, else False.
    :rtype: bool
    """
    if not isinstance(value, list):
        errors.append(f"Value for key '{key}' must be a list, got {type(value).__name__}.")
        return False
    return True


def _validate_list_items(key: str, value: list, errors: list) -> None:
    """
    Validates that all items within a list are of type string. If an item in the list is not
    a string, an error message is appended to the `errors` list indicating the invalid type
    and its position.

    :param key: The key representing the list being validated.
    :type key: str
    :param value: The list of items to validate.
    :type value: list
    :param errors: A list to store validation error messages.
    :type errors: list
    :return: This function does not return a value; it appends error messages to the provided errors list.
    :rtype: None
    """
    for i, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(
                f"Value at {key}[{i}] must be a string, got {type(item).__name__}."
            )


def _count_synonyms(data: dict) -> int:
    """
    Counts the total number of synonyms present in the input dictionary. The
    function iterates over the values of the dictionary, checks if they are lists,
    and, if so, sums up the lengths of these lists to calculate the total count
    of synonyms.

    :param data: The input dictionary where keys are typically words and values
        are expected to be lists of synonyms.
    :type data: dict
    :return: The total count of synonyms from the dictionary.
    :rtype: int
    """
    return sum(len(v) for v in data.values() if isinstance(v, list))


def validate_domain_synonyms(file_path: str) -> dict:
    """
    Validates the format and structure of the domain synonyms JSON file specified
    by the file path. The function checks for the following conditions:
    1. The file must contain a dictionary at the top level.
    2. Keys of the dictionary must be strings.
    3. Values associated with each key must be lists.
    4. All elements within the lists must be strings.
    Returns a dictionary indicating whether the file is valid, a list of error
    messages if any issues are identified, the total count of terms (keys), and
    the total count of synonyms (all items across the lists).
    :param file_path: Path to the JSON file containing domain synonyms
    :type file_path: str
    :return: A dictionary containing validation results, including error messages,
        the count of terms, and the count of synonyms.
    :rtype: dict
    """
    errors = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("DOMAIN_SYNONYMS file must contain a dictionary at the top level.")

    for key, value in data.items():
        _validate_key(key, errors)

        is_valid_list = _validate_value_is_list(key, value, errors)
        if not is_valid_list:
            continue

        _validate_list_items(key, value, errors)

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "count_terms": len(data),
        "count_synonyms": _count_synonyms(data)
    }


def _normalize_key(key: str) -> str:
    """
    Normalize a given key by converting it to lowercase and removing any characters
    that are not alphanumeric. This ensures that keys are consistent and follow
    a unified format.

    :param key: The input key string to be normalized.
    :type key: str
    :return: A normalized string containing only lowercase alphanumeric characters.
    :rtype: str
    """
    return re.sub(r'[^a-z0-9]', '', key.lower())


def _generate_pascal_case(base: str) -> str:
    """
    Generates a PascalCase string from a given base string.

    This function takes a base string, splits it into words based on spaces,
    capitalizes the first letter of each word, and then joins them into a
    single PascalCase formatted string.

    :param base: The input string to be converted into PascalCase
    :type base: str
    :return: A string converted to PascalCase
    :rtype: str
    """
    return ''.join(word.capitalize() for word in base.split())


def _generate_snake_case(base: str) -> str:
    """
    Generates a snake_case formatted string from a given base string.

    This function takes a string input and transforms it into a snake_case format by
    splitting the string at whitespace and joining the parts with underscores.

    :param base: The input string to be transformed into snake_case.
    :type base: str
    :return: A snake_case formatted string derived from the input string.
    :rtype: str
    """
    return '_'.join(base.split())


def _generate_kebab_case(base: str) -> str:
    """
    Generates a kebab-case (dash-separated) version of the given string.

    The method splits the given string by spaces and joins its components using
    dashes ('-'). This transformation can be useful in creating identifiers that
    adhere to kebab-case formatting styles.

    :param base: The input string to be transformed into kebab-case.
                 The string should be space-separated if multiple words are present.
    :type base: str
    :return: A kebab-case formatted string derived from the input.
    :rtype: str
    """
    return '-'.join(base.split())


def build_category_key_map(category_keywords: dict) -> dict:
    """
    Builds a mapping of various key formats to their canonical forms.

    This function generates a mapping of normalized strings, case format variants
    (e.g., camelCase, PascalCase, snake_case, kebab-case), and their original canonical keys,
    based on the input dictionary of category keywords.

    :param category_keywords: A dictionary of category keywords with canonical
        string keys that serve as the base for generating various key formats.
    :type category_keywords: dict

    :return: A dictionary mapping different formatted keys (e.g., normalized,
        case variants) to their corresponding canonical keys from the input.
    :rtype: dict
    """
    key_map = {}

    for canonical in category_keywords.keys():
        base = canonical.lower()

        # Add normalized key (alphanumeric only)
        normalized = _normalize_key(canonical)
        key_map[normalized] = canonical

        # Add case format variants
        pascal = _generate_pascal_case(base)
        key_map[pascal.lower()] = canonical  # camelCase: terrainandbathymetry
        key_map[pascal] = canonical  # PascalCase: TerrainAndBathymetry

        snake = _generate_snake_case(base)
        key_map[snake] = canonical  # snake_case: terrain_and_bathymetry

        kebab = _generate_kebab_case(base)
        key_map[kebab] = canonical  # kebab-case: terrain-and-bathymetry

    return key_map


def _normalise_key(key: str) -> str:
    """
    Normalizes a given key by transforming it to lowercase and removing any
    characters that are not letters (a-z) or digits (0-9).

    :param key: The input string to be normalized.
    :type key: str
    :return: A normalized string consisting only of lowercase letters and digits.
    :rtype: str
    """
    return re.sub(r'[^a-z0-9]', '', key.lower())


# ... existing code ...

def normalise_category_keys(imported_dict: dict, category_keywords: dict) -> dict:
    """
    Normalizes the keys of the input dictionary according to a predefined category key map
    derived from given category keywords.

    This function helps standardize the keys in a dictionary by:
    1. Normalizing keys to lowercase alphanumeric strings by removing any non-alphanumeric
       characters.
    2. Mapping normalized keys to their corresponding standard category keys using the
       category keywords provided.

    After processing, any key that does not match an entry in the key map will remain
    unchanged in the returned dictionary.

    :param imported_dict: The input dictionary whose keys need to be normalized.
    :param category_keywords: A dictionary defining the relation between category keywords
        and standardized category names.
    :return: A new dictionary with normalized and mapped keys.
    :rtype: dict
    """
    key_map = build_category_key_map(category_keywords)

    normalised_dict = {}
    for k, v in imported_dict.items():
        norm_k = _normalise_key(k)
        mapped_key = key_map.get(norm_k, k)  # fallback to original if not found
        normalised_dict[mapped_key] = v

    return normalised_dict


def humanise_key(key: str) -> str:
        """
        Transforms a string key into a human-readable format by replacing specific
        characters and adjusting case. Underscores and hyphens are replaced with
        spaces. Spaces are added before capitalized letters (unless at the start),
        and the result is normalized and converted to title case.
        :param key: The input key to be transformed.
        :return: A human-readable string derived from the input key.
        """
        # Replace underscores and hyphens with spaces
        result = key.replace("_", " ").replace("-", " ")
        # Add spaces before capital letters (but not at start)
        result = re.sub(_CAMEL_CASE_PATTERN, ' ', result)
        # Normalise whitespace and title case
        result = re.sub(_WHITESPACE_NORMALIZATION_PATTERN, ' ', result).strip()
        return result


def normalise_key(name: str) -> str:
    """
    Normalize a camel case or Pascal case string to a space-separated string.
    This function takes a camel case or Pascal case string, separates each word
    by inserting a space character, and removes any leading or trailing spaces
    from the resulting string.
    :param name: The input string in camel case or Pascal case format.
    :type name: str
    :return: A string with each word separated by a space and stripped of
        leading or trailing spaces.
    :rtype: str
    """
    words_separated_uppercase = re.sub(_UPPERCASE_LOWERCASE_PATTERN, r'\1 \2', name)
    return re.sub(_LOWERCASE_DIGIT_UPPERCASE_PATTERN, r'\1 \2', words_separated_uppercase).strip()


def _load_taxonomy_json(json_path: str) -> dict:
    """
    Loads the taxonomy JSON file and extracts the MissionPlanningTaxonomy root.

    :param json_path: Path to the JSON file containing taxonomy data.
    :type json_path: str
    :return: The taxonomy dictionary.
    :rtype: dict
    """
    with open(json_path, "r") as f:
        taxonomy = json.load(f)
    return taxonomy.get("MissionPlanningTaxonomy", taxonomy)


def _process_child_concepts(parent_key: str, parent_val: dict) -> tuple[dict[str, dict], set[str]]:
    """
    Processes child concepts for a given parent category, extracting child keywords
    and aggregating all keywords for the parent.

    :param parent_key: The parent category key name.
    :type parent_key: str
    :param parent_val: The parent category value containing concepts.
    :type parent_val: dict
    :return: A tuple containing:
        - Dictionary of child keywords with their metadata
        - Set of all keywords associated with this parent
    :rtype: tuple[dict[str, dict], set[str]]
    """
    parent_name = normalise_key(parent_key)
    child_keywords = {}
    parent_keyword_set = set()

    for child_key, keywords in parent_val.get("Concepts", {}).items():
        child_name = normalise_key(child_key)
        child_keywords[child_name] = {
            "keywords": keywords,
            "parent": parent_name
        }
        parent_keyword_set.update(keywords)

    return child_keywords, parent_keyword_set


def load_taxonomy(json_path: str):
    """
    Loads and processes a taxonomy from a JSON file, extracting parent and child
    keywords along with their hierarchical relationships. The function normalizes
    keys, organizes parent-child relationships, and structures the data into two
    dictionaries: one containing keywords grouped by normalized parent names and
    another containing child concepts with their attributes.

    :param json_path: Path to the JSON file containing taxonomy data.
    :type json_path: str
    :return: A tuple containing two dictionaries:
        1. A dictionary where keys are normalized parent names and values are
           sorted lists of keywords associated with those parents.
        2. A dictionary where keys are normalized child names and values are
           dictionaries containing associated keywords and parent names.
    :rtype: tuple[dict, dict]
    """
    taxonomy = _load_taxonomy_json(json_path)

    parent_keywords = {}
    child_keywords = {}

    for parent_key, parent_val in taxonomy.items():
        child_data, parent_keyword_set = _process_child_concepts(parent_key, parent_val)
        child_keywords.update(child_data)

        parent_name = normalise_key(parent_key)
        parent_keywords[parent_name] = sorted(list(parent_keyword_set))

    return parent_keywords, child_keywords


def _normalize_and_group_terms(data: dict) -> dict[str, set[str]]:
    """
    Normalizes and groups terms from raw data dictionary.

    :param data: Dictionary mapping categories to lists of terms.
    :type data: dict
    :return: Dictionary mapping categories to sets of normalized terms.
    :rtype: dict[str, set[str]]
    """
    return {k: {t.lower().strip() for t in v} for k, v in data.items()}


def load_protected_terms(path: str, flatten: bool = True) -> set[str] | dict[str, set[str]]:
    """
    Loads and processes protected terms from a JSON file at the specified path. The function reads
    the JSON file, parses its content to create a dictionary where keys are categories or group names
    and values are sets of terms specific to those groups. If the `flatten` parameter is set to True,
    the function merges all the grouped terms into a single set.

    :param path: The file path to the JSON file containing the protected terms.
    :type path: str
    :param flatten: Whether to merge all grouped terms into a single set. Defaults to True.
    :type flatten: bool, optional
    :return: A set of all terms if `flatten` is True, otherwise a dictionary grouping terms into categories.
    :rtype: set[str] | dict[str, set[str]]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = _normalize_and_group_terms(data)

    if flatten:
        return set().union(*grouped.values())
    return grouped


def _normalize_rule_list(rules: list) -> list:
    """
    Normalize a list of rules by converting to lowercase and stripping whitespace.

    :param rules: List of rule strings to normalize.
    :type rules: list
    :return: List of normalized rule strings.
    :rtype: list
    """
    return [rule.lower().strip() for rule in rules]


def load_conflict_rules(path: str) -> dict:
    """
    Load conflict rules from a JSON file and normalize the data to lowercase for consistent
    matching. The normalization ensures that all keys and list values in the rules dictionary
    are converted to lowercase.
    :param path: The file path to the JSON file containing conflict rules.
    :type path: str
    :return: A dictionary containing normalized conflict rules with "protected" and
        "synonyms" lists for each context.
    :rtype: dict
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # normalise to lowercase for consistent matching
    normalized = {}
    for ctx, rules in data.items():
        normalized[ctx.lower()] = {
            "protected": _normalize_rule_list(rules.get("protected", [])),
            "synonyms": _normalize_rule_list(rules.get("synonyms", []))
        }
    return normalized


def load_taxonomy_configuration(taxonomy_path: str) -> tuple[dict, dict]:
    """
    Loads and processes a taxonomy configuration file, extracting pertinent information
    such as the full taxonomy and its category keywords. This function reads a JSON file
    specified by the given path, processes its content to extract human-readable categories and their
    respective keywords, and prints a summary of the keyword count for debugging purposes.

    :param taxonomy_path: The file path to the taxonomy configuration file.
                          It is expected to be a valid JSON file.
    :return: A tuple containing two elements:
             1. The raw taxonomy dictionary loaded from the file.
             2. A dictionary of category keywords with human-readable keys and their associated values.
    """
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy_raw = json.load(f)

    # Extract the actual taxonomy data from the wrapper
    if "MissionPlanningTaxonomy" in taxonomy_raw:
        taxonomy = taxonomy_raw["MissionPlanningTaxonomy"]
    else:
        taxonomy = taxonomy_raw

    category_keywords = {humanise_key(k): v for k, v in taxonomy.items()}

    # Debug print
    print("\nCategory Keyword Summary")
    print("==========================")
    for cat, kws in category_keywords.items():
        print(f"{cat}: {len(kws)} keywords")

    return taxonomy_raw, category_keywords


def load_and_validate_domain_synonyms(synonyms_path: str) -> dict:
    """
    Loads and validates domain synonyms from a given file path. The function checks
    the validity of the synonyms file, prints a detailed report of the validation,
    and loads the content into a dictionary. The dictionary keys and values are
    converted to lowercase for consistency.

    :param synonyms_path: Path to the JSON file containing domain synonyms.
    :type synonyms_path: str
    :return: A dictionary where keys are domain terms in lowercase and values
        are lists of their respective synonyms in lowercase.
    :rtype: dict
    """
    report = validate_domain_synonyms(synonyms_path)

    print("\nDomain Synonym Validation Report:")
    print("====================================")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Terms: {report['count_terms']}")
    print(f"  Synonyms: {report['count_synonyms']}")

    if not report["is_valid"]:
        print("\nErrors found:")
        for err in report["errors"]:
            print(f"  - {err}")

    with open(synonyms_path, 'r', encoding='utf-8') as f:
        domain_synonyms = {k.lower(): [s.lower() for s in v] for k, v in json.load(f).items()}

    print("Loaded terms:", len(domain_synonyms))
    print("Sample mission synonyms:", domain_synonyms.get("mission"))

    return domain_synonyms


def load_validation_configuration(
        protected_terms_path: str,
        conflict_rules_path: str,
        domain_synonyms: dict
) -> ValidationRules:
    """
    Load and parse the validation configuration from the provided paths and inputs.
    This includes loading protected terms, conflict rules, and domain-specific
    synonym mappings, which are used to define validation rules for the application.

    :param protected_terms_path: The path to the file containing protected terms.
    :param conflict_rules_path: The path to the file containing conflict rules.
    :param domain_synonyms: A dictionary mapping domain terms to their synonyms.
    :return: The constructed ValidationRules object based on the provided inputs.
    """
    protected_terms = load_protected_terms(protected_terms_path, flatten=False)
    conflict_rules = load_conflict_rules(conflict_rules_path)

    return ValidationRules(
        conflict_rules=conflict_rules,
        protected_terms=protected_terms,
        domain_synonyms=domain_synonyms
    )


def save_suppression_report(suppression_log: list, output_path: str = './output/suppression_log.json') -> None:
    """
    Saves the suppression log to a specified output file in JSON format, providing summary statistics
    about the suppressions recorded. Useful for reviewing and debugging suppressed triggers or
    terms logged during runtime.

    :param suppression_log: List of suppression log entries recorded during runtime.
    :param output_path: Path to save the suppression log file. Defaults to './output/suppression_log.json'.
    :return: None
    """
    if len(suppression_log) == 0:
        print("\nNo suppressions logged during this run.")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(suppression_log, f, indent=2, ensure_ascii=False)

    print(f"\nSuppression log saved to: {output_path}")
    print(f"Total suppressions logged: {len(suppression_log)}")

    # Print summary statistics
    protected_count = sum(1 for entry in suppression_log if entry.get('protected_trigger'))
    synonym_count = len(suppression_log) - protected_count
    print(f"  - Protected term suppressions: {protected_count}")
    print(f"  - Synonym unavailable suppressions: {synonym_count}")


if __name__ == "__main__":
    # Load taxonomy configuration
    marine_planning_taxonomy_raw, CATEGORY_KEYWORDS = load_taxonomy_configuration(
        './docs/mission_planning_taxonomy.json'
    )

    # Load and validate domain synonyms
    DOMAIN_SYNONYMS = load_and_validate_domain_synonyms('./docs/domain_synonyms.json')

    # Load validation configuration
    validation_rules = load_validation_configuration(
        protected_terms_path='./docs/protected_terms.json',
        conflict_rules_path='./docs/conflict_rules.json',
        domain_synonyms=DOMAIN_SYNONYMS
    )

    # Extract PROTECTED_TERMS from validation_rules for use in augmentation
    PROTECTED_TERMS = validation_rules.protected_terms

    # Run classification for all scenarios
    for scenario_num in [1, 2, 3]:
        print(f"\n{'#' * 60}")
        for question_num in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
            print(f"Running classification for Scenario {scenario_num}, Question {question_num}")
            main(question_num, scenario_num, marine_planning_taxonomy_raw,
                 validation_rules, merge_to_source=False)

    # After all scenarios are classified, merge into a single file
    print(f"\n{'#' * 60}")
    print(f"# Merging all classifications into master file")
    print(f"{'#' * 60}\n")
    merge_all_classifications()

    # Save a suppression report
    save_suppression_report(SUPPRESSION_LOG)
