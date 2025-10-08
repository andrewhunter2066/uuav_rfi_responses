#!/usr/bin/env python3

import pandas as pd
import numpy as np
import random
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

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
    r'\bcomms\b': 'communications'
}

# Regex pattern to keep only word characters, spaces, and hyphens
PUNCTUATION_PATTERN = r'[^\w\s\-]'

# Define keywords for each category as a module constant
Q1_CATEGORY_KEYWORDS = {
    "terrain and seafloor features": [
        "seafloor topography",
        "seafloor texture",
        "seafloor complexity",
        "seafloor classification",
        "seabed complexity",
        "seabed features",
        "bottom texture",
        "terrain complexity",
        "seabed gradient"
    ],
    "environmental conditions": [
        "ephemeral current",
        "tides",
        "current",
        "tidal stream",
        "ocean currents",
        "water column",
        "water conditions",
        "wave height",
        "environmental factors",
        "environmental conditions",
        "water column"
    ],
    "navigation and localization": [
        "gps requirements",
        "navigation accuracy",
        "navigation errors",
        "positioning",
        "gps fix",
        "navigation systems",
        "navigational charts",
        "route data"
    ],
    "vehicle capabilities": [
        "uuv limits",
        "endurance",
        "vehicle",
        "minimum depth",
        "maximum depth",
        "range",
        "speed",
        "vehicle mobility",
        "uuv capabilities",
        "uuv limits",
        "ascent",
        "descent",
        "dive",
        "timings hard left right"
    ],
    "sensors and data collection": [
        "sensor collection requirements",
        "sensor specifications",
        "multibeam echo sounder coverage",
        "resolution",
        "overlap",
        "gaps",
        "sensor performance",
        "active sensor"
    ],
    "surveillance and traffic considerations": [
        "surface subsea traffic",
        "surface traffic",
        "subsea traffic",
        "hazards along route",
        "no-go zones",
        "threats to a mission",
        "threats"
    ],
    "operational and logistical factors": [
        "launch recovery locations",
        "survey duration",
        "transit time to beach",
        "mission collection requirements",
        "search areas",
        "swept channel planning",
        "area of operations",
        "route planning",
        "survey planning",
        "constraints",
        "beach landings"
    ],
    "support data and historical information": [
        "historical data",
        "similar beaches",
        "environmental conditions",
        "mission success factors"
    ],
    "threat and risk assessment": [
        "cyber threat",
        "cyber",
        "hazards",
        "threats",
        "risks",
        "no-go zone",
        "navigational hazards",
        "communication loss"
    ],
    "data quality": [
        "accuracy requirements",
        "confidence",
        "equipment limits",
        "mission collection requirements"
    ]
}


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
def _extract_lemma_synonyms(synset: wordnet.synset, original_word: str) -> set[str]:
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
    The function utilises WordNet synsets to extract synonyms. Each synonym is
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

    return ' '.join(augmented_words)


# --- Simple keyword-based classification ---
# Since transformers may also have dependency issues, implementing a keyword-based classifier
def _calculate_category_scores(text_lower: str, q: str) -> dict[str, int]:
    """
    Calculates and returns category scores based on a text and a specific query.

    This function computes a score for each category based on the occurrence of
    keywords in the given text. The category keywords are selected based on the
    provided query. The score for a category is the total count of keywords from
    that category that appear in the text.

    :param text_lower: A lowercase string to be evaluated against category
        keywords.
        Example: "this is a sample text".
    :param q: A string representing the query to determine which category
        keywords to use. Example: "Q1".
    :return: A dictionary where keys are categories (strings) and values are
        scores (integers) representing the count of matching keywords for
        each category.
    :rtype: dict[str, int]
    """
    if q == "Q1":
        category_keywords = Q1_CATEGORY_KEYWORDS
    scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[category] = score
    return scores


def classify_response(text: str, question: str) -> tuple[str, dict[str, float]]:
    """
    Analyses a given response text and a question to classify the response into a category
    based on calculated scores. The function returns the predicted category as well as
    the normalised scores for all potential categories.

    :param text: Input response text for classification
    :type text: str
    :param question: The reference question to help in classification
    :type question: str
    :return: A tuple containing the predicted category as a string and a dictionary of
             normalised scores for all categories.
    :rtype: tuple[str, dict[str, float]]
    """
    text_lower = text.lower()

    # Calculate scores for each category
    scores = _calculate_category_scores(text_lower, question)

    # Get the category with the highest score
    predicted_category = max(scores, key=scores.get, default='uncategorised')
    if scores.get(predicted_category, 0) == 0:
        predicted_category = 'uncategorised'

    # Normalise scores
    total_score = sum(scores.values()) if sum(scores.values()) > 0 else 1
    normalized_scores = {k: v / total_score for k, v in scores.items()}

    return predicted_category, normalized_scores


def load_and_filter_responses(file_path: str, scenario_number: int, question_prefix: str) -> pd.Series:
    """
    Loads response data from CSV and filters by scenario and question criteria.

    :param file_path: Path to the input CSV file containing responses.
    :param scenario_number: Scenario number to filter responses.
    :param question_prefix: Question prefix to filter responses.
    :return: Series of filtered response texts.
    :raises FileNotFoundError: If the input CSV file is not found.
    """
    response_set = pd.read_csv(file_path)
    df = pd.DataFrame(response_set)
    df['ResponseText'] = df['ResponseText'].str.replace(r'[/]', ' ', regex=True)

    filtered_responses = df[
        (df['ScenarioNumber'] == scenario_number) &
        (df['Question'].str.startswith(question_prefix))
        ]['ResponseText']

    return filtered_responses


def preprocess_responses(response_series: pd.Series) -> list[str]:
    """
    Preprocesses and deduplicates response texts.

    :param response_series: Series of raw response texts.
    :return: List of preprocessed unique responses.
    """
    responses = [
        preprocess(r)
        for r in response_series
        if isinstance(r, str) and r.strip() != ''
    ]
    return list(set(responses))


def augment_response_dataset(responses: list[str], aug_prob: float = 0.3,
                             max_synonyms: int = 2, sample_size: int = 500) -> list[str]:
    """
    Augments response dataset using synonym replacement and samples the result.

    :param responses: List of preprocessed responses.
    :param aug_prob: Probability of word augmentation.
    :param max_synonyms: Maximum number of synonyms to use per word.
    :param sample_size: Maximum number of responses to sample.
    :return: List of augmented and sampled responses.
    """
    augmented_responses = []
    for resp in responses:
        augmented = augment_text(resp, aug_prob=aug_prob, max_synonyms=max_synonyms)
        augmented_responses.append(augmented)
        augmented_responses.append(resp)

    return random.sample(augmented_responses, min(sample_size, len(augmented_responses)))


def classify_responses(responses: list[str], question: str) -> list[dict]:
    """
    Classifies a list of responses based on a given question. Each response is
    analysed to determine its predicted category and associated scores, which
    are then aggregated into a structured list of dictionaries.

    :param responses: A list of textual responses to be classified.
    :param question: A string representing the question related to the responses.
    :return: A list of dictionaries, each containing the response, its predicted
             category, and associated scores.
    """
    results = []
    for resp in responses:
        predicted_category, all_scores = classify_response(resp, question)
        results.append({
            'response': resp,
            'predicted_category': predicted_category,
            'all_scores': str(all_scores)
        })
    return results


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
    print(df['predicted_category'].value_counts())


def main(question: str, scenario: int):
    """
    Executes the main process for loading, filtering, preprocessing, augmenting,
    classifying responses, and saving results based on input question and scenario.

    The function undergoes several steps:
    1. Loads and filters a dataset of responses based on input criteria.
    2. Preprocesses filtered responses for further processing.
    3. Augments the dataset by introducing controlled variations (e.g. synonyms).
    4. Classifies the augmented responses using a classification model.
    5. Saves the classification results to an output file.

    :param question: The question string used to filter and classify responses.
    :type question: str
    :param scenario: The scenario string used for naming the output files.
    :type scenario: str
    :return: None
    """
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

    # Classify responses
    results = classify_responses(responses_aug, question)

    # Save and display results
    save_and_display_results(
        results=results,
        output_path=f"./output/S{scenario}_{question}_classification_results.csv"
    )


if __name__ == "__main__":
    main("Q1",3)
