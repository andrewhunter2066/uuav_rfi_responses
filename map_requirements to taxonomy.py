"""
Map requirements in a CSV file to taxonomy concepts (keyword and semantic match).

Usage:
    python map_requirements_to_taxonomy.py --requirements requirements.csv --taxonomy taxonomy.json
    --output mapping_results.csv
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ---------------------------------------------------------------------
# Config:
# ---------------------------------------------------------------------
@dataclass
class AlgorithmConfig:
    """Configuration for scoring and classification algorithms."""
    # Scoring weights
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # Auto-accept thresholds
    auto_combined_threshold: float = 0.75
    auto_semantic_threshold: float = 0.80
    auto_keyword_threshold: float = 70.0

    # Review thresholds
    review_combined_threshold: float = 0.65
    review_semantic_threshold: float = 0.65
    review_keyword_threshold: float = 60.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if not abs(self.semantic_weight + self.keyword_weight - 1.0) < 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {self.semantic_weight + self.keyword_weight}"
            )

        if not (0 <= self.semantic_weight <= 1 and 0 <= self.keyword_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")

        if self.auto_combined_threshold < self.review_combined_threshold:
            raise ValueError("Auto threshold must be >= review threshold")

    def calculate_combined_score(self, semantic_score: float, keyword_score: float) -> float:
        """Calculate weighted combined score."""
        return (semantic_score * self.semantic_weight +
                keyword_score / 100 * self.keyword_weight)

    def classify_match(self, semantic_score: float, keyword_score: float,
                       combined_score: float) -> str:
        """Classify match based on configured thresholds."""
        if combined_score >= self.auto_combined_threshold or (
                semantic_score >= self.auto_semantic_threshold and
                keyword_score >= self.auto_keyword_threshold
        ):
            return "auto"
        elif combined_score >= self.review_combined_threshold or (
                semantic_score >= self.review_semantic_threshold and
                keyword_score >= self.review_keyword_threshold
        ):
            return "review"
        else:
            return "discard"


# Default algorithm configuration
DEFAULT_ALGORITHM_CONFIG = AlgorithmConfig()

# Configuration constants
BASE_PROJECT_PATH = Path("C:/Users/Andrew/PycharmProjects/RAN")
MISSION_REQUIREMENTS_FILE = BASE_PROJECT_PATH / "Reports/uuv-mission-phase-requirements.csv"
TAXONOMY_FILE = BASE_PROJECT_PATH / "rfi/rfi_data/output/reduced_mission_planning_taxonomy.json"
PROJECT_REQUIREMENTS_FILE = BASE_PROJECT_PATH / "rfi/rfi_data/output/uuv_requirements.csv"


@dataclass
class MappingConfig:
    """Configuration for requirement mapping operations."""
    mode: str
    mission_requirements: Path
    csv_output: str
    json_output: str
    top_n: int = 3
    taxonomy: Optional[Path] = None
    project_requirements: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration after initialisation."""
        if self.mode not in ("taxonomy", "reqmap"):
            raise ValueError(f"Invalid mode: '{self.mode}'. Must be 'taxonomy' or 'reqmap'")

        if self.mode == "taxonomy" and not self.taxonomy:
            raise ValueError("taxonomy mode requires 'taxonomy' parameter")

        if self.mode == "reqmap" and not self.project_requirements:
            raise ValueError("reqmap mode requires 'project_requirements' parameter")

        # Validate that input files exist
        if not self.mission_requirements.exists():
            raise FileNotFoundError(f"Mission requirements file not found: {self.mission_requirements}")

        if self.taxonomy and not self.taxonomy.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy}")

        if self.project_requirements and not self.project_requirements.exists():
            raise FileNotFoundError(f"Project requirements file not found: {self.project_requirements}")

    def execute(self):
        """Execute the mapping based on configuration."""
        if self.mode == "taxonomy":
            map_requirements(
                str(self.mission_requirements),
                str(self.taxonomy),
                self.csv_output,
                self.json_output,
                self.top_n
            )
        elif self.mode == "reqmap":
            map_requirements_to_requirements(
                str(self.mission_requirements),
                str(self.project_requirements),
                self.csv_output,
                self.json_output,
                self.top_n
            )


# Predefined configurations
TAXONOMY_MAPPING_CONFIG = MappingConfig(
    mode="taxonomy",
    mission_requirements=MISSION_REQUIREMENTS_FILE,
    taxonomy=TAXONOMY_FILE,
    top_n=3,
    csv_output="output/mapping_mission_results.csv",
    json_output="output/mapping_mission_results.json"
)

REQMAP_MAPPING_CONFIG = MappingConfig(
    mode="reqmap",
    mission_requirements=MISSION_REQUIREMENTS_FILE,
    project_requirements=PROJECT_REQUIREMENTS_FILE,
    top_n=3,
    csv_output="output/mapping_project_results.csv",
    json_output="output/mapping_project_results.json"
)


def flatten_taxonomy(taxonomy: dict, parent: str = "") -> list:
    """
    Flattens a hierarchical taxonomy dictionary into a list of flattened
    taxonomy entries. Each entry includes the path of the taxonomy node,
    its description, and associated terms.
    Args:
        taxonomy (dict): The hierarchical taxonomy structure containing
            nested concepts and their descriptions.
        parent (str, optional): The parent path for constructing
            hierarchical taxonomy paths. Defaults to an empty string.
    Returns:
        list: A list of flattened taxonomy entries, where each entry
            is represented as a dictionary with keys `path`, `description`,
            and `terms`.
    """

    def _create_taxonomy_entry(taxonomy_path: str, description: str, terms: list) -> dict:
        """Creates a single taxonomy entry dictionary."""
        return {
            "path": taxonomy_path,
            "description": description,
            "terms": terms
        }

    rows = []
    for key, value in taxonomy.items():
        taxonomy_path = f"{parent} > {key}" if parent else key
        description = value.get("Description", "")
        terms = value.get("Terms", [])
        has_nested_concepts = "Concepts" in value

        if has_nested_concepts:
            rows.extend(flatten_taxonomy(value["Concepts"], taxonomy_path))
        else:
            rows.append(_create_taxonomy_entry(taxonomy_path, description, terms))

    return rows


def _calculate_keyword_score(requirement_text: str, row) -> float:
    """
    Calculates the keyword score between a given requirement text and row data.

    This function computes a similarity score based on the tokens present in the
    requirement text and a row data source. The similarity comparison is performed
    using the `token_set_ratio` function from the `fuzz` module. The comparison
    prioritises the "terms" field in the row and falls back to the "description"
    field if "terms" is empty or contains only whitespace.

    Args:
        requirement_text: The requirement text string to compare.
        row: A dictionary-like object containing "terms" and "description" fields
            for the comparison source.

    Returns:
        An float score representing the similarity between the requirement text
        and the row data.
    """
    terms = " ".join(row["terms"])
    text_to_compare = terms if terms.strip() else row["description"]
    return fuzz.token_set_ratio(requirement_text.lower(), text_to_compare.lower())


def _calculate_semantic_scores(requirement_text: str, taxonomy_df, model) -> list:
    """
    Calculates semantic similarity scores between a requirement text and a taxonomy dataset
    based on their embeddings. It uses a machine learning model to encode the text
    and computes the cosine similarity between the encoded representations.

    Parameters:
        requirement_text: str
            The input text representing a requirement to compare with the taxonomy dataset.
        taxonomy_df
            A data frame containing a taxonomy dataset with textual descriptions to match against.
        model
            A machine learning model with encoding capabilities, used to compute text embeddings.

    Returns:
        list
            A list of semantic similarity scores calculated between the input requirement text
            and each description within the taxonomy dataset.
    """
    requirement_embedding = model.encode(requirement_text, convert_to_tensor=True)
    taxonomy_embeddings = model.encode(taxonomy_df["description"].tolist(), convert_to_tensor=True)
    cosine_scores = util.cos_sim(requirement_embedding, taxonomy_embeddings).squeeze().tolist()
    return cosine_scores


def _calculate_combined_scores(taxonomy_df, config: AlgorithmConfig = DEFAULT_ALGORITHM_CONFIG) -> None:
    """
    Calculates the combined scores for the taxonomy dataframe.

    This function computes a 'combined_score' for each entry in the provided
    taxonomy dataframe by blending its 'semantic_score' and 'keyword_score'.
    The calculation involves applying a predefined weight to each component
    to generate an overall score. The new 'combined_score' is then added
    as a column to the input dataframe.

    Args:
        taxonomy_df: A Pandas DataFrame containing the 'semantic_score' and
                     'keyword_score' columns used for calculating the
                     'combined_score'.
        config: Algorithm configuration containing weights and thresholds.

    Returns:
        None: The function modifies the input dataframe in place and does not
        return a value.
    """
    taxonomy_df["combined_score"] = (
            taxonomy_df["semantic_score"] * config.semantic_weight +
            taxonomy_df["keyword_score"] / 100 * config.keyword_weight
    )


def find_best_matches(requirement_text: str, taxonomy_df: pd.DataFrame, model,
                      top_n=5, config: AlgorithmConfig = DEFAULT_ALGORITHM_CONFIG) -> pd.DataFrame:
    """
    Find and rank the best matching taxonomy entries for a given requirement text.

    This function calculates a combination of keyword similarity and semantic
    similarity between the given requirement text and each entry in a taxonomy
    DataFrame. The function leverages fuzzy matching for keyword similarity and
    uses embeddings for semantic similarity. It outputs the top N matches based
    on a weighted score.

    Parameters:
    requirement_text: str
        The text of the requirement to be matched against the taxonomy entries.
    taxonomy_df: pandas.DataFrame
        A DataFrame containing taxonomy data with columns 'terms' and 'description'.
    model
        A pre-trained model supporting the `encode` method to generate
        embeddings for semantic similarity calculation.
    top_n: int, optional
        The number of top matches to return, default is 5.
    config: AlgorithmConfig
        Algorithm configuration containing weights and thresholds.

    Returns:
    pandas.DataFrame
        A DataFrame containing the top N taxonomy matches, including their path,
        description, combined score, keyword score, and semantic score.

    Raises:
    None
    """
    # Calculate keyword/fuzzy match scores
    keyword_scores = [
        _calculate_keyword_score(requirement_text, row)
        for _, row in taxonomy_df.iterrows()
    ]
    taxonomy_df["keyword_score"] = keyword_scores

    # Calculate semantic similarity scores using embeddings
    taxonomy_df["semantic_score"] = _calculate_semantic_scores(
        requirement_text, taxonomy_df, model
    )

    # Calculate combined weighted scores
    _calculate_combined_scores(taxonomy_df, config)

    return taxonomy_df.nlargest(top_n, "combined_score")[
        ["path", "description", "combined_score", "keyword_score", "semantic_score"]]


def _extract_taxonomy_parts(path: str) -> tuple[str, str]:
    """
    Extracts the taxonomy domain and domain concept from a given taxonomy path.

    Splits the input string using the separator ' > ' to parse the taxonomy
    details. The resulting components are returned as the taxonomy domain and
    the domain concept. If the input path contains fewer parts than expected,
    empty strings are substituted for the missing components.

    Parameters:
        path (str): The taxonomy path string, where parts are separated by ' > '.

    Returns:
        tuple: A tuple containing the taxonomy domain as the first element
        and the domain concept as the second element. If no valid parts
        exist, empty strings are returned.
    """
    path_parts = path.split(' > ')
    taxonomy_domain = path_parts[0] if len(path_parts) > 0 else ""
    domain_concept = path_parts[1] if len(path_parts) > 1 else ""
    return taxonomy_domain, domain_concept


def _create_csv_result_entry(row: dict, match: dict, req_text: str,
                             config: AlgorithmConfig = DEFAULT_ALGORITHM_CONFIG) -> dict:
    """
    Creates a dictionary entry for a CSV result based on the provided data.

    This function processes data extracted from a row, match information, and
    requirement text to generate a structured dictionary suitable for CSV output.
    It classifies the match, computes rounded scores, and organises the data in
    a readable format for further processing or storing.

    Parameters:
    row: dict
        A dictionary representing a row containing details such as the
        requirement ID.
    match: dict
        A dictionary containing match details, such as the match path and scores
        (semantic, keyword, and combined).
    req_text: str
        A string representing the requirement text that corresponds to the row.
    config: AlgorithmConfig
        Algorithm configuration containing weights and thresholds.

    Returns:
    dict
        A dictionary with keys such as 'requirement_id', 'requirement_text',
        'taxonomy_domain', 'domain_concept', 'combined_score', 'keyword_score',
        'semantic_score', and 'classification', containing the processed and
        classified data.
    """
    taxonomy_domain, domain_concept = _extract_taxonomy_parts(match["path"])
    label = config.classify_match(
        semantic_score=round(match["semantic_score"], 3),
        keyword_score=round(match["keyword_score"], 1),
        combined_score=round(match["combined_score"], 3)
    )
    return {
        "requirement_id": row.get("id", ""),
        "requirement_text": req_text,
        "taxonomy_domain": taxonomy_domain,
        "domain_concept": domain_concept,
        "combined_score": round(match["combined_score"], 3),
        "keyword_score": round(match["keyword_score"], 1),
        "semantic_score": round(match["semantic_score"], 3),
        "classification": label
    }


def _create_json_result_entry(row: dict, matches: pd.DataFrame, req_text: str) -> dict:
    """
    Generates a JSON-compatible result entry from a given row and matches.

    The function processes provided match data to extract taxonomy components, calculate scores,
    and classify the matches. It organises this information into a structured dictionary alongside
    requirement details. The output is suitable for integration into JSON responses.

    Arguments:
    row (dict): A dictionary representing a row of data that may include requirement details.
    matches (DataFrame): A Pandas DataFrame containing match data, including scores and paths.
    req_text (str): The requirement text associated with the row.

    Returns:
    dict: A structured dictionary with requirement details and match information.
    """
    match_list = []
    for _, m in matches.iterrows():
        taxonomy_domain, domain_concept = _extract_taxonomy_parts(m["path"])
        match_list.append({
            "taxonomy_domain": taxonomy_domain,
            "domain_concept": domain_concept,
            "combined_score": round(m["combined_score"], 3),
            "keyword_score": round(m["keyword_score"], 1),
            "semantic_score": round(m["semantic_score"], 3),
            "classification": classify_match(
                round(m["semantic_score"], 3),
                round(m["keyword_score"], 1),
                round(m["combined_score"], 3)
            )
        })

    return {
        "requirement_id": row.get("id", ""),
        "requirement_text": req_text,
        "matches": match_list
    }


def map_requirements(requirements_csv: str, taxonomy_json: str, output_csv: str, output_json: str, top_n=3) -> None:
    """
    Maps requirements from a CSV file to taxonomy concepts using a semantic similarity model.
    The function processes a taxonomy from a JSON file and requirements from a CSV file.
    It employs a pre-trained sentence transformer model to compute semantic similarity
    between requirements and taxonomy concepts. Results are saved in both flat CSV format
    and structured JSON format.
    Arguments:
        requirements_csv: Path to the input CSV file containing requirements.
        taxonomy_json: Path to the input JSON file containing taxonomy data.
        output_csv: Path where the resulting mapped CSV file will be saved.
        output_json: Path where the resulting mapped JSON file will be saved.
        top_n: Optional; the number of top matches to retrieve for each requirement.
               Default is 3.
    Raises:
        IOError: If the input or output files cannot be accessed.
        ValueError: If the necessary columns are missing in the requirements CSV.
    """
    print(f"\n📘 Loading taxonomy: {taxonomy_json}")
    with open(taxonomy_json, "r", encoding="utf-8") as f:
        taxonomy_root = json.load(f)
    if len(taxonomy_root) == 1 and isinstance(next(iter(taxonomy_root.values())), dict):
        taxonomy_root = next(iter(taxonomy_root.values()))
    taxonomy_flat = flatten_taxonomy(taxonomy_root)
    taxonomy_df = pd.DataFrame(taxonomy_flat)
    print(f"✅ Taxonomy entries: {len(taxonomy_df)}")
    print(f"\n📗 Loading requirements: {requirements_csv}")
    req_df = pd.read_csv(requirements_csv)
    # Rename for convenience
    if "Requirement Statement" in req_df.columns:
        req_df.rename(columns={"Requirement Statement": "requirement_text"}, inplace=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results_flat = []  # for CSV
    results_json = []  # for JSON
    for _, row in tqdm(req_df.iterrows(), total=len(req_df), desc="Mapping"):
        req_text = row["requirement_text"]
        matches = find_best_matches(req_text, taxonomy_df.copy(), model, top_n=top_n)

        # CSV-style (flat) results
        for _, match in matches.iterrows():
            results_flat.append(_create_csv_result_entry(row, match, req_text))

        # JSON-structured results
        results_json.append(_create_json_result_entry(row, matches, req_text))

    # Save CSV
    pd.DataFrame(results_flat).to_csv(output_csv, index=False)
    print(f"💾 CSV mapping results saved to: {output_csv}")
    # Save JSON
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results_json, jf, indent=2, ensure_ascii=False)
    print(f"💾 JSON mapping results saved to: {output_json}")


def classify_match(semantic_score: float, keyword_score: float, combined_score: float,
                   config: AlgorithmConfig = DEFAULT_ALGORITHM_CONFIG) -> str:
    """
    Classifies a match based on provided semantic, keyword, and combined scores.

    The match is categorised into one of three classes: "auto", "review", or "discard".
    The classification depends on thresholds for semantic score, keyword score, and
    combined score being met.

    Parameters:
    semantic_score: float
        The semantic similarity score of the match, typically between 0 and 1.
    keyword_score: float
        The keyword match score of the match, typically between 0 and 100.
    combined_score: float
        The overall score combining various factors, typically between 0 and 1.
    config: AlgorithmConfig
        Algorithm configuration containing weights and thresholds.

    Returns:
    str
        The classification of the match as "auto", "review", or "discard".
    """
    return config.classify_match(semantic_score, keyword_score, combined_score)


def _normalise_mission_dataframe(mission_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the mission DataFrame by renaming specific columns to a standardised format.

    This function takes a DataFrame as input and checks for specific column names.
    If the column "Requirement Statement" exists, it renames it to "requirement_text".
    Additionally, it standardises the column representing an ID by renaming "id" or "ID"
    to "requirement_id". The modified DataFrame is then returned.

    Args:
        mission_df (pd.DataFrame): The input DataFrame containing mission-related data.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns for standardisation.
    """
    if "Requirement Statement" in mission_df.columns:
        mission_df.rename(columns={"Requirement Statement": "requirement_text"}, inplace=True)

    if "id" in mission_df.columns:
        mission_df.rename(columns={"id": "requirement_id"}, inplace=True)
    elif "ID" in mission_df.columns:
        mission_df.rename(columns={"ID": "requirement_id"}, inplace=True)

    return mission_df


def _normalise_project_dataframe(project_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises the column names of a project DataFrame.

    This function renames specific columns in a given DataFrame to standardised
    column names for uniformity and easier processing.

    Args:
        project_df (pd.DataFrame): The input DataFrame containing project data.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    project_df.rename(columns={
        "requirement_text": "req_text",
        "requirement_id": "req_id",
        "requirement_name": "req_name"
    }, inplace=True)
    return project_df


def _compute_match_scores(mission_text: str, project_df: pd.DataFrame,
                          project_embeddings, model) -> tuple:
    """
    Computes match scores between a given mission text and a collection of project
    descriptions using a combination of keyword similarity and semantic similarity.

    Parameters:
    mission_text : str
        The text of the mission or job description.
    project_df : pandas.DataFrame
        A DataFrame containing project information, including a column "req_text"
        that holds the text against which the mission text is to be compared.
    project_embeddings
        Precomputed embeddings for the project texts, used for semantic similarity
        calculations.
    model
        The language model used to encode the mission text into embeddings.

    Returns:
    tuple
        A tuple containing three elements:
        - A list of keyword similarity scores (int) indicating the similarity
          between the mission text and each entry in the project DataFrame based
          on token overlap.
        - A list of semantic similarity scores (float) derived from cos_sim
          comparisons of mission embeddings with the project embeddings.
        - A list of combined scores (float) that blend the semantic similarity
          scores and keyword similarity scores in a weighted score.
    """
    keyword_scores = [
        fuzz.token_set_ratio(mission_text.lower(), t.lower())
        for t in project_df["req_text"]
    ]

    mission_emb = model.encode(mission_text, convert_to_tensor=True)
    sim_scores = util.cos_sim(mission_emb, project_embeddings).squeeze().tolist()

    combined_scores = [
        0.7 * sim + 0.3 * (kw / 100) for sim, kw in zip(sim_scores, keyword_scores)
    ]

    return keyword_scores, sim_scores, combined_scores


def _create_flat_result(mission_id: str, mission_text: str, match: pd.Series,
                        sim_scores: list, keyword_scores: list) -> dict:
    """
    Creates a structured dictionary representing the flattened result of a matching process.

    This function generates a dictionary containing relevant details about the mission and
    project requirements, including their identifiers, text, names, and various associated
    scores resulting from the matching process. It also determines a classification for
    the match based on the given scores.

    Parameters:
        mission_id: str
            Identifier of the mission requirement.
        mission_text: str
            Text content of the mission requirement.
        match: pd.Series
            Series containing information about the matched project requirement.
        sim_scores: list
            List of semantic similarity scores for the project requirements.
        keyword_scores: list
            List of keyword similarity scores for the project requirements.

    Returns:
        dict: A dictionary containing the flattened matching result with combined, semantic,
              and keyword scores, along with classification and details about the mission
              and project requirements.
    """
    return {
        "mission_requirement_id": mission_id,
        "mission_requirement_text": mission_text,
        "project_requirement_id": match["req_id"],
        "project_requirement_name": match["req_name"],
        "combined_score": round(match["combined_score"], 3),
        "semantic_score": round(sim_scores[match.name], 3),
        "keyword_score": round(keyword_scores[match.name], 1),
        "classification": classify_match(
            semantic_score=round(sim_scores[match.name], 3),
            keyword_score=round(keyword_scores[match.name], 1),
            combined_score=round(match["combined_score"], 3)
        )
    }


def _create_json_match(match: pd.Series, sim_scores: list, keyword_scores: list) -> dict:
    """
    Creates a JSON representation of a match, including various scores and classification.

    Args:
        match (pd.Series): A pandas Series object representing a single match, including
            its details such as requirement ID, requirement name, and combined score.
        sim_scores (list): A list of semantic similarity scores where each score corresponds
            to a specific match.
        keyword_scores (list): A list of keyword matching scores where each score corresponds
            to a specific match.

    Returns:
        dict: A dictionary containing the match details with corresponding scores and classification.
    """
    return {
        "project_requirement_id": match["req_id"],
        "project_requirement_name": match["req_name"],
        "combined_score": round(match["combined_score"], 3),
        "semantic_score": round(sim_scores[match.name], 3),
        "keyword_score": round(keyword_scores[match.name], 1),
        "classification": classify_match(
            round(sim_scores[match.name], 3),
            round(keyword_scores[match.name], 1),
            round(match["combined_score"], 3)
        )
    }


def map_requirements_to_requirements(
        mission_csv: str,
        project_csv: str,
        output_csv: str,
        output_json: str,
        top_n: int = 3
) -> None:
    """
    Maps mission phase requirements to project requirements based on semantic similarity and keyword
    matching scores. Outputs the mapping results in both CSV and JSON formats.

    Arguments:
        mission_csv (str): Path to the input CSV containing mission phase requirements.
        project_csv (str): Path to the input CSV containing project requirements.
        output_csv (str): Path to the output CSV file where flat mapping results will be saved.
        output_json (str): Path to the output JSON file where nested mapping results will be saved.
        top_n (int): The number of top matching project requirements to retrieve for each mission
        requirement. Defaults to 3.

    Raises:
        FileNotFoundError: If the provided file paths for mission_csv or project_csv do not exist.
        ValueError: If the input CSV files do not contain the expected columns.

    Returns:
        None
    """

    print(f"\n📘 Loading Mission Phase Requirements: {mission_csv}")
    mission_df = pd.read_csv(mission_csv)
    mission_df = _normalise_mission_dataframe(mission_df)

    print(f"📗 Loading Project Requirements: {project_csv}")
    project_df = pd.read_csv(project_csv)
    project_df = _normalise_project_dataframe(project_df)
    print(f"✅ Project requirements loaded: {len(project_df)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    project_embeddings = model.encode(project_df["req_text"].tolist(), convert_to_tensor=True)

    results_flat = []
    results_json = []

    for _, mission_row in tqdm(mission_df.iterrows(), total=len(mission_df), desc="Mapping"):
        mission_text = mission_row["requirement_text"]
        mission_id = mission_row.get("requirement_id", "")

        keyword_scores, sim_scores, combined_scores = _compute_match_scores(
            mission_text, project_df, project_embeddings, model
        )

        project_df["combined_score"] = combined_scores
        top_matches = project_df.nlargest(top_n, "combined_score")

        for _, match in top_matches.iterrows():
            results_flat.append(_create_flat_result(
                mission_id, mission_text, match, sim_scores, keyword_scores
            ))

        results_json.append({
            "mission_requirement_id": mission_id,
            "mission_requirement_text": mission_text,
            "matches": [
                _create_json_match(m, sim_scores, keyword_scores)
                for _, m in top_matches.iterrows()
            ],
        })

    pd.DataFrame(results_flat).to_csv(output_csv, index=False)
    print(f"💾 CSV results written to {output_csv}")

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results_json, jf, indent=2, ensure_ascii=False)
    print(f"💾 JSON results written to {output_json}")


def _validate_taxonomy_mode(mission_requirements, taxonomy, csv_output, json_output):
    """Validates required parameters for taxonomy mode."""
    missing = []
    if not mission_requirements:
        missing.append("mission_requirements")
    if not taxonomy:
        missing.append("taxonomy")
    if not csv_output:
        missing.append("csv_output")
    if not json_output:
        missing.append("json_output")

    if missing:
        raise ValueError(
            f"taxonomy mode requires: {', '.join(missing)}"
        )


def _validate_reqmap_mode(mission_requirements, project_requirements, csv_output, json_output):
    """Validates required parameters for reqmap mode."""
    missing = []
    if not mission_requirements:
        missing.append("mission_requirements")
    if not project_requirements:
        missing.append("project_requirements")
    if not csv_output:
        missing.append("csv_output")
    if not json_output:
        missing.append("json_output")

    if missing:
        raise ValueError(
            f"reqmap mode requires: {', '.join(missing)}"
        )


def run_mapping(mode="taxonomy", mission_requirements=None, taxonomy=None,
                project_requirements=None, top_n=3, csv_output=None, json_output=None):
    """
    Runs the mapping process based on the specified mode. The function supports two
    modes: 'taxonomy' and 'reqmap'. Depending on the chosen mode, it performs
    requirement mappings either using taxonomy or between two sets of requirements.
    Appropriate files for input and output need to be specified according to the mode.
    Args:
        mode (str): The operational mode of the function. It must be either 'taxonomy'
            or 'reqmap'. Defaults to 'taxonomy'.
        mission_requirements: A file or structure containing the mission requirements.
        taxonomy: A file or structure containing the taxonomy data needed for mapping
            when running in 'taxonomy' mode.
        project_requirements: A file or structure containing project-specific
            requirements for mapping when running in 'reqmap' mode.
        top_n (int): The number of top matches to consider during the mapping.
            Defaults to 3.
        csv_output: The file where the mapping results will be output in CSV format.
        json_output: The file where the mapping results will be output in JSON format.
    Raises:
        ValueError: If required arguments for the selected mode are missing or if
            an invalid mode is specified.
    """
    mode_config = {
        "taxonomy": {
            "validator": lambda: _validate_taxonomy_mode(
                mission_requirements, taxonomy, csv_output, json_output
            ),
            "executor": lambda: map_requirements(
                mission_requirements, taxonomy, csv_output, json_output, top_n
            )
        },
        "reqmap": {
            "validator": lambda: _validate_reqmap_mode(
                mission_requirements, project_requirements, csv_output, json_output
            ),
            "executor": lambda: map_requirements_to_requirements(
                mission_requirements, project_requirements, csv_output, json_output, top_n
            )
        }
    }

    if mode not in mode_config:
        raise ValueError(
            f"Unknown mode: '{mode}'. Must be one of: {', '.join(mode_config.keys())}"
        )

    config = mode_config[mode]
    config["validator"]()
    config["executor"]()


if __name__ == "__main__":
    TAXONOMY_MAPPING_CONFIG.execute()
    REQMAP_MAPPING_CONFIG.execute()
