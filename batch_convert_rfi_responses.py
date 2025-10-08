#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import re
import json
from datetime import datetime

# --- Define constants ---
QUESTION_HEADERS = {
    "Q1": "Q1 - What test scenarios and supporting data does the system  need to provide to support mission planners "
          "in evaluating route options, based on calculable or estimable factors (e.g., time, usage)?",
    "Q2": "Q2 - What estimated factors should be considered?",
    "Q3": "Q3 - What guidance can be provided regarding the estimation approach?",
    "Q4": "Q4 - What test scenarios and supporting data does the system  need to provide to support mission planners "
          "in evaluating risks for a route?",
    "Q5": "Q5 - When evaluating risks, what specific risks should the system flag?",
    "Q6": "Q6 - What test scenarios and supporting data does the system  need to provide to support mission planners "
          "in recording mission planning  data for future review, reuse,  or refinement?"
}

FIELD_OBSERVATIONS_HEADERS = {
    "Q1": "Task 1 - Report of current risk profile performance and management systems effectiveness",
    "Q2": "Task 2 - Analyse vehicular performance profiles during adverse weather, environmental or restricted "
          "waterways.",
    "Q3": "Task 3 - Investigate the quality of sounding operations during mission execution",
    "Q4": "Task 4 - Capture 'OFFICIAL' level task identifiers and missions plans",
    "Q5": "Task 5 - Report risk deficiencies or notifiable incidents (i.e. comms, inoperable UUV, etc.)",
    "Q6": "Task 6 - Are the current standards adequate for format and capacity?"
}

QUESTION_TYPE_MAP = {
    "Q1": "Route Evaluation",
    "Q2": "Route Evaluation",
    "Q3": "Route Evaluation",
    "Q4": "Risk Evaluation",
    "Q5": "Risk Evaluation",
    "Q6": "Data Records"
}

# File locations
INPUT_FOLDER = "C:/Users/Andrew/Documents/GitHub/RAN/rfi/rfi_data/cleaned_ran_rfi_responses"
OUTPUT_FOLDER = "input/normalised_individual_responses"
OUTPUT_CSV = "input/normalised_all_responses.csv"
OUTPUT_JSON = "input/normalised_all_responses.json"

# Scenario parsing patterns and constants
SCENARIO_PATTERN = r"Scenario\s*([0-9]+)\s*[:\-]?\s*(.*)"
FIELD_OBSERVATIONS_PATTERN = r"Field Observations - User Input"
FIELD_OBSERVATIONS_NUMBER = "4"
FIELD_OBSERVATIONS_LABEL = "Field Observations - User Input"
UNKNOWN_SCENARIO_NUMBER = "Unknown"

# Constants for sheet type detection
FIELD_KEYWORD = "field"
OBSERVATION_KEYWORD = "observation"

# Define constants at module level
NORMALIZED_COLUMNS_ORDER = [
    "Respondent", "Scenario", "ScenarioNumber", "ScenarioLabel",
    "Question", "QuestionType", "ResponseText",
    "SourceDate", "FollowupNeeded", "Status", "Version", "ChangeNote"
]

RESPONDENT_MAPPING_FILE = Path("secure/sme_mapping.csv")


class PseudonymMapper:
    """
    Manages pseudonym mappings for respondent identifiers, providing functionality
    to load, generate, persist, and apply pseudonyms to datasets.
    """

    def __init__(self, mapping_file_path: Path):
        """
        Initialise the pseudonym mapper with a mapping file path.

        :param mapping_file_path: Path to the CSV file storing pseudonym mappings
        """
        self.mapping_file_path = Path(mapping_file_path)
        self._mapping = {}
        self._next_index = 1
        self._load_existing_mappings()

    def _load_existing_mappings(self) -> None:
        """Load existing pseudonym mappings from a file if it exists."""
        self.mapping_file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.mapping_file_path.exists():
            existing_df = pd.read_csv(self.mapping_file_path, dtype=str)
            self._mapping = existing_df.set_index("Respondent")["Pseudonym"].to_dict()
            self._next_index = len(self._mapping) + 1
        else:
            self._mapping = {}
            self._next_index = 1

    def get_or_create_pseudonym(self, respondent: str) -> str:
        """
        Get an existing pseudonym for a respondent or create a new one.

        :param respondent: Original respondent identifier
        :return: Pseudonym for the respondent
        """
        if respondent not in self._mapping:
            pseudonym = f"R{self._next_index:03d}"
            self._mapping[respondent] = pseudonym
            self._next_index += 1

        return self._mapping[respondent]

    def generate_new_mappings(self, respondents: pd.Series) -> list[dict]:
        """
        Generate new pseudonym mappings for respondents not already in the mapping.

        :param respondents: Series of respondent identifiers
        :return: List of new mapping records with Respondent, Pseudonym, and Created fields
        """
        new_mappings = []

        for respondent in respondents.unique():
            if respondent not in self._mapping:
                pseudonym = self.get_or_create_pseudonym(respondent)
                new_mappings.append({
                    "Respondent": respondent,
                    "Pseudonym": pseudonym,
                    "Created": datetime.now().strftime("%Y-%m-%d")
                })

        return new_mappings

    def save_mappings(self, new_mappings: list[dict]) -> None:
        """
        Persist new mappings to the mapping file.

        :param new_mappings: List of new mapping records to append
        """
        if not new_mappings:
            print("No new respondents to map.")
            return

        # Load existing data or create empty DataFrame
        if self.mapping_file_path.exists():
            existing_df = pd.read_csv(self.mapping_file_path, dtype=str)
        else:
            existing_df = pd.DataFrame(columns=["Respondent", "Pseudonym", "Created"])

        # Append new mappings
        new_df = pd.DataFrame(new_mappings)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(self.mapping_file_path, index=False)

        print(f"Updated mapping file with {len(new_mappings)} new entries.")

    def apply_to_dataframe(self, df: pd.DataFrame, column: str = "Respondent") -> pd.DataFrame:
        """
        Apply pseudonym mappings to a dataframe column.

        :param df: DataFrame containing respondent data
        :param column: Name of the column to pseudonymize
        :return: DataFrame with pseudonymized column
        """
        df[column] = df[column].map(self._mapping)
        return df


def parse_scenario(scenario_text: str) -> tuple[str, str]:
    """
    Parses a scenario text string to extract a scenario number and label. The function aims to identify pattern-matched
    scenario text. If no match is found with predefined patterns, it returns default values.

    :param scenario_text: str
        The input string representing the scenario description, which may include both the scenario number and the
        label.

    :return: tuple
        Returns a tuple `(number, label)`. The `number` corresponds to the scenario identifier or "Unknown" if it cannot
        be determined. The `label` is the descriptive text of the scenario, extracted from the input string, or a
        default label if not present.
    """
    normalised_text = scenario_text.strip()

    # Try to match a standard scenario pattern
    scenario_match = re.match(SCENARIO_PATTERN, normalised_text, re.IGNORECASE)
    if scenario_match:
        number = scenario_match.group(1).strip()
        label = scenario_match.group(2).strip() if scenario_match.group(2) else f"Scenario {number}"
        return number, label

    # Try to match a field observations pattern
    if re.match(FIELD_OBSERVATIONS_PATTERN, normalised_text, re.IGNORECASE):
        return FIELD_OBSERVATIONS_NUMBER, FIELD_OBSERVATIONS_LABEL

    # Return unknown scenario with original text as label
    return UNKNOWN_SCENARIO_NUMBER, normalised_text


def _is_field_observation_sheet(sheet_name: str) -> bool:
    """
    Determines if a sheet is a field observation sheet based on its name.

    :param sheet_name: The name of the sheet to check
    :type sheet_name: str
    :return: True if the sheet is a field observation sheet, False otherwise
    :rtype: bool
    """
    normalised_sheet_name = sheet_name.lower().strip()
    return FIELD_KEYWORD in normalised_sheet_name and OBSERVATION_KEYWORD in normalised_sheet_name


def get_question_headers(sheet_name: str) -> dict[str, str]:
    """
    Retrieves a set of headers based on the provided sheet name. The function determines
    which headers to return by evaluating specific keywords in the sheet name. If the
    sheet name includes both "field" and "observation", it returns a predefined set of
    field observation headers. Otherwise, it returns a generic set of question headers.

    :param sheet_name: The name of the sheet to process
    :type sheet_name: str
    :return: A predefined set of headers corresponding to the specified sheet name
    :rtype: dict[str, str]
    """
    if _is_field_observation_sheet(sheet_name):
        return FIELD_OBSERVATIONS_HEADERS
    else:
        return QUESTION_HEADERS


def detect_question_type(qid: str) -> str:
    """
    Detect the type of question based on the given question ID.

    This function uses a predefined mapping (`QUESTION_TYPE_MAP`) to retrieve the type
    of question corresponding to its ID. If the ID is not found in the map, the function
    returns "Unknown".

    :param qid: The question ID for which the question type is to be determined.
    :type qid: str
    :return: The detected question type or "Unknown" if the ID is not found in the map.
    :rtype: str
    """
    return QUESTION_TYPE_MAP.get(qid, "Unknown")


def _extract_source_date(input_path: Path) -> str:
    """Extract the source date from file modification time."""
    file_mtime = Path(input_path).stat().st_mtime
    return datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")


def _get_respondent_name(respondent_series: pd.Series, row_index: int, fallback_name: str) -> str:
    """Get the respondent name from the series or use fallback."""
    respondent_val = respondent_series.iloc[row_index - 1]
    if pd.notna(respondent_val):
        return str(respondent_val).strip()
    return fallback_name


def _create_response_record(respondent: str, sheet_name: str, scenario_number: str, scenario_label: str,
                            question: str, question_type: str, response_text: str, source_date: str) -> dict:
    """Create a standardised response record dictionary."""
    return {
        "Respondent": respondent,
        "Scenario": sheet_name,
        "ScenarioNumber": scenario_number,
        "ScenarioLabel": scenario_label,
        "Question": question,
        "QuestionType": question_type,
        "ResponseText": response_text,
        "SourceDate": source_date,
        "FollowupNeeded": "",
        "Status": "Draft",
        "Version": "v0.1",
        "ChangeNote": "Initial merge",
    }


def _process_sheet_responses(df: pd.DataFrame, sheet_name: str, scenario_number: str, scenario_label: str,
                             question_headers: dict[str, str], source_date: str, fallback_name: str) -> list[dict]:
    """
    Processes the responses from a specified sheet and compiles them into a list of structured records.

    The method parses a DataFrame representing the worksheet, extracts responses based on predefined question
    headers, and creates a response record for each respondent and question-answer pair. It accounts for
    cell formatting, missing responses, and labels when constructing the output.

    :param df: The DataFrame that contains worksheet data.
    :type df: pd.DataFrame
    :param sheet_name: The name of the worksheet being processed.
    :param scenario_number: A string identifier for the scenario being analysed.
    :param scenario_label: Descriptive label for the scenario.
    :param question_headers: A dictionary where keys are question IDs and values are corresponding
        question details.
    :param source_date: The date the source data was received.
    :param fallback_name: The fallback name to use when a respondent name cannot be resolved.
    :return: A list of dictionaries, where each dictionary corresponds to a structured response record.
    :rtype: list[dict]
    """
    rows = []
    respondent_col = 2
    respondent_series = df.iloc[:, respondent_col]
    question_cols = list(range(3, 9))  # columns D – I (0-based indices)

    for col_index, qid in zip(question_cols, question_headers.keys()):
        question = question_headers[qid]
        question_type = detect_question_type(qid)
        responses = df.iloc[5:, col_index]

        for row_index, cell in enumerate(responses, start=6):
            if pd.isna(cell) or str(cell).strip() == "":
                break

            response_text = str(cell).strip()
            respondent = _get_respondent_name(respondent_series, row_index, fallback_name)

            record = _create_response_record(
                respondent, sheet_name, scenario_number, scenario_label,
                question, question_type, response_text, source_date
            )
            rows.append(record)

    return rows


def normalize_rfi_excel(input_path: Path) -> list[dict]:
    """
    Normalise responses from an Excel-based RFI (Request For Information) data file by
    extracting scenarios, questions, respondents, and responses. This function processes
    each sheet in the provided Excel file (skipping instructional sheets) and aggregates
    the data into a structured tabular format.
    :param input_path: Path to the input Excel file containing RFI data.
    :type input_path: Path
    :return: A list of dictionaries, where each dictionary represents a single response
        from the Excel file with associated metadata such as respondent, scenarios, question,
        and other details.
    :rtype: list[dict]
    """
    xl = pd.ExcelFile(input_path)
    all_rows = []

    source_date = _extract_source_date(input_path)
    fallback_name = Path(input_path).stem

    for sheet_name in xl.sheet_names[1:]:  # skip instructions
        df = xl.parse(sheet_name, header=None)
        scenario_number, scenario_label = parse_scenario(sheet_name)
        question_headers = get_question_headers(sheet_name)

        sheet_rows = _process_sheet_responses(
            df, sheet_name, scenario_number, scenario_label,
            question_headers, source_date, fallback_name
        )
        all_rows.extend(sheet_rows)

    return all_rows


def _process_excel_files(input_folder: Path, output_folder: Path, per_respondent: bool) -> list[dict]:
    """
    Process all Excel files in the input folder and normalise their data.

    :param input_folder: Path to folder containing Excel files
    :param output_folder: Path to folder for saving individual respondent CSVs
    :param per_respondent: Whether to save individual CSV files per respondent
    :return: List of all normalised response records
    """
    all_data = []
    excel_files = list(input_folder.glob("*.xlsx"))

    if not excel_files:
        print(f"No Excel files found in {input_folder}")
        return all_data

    for file in excel_files:
        print(f"Processing: {file.name}")
        rows = normalize_rfi_excel(file)
        all_data.extend(rows)

        if per_respondent:
            out_csv = output_folder / f"{file.stem}_normalised.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  └─ Saved individual CSV → {out_csv.name}")

    return all_data


def _create_response_id_column(df: pd.DataFrame) -> None:
    """
    Creates and populates the ResponseID column with sequential integers starting from 1.

    :param df: The DataFrame to modify in-place.
    :type df: pd.DataFrame
    :return: None
    """
    df.insert(0, "ResponseID", range(1, len(df) + 1))


def _get_next_available_id(df: pd.DataFrame) -> int:
    """
    Determines the next available ResponseID based on the maximum existing ID.

    :param df: The DataFrame containing ResponseID column.
    :type df: pd.DataFrame
    :return: The next available integer ID (max_id + 1, or 1 if no valid IDs exist).
    :rtype: int
    """
    max_id = df["ResponseID"].max()
    return 1 if pd.isna(max_id) else int(max_id) + 1


def _fill_missing_response_ids(df: pd.DataFrame) -> None:
    """
    Identifies and fills missing ResponseID values with unique sequential integers.
    Prints the number of missing IDs that were populated.

    :param df: The DataFrame to modify in-place.
    :type df: pd.DataFrame
    :return: None
    """
    missing_ids = df["ResponseID"].isna()
    if not missing_ids.any():
        return

    next_id = _get_next_available_id(df)
    num_missing = missing_ids.sum()
    new_ids = range(next_id, next_id + num_missing)

    df.loc[missing_ids, "ResponseID"] = list(new_ids)
    print(f"Populated {num_missing} missing ResponseID(s)")


def _ensure_response_id_integrity(df: pd.DataFrame) -> None:
    """
    Ensures ResponseID column exists, fills missing values, and converts to integer type.

    :param df: The DataFrame to modify in-place.
    :type df: pd.DataFrame
    :return: None
    """
    if "ResponseID" not in df.columns:
        _create_response_id_column(df)
    else:
        _fill_missing_response_ids(df)
        df["ResponseID"] = df["ResponseID"].astype(int)


def add_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a unique "ResponseID" to the provided DataFrame and ensures its consistency.
    If the "ResponseID" column does not exist, it will be created and populated with
    unique sequential integers starting from 1. If existing "ResponseID" entries have
    missing values, those will be filled with unique values based on the maximum
    existing "ResponseID". Additionally, an "id" column is added, corresponding
    to the 1-based index of the DataFrame.

    :param df: The input DataFrame to be modified.
    :type df: pd.DataFrame
    :return: The modified DataFrame with the ensured "ResponseID" and added "id" column.
    :rtype: pd.DataFrame
    """
    _ensure_response_id_integrity(df)
    return df


def _save_consolidated_data(df: pd.DataFrame, output_csv: Path, output_json: Path = None) -> None:
    """
    Saves the consolidated and normalized DataFrame to specified CSV file and optionally to a
    JSON file. The DataFrame is first processed with an identifier added. Once processed, it
    is saved as a CSV file to the specified path. If an optional JSON output path is provided,
    the DataFrame is also serialized and saved in JSON format.

    :param df: DataFrame to be processed and saved
    :type df: pd.DataFrame
    :param output_csv: Path to save the processed DataFrame as a CSV file
    :type output_csv: Path
    :param output_json: Optional path to save the processed DataFrame as a JSON file
    :type output_json: Path, optional
    :return: None
    :rtype: None
    """
    df = add_id(df)
    df.to_csv(output_csv, index=False)
    print(f"Saved combined normalised CSV → {output_csv}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        print(f"Saved combined JSON → {output_json}")


def batch_normalise_rfi(input_folder: str,
                        output_folder: str,
                        output_csv: str,
                        output_json=None,
                        per_respondent=False) -> None:
    """
    Processes and normalises RFI (Request for Information) data from Excel files in the specified input folder.
    The function supports saving normalised data for each respondent into separate CSV files if
    the `per_respondent` flag is set. Additionally, it creates a consolidated CSV file and an optional
    JSON file with all the normalised and anonymised data combined. The respondent names are pseudonymized
    for privacy, and the mapping file is saved securely.

    :param input_folder: Path to the folder containing Excel files for processing.
    :param output_folder: Path to the folder to save processed files.
    :param output_csv: Path to save the consolidated normalised CSV file.
    :param output_json: (Optional) Path to save the consolidated JSON file. If not provided, JSON saving is skipped.
    :param per_respondent: Boolean flag. If True, saves normalised data for each respondent into individual CSV files.
    :return: None
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_data = _process_excel_files(input_folder, output_folder, per_respondent)

    if not all_data:
        return

    df_all = pd.DataFrame(all_data)
    df_all = df_all[NORMALIZED_COLUMNS_ORDER]

    df_all = pseudonymise_respondents(df_all, RESPONDENT_MAPPING_FILE)

    _save_consolidated_data(df_all, Path(output_csv), Path(output_json) if output_json else None)


def pseudonymise_respondents(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    """
    Pseudonymises the "Respondent" field in a given dataset using a mapping file. If a
    mapping for a respondent does not exist, a new pseudonym is generated, saved to the
    mapping file, and applied to the dataset. Existing mappings will be preserved and
    used when possible.

    :param df: pandas.DataFrame
        Input dataframe containing a "Respondent" column to be pseudonymised.
    :param mapping_path: Path
        File path to the mapping CSV file. This file stores relationships between original
        respondent identifiers and their pseudonyms. The directory will be created if
        it does not exist.
    :return: pandas.DataFrame
        Modified dataframe where the "Respondent" column has been replaced with
        pseudonyms.
    """
    mapper = PseudonymMapper(mapping_path)
    new_mappings = mapper.generate_new_mappings(df["Respondent"])
    mapper.save_mappings(new_mappings)
    return mapper.apply_to_dataframe(df)


def run_batch_normalization() -> None:
    """
    Entry point for batch RFI normalization with default configuration.
    Processes Excel files from the input folder and generates normalised outputs
    with pseudonymized respondent data.
    """
    batch_normalise_rfi(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        output_csv=OUTPUT_CSV,
        output_json=OUTPUT_JSON,
        per_respondent=True
    )


if __name__ == "__main__":
    run_batch_normalization()
