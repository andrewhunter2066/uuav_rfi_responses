#!/usr/bin/env python3
"""
merge_rfi_requirements_pseudonym.py
-----------------------------------
Merge SME RFI input CSVs with pseudonymization, validation, and priority scoring.
Option A: Pseudonymization (reversible for follow-up).
"""
import os
import glob
from typing import Optional

import pandas as pd

# --- Configuration ---
INPUT_DIR = "./input"
OUTPUT_CSV = "./derived/requirements_rfi_collation.csv"
PSEUDONYM_FILE = "./secure/sme_mapping.csv"
DERIVED_DIR = "./derived"
EXPORT_JSONLD = True
EXPORT_NEO4J = True
# Priority score calculation constants
IMPORTANCE_WEIGHT = 0.5
FREQUENCY_WEIGHT = 0.4
FEASIBILITY_WEIGHT = 0.1
DEFAULT_IMPORTANCE = 0
DEFAULT_FREQUENCY = 0
DEFAULT_FEASIBILITY = 3
SCORE_DECIMAL_PLACES = 2


def is_valid_name(name: str) -> bool:
    """
    Checks if a name is valid (not empty, not NaN).

    :param name: The name to validate.
    :type name: str
    :return: True if the name is valid, False otherwise.
    :rtype: bool
    """
    return not (pd.isna(name) or name.strip() == "")


def get_pseudonym(name: str, mapping: dict) -> Optional[str]:
    """
    Generates a pseudonym for a given name based on a mapping. If the name is already
    present in the mapping, the corresponding pseudonym is returned. If the name is
    not in the mapping, a new pseudonym is generated, added to the mapping, and
    returned. If the given name is empty or NaN, None is returned.

    :param name: The original name for which a pseudonym needs to be generated.
    :type name: str
    :param mapping: A dictionary that holds the mapping between the original names
        and their pseudonyms.
    :type mapping: dict
    :return: The pseudonym corresponding to the given name, or None if the name is
        empty or NaN.
    :rtype: Optional[str]
    """
    if not is_valid_name(name):
        return None
    if name in mapping:
        return mapping[name]
    pseudonym = f"SME{len(mapping) + 1:03d}"
    mapping[name] = pseudonym
    return pseudonym


def compute_priority_score(row: dict) -> Optional[float]:
    """
    Computes the priority score for a given row by evaluating its importance,
    frequency, and feasibility. The computation uses a weighted formula where
    importance contributes 50%, frequency contributes 40%, and feasibility
    contributes 10%. The score is then rounded to two decimal places. If any
    unexpected error occurs, returns None.
    :param row: Dictionary containing attributes 'Importance', 'Frequency', and
                'Feasibility'. All attributes are expected to be numeric or
                convertible to float.
    :type row: dict
    :return: A computed priority score rounded to two decimals, or None if an
             error occurs during computation.
    :rtype: float or None
    """
    try:
        importance = float(row.get("Importance", DEFAULT_IMPORTANCE))
        frequency = float(row.get("Frequency", DEFAULT_FREQUENCY))
        feasibility = float(row.get("Feasibility", DEFAULT_FEASIBILITY))

        weighted_score = (
                importance * IMPORTANCE_WEIGHT +
                frequency * FREQUENCY_WEIGHT +
                feasibility * FEASIBILITY_WEIGHT
        )

        return round(weighted_score, SCORE_DECIMAL_PLACES)
    except Exception:
        return None


def ensure_output_directories() -> None:
    """
    Ensures that required output directories exist before proceeding with further
    operations. This function creates the directories if they do not already
    exist, using the predefined paths for derived and pseudonym files.

    :raises OSError: If the directory creation fails for any reason.
    :return: None
    """
    os.makedirs(DERIVED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PSEUDONYM_FILE), exist_ok=True)


def load_and_merge_csv_files() -> pd.DataFrame:
    """
    Load and combine multiple CSV files into one unified DataFrame. This function searches for
    all CSV files within a specified input directory, loads their contents into individual
    DataFrames, strips leading and trailing spaces from column names, and adds a new column
    indicating the source file name for traceability. Finally, it concatenates all these
    DataFrames into a single DataFrame and returns it.

    :raises FileNotFoundError: If no CSV files are found in the specified input directory.
    :return: A unified DataFrame containing the data from all CSV files within the
        specified input directory.
    :rtype: pd.DataFrame
    """
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    all_frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df["SourceFile"] = os.path.basename(f)
        all_frames.append(df)

    return pd.concat(all_frames, ignore_index=True)


def load_pseudonym_mapping() -> dict:
    """
    Loads a mapping of pseudonyms from a CSV file. The method attempts to read
    a predefined file containing SME names and their corresponding pseudonyms.
    If the file exists, the method reads it, strips any leading or trailing
    spaces in column names, and creates a dictionary mapping SME names to their
    pseudonyms. If the file does not exist, an empty dictionary is returned.

    :return: A dictionary where keys are SME names and values are their
        corresponding pseudonyms.
    :rtype: dict
    """
    if os.path.exists(PSEUDONYM_FILE):
        mapping_df = pd.read_csv(PSEUDONYM_FILE)
        mapping_df.columns = mapping_df.columns.str.strip()
        return dict(zip(mapping_df["SMEName"], mapping_df["Pseudonym"]))
    return {}


def pseudonymize_sme_data(merged: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Pseudonymizes SME (Subject Matter Expert) data by replacing the SME names with pseudonyms.
    The function processes a DataFrame by applying pseudonymization to a specific column and updates
    the mapping file used for the pseudonymization process. It removes certain columns deemed unnecessary
    for the post-processed data.

    :param merged: A Pandas DataFrame containing SME data. This is the main dataset
        requiring pseudonymization.
    :type merged: pd.DataFrame

    :param mapping: A dictionary mapping SME names to pseudonyms. It serves as a reference for
        assigning or retrieving pseudonyms during processing.
    :type mapping: dict

    :return: A Pandas DataFrame with pseudonymized "SME_ID" data and modified structure after dropping
        unnecessary columns.
    :rtype: pd.DataFrame
    """
    merged["SME_ID"] = merged["SMEName"].apply(lambda name: get_pseudonym(name, mapping))
    merged.drop(columns=["SMEName", "SourceFile"], inplace=True)

    # Update mapping CSV
    mapping_df = pd.DataFrame(list(mapping.items()), columns=["SMEName", "Pseudonym"])
    mapping_df.to_csv(PSEUDONYM_FILE, index=False)

    return merged


def validate_merged_data(merged: pd.DataFrame) -> list:
    """
    Validates the merged DataFrame to ensure that it complies with specific requirements.

    This function performs several checks on the provided DataFrame, such as identifying
    missing values in specific columns, detecting duplicate entries for RequirementID,
    and ensuring that required fields like Priority and Importance are not blank.
    It returns a list of error messages if any issues are found.

    :param merged: The merged DataFrame that contains the data to be validated. It is
        expected to have the columns 'RequirementID', 'Priority', and 'Importance'.
    :type merged: pd.DataFrame
    :return: A list of string messages describing validation errors found in the
        DataFrame, if any.
    :rtype: list
    """
    errors = []

    # Missing RequirementID
    if merged["RequirementID"].isnull().any():
        errors.append("Some rows have missing RequirementID.")

    # Duplicate RequirementID
    duplicates = merged[merged["RequirementID"].duplicated(keep=False)]
    if not duplicates.empty:
        errors.append(f"Duplicate RequirementIDs: {', '.join(duplicates['RequirementID'].astype(str).unique())}")

    # Blank Priority
    blank_priority = merged[merged["Priority"].isnull() | (merged["Priority"].astype(str).str.strip() == "")]
    if not blank_priority.empty:
        errors.append(f"{len(blank_priority)} rows have blank Priority.")

    # Blank Importance
    blank_importance = merged[merged["Importance"].isnull()]
    if not blank_importance.empty:
        errors.append(f"{len(blank_importance)} rows have blank Importance.")

    return errors


def apply_default_values(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Applies default values to specific columns in a dataframe.

    This function ensures that certain columns within the provided dataframe
    have default values. If a column does not already exist or contain a
    value, a default will be assigned to it. Default values include
    specific data for the following columns: 'Feasibility', 'Status',
    'Version', and 'ChangeNote'. This function modifies the dataframe
    in-place and returns the updated dataframe.

    :param merged: A pandas DataFrame object. This is the dataframe on
        which default values will be applied.
    :return: The updated pandas DataFrame with default values added where
        applicable.
    """
    merged["Feasibility"] = merged.get("Feasibility", 3)
    merged["Status"] = "Draft"
    merged["Version"] = "v0.1"
    merged["ChangeNote"] = "Initial merge"
    return merged


def export_jsonld(merged: pd.DataFrame) -> None:
    """
    Exports a pandas DataFrame to a JSON-LD file format. The function uses a
    pre-defined context to specify vocabularies and mappings for the data,
    allowing for semantic representation. The converted JSON-LD data contains a
    graph of records generated from the DataFrame.

    :param merged: The pandas DataFrame containing data to be converted
        into JSON-LD format.
    :type merged: pd.DataFrame
    :return: None
    """
    import json

    context = {
        "@vocab": "https://example.org/rfi#",
        "RequirementID": "rfi:RequirementID",
        "RequirementStatement": "rfi:RequirementStatement",
        "SME_ID": "rfi:SME_ID",
        "PriorityScore": "rfi:PriorityScore",
        "Priority": "rfi:Priority"
    }
    jsonld_data = {"@context": context, "@graph": merged.to_dict(orient="records")}

    with open(os.path.join(DERIVED_DIR, "requirements_rfi_collation.jsonld"), "w") as f:
        json.dump(jsonld_data, f, indent=2)

    print("JSON-LD export complete.")


def export_neo4j(merged: pd.DataFrame) -> None:
    """
    Exports data into CSV files formatted for Neo4j import. This function creates two separate CSV files:
    one containing nodes representing requirements and another containing relationships representing the
    proposals made by SMEs for these requirements.

    :param merged: A pandas DataFrame containing columns such as `RequirementID`, `RequirementStatement`,
                   `PriorityScore`, and `SME_ID`. This data will be transformed and written to Neo4j-compatible
                   CSV files.
    :return: None
    """
    # Nodes: Requirements
    nodes = merged[["RequirementID", "RequirementStatement", "PriorityScore"]].drop_duplicates()
    nodes[":LABEL"] = "Requirement"
    nodes.rename(columns={
        "RequirementID": "id:ID",
        "RequirementStatement": "title",
        "PriorityScore": "priority_score"
    }, inplace=True)
    nodes.to_csv(os.path.join(DERIVED_DIR, "nodes_requirements.csv"), index=False)

    # Relationships: SME -> Requirement
    rels = merged[["RequirementID", "SME_ID"]].dropna()
    rels[":TYPE"] = "PROPOSED_BY"
    rels.rename(columns={"RequirementID": ":START_ID", "SME_ID": ":END_ID"}, inplace=True)
    rels.to_csv(os.path.join(DERIVED_DIR, "relationships_sme.csv"), index=False)

    print("Neo4j CSV export complete.")


def main() -> None:
    """
    Main function for processing and managing SME data and requirement priorities.

    This function orchestrates the entire workflow:
    1. Sets up output directories
    2. Loads and merges SME data from CSV files
    3. Pseudonymizes SME names for anonymity
    4. Validates the merged dataset
    5. Applies default values and computes priority scores
    6. Saves processed data to CSV
    7. Optionally exports to JSON-LD and Neo4j formats

    :return: None
    """
    # Ensure output directories exist
    ensure_output_directories()

    # Load and merge SME CSVs
    merged = load_and_merge_csv_files()

    # Pseudonymize SME Names
    mapping = load_pseudonym_mapping()
    merged = pseudonymize_sme_data(merged, mapping)

    # Compute priority scores
    merged["PriorityScore"] = merged.apply(compute_priority_score, axis=1)

    # Validate merged data
    errors = validate_merged_data(merged)
    if errors:
        print("\n".join(errors))
        # Optionally, you could exit here if you want to enforce strict validation
        # import sys; sys.exit(1)
    else:
        print("Validation passed.")

    # Set defaults for missing synthesis columns
    merged = apply_default_values(merged)

    # Save merged collation CSV
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Collated CSV saved: {OUTPUT_CSV}")

    # Optional Graph Exports
    if EXPORT_JSONLD or EXPORT_NEO4J:
        if EXPORT_JSONLD:
            export_jsonld(merged)
        if EXPORT_NEO4J:
            export_neo4j(merged)


if __name__ == "__main__":
    main()
