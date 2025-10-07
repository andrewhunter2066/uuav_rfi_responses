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

# File location
INPUT_FOLDER = "C:/Users/Andrew/Documents/GitHub/RAN/rfi/rfi_data/cleaned_ran_rfi_responses"
OUTPUT_FOLDER = "input/normalised_individual_responses"
OUTPUT_CSV = "input/normalized_all_responses.csv"
OUTPUT_JSON = "input/normalized_all_responses.json"


# --- Helper functions ---
def parse_scenario(scenario_text):
    """Extract scenario number and label."""
    m = re.match(r"Scenario\s*([0-9]+)\s*[:\-]?\s*(.*)", scenario_text.strip(), re.IGNORECASE)
    if m:
        number = m.group(1).strip()
        label = m.group(2).strip() if m.group(2) else f"Scenario {number}"
        return number, label
    fo = re.match(r"Field Observations - User Input", scenario_text.strip(), re.IGNORECASE)
    if fo:
        return "4", "Field Observations - User Input"
    return "Unknown", scenario_text.strip()


def get_question_headers(sheet_name):
    """Return the appropriate question header mapping based on the sheet name."""
    name = sheet_name.lower().strip()
    if "field" in name and "observation" in name:
        return FIELD_OBSERVATIONS_HEADERS
    else:
        return QUESTION_HEADERS


def detect_question_type(qid):
    return QUESTION_TYPE_MAP.get(qid, "Unknown")


def normalize_rfi_excel(input_path):
    """Parse a single respondent workbook into a list of normalised rows."""
    xl = pd.ExcelFile(input_path)
    all_rows = []

    # Get file modification date
    file_mtime = Path(input_path).stat().st_mtime
    source_date = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")

    for sheet_name in xl.sheet_names[1:]:  # skip instructions
        df = xl.parse(sheet_name, header=None)
        scenario_number, scenario_label = parse_scenario(sheet_name)

        # Select question headers depending on sheet type
        question_headers = get_question_headers(sheet_name)

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
                respondent_val = respondent_series.iloc[row_index - 1]
                respondent = (
                    str(respondent_val).strip()
                    if pd.notna(respondent_val)
                    else Path(input_path).stem
                )

                all_rows.append({
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
                })

    return all_rows


def batch_normalize_rfi(input_folder, output_folder, output_csv, output_json=None, per_respondent=False):
    """Process all Excel files in the folder."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    all_data = []

    excel_files = list(input_folder.glob("*.xlsx"))
    if not excel_files:
        print(f"No Excel files found in {input_folder}")
        return

    for file in excel_files:
        print(f"Processing: {file.name}")
        rows = normalize_rfi_excel(file)
        all_data.extend(rows)

        if per_respondent:
            out_csv = output_folder / f"{file.stem}_normalized.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  └─ Saved individual CSV → {out_csv.name}")

    df_all = pd.DataFrame(all_data)
    columns_order = [
        "Respondent", "Scenario", "ScenarioNumber", "ScenarioLabel",
        "Question", "QuestionType", "ResponseText",
        "SourceDate", "FollowupNeeded", "Status", "Version", "ChangeNote"
    ]
    df_all = df_all[columns_order]

    # Anonymize respondent names and persist mapping
    mapping_file = Path("secure/sme_mapping.csv")
    df_all = pseudonymise_respondents(df_all, mapping_file)

    df_all.to_csv(output_csv, index=False)
    print(f"Saved combined normalized CSV → {output_csv}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(df_all.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
        print(f"Saved combined JSON → {output_json}")


def pseudonymise_respondents(df, mapping_path):
    """Replace respondent names with pseudonyms, maintaining a secure mapping."""
    mapping_path = Path(mapping_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing mapping if it exists
    if mapping_path.exists():
        existing_map = pd.read_csv(mapping_path, dtype=str)
    else:
        existing_map = pd.DataFrame(columns=["Respondent", "Pseudonym", "Created"])

    current_map = existing_map.set_index("Respondent")["Pseudonym"].to_dict()
    next_index = len(current_map) + 1

    # Assign new pseudonyms as needed
    new_rows = []
    pseudonyms = []
    for respondent in df["Respondent"].unique():
        if respondent not in current_map:
            pseudonym = f"R{next_index:03d}"
            current_map[respondent] = pseudonym
            new_rows.append({
                "Respondent": respondent,
                "Pseudonym": pseudonym,
                "Created": datetime.now().strftime("%Y-%m-%d")
            })
            next_index += 1

    # Append any new mappings to the file
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated_map = pd.concat([existing_map, new_df], ignore_index=True)
        updated_map.to_csv(mapping_path, index=False)
        print(f"Updated mapping file with {len(new_rows)} new entries.")
    else:
        print("No new respondents to map.")

    # Apply mapping to the main dataframe
    df["Respondent"] = df["Respondent"].map(current_map)

    return df


if __name__ == "__main__":
    batch_normalize_rfi(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        output_csv=OUTPUT_CSV,
        output_json=OUTPUT_JSON,
        per_respondent=True
    )
