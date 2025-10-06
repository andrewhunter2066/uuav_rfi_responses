#!/usr/bin/env python3
"""
merge_rfi_requirements_pseudonym.py
-----------------------------------
Merge SME RFI input CSVs with pseudonymization, validation, and priority scoring.
Option A: Pseudonymization (reversible for follow-up).
"""
import os
import glob
import pandas as pd

# --- Configuration ---
INPUT_DIR = "./input"
OUTPUT_CSV = "./derived/requirements_rfi_collation.csv"
PSEUDONYM_FILE = "./secure/sme_mapping.csv"
DERIVED_DIR = "./derived"
EXPORT_JSONLD = True
EXPORT_NEO4J = True

# Ensure output directories exist
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PSEUDONYM_FILE), exist_ok=True)

# --- Load SME CSVs ---
csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

all_frames = []
for f in csv_files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    df["SourceFile"] = os.path.basename(f)
    all_frames.append(df)

merged = pd.concat(all_frames, ignore_index=True)

# --- Pseudonymize SME Names ---
# Load existing mapping if present
if os.path.exists(PSEUDONYM_FILE):
    mapping_df = pd.read_csv(PSEUDONYM_FILE)
    mapping_df.columns = mapping_df.columns.str.strip()
    mapping = dict(zip(mapping_df["SMEName"], mapping_df["Pseudonym"]))
else:
    mapping = {}
    mapping_df = pd.DataFrame(columns=["SMEName", "Pseudonym"])


# Assign new pseudonyms for unknown SME names
def get_pseudonym(name):
    if pd.isna(name) or name.strip() == "":
        return None
    if name in mapping:
        return mapping[name]
    pseudonym = f"SME{len(mapping) + 1:03d}"
    mapping[name] = pseudonym
    return pseudonym


merged["SME_ID"] = merged["SMEName"].apply(get_pseudonym)
# Drop both SMEName and SourceFile to ensure full anonymisation
merged.drop(columns=["SMEName", "SourceFile"], inplace=True)

# Update mapping CSV
mapping_df = pd.DataFrame(list(mapping.items()), columns=["SMEName", "Pseudonym"])
mapping_df.to_csv(PSEUDONYM_FILE, index=False)


# --- Priority Score Calculation ---
def compute_priority_score(row):
    try:
        importance = float(row.get("Importance", 0))
        frequency = float(row.get("Frequency", 0))
        feasibility = float(row.get("Feasibility", 3))  # default = 3
        return round((importance * 0.5 + frequency * 0.4 + feasibility * 0.1), 2)
    except Exception:
        return None


merged["PriorityScore"] = merged.apply(compute_priority_score, axis=1)

# --- Validation ---
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

if errors:
    print("\n".join(errors))
    # Optionally, you could exit here if you want to enforce strict validation
    # import sys; sys.exit(1)
else:
    print("Validation passed.")

# --- Set defaults for missing synthesis columns ---
merged["Feasibility"] = merged.get("Feasibility", 3)
merged["Status"] = "Draft"
merged["Version"] = "v0.1"
merged["ChangeNote"] = f"Initial merge"

# --- Save merged collation CSV ---
merged.to_csv(OUTPUT_CSV, index=False)
print(f"Collated CSV saved: {OUTPUT_CSV}")

# --- Optional Graph Exports ---
if EXPORT_JSONLD or EXPORT_NEO4J:
    if EXPORT_JSONLD:
        import json

        context = {"@vocab": "https://example.org/rfi#", "RequirementID": "rfi:RequirementID",
                   "RequirementStatement": "rfi:RequirementStatement", "SME_ID": "rfi:SME_ID",
                   "PriorityScore": "rfi:PriorityScore", "Priority": "rfi:Priority"}
        jsonld_data = {"@context": context, "@graph": merged.to_dict(orient="records")}
        with open(os.path.join(DERIVED_DIR, "requirements_rfi_collation.jsonld"), "w") as f:
            json.dump(jsonld_data, f, indent=2)
        print("JSON-LD export complete.")

    if EXPORT_NEO4J:
        # Nodes: Requirements
        nodes = merged[["RequirementID", "RequirementStatement", "PriorityScore"]].drop_duplicates()
        nodes[":LABEL"] = "Requirement"
        nodes.rename(columns={"RequirementID": "id:ID", "RequirementStatement": "title",
                              "PriorityScore": "priority_score"}, inplace=True)
        nodes.to_csv(os.path.join(DERIVED_DIR, "nodes_requirements.csv"), index=False)

        # Relationships: SME -> Requirement
        rels = merged[["RequirementID", "SME_ID"]].dropna()
        rels[":TYPE"] = "PROPOSED_BY"
        rels.rename(columns={"RequirementID": ":START_ID", "SME_ID": ":END_ID"}, inplace=True)
        rels.to_csv(os.path.join(DERIVED_DIR, "relationships_sme.csv"), index=False)

        print("Neo4j CSV export complete.")
