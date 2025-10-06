#!/usr/bin/env python3
"""
merge_rfi_requirements.py
-------------------------
Merge SME input CSVs (requirements elicitation) into a single collation file.
Designed for use in a GitHub repo workflow or local script.

Usage:
    python merge_rfi_requirements.py --input_dir ./input --output ./derived/requirements_rfi_collation.csv
"""
import os
import glob
import pandas as pd
import argparse
from datetime import date


INPUT_DIR = "./input"
OUTPUT_FILE = "./derived/requirements_rfi_collation.csv"


def compute_priority_score(row):
    """Compute a weighted score (favouring Importance and Frequency)."""
    try:
        importance = float(row.get("Importance", 0))
        frequency = float(row.get("Frequency", 0))
        feasibility = float(row.get("Feasibility", 3))  # default = 3 if unknown
        return round((importance * 0.5 + frequency * 0.4 + feasibility * 0.1), 2)
    except Exception:
        return None


def merge_rfi_csvs(input_dir, output_file):
    # Find all CSVs in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    all_frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["SourceFile"] = os.path.basename(f)
        all_frames.append(df)

    merged = pd.concat(all_frames, ignore_index=True)

    # Add synthesis columns
    merged["Feasibility"] = merged.get("Feasibility", 3)
    merged["PriorityScore"] = merged.apply(compute_priority_score, axis=1)
    merged["Status"] = "Draft"
    merged["Version"] = "v0.1"
    merged["ChangeNote"] = f"Initial merge on {date.today().isoformat()}"

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)

    print(f"Collated {len(merged)} requirements from {len(csv_files)} files.")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':

    merge_rfi_csvs(INPUT_DIR, OUTPUT_FILE)
