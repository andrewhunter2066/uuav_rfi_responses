from pathlib import Path
import shutil

# Source files (generated locally)
# Since files are in project_root/input/, set SOURCE_DIR to the project root
SOURCE_DIR = Path(__file__).parent  # Points to the directory containing this script
FILES_TO_MOVE_TO_CLEAN = [
    "input/normalised_all_responses.csv",
    "input/normalised_all_responses.json",
    "input/rfi_response_data_dictionary_draft.csv",
    "input/rfi_response_data_dictionary_schema.json",
]

FILES_TO_MOVE_TO_OUTPUT = [
    "docs/Q1 Classification Schema.md",
    "docs/mission_planning_taxonomy.json",
    "docs/synonym-protection pairing matrix.md",
    "docs/Gap Analysis against initial Taxonomy.md",
    "docs/Q2 Classification Schema.md",
    "output/normalised_all_classified_responses.csv",
    "output/q1_vocabulary.jsonld",
    "output/S1_Q1_classification_results.csv",
    "output/S1_Q1_classification_summary.json",
    "output/S1_Q1_classification_summary_child.csv",
    "output/S1_Q1_classification_summary_parent.csv",
    "output/S1_Q2_classification_results.csv",
    "output/S1_Q2_classification_summary.json",
    "output/S1_Q2_classification_summary_child.csv",
    "output/S1_Q2_classification_summary_parent.csv",
    "output/S1_Q3_classification_results.csv",
    "output/S1_Q3_classification_summary.json",
    "output/S1_Q3_classification_summary_child.csv",
    "output/S1_Q3_classification_summary_parent.csv",
    "output/S1_Q4_classification_results.csv",
    "output/S1_Q4_classification_summary.json",
    "output/S1_Q4_classification_summary_child.csv",
    "output/S1_Q4_classification_summary_parent.csv",
    "output/S1_Q5_classification_results.csv",
    "output/S1_Q5_classification_summary.json",
    "output/S1_Q5_classification_summary_child.csv",
    "output/S1_Q5_classification_summary_parent.csv",
    "output/S1_Q6_classification_results.csv",
    "output/S1_Q6_classification_summary.json",
    "output/S1_Q6_classification_summary_child.csv",
    "output/S1_Q6_classification_summary_parent.csv",
    "output/S2_Q1_classification_results.csv",
    "output/S2_Q1_classification_summary.json",
    "output/S2_Q1_classification_summary_child.csv",
    "output/S2_Q1_classification_summary_parent.csv",
    "output/S2_Q2_classification_results.csv",
    "output/S2_Q2_classification_summary.json",
    "output/S2_Q2_classification_summary_child.csv",
    "output/S2_Q2_classification_summary_parent.csv",
    "output/S2_Q3_classification_results.csv",
    "output/S2_Q3_classification_summary.json",
    "output/S2_Q3_classification_summary_child.csv",
    "output/S2_Q3_classification_summary_parent.csv",
    "output/S2_Q4_classification_results.csv",
    "output/S2_Q4_classification_summary.json",
    "output/S2_Q4_classification_summary_child.csv",
    "output/S2_Q4_classification_summary_parent.csv",
    "output/S2_Q5_classification_results.csv",
    "output/S2_Q5_classification_summary.json",
    "output/S2_Q5_classification_summary_child.csv",
    "output/S2_Q5_classification_summary_parent.csv",
    "output/S2_Q6_classification_results.csv",
    "output/S2_Q6_classification_summary.json",
    "output/S2_Q6_classification_summary_child.csv",
    "output/S2_Q6_classification_summary_parent.csv",
    "output/S3_Q1_classification_results.csv",
    "output/S3_Q1_classification_summary.json",
    "output/S3_Q1_classification_summary_child.csv",
    "output/S3_Q1_classification_summary_parent.csv",
    "output/S3_Q2_classification_results.csv",
    "output/S3_Q2_classification_summary.json",
    "output/S3_Q2_classification_summary_child.csv",
    "output/S3_Q2_classification_summary_parent.csv",
    "output/S3_Q3_classification_results.csv",
    "output/S3_Q3_classification_summary.json",
    "output/S3_Q3_classification_summary_child.csv",
    "output/S3_Q3_classification_summary_parent.csv",
    "output/S3_Q4_classification_results.csv",
    "output/S3_Q4_classification_summary.json",
    "output/S3_Q4_classification_summary_child.csv",
    "output/S3_Q4_classification_summary_parent.csv",
    "output/S3_Q5_classification_results.csv",
    "output/S3_Q5_classification_summary.json",
    "output/S3_Q5_classification_summary_child.csv",
    "output/S3_Q5_classification_summary_parent.csv",
    "output/S3_Q6_classification_results.csv",
    "output/S3_Q6_classification_summary.json",
    "output/S3_Q6_classification_summary_child.csv",
    "output/S3_Q6_classification_summary_parent.csv",
    "output/S4_Task 1_classification_results.csv",
    "output/S4_Task 1_classification_summary.json",
    "output/S4_Task 1_classification_summary_child.csv",
    "output/S4_Task 1_classification_summary_parent.csv",
    "output/S4_Task 2_classification_results.csv",
    "output/S4_Task 2_classification_summary.json",
    "output/S4_Task 2_classification_summary_child.csv",
    "output/S4_Task 2_classification_summary_parent.csv",
    "output/S4_Task 3_classification_results.csv",
    "output/S4_Task 3_classification_summary.json",
    "output/S4_Task 3_classification_summary_child.csv",
    "output/S4_Task 3_classification_summary_parent.csv",
    "output/S4_Task 4_classification_results.csv",
    "output/S4_Task 4_classification_summary.json",
    "output/S4_Task 4_classification_summary_child.csv",
    "output/S4_Task 4_classification_summary_parent.csv",
    "output/S4_Task 5_classification_results.csv",
    "output/S4_Task 5_classification_summary.json",
    "output/S4_Task 5_classification_summary_child.csv",
    "output/S4_Task 5_classification_summary_parent.csv",
    "output/S4_Task 6_classification_results.csv",
    "output/S4_Task 6_classification_summary.json",
    "output/S4_Task 6_classification_summary_child.csv",
    "output/S4_Task 6_classification_summary_parent.csv",
]

# Target repo (can be local or remote)
TARGET_REPO_DIR = Path("C:/Users/Andrew/Documents/GitHub/RAN").expanduser()
TARGET_CLEAN_SUBDIR = TARGET_REPO_DIR / "rfi/rfi_data/cleaned_merged_responses"
TARGET_OUTPUT_SUBDIR = TARGET_REPO_DIR / "rfi/rfi_data/output"

TARGET_CLEAN_SUBDIR.mkdir(parents=True, exist_ok=True)
TARGET_OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)


def move_files(files_to_move: list[str], target_subdir: Path):
    """
    Move a list of files to a specified target subdirectory. Each file from the
    list will be copied from the SOURCE_DIR to the given target subdirectory.
    The method ensures that only the filename part of the source path is used
    to build the destination path.

    :param files_to_move: A list of file names (as strings) to be moved from the
        source directory to the target directory.
    :param target_subdir: Path object representing the target subdirectory where
        the files will be copied.
    :return: None
    """
    for fname in files_to_move:
        src = SOURCE_DIR / fname
        dest = target_subdir / Path(fname).name  # Use only filename for destination

        # Add error handling for better debugging
        if not src.exists():
            print(f"ERROR: Source file not found: {src}")
            continue

        shutil.copy2(src, dest)
        print(f"Copied {src.name} → {dest}")


if __name__ == "__main__":
    move_files(FILES_TO_MOVE_TO_CLEAN, TARGET_CLEAN_SUBDIR)
    move_files(FILES_TO_MOVE_TO_OUTPUT, TARGET_OUTPUT_SUBDIR)
