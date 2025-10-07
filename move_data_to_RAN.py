from pathlib import Path
import shutil

# Source files (generated locally)
# Since files are in project_root/input/, set SOURCE_DIR to the project root
SOURCE_DIR = Path(__file__).parent  # Points to the directory containing this script
FILES_TO_MOVE = [
    "input/normalised_all_responses.csv",
    "input/normalised_all_responses.json",
    "input/rfi_response_data_dictionary_draft.csv",
    "input/rfi_response_data_dictionary_schema.json"
]

# Target repo (can be local or remote)
TARGET_REPO_DIR = Path("C:/Users/Andrew/Documents/GitHub/RAN").expanduser()
TARGET_SUBDIR = TARGET_REPO_DIR / "rfi/rfi_data/cleaned_merged_responses"

TARGET_SUBDIR.mkdir(parents=True, exist_ok=True)

for fname in FILES_TO_MOVE:
    src = SOURCE_DIR / fname
    dest = TARGET_SUBDIR / Path(fname).name  # Use only filename for destination

    # Add error handling for better debugging
    if not src.exists():
        print(f"ERROR: Source file not found: {src}")
        continue

    shutil.copy2(src, dest)
    print(f"Copied {src.name} → {dest}")
