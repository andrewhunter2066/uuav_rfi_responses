"""
Map requirements in a CSV file to taxonomy concepts (keyword + semantic match).

Usage:
    python map_requirements_to_taxonomy.py --requirements requirements.csv --taxonomy taxonomy.json --output mapping_results.csv
"""
import json
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ---------------------------------------------------------------------
# 1. Flatten nested taxonomy into a searchable list
# ---------------------------------------------------------------------
def flatten_taxonomy(taxonomy: dict, parent: str = "") -> list:
    """
    Recursively flatten nested taxonomy into a list of domain/concept entries.
    Each entry includes full path, description, and terms.
    """
    rows = []
    for key, value in taxonomy.items():
        name = f"{parent} > {key}" if parent else key
        desc = value.get("Description", "")
        terms = value.get("Terms", [])
        if "Concepts" in value:
            rows.extend(flatten_taxonomy(value["Concepts"], name))
        else:
            rows.append({"path": name, "description": desc, "terms": terms})
    return rows


# ---------------------------------------------------------------------
# 2. Compute best taxonomy matches for a given requirement
# ---------------------------------------------------------------------
def find_best_matches(requirement_text, taxonomy_df, model, top_n=5):
    """
    Find top matching taxonomy entries for a given requirement.
    Combines keyword/fuzzy matching and semantic similarity.
    """
    # --- Keyword / fuzzy match score ---
    keyword_scores = []
    for i, row in taxonomy_df.iterrows():
        terms = " ".join(row["terms"])
        if terms.strip():
            score = fuzz.token_set_ratio(requirement_text.lower(), terms.lower())
        else:
            score = fuzz.token_set_ratio(requirement_text.lower(), row["description"].lower())
        keyword_scores.append(score)

    taxonomy_df["keyword_score"] = keyword_scores

    # --- Semantic similarity using embeddings ---
    emb_req = model.encode(requirement_text, convert_to_tensor=True)
    emb_tax = model.encode(taxonomy_df["description"].tolist(), convert_to_tensor=True)
    cos_scores = util.cos_sim(emb_req, emb_tax).squeeze().tolist()

    taxonomy_df["semantic_score"] = cos_scores

    # --- Combine weighted score (tune weights as desired) ---
    taxonomy_df["combined_score"] = taxonomy_df["semantic_score"] * 0.7 + taxonomy_df["keyword_score"] / 100 * 0.3

    return taxonomy_df.nlargest(top_n, "combined_score")[
        ["path", "description", "combined_score", "keyword_score", "semantic_score"]]


# ---------------------------------------------------------------------
# 3. Main mapping routine
# ---------------------------------------------------------------------
def map_requirements(requirements_csv, taxonomy_json, output_csv, output_json, top_n=3):
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
            # Split taxonomy_path by '>'
            path_parts = match["path"].split(' > ')
            taxonomy_domain = path_parts[0] if len(path_parts) > 0 else ""
            domain_concept = path_parts[1] if len(path_parts) > 1 else ""
            label = classify_match(
                semantic_score=round(match["semantic_score"], 3),
                keyword_score=round(match["keyword_score"], 1),
                combined_score=round(match["combined_score"], 3)
            )

            results_flat.append({
                "requirement_id": row.get("id", ""),
                "requirement_text": req_text,
                "taxonomy_domain": taxonomy_domain,
                "domain_concept": domain_concept,
                "combined_score": round(match["combined_score"], 3),
                "keyword_score": round(match["keyword_score"], 1),
                "semantic_score": round(match["semantic_score"], 3),
                "classification": label
            })

        # JSON-structured results
        json_entry = {
            "requirement_id": row.get("id", ""),
            "requirement_text": req_text,
            "matches": [
                {
                    "taxonomy_domain": m["path"].split(' > ')[0] if len(m["path"].split(' > ')) > 0 else "",
                    "domain_concept": m["path"].split(' > ')[1] if len(m["path"].split(' > ')) > 1 else "",
                    "combined_score": round(m["combined_score"], 3),
                    "keyword_score": round(m["keyword_score"], 1),
                    "semantic_score": round(m["semantic_score"], 3),
                    "classification": classify_match(
                        round(m["semantic_score"], 3),
                        round(m["keyword_score"], 1),
                        round(m["combined_score"], 3)
                    )
                }
                for _, m in matches.iterrows()
            ],
        }
        results_json.append(json_entry)

    # Save CSV
    pd.DataFrame(results_flat).to_csv(output_csv, index=False)
    print(f"💾 CSV mapping results saved to: {output_csv}")

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results_json, jf, indent=2, ensure_ascii=False)
    print(f"💾 JSON mapping results saved to: {output_json}")


def classify_match(semantic_score, keyword_score, combined_score):
    """
    Classify a match as auto-accepted, needs-review, or discard.
    Returns one of: 'auto', 'review', 'discard'.
    """
    if combined_score >= 0.75 or (semantic_score >= 0.80 and keyword_score >= 70):
        return "auto"
    elif combined_score >= 0.65 or (semantic_score >= 0.65 and keyword_score >= 60):
        return "review"
    else:
        return "discard"


def map_requirements_to_requirements(
        mission_csv: str,
        project_csv: str,
        output_csv: str,
        output_json: str,
        top_n: int = 3
):
    """
    Map mission-phase requirements to project requirements using embedding and fuzzy similarity.
    """

    print(f"\n📘 Loading Mission Phase Requirements: {mission_csv}")
    mission_df = pd.read_csv(mission_csv)
    if "Requirement Statement" in mission_df.columns:
        mission_df.rename(columns={"Requirement Statement": "requirement_text"}, inplace=True)

    # Handle mission requirement ID column naming
    if "id" in mission_df.columns:
        mission_df.rename(columns={"id": "requirement_id"}, inplace=True)
    elif "ID" in mission_df.columns:
        mission_df.rename(columns={"ID": "requirement_id"}, inplace=True)
    # If requirement_id already exists, no need to rename

    print(f"📗 Loading Project Requirements: {project_csv}")
    project_df = pd.read_csv(project_csv)
    # Normalise column naming
    project_df.rename(columns={
        "requirement_text": "req_text",
        "requirement_id": "req_id",
        "requirement_name": "req_name"
    }, inplace=True)

    print(f"✅ Project requirements loaded: {len(project_df)}")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Precompute project embeddings
    project_embeddings = model.encode(project_df["req_text"].tolist(), convert_to_tensor=True)

    results_flat = []
    results_json = []

    for _, mission_row in tqdm(mission_df.iterrows(), total=len(mission_df), desc="Mapping"):
        mission_text = mission_row["requirement_text"]
        mission_id = mission_row.get("requirement_id", "")

        # Compute fuzzy keyword scores
        keyword_scores = [
            fuzz.token_set_ratio(mission_text.lower(), t.lower())
            for t in project_df["req_text"]
        ]

        # Compute semantic similarity
        mission_emb = model.encode(mission_text, convert_to_tensor=True)
        sim_scores = util.cos_sim(mission_emb, project_embeddings).squeeze().tolist()

        # Combine weighted score
        combined_scores = [
            0.7 * sim + 0.3 * (kw / 100) for sim, kw in zip(sim_scores, keyword_scores)
        ]

        project_df["combined_score"] = combined_scores
        top_matches = project_df.nlargest(top_n, "combined_score")

        # Save to flat CSV form
        for _, match in top_matches.iterrows():
            label = classify_match(
                semantic_score=round(sim_scores[match.name], 3),
                keyword_score=round(keyword_scores[match.name], 1),
                combined_score=round(match["combined_score"], 3)
            )
            results_flat.append({
                "mission_requirement_id": mission_id,
                "mission_requirement_text": mission_text,
                "project_requirement_id": match["req_id"],
                "project_requirement_name": match["req_name"],
                "combined_score": round(match["combined_score"], 3),
                "semantic_score": round(sim_scores[match.name], 3),
                "keyword_score": round(keyword_scores[match.name], 1),
                "classification": label
            })

        # Save to nested JSON form
        results_json.append({
            "mission_requirement_id": mission_id,
            "mission_requirement_text": mission_text,
            "matches": [
                {
                    "project_requirement_id": m["req_id"],
                    "project_requirement_name": m["req_name"],
                    "combined_score": round(m["combined_score"], 3),
                    "semantic_score": round(sim_scores[m.name], 3),
                    "keyword_score": round(keyword_scores[m.name], 1),
                    "classification": classify_match(
                        round(sim_scores[m.name], 3),
                        round(keyword_scores[m.name], 1),
                        round(m["combined_score"], 3)
                    )
                }
                for _, m in top_matches.iterrows()
            ],
        })

    # Write results
    pd.DataFrame(results_flat).to_csv(output_csv, index=False)
    print(f"💾 CSV results written to {output_csv}")

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results_json, jf, indent=2, ensure_ascii=False)
    print(f"💾 JSON results written to {output_json}")


def run_mapping(mode="taxonomy", mission_requirements=None, taxonomy=None,
                project_requirements=None, top_n=3, csv_output=None, json_output=None):
    """
    Programmatically run mapping without command-line arguments.

    :param mode: Either "taxonomy" or "reqmap"
    :param mission_requirements: Path to mission requirements CSV
    :param taxonomy: Path to taxonomy JSON (for taxonomy mode)
    :param project_requirements: Path to project requirements CSV (for reqmap mode)
    :param top_n: Number of top matches to return
    :param csv_output: Path to output CSV file
    :param json_output: Path to output JSON file
    """
    if mode == "taxonomy":
        if not all([mission_requirements, taxonomy, csv_output, json_output]):
            raise ValueError("taxonomy mode requires mission_requirements, taxonomy, csv_output, and json_output")
        map_requirements(mission_requirements, taxonomy, csv_output, json_output, top_n)
    elif mode == "reqmap":
        if not all([mission_requirements, project_requirements, csv_output, json_output]):
            raise ValueError(
                "reqmap mode requires mission_requirements, project_requirements, csv_output, and json_output")
        map_requirements_to_requirements(mission_requirements, project_requirements, csv_output, json_output, top_n)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'taxonomy' or 'reqmap'")


# ---------------------------------------------------------------------
# 4. CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_mapping(
        mode="taxonomy",
        mission_requirements="C:/Users/Andrew/PycharmProjects/RAN/Reports/uuv-mission-phase-requirements.csv",
        taxonomy="C:/Users/Andrew/PycharmProjects/RAN/rfi/rfi_data/output/reduced_mission_planning_taxonomy.json",
        top_n=3,
        csv_output="output/mapping_mission_results.csv",
        json_output="output/mapping_mission_results.json"
    )

    run_mapping(
        mode="reqmap",
        mission_requirements="C:/Users/Andrew/PycharmProjects/RAN/Reports/uuv-mission-phase-requirements.csv",
        project_requirements="C:/Users/Andrew/PycharmProjects/RAN/rfi/rfi_data/output/uuv_requirements.csv",
        top_n=3,
        csv_output="output/mapping_project_results.csv",
        json_output="output/mapping_project_results.json"
    )

