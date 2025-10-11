#!/usr/bin/env python3

import pandas as pd
import random
import re
import nltk
import os
import json
from nltk.corpus import stopwords, wordnet
from datetime import datetime
from typing import List

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
    r'\bcomms\b': 'communications',
    r'\bhipap\b': 'high precision acoustic positioning',
    r'\be.g.\b': '',
    r'\b%\b': 'percent',
}

# Regex pattern to keep only word characters, spaces, and hyphens
PUNCTUATION_PATTERN = r'[^\w\s\-]'


# Words to NEVER replace (protected terms)
PROTECTED_TERMS = {
    # --- Analytical & Data Management ---
    "ai", "artificial intelligence", "cf convention", "confidence interval",
    "data lineage", "data provenance", "error budget", "estimation factor",
    "geojson", "geospatial data", "json-ld", "knowledge graph",
    "machine learning", "metadata", "metadata schema", "ml",
    "netcdf", "netcdf-cf", "ontology", "performance model",
    "probability distribution", "python", "r", "sdk", "semantic model",
    "sensitivity analysis", "taxonomy", "test set", "training data",
    "uncertainty model", "validation data",

    # --- Communications & Control ---
    "acoustic modem", "bandwidth allocation", "command and control", "command latency",
    "comms", "communication handshake", "communication outage",
    "communications plan", "control architecture", "control interface",
    "control loop", "c2", "downlink", "latency", "link stability",
    "mission controller", "mission monitoring", "mission planning system",
    "operator console", "real-time control", "telemetry", "telemetry link",
    "uplink", "uplink path",

    # --- Data Products & Requirements ---
    "classified map", "coverage map", "coverage polygon", "data deliverable",
    "data fusion", "data product", "deliverable product", "geotiff",
    "point cloud", "sensor fusion", "survey deliverable", "survey footprint",

    # --- Environmental & Oceanographic ---
    "bathymetric grid", "beach gradient", "conductivity", "ctd",
    "current", "current shear", "density", "downwelling", "east australian current",
    "eac", "environmental restriction", "halocline", "habitat zone",
    "hydrodynamics", "hydrography", "hydrostatic pressure",
    "internal wave", "ocean front", "oceanographic conditions",
    "pressure", "salinity", "sea floor roughness", "sea state",
    "sediment", "sediment transport", "sensitive habitat", "substrate",
    "substrate composition", "swell", "temperature", "thermocline",
    "tidal", "tide", "turbidity", "turbulent kinetic energy",
    "upwelling", "visibility", "water column", "wave height",

    # --- Historical & Contextual ---
    "archival data", "baseline dataset", "historical", "historical bathymetry",
    "historical survey", "legacy mission", "mission archive", "past operations",
    "reference dataset", "reference data",

    # --- Mission Parameters & Objectives ---
    "area coverage rate", "coverage requirement", "mission", "mission duration",
    "mission feasibility", "mission objective", "mission phase",
    "mission profile", "mission readiness", "mission rehearsal",
    "mission risk", "mission sequencing", "mission segment",
    "mission tempo", "mission timeline", "mission validation",
    "search pattern", "survey objective", "time on task",

    # --- Navigation & Positioning ---
    "accuracy", "ais", "bearing", "chart datum", "chart projection",
    "coordinate reference system", "depth", "doppler velocity log",
    "dvl", "epsg", "fix", "fix accuracy", "geo-reference", "gnss",
    "gps", "heading", "inertial drift", "inertial navigation system",
    "ins", "local datum", "navigation", "navigation solution",
    "no-go zone", "navigational hazard", "position", "position fix",
    "position uncertainty", "route", "route plan", "survey coverage",
    "survey speed", "trackline", "waypoint", "zone of confidence",

    # --- Operational Logistics ---
    "access corridor", "area of interest", "area of operation", "area of operations",
    "deployment", "deployment vessel", "launch", "launch point",
    "launch sequence", "launch site", "mission logistics", "operation window",
    "operational area", "recovery", "recovery location", "recovery operation",
    "recovery point", "recovery track", "staging area", "support vessel",
    "survey planning", "transit", "transit corridor",

    # --- Organisational / Workflow & Control ---
    "command latency", "control architecture", "mission control",
    "system configuration", "system validation",

    # --- Programmatic / Org-Specific ---
    "api", "arcgis", "aodn", "emii", "geopandas", "geoserver", "imos",
    "matlab", "nmea", "pandas", "postgis", "qgis",

    # --- Risk, Safety & Regulation ---
    "collision avoidance", "contingency plan", "contingency procedure",
    "critical system", "environmental hazard", "exclusion zone",
    "fail-safe", "failure mode", "loss of communications", "loss of signal",
    "mitigation measure", "redundancy", "redundant system",
    "restricted area", "restricted zone", "risk", "risk assessment",
    "risk factor", "risk likelihood", "risk mitigation strategy",
    "risk severity", "safety", "safety case", "safety margin",
    "submerged hazard", "threat",

    # --- Sensor & Data Quality ---
    "acoustic", "acoustic interference", "calibration coefficient",
    "calibration status", "confidence zone", "data quality",
    "data quality flag", "data resolution", "fix frequency",
    "ping", "ping rate", "positional uncertainty", "sampling rate",
    "sensor", "sensor calibration", "signal to noise ratio", "snr",
    "update frequency",

    # --- Survey & Mapping ---
    "coverage", "digital elevation model", "dem", "line plan",
    "metadata", "mosaic", "overlap", "survey", "survey grid",
    "survey line", "swath", "track spacing",

    # --- Temporal & Mission Management ---
    "launch window", "mission debrief", "mission execution",
    "mission window", "recovery window", "time synchronization",
    "timestamp", "utc",

    # --- Terrain & Bathymetry ---
    "bathy grid", "bathymetry", "bottom texture", "bottom type",
    "complexity", "gradient", "multibeam", "seabed", "seabed morphology",
    "seafloor", "seafloor complexity", "slope", "terrain", "terrain model",
    "topography", "vector electronic navigational charts",

    # --- Vehicle, Platform & System Terms ---
    "altitude", "asv", "attitude", "auv", "autonomous surface vehicle",
    "autonomous underwater vehicle", "battery", "battery endurance",
    "battery life", "ballast", "climb rate", "communication",
    "descent rate", "duty cycle", "endurance", "equipment limitations",
    "hover capability", "hull", "launch and recovery system", "lars",
    "maneuverability", "maximum", "minimum", "mission abort condition",
    "mission endurance", "oem", "payload", "payload bay",
    "payload capacity", "pitch", "platform", "power budget",
    "power consumption", "power constraint", "propulsion", "range",
    "roll", "rov", "speed", "stability", "thruster", "turn radius",
    "turnaround time", "usv", "vehicle", "vehicle controller",
    "vehicle fault", "vessel", "yaw", "unmanned underwater vehicle",
}


# Synonym - Protection conflict rules
CONFLICT_RULES = {
    # --- Temporal / Oceanographic context ---
    "temporal": {
        "protected": [
            "current", "tidal", "tide", "eac", "east australian current",
            "temporal resolution", "time series", "timestamp", "utc",
            "time synchronization", "mission timeline"
        ],
        "synonyms": ["time", "duration", "interval", "period", "timespan"]
    },

    # --- Terrain / Bathymetric context ---
    "terrain_bathymetry": {
        "protected": [
            "depth", "bathymetry", "bathymetric grid", "seafloor", "seabed",
            "gradient", "slope", "topography", "terrain", "bathy grid",
            "chart datum", "vertical datum", "bottom texture"
        ],
        "synonyms": ["depth", "area", "layer", "zone", "surface"]
    },

    # --- Environmental / Oceanographic context ---
    "environmental": {
        "protected": [
            "sea state", "wave height", "turbidity", "sediment", "substrate",
            "temperature", "salinity", "water column", "density", "conductivity",
            "eac", "east australian current", "hydrodynamics", "hydrography"
        ],
        "synonyms": ["environment", "conditions", "ocean", "marine", "weather"]
    },

    # --- Survey / Mapping context ---
    "survey": {
        "protected": [
            "survey", "multibeam echo sounder", "mbes", "coverage map",
            "swath", "line plan", "track spacing", "geotiff",
            "digital elevation model", "dem", "survey grid", "coverage polygon"
        ],
        "synonyms": ["mapping", "data", "collection", "surveying", "charting"]
    },

    # --- Navigation / Positioning context ---
    "navigation_and_positioning": {
        "protected": [
            "gps", "gnss", "positioning", "navigation", "ins",
            "inertial navigation system", "doppler velocity log", "dvl",
            "fix", "bearing", "waypoint", "route", "trackline",
            "coordinate reference system", "epsg", "geofence", "chart datum"
        ],
        "synonyms": ["location", "path", "trajectory", "movement", "space"]
    },

    # --- Mission Operations context ---
    "mission_ops": {
        "protected": [
            "launch", "recovery", "transit", "endurance", "mission duration",
            "mission phase", "mission controller", "operator console",
            "mission segment", "mission readiness", "mission objective"
        ],
        "synonyms": ["mission", "operation", "activity", "task", "objective"]
    },

    # --- Vehicle / Platform context ---
    "platform": {
        "protected": [
            "launch and recovery system", "lars", "auv", "usv", "rov", "asv", "uuv",
            "autonomous underwater vehicle", "autonomous surface vehicle",
            "remotely operated vehicle", "vessel", "platform", "hull", "payload",
            "unmanned underwater vehicle",
        ],
        "synonyms": ["vehicle", "capability", "system", "asset"]
    },

    # --- Propulsion / Power context ---
    "propulsion": {
        "protected": [
            "propulsion", "thruster", "battery", "ballast", "energy",
            "power system", "power budget", "battery endurance"
        ],
        "synonyms": ["drive", "movement", "motor", "thrust"]
    },

    # --- Communications / Networking context ---
    "comms": {
        "protected": [
            "comms", "telemetry", "acoustic modem", "satellite link",
            "uplink", "downlink", "api", "sdk", "network interface", "data uplink",
            "bandwidth allocation", "link stability", "latency"
        ],
        "synonyms": ["communication", "connectivity", "link", "signal"]
    },

    # --- Data Processing / Modeling context ---
    "data_processing": {
        "protected": [
            "machine learning", "artificial intelligence", "ai", "ml",
            "training data", "test set", "validation data",
            "python", "r", "matlab", "pandas", "xarray", "geopandas",
            "data fusion", "data pipeline"
        ],
        "synonyms": ["analytics", "modeling", "processing", "prediction"]
    },

    # --- Semantic / Integration context ---
    "semantic_data": {
        "protected": [
            "knowledge graph", "ontology", "taxonomy", "json-ld",
            "metadata schema", "controlled vocabulary", "data dictionary",
            "semantic model", "cf convention", "netcdf-cf"
        ],
        "synonyms": ["data", "classification", "schema", "structure"]
    },

    # --- Risk / Safety / Compliance context ---
    "risk": {
        "protected": [
            "risk", "threat", "hazard", "cyber risk", "no-go zone",
            "collision avoidance", "restricted area", "safety case", "fail-safe",
            "redundancy", "contingency plan", "risk assessment",
            "risk likelihood", "risk severity", "mitigation measure"
        ],
        "synonyms": ["important", "limitation", "consideration", "danger", "vulnerability"]
    },

    # --- Logistics / Operational Planning context (NEW) ---
    "logistics": {
        "protected": [
            "deployment", "launch point", "recovery point", "access corridor",
            "mission window", "staging area", "support vessel", "turnaround time"
        ],
        "synonyms": ["planning", "coordination", "support", "setup"]
    },

    # --- Data Products / Deliverables context (NEW) ---
    "data_products": {
        "protected": [
            "data product", "survey deliverable", "coverage polygon",
            "data quality", "zone of confidence", "confidence zone"
        ],
        "synonyms": ["output", "dataset", "map", "deliverable"]
    },

    # --- Performance / Evaluation context (NEW) ---
    "performance": {
        "protected": [
            "efficiency", "throughput", "latency", "accuracy",
            "timing", "delay", "performance model"
        ],
        "synonyms": ["speed", "effectiveness", "responsiveness"]
    },

    # --- Historical / Contextual Data context (NEW) ---
    "historical": {
        "protected": [
            "historical", "legacy mission", "past operations",
            "reference data", "archive", "baseline dataset"
        ],
        "synonyms": ["previous", "contextual", "reference", "benchmark"]
    }
}


# Which method to use, domain-specific (True) or general NLP (False)
DOMAIN_SPECIFIC = True

# Suppression Log
SUPPRESSION_LOG = []


def build_category_keywords(taxonomy: dict, lowercase: bool = True, deduplicate: bool = True) -> dict[str, list[str]]:
    """
    Build flat keyword lists per taxonomy category for classification.
    Traverses the taxonomy hierarchy and collects all terms under 'Concepts'.

    Args:
        taxonomy (dict): The full taxonomy dict.
        lowercase (bool): Whether to normalise keywords to lowercase.
        deduplicate (bool): Whether to deduplicate keyword lists.

    Returns:
        dict[str, list[str]]: Mapping of category → keyword list.
    """
    category_keywords = {}

    # Expect the taxonomy to be unwrapped at this point
    # Each top-level key should be a category with a "Concepts" sub-dictionary
    for category, content in taxonomy.items():
        keywords = []
        if isinstance(content, dict) and "Concepts" in content:
            for group_terms in content["Concepts"].values():
                if isinstance(group_terms, list):
                    keywords.extend(group_terms)

        # Optional normalization
        if lowercase:
            keywords = [kw.lower().strip() for kw in keywords]

        # Optional deduplication while preserving order
        if deduplicate:
            seen = set()
            keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]

        category_keywords[category] = keywords

    return category_keywords


def tokenize_text(text: str) -> List[str]:
    """
    Split text into candidate tokens/phrases for validation.
    Keeps multi-word domain terms intact if possible.
    """
    # Normalise and keep multi-word phrases (basic approach)
    tokens = re.findall(r"[A-Za-z0-9\-\s]+", text)
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    return tokens


def document_validate(text: str,
                      context: str,
                      CONFLICT_RULES: dict,
                      PROTECTED_TERMS: set,
                      DOMAIN_SYNONYMS: dict,
                      strict_mode: bool = False,
                      export_csv: str = None,
                      export_json: str = None) -> dict:
    """
    Validate a whole text document against conflict rules, with composite context support.

    Args:
        text (str): Raw document text.
        context (str): Context (single or composite like 'mission_ops+vehicle').
        CONFLICT_RULES (dict): Context conflict rules.
        PROTECTED_TERMS (set): Global protected terms.
        DOMAIN_SYNONYMS (dict): Domain synonym mappings.
        strict_mode (bool): If True, cross-context validation is enabled.
        export_csv (str): Path to export CSV results.
        export_json (str): Path to export JSON results.

    Returns:
        dict: Structured validation report.
    """

    tokens = tokenize_text(text)
    report = batch_validate(tokens, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode)

    # Export if requested
    df = pd.DataFrame(report["results"])
    if export_csv:
        df.to_csv(export_csv, index=False)
    if export_json:
        df.to_json(export_json, orient="records", indent=2)

    return report


def validate_responses_csv(input_csv,
                           output_dir,
                           CONFLICT_RULES,
                           PROTECTED_TERMS,
                           DOMAIN_SYNONYMS,
                           strict_mode=False):
    """
    Validate all responses in a normalised SME CSV (e.g. normalised_all_responses.csv).
    Generates per-scenario, per-question conflict reports.

    Args:
        input_csv (str): Path to the normalised responses CSV.
        output_dir (str): Directory to store validation reports.
        CONFLICT_RULES (dict): Context-specific conflict rules.
        PROTECTED_TERMS (set): Global protected terms.
        DOMAIN_SYNONYMS (dict): Domain-level synonym mappings.
        strict_mode (bool): If True, check for cross-context conflicts.

    Returns:
        dict: Summary statistics across all scenarios/questions.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    all_summary = {}

    for (scenario, question), group in df.groupby(["ScenarioNumber", "Question"]):
        context = map_question_to_context(question)  # You need a mapping fn
        text_block = " ".join(group["ResponseText"].dropna().astype(str))

        # Validate document-level
        report = document_validate(
            text=text_block,
            context=context,
            CONFLICT_RULES=CONFLICT_RULES,
            PROTECTED_TERMS=PROTECTED_TERMS,
            DOMAIN_SYNONYMS=DOMAIN_SYNONYMS,
            strict_mode=strict_mode,
            export_csv=os.path.join(output_dir, f"S{scenario}_{question}_conflict_report.csv"),
            export_json=os.path.join(output_dir, f"S{scenario}_{question}_conflict_report.json")
        )

        # Store summary
        all_summary[(scenario, question)] = report["summary"]

    return all_summary


def map_question_to_context(question: str) -> str:
    """
    Map RFI question text to taxonomy-aligned validation context.
    Returns a taxonomy category key (or a composite label if multi-domain).
    """

    q = question.lower().strip()

    # --- Q1: Data required for a scenario (what inputs are needed)
    # Covers terrain, environmental data, vehicle constraints, and navigation safety
    if q.startswith("q1"):
        return "terrain_bathymetry+environmental+vehicle+navigation"

    # --- Q2: Estimated factors (what variables should be considered)
    # Aligns strongly with mission ops, vehicle performance, and environmental conditions
    elif q.startswith("q2"):
        return "mission_ops+vehicle+environmental+performance"

    # --- Q3: Estimation guidance (what methods/models should be used)
    # Links to data processing, modelling, historical data, and risk assessment
    elif q.startswith("q3"):
        return "data_processing+historical+risk+performance"

    # --- Q4: Risks to be assessed (risk identification and assessment methods)
    elif q.startswith("q4"):
        return "risk+environmental+vehicle"

    # --- Q5: Specific risks flagged (hazards, safety, mission-critical risk)
    elif q.startswith("q5"):
        return "risk+threats+mission_ops"

    # --- Q6: Documentation & records (knowledge management and data quality)
    elif q.startswith("q6"):
        return "semantic_data+data_products+historical"

    # --- Fallback
    else:
        return "general"


def log_suppression(word, protected, context_snippet):
    SUPPRESSION_LOG.append({
        "word": word,
        "protected_trigger": protected,
        "context_snippet": context_snippet[:120]
    })


def should_skip_synonym(text, synonym_group, conflict_rules):
    """Check if a synonym group should be skipped based on protected term co-occurrence."""
    text_lower = text.lower()
    for rule in conflict_rules.values():
        if synonym_group in rule["synonyms"]:
            for protected in rule["protected"]:
                if protected in text_lower:
                    return True
    return False


def augment_text_minimal(text: str, aug_prob=0.1) -> str:
    """
    Augments text with minimal character-level variations that preserve meaning.
    Suitable for technical/jargon-dense text where synonym replacement is risky.

    Techniques:
    - Random word order swap (adjacent words only, rarely)
    - Punctuation variation
    - Article insertion/removal (a/an/the)

    :param text: The input text to augment.
    :param aug_prob: Probability of applying augmentation.
    :return: Lightly augmented text.
    """
    words = text.split()
    word_count = len(words)

    if word_count < 3 or random.random() > aug_prob:
        return text

    augmented_words = words.copy()

    # Occasionally swap adjacent non-technical words
    if word_count > 3:
        idx = random.randint(0, word_count - 2)
        # Only swap if neither word is in protected terms
        if (augmented_words[idx].lower() not in PROTECTED_TERMS and
                augmented_words[idx + 1].lower() not in PROTECTED_TERMS):
            augmented_words[idx], augmented_words[idx + 1] = \
                augmented_words[idx + 1], augmented_words[idx]

    return ' '.join(augmented_words)


def augment_text_structural(text: str) -> str:
    """
    Creates structural variations while preserving technical content.

    Examples:
    - "X is needed for Y" → "Y requires X"
    - "System must support X" → "X support is required"

    :param text: The input text to augment.
    :return: Structurally varied text.
    """
    # Simple pattern-based transformations
    transformations = [
        (r'\b(\w+) is needed for (\w+)', r'\2 requires \1'),
        (r'\bsystem must support (\w+)', r'\1 support is required'),
        (r'\b(\w+) capability', r'capability to \1'),
        (r'\brequirement for (\w+)', r'\1 requirement'),
    ]

    augmented = text
    # Apply one random transformation if a pattern matches
    random.shuffle(transformations)
    for pattern, replacement in transformations:
        if re.search(pattern, augmented, re.IGNORECASE):
            augmented = re.sub(pattern, replacement, augmented, count=1, flags=re.IGNORECASE)
            break

    return augmented


def check_conflict(word, context, conflict_rules):
    """
    Validate whether a given word conflicts with the protected or synonym sets
    of a specific semantic context.

    Parameters
    ----------
    word : str
        The candidate word or phrase to test.
    context : str
        The domain context (e.g. 'temporal', 'survey', 'mission_ops').
    conflict_rules : dict
        The CONFLICT_RULES structure defining protected/synonym sets.

    Returns
    -------
    dict
        A structured validation result:
        {
            "status": "protected" | "synonym_conflict" | "allowed" | "unknown_context",
            "context": context,
            "reason": str
        }

    Examples
    --------
    >>> check_conflict("bathymetry", "terrain_bathymetry", CONFLICT_RULES)
    {'status': 'protected', 'context': 'terrain_bathymetry', 'reason': 'Exact match in protected terms.'}

    >>> check_conflict("time", "temporal", CONFLICT_RULES)
    {'status': 'synonym_conflict', 'context': 'temporal', 'reason': 'Matches synonym list; may conflict semantically.'}

    >>> check_conflict("fish", "environmental", CONFLICT_RULES)
    {'status': 'allowed', 'context': 'environmental', 'reason': 'No match found in protected or synonym lists.'}
    """
    # Normalise inputs
    word = word.strip().lower()
    context = context.strip().lower()

    # Validate context
    if context not in conflict_rules:
        return {
            "status": "unknown_context",
            "context": context,
            "reason": f"Context '{context}' not defined in conflict rules."
        }

    rules = conflict_rules[context]
    protected = set(w.lower() for w in rules.get("protected", []))
    synonyms = set(w.lower() for w in rules.get("synonyms", []))

    # Exact match checks
    if word in protected:
        return {
            "status": "protected",
            "context": context,
            "reason": "Exact match in protected terms."
        }

    if word in synonyms:
        return {
            "status": "synonym_conflict",
            "context": context,
            "reason": "Matches synonym list; may conflict semantically."
        }

    # Partial match heuristic (e.g., "mission phase" contains "mission")
    for p in protected:
        if re.search(rf"\b{re.escape(p)}\b", word):
            return {
                "status": "protected",
                "context": context,
                "reason": f"Contains protected token '{p}'."
            }

    for s in synonyms:
        if re.search(rf"\b{re.escape(s)}\b", word):
            return {
                "status": "synonym_conflict",
                "context": context,
                "reason": f"Contains synonym token '{s}'."
            }

    # Otherwise safe
    return {
        "status": "allowed",
        "context": context,
        "reason": "No match found in protected or synonym lists."
    }


def check_conflict_all(word, conflict_rules):
    """
    Run check_conflict across all defined contexts and return all conflicts found.
    """
    results = []
    for ctx in conflict_rules.keys():
        result = check_conflict(word, ctx, conflict_rules)
        if result["status"] in ("protected", "synonym_conflict"):
            results.append(result)
    return results or [{
        "status": "allowed",
        "context": None,
        "reason": "No conflicts found in any context."
    }]


def validate_term(term, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode=False):
    """
    Validate a term against protected terms and conflict rules.
    Supports composite contexts separated by '+' (e.g. "mission_ops+vehicle").
    """

    term_lower = term.lower().strip()
    contexts = [c.strip().lower() for c in context.split("+")]

    all_results = []
    final_status = "valid"
    final_details = []

    # --- Global protection check
    if term_lower in PROTECTED_TERMS:
        return {
            "term": term,
            "context": context,
            "status": "protected",
            "details": f"Globally protected technical term – must not be replaced."
        }

    # --- Check each context individually
    for ctx in contexts:
        if ctx not in CONFLICT_RULES:
            continue

        rules = CONFLICT_RULES[ctx]
        protected = [p.lower() for p in rules.get("protected", [])]
        synonyms = [s.lower() for s in rules.get("synonyms", [])]

        if term_lower in protected:
            all_results.append(("protected", f"Protected within context '{ctx}'."))
        elif term_lower in synonyms:
            all_results.append(("conflict", f"Potential synonym conflict in context '{ctx}'."))

        # Strict mode: check cross-context conflicts
        if strict_mode:
            cross_conflicts = []
            for other_ctx, other_rules in CONFLICT_RULES.items():
                if other_ctx == ctx:
                    continue
                if term_lower in [p.lower() for p in other_rules.get("protected", [])]:
                    cross_conflicts.append(other_ctx)
            if cross_conflicts:
                all_results.append(("cross_conflict", f"Protected in other context(s): {', '.join(cross_conflicts)}"))

    # --- Domain-level synonym conflict
    for key, syns in DOMAIN_SYNONYMS.items():
        all_syns = [s.lower() for s in syns] + [key.lower()]
        if term_lower in all_syns and key.lower() in PROTECTED_TERMS:
            all_results.append(("conflict", f"'{term}' overlaps with protected domain term '{key}'."))

    # --- Merge results
    if all_results:
        # Priority: protected > cross_conflict > conflict > valid
        priorities = {"protected": 3, "cross_conflict": 2, "conflict": 1, "valid": 0}
        final_status = max(all_results, key=lambda x: priorities[x[0]])[0]
        final_details = [msg for _, msg in all_results]
    else:
        final_status = "valid"
        final_details = ["No conflict detected."]

    return {
        "term": term,
        "context": context,
        "status": final_status,
        "details": "; ".join(final_details)
    }


def batch_validate(terms, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode=False):
    """
    Batch-validate a list of terms against protected and conflict rules.
    Handles composite contexts (split by '+').

    Args:
        terms (list[str]): Words or phrases to validate.
        context (str): Active domain context (or composite, e.g. 'mission_ops+vehicle').
        CONFLICT_RULES (dict): Context rules containing 'protected' and 'synonyms' lists.
        PROTECTED_TERMS (set): Globally protected terms.
        DOMAIN_SYNONYMS (dict): Domain-level synonym mappings.
        strict_mode (bool): If True, check for cross-context conflicts.

    Returns:
        dict: {
            "context": str,
            "results": [ {"term": str, "status": str, "details": str} ],
            "summary": {"valid": int, "protected": int, "conflict": int, "cross_conflict": int}
        }
    """

    results = []
    summary = {"valid": 0, "protected": 0, "conflict": 0, "cross_conflict": 0}

    for term in terms:
        result = validate_term(term, context, CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS, strict_mode)
        results.append({
            "term": result["term"],
            "status": result["status"],
            "details": result["details"]
        })
        summary[result["status"]] = summary.get(result["status"], 0) + 1

    return {
        "context": context,
        "results": results,
        "summary": summary
    }


def get_domain_synonyms(word: str, domain_synonyms: dict[str, list[str]], protected_terms: set[str]) -> list[str]:
    """
    Retrieve domain-appropriate synonyms from a controlled dictionary with protection logic.

    This function:
      - Checks for protected terms (and submatches of multi-word protected terms)
      - Normalizes input for consistent matching
      - Handles plural/singular equivalence
      - Ensures synonyms are safe and not recursive
      - Optionally supports fuzzy matching for near-misses (e.g. 'depths' → 'depth')

    :param word: The word or phrase to find synonyms for.
    :param domain_synonyms: Controlled mapping of domain terms → synonym lists.
    :param protected_terms: Set of protected technical or domain-specific terms.
    :return: List of domain-safe synonyms, or [] if none is found, or the term is protected.
    """

    # Normalize input
    word_norm = word.strip().lower()

    # --- Protected term check (exact and substring) ---
    if any(re.fullmatch(rf"\b{re.escape(term)}\b", word_norm) or term in word_norm for term in protected_terms):
        return []

    # --- Direct lookup ---
    if word_norm in domain_synonyms:
        return domain_synonyms[word_norm]

    # --- Reverse lookup (catch when a synonym is used instead of its key) ---
    for key, syns in domain_synonyms.items():
        if word_norm in [s.lower() for s in syns]:
            return [key] + [s for s in syns if s.lower() != word_norm]

    # --- Singular/plural normalization ---
    # Handle simple plural/singular equivalence (e.g., "routes" → "route")
    if word_norm.endswith('s') and word_norm[:-1] in domain_synonyms:
        return domain_synonyms[word_norm[:-1]]
    elif f"{word_norm}s" in domain_synonyms:
        return domain_synonyms[f"{word_norm}s"]

    # --- Fuzzy fallback (minor typo tolerance) ---
    # For example, "acqusition" → "acquisition"
    for key in domain_synonyms.keys():
        if abs(len(key) - len(word_norm)) <= 2 and sum(a != b for a, b in zip(key, word_norm)) <= 2:
            return domain_synonyms[key]

    # --- No match found ---
    return []


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
def _extract_lemma_synonyms(synset, original_word: str) -> set[str]:
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
    The function uses WordNet synsets to extract synonyms. Each synonym is
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
        if DOMAIN_SPECIFIC:
            synonyms = get_domain_synonyms(words[idx], DOMAIN_SYNONYMS, PROTECTED_TERMS)
        else:
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

    augmented_text = ' '.join(augmented_words)

    # Optionally, apply minimal structural variation
    if random.random() < 0.3:  # 30% chance of structural change
        augmented_text = augment_text_structural(augmented_text)

    return augmented_text


# --- Simple keyword-based classification ---
# Since transformers may also have dependency issues, implementing a keyword-based classifier
def _calculate_category_scores(text_lower: str, category_keywords: dict[str, list[str]]) -> dict[str, int]:
    """
    Calculates category scores based on keyword matches in text.

    Args:
        text_lower (str): Input text (lowercased).
        category_keywords (dict): Mapping of category -> keyword list.

    Returns:
        dict[str, int]: Scores per category.
    """
    scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[category] = score
    return scores


def classify_response(text: str, taxonomy: dict) -> tuple[str, dict[str, float]]:
    """
    Classify a response into a taxonomy category using keywords from taxonomy.

    Args:
        text (str): Response text.
        taxonomy (dict): Full taxonomy dict.

    Returns:
        tuple[str, dict[str, float]]:
            - predicted category
            - normalized scores
    """
    text_lower = text.lower()

    # Build category keywords dynamically from taxonomy
    category_keywords = build_category_keywords(taxonomy, lowercase=True, deduplicate=True)

    # Score categories
    scores = _calculate_category_scores(text_lower, category_keywords)

    # Pick best-scoring category
    predicted_category = max(scores, key=scores.get, default='uncategorised')
    if scores.get(predicted_category, 0) == 0:
        predicted_category = 'uncategorised'

    # Normalize scores
    total_score = sum(scores.values()) if sum(scores.values()) > 0 else 1
    normalized_scores = {k: v / total_score for k, v in scores.items()}

    return predicted_category, normalized_scores


def load_and_filter_responses(file_path: str, scenario_number: int, question_prefix: str) -> pd.DataFrame:
    """
    Loads response data from CSV and filters by scenario and question criteria.

    :param file_path: Path to the input CSV file containing responses.
    :param scenario_number: Scenario number to filter responses.
    :param question_prefix: Question prefix to filter responses.
    :return: Series of filtered response texts.
    :raises FileNotFoundError: If the input CSV file is not found.
    """
    df = pd.read_csv(file_path)
    # df = pd.DataFrame(response_set)
    df['ResponseText'] = df['ResponseText'].str.replace(r'[/]', ' ', regex=True)

    filtered_df = df[
        (df['ScenarioNumber'] == scenario_number) &
        (df['Question'].str.startswith(question_prefix))
        ][['ResponseID','ResponseText']]

    return filtered_df


def preprocess_responses(response_series: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and deduplicates response texts.

    :param response_series: Series of raw response texts.
    :return: List of preprocessed unique responses.
    """
    processed_rows = []
    for _, row in response_series.iterrows():
        if isinstance(row['ResponseText'], str) and row['ResponseText'].strip() != '':
            cleaned_text = preprocess(row['ResponseText'])
            processed_rows.append({'ResponseID': row['ResponseID'], 'ResponseText': cleaned_text})
    return pd.DataFrame(processed_rows).drop_duplicates(subset=['ResponseText'], keep='first')


def augment_response_dataset(responses: pd.DataFrame, aug_prob: float = 0.3,
                             max_synonyms: int = 2, sample_size: int = 500) -> pd.DataFrame:
    """
    Augments response dataset using synonym replacement and samples the result.
    Maintains ResponseID throughout the augmentation process.

    :param responses: DataFrame with ResponseID and ResponseText columns.
    :param aug_prob: Probability of word augmentation.
    :param max_synonyms: Maximum number of synonyms to use per word.
    :param sample_size: Maximum number of responses to the sample.
    :return: DataFrame of augmented and sampled responses with ResponseIDs preserved.
    """
    augmented_rows = []
    # Create both original and augmented versions for each response
    for _, row in responses.iterrows():
        response_id = row['ResponseID']
        response_text = row['ResponseText']
        # Add original response
        augmented_rows.append({
            'ResponseID': response_id,
            'ResponseText': response_text
        })
        # Add an augmented version
        augmented_text = augment_text(response_text, aug_prob=aug_prob, max_synonyms=max_synonyms)
        augmented_rows.append({
            'ResponseID': response_id,
            'ResponseText': augmented_text
        })

    # Convert to DataFrame
    augmented_df = pd.DataFrame(augmented_rows)

    # Sample if needed (maintaining ResponseID association)
    if len(augmented_df) > sample_size:
        augmented_df = augmented_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return augmented_df


def classify_responses(responses: pd.DataFrame, taxonomy: dict ) -> list[dict]:
    """
    Classify responses with optional validation flags attached.
    """
    results = []
    for _, resp in responses.iterrows():
        resp_id = resp['ResponseID']
        text = resp['ResponseText']
        predicted_category, all_scores = classify_response(text, taxonomy)

        results.append({
            'ResponseID': resp_id,
            'response': text,
            'predicted_category': predicted_category,
            'all_scores': str(all_scores),
            # --- carry validation flags if present ---
            'has_protected': resp.get('has_protected', False),
            'has_conflict': resp.get('has_conflict', False),
            'has_cross_conflict': resp.get('has_cross_conflict', False),
            'validation_summary': str(resp.get('validation_summary', {}))
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


def main(question: str, scenario: int, taxonomy: dict, merge_to_source: bool = False, strict_mode: bool = True):
    """
    Extended main pipeline:
      1. Load and filter responses
      2. Preprocess responses
      3. Augment dataset
      4. Validate responses (protected/conflict terms)
      5. Classify validated responses
      6. Save classification results
      7. Optionally merge results to source
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

    # --- Run validation before classification ---
    context = map_question_to_context(question)
    validation_reports = {}

    enriched_responses = []
    for _, row in responses_aug.iterrows():
        text = row['ResponseText']
        report = document_validate(
            text=text,
            context=context,
            CONFLICT_RULES=CONFLICT_RULES,
            PROTECTED_TERMS=PROTECTED_TERMS,
            DOMAIN_SYNONYMS=DOMAIN_SYNONYMS,
            strict_mode=strict_mode
        )
        flags = extract_validation_flags(report)

        enriched_responses.append({
            "ResponseID": row["ResponseID"],
            "ResponseText": text,
            **flags
        })

    validated_df = pd.DataFrame(enriched_responses)

    # Save validation logs per scenario/question
    validation_output_path = f"./output/S{scenario}_{question}_validation_summary.json"
    pd.DataFrame(validation_reports).to_json(validation_output_path, orient="records", indent=2)
    print(f"Validation report saved: {validation_output_path}")

    # --- Classify only validated responses ---
    results = classify_responses(validated_df, taxonomy)

    classification_output_path = f"./output/S{scenario}_{question}_classification_results.csv"
    save_and_display_results(results, classification_output_path)

    # Create a summary report
    summary_output_path = f"./output/S{scenario}_{question}_classification_summary.csv"
    save_summary_report(results, summary_output_path)

    # Optionally merge classifications to source
    if merge_to_source:
        print("\n" + "=" * 60)
        print("Merging ALL classifications back to source data...")
        print("=" * 60)
        merge_all_classifications()


def extract_validation_flags(report: dict) -> dict:
    """
    Extracts simple boolean flags from a validation report summary.
    """
    summary = report.get("summary", {})
    return {
        "has_protected": summary.get("protected", 0) > 0,
        "has_conflict": summary.get("conflict", 0) > 0,
        "has_cross_conflict": summary.get("cross_conflict", 0) > 0,
        "validation_summary": summary
    }


def merge_classifications_to_source(
        classification_results_path: str,
        original_data_path: str,
        output_path: str
) -> None:
    """
    Merges classification results back into the original normalised responses dataset.
    Handles multiple classifications per response without duplicates.

    :param classification_results_path: Path to classification results CSV.
    :param original_data_path: Path to original normalised responses CSV.
    :param output_path: Path to save merged output CSV.
    """
    # Load classification results
    classifications_df = pd.read_csv(classification_results_path)

    # Load original normalized data
    original_df = pd.read_csv(original_data_path)

    # Group by ResponseID and aggregate unique categories
    # This handles the case where augmentation created multiple entries per ResponseID
    aggregated_classifications = classifications_df.groupby('ResponseID').agg({
        'predicted_category': lambda x: '|'.join(sorted(set(x))),  # Unique categories separated by |
        'all_scores': 'first'  # Take first scores (they should be similar for the same ResponseID)
    }).reset_index()

    # Rename columns for clarity
    aggregated_classifications.rename(columns={
        'predicted_category': 'PredictedCategories'
    }, inplace=True)

    # Merge back to original data
    merged_df = original_df.merge(
        aggregated_classifications[['ResponseID', 'PredictedCategories']],
        on='ResponseID',
        how='left'
    )

    # Fill NaN values for responses that weren't classified
    merged_df['PredictedCategories'].fillna('not_classified', inplace=True)

    # Save to a new file
    merged_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Merged classifications saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(merged_df)}")
    print(f"Classified responses: {len(merged_df[merged_df['PredictedCategories'] != 'not_classified'])}")
    print(f"Unclassified responses: {len(merged_df[merged_df['PredictedCategories'] == 'not_classified'])}")

    # Show the distribution of classification patterns
    print(f"\nClassification patterns:")
    category_counts = merged_df['PredictedCategories'].value_counts().head(10)
    print(category_counts)

    # Count responses with multiple categories
    multi_category = merged_df[merged_df['PredictedCategories'].str.contains('\|', na=False)]
    print(f"\nResponses with multiple categories: {len(multi_category)}")


def batch_merge_all_scenarios(question: str, scenarios: list[int]) -> None:
    """
    Merges classification results for multiple scenarios into the source data.

    :param question: Question identifier (e.g. "Q1")
    :param scenarios: List of scenario numbers to process
    """
    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Processing Scenario {scenario}, Question {question}")
        print(f"{'=' * 60}")

        classification_results_path = f"./output/S{scenario}_{question}_classification_results.csv"
        original_data_path = './input/normalised_all_responses.csv'
        output_path = f"./output/S{scenario}_{question}_normalised_classified_responses.csv"

        try:
            merge_classifications_to_source(
                classification_results_path=classification_results_path,
                original_data_path=original_data_path,
                output_path=output_path
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not process Scenario {scenario} - {e}")
            continue


def create_master_classified_file(question: str, scenarios: list[int]) -> None:
    """
    Creates a single master file with classifications from all scenarios.
    Each scenario's classifications are added as separate columns.

    :param question: Question identifier (e.g. "Q1")
    :param scenarios: List of scenario numbers to include
    """
    # Load original data
    master_df = pd.read_csv('./input/normalised_all_responses.csv')

    # Merge classifications from each scenario
    for scenario in scenarios:
        classification_path = f"./output/S{scenario}_{question}_classification_results.csv"

        try:
            # Load and aggregate classifications for this scenario
            classifications_df = pd.read_csv(classification_path)

            # Aggregate unique categories per ResponseID
            aggregated = classifications_df.groupby('ResponseID').agg({
                'predicted_category': lambda x: '|'.join(sorted(set(x)))
            }).reset_index()

            # Rename column with scenario number
            column_name = f'S{scenario}_{question}_Categories'
            aggregated.rename(columns={'predicted_category': column_name}, inplace=True)

            # Merge to master
            master_df = master_df.merge(
                aggregated,
                on='ResponseID',
                how='left'
            )

            # Fill NaN
            master_df[column_name].fillna('', inplace=True)

            print(f"Added classifications for Scenario {scenario}")

        except FileNotFoundError:
            print(f"Warning: Classification file not found for Scenario {scenario}")
            continue

    # Save the master file
    output_path = './output/normalised_all_classified_responses.csv'
    master_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Master classified file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nColumns added:")
    for scenario in scenarios:
        col_name = f'S{scenario}_{question}_Categories'
        if col_name in master_df.columns:
            print(f"  - {col_name}")

    print(f"\nTotal responses: {len(master_df)}")


def merge_all_classifications_single_column(
        classification_files: list[tuple[int, str]],
        original_data_path: str,
        output_path: str
) -> None:
    """
    Merges all classification results into the original data with a single classification column.
    Each response belongs to only one Scenario+Question combination, so only one classification applies.

    Updates version control fields for classified responses:
    - Classification: The predicted category/categories
    - Version: Updated to "v0.2" for classified responses
    - ChangeNote: Set to "Initial classification" for classified responses
    - ChangeDate: Set to the current date for classified responses

    :param classification_files: List of tuples (scenario_number, question) to process.
    :param original_data_path: Path to original normalised responses CSV.
    :param output_path: Path to save merged output CSV.
    """
    # Load original data
    original_df = pd.read_csv(original_data_path)

    current_date = datetime.now().strftime('%Y-%m-%d')

    # Initialise classification column if it doesn't exist
    if 'Classification' not in original_df.columns:
        original_df['Classification'] = 'not_classified'

        # Ensure version control columns exist
        if 'Version' not in original_df.columns:
            original_df['Version'] = original_df.get('Version', 'v0.1')
        if 'ChangeNote' not in original_df.columns:
            original_df['ChangeNote'] = original_df.get('ChangeNote', 'Initial merge')
        if 'ChangeDate' not in original_df.columns:
            original_df['ChangeDate'] = ''

        # Process each classification file
        for scenario, question in classification_files:
            classification_path = f"./output/S{scenario}_{question}_classification_results.csv"

            try:
                # Get the last modified date of the classification file
                file_mod_timestamp = os.path.getmtime(classification_path)
                file_mod_date = datetime.fromtimestamp(file_mod_timestamp).strftime("%Y-%m-%d")

                # Load classifications
                classifications_df = pd.read_csv(classification_path)

                # Aggregate unique categories per ResponseID
                # (handles augmentation where same ResponseID appears multiple times)
                aggregated = classifications_df.groupby('ResponseID').agg({
                    'predicted_category': lambda x: '|'.join(sorted(set(x)))
                }).reset_index()

                # For each ResponseID in this scenario+question, update the classification
                for _, row in aggregated.iterrows():
                    response_id = row['ResponseID']
                    category = row['predicted_category']

                    # Update only rows matching this ResponseID
                    mask = original_df['ResponseID'] == response_id

                    # Update classification
                    original_df.loc[mask, 'Classification'] = category

                    # Update version control fields ONLY for classified responses
                    # (i.e. not 'uncategorised' or 'not_classified')
                    if category not in ['uncategorised', 'not_classified', '']:
                        original_df.loc[mask, 'Version'] = 'v0.2'
                        original_df.loc[mask, 'ChangeNote'] = 'Initial classification'
                        original_df.loc[mask, 'ChangeDate'] = file_mod_date

                print(f"Merged classifications for Scenario {scenario}, {question} (file date: {file_mod_date})")
            except FileNotFoundError:
                print(f"Classification file not found for Scenario {scenario}, {question}")
                continue

    # Save merged data
    original_df.to_csv(output_path, index=False)

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Merged file saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(original_df)}")

    # Count classified responses (excluding 'not_classified' and 'uncategorised')
    classified = original_df[
        (original_df['Classification'] != 'not_classified') &
        (original_df['Classification'] != 'uncategorised') &
        (original_df['Classification'] != '')
        ]
    print(f"Classified responses: {len(classified)}")
    print(f"Unclassified responses: {len(original_df) - len(classified)}")

    # Show version distribution
    print(f"\nVersion distribution:")
    print(original_df['Version'].value_counts())

    # Show distribution of classification patterns
    print(f"\nTop 10 classification patterns:")
    print(original_df['Classification'].value_counts().head(10))

    # Count multi-category responses
    multi_category = original_df[original_df['Classification'].str.contains('\|', na=False)]
    print(f"\nResponses with multiple categories: {len(multi_category)}")

    # Show summary of updated records
    updated_records = original_df[original_df['Version'] == 'v0.2']
    print(f"\nRecords updated to v0.2: {len(updated_records)}")
    print(f"Classification date: {current_date}")


def merge_all_classifications():
    """
    Convenience function to merge multiple question classifications from multiple scenarios.
    """
    classification_files = [
        (1, 'Q1'),
        (2, 'Q1'),
        (3, 'Q1'),
        (1, 'Q2'),
        (2, 'Q2'),
        (3, 'Q2'),
        (1, 'Q3'),
        (2, 'Q3'),
        (3, 'Q3')
    ]

    merge_all_classifications_single_column(
        classification_files=classification_files,
        original_data_path='./input/normalised_all_responses.csv',
        output_path='./output/normalised_all_classified_responses.csv'
    )


def save_summary_report(results: list[dict], summary_path: str) -> None:
    """
    Creates a summary CSV report of classification results with validation flags.

    Args:
        results (list[dict]): Per-response classification and validation results.
        summary_path (str): Path to save the summary CSV.
    """
    df = pd.DataFrame(results)

    # Group by category and flag states
    summary = df.groupby(
        ["predicted_category", "has_protected", "has_conflict", "has_cross_conflict"]
    ).size().reset_index(name="count")

    # Save to CSV
    summary.to_csv(summary_path, index=False)

    print(f"\nSummary report saved to: {summary_path}")
    print(summary.head(10))


def validate_domain_synonyms(file_path: str) -> dict:
    """
    Validate the structure of the DOMAIN_SYNONYMS JSON file.

    Expected structure:
    {
        "term": ["synonym1", "synonym2", ...],
        ...
    }

    Rules:
    - Keys must be strings.
    - Values must be lists.
    - Each list item must be a string.
    """

    errors = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("DOMAIN_SYNONYMS file must contain a dictionary at the top level.")

    for key, value in data.items():
        if not isinstance(key, str):
            errors.append(f"Key {key} is not a string.")

        if not isinstance(value, list):
            errors.append(f"Value for key '{key}' must be a list, got {type(value).__name__}.")
            continue

        for i, item in enumerate(value):
            if not isinstance(item, str):
                errors.append(
                    f"Value at {key}[{i}] must be a string, got {type(item).__name__}."
                )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "count_terms": len(data),
        "count_synonyms": sum(len(v) for v in data.values() if isinstance(v, list))
    }


def build_category_key_map(category_keywords: dict) -> dict:
    """
    Build a normalisation map from many possible variants (camelCase, snake_case,
    no-spaces, etc.) to the canonical CATEGORY_KEYWORDS labels.

    Args:
        category_keywords (dict): The canonical category keywords dict where keys are
                                  human-readable category labels (e.g. "Terrain and Bathymetry").

    Returns:
        dict: Mapping of normalised variants → canonical labels.
    """
    key_map = {}

    for canonical in category_keywords.keys():
        base = canonical.lower()

        # Normalise by removing spaces, hyphens, underscores
        normalized = re.sub(r'[^a-z0-9]', '', base)
        key_map[normalized] = canonical

        # Variants: camelCase style, snake_case, kebab-case
        camel = ''.join(word.capitalize() for word in base.split())
        key_map[camel.lower()] = canonical  # terrainandbathymetry
        key_map[camel] = canonical          # TerrainAndBathymetry

        snake = '_'.join(base.split())
        key_map[snake] = canonical          # terrain_and_bathymetry

        kebab = '-'.join(base.split())
        key_map[kebab] = canonical          # terrain-and-bathymetry

    return key_map


def normalise_category_keys(imported_dict: dict, category_keywords: dict) -> dict:
    """
    Remap imported category keys (in arbitrary formatting) to the canonical keys
    from CATEGORY_KEYWORDS.

    Args:
        imported_dict (dict): Imported dictionary with potentially inconsistent keys.
        category_keywords (dict): Canonical CATEGORY_KEYWORDS dict.

    Returns:
        dict: Dictionary with keys remapped to canonical labels.
    """
    key_map = build_category_key_map(category_keywords)

    def normalise_key(key: str) -> str:
        return re.sub(r'[^a-z0-9]', '', key.lower())

    normalized_dict = {}
    for k, v in imported_dict.items():
        norm_k = normalise_key(k)
        mapped_key = key_map.get(norm_k, k)  # fallback to original if not found
        normalized_dict[mapped_key] = v

    return normalized_dict


def humanise_key(key: str) -> str:
    """
    Convert taxonomy keys like 'MissionPlanningTaxonomy'
    into 'Mission Planning Taxonomy'.
    Handles CamelCase, PascalCase, snake_case, and kebab-case.
    """
    # Replace underscores and hyphens with spaces
    key = key.replace("_", " ").replace("-", " ")
    # Add spaces before capital letters (but not at start)
    key = re.sub(r'(?<!^)(?=[A-Z])', ' ', key)
    # Normalise whitespace and title case
    key = re.sub(r'\s+', ' ', key).strip()
    return key


if __name__ == "__main__":
    # Define keywords for each category as a module constant
    # Load the JSON file
    with open('./docs/mission_planning_taxonomy.json', 'r') as f:
        marine_planning_taxonomy_raw = json.load(f)

    # Extract the actual taxonomy data from the wrapper
    if "MissionPlanningTaxonomy" in marine_planning_taxonomy_raw:
        marine_planning_taxonomy = marine_planning_taxonomy_raw["MissionPlanningTaxonomy"]
    else:
        marine_planning_taxonomy = marine_planning_taxonomy_raw

    CATEGORY_KEYWORDS = {humanise_key(k): v for k, v in marine_planning_taxonomy.items()}
    # Debug print
    print("\nCategory Keyword Summary")
    print("==========================")
    for cat, kws in CATEGORY_KEYWORDS.items():
        print(f"{cat}: {len(kws)} keywords")

    # Load DOMAIN_SYNONYMS from JSON file
    report = validate_domain_synonyms("./docs/domain_synonyms.json")
    print("\nDomain Synonym Validation Report:")
    print("====================================")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Terms: {report['count_terms']}")
    print(f"  Synonyms: {report['count_synonyms']}")
    if not report["is_valid"]:
        print("\nErrors found:")
        for err in report["errors"]:
            print(f"  - {err}")

    with open('./docs/domain_synonyms.json', 'r', encoding='utf-8') as f:
        DOMAIN_SYNONYMS = {k.lower(): [s.lower() for s in v] for k, v in json.load(f).items()}
    print("Loaded terms:", len(DOMAIN_SYNONYMS))
    print("Sample mission synonyms:", DOMAIN_SYNONYMS.get("mission"))
    # Run classification for all scenarios
    for scenario_num in [1, 2, 3]:
        print(f"\n{'#' * 60}")
        print(f"# Processing Scenario {scenario_num}")
        print(f"{'#' * 60}\n")
        main("Q3", scenario_num, CATEGORY_KEYWORDS, merge_to_source=False)

    # After all scenarios are classified, merge into a single file
    print(f"\n{'#' * 60}")
    print(f"# Merging all classifications into master file")
    print(f"{'#' * 60}\n")
    merge_all_classifications()

    # Save a suppression report
    if len(SUPPRESSION_LOG) > 0:
        suppression_log_path = './output/suppression_log.csv'
        with open(suppression_log_path, 'w') as f:
            f.write('\n'.join(SUPPRESSION_LOG))
        print(f"\nSuppression log saved to: {suppression_log_path}")

