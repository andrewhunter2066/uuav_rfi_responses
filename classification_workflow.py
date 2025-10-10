#!/usr/bin/env python3

import pandas as pd
import random
import re
import nltk
import os
from nltk.corpus import stopwords, wordnet
from datetime import datetime

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
}

# Regex pattern to keep only word characters, spaces, and hyphens
PUNCTUATION_PATTERN = r'[^\w\s\-]'

# Define keywords for each category as a module constant
CATEGORY_KEYWORDS = {

    "Terrain and Bathymetry": [
        "terrain", "bathymetry", "seabed", "bottom texture", "slope", "gradient",
        "topography", "complexity", "seafloor", "multibeam", "coverage", "resolution",
        "overlap", "gap", "vector electronic navigational charts", "zone of confidence",
        "bathymetric model", "surface model", "digital elevation model", "DEM",
        "depth profile", "bottom type", "substrate", "geomorphology"
    ],

    "Environmental and Oceanographic Conditions": [
        "current", "tidal", "stream", "EAC", "East Australian Current", "tide",
        "ephemeral", "wave height", "water column", "temperature", "salinity",
        "environmental", "ocean", "conditions", "hydrodynamics", "sea state",
        "wind", "drift", "surface current", "subsurface current", "turbidity",
        "pressure", "density", "conductivity", "CTD", "environmental forecast",
        "metocean", "hindcast", "oceanographic model"
    ],

    "Vehicle Capabilities and Constraints": [
        "speed", "endurance", "maximum depth", "minimum depth", "sensor range",
        "communication", "launch and recovery", "abort angles", "OEM",
        "equipment limitations", "vehicle", "range", "fitted", "parameters",
        "speed requirements", "battery capacity", "payload", "thruster",
        "autonomy", "navigation limits", "operating envelope", "mission duration",
        "power consumption", "fail-safe", "mission abort", "stability"
    ],

    "Navigation and Positioning": [
        "GPS", "Global Positioning System", "navigation", "accuracy", "fixing frequency",
        "navigational hazards", "route", "depth", "position", "electronic navigational chart",
        "no-go zones", "vector chart", "coordinates", "waypoint", "geofence", "heading",
        "bearing", "trajectory", "trackline", "positioning system", "INS", "DVL",
        "GNSS", "localization", "georeferencing", "datum", "EPSG", "coordinate reference system"
    ],

    "Mission Parameters and Objectives": [
        "mission", "objective", "task", "time-on-task", "timings", "hard left right",
        "collection requirements", "search area", "zone of operation", "area of operation",
        "survey", "planning", "transit time", "route options", "evaluation",
        "mission planning", "survey feasibility", "mission phase", "mission profile",
        "mission goal", "operational constraint", "time window", "launch window",
        "recovery window", "time synchronization", "UTC", "timestamp",
        "mission segment", "mission schedule"
    ],

    "Threats and Risk Management": [
        "threat", "cyber", "risk", "assessment", "loss of communications",
        "loss of vehicle", "safety", "abort", "no-go", "failure", "contingency",
        "recovery", "hazard", "collision avoidance", "redundancy", "fail-safe",
        "safety case", "emergency procedure", "risk mitigation", "contingency plan",
        "system failure", "environmental risk"
    ],

    "Historical and Contextual Data": [
        "historical", "previous mission", "similar mission", "reference data",
        "legacy", "archive", "past operations", "benchmark", "mission archive",
        "data provenance", "historical record", "baseline", "mission report",
        "comparative analysis", "trend", "reference dataset"
    ],

    "Data Products and Requirements": [
        "coverage maps", "survey data", "data product", "sensor data", "multibeam",
        "swath width", "resolution", "classification", "infrastructure", "overlap",
        "metadata", "data format", "supporting data", "zone of confidence",
        "data quality", "data standard", "data validation", "quality control",
        "data fusion", "post-processing", "data product specification",
        "geotiff", "point cloud", "DEM", "bathymetric grid", "file format"
    ],

    "Communications and Control": [
        "communication", "link", "telemetry", "control", "signal", "frequency",
        "loss of communications", "vehicle communication requirements", "bandwidth",
        "uplink", "downlink", "satellite", "acoustic modem", "data link",
        "network", "latency", "reliability", "comms channel", "signal loss"
    ],

    "Operational Logistics": [
        "launch", "recovery", "deployment", "transit", "area of operation",
        "survey planning", "constraints", "200 metre width", "approach route",
        "landing", "access", "launch point", "recovery location",
        "staging area", "support vessel", "crew", "equipment transport",
        "operational window", "base port", "site access", "mobilization",
        "demobilization", "mission setup", "field logistics"
    ],
}


# Domain-specific controlled synonyms (safe replacements for technical text)
DOMAIN_SYNONYMS = {
    # Navigation & positioning
    "location": ["position", "coordinates", "fix", "geo-location"],
    "path": ["route", "trajectory", "track", "course", "transit"],
    "area": ["region", "zone", "sector", "operating area", "AOI", "area of operation"],
    "navigation": ["piloting", "routing", "pathfinding", "wayfinding"],
    "hazard": ["obstacle", "danger", "no-go zone", "restriction", "constraint"],
    "positioning": ["fixing", "localization", "geolocation", "position estimate"],
    "accuracy": ["precision", "error", "uncertainty", "deviation"],
    "drift": ["offset", "position drift", "bias"],

    # Marine environment
    "water": ["marine", "aquatic", "hydrographic", "oceanic"],
    "depth": ["underwater depth", "bathymetry", "seafloor elevation"],
    "surface": ["sea surface", "ocean surface", "top layer"],
    "current": ["flow", "stream", "drift", "tide", "EAC", "East Australian Current", "circulation"],
    "tide": ["tidal stream", "tidal current", "tidal flow"],
    "wave": ["sea state", "swell", "wave height", "wave period"],
    "seabed": ["seafloor", "bottom", "substrate"],
    "gradient": ["slope", "incline", "angle"],
    "environment": ["conditions", "weather", "climate", "hydrodynamics",  "ocean conditions"],
    "temperature": ["sea temperature", "water temperature", "thermal gradient"],
    "salinity": ["salt concentration", "conductivity"],
    "turbidity": ["clarity", "suspended sediment", "visibility"],

    # Operations & planning
    "mission": ["operation", "task", "objective", "sortie", "assignment"],
    "survey": ["assessment", "examination", "mapping", "reconnaissance"],
    "data": ["information", "dataset", "records",  "telemetry", "signal data"],
    "collection": ["gathering", "acquisition", "measurement", "sampling"],
    "planning": ["preparation", "scheduling", "mission design"],
    "evaluation": ["assessment", "analysis", "review", "appraisal"],
    "operation": ["mission", "activity", "undertaking"],
    "maintenance": ["servicing", "upkeep", "repair"],
    "logistics": ["supply chain", "support", "replenishment"],
    "latency": ["delay", "lag", "transmission delay"],
    "availability": ["uptime", "readiness", "serviceability"],
    "resource": ["asset", "supply", "input"],

    # Capabilities & vehicle characteristics
    "capability": ["ability", "capacity", "performance", "functionality"],
    "limitation": ["constraint", "restriction", "bound", "threshold"],
    "requirement": ["need", "specification", "criterion", "parameter"],
    "vehicle": ["platform", "AUV", "UUV", "ROV", "vessel", "craft"],
    "sensor": ["instrument", "payload", "detector"],
    "speed": ["velocity", "cruise speed", "transit speed"],
    "endurance": ["range", "duration", "loiter time"],
    "launch": ["deployment", "release"],
    "recovery": ["retrieval", "return", "rescue", "restoration"],
    "communication": ["comms", "link", "data link", "telemetry", "signal"],
    "abort": ["terminate", "cancel", "cease operation"],
    "power": ["energy", "battery capacity", "consumption", "load"],
    "payload": ["sensor package", "instrument load", "carried sensors"],
    "autonomy": ["automation", "self-operation", "independence"],
    "stability": ["control", "balance", "steady state"],
    "maneuverability": ["agility", "handling", "turn radius"],

    # Data products & mapping
    "chart": ["map", "navigational chart", "ENC", "electronic chart"],
    "coverage": ["extent", "area covered", "swath"],
    "resolution": ["granularity", "detail", "precision"],
    "overlap": ["redundancy", "intersection"],
    "metadata": ["data descriptor", "annotation"],
    "frequency": ["update rate", "sampling rate", "temporal resolution"],
    "uncertainty": ["error", "variance", "confidence"],
    "processing": ["analysis", "filtering", "post-processing", "fusion"],
    "fusion": ["integration", "combination", "data merging"],

    # Risk & contingency
    "risk": ["hazard", "threat", "danger", "exposure", "vulnerability"],
    "threat": ["risk", "hazard", "cyber risk", "environmental risk"],
    "loss": ["failure", "breakdown", "malfunction", "outage"],
    "contingency": ["backup", "fallback", "emergency plan"],
    "safety": ["protection", "security"],
    "probability": ["likelihood", "chance", "possibility"],
    "severity": ["impact", "consequence", "magnitude"],
    "mitigation": ["control", "prevention", "response"],
    "redundancy": ["backup", "fallback", "failover"],
    "reliability": ["robustness", "dependability", "resilience"],

    # Temporal & performance
    "duration": ["timeframe", "period", "interval", "timespan",],
    "time": ["timing", "elapsed time"],
    "estimate": ["approximation", "projection", "forecast"],
    "efficiency": ["performance", "effectiveness"],
    "timing": ["schedule", "interval", "timeframe"],
    "delay": ["latency", "hold-up", "pause"],
    "performance": ["efficiency", "effectiveness", "throughput"],

    # General analytical & operational terms
    "analysis": ["assessment", "evaluation", "study", "review"],
    "model": ["simulation", "representation", "algorithm", "forecast model"],
    "system": ["platform", "application", "software"],
    "important": ["critical", "essential", "significant", "vital"],
    "needed": ["required", "necessary", "essential"],
    "provide": ["supply", "furnish", "deliver"],
    "use": ["utilize", "employ", "apply"],
    "show": ["display", "indicate", "demonstrate"],
    "large": ["substantial", "significant", "major"],
    "small": ["minimal", "limited", "minor"],
    "estimation": ["approximation", "prediction", "inference"],
    "parameter": ["variable", "input", "factor"],
    "trend": ["pattern", "tendency", "trajectory"],
    "validation": ["verification", "cross-check", "benchmarking"],
}

# Words to NEVER replace (protected terms)
PROTECTED_TERMS = {
    # --- Existing Core Terms ---
    "unmanned underwater vehicle", "uuv",
    "global positioning system", "gps",
    "multibeam echo sounder", "mbes",
    "area of operations",
    "seafloor", "seabed", "bathymetry", "sonar", "acoustic",
    "navigation", "positioning", "sensor", "endurance",
    "tide", "tidal", "current", "ephemeral",
    "depth", "range", "speed", "vehicle",
    "terrain", "topography", "gradient",
    "minimum", "maximum", "accuracy", "resolution",
    "launch", "recovery", "deployment", "transit",
    "threat", "hazard", "risk", "surveillance",

    # --- Navigation & Geospatial ---
    "waypoint", "route", "trackline", "fix", "heading",
    "bearing", "geofence", "ais", "chart datum", "coordinate reference system",
    "epsg", "gnss", "ins", "dvl", "doppler velocity log", "inertial navigation system",

    # --- Environmental & Oceanographic ---
    "salinity", "temperature", "conductivity", "ctd", "water column",
    "wave height", "sea state", "currents", "east australian current", "eac",
    "sediment", "substrate", "turbidity", "density", "pressure", "hydrography",
    "hydrodynamics", "bathymetric grid",
    "turbidity", "visibility", "swell", "surf zone",
    "sediment transport", "beach gradient", "substrate composition",
    "habitat zone", "sensitive habitat", "environmental restriction",
    "oceanographic conditions", "sea floor roughness",

    # --- Vehicle, Platform & System Terms ---
    "auv", "rov", "usv", "asv", "vessel", "autonomous surface vehicle",
    "autonomous underwater vehicle", "remotely operated vehicle",
    "launch and recovery system", "lars",
    "payload", "hull", "propulsion", "battery", "thruster", "ballast",
    "comms", "telemetry", "acoustic modem", "satellite link",
    "mission controller", "operator console",
    "battery life", "power consumption", "power budget",
    "autonomy", "maneuverability", "stability",
    "pitch", "roll", "yaw", "altitude", "attitude",
    "turnaround time", "maintenance interval",
    "payload capacity", "sensor payload",
    "mission endurance", "duty cycle", "mission tempo",

    # --- Survey & Mapping ---
    "survey line", "survey grid", "swath", "coverage map",
    "overlap", "track spacing", "line plan", "mosaic", "geotiff",
    "digital elevation model", "dem", "point cloud",
    "metadata", "data quality flag", "ping rate", "ping",

    # --- Navigation ---
    "survey speed", "survey coverage", "route safety",
    "area of operation", "zone of confidence", "no-go zone", "restricted zone",
    "navigational hazard", "submerged hazard", "route plan",

    # --- Sensor & Data Quality ---
    "sensor calibration", "calibration status", "calibration coefficient",
    "update frequency", "fix frequency", "sampling rate",
    "data quality", "data resolution", "signal to noise ratio", "snr",
    "acoustic interference", "positional uncertainty", "confidence zone",

    # --- Risk, Safety & Regulation ---
    "no-go zone", "restricted area", "exclusion zone",
    "collision avoidance", "safety case", "risk assessment", "fail-safe",
    "redundancy", "contingency plan", "risk factor", "risk likelihood", "risk severity",
    "mitigation measure", "contingency procedure", "safety margin",
    "failure mode", "critical system", "redundant system",

    # --- Temporal & Mission Management ---
    "time on task", "mission duration", "mission segment", "mission phase",
    "launch window", "recovery window", "time synchronization", "utc", "timestamp",

    # --- Analytical & Data Management ---
    "machine learning", "artificial intelligence", "ai", "ml",
    "training data", "test set", "validation data",
    "geospatial data", "netcdf", "geojson", "json-ld", "metadata schema",
    "knowledge graph", "ontology", "taxonomy", "semantic model",
    "estimation factor", "estimated parameter", "performance model",
    "error budget", "uncertainty model", "probability distribution",
    "confidence interval", "sensitivity analysis",

    # --- Organisational / Workflow & Control terms
    "mission planning system", "mission control", "control interface",
    "vehicle controller", "command and control", "c2", "telemetry link",
    "communications plan", "uplink", "downlink", "latency",

    # --- Programmatic / Org-specific Terms (found in RFI responses) ---
    "imos", "emii", "aodn", "nmea", "csv", "api", "sdk",
    "postgis", "geoserver", "arcgis", "qgis", "python", "r",
    "matlab", "pandas", "xarray", "geopandas",

    # --- Operational Areas ---
    "area of interest", "operational area", "search area",
    "launch point", "recovery point", "access corridor",
    "mission window", "deployment window", "survey corridor",
}

# Synonym–Protection conflict rules
CONFLICT_RULES = {
    # --- Temporal / Oceanographic context ---
    "temporal": {
        "protected": [
            "current", "tidal", "tide", "eac", "east australian current",
            "temporal resolution", "time series", "timestamp", "utc"
        ],
        "synonyms": ["time", "duration", "interval", "period"]
    },

    # --- Survey / Mapping context ---
    "survey": {
        "protected": [
            "survey", "multibeam echo sounder", "mbes", "coverage map",
            "swath", "line plan", "track spacing", "geotiff", "digital elevation model"
        ],
        "synonyms": ["survey", "data", "mapping", "collection"]
    },

    # --- Spatial / Geospatial context ---
    "spatial": {
        "protected": [
            "navigation", "route", "trackline", "trajectory", "heading",
            "geofence", "waypoint", "coordinate reference system", "epsg"
        ],
        "synonyms": ["path", "location", "position", "space"]
    },

    # --- Vertical / Bathymetric context ---
    "vertical": {
        "protected": [
            "depth", "bathymetry", "seafloor", "seabed", "vertical datum",
            "chart datum", "gradient", "topography"
        ],
        "synonyms": ["depth", "area", "layer", "zone"]
    },

    # --- Risk / Safety / Compliance context ---
    "risk": {
        "protected": [
            "risk", "threat", "hazard", "cyber risk", "no-go zone",
            "collision avoidance", "restricted area", "safety case", "fail-safe",
            "redundancy", "contingency plan"
        ],
        "synonyms": ["important", "limitation", "consideration", "danger"]
    },

    # --- Mission Operations context ---
    "mission_ops": {
        "protected": [
            "launch", "recovery", "transit", "endurance", "mission duration",
            "mission phase", "mission controller", "operator console"
        ],
        "synonyms": ["mission", "operation", "activity", "task"]
    },

    # --- Navigation / Positioning context ---
    "navigation": {
        "protected": [
            "gps", "gnss", "positioning", "navigation", "ins", "inertial navigation system",
            "doppler velocity log", "dvl", "fix", "bearing"
        ],
        "synonyms": ["accuracy", "location", "tracking", "movement"]
    },

    # --- Sensors / Instrumentation context ---
    "sensors": {
        "protected": [
            "sensor", "sonar", "acoustic", "telemetry", "ctd", "conductivity",
            "temperature", "pressure", "salinity", "hydrography", "hydrodynamics"
        ],
        "synonyms": ["data", "collection", "measurement"]
    },

    # --- Semantic / Data Integration context ---
    "semantic_data": {
        "protected": [
            "knowledge graph", "ontology", "taxonomy", "json-ld", "metadata schema",
            "controlled vocabulary", "data dictionary", "semantic model"
        ],
        "synonyms": ["data", "classification", "schema"]
    },

    # --- Platform / Vehicle context ---
    "platform": {
        "protected": [
            "launch and recovery system", "lars", "auv", "usv", "rov", "asv", "uuv",
            "autonomous underwater vehicle", "autonomous surface vehicle",
            "remotely operated vehicle", "vessel", "platform"
        ],
        "synonyms": ["vehicle", "capability", "system"]
    },

    # --- Power / Propulsion context (NEW) ---
    "propulsion": {
        "protected": [
            "propulsion", "thruster", "battery", "ballast", "energy", "power system"
        ],
        "synonyms": ["drive", "movement", "motor"]
    },

    # --- Communication / Network context (NEW) ---
    "comms": {
        "protected": [
            "comms", "telemetry", "acoustic modem", "satellite link",
            "api", "sdk", "network interface", "data uplink"
        ],
        "synonyms": ["communication", "connectivity", "link"]
    },

    # --- Data Science / Processing context (NEW) ---
    "data_processing": {
        "protected": [
            "machine learning", "artificial intelligence", "ai", "ml",
            "training data", "test set", "validation data",
            "python", "r", "matlab", "pandas", "xarray", "geopandas"
        ],
        "synonyms": ["analytics", "modeling", "processing"]
    },

    # --- Environmental / Oceanographic context (NEW) ---
    "environmental": {
        "protected": [
            "sea state", "wave height", "turbidity", "sediment", "substrate",
            "density", "temperature", "salinity", "water column", "eac"
        ],
        "synonyms": ["environment", "conditions", "ocean"]
    }
}

# Which method to use, domain-specific (True) or general NLP (False)
DOMAIN_SPECIFIC = True

# Suppression Log
SUPPRESSION_LOG = []


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
    # Apply one random transformation if pattern matches
    random.shuffle(transformations)
    for pattern, replacement in transformations:
        if re.search(pattern, augmented, re.IGNORECASE):
            augmented = re.sub(pattern, replacement, augmented, count=1, flags=re.IGNORECASE)
            break

    return augmented


def get_domain_synonyms(word: str, domain_synonyms: dict[str, list[str]], protected_terms: set[str]) -> list[str]:
    """
    Retrieve domain-appropriate synonyms from a controlled dictionary with protection logic.

    This function:
      - Checks for protected terms (and submatches of multi-word protected terms)
      - Normalizes input for consistent matching
      - Handles plural/singular equivalence
      - Ensures synonyms are safe and not recursive
      - Optionally supports fuzzy matching for near-misses (e.g., 'depths' → 'depth')

    :param word: The word or phrase to find synonyms for.
    :param domain_synonyms: Controlled mapping of domain terms → synonym lists.
    :param protected_terms: Set of protected technical or domain-specific terms.
    :return: List of domain-safe synonyms, or [] if none found or term is protected.
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

    # Optionally apply minimal structural variation
    if random.random() < 0.3:  # 30% chance of structural change
        augmented_text = augment_text_structural(augmented_text)

    return augmented_text


# --- Simple keyword-based classification ---
# Since transformers may also have dependency issues, implementing a keyword-based classifier
def _calculate_category_scores(text_lower: str, q: str) -> dict[str, int]:
    """
    Calculates and returns category scores based on a text and a specific query.

    This function computes a score for each category based on the occurrence of
    keywords in the given text. The category keywords are selected based on the
    provided query. The score for a category is the total count of keywords from
    that category that appear in the text.

    :param text_lower: A lowercase string to be evaluated against category
        keywords.
        Example: "this is a sample text".
    :param q: A string representing the query to determine which category
        keywords to use. Example: "Q1".
    :return: A dictionary where keys are categories (strings) and values are
        scores (integers) representing the count of matching keywords for
        each category.
    :rtype: dict[str, int]
    """
    if q[0] == "Q":
        category_keywords = CATEGORY_KEYWORDS
    else:
        raise ValueError("Invalid question. Please provide a valid question to assess.")
    scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[category] = score
    return scores


def classify_response(text: str, question: str) -> tuple[str, dict[str, float]]:
    """
    Analyses a given response text and a question to classify the response into a category
    based on calculated scores. The function returns the predicted category as well as
    the normalised scores for all potential categories.

    :param text: Input response text for classification
    :type text: str
    :param question: The reference question to help in classification
    :type question: str
    :return: A tuple containing the predicted category as a string and a dictionary of
             normalised scores for all categories.
    :rtype: tuple[str, dict[str, float]]
    """
    text_lower = text.lower()

    # Calculate scores for each category
    scores = _calculate_category_scores(text_lower, question)

    # Get the category with the highest score
    predicted_category = max(scores, key=scores.get, default='uncategorised')
    if scores.get(predicted_category, 0) == 0:
        predicted_category = 'uncategorised'

    # Normalise scores
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


def classify_responses(responses: pd.DataFrame, question: str) -> list[dict]:
    """
    Classifies a list of responses based on a given question. Each response is
    analysed to determine its predicted category and associated scores, which
    are then aggregated into a structured list of dictionaries.

    :param responses: A list of textual responses to be classified.
    :param question: A string representing the question related to the responses.
    :return: A list of dictionaries, each containing the response, its predicted
             category, and associated scores.
    """
    results = []
    for _, resp in responses.iterrows():
        resp_id = resp['ResponseID']
        text = resp['ResponseText']
        predicted_category, all_scores = classify_response(text, question)
        results.append({
            'ResponseID': resp_id,
            'response': text,
            'predicted_category': predicted_category,
            'all_scores': str(all_scores)
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


def main(question: str, scenario: int, merge_to_source: bool = False):
    """
    Executes the main process for loading, filtering, preprocessing, augmenting,
    classifying responses, and saving results based on input question and scenario.

    The function undergoes several steps:
    1. Loads and filters a dataset of responses based on input criteria.
    2. Preprocesses filtered responses for further processing.
    3. Augments the dataset by introducing controlled variations (e.g. synonyms).
    4. Classifies the augmented responses using a classification model.
    5. Saves the classification results to an output file.
    6. Optionally merges classifications back to the source data (call separately after all scenarios).

    :param question: The question is used to filter and classify responses.
    :type question: str
    :param scenario: The scenario number used for filtering and naming output files.
    :type scenario: int
    :param merge_to_source: Whether to merge ALL classifications back to source (do this after running all scenarios).
    :type merge_to_source: bool
    :return: None
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

    # Classify responses
    results = classify_responses(responses_aug, question)

    # Save and display results
    classification_output_path = f"./output/S{scenario}_{question}_classification_results.csv"
    save_and_display_results(
        results=results,
        output_path=classification_output_path
    )

    # Optionally merge all classifications (do this after running all scenarios)
    if merge_to_source:
        print("\n" + "=" * 60)
        print("Merging ALL classifications back to source data...")
        print("=" * 60)
        merge_all_classifications()


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
        'all_scores': 'first'  # Take first scores (they should be similar for same ResponseID)
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

    # Save to new file
    merged_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Merged classifications saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nTotal responses: {len(merged_df)}")
    print(f"Classified responses: {len(merged_df[merged_df['PredictedCategories'] != 'not_classified'])}")
    print(f"Unclassified responses: {len(merged_df[merged_df['PredictedCategories'] == 'not_classified'])}")

    # Show distribution of classification patterns
    print(f"\nClassification patterns:")
    category_counts = merged_df['PredictedCategories'].value_counts().head(10)
    print(category_counts)

    # Count responses with multiple categories
    multi_category = merged_df[merged_df['PredictedCategories'].str.contains('\|', na=False)]
    print(f"\nResponses with multiple categories: {len(multi_category)}")


def batch_merge_all_scenarios(question: str, scenarios: list[int]) -> None:
    """
    Merges classification results for multiple scenarios into the source data.

    :param question: Question identifier (e.g., "Q1")
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
    - ChangeDate: Set to current date for classified responses

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
                    # (i.e., not 'uncategorised' or 'not_classified')
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
        (3, 'Q2')
    ]

    merge_all_classifications_single_column(
        classification_files=classification_files,
        original_data_path='./input/normalised_all_responses.csv',
        output_path='./output/normalised_all_classified_responses.csv'
    )


if __name__ == "__main__":
    # Run classification for all scenarios
    for scenario_num in [1, 2, 3]:
        print(f"\n{'#' * 60}")
        print(f"# Processing Scenario {scenario_num}")
        print(f"{'#' * 60}\n")
        main("Q2", scenario_num, merge_to_source=False)

    # After all scenarios are classified, merge into single file
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

