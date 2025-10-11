#!/usr/bin/env python3

from classification_workflow import (
    validate_term,
    batch_validate,
    document_validate,
    CONFLICT_RULES,
    PROTECTED_TERMS,
    DOMAIN_SYNONYMS
)

# --- Expected results table for sanity check ---
EXPECTED_RESULTS = {
    ("bathymetry", "terrain_bathymetry"): "protected",       # In protected list
    ("time", "temporal"): "conflict",                       # Synonym conflict
    ("path", "navigation_and_positioning"): "conflict",     # Synonym conflict
    ("accuracy", "performance"): "cross_conflict",          # Protected in navigation
    ("mission", "mission_ops"): "protected",                # Protected
    ("vehicle", "platform"): "protected",                   # Protected
    ("fusion", "data_processing"): "valid",                 # Allowed domain synonym
    ("speed", "performance"): "conflict",                   # Ambiguous in performance
    ("chart", "survey"): "cross_conflict",                  # Cross-domain clash
    ("unknown", "general"): "valid"                         # Should pass
}


def run_single_term_tests():
    print("\n=== Single Term Validation Tests ===")
    for (term, ctx), expected in EXPECTED_RESULTS.items():
        result = validate_term(
            term, ctx,
            CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS,
            strict_mode=True
        )
        actual = result["status"]
        status_flag = "✅" if actual == expected else "❌"
        print(f"{status_flag} Term: '{term}' | Context: {ctx} | Expected: {expected} | Got: {actual}")
        if actual != expected:
            print(f"    Details: {result['details']}")


def run_batch_tests():
    terms = ["bathymetry", "time", "mission", "fusion", "accuracy", "unknown"]
    context = "temporal+terrain_bathymetry+performance"

    print("\n=== Batch Validation Test ===")
    batch_result = batch_validate(
        terms,
        context,
        CONFLICT_RULES, PROTECTED_TERMS, DOMAIN_SYNONYMS,
        strict_mode=True
    )
    for r in batch_result["results"]:
        print(r)
    print("Summary:", batch_result["summary"])


def run_document_test():
    test_text = """
    The mission duration depends on bathymetry and current conditions.
    Accuracy of navigation is critical, and timing must align with mission phase.
    Redundancy should be applied for safety, and data fusion improves reliability.
    """
    context = "mission_ops+temporal+navigation_and_positioning+risk+data_processing"

    print("\n=== Document Validation Test ===")
    report = document_validate(
        text=test_text,
        context=context,
        CONFLICT_RULES=CONFLICT_RULES,
        PROTECTED_TERMS=PROTECTED_TERMS,
        DOMAIN_SYNONYMS=DOMAIN_SYNONYMS,
        strict_mode=True
    )

    for r in report["results"]:
        print(r)
    print("Summary:", report["summary"])


if __name__ == "__main__":
    run_single_term_tests()
    run_batch_tests()
    run_document_test()
