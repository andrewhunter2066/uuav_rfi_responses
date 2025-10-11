import json

from classification_workflow import DOMAIN_SYNONYMS  # if it's already defined somewhere

# Save to JSON
with open("docs/domain_synonyms.json", "w", encoding="utf-8") as f:
    json.dump(DOMAIN_SYNONYMS, f, indent=2, ensure_ascii=False)

print("DOMAIN_SYNONYMS saved to domain_synonyms.json")