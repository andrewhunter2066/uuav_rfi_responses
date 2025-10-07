import pandas as pd
import json

# Load the merged data
df = pd.read_csv("./input/rfi_response_data_dictionary_draft.csv")

schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "RFI Merged Data Schema",
    "type": "object",
    "properties": {}
}

for _, row in df.iterrows():
    field = row["FieldName"]
    dtype = row["DataType"]
    field_type = "string"
    if "int" in dtype: field_type = "integer"
    elif "float" in dtype: field_type = "number"
    elif "bool" in dtype: field_type = "boolean"

    schema["properties"][field] = {
        "type": field_type,
        "description": row["Description"]
    }

with open("./input/rfi_response_data_dictionary_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("JSON Schema written → data_dictionary_schema.json")