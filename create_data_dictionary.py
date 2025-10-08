import pandas as pd

dtype_overrides = {
    "FollowupNeeded": "boolean",   # use pandas nullable boolean type
    "Status": "string",
    "Version": "string",
    "ChangeNote": "string",
}

column_descriptions = {
    "ResponseID": "Unique identifier for each response record.",
    "Respondent": "Anonymized identifier for the subject matter expert (SME).",
    "Scenario": "UUAV Scenario name",
    "ScenarioNumber": "Sequential number representing the scenario order.",
    "ScenarioLabel": "Descriptive label of the scenario used in the RFI.",
    "Question": "Full text of the question (Q1–Q6).",
    "QuestionType": "Category of the question, such as RouteEvaluation, RiskEvaluation, or DataRecord.",
    "ResponseText": "The raw text response provided by the SME for the given question.",
    "SourceDate": "Date Excel file received from RAM/SME. (ISO format).",
    "FollowupNeeded": "Indicates whether further clarification or follow-up with the SME is required.",
    "Status": "Status of the record (e.g., Draft, Reviewed, Final).",
    "Version": "Version label assigned to this merged data batch.",
    "ChangeNote": "Short note describing what changed in this merge or version.",
    "Classification": "Classification results for each scenario and question. Only included in classified outputs.",
    "ChangeDate": "Date of last change to this record. Only included when first change is made."
}


# Load the merged data
df = pd.read_csv("./output/normalised_all_classified_responses.csv", dtype=dtype_overrides)

# Generate auto summary
data_dict = pd.DataFrame({
    "FieldName": df.columns,
    "DataType": [str(df[col].dtype) for col in df.columns],
    "NonNullCount": [df[col].notnull().sum() for col in df.columns],
    "UniqueCount": [df[col].nunique(dropna=True) for col in df.columns],
    "ExampleValue": [df[col].dropna().iloc[0] if df[col].notna().any() else "" for col in df.columns],
    "Description": [column_descriptions.get(col, "") for col in df.columns],
})

# Save draft dictionary
data_dict.to_csv("./input/rfi_response_data_dictionary_draft.csv", index=False)

print("Draft data dictionary saved as data_dictionary_draft.csv")
