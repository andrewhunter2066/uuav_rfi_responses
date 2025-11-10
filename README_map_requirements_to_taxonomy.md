# Requirements Mapping Module

## Overview

This module provides intelligent mapping capabilities for requirements analysis using hybrid semantic and keyword-based matching. It supports two primary mapping modes:

1. **Taxonomy Mapping** - Maps mission requirements to hierarchical taxonomy concepts
2. **Requirement-to-Requirement Mapping** - Maps mission phase requirements to project requirements

## Features

### Hybrid Matching Algorithm
- **Semantic Similarity**: Uses sentence transformers (`all-MiniLM-L6-v2`) for deep semantic understanding
- **Keyword Matching**: Employs fuzzy string matching (RapidFuzz) for lexical similarity
- **Weighted Scoring**: Configurable weights combine both approaches (default: 70% semantic, 30% keyword)

### Automated Classification
Each match is automatically classified into one of three categories:
- **Auto**: High-confidence matches meeting strict thresholds
- **Review**: Medium-confidence matches requiring human review
- **Discard**: Low-confidence matches below acceptance thresholds

### Flexible Configuration
- **AlgorithmConfig**: Tune scoring weights and classification thresholds
- **MappingConfig**: Configure input/output paths, modes, and top-N results
- Built-in validation ensures configuration consistency and file existence

### Multiple Output Formats
- **CSV**: Flat format for spreadsheet analysis and filtering
- **JSON**: Hierarchical format preserving requirement-to-matches relationships

## Architecture

### Core Components

#### Configuration Classes
- `AlgorithmConfig`: Encapsulates scoring weights and classification thresholds with validation
- `MappingConfig`: Manages execution parameters with built-in path validation and mode-specific requirements

#### Mapping Functions
- `map_requirements()`: Maps requirements to taxonomy concepts
- `map_requirements_to_requirements()`: Maps mission requirements to project requirements
- `find_best_matches()`: Core matching engine with configurable top-N results

#### Helper Functions
- `flatten_taxonomy()`: Converts hierarchical taxonomy JSON to flat DataFrame
- `_create_match_dict()`: Unified match dictionary creation for both mapping modes
- `_calculate_combined_scores()`: Applies weighted scoring algorithm
- Classification and normalization utilities

## Usage

### Basic Execution
```python
python from map_requirements_to_taxonomy import MappingConfig, AlgorithmConfig

# Taxonomy mapping with default settings
config = MappingConfig   config.execute()
# Requirement-to-requirement mapping
config = MappingConfig   config.execute()
``` 

### Custom Algorithm Configuration
```python
# Create custom thresholds for stricter matching
strict_config = AlgorithmConfig  
# Use custom config in matching
matches = find_best_matches  
``` 

## Input Format Requirements

### Requirements CSV
Must contain at minimum:
- `Requirement Statement` or `requirement_text`: The requirement text
- `id` or `ID` or `requirement_id`: Unique identifier (optional but recommended)

### Taxonomy JSON
Hierarchical structure with:
- Nested `Concepts` dictionaries
- `Description` fields for each concept
- Optional `Terms` arrays for keyword matching

### Project Requirements CSV (for reqmap mode)
- `requirement_text`: The requirement text
- `requirement_id`: Unique identifier
- `requirement_name`: Human-readable name

## Output Format

### CSV Output
Flattened rows with columns:
- `requirement_id`, `requirement_text`
- `taxonomy_domain`, `domain_concept` (taxonomy mode) or `project_requirement_id`, `project_requirement_name` (reqmap mode)
- `combined_score`, `semantic_score`, `keyword_score`
- `classification` (auto/review/discard)

### JSON Output
Hierarchical structure preserving requirement-to-matches relationships:
```json
{
  "mission_requirement_id": "L01",
  "mission_requirement_text": "The system shall support safe deployment of the UUV from a surface vessel under operational marine conditions",
  "matches": [
    {
      "project_requirement_id": "AURRP-18",
      "project_requirement_name": "Obstacle Avoidance",
      "combined_score": 0.336,
      "semantic_score": 0.293,
      "keyword_score": 43.6,
      "classification": "discard"
    }
  ]
}
```
## Dependencies

- `pandas`: DataFrame manipulation
- `sentence-transformers`: Semantic embeddings
- `rapidfuzz`: Fuzzy string matching
- `tqdm`: Progress visualization
- `pathlib`: Path handling

## Algorithm Details

### Scoring Formula
```python
combined_score = (semantic_score × semantic_weight) + (keyword_score/100 × keyword_weight)
```
### Default Thresholds
- **Auto**: combined ≥ 0.75 OR (semantic ≥ 0.80 AND keyword ≥ 70)
- **Review**: combined ≥ 0.65 OR (semantic ≥ 0.65 AND keyword ≥ 60)
- **Discard**: Below review thresholds

## Extensibility

The module is designed for easy extension:
- Add new mapping modes by extending `MappingConfig`
- Implement custom scoring algorithms by subclassing `AlgorithmConfig`
- Support additional output formats by adding new result creation functions
- Integrate different embedding models by modifying `find_best_matches()`

