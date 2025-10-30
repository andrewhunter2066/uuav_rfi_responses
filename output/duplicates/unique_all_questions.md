# Unique Responses by Scenario

---

## Questions and Tasks

> **Q1**: What test scenarios and supporting data does the system need to provide to support mission planners in evaluating route options, based on calculable or estimable factors (e.g. time, usage)?
>
> **Q2**: What estimable factors should be considered?
>
> **Q3**: What guidance can be provided regarding the estimation approach?
>
> **Q4**: What test scenarios and supporting data does the system need to provide to support mission planners in evaluating risks for a route?
>
> **Q5**: When evaluating risks, what specific risks should the system flag?
>
> **Q6**: What test scenarios and supporting data does the system need to provide to support mission planners in recording mission planning data for future review, reuse, or refinement?
>
> **Task 3**: Investigate the quality of sounding operations during mission execution.
>
> **Task 4**: Capture 'OFFICIAL' level task identifiers and missions plans.
>

---

## Scenario 1

| Q   | QuestionType     | ResponseText                                      | PredictedParent   | PredictedChild   |
|:----|:-----------------|:--------------------------------------------------|:------------------|:-----------------|
| Q1  | Route Evaluation | dive/ascent process/time                          |                   |                  |
| Q2  | Route Evaluation | Turnaround time and coordination of multiple UUVs |                   |                  |
| Q2  | Route Evaluation | max depth of vehicle                              |                   |                  |
| Q2  | Route Evaluation | min depth of vehicle                              |                   |                  |
| Q3  | Route Evaluation | rules of engagement, line spacing                 |                   |                  |
| Q3  | Route Evaluation | Informs AO. Informs if other assets are required  |                   |                  |
| Q3  | Route Evaluation | Confidence in understanding of AO                 |                   |                  |
| Q4  | Risk Evaluation  | max depth of vehicle                              |                   |                  |
| Q4  | Risk Evaluation  | min depth of vehicle                              |                   |                  |
| Q5  | Risk Evaluation  | Potential/likelihood of collision                 |                   |                  |
| Q5  | Risk Evaluation  | Factor of damage if a grounding event occurs      |                   |                  |
| Q5  | Risk Evaluation  | Total time and location on surface                |                   |                  |
| Q6  | Data Records     | Initial test check lists (pre-dive, etc.)         |                   |                  |
| Q6  | Data Records     | Specific job files to load into a template        |                   |                  |
| Q6  | Data Records     | track of interest                                 |                   |                  |
| Q6  | Data Records     | Automated procedure to record contacts            |                   |                  |

## Scenario 2

| Q   | QuestionType     | ResponseText                          | PredictedParent   | PredictedChild   |
|:----|:-----------------|:--------------------------------------|:------------------|:-----------------|
| Q1  | Route Evaluation | max depth of vehicle                  |                   |                  |
| Q1  | Route Evaluation | active sensor collection requirements |                   |                  |
| Q1  | Route Evaluation | min depth of vehicle                  |                   |                  |

## Scenario 3

| Q   | QuestionType     | ResponseText                                                   | PredictedParent   | PredictedChild   |
|:----|:-----------------|:---------------------------------------------------------------|:------------------|:-----------------|
| Q2  | Route Evaluation | Habitat zone sensitivity and restrictions.                     |                   |                  |
| Q5  | Risk Evaluation  | Presence of marine life or protected habitat zones.            |                   |                  |
| Q5  | Risk Evaluation  | Sensor interference from bottom clutter or acoustic shadowing. |                   |                  |
| Q6  | Data Records     | Feature detection records with confidence scores.              |                   |                  |

## Scenario 4

| Q      | QuestionType     | ResponseText                                | PredictedParent   | PredictedChild   |
|:-------|:-----------------|:--------------------------------------------|:------------------|:-----------------|
| Task 3 | Route Evaluation | Data is fit for purpose but could be better |                   |                  |
| Task 4 | Risk Evaluation  | AO search                                   |                   |                  |
| Task 4 | Risk Evaluation  | Acquire target                              |                   |                  |

