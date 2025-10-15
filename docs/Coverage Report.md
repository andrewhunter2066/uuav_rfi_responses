## Coverage Insights—Requirements Coverage by Domain

### Well(?) mapped domains:

- NavigationAndPositioning (AURRP-01, 03, 15, 16, 18)

- TerrainAndBathymetry (AURRP-01, 02, 16, 18)

- EnvironmentalAndOceanographicConditions (AURRP-01, 02, 04, 16, 17)

- DataProductsAndRequirements (AURRP-02, 04, 05, 09, 11, 13, 19)

- MissionAuditAndTraceability (AURRP-02, 05, 06, 07, 11, 14, 19)

- StandardsAndGovernance (AURRP-05, 07, 10, 12, 13, 14, 19)

### Light coverage so far:

- HistoricalAndContextualData (only AURRP-09 — via legacy fusion).

- IntegrationAndFeedback (only AURRP-06).

### Unmapped top-level domains (still 0):

- CommunicationsAndControl

- OperationalLogistics

- TestAndEvaluationScenarios

---

## Domain-Centric Coverage

### Best covered domains:

- StandardsAndGovernance (39%) – seven requirements touch this.

- DataProductsAndRequirements (33%) – lots of sensor/data/AI-related requirements.

- MissionAuditAndTraceability (33%) – driven by provenance and AI self-documentation.

- NavigationAndPositioning (28%) – strong, but many navigation sub-concepts unmapped.

- EnvironmentalAndOceanographicConditions (28%) – well covered via environmental ingestion and uncertainty.

### Light coverage domains:

- HistoricalAndContextualData (6%) – only touched by data fusion/legacy.

- IntegrationAndFeedback (6%) – only touched by productivity enhancement.

### Uncovered domains:

- CommunicationsAndControl (0%)

- OperationalLogistics (0%)

- TestAndEvaluationScenarios (0%)

## Implications

This shows clear gaps in requirement coverage:

- No current requirements address comms, field logistics, or test/eval scenarios.

- Historical data re-use and post-mission feedback loops are barely covered.

- Future opportunity to raise new requirements or clarify existing ones to ensure these domains aren’t missed in system design!!