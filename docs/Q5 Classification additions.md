## Mission Planning Taxonomy – Q5 Amendments

This document summarises the changes applied to the Mission Planning Taxonomy based on the responses to Q5: “When evaluating risks, what specific risks should the system flag?”

### New Parent Category
#### AuditAndAccountability

**Rationale:** Q5 responses emphasized the need for **logging**, **metadata capture**, **traceability**, and **accountability** in mission planning systems.

#### New Concepts:

* **MissionLogging**
  * logging of route decisions
  * timestamped mission events
  * mission logs
  * survey track overlays
  * approach conditions logs

* **MetadataAndRecords**
  * MBES metadata logs
  * ping rate logs
  * swath width logs
  * annotated feature detection outputs
  * annotated beach selection rationale
  * feature detection records with confidence scores

* **Accountability**
  * signatory information
  * user accounts
  * job file templates
  * initial test checklists

**RelatedTo**: DataProductsAndRequirements, ThreatsAndRiskManagement, EstimationAndUncertaintyModeling

---

### Updates to Existing Categories
**ThreatsAndRiskManagement**
* Added **FailureResponse**:
  * systematic anomaly reporting
  * automated procedure to record contacts

---

**HistoricalAndContextualData**
* Extended **AnalysisAndLearning**:
  * post-mission feedback
  * post-mission reports on anomalies
  * lessons learned
  * feedback loops into planning

---

**DataProductsAndRequirements**
*  Extended **DataIntegrityAndResilience**:
  * holes in data
  * sensor performance issues
  * depth discrepancies
  * track of interest
  * automated contact recording
 
---

**EstimationAndUncertaintyModeling**
* Extended **ConfidenceScoring**:
  * annotated confidence scores
  * confidence in beach viability

---

### Summary of Amendments

* **New parent category**: `AuditAndAccountability` (system logging, metadata, accountability).
* **ThreatsAndRiskManagement**: broader failure response including automated anomaly reporting.
* **HistoricalAndContextualData**: stronger emphasis on **feedback loops** and **post-mission learning**.
* **DataProductsAndRequirements**: resilience to **data gaps** and **discrepancies** explicitly covered.
* **EstimationAndUncertaintyModeling**: confidence scoring expanded with annotations and beach viability.

---

## Mission Planning Taxonomy – Q5 Amendments (Side-by-Side Diff)
| **Category**                         | **Before Q5**                                                                                                                                                                                                                                                                                                                                                                                                                     | **After Q5**                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NEW – AuditAndAccountability**     | ❌ *Not present*                                                                                                                                                                                                                                                                                                                                                                                                                   | ✅ **MissionLogging** (route decision logs, timestamped mission events, survey overlays, approach conditions) <br> ✅ **MetadataAndRecords** (MBES metadata logs, ping rate logs, annotated feature detection outputs, annotated beach rationale, feature detection confidence scores) <br> ✅ **Accountability** (signatory info, user accounts, job file templates, initial test checklists) |
| **ThreatsAndRiskManagement**         | Hazards (risk, loss of comms, loss of vehicle, environmental risk, adversary influence) <br> ResponseAndRecovery (abort, recovery, contingency) <br> SafetyAndMitigation (safety, redundancy, collision avoidance, cyber risk) <br> AdversarialAndDetectionRisks (detection, interference) <br> FailureResponse (actions on loss of comms/vehicle) <br> CollisionAndObstruction (collision avoidance, bottom object interference) | ➕ **FailureResponse extended**: includes *systematic anomaly reporting*, *automated procedure to record contacts*                                                                                                                                                                                                                                                                           |
| **HistoricalAndContextualData**      | MissionHistory (historical missions, archives, reports) <br> DataContext (reference data, baselines, benchmarks) <br> AnalysisAndLearning (comparative analysis, trends, historical performance, beach assessments, landing success rates)                                                                                                                                                                                        | ➕ **AnalysisAndLearning extended**: adds *post-mission feedback*, *post-mission reports on anomalies*, *lessons learned*, *feedback loops into planning*                                                                                                                                                                                                                                    |
| **DataProductsAndRequirements**      | SurveyData (survey data, coverage maps, swath width, infrastructure) <br> DataQuality (resolution, overlap, zone of confidence, validation, quality control, fusion, confidence scoring) <br> MetadataAndFormat (metadata, data format, standards, geotiff, point cloud, file format) <br> Outputs (data product, post-processing, DEM, bathymetric grid)                                                                         | ➕ **DataIntegrityAndResilience extended**: adds *holes in data*, *sensor performance issues*, *depth discrepancies*, *track of interest*, *automated contact recording*                                                                                                                                                                                                                     |
| **EstimationAndUncertaintyModeling** | UncertaintyModeling (environmental/bathymetric models, predictive modeling, variance analysis, standard deviation thresholds/limits) <br> ConfidenceScoring (feature detection confidence, classification reliability, route confidence, confidence in maintaining position, AO confidence) <br> PredictiveEstimation (predictive modeling for sonar, beach access, viability scoring, benchmarks from historical data)           | ➕ **ConfidenceScoring extended**: adds *annotated confidence scores*, *confidence in beach viability*                                                                                                                                                                                                                                                                                       |
