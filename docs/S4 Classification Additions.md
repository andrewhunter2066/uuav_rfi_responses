## Proposed Taxonomy Extensions for Scenario 4
### 1) Route Option Evaluation (Tasks 1–3)
#### A) EstimationAndUncertaintyModeling → RiskFusionAndEffectiveness
**New Concepts**

* **DynamicRiskFusion:** ["real-time oceanographic data fusion for risk", "fusion of UUV performance metrics into risk model", "adaptive thresholds", "risk model drift monitoring", "online model recalibration"]

* **EffectivenessAssessment:** ["risk profile effectiveness", "management system effectiveness", "SOP adherence effectiveness", "feedback-driven threshold tuning", "learning from past missions"]

**RelatedTo (add)**

["HistoricalAndContextualData", "AuditAndAccountability", "MissionParametersAndObjectives"]

---

#### B) VehicleCapabilitiesAndConstraints → AdverseConditionsAndRestrictedWaters
**New Concepts**

* **AdverseConditionPerformance:** ["performance degradation in high-current zones", "shallow water handling", "acoustically complex environment effects", "stability under sea state", "propulsion efficiency vs conditions", "sensor reliability under stress"]

* **RestrictedWaterwayOps:** ["traffic separation scheme awareness", "channel width constraints", "under-keel clearance", "turning basin limits", "speed restrictions in restricted waters", "vessel traffic overlays for surface GPS fixes"]

**RelatedTo (add)**

*  ["NavigationAndPositioning", "EnvironmentalAndOceanographicConditions", "ThreatsAndRiskManagement"]

---

#### C) NavigationAndPositioning → TrafficAndSurfaceFix Risks

**New Concepts**

* **TrafficInfluence**: ["vessel traffic considerations for GPS surface fixing", "high-traffic lane risk", "shipping channel exposure", "surface fix timing windows"]

**RelatedTo (add)**

* ["ThreatsAndRiskManagement", "OperationalLogistics"]

---

#### D) DataProductsAndRequirements → SoundingOperationsQuality

**New Concepts**

* **AcquisitionGeometry**: ["UUV altitude bounds for quality", "pitch and roll envelopes", "heave/attitude impact on MBES", "environmental noise influence"]

* **BenchmarkAndQC**: ["MBES performance vs expected resolution", "DVL performance benchmarks", "coverage conformance checks", "soundings fit-for-purpose assessment"]

* **GapAndAnomalyFlags**: ["automated data gap detection", "anomaly flagging for review", "depth discrepancy detection"]

**RelatedTo (add)**

* ["EstimationAndUncertaintyModeling", "AuditAndAccountability"]

---

### 2) Risk Evaluation (Tasks 4–5)
#### E) MissionParametersAndObjectives → MissionIdentificationAndTraceability

**New Concepts**

* **TaskAndPhaseTagging**: ["timestamped task identifiers", "objective tagging per phase", "environmental overlays on mission plan", "phase-level metadata tags", "OFFICIAL classification tagging (task identifiers)"]

**RelatedTo (add)**

* ["AuditAndAccountability", "StandardsAndInteroperability"]

---

#### F) ThreatsAndRiskManagement → IncidentManagementAndNotifications

**New Concepts**

* **NotifiableIncidents**: ["notifiable incident logging", "root cause analysis", "operational impact assessment"]

* **AlertingAndDebrief**: ["automated risk alerts", "post-mission debrief triggers", "navigation drift with geospatial context", "comms dropout geotagging", "sensor failure with geospatial context"]

* **DeconflictionAndLiaison**: ["communication channel deconfliction", "liaison-driven deconfliction", "mission orders for deconfliction"]

**RelatedTo (add)**

* ["AuditAndAccountability", "CommunicationsAndControl"]

---

### 3) Data Records (Task 6)
#### G) StandardsAndInteroperability (NEW top-level node)

**Purpose**: Capture format adequacy, capacity/scalability, and interoperability (including classification level handling such as “OFFICIAL”).

**Concepts**

* **StandardsAdequacy**: ["current standards adequacy", "flexibility for emerging data types", "AI-generated insights support", "schema evolution"]

* **FormatAndContent**: ["structured and unstructured data support", "imagery support", "acoustic log support", "rich metadata for AI outputs"]

* **InteroperabilityProfiles**: ["defence data standards integration", "OEM tool compatibility", "template/job file interoperability"]

* **CapacityAndScalability**: ["multi-UUV scale-out", "high-resolution dataset capacity", "throughput & storage elasticity"]

* **ClassificationAndHandling**: ["OFFICIAL level tagging", "security markings in plans", "access controls by marking"]

**RelatedTo**

* ["DataProductsAndRequirements", "AuditAndAccountability", "MissionParametersAndObjectives"]

---

### 4) Cross-cutting Governance & People
#### H) HistoricalAndContextualData → GovernanceAndPractices

**New Concepts**

* **RiskGovernance**: ["security risk assessment usage", "standard operating procedures linkage", "governance artifacts"]

* **PeopleAndCompetency**: ["suitably qualified and experienced personnel (SQEP)", "mission commander qualifications", "role-based accountability links"]

**RelatedTo (add)**

* ["AuditAndAccountability", "EstimationAndUncertaintyModeling"]

---

### 5) Audit & Metrics Enhancements
#### I) AuditAndAccountability → EffectivenessMetrics

**New Concepts**

* **RiskSystemKPIs**: ["risk profile KPI dashboard", "management effectiveness metrics", "adaptive-learning improvement tracking"]

* **ComparativeAnalytics**: ["cross-platform comparative analysis", "across mission types benchmarking", "operational limit refinement metrics"]

**RelatedTo (add)**

* ["EstimationAndUncertaintyModeling", "VehicleCapabilitiesAndConstraints"]

---

## Change Log (Scenario 4 Updates)
| **Category**                      | **New Subcategory**             | **Concepts Added**                                                                                                                 |
| --------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| ThreatsAndRiskManagement          | RiskPerformanceAndEffectiveness | Dynamic risk profiling, adaptive learning, SOPs, OEM risk tools, human factors                                                     |
| ThreatsAndRiskManagement          | IncidentReporting               | Incident logs, comms/sensor/nav failures, alerts, debriefs, deconfliction                                                          |
| VehicleCapabilitiesAndConstraints | AdversePerformanceProfiles      | Performance degradation (currents, shallow, acoustics), UUV stability, propulsion efficiency, comparative analysis, vessel traffic |
| TerrainAndBathymetry              | SoundingOperationsQuality       | Altitude/pitch/roll effects, MBES/DVL benchmarking, anomaly/data gap flagging, fit-for-purpose assessment                          |
| AuditAndAccountability            | MissionRecordkeeping            | Timestamped identifiers, metadata tagging, Defence standards integration, AO search, target acquisition                            |
| DataProductsAndRequirements       | StandardsAndCapacity            | Adequacy of standards, flexible formats, structured/unstructured data, imagery/acoustic logs, AI insights, multi-UUV scaling       |
