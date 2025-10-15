| **Original Node**                                                      | **Action**        | **Merged / Renamed Into**                                                                | **Notes**                                                                                                   |
|------------------------------------------------------------------------|-------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| *AdversePerformanceProfiles*                                           | Merged            | **AdverseOperationalPerformance**                                                        | Combined with *AdverseConditionsAndRestrictedWaters* to avoid duplicate categories.                         |
| *AdverseConditionsAndRestrictedWaters*                                 | Merged            | **AdverseOperationalPerformance**                                                        | All adverse condition and restricted waters performance captured in one consolidated node.                  |
| *IncidentReporting*                                                    | Merged            | **IncidentManagementAndRecording**                                                       | Consolidated with incident logging, notifications, and recorded events.                                     |
| *IncidentManagementAndNotifications*                                   | Merged            | **IncidentManagementAndRecording**                                                       | Prevented overlapping sub-categories for incident logs.                                                     |
| *RecordedRiskEvents*                                                   | Merged            | **IncidentManagementAndRecording**                                                       | Recorded events and anomalies absorbed into incident management.                                            |
| *ProceduralRecording*                                                  | Merged            | **IncidentManagementAndRecording**                                                       | Consolidated into incident handling since “procedural” overlaps with automated contact recording.           |
| *AuditAndAccountability*                                               | Merged            | **MissionAuditAndTraceability**                                                          | Merged with *KnowledgeManagementAndTraceability* to unify mission log, audit, provenance, and traceability. |
| *KnowledgeManagementAndTraceability*                                   | Merged            | **MissionAuditAndTraceability**                                                          | All decision provenance, retention, templates, and search rolled into audit/traceability domain.            |
| *StandardsAndInteroperability*                                         | Merged            | **StandardsAndGovernance**                                                               | Merged with governance and access control for a unified standards/compliance/authority category.            |
| *GovernanceSecurityAndAccess*                                          | Merged            | **StandardsAndGovernance**                                                               | Simplified by combining identity, access, compliance, and security under standards/governance.              |
| *SoundingOperationsQuality* (duplicate)                                | Removed Duplicate | Retained under **TerrainAndBathymetry**                                                  | Removed duplicate copy from *DataProductsAndRequirements* to avoid redundancy.                              |
| *PerformanceConstraints* vs *AdversePerformanceProfiles*               | Rationalized      | **PerformanceConstraints** (kept minimal) + **AdverseOperationalPerformance** (combined) | Clarified difference: baseline vs stressed performance.                                                     |
| *RiskPerformanceAndEffectiveness* + *RiskFusionAndEffectiveness*       | Merged            | **RiskModelEffectiveness**                                                               | Unified into one subdomain covering adaptive learning, real-time fusion, effectiveness metrics.             |
| *ChecklistAndVerification* (Audit) + *Pre-dive checklists* (Test/Eval) | Aligned           | **ChecklistAndVerification** (Audit domain)                                              | Pre-mission checklists consistently placed under audit/verification.                                        |
| *GovernanceAndPractices* (Historical domain)                           | Folded            | **StandardsAndGovernance**                                                               | Governance, SQEP, SOP usage merged into governance/security.                                                |
| *PostMissionAnalysis* (Historical)                                     | Folded            | **IntegrationAndFeedback**                                                               | Consolidated into integration/feedback loop to avoid overlap with analysis & learning.                      |

### Key Outcomes

1. **Reduced top-level domains** from ~15 → **12**, making the taxonomy easier to navigate.

2. **Removed duplication** (e.g. sounding quality under two domains).

3. **Clarified boundaries**:

  - Vehicle performance → one place.

  - Risk/incident logging → one place.

  - Governance/standards/security → one place.

  - Audit, accountability, traceability → one unified place.

4. **Maintained full coverage** of Scenario 1–4 responses, ensuring no concepts were lost.