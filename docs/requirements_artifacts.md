## Typical Software development artefacts

For simplicity assume the UUAV prototype consists for the following:

- Data (raw, processed, metadata)

- Services (APIs, external/internal integrations)

- Workflows (process flows, automation, orchestration)

- UI/UX (mockups, screens, assets, accessibility)

- Architecture (components, infrastructure, configs)

- Ops/Governance (monitoring, security, compliance, testing, docs)

When these are mapped against the requirements and taxonomy we get the following observations:

- Heavy Data requirements: AURRP-01, 02, 03, 04, 09, 11, 14, 16, 17, 18, 19 → almost two-thirds of requirements produce or depend on data artifacts.

- Services appear less frequently: Only explicit in sensor interfaces (02), environmental ingestion (04), data fusion (09), and flexibility (13).

- Workflows dominate mission planning: Route planning, optimisation, user suitability, productivity enhancement.

- UI/UX is modest but crucial: AURRP-05 (LLM interactions), AURRP-06 (productivity), AURRP-07 (suitability).

- Architecture & Ops/Governance are everywhere: nearly every requirement has architectural or governance implications — shows this system is safety/security critical.

- Documentation emerges explicitly in AI-related requirements (AURRP-14, AURRP-19).