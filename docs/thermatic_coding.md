# Thematic Coding

## Scenario 1

### Q1
- Terrain analysis feature detection (identify features that exceed sensor capability)
- Ocean currents
- Vehicle mobility characteristics
- Surface/subsea traffic
- Seafloor topography
- Infrastructure on route
- UUV limits - depth, range, speed, etc.
- Seafloor texture
- Cyber threat
- Environmental factors (water column)
- Launch/recovery locations
- Sensor collection requirements
- GPS requirements
- Hazards along route
- Navigation accuracy
- UUV communication requirements
- No-go zones
- Mission collection requirements
- Search areas

### Q2
- Acoustic positioning limits
- GPS availability 
- Battery life
- Turn around time (?)
- Coordination (Hive ?)
- Sensor limitations
- UUV specifications
- Navigation hazards
- No-go zones
- route confidence

### Q3
- Statistics/Variance (of what?, time, distance, etc.)
- Environmental uncertainty (of what? currents, weather, etc.)
- rules of engagement
- Surfacing risk
- planned route versus actual route
- Host vehicle availability
- Collision avoidance

### Q4
- Sensor (GPS, sonar) denied environments
- Sensor failure (DVL)
- Surfacing in high-traffic areas
- Collision avoidance
- ocean Currents
- Weather patterns
- Sea floor classification
- UUV limits
- Navigation hazards
- Route confidence

### Q5
- Navigation drift
- Environmental anomalies
- Likelihood of collision
- Battery capacity
- Acoustic performance
- Grounding risk
- Proportion of route covered
- Water column buoyancy

### Q6
- Logging of all route decisions
- Timestamp and position of mission events
- Integration with post-mission tools
- Pre-dive checks
- Appropriate security access
- Job template files
- Time on task v completion
- Data gaps
- Depth discrepancies
- Automated provenance

## Scenario 2

### Q1
- Swept channel planning
- MBES sensor coverage - consider sensor specifications
  - resolution
  - overlap
  - gaps
- Survey duration based on 
  - speed
  - swath width
  - environmental conditions
- Seabed complexity
- Surface/subsurface traffic
- Sensor specification
- Seafloor texture
- Ocean currents
- Threats to a mission
- Environmental conditions
- Launch and recovery locations
- UUV limits
- Sensor collection requirements
- GPS requirements
- Hazards along route
- Navigation accuracy
- UUV communication requirements
- No-go zones
- Mission collection requirements 

### Q2
- Swath width vs depth profile
- Speed of a survey vrs quality of a survey
- vehicle stability
- environmental factors
  - turbidity
  - seabed composition
  - acoustic interference
- Sensor calibration
- UUV limitations
- GPS limitations
- Hazards along route
- Navigation accuracy
- UUV communication requirements
- No-go zones
- Mission collection requirements

### Q3
- Bathymetric uncertainty
- Statistical performance measures
- Benchmark data against previous surveys
- Predictive modelling for sonar performance
- Environmental conditions
- Rules of engagement
- Route planning confidence
- Other asset requirements
- Host vehicle options
- Vehicle safety
- Navigational hazards

### Q4
- Terrain shadowing
- Data loss due to environmental interference
- Navigation drift
- Overlap errors/differences
- collision avoidance
- ocean conditions
- Threats to a mission
- Environmental conditions
- Launch and recovery locations
- UUV limits
- Sensor collection requirements
- GPS requirements
- Hazards along route
- Navigation accuracy
- UUV communication requirements
- No-go zones

### Q5
- Gaps in coverage
- Calibration drift
- Sensor specification limits
- Environmental factors limiting sensors
- Collision avoidance
- Battery capacity
- Acoustic performance
- Grounding risk
- Proportion of route covered
- Water column buoyancy

### Q6
- Metadata logs
- Feature detection with confidence scores
- Survey track overlaps
- Post-mission analysis reports
- Pre-mission checks
- Security access
- Job template files
- Time on task v completion
- Data gaps
- Depth discrepancies
- Automated provenance

## Scenario 3

### Q1
- Transit time to beach
- Seafloor complexity
- Environmental conditions
  - wave height
  - current
- Historical data for similar beaches
- Surface/sub sea traffic
- UUV limits
- Sensor collection requirements
- GPS requirements
- Hazards along route
- Navigation accuracy
- UUV communication requirements
- No-go zones
- Mission collection requirements

### Q2
- Beach gradient and substrate composition
- Habitat restrictions
- Ocean conditions
  - Swell
  - tide
  - sediment transport
- Bottom object density and distribution
- Navigation risk
  - submerged hazards
  - surf zone dynamics
- Sensor limitations
- UUV limits
- Sensor collection requirements
- GPS requirements
- Hazards along route
- Navigation accuracy
- UUV communication requirements
- No-go zones
- Mission collection requirements

### Q3
- Use of existing base map models
- Viability scoring with respect to
  - stealth
  - speed
  - safety
- Incorporate predictive modelling
- Reference historical beach assessments and landing success
- Environmental conditions
- Rules of engagement
- Route planning confidence
- Other asset requirements
- Host vehicle options
- Vehicle safety
- Navigational hazards
- Surfacing risk

### Q4
- Habitat disruption
- Environmental compliance
- Sensor interference due to seafloor composition
- Surf dynamics
- Detection in high-traffic areas
- Collision avoidance
- Sensor performance
- Ocean conditions
- Threats to a mission
- Environmental conditions
- Launch and recovery locations
- UUV limits
- Sensor collection requirements
- GPS requirements
- Hazards along route
- Navigation accuracy
- Seafloor classification
- Communication loss

### Q5
- Unsuitable beach topography
- Unsuitable beach composition
- Marine life
- Protected zones
- Beach anomalies
  - rips
  - sediment plumes
- Seafloor clutter or shadowing
- Battery capacity
- Acoustic performance
- Grounding risk
- Proportion of route covered
- Water column buoyancy
- Collision avoidance
- Time on surface

### Q6
- Documentation of beach selection rationale
- Mission logs
- Feature detection records
- Post-mission analysis reports
- Pre-mission checks
- Security access
- Job template files
- Time on task v completion
- Data gaps
- Depth discrepancies
- Automated provenance

## Scenario 4

### T1
- Baseline risk profiling, but lack dynamic environmental integration
- Must incorporate realtime ocean environmental data
- Must include UUV performance specifications
- Need adaptive learning techniques
- Standard operating procedures
- Environmental limits must be documented
- UUV capability (limits)
- Sensor capabilities (limits)
- Personnel suitably trained/experienced

### T2
- Performance degradation in 
  - high current zones
  - Shallow water
  - Complex acoustic environments
- UUV performance must be documented in all conditions
- Comparative analysis is essential for refining operational limits
- Ocean currents
- Environmental conditions
- Ocean traffic when surfacing

### T3
- Sounding quality affected by UUV attitude
- Need to benchmark sensors against sensor specifications
- Need to flag data gaps for post-mission review

### T4
- Mission plan events to be time stamped and fixed in location
- Mission phases to be tagged with metadata
- Integrate defense data standards
- Data to be fit for purpose

### T5
- Incident logging to include root cause analysis
- data flow dropouts to be flagged and logged geographically
- Risk flags trigger automated alerts and debrief requirements

### T6
- Current standards are not flexible enough
- Need to support structured and unstructured data
- Systems must scale

