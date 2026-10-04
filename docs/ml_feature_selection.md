# ML Feature Selection — US Traffic Accident Severity

## 1. Objective and Problem Formulation
The goal of this machine learning pipeline is to predict the severity of traffic accidents in the United States on a multiclass scale of 1 to 4:
- **Severity 1**: Minor disruption / short delay
- **Severity 2**: Moderate traffic delay (dominant operational class)
- **Severity 3**: Serious traffic disruption / lane closures
- **Severity 4**: Critical impact / long delays and major closures

To ensure real-world predictive validity and prevent data leakage, features must be restricted strictly to information available at or prior to the occurrence and initial reporting of an accident.

---

## 2. Selected Feature Set

The selected feature set encompasses 28 attributes partitioned into four primary categories:

### A. Geographic and Spatial Attributes (2 features)
- `Start_Lat` (Double): Latitude of accident start location.
- `Start_Lng` (Double): Longitude of accident start location.
*Rationale*: Geographic coordinates capture spatial clustering, regional road design standards, terrain, and urbanization density.

### B. Impact Extent (1 feature)
- `Distance(mi)` (Double): Length of the road extent affected by the accident.
*Rationale*: Strong indicator of queue length and physical roadway corridor disruption.

### C. Meteorological Features (9 features)
- `Temperature(F)` (Double): Ambient temperature at nearby weather station.
- `Humidity(%)` (Double): Relative humidity.
- `Pressure(in)` (Double): Atmospheric pressure.
- `Visibility(mi)` (Double): Visibility distance.
- `Wind_Speed(mph)` (Double): Wind speed.
- `Precipitation(in)` (Double): Precipitation amount recorded during the observation period.
- `Weather_Condition` (String): Categorical condition (e.g., Fair, Clear, Rain, Snow, Fog).
- `Timezone` (String): US timezone (US/Eastern, US/Central, US/Mountain, US/Pacific).
- `State` (String): US State code (49 states/districts).
*Rationale*: Adverse weather, surface friction degradation, and reduced visibility affect driver braking and impact speeds.

### D. Day/Night and Twilight Indicators (4 features)
- `Sunrise_Sunset` (String): Day vs. Night indicator.
- `Civil_Twilight` (String): Day vs. Night based on sun 6° below horizon.
- `Nautical_Twilight` (String): Day vs. Night based on sun 12° below horizon.
- `Astronomical_Twilight` (String): Day vs. Night based on sun 18° below horizon.
*Rationale*: Natural illumination levels directly influence driver reaction time and collision severity.

### E. Road and Infrastructure Indicators (12 Boolean features)
- `Amenity` (Boolean -> 0/1)
- `Bump` (Boolean -> 0/1)
- `Crossing` (Boolean -> 0/1)
- `Give_Way` (Boolean -> 0/1)
- `Junction` (Boolean -> 0/1)
- `No_Exit` (Boolean -> 0/1)
- `Railway` (Boolean -> 0/1)
- `Roundabout` (Boolean -> 0/1)
- `Station` (Boolean -> 0/1)
- `Stop` (Boolean -> 0/1)
- `Traffic_Calming` (Boolean -> 0/1)
- `Traffic_Signal` (Boolean -> 0/1)
*Rationale*: Roadway fixtures identify intersection controls, speed management infrastructure, and highway interchange merge zones.

### F. Derived Temporal Features (4 features)
- `Year` (Double): 2016 - 2023.
- `Month` (Double): 1 - 12 (seasonal variation).
- `DayOfWeek` (Double): 1 - 7 (weekday vs. weekend patterns).
- `Hour` (Double): 0 - 23 (commute rush hours vs. overnight).
*Rationale*: Extracted from `Start_Time`. Accounts for traffic density cycles, commuting volume, and seasonal patterns.

---

## 3. Excluded Columns and Leakage Prevention

The following columns are strictly excluded from the feature space:

| Excluded Column | Reason for Exclusion |
|:---|:---|
| **Severity** | Target label variable (must not be in feature vector). |
| **ID** | Arbitrary primary key string with no generalizable predictive value. |
| **Description** | Free-text narrative entered post-incident; often contains explicit descriptions of duration and emergency response (target leakage). |
| **Street** | Extremely high cardinality (>170,000 distinct values); prone to severe overfitting. |
| **City** | High cardinality (>13,000 distinct values); covered by `Start_Lat`, `Start_Lng`, and `State`. |
| **County** | High cardinality (>1,800 distinct values); redundant with coordinates. |
| **Zipcode** | High cardinality (>19,000 distinct values); sparse and redundant. |
| **Airport_Code** | High cardinality weather station identifier; weather values themselves are already included. |
| **End_Time** | **Critical Target Leakage**: Encodes accident duration ($End\_Time - Start\_Time$), which is not available at accident occurrence. |
| **End_Lat, End_Lng** | High missingness (44.18%); redundant with `Start_Lat`, `Start_Lng`, and `Distance(mi)`. |
| **Weather_Timestamp** | Exact observation timestamp of airport station; redundant with `Start_Time`. |
| **Turning_Loop** | **Zero Variance**: Value is `False` in 100.0% of all 7,728,394 rows. |
| **Source** | Data provider reporting artifact with no physical driving meaning. |
| **Country** | Constant value (`US`) across 100.0% of rows. |
| **Start_Time** | Raw timestamp dropped after extracting `Year`, `Month`, `DayOfWeek`, and `Hour`. |

---

## 4. Leakage Verification Checklist
Prior to pipeline fitting, the training script programmatically asserts that zero excluded fields exist in the assembler input column list.
