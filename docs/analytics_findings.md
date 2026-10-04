# US Traffic Accident Severity Analytics

## 1. Dataset Overview

The dataset analyzed represents comprehensive nationwide US traffic accident records sourced from multiple data APIs between January 2016 and April 2023. The data pipeline loaded and processed the full dataset using Apache Spark 4.2.0 over an explicit 46-column typed schema and Parquet storage:

- **Total Recorded Accidents**: 7,728,394 rows
- **Total Variables**: 46 columns (temporal, geographic, spatial, meteorological, and road infrastructure features)
- **Primary Key Integrity**: 7,728,394 distinct `ID` values (zero duplicate IDs across the entire dataset)
- **Temporal Coverage**: January 15, 2016 01:48:33 UTC to April 01, 2023 05:00:00 UTC (~7.25 years)
- **Geographic Breadth**: 49 US states (including District of Columbia; excludes Alaska and Hawaii), across 13,678 unique cities and 1,871 counties
- **Storage Profile**: Raw CSV footprint of 3,058,183,727 bytes (~2.85 GB) compressed to a columnar Parquet footprint of 761,426,478 bytes (~726.15 MB), delivering a ~4:1 compression ratio with zero row or data loss

*Traceability: `outputs/tables/dataset_overview.csv`*

---

## 2. Severity Distribution

Accident severity in this dataset is graded on an integer scale from 1 (minimal impact on traffic / short delay) to 4 (critical impact / long delays and major closures). The measured nationwide distribution is heavily skewed toward moderate severity:

| Severity Level | Accident Count | Percentage (%) | Description |
|:---:|:---:|:---:|:---|
| **1** | 67,366 | 0.87% | Minor disruption |
| **2** | 6,156,981 | 79.67% | Moderate traffic delay (dominant class) |
| **3** | 1,299,337 | 16.81% | Serious traffic disruption |
| **4** | 204,710 | 2.65% | Severe closure / critical impact |
| **Total** | **7,728,394** | **100.00%** | Sum strictly verified |

- **Extreme Class Imbalance**: Severity 2 comprises approximately 79.67% of all recorded events. Combined, lower-to-medium impact incidents (Severity 1 and 2) represent 80.54% of records.
- **High Severity Events**: Critical severity events (Severity 4) account for 2.65% (204,710 incidents), while Severity 3 events account for 16.81% (1,299,337 incidents).
- **Implication for Modeling**: This class imbalance necessitates stratified evaluation (macro F1-score, balanced precision/recall) rather than simple accuracy in downstream classification.

*Traceability: `outputs/tables/severity_distribution.csv`, `outputs/charts/severity_distribution.png`*

---

## 3. Temporal Patterns

Analysis of `Start_Time` across temporal granularities reveals distinct operational cycles:

### Yearly Volume
- **2016**: 410,821 (5.32%)
- **2017**: 717,290 (9.28%)
- **2018**: 893,426 (11.56%)
- **2019**: 954,302 (12.35%)
- **2020**: 1,161,598 (15.03%)
- **2021**: 1,412,433 (18.28%) — *Peak complete recording year*
- **2022**: 1,268,806 (16.42%)
- **2023**: 166,552 (2.16%) — *Truncated: records only span up to April 1, 2023*

The upward trend between 2016 and 2021 reflects expansion in automated traffic sensor coverage and reporting provider integration over time, in addition to changes in traffic volume.

### Monthly Volume
Accident counts peak in autumn and winter:
- **December**: 854,845 (11.06%) — Highest monthly volume
- **November**: 766,801 (9.92%)
- **January**: 737,511 (9.54%)
- **July**: 452,713 (5.86%) — Lowest monthly volume

### Day-of-Week Distribution
- **Weekdays (Monday–Friday)**: Account for 5,915,960 accidents (**76.55%** of all records), averaging ~1.18 million accidents per weekday. Friday exhibits the highest individual volume (1,237,229; 16.01%).
- **Weekends (Saturday–Sunday)**: Account for 1,812,434 accidents (**23.45%** of all records), with Saturday recording 579,153 (7.49%) and Sunday recording 490,115 (6.34%).

### Hourly Rush-Hour Pattern
Hourly volume exhibits a prominent bimodal commute distribution (local time):
- **Morning Peak**: 7:00 AM (546,789; 7.08%) and 8:00 AM (541,643; 7.01%).
- **Evening Peak**: 4:00 PM (520,177; 6.73%) and 5:00 PM (516,626; 6.68%).
- **Midday Plateau**: 10:00 AM to 12:00 PM stabilizes at ~315,000–322,000 accidents per hour (~4.1%).
- **Overnight Minimum**: 3:00 AM records the lowest volume across all 24 hours (74,229; 0.96%).

*Traceability: `outputs/tables/accidents_by_year.csv`, `outputs/tables/accidents_by_month.csv`, `outputs/tables/accidents_by_day_of_week.csv`, `outputs/tables/accidents_by_hour.csv`, `outputs/charts/accidents_by_year.png`, `outputs/charts/accidents_by_month.png`, `outputs/charts/accidents_by_day_of_week.png`, `outputs/charts/accidents_by_hour.png`, `outputs/charts/accidents_by_year_severity.png`*

---

## 4. Geographic Patterns

### State-Level Concentration
The top 10 states contribute 5,244,043 incidents, representing **67.85%** of all nationwide records:

1. **California (CA)**: 1,741,433 accidents (22.53%)
2. **Florida (FL)**: 880,192 accidents (11.39%)
3. **Texas (TX)**: 582,837 accidents (7.54%)
4. **South Carolina (SC)**: 382,557 accidents (4.95%)
5. **New York (NY)**: 347,960 accidents (4.50%)
6. **North Carolina (NC)**: 338,199 accidents (4.38%)
7. **Virginia (VA)**: 303,301 accidents (3.92%)
8. **Pennsylvania (PA)**: 296,620 accidents (3.84%)
9. **Minnesota (MN)**: 192,084 accidents (2.49%)
10. **Oregon (OR)**: 179,660 accidents (2.32%)

### City-Level Concentration
The top 5 municipal areas with the highest recorded accident counts are:
1. **Miami, FL**: 186,768 (2.42%)
2. **Houston, TX**: 169,428 (2.19%)
3. **Los Angeles, CA**: 156,491 (2.02%)
4. **Charlotte, NC**: 138,345 (1.79%)
5. **Dallas, TX**: 130,303 (1.69%)

### Analytical Caution: Volume vs. Risk
> **Important Distinction**: These recorded figures indicate raw reporting volume within the data provider network. They must **not** be interpreted as localized accident risk or driver danger. True accident risk requires normalization by exposure denominators (e.g., annual vehicle miles traveled [VMT], registered vehicles, lane miles, and commuter population), which are outside the scope of this raw incident feed.

*Traceability: `outputs/tables/top_10_states.csv`, `outputs/tables/top_20_cities.csv`, `outputs/tables/top_20_counties.csv`, `outputs/charts/top_10_states.png`, `outputs/charts/top_20_cities.png`, `outputs/tables/severity_by_state.csv`, `outputs/charts/severity_by_state.png`*

---

## 5. Weather Patterns

### Predominant Weather Conditions
Most recorded accidents occurred during benign, clear, or common atmospheric conditions:
- **Fair**: 2,560,802 (33.13%)
- **Mostly Cloudy**: 1,016,195 (13.15%)
- **Cloudy**: 817,082 (10.57%)
- **Clear**: 808,743 (10.46%)
- **Partly Cloudy**: 698,972 (9.04%)
- **Overcast**: 382,866 (4.95%)
- **Light Rain**: 352,957 (4.57%)
- **Light Snow**: 128,680 (1.67%)
- **Fog**: 99,238 (1.28%)
- **Rain**: 84,331 (1.09%)

Non-adverse weather ("Fair", "Clear", and cloudy categories) accounts for over **81%** of all recorded incidents. Observational finding: accidents occur overwhelmingly during standard, dry driving conditions when traffic volume is highest. Adverse weather (rain, snow, fog) is present in a measurable minority (~9-10%) of records.

### Meteorological Variable Summary
- **Temperature (F)**: Mean = 61.66°F, Median = 64.00°F (Min: -89.0°F, Max: 207.0°F; 2.12% missing)
- **Humidity (%)**: Mean = 64.83%, Median = 67.00% (Min: 1.0%, Max: 100.0%; 2.25% missing)
- **Pressure (in)**: Mean = 29.54 in, Median = 29.86 in (Min: 0.00 in, Max: 58.63 in; 1.82% missing)
- **Visibility (mi)**: Mean = 9.09 mi, Median = 10.00 mi (Min: 0.00 mi, Max: 140.0 mi; 2.29% missing)
- **Wind Speed (mph)**: Mean = 7.69 mph, Median = 7.00 mph (Min: 0.00 mph, Max: 1,087.0 mph; 7.39% missing)
- **Precipitation (in)**: Mean = 0.01 in, Median = 0.00 in (Min: 0.00 in, Max: 36.47 in; 28.51% missing)

*Note: Extreme values (e.g., Wind Speed of 1,087 mph or Temperature of 207°F) represent sensor/ingestion anomalies documented during Stage 4 and should be handled during feature preprocessing prior to model training.*

*Traceability: `outputs/tables/top_weather_conditions.csv`, `outputs/tables/weather_severity_summary.csv`, `outputs/tables/weather_numeric_summary.csv`, `outputs/charts/top_weather_conditions.png`, `outputs/charts/weather_severity_summary.png`*

---

## 6. Road / Infrastructure Analysis

Thirteen boolean indicators identify infrastructure attributes near the incident location:

| Infrastructure Feature | True Count | False Count | True Percentage (%) |
|:---|:---:|:---:|:---:|
| **Traffic_Signal** | 1,143,772 | 6,584,622 | 14.800% |
| **Crossing** | 873,763 | 6,854,631 | 11.306% |
| **Junction** | 571,342 | 7,157,052 | 7.393% |
| **Stop** | 214,371 | 7,514,023 | 2.774% |
| **Station** | 201,901 | 7,526,493 | 2.612% |
| **Amenity** | 96,334 | 7,632,060 | 1.246% |
| **Railway** | 66,979 | 7,661,415 | 0.867% |
| **Give_Way** | 36,582 | 7,691,812 | 0.473% |
| **No_Exit** | 19,545 | 7,708,849 | 0.253% |
| **Traffic_Calming** | 7,598 | 7,720,796 | 0.098% |
| **Bump** | 3,514 | 7,724,880 | 0.045% |
| **Roundabout** | 249 | 7,728,145 | 0.003% |
| **Turning_Loop** | 0 | 7,728,394 | 0.000% |

### Key Observations
1. **Traffic Signals and Crossings**: Most prevalent features, associated with 14.80% and 11.31% of incidents respectively.
2. **Junction Severity Profile**: Incidents occurring near Highway Junctions exhibit a markedly higher proportion of severe traffic impacts: **23.19% Severity 3** and **3.60% Severity 4** (total high-severity share = 26.79%), compared to Crossings (5.50% Severity 3) and Traffic Signals (7.75% Severity 3). Highway interchange complexity correlates with extended queueing and road blockage.
3. **Degenerate Feature**: `Turning_Loop` is 100% False (zero True occurrences across all 7.7M rows) and provides zero variance for predictive modeling.

*Traceability: `outputs/tables/infrastructure_features.csv`, `outputs/tables/infrastructure_severity_summary.csv`, `outputs/charts/infrastructure_features.png`*

---

## 7. Accident Distance Extent Analysis

`Distance(mi)` measures the length of the road extent impacted by the incident:

- **Mean Extent**: 0.5618 miles (~2,966 feet)
- **Median Extent**: 0.0300 miles (~158 feet)
- **Minimum**: 0.0000 miles
- **Maximum**: 441.75 miles

### Distance Bucket Distribution
- **0.0 miles (Point Incident)**: 3,302,161 accidents (**42.73%**)
- **0.0 – 0.1 miles**: 1,185,298 accidents (**15.34%**)
- **0.1 – 0.5 miles**: 1,389,805 accidents (**17.98%**)
- **0.5 – 1.0 miles**: 740,906 accidents (**9.59%**)
- **1.0 – 5.0 miles**: 965,010 accidents (**12.49%**)
- **5.0 – 10.0 miles**: 109,696 accidents (**1.42%**)
- **10.0 – 25.0 miles**: 30,833 accidents (**0.40%**)
- **25.0+ miles**: 4,685 accidents (**0.06%**)

### Distance Extent by Severity
- **Severity 1**: Mean = 0.1145 mi, Median = 0.0000 mi
- **Severity 2**: Mean = 0.5649 mi, Median = 0.0680 mi
- **Severity 3**: Mean = 0.4236 mi, Median = 0.0000 mi
- **Severity 4**: Mean = 1.4958 mi, Median = 0.4730 mi

Severity 4 incidents produce a mean road blockage extent nearly **3 times larger** than Severity 2 and over **13 times larger** than Severity 1, consistent with severe multi-vehicle collisions requiring extended emergency response corridors.

*Traceability: `outputs/tables/distance_summary.csv`, `outputs/tables/distance_buckets.csv`, `outputs/tables/distance_by_severity.csv`, `outputs/charts/distance_buckets.png`*

---

## 8. Data Quality Summary

Consolidating measurements from Stage 3C, Stage 4, and Stage 6:

1. **End Location Missingness**: `End_Lat` and `End_Lng` are missing in 3,414,349 records (**44.18%**), matching instances where point-source sensors logged single-coordinate coordinates.
2. **Precipitation Missingness**: `Precipitation(in)` is absent in 2,203,586 records (**28.51%**), primarily corresponding to automated weather stations that report precipitation only during active rainfall events.
3. **Wind Chill Missingness**: `Wind_Chill(F)` is absent in 1,996,009 records (**25.83%**), as wind chill is defined and recorded only during cold weather conditions.
4. **Physical Outliers / Sensor Anomalies**: Maximum recorded Wind Speed of 1,087 mph, Maximum Temperature of 207°F, Maximum Visibility of 140 miles, and Maximum Pressure of 58.63 in reflect sensor calibration errors or parsing artifacts that should be robustly clamped or imputed.
5. **Zero-Variance Feature**: `Turning_Loop` has zero positive instances across all 7,728,394 rows and should be dropped during feature selection.
6. **Class Imbalance**: Severity 2 comprises 79.67% while Severity 1 comprises only 0.87% and Severity 4 comprises 2.65%.

---

## 9. Key Evidence-Based Findings

1. **Predominance of Moderate Severity**: Approximately 79.67% of recorded US traffic accidents are classified as Severity 2, while critical Severity 4 accidents represent only 2.65% (`outputs/tables/severity_distribution.csv`).
2. **Geographic Clustering**: Three states (California, Florida, Texas) account for 41.46% (3,204,462 records) of all nationwide traffic accident reports (`outputs/tables/top_10_states.csv`).
3. **Commuter Rush-Hour Bimodality**: Hourly accident volume peaks sharply during weekday morning commute (7:00–8:00 AM, ~14.09% combined) and evening commute (4:00–5:00 PM, ~13.41% combined) (`outputs/tables/accidents_by_hour.csv`).
4. **Weekday Workweek Concentration**: Over 76.5% of accidents occur Monday through Friday, with Friday exhibiting the highest single-day count (1,237,229 accidents; 16.01%) (`outputs/tables/accidents_by_day_of_week.csv`).
5. **Autumn / Early Winter Peak**: Accident recordings culminate in November and December, with December registering the maximum annual volume (854,845; 11.06%) (`outputs/tables/accidents_by_month.csv`).
6. **Non-Adverse Weather Dominance**: Over 81% of accidents occur during fair, clear, or cloudy weather conditions rather than precipitation or storm events (`outputs/tables/top_weather_conditions.csv`).
7. **Junctions Amplify Severe Delays**: Highway junctions exhibit a high-severity rate (Severity 3 + 4) of 26.79%, compared to traffic signals (9.51%) and crossings (7.04%) (`outputs/tables/infrastructure_severity_summary.csv`).
8. **Point vs. Corridor Impacts**: 42.73% of accidents are localized point events (`Distance(mi) == 0.0`), while Severity 4 accidents average an impacted corridor length of 1.4958 miles (`outputs/tables/distance_by_severity.csv`).
9. **Controlled Intersections Most Frequent Infrastructure Feature**: Traffic signals (14.80%) and pedestrian crossings (11.31%) are the most common infrastructure attributes identified at accident locations (`outputs/tables/infrastructure_features.csv`).
10. **Zero-Variance Feature Identified**: `Turning_Loop` is 100% False across all 7,728,394 rows and should be removed from machine learning feature pipelines (`outputs/tables/infrastructure_features.csv`).
11. **Primary Key Completeness**: The dataset demonstrates 100% unique primary key coverage with zero duplicate IDs across 7,728,394 records (`outputs/tables/dataset_overview.csv`).
12. **High-Ratio Lossless Compression**: Converting raw CSV (3.058 GB) to snappy-compressed Parquet (726.15 MB) achieved a ~4:1 compression ratio while preserving exact schemas and accelerating analytic scan speeds.
