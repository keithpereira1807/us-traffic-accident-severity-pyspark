# US Traffic Accident Severity Analytics Using PySpark
## Big Data Analytics — Comprehensive Academic Project Report

[![Python Version](https://img.shields.io/badge/Python-3.13.2-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-orange.svg)](https://spark.apache.org/)
[![Java Version](https://img.shields.io/badge/Java-17%20(Adoptium)-red.svg)](https://adoptium.net/)
[![License](https://img.shields.io/badge/License-Academic%20Use%20Only-green.svg)]()

---

## 1. Project Overview

Traffic accidents across the United States represent a major societal challenge, resulting in tragic loss of life, severe injuries, economic disruption, and extensive roadway delays. Modern transportation monitoring systems capture massive streams of vehicular, spatial, meteorological, and infrastructure telemetry. However, extracting actionable insights from millions of multi-dimensional records requires high-performance, distributed computing frameworks.

This project implements an end-to-end **Big Data Analytics (BDA)** pipeline utilizing **Apache Spark (PySpark 4.2.0)** to ingest, clean, explore, and model over **7.7 million traffic accident records** spanning February 2016 through April 2023. Operating within the constraints of a standard personal computer (~7.37 GB RAM), this project demonstrates how distributed memory-conscious data engineering, columnar storage, Spark SQL, and PySpark MLlib can process massive datasets without out-of-memory failures.

> **Methodological Note on Causality**:  
> All analyses and relationships documented in this study represent **observational associations** derived from historical event logs. In accordance with rigorous data science principles, observed correlations (e.g., higher accident counts during peak commute hours or under specific weather conditions) reflect reporting density and environmental co-occurrence rather than causal mechanisms.

---

## 2. Problem Statement

Transportation agencies and emergency response centers are challenged by the volume, velocity, and variety of accident telemetry. Raw accident logs contain complex combinations of spatial coordinates, precise timestamps, weather observations, and localized roadway fixtures. Key analytical challenges include:
1. **Scalability**: Processing gigabyte-scale datasets containing millions of rows on commodity hardware without memory exhaustion.
2. **Severe Class Imbalance**: Less than 1% of accidents fall into the most minor category (Severity 1) and only ~2.6% into critical road closures (Severity 4), while ~80% occupy moderate disruption (Severity 2). Naive machine learning models degenerate to majority-class predictors.
3. **Missingness & Sensor Telemetry**: Severe missingness across weather indicators (up to 44% in secondary coordinates and 28% in precipitation) requires disciplined, non-destructive imputation rather than row deletion.
4. **Predictive Validity**: Formulating defensible feature sets strictly observable at the onset of an incident to prevent target leakage.

This project investigates nationwide spatiotemporal patterns and evaluates whether initial environmental and infrastructure conditions can predict accident severity using distributed machine learning.

---

## 3. Objectives

1. **Scalable Ingestion**: Ingest over 7.7 million raw CSV records using PySpark with an explicit schema and convert to Snappy-compressed columnar Parquet format.
2. **Data Profiling & Quality Assessment**: Quantify missingness patterns, investigate physical sensor anomalies, and establish deterministic data hygiene.
3. **Temporal Analytics**: Analyze diurnal, day-of-week, monthly, and multi-year accident trends.
4. **Geographic Analytics**: Identify spatial distribution, state-level concentrations, and urban hot-spots across the continental United States.
5. **Environmental & Roadway Analytics**: Evaluate relationships between accident frequency, meteorological conditions, and physical road infrastructure.
6. **Corridor Impact Analysis**: Examine the physical linear road extent (`Distance(mi)`) impacted across severity levels.
7. **Declarative Querying**: Demonstrate advanced big data aggregation capabilities using Spark SQL queries and views.
8. **Class Imbalance Mitigation**: Formulate and implement training-partition balanced class loss weighting natively within PySpark MLlib.
9. **Distributed Machine Learning**: Train, evaluate, and compare three multiclass classifiers (Multinomial Logistic Regression, Decision Tree, Random Forest) on held-out test data.
10. **Academic Presentation**: Deliver a reproducible, interactive Jupyter Notebook, structured documentation, and viva defense resources.

---

## 4. Dataset

The project utilizes the **US Accidents Dataset (March 2023 release)**, one of the most comprehensive open traffic datasets available:

| Dimension / Metric | Verified Value | Description |
|---|---|---|
| **Total Records** | **7,728,394** | Exactly 7.728 million individual accident occurrences |
| **Total Attributes** | **46** | Spatiotemporal, meteorological, and infrastructure features |
| **Distinct Primary Keys** | **7,728,394** | 100% unique primary keys (`ID` column) |
| **Temporal Extent** | **2016-01-15 01:48:33 to 2023-04-01 05:00:00** | ~7.2 years of continuous coverage |
| **Geographic Extent** | **49 US States** | Contiguous US plus District of Columbia (13,678 cities, 1,871 counties) |
| **Raw Storage Size** | **3,058,183,727 bytes (~2.85 GB)** | Uncompressed raw CSV file |
| **Optimized Storage Size** | **~726 MB** | Snappy-compressed columnar Parquet partition |

### Target Variable: `Severity`
`Severity` measures the degree of impact on traffic delay and road capacity on an ordinal scale of 1 to 4:
- **Severity 1**: Shortest delay / minor traffic impact — **67,366 records (0.87%)**
- **Severity 2**: Moderate traffic delay / short disruption — **6,156,981 records (79.67%)** *(Dominant Class)*
- **Severity 3**: Significant traffic disruption / lane closures — **1,299,337 records (16.81%)**
- **Severity 4**: Critical event / long delay / major road closure — **204,710 records (2.65%)**

> **Repository Storage Notice**:  
> The raw 2.85 GB CSV file (`US_Accidents_March23.csv`) and generated Parquet partitions are **excluded from GitHub version control** via `.gitignore`. Instructions for downloading the dataset and placing it locally are provided in the *How to Run* section.

---

## 5. Technologies Used

- **Programming Language**: Python 3.13.2
- **Distributed Computing Engine**: Apache Spark (PySpark 4.2.0)
- **Java Runtime**: OpenJDK 17 (Adoptium 17.0.20.101-hotspot)
- **Hadoop Ecosystem**: Apache Hadoop 3.5.0 client libraries with native Windows binaries (`winutils.exe`, `hadoop.dll` under `C:\hadoop\bin`)
- **Query APIs**: Spark DataFrames and Spark SQL
- **Machine Learning Library**: PySpark MLlib (`Pipeline`, `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `Imputer`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`)
- **Data Visualization & Analytics**: Matplotlib, NumPy, and Pandas (used strictly for rendering small aggregated summary tables)
- **Interactive Delivery**: Jupyter Notebook (`notebooks/US_Traffic_Accident_Severity_Analytics_Final.ipynb`)
- **Version Control**: Git / GitHub

### Hardware & Resource Considerations
The entire pipeline was engineered to execute locally on a host computer with **approximately 7.37 GB RAM**:
- Single-node execution: `master("local[1]")`
- Strict driver memory ceiling: `spark.driver.memory="2g"`
- Reduced shuffle partitions: `spark.sql.shuffle.partitions="2"`
- Sequential training execution: Models were trained one at a time to prevent garbage collection thrashing and memory exhaustion.
- No full dataset conversions: `collect()` or `toPandas()` were strictly prohibited on raw data partitions.

---

## 6. System / Project Architecture

The following diagram illustrates the end-to-end data flow and execution lifecycle:

```text
       Raw Dataset (US_Accidents_March23.csv, 2.85 GB)
                             │
                             ▼
               [Stage 2] Dataset Verification
                             │
                             ▼
         [Stage 3A/B] Explicit Schema PySpark Ingestion
                             │
                             ▼
         [Stage 5/6] Parquet Columnar Conversion (~726 MB)
                             │
                             ▼
         [Stage 3C/4] Data Profiling & Quality Analysis
                             │
                             ▼
              [Stage 6] Distributed BDA Analytics
        ┌────────────────────┬────────────────────┐
        ▼                    ▼                    ▼
Temporal Analysis    Geographic Analysis    Weather & Infrastructure
 (Hour, Day, Year)     (State, City)        (Rain, Signal, Junction)
        └────────────────────┬────────────────────┘
                             │
                             ▼
               [Stage 6] Spark SQL Demonstrations
             (5 Core Analytical Aggregations)
                             │
                             ▼
         [Stage 7] ML Feature Engineering & Selection
           (32 Features Selected, 16 Excluded)
                             │
                             ▼
       [Stage 7] Train / Test Split (80% / 20%, Seed 42)
                             │
                             ▼
      [Stage 7] Balanced Class Weighting (Training Set Only)
        (w1 = 28.62, w2 = 0.31, w3 = 1.49, w4 = 9.42)
                             │
                             ▼
          [Stage 7] Sequential MLlib Model Training
        ┌────────────────────┬────────────────────┐
        ▼                    ▼                    ▼
Logistic Regression    Decision Tree       Random Forest
 (Multinomial Softmax) (maxDepth=8)        (25 Trees, Depth=8)
        └────────────────────┬────────────────────┘
                             │
                             ▼
      [Stage 7] Test Set Evaluation (1,545,478 Records)
   (Accuracy, Weighted F1, Macro F1, Per-Class Confusion Matrices)
                             │
                             ▼
     [Stage 8/9] Final Jupyter Notebook & Project Packaging
```

---

## 7. Data Processing Pipeline

1. **Ingestion with Explicit Schema**: Rather than forcing a costly double-scan with `inferSchema=True`, a 46-attribute `StructType` schema was enforced directly upon reading the raw CSV, guaranteeing instant type assignment and eliminating parsing ambiguity.
2. **Columnar Parquet Migration**: The dataset was transformed into partitioned Parquet format with Snappy compression (`data/processed/us_accidents_parquet/`). This reduced disk consumption from 2.85 GB to ~726 MB (a 74% compression saving) while unlocking column pruning and predicate pushdown.
3. **Data Quality & Anomaly Profiling**:
   - **Coordinate Gaps**: `End_Lat` and `End_Lng` exhibit **44.03% missingness** (3,402,773 rows), identifying them as secondary feed attributes unsuitable for baseline prediction.
   - **Precipitation Missingness**: 28.51% of records lack precipitation data because many stations report only when measurable rain/snow occurs.
   - **Extreme Values**: Profiling discovered physical measurement anomalies, such as maximum wind speeds of 1,087 mph and sea-level atmospheric pressures up to 58.63 in.
   - **Zero Variance**: `Turning_Loop` was found to be constant `False` across all 7,728,394 rows and was eliminated.
4. **Leakage-Safe Preprocessing**:
   - Numeric missing values are imputed via **median imputation** (`Imputer(strategy="median")`), learned exclusively on the training partition.
   - Categorical nulls are coalesced to `"Unknown"` and encoded using `StringIndexer(handleInvalid="keep")` and `OneHotEncoder(handleInvalid="keep")`.
   - Raw `Start_Time` was converted into cyclical integer columns (`Year`, `Month`, `DayOfWeek`, `Hour`) and then dropped.

---

## 8. Exploratory and Big Data Analytics

Stage 6 generated **23 structured analytical CSV tables** and **13 publication-quality visualizations** saved under `outputs/tables/` and `outputs/charts/`.

### Major Analytical Findings:
- **Temporal Patterns**:
  - **Weekday Dominance**: **76.55%** of all recorded accidents occur on weekdays (Monday through Friday), peaking on **Friday with 1,237,229 records (16.01%)**.
  - **Commuter Spikes**: Accident volume exhibits a distinct bimodal diurnal distribution peaking during morning rush hour (**7:00–8:00 AM: 14.09%**) and evening rush hour (**4:00–5:00 PM: 13.41%**). The overnight trough occurs at **3:00 AM (0.96%)**.
- **Geographic Patterns**:
  - The top three states by accident volume are **California (1,741,433 records, 22.53%)**, **Florida (880,192 records, 11.39%)**, and **Texas (582,837 records, 7.54%)**. Together, these three states account for **41.46%** of all nationwide records.
  - *Contextual Caveat*: These figures represent raw reporting volumes, not population- or traffic-normalized rates.
- **Weather Conditions**:
  - Over **81%** of accidents occur during fair, clear, or cloudy weather conditions (*Fair*: 32.84%, *Mostly Cloudy*: 13.11%, *Clear*: 11.34%). Benign weather accounts for most collisions simply because the majority of vehicular travel occurs in good weather.
- **Infrastructure Attributes**:
  - Accidents frequently occur near traffic management infrastructure: `Traffic_Signal` (14.80%), `Crossing` (11.31%), and `Junction` (7.28%).
  - Records marked with `Junction=True` exhibit an observably higher proportion of Severity 3 and Severity 4 events, consistent with the higher speed differentials typical of highway interchange merge zones.
- **Corridor Impact (`Distance(mi)`)**:
  - **42.73%** of records have `Distance(mi) = 0.0` (point-source accidents).
  - Severity 4 accidents disrupt a substantially longer road corridor (mean of **1.50 miles**, median **0.47 miles**) compared to Severity 2 (mean **0.30 miles**).

---

## 9. Machine Learning Methodology

### Problem Formulation
Accident severity prediction was formulated as an ordinal 4-class classification problem:
$$\mathcal{Y} \in \{1, 2, 3, 4\} \longrightarrow \{0.0, 1.0, 2.0, 3.0\}$$
All four classes were retained; the target was not collapsed into binary classification.

### Train / Test Partitioning
- **Split Ratio**: 80% Training, 20% Testing (fixed random seed `42`)
- **Training Set**: 6,182,916 records
- **Testing Set**: 1,545,478 records

### Feature Selection (32 Predictors)
- **Numeric (9)**: `Start_Lat`, `Start_Lng`, `Distance(mi)`, `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`
- **Categorical (7)**: `State`, `Timezone`, `Weather_Condition`, `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`
- **Infrastructure Boolean (12)**: `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`
- **Derived Temporal (4)**: `Year`, `Month`, `DayOfWeek`, `Hour`

### 16 Excluded Features & Leakage Prevention
To prevent data leakage, target contamination, and high-missingness bias, 16 features were strictly excluded:
- `Severity` (target variable)
- `ID`, `Description` (unique keys and text narrative)
- `Street`, `City`, `County`, `Zipcode`, `Airport_Code` (high-cardinality nominal proxies prone to memorization)
- `End_Time` (encodes incident duration, unavailable at onset)
- `End_Lat`, `End_Lng` (~44% missingness)
- `Weather_Timestamp` (redundant with start time)
- `Turning_Loop` (constant `False`)
- `Source`, `Country` (provenance metadata and constant `"US"`)
- `Start_Time` (raw timestamp dropped after extracting cyclical components)

---

## 10. Class Imbalance Handling

Because Severity 2 represents **79.67%** of the dataset, unweighted optimization leads models to prioritize the majority class at the expense of rare, high-consequence accidents. Rather than applying computationally intractable oversampling techniques like SMOTE (which causes memory exhaustion on 7.7M rows), **balanced class weighting** was implemented directly in the PySpark MLlib objective function:

$$w_c = \frac{N_{\text{train}}}{4 \times N_{\text{train}, c}}$$

### Calculated Training Partition Weights
- **Severity 1** ($N = 54,009$): **$w_1 = 28.6198$**
- **Severity 2** ($N = 4,925,105$): **$w_2 = 0.3138$**
- **Severity 3** ($N = 1,039,766$): **$w_3 = 1.4866$**
- **Severity 4** ($N = 164,036$): **$w_4 = 9.4231$**

Class weights were derived **strictly from the training partition** and passed via `weightCol="class_weight"` to each estimator.

---

## 11. Models Used

Three PySpark MLlib supervised classifiers were trained sequentially under strict resource controls:

1. **Multinomial Logistic Regression**:
   - Generalized linear model with softmax probability mapping (`family="multinomial"`).
   - Parameters: `maxIter=20`, `regParam=0.01`, `weightCol="class_weight"`.
   - Training time: 324.69 s (~5.4 min).
2. **Decision Tree Classifier**:
   - Non-parametric recursive binary tree with multiclass impurity minimization.
   - Parameters: `maxDepth=8`, `minInstancesPerNode=20`, `weightCol="class_weight"`.
   - Training time: 940.27 s (~15.7 min).
3. **Random Forest Classifier**:
   - Bagged ensemble of 25 randomized decision trees.
   - Parameters: `numTrees=25`, `maxDepth=8`, `minInstancesPerNode=20`, `seed=42`, `weightCol="class_weight"`.
   - Training time: 1698.93 s (~28.3 min).

---

## 12. Model Results

Models were evaluated on the independent **1,545,478-record held-out test partition**. Because of extreme class imbalance, performance is assessed using **Macro F1**, **Weighted F1**, and **Weighted Precision/Recall** in addition to raw accuracy.

### Summary Model Comparison

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Training Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.4208 | 0.7934 | 0.4208 | 0.5082 | 0.2930 | 324.69 |
| **Decision Tree** | **0.4738** | **0.8378** | **0.4738** | **0.5462** | **0.3656** | 940.27 |
| **Random Forest** | 0.3869 | 0.8282 | 0.3869 | 0.4628 | 0.2934 | 1698.93 |

> *"Within this experiment and on the held-out test set, the Decision Tree produced the highest measured Accuracy, Weighted F1 and Macro F1 among the three evaluated models."*

### Per-Class Evaluation Breakdown

| Model | Severity Level | Precision | Recall | F1-Score | Test Support |
|---|---|---:|---:|---:|---:|
| **Logistic Regression** | Severity 1 | 0.0348 | **0.8050** | 0.0667 | 13,357 |
| | Severity 2 | 0.9161 | 0.3826 | 0.5398 | 1,231,876 |
| | Severity 3 | 0.3633 | 0.5617 | 0.4413 | 259,571 |
| | Severity 4 | 0.0699 | **0.5508** | 0.1240 | 40,674 |
| **Decision Tree** | Severity 1 | **0.1394** | **0.9357** | **0.2426** | 13,357 |
| | Severity 2 | 0.9631 | 0.4050 | 0.5702 | 1,231,876 |
| | Severity 3 | 0.3986 | 0.7148 | 0.5118 | 259,571 |
| | Severity 4 | **0.0748** | **0.8685** | **0.1377** | 40,674 |
| **Random Forest** | Severity 1 | 0.0573 | **0.8873** | 0.1076 | 13,357 |
| | Severity 2 | 0.9589 | 0.3190 | 0.4787 | 1,231,876 |
| | Severity 3 | 0.3666 | 0.6101 | 0.4580 | 259,571 |
| | Severity 4 | 0.0699 | **0.8537** | 0.1293 | 40,674 |

### Impact of Balanced Class Weighting
Under balanced class weighting, the penalty for missing a minority instance was scaled up proportionally ($w_1 = 28.62$, $w_4 = 9.42$). As a result, the Decision Tree model achieved:
- **93.57% Recall on Severity 1** (12,498 correctly detected out of 13,357).
- **86.85% Recall on Severity 4** (35,325 correctly detected out of 40,674).
While overall raw accuracy dropped from the naive majority baseline of ~79.7% to 47.4% due to false alarms on the majority class, this trade-off is operationally advantageous: failing to anticipate a major highway closure has far more severe consequences than responding to a false alert.

---

## 13. Key Findings

1. **Extreme Class Imbalance**: Severity 2 dominates at **79.67%**, while Severity 1 (0.87%) and Severity 4 (2.65%) represent small minority classes.
2. **Weekday Concentration**: **76.55%** of accidents occur on weekdays, peaking on Friday (**16.01%**).
3. **Commuter Alignment**: Accidents concentrate heavily during morning rush hour (**7–8 AM: 14.09%**) and evening rush hour (**4–5 PM: 13.41%**).
4. **Geographic Clustering**: Three states (**CA, FL, TX**) account for **41.46%** of all recorded incidents nationwide.
5. **Prevalence of Fair Weather**: Over **81%** of accidents occur under fair or cloudy skies due to higher overall travel exposure.
6. **Infrastructure Associations**: Highway interchanges and junctions (`Junction=True`) exhibit a higher proportion of severe (Severity 3 and 4) events.
7. **Spatial Impact Correlation**: Severe accidents (Severity 4) impact an average corridor length of **1.50 miles**, compared to 0.30 miles for Severity 2.
8. **Class Weighting Efficacy**: Balanced class weights boosted minority class recall to **93.57%** (Severity 1) and **86.85%** (Severity 4) in the Decision Tree model.
9. **Algorithm Performance**: The Decision Tree classifier demonstrated superior overall performance, leading across Accuracy (0.4738), Weighted F1 (0.5462), and Macro F1 (0.3656).
10. **Scalable Big Data Execution**: Apache Spark successfully ingested, cleaned, engineered features, and trained ML pipelines on 7.7M rows within a constrained local memory budget (~7.37 GB RAM).

---

## 14. Project Structure

```text
BDA PROJECT/
├── .gitignore                                      # Comprehensive Git ignore rules
├── README.md                                       # Comprehensive academic project report
├── requirements.txt                                # Python dependency manifest
├── US_Accidents_March23.csv                        # [Local Raw Dataset, 2.85 GB - Excluded from Git]
│
├── data/
│   └── processed/
│       └── us_accidents_parquet/                   # [Local Parquet Data, ~726 MB - Excluded from Git]
│
├── docs/                                           # Academic and Technical Documentation
│   ├── analytics_findings.md                       # Comprehensive analytical findings
│   ├── ml_feature_selection.md                     # Feature selection & leakage prevention rationale
│   ├── ml_preprocessing.md                         # Preprocessing & class weighting methodology
│   ├── ml_results.md                               # Detailed ML performance metrics & per-class tables
│   └── viva_questions.md                           # 30 viva examination questions & answers
│
├── notebooks/                                      # Interactive Delivery
│   └── US_Traffic_Accident_Severity_Analytics_Final.ipynb  # Complete 22-section final notebook
│
├── outputs/                                        # Generated Artifacts
│   ├── charts/                                     # 18 Publication-quality visualizations
│   │   ├── accidents_by_day_of_week.png
│   │   ├── accidents_by_hour.png
│   │   ├── accidents_by_month.png
│   │   ├── accidents_by_year.png
│   │   ├── confusion_matrix_decision_tree.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── confusion_matrix_random_forest.png
│   │   ├── distance_buckets.png
│   │   ├── infrastructure_features.png
│   │   ├── model_comparison_metrics.png
│   │   ├── severity_distribution.png
│   │   ├── top_10_states.png
│   │   ├── top_20_cities.png
│   │   ├── train_test_class_distribution.png
│   │   └── weather_severity_summary.png
│   ├── tables/                                     # 23 Analytical & evaluation CSV tables
│   │   ├── dataset_overview.csv
│   │   ├── distance_summary.csv
│   │   ├── infrastructure_features.csv
│   │   ├── model_comparison.csv
│   │   ├── per_class_metrics.csv
│   │   ├── severity_distribution.csv
│   │   ├── sql_severity_summary.csv
│   │   ├── top_10_states.csv
│   │   └── top_weather_conditions.csv
│   └── models/                                     # [Trained PySpark Models - Excluded from Git]
│       ├── decision_tree/
│       ├── logistic_regression/
│       └── random_forest/
│
└── src/                                            # Modular Production Source Scripts
    ├── convert_to_parquet.py                       # Stage 5/6: Parquet conversion pipeline
    ├── count_full_dataset.py                       # Stage 3B: Deterministic row count verification
    ├── ingestion_smoke_test.py                     # Stage 3A: Explicit schema ingestion test
    ├── investigate_anomalies.py                    # Stage 4: Sensor & telemetry anomaly analysis
    ├── parquet_smoke_test.py                       # Stage 5A: Hadoop winutils filesystem test
    ├── profile_dataset.py                          # Stage 3C: Full dataset missingness profiling
    ├── run_bda_analytics.py                        # Stage 6: Distributed BDA analytics & charts
    └── train_ml_models.py                          # Stage 7: Sequential MLlib model training
```

---

## 15. Hardware / Resource Considerations

Executing distributed algorithms on 7.7 million rows on a machine with ~7.37 GB of RAM required careful memory management:
- **Single-Core Master (`local[1]`)**: Prevents multithreaded task contention and out-of-memory spikes.
- **Controlled Driver Memory (`2g`)**: Limits JVM heap to safe boundaries, leaving adequate headroom for OS and background operations.
- **Shuffle Optimization (`spark.sql.shuffle.partitions=2`)**: Reduces partition task metadata overhead during aggregations.
- **Zero Full-Dataset Collects**: Data transformations rely on lazy evaluation and streaming operations.

---

## 16. How to Run the Project

### Step 1: Environment Prerequisites
- **Python**: Version 3.10+ (tested on Python 3.13.2)
- **Java**: Java 17 (OpenJDK Adoptium 17.0.20.101-hotspot recommended)
- **Hadoop Winutils (Windows only)**: Hadoop 3.4+ `winutils.exe` and `hadoop.dll` placed under `C:\hadoop\bin` with `HADOOP_HOME=C:\hadoop`.

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Dataset Placement
Download `US_Accidents_March23.csv` and place it directly into the project root directory:
```text
D:\Users\Admin\Desktop\BDA PROJECT\US_Accidents_March23.csv
```

### Step 4: Run Ingestion and Parquet Conversion
```bash
python src/convert_to_parquet.py
```

### Step 5: Execute Analytics Pipeline
```bash
python src/run_bda_analytics.py
```

### Step 6: Execute Machine Learning Training Pipeline
```bash
python src/train_ml_models.py
```

### Step 7: Launch the Final Presentation Notebook
```bash
jupyter notebook notebooks/US_Traffic_Accident_Severity_Analytics_Final.ipynb
```
*Note: The final notebook is pre-configured to load existing outputs and metrics without re-running the 7.7M-row model training.*

---

## 17. Limitations

1. **Observational Dataset**: The dataset records reported accidents; statistical correlations do not establish causality.
2. **Lack of Traffic Exposure Denominators**: Incident counts are not normalized by Vehicle Miles Traveled (VMT) or roadway capacity, precluding risk-rate conclusions.
3. **Severe Natural Class Imbalance**: The 80% concentration of Severity 2 requires heavy loss weighting, which inherently increases false alarms on the majority class.
4. **Sensor & Meteorological Missingness**: Missingness in precipitation (28.5%) and coordinates (44.0%) required imputation and feature exclusions.
5. **Reporting Heterogeneity**: State reporting practices and sensor densities vary across jurisdictions.
6. **Local Single-Node Constraints**: Hyperparameter optimization was bounded to conservative tree depths (`maxDepth=8`) to prevent memory exhaustion.
7. **No Real-World Deployment Claims**: Model performance reflects offline test set evaluation and cannot guarantee live operational accuracy.

---

## 18. Future Scope

1. **Exposure Rate Integration**: Merge federal VMT and state traffic sensor feeds to calculate true risk rates per vehicle mile.
2. **Spatial Hotspot Modeling**: Apply spatial point process algorithms (e.g., Kernel Density Estimation) to identify micro-level crash clusters.
3. **Gradient Boosting on Distributed Clusters**: Deploy Spark MLlib's Gradient-Boosted Trees (GBTs) or XGBoost on a multi-node Apache Spark cluster.
4. **Temporal Cross-Validation**: Evaluate models using forward-chaining rolling-window splits to test resilience across different years.
5. **Real-Time Streaming Pipelines**: Ingest live incident telemetry via Apache Kafka and Spark Structured Streaming for real-time severity prediction.
6. **Model Interpretability**: Incorporate SHAP (SHapley Additive exPlanations) to explain feature contributions for individual predictions.

---

## 19. Conclusion

This project successfully demonstrated the application of **Apache Spark (PySpark)** for end-to-end Big Data Analytics on a massive real-world dataset of **7,728,394 traffic accident records**:
- Successfully processed 3.06 GB of raw CSV data using an explicit schema and converted it to high-performance Snappy Parquet format.
- Uncovered nationwide temporal, geographic, weather, and infrastructure patterns across 49 states.
- Implemented an automated ML pipeline with median imputation, string indexing, one-hot encoding, and balanced class loss weighting.
- Evaluated three MLlib classifiers on 1.55 million test records, with the **Decision Tree Classifier** demonstrating the best overall balance (Accuracy: 0.4738, Weighted F1: 0.5462, Macro F1: 0.3656) and successfully capturing **86.85% of critical Severity 4 accidents** and **93.57% of Severity 1 incidents**.

These results highlight both the power of distributed computing for large-scale transportation analytics and the essential role of balanced loss functions when modeling extreme real-world class imbalances.

---

## 20. Academic Note

This project was developed as a comprehensive Big Data Analytics college project demonstrating mastery of distributed data processing, data hygiene, exploratory analysis, distributed machine learning, class imbalance handling, and technical documentation. All analytical findings and metrics are grounded in verified outputs from the 7.7M-row dataset.
