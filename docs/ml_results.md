# Machine Learning Results — US Traffic Accident Severity Prediction

## 1. ML Objective
The objective of this stage is to build, evaluate, and compare memory-conscious, scalable machine learning classification pipelines in PySpark MLlib to predict traffic accident severity across the United States. Traffic accidents represent substantial societal, public health, and economic costs. Accurately anticipating accident severity from initial spatiotemporal, weather, and roadway infrastructure characteristics enables transportation authorities and emergency responders to deploy appropriate resources, optimize emergency routing, and mitigate secondary collisions.

---

## 2. Target Variable
The target variable is **`Severity`**, representing the impact of the accident on traffic delay and road capacity on an ordinal scale:
- **Severity 1**: Minor disruption / shortest delay (~0.87% of total records)
- **Severity 2**: Moderate disruption / short delay (~79.67% of total records; dominant class)
- **Severity 3**: Serious traffic disruption / significant congestion (~16.81% of total records)
- **Severity 4**: Critical event / major road closure and long delays (~2.65% of total records)

In PySpark MLlib, multiclass classifiers require zero-indexed labels. Therefore, `Severity` (values 1, 2, 3, 4) is mapped to numeric labels `0.0, 1.0, 2.0, 3.0` during training, while keeping all 4 distinct classes. It is **not** collapsed into a binary classification problem.

---

## 3. Feature Selection
A 32-attribute feature set was constructed based on domain defensibility and strict operational validity (available at the initial incident reporting window):
- **Numeric Features (9)**: `Start_Lat`, `Start_Lng`, `Distance(mi)`, `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`
- **Categorical Features (7)**: `State`, `Timezone`, `Weather_Condition`, `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`
- **Infrastructure Boolean Features (12)**: `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal` (converted to 0.0/1.0)
- **Temporal Features (4)**: Derived `Year`, `Month`, `DayOfWeek`, and `Hour` extracted from `Start_Time`.

### Strict Leakage Exclusion
The following 16 attributes were explicitly excluded to prevent target leakage, high missingness bias, or artificial predictive inflation:
- `Severity` (target variable itself)
- `ID`, `Description` (unique record keys and text narrative)
- `Street`, `City`, `County`, `Zipcode`, `Airport_Code` (high-cardinality nominal proxies that risk severe overfitting)
- `End_Time` (contains accident duration, which is inherently unknown at incident start)
- `End_Lat`, `End_Lng` (~44% missingness in dataset)
- `Weather_Timestamp` (redundant with incident temporal features)
- `Turning_Loop` (constant `False` across all 7.7M rows)
- `Source`, `Country` (provenance metadata and single-value constant `"US"`)
- `Start_Time` (raw timestamp dropped after extracting orthogonal components)

---

## 4. Preprocessing
The preprocessing pipeline was encapsulated in a single, reproducible PySpark `Pipeline`:
1. **Median Imputation**: `Imputer` fitted strictly on training data for numeric predictors.
2. **String Indexing**: `StringIndexer(handleInvalid="keep")` mapping categorical string levels to integer indices.
3. **One-Hot Encoding**: `OneHotEncoder(handleInvalid="keep")` producing sparse categorical indicator vectors.
4. **Vector Assembly**: `VectorAssembler(handleInvalid="keep")` concatenating all numeric, temporal, boolean, and encoded vectors into a single feature vector (`features`).

---

## 5. Missing Value Handling
- **Numeric Missingness**: Missing values in weather variables (such as precipitation, visibility, and wind speed) were imputed using the median learned from the training partition. The median was selected because weather parameters exhibit heavy right-skew and physical measurement outliers.
- **Categorical Missingness**: Categorical nulls were systematically coalesced to `"Unknown"` prior to indexing. The `handleInvalid="keep"` setting on indexers and encoders assigns any unobserved levels to an explicit reserve bin.
- **Boolean Features**: Missing boolean values were imputed to `0.0`.
- **Temporal Features**: Guarded with `F.coalesce` fallbacks to guarantee zero missing values.

No rows were dropped, ensuring full retention of all records across partitions.

---

## 6. Class Imbalance
The dataset exhibits extreme class imbalance: Severity 2 accounts for nearly 80% of all incidents, while Severity 1 accounts for under 0.9%. Blind models trained without intervention degenerate into majority-class predictors, ignoring critical severe incidents.

Rather than computationally prohibitive SMOTE (which causes out-of-memory errors on 7.7M rows), **balanced class weighting** was implemented natively via `weightCol`:
$$w_c = \frac{N_{\text{train}}}{4 \times N_{\text{train}, c}}$$

### Calculated Training Set Weights
- **Severity 1** ($N = 54,009$): **$w_1 = 28.6198$**
- **Severity 2** ($N = 4,925,105$): **$w_2 = 0.3138$**
- **Severity 3** ($N = 1,039,766$): **$w_3 = 1.4866$**
- **Severity 4** ($N = 164,036$): **$w_4 = 9.4231$**

Class weights were computed **strictly from the training partition** and passed to each classifier's loss function.

---

## 7. Train/Test Split
The dataset was split using an 80/20 train/test partition with fixed random seed `42`:
- **Training Set (80%)**: 6,182,916 records
  - Severity 1: 54,009 (0.87%)
  - Severity 2: 4,925,105 (79.66%)
  - Severity 3: 1,039,766 (16.82%)
  - Severity 4: 164,036 (2.65%)
- **Testing Set (20%)**: 1,545,478 records
  - Severity 1: 13,357 (0.86%)
  - Severity 2: 1,231,876 (79.71%)
  - Severity 3: 259,571 (16.80%)
  - Severity 4: 40,674 (2.63%)

The class proportions between training and testing splits match within 0.05%, confirming an unbiased split.

---

## 8. Algorithms
Three PySpark MLlib classifiers were trained sequentially under strict resource controls:
1. **Multinomial Logistic Regression**:
   - `family="multinomial"`, `maxIter=20`, `regParam=0.01`, `weightCol="class_weight"`
   - Training time: 324.69 seconds (~5.4 min)
2. **Decision Tree Classifier**:
   - `maxDepth=8`, `minInstancesPerNode=20`, `weightCol="class_weight"`
   - Training time: 940.27 seconds (~15.7 min)
3. **Random Forest Classifier**:
   - `numTrees=25`, `maxDepth=8`, `minInstancesPerNode=20`, `seed=42`, `weightCol="class_weight"`
   - Training time: 1698.93 seconds (~28.3 min)

---

## 9. Evaluation Metrics
Because of the heavy 80% majority dominance of Severity 2, standard raw accuracy is misleading (a naive dummy classifier predicting always Severity 2 would achieve 79.71% accuracy while achieving 0% recall on severe accidents).

Therefore, performance was evaluated across:
- **Accuracy**: Overall fraction of correct predictions across all classes.
- **Weighted Precision & Weighted Recall**: Precision and recall weighted by class support.
- **Weighted F1-Score**: Harmonic mean of weighted precision and recall.
- **Macro F1-Score**: Unweighted arithmetic average of F1 across all 4 classes ($\frac{1}{4}\sum_{c=1}^4 F1_c$), giving equal weight to minority and majority classes.
- **Per-Class Precision, Recall, F1, and Support**: Exact class-specific breakdown.

---

## 10. Results

### Overall Model Comparison (Test Set: 1,545,478 Records)

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | Training Time (s) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.4208 | 0.7934 | 0.4208 | 0.5082 | 0.2930 | 324.69 |
| **Decision Tree** | **0.4738** | **0.8378** | **0.4738** | **0.5462** | **0.3656** | 940.27 |
| **Random Forest** | 0.3869 | 0.8282 | 0.3869 | 0.4628 | 0.2934 | 1698.93 |

*Decision Tree achieved the strongest performance across all summary metrics, with the highest Accuracy (0.4738), Weighted F1 (0.5462), and Macro F1 (0.3656).*

---

## 11. Per-Class Performance

### Per-Class Evaluation Breakdown

| Model | Severity Class | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|---|
| **Logistic Regression** | Severity 1 | 0.0348 | 0.8050 | 0.0667 | 13,357 |
| | Severity 2 | 0.9161 | 0.3826 | 0.5398 | 1,231,876 |
| | Severity 3 | 0.3633 | 0.5617 | 0.4413 | 259,571 |
| | Severity 4 | 0.0699 | 0.5508 | 0.1240 | 40,674 |
| **Decision Tree** | Severity 1 | **0.1394** | **0.9357** | **0.2426** | 13,357 |
| | Severity 2 | 0.9631 | 0.4050 | 0.5702 | 1,231,876 |
| | Severity 3 | 0.3986 | 0.7148 | 0.5118 | 259,571 |
| | Severity 4 | **0.0748** | **0.8685** | **0.1377** | 40,674 |
| **Random Forest** | Severity 1 | 0.0573 | 0.8873 | 0.1076 | 13,357 |
| | Severity 2 | 0.9589 | 0.3190 | 0.4787 | 1,231,876 |
| | Severity 3 | 0.3666 | 0.6101 | 0.4580 | 259,571 |
| | Severity 4 | 0.0699 | 0.8537 | 0.1293 | 40,674 |

### Analysis of Minority Class Detection (Severity 1 and Severity 4)
1. **Minority Recall Surge**:
   Under balanced class weighting, the penalty for missing a minority instance was scaled up proportionally ($w_1 = 28.62$, $w_4 = 9.42$). Consequently, all three models learned decision boundaries prioritizing minority detection:
   - **Severity 1 Recall**: Decision Tree successfully identified **93.57%** of all Severity 1 accidents, Random Forest achieved **88.73%**, and Logistic Regression captured **80.50%**.
   - **Severity 4 Recall**: For critical closures, Decision Tree captured **86.85%** of true Severity 4 events, Random Forest captured **85.37%**, and Logistic Regression captured **55.08%**.
2. **Precision vs. Recall Tradeoff**:
   Because the models aggressively flag suspected minority cases, precision on Severity 1 and 4 is low (e.g., 0.1394 and 0.0748 for Decision Tree). Many Severity 2 accidents were classified into higher/lower severity bins.
3. **Operational Implication**:
   In emergency traffic management, false negatives on critical accidents (failing to detect a major highway closure) are significantly more hazardous than false positives. The class-weighted models prioritize sensitivity to high-severity events over raw majority-class accuracy.

---

## 12. Limitations

1. **Severe Natural Class Imbalance**:
   With Severity 2 representing ~80% of data, standard optimization inherently favors the majority class unless heavily weighted. Balancing weights trades off overall accuracy to recover minority recall.
2. **Missing Meteorological Observations**:
   Weather fields had missingness rates up to 33% (Precipitation) and 3% (Wind Speed). Median imputation, while robust, introduces localized bias during non-standard weather events.
3. **Observational & Reporting Disparities**:
   The US Accidents dataset relies on multiple aggregator APIs and municipal reporting channels. Certain states (e.g., California, Florida, Texas) have higher reporting frequencies due to sensor density rather than solely higher per-capita crash rates.
4. **Absence of Exposure Denominators**:
   The dataset lacks vehicle miles traveled (VMT), real-time traffic volumes, and roadway capacity metrics. True per-vehicle collision likelihood cannot be inferred.
5. **Absence of Causal Inference**:
   Correlations between features (such as traffic signals, weather conditions, or time of day) and accident severity reflect observational associations, not causal links.
6. **Local Hardware Constraints**:
   Training was executed on a single-node configuration (`local[1]`, 2 GB driver memory) on a 7.37 GB RAM machine. Model architectures were bounded to moderate depths (`maxDepth=8`, 25 trees) to prevent out-of-memory errors on 7.7 million records.
