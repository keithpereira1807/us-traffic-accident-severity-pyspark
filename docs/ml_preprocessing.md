# ML Data Preprocessing and Class Imbalance Strategy

## 1. Overview
This document details the preprocessing pipeline and class imbalance mitigation strategy implemented for the multiclass accident severity prediction models using PySpark MLlib.

---

## 2. Missing Value Imputation Strategy

The raw dataset exhibits significant missingness in meteorological indicators. Dropping missing rows would discard over 30% of the entire dataset. The following missing-value handling strategy is applied:

### Numeric Features
Features: `Start_Lat`, `Start_Lng`, `Distance(mi)`, `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`.
- **Strategy**: Median Imputation via PySpark `Imputer(strategy="median")`.
- **Why Median**: Numeric weather features exhibit severe skewness and physical outliers (e.g., wind speed max of 1,087 mph; pressure max of 58.63 in). The median provides a robust central tendency estimate insensitive to extreme outliers.
- **Leakage Prevention**: The `Imputer` estimator is fitted strictly on the **training set** (`train_df`). The learned training medians are subsequently applied to transform both the training and test sets.

### Categorical Features
Features: `State`, `Timezone`, `Weather_Condition`, `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`.
- **Strategy**: Missing values are filled with an explicit `"Unknown"` category string (`F.coalesce(F.col(c), F.lit("Unknown"))`).
- **Encoding**: Processed via `StringIndexer(handleInvalid="keep")` followed by `OneHotEncoder(handleInvalid="keep")`. This ensures unseen categories in future inference or test partitions are safely mapped to a dedicated index without runtime exceptions.

### Boolean Infrastructure Features
Features: `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`.
- **Strategy**: Converted safely into numeric double representations:
  $$\text{value} = \begin{cases} 1.0 & \text{if True} \\ 0.0 & \text{otherwise (False or Null)} \end{cases}$$

---

## 3. Temporal Feature Derivation

The raw `Start_Time` timestamp is decomposed into orthogonal cyclical components:
- `Year`: Four-digit calendar year (2016–2023)
- `Month`: Calendar month (1–12)
- `DayOfWeek`: Integer day of the week (1=Sunday, 7=Saturday)
- `Hour`: Local hour of the day (0–23)

Raw `Start_Time` is dropped after extraction to eliminate non-vectorizable timestamp types.

---

## 4. Class Imbalance Mitigation Strategy

### Measured Class Distribution
The target variable `Severity` exhibits extreme class imbalance:
- **Severity 1**: 67,366 records (~0.87%) — Minority Class
- **Severity 2**: 6,156,981 records (~79.67%) — Majority Class
- **Severity 3**: 1,299,337 records (~16.81%)
- **Severity 4**: 204,710 records (~2.65%) — Minority Class

### Rejection of SMOTE
Synthetic Minority Over-sampling Technique (SMOTE) requires $k$-nearest neighbor graph computation in high-dimensional feature spaces. On a 7.7 million record dataset with 7.37 GB of host RAM, global SMOTE is computationally intractable, causes out-of-memory errors, and generates synthetic artifacts along linear interpolations.

### Balanced Class Weight Formulation
Class imbalance is resolved directly inside the PySpark MLlib optimization objective using loss weighting (`weightCol`):

For each severity class $c \in \{1, 2, 3, 4\}$ (indexed as $0, 1, 2, 3$):
$$w_c = \frac{N_{\text{train}}}{K \times N_{\text{train}, c}}$$
where:
- $N_{\text{train}}$ = total number of training records
- $K$ = number of classes ($K = 4$)
- $N_{\text{train}, c}$ = count of training records belonging to class $c$

### Strict Leakage Rules for Class Weights
1. Class weights $w_c$ are calculated **strictly from the training partition** ($80\%$).
2. The test partition ($20\%$) has zero influence on the weight calculations.
3. Class weights are passed to the classifiers via `weightCol="class_weight"`, supported natively by `LogisticRegression`, `DecisionTreeClassifier`, and `RandomForestClassifier`.
