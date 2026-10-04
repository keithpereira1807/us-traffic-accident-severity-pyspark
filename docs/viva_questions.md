# US Traffic Accident Severity Analytics Using PySpark
## Comprehensive Viva / Oral Examination Guide (Q&A)

This document contains 30 key questions and concise, technically rigorous answers tailored for project viva, oral examination, and technical defense.

---

### 1. Why did you choose PySpark for this project?
PySpark provides a scalable distributed computing framework that handles dataset sizes exceeding local host memory. Unlike single-machine tools like Pandas that require loading the entire dataset into RAM, Spark utilizes lazy evaluation, Catalyst query optimization, Tungsten binary processing, and partitioning to execute transformations efficiently on large volumes.

### 2. Why is this considered a Big Data problem?
The dataset consists of **7,728,394 records across 46 attributes (~3.06 GB raw CSV)**. On standard personal computers with limited memory (such as 7.37 GB RAM), standard in-memory libraries experience out-of-memory (OOM) crashes. Processing, aggregating, and modeling millions of records reliably necessitates distributed, memory-conscious big data engineering.

### 3. What are the dimensions and scope of the dataset?
- **Total Records**: 7,728,394 rows
- **Total Attributes**: 46 columns
- **Distinct Primary Keys**: 7,728,394 unique `ID` values
- **Temporal Span**: January 15, 2016 through April 1, 2023 (~7.2 years)
- **Geographic Coverage**: 49 US States, 13,678 cities, and 1,871 counties

### 4. Why did you convert the raw CSV into Parquet format?
Parquet is an open-source, columnar storage format. Compared to row-oriented CSV:
- It supports **Snappy compression**, drastically reducing disk footprint (~726 MB vs. 3.06 GB).
- It enables **column pruning** (reading only requested columns) and **predicate pushdown** (filtering rows at the storage level using column min/max statistics).
- It embeds strict metadata and schemas, avoiding repetitive string-to-type parsing.

### 5. Why did you use an explicit schema instead of `inferSchema=True`?
Using `inferSchema=True` forces Spark to scan the entire 7.7 million-row CSV file twice—once to determine data types and once to read the data. Providing an explicit `StructType` schema executes in a single pass, prevents ambiguous type guessing, ensures deterministic column data types, and significantly speeds up ingestion.

### 6. What is a Spark DataFrame?
A Spark DataFrame is a distributed collection of structured data organized into named columns, conceptually equivalent to a table in a relational database. Under the hood, it is built on top of resilient distributed datasets (RDDs) and optimized using Spark's Catalyst engine for execution plan generation.

### 7. What is Spark SQL and how was it utilized?
Spark SQL is Apache Spark's module for structured data processing. It allows querying DataFrames using ANSI SQL syntax via temporary views (`df.createOrReplaceTempView("us_accidents")`). In this project, Spark SQL was used to execute multi-dimensional aggregations, including state accident volumes, annual growth rates, and severity cross-tabulations.

### 8. Why was `Severity` chosen as the target variable?
`Severity` represents the degree of impact on roadway traffic capacity and delay on an ordinal scale of 1 to 4. Anticipating incident severity enables transportation departments, traffic management centers, and emergency services to optimize resource deployment, emergency dispatch routing, and secondary collision countermeasures.

### 9. Why is class imbalance a major problem in this dataset?
The ground truth distribution is heavily skewed:
- **Severity 1**: 0.87% (67,366)
- **Severity 2**: 79.67% (6,156,981 — dominant class)
- **Severity 3**: 16.81% (1,299,337)
- **Severity 4**: 2.65% (204,710)
A naive, unweighted classifier predicting only Severity 2 would achieve ~79.7% raw accuracy while missing 100% of critical road closures (Severity 4) and minor disruptions (Severity 1).

### 10. What is balanced class weighting?
Balanced class weighting adjusts the loss function objective so that misclassifications of minority classes incur larger penalties than errors on majority classes. The weight for class $c$ is calculated as:
$$w_c = \frac{N_{\text{train}}}{K \times N_{\text{train}, c}}$$
Where $K=4$ classes, $N_{\text{train}}$ is total training rows, and $N_{\text{train}, c}$ is the count for class $c$.

### 11. Why must class weights be calculated strictly on the training partition?
Calculating class weights (or any preprocessing statistics) across the entire dataset introduces **data leakage**, allowing information from the unseen test set to influence model parameter estimation. Calculating weights strictly on the 80% training split preserves test set independence.

### 12. What is data leakage and how was it prevented?
Data leakage occurs when information from outside the training dataset or information not available at prediction time is inadvertently used to train the model. It was prevented by:
- Splitting data into 80% train / 20% test partitions before fitting estimators.
- Fitting imputation, string indexing, and encoding transformers solely on the training partition.
- Explicitly excluding post-incident and proxy leakage attributes.

### 13. Which features were excluded due to leakage, missingness, or variance?
16 features were explicitly excluded:
- `Severity` (target variable)
- `ID`, `Description` (unique identifiers / text)
- `Street`, `City`, `County`, `Zipcode`, `Airport_Code` (high-cardinality nominal proxies prone to overfitting)
- `End_Time` (contains accident duration, inherently unknown at incident onset)
- `End_Lat`, `End_Lng` (~44% missingness)
- `Weather_Timestamp` (redundant with start time)
- `Turning_Loop` (constant `False` across all rows)
- `Source`, `Country` (provenance metadata / constant `"US"`)
- `Start_Time` (raw timestamp dropped after extracting orthogonal components)

### 14. Why did you use median imputation rather than mean imputation for numeric values?
Meteorological variables (e.g., wind speed, precipitation, pressure) exhibit heavy skewness and telemetry outliers (e.g., recorded wind speeds of 1,087 mph). The median provides a robust measure of central tendency that is unaffected by extreme outliers.

### 15. What is `StringIndexer` in PySpark MLlib?
`StringIndexer` maps categorical string columns into numerical category indices ordered by frequency (e.g., the most frequent category is assigned index 0.0). Setting `handleInvalid="keep"` maps any unseen category during test inference to a dedicated index bin without throwing runtime exceptions.

### 16. What is `OneHotEncoder` in PySpark MLlib?
`OneHotEncoder` maps indexed categorical integers into binary sparse vectors. This prevents linear and distance-based algorithms from interpreting arbitrary integer index orderings as ordinal or metric relationships.

### 17. Why is `VectorAssembler` required in PySpark MLlib?
PySpark MLlib estimators require all feature columns to be combined into a single unified vector column (conventionally named `"features"`). `VectorAssembler` concatenates numeric doubles, boolean doubles, and sparse one-hot encoded vectors into this format.

### 18. Why was Multinomial Logistic Regression selected?
Logistic Regression serves as a disciplined, interpretable linear baseline. In multiclass mode (`family="multinomial"`), it fits a softmax generalized linear model, optimizing class probabilities via cross-entropy loss with native support for `weightCol`.

### 19. Why was the Decision Tree Classifier selected?
Decision Trees handle non-linear feature interactions, mixed categorical/continuous data, and threshold boundaries without requiring feature normalization. In PySpark, it natively supports `weightCol` and multiclass labels.

### 20. Why was Random Forest Classifier selected?
Random Forest is an ensemble method that constructs multiple randomized decision trees and aggregates their predictions via bagging. It reduces variance, mitigates single-tree overfitting, and typically improves generalization on complex datasets.

### 21. What is Accuracy and what is its formula?
Accuracy measures the proportion of correct predictions across all classes:
$$\text{Accuracy} = \frac{\text{Total Correct Predictions}}{\text{Total Instances}} = \frac{\sum_{c=1}^4 \text{TP}_c}{N}$$

### 22. What is Precision?
Precision measures the proportion of true positives among all instances predicted as that class:
$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$
It reflects how reliable a positive prediction is for a given severity level.

### 23. What is Recall?
Recall (sensitivity) measures the proportion of actual class instances correctly identified:
$$\text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
High recall indicates the model rarely misses instances of that severity level.

### 24. What is the F1-Score?
The F1-score is the harmonic mean of precision and recall:
$$F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
It balances false positives and false negatives into a single balanced metric.

### 25. What is Macro F1 and why is it crucial for imbalanced datasets?
Macro F1 is the unweighted arithmetic mean of F1-scores across all individual classes:
$$\text{Macro } F1 = \frac{1}{K} \sum_{c=1}^K F1_c$$
Unlike Weighted F1 (which is dominated by the majority Severity 2 class), Macro F1 gives equal importance to every class, providing an honest assessment of minority-class performance.

### 26. Why is raw accuracy alone insufficient for evaluating these models?
Because Severity 2 constitutes 79.67% of the dataset, a trivial model predicting only Severity 2 achieves 79.67% accuracy while having **0% recall** on Severity 1, 3, and 4. Raw accuracy masks total failure on critical minority classes.

### 27. Which model produced the highest measured performance in this experiment?
On the 1,545,478-record test partition:
- **Decision Tree Classifier** achieved the highest overall metrics:
  - **Accuracy**: **0.4738**
  - **Weighted F1**: **0.5462**
  - **Macro F1**: **0.3656**
- Logistic Regression: Accuracy 0.4208, Weighted F1 0.5082, Macro F1 0.2930
- Random Forest: Accuracy 0.3869, Weighted F1 0.4628, Macro F1 0.2934

### 28. What was the exact operational impact of balanced class weighting?
Balanced class weighting successfully shifted the models' attention to rare, critical events:
- **Severity 1 Recall**: Reached **93.57%** in Decision Tree (vs. near 0% under unweighted training).
- **Severity 4 Recall**: Reached **86.85%** in Decision Tree (and 85.37% in Random Forest).
The trade-off was lower minority precision (~7%–14%) and reduced majority accuracy, which is operationally desirable for safety-critical incident response where failing to detect a severe highway closure has far worse consequences than a false alarm.

### 29. What are the key empirical findings from the analytics?
1. **Temporal**: 76.55% of accidents occur on weekdays (Friday peak: 16.01%); diurnal peaks align with commute hours (7–8 AM: 14.09%, 4–5 PM: 13.41%).
2. **Geographic**: California, Florida, and Texas account for 41.46% of records.
3. **Weather**: Over 81% of accidents occur under fair or cloudy skies due to higher exposure volume.
4. **Infrastructure**: Highway junctions (`Junction=True`) exhibit a higher proportion of severe (Severity 3 and 4) accidents.
5. **Corridor Extent**: Severity 4 accidents impact an average road extent of 1.50 miles, compared to 0.30 miles for Severity 2.

### 30. What are the primary academic limitations of this study?
- **Observational Data**: Associations do not prove causality.
- **No Exposure Denominators**: Counts reflect incident volume, not per-capita or Vehicle Miles Traveled (VMT) risk rates.
- **Sensor Missingness**: High missingness in weather and coordinate fields required imputation.
- **Hardware Constraints**: Models were trained on single-node execution (`local[1]`, 2 GB driver RAM), limiting hyperparameter search and tree depth.
- **No Production Claim**: Results reflect historical offline evaluation and do not represent production-deployed operational guarantees.
