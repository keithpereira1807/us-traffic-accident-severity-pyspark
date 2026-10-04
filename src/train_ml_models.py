"""
STAGE 7 — MACHINE LEARNING PIPELINE, CLASS IMBALANCE HANDLING, AND MODEL TRAINING
US Traffic Accident Severity Analytics Using PySpark

Trains and evaluates:
1. Multiclass Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Features balanced class weights, strict leakage prevention, Spark ML pipelines,
sequential model training, and memory-conscious resource controls.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# Ensure JAVA_HOME, HADOOP_HOME and Python paths are properly configured
if "JAVA_HOME" not in os.environ or not os.environ["JAVA_HOME"]:
    adoptium_jdk = Path(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")
    if adoptium_jdk.exists():
        os.environ["JAVA_HOME"] = str(adoptium_jdk)
        os.environ["PATH"] = str(adoptium_jdk / "bin") + os.pathsep + os.environ.get("PATH", "")

if "HADOOP_HOME" not in os.environ or not os.environ["HADOOP_HOME"]:
    hadoop_dir = Path(r"C:\hadoop")
    if hadoop_dir.exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        os.environ["PATH"] = str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "US_Accidents_March23.csv"
PARQUET_DIR = PROJECT_ROOT / "data" / "processed" / "us_accidents_parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

EXPECTED_ROWS = 7_728_394
EXPECTED_CSV_BYTES = 3_058_183_727


def compute_metrics_and_cm(preds_df, num_classes=4):
    """
    Computes exact confusion matrix, per-class metrics, and macro/weighted F1
    from a single aggregated group-by on test predictions.
    """
    cm_rows = preds_df.groupBy("label", "prediction").agg(F.count(F.lit(1)).alias("count")).collect()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for r in cm_rows:
        true_l = int(float(r["label"]))
        pred_l = int(float(r["prediction"]))
        if 0 <= true_l < num_classes and 0 <= pred_l < num_classes:
            cm[true_l, pred_l] = r["count"]

    total_samples = cm.sum()
    total_correct = np.trace(cm)
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0

    per_class = []
    f1_list = []
    weighted_prec_sum = 0.0
    weighted_rec_sum = 0.0
    weighted_f1_sum = 0.0

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class.append({
            "Severity": c + 1,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "Support": int(support),
        })
        f1_list.append(f1)
        weighted_prec_sum += prec * support
        weighted_rec_sum += rec * support
        weighted_f1_sum += f1 * support

    macro_f1 = float(np.mean(f1_list))
    weighted_precision = float(weighted_prec_sum / total_samples) if total_samples > 0 else 0.0
    weighted_recall = float(weighted_rec_sum / total_samples) if total_samples > 0 else 0.0
    weighted_f1 = float(weighted_f1_sum / total_samples) if total_samples > 0 else 0.0

    summary_metrics = {
        "Accuracy": round(accuracy, 4),
        "Weighted_Precision": round(weighted_precision, 4),
        "Weighted_Recall": round(weighted_recall, 4),
        "Weighted_F1": round(weighted_f1, 4),
        "Macro_F1": round(macro_f1, 4),
    }

    return summary_metrics, per_class, cm


def plot_confusion_matrix(cm, model_name, output_path):
    """Saves a publication-quality confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    classes = ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    # Annotate numbers and percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            row_sum = row_sums[i, 0]
            pct = (val / row_sum * 100) if row_sum > 0 else 0.0
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=8.5)

    ax.set_title(f"Confusion Matrix — {model_name}\n(Test Set Predictions)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Severity", fontsize=11, labelpad=8)
    ax.set_ylabel("True Severity", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {output_path.name}")


def main() -> None:
    t_start = time.perf_counter()

    # Step 18: Verify raw CSV before ML
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")
    csv_size_before = RAW_CSV.stat().st_size
    print("=== PART 18: RAW DATA INTEGRITY (PRE-ML) ===")
    print(f"RAW CSV SIZE BEFORE: {csv_size_before} bytes")
    if csv_size_before != EXPECTED_CSV_BYTES:
        raise ValueError(f"Raw CSV size mismatch: expected {EXPECTED_CSV_BYTES}, got {csv_size_before}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("us_accidents_stage7_ml")
        .master("local[1]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.maxResultSize", "128m")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.hadoop.home.dir", os.environ.get("HADOOP_HOME", r"C:\hadoop"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        # =====================================================================
        # PART 1 — LOAD VERIFIED PARQUET
        # =====================================================================
        print("\n=== ML DATASET LOAD ===")
        df_raw = spark.read.parquet(str(PARQUET_DIR))
        total_rows = df_raw.count()
        total_cols = len(df_raw.columns)
        distinct_ids = df_raw.select(F.countDistinct("ID")).collect()[0][0]

        sev_dist_raw = df_raw.groupBy("Severity").agg(F.count(F.lit(1)).alias("count")).orderBy("Severity").collect()
        sev_dist_str = ", ".join(f"Severity {r['Severity']}: {r['count']:,}" for r in sev_dist_raw)

        print(f"Rows: {total_rows}")
        print(f"Columns: {total_cols}")
        print(f"Severity distribution: {sev_dist_str}")
        print(f"Distinct IDs: {distinct_ids}")

        if total_rows != EXPECTED_ROWS:
            raise RuntimeError(f"Row count mismatch: expected {EXPECTED_ROWS}, got {total_rows}")
        if distinct_ids != EXPECTED_ROWS:
            raise RuntimeError(f"Distinct ID mismatch: expected {EXPECTED_ROWS}, got {distinct_ids}")

        # =====================================================================
        # PART 2 — DEFINE TARGET
        # =====================================================================
        print("\n=== PART 2: TARGET DEFINITION ===")
        print("Target: Severity (Multiclass 1, 2, 3, 4 -> mapped to 0.0, 1.0, 2.0, 3.0)")
        for r in sev_dist_raw:
            pct = (r["count"] / total_rows) * 100
            print(f"  Class Severity {r['Severity']}: {r['count']:,} records ({pct:.2f}%)")

        # =====================================================================
        # PART 3 & 17 — FEATURE SELECTION AND LEAKAGE CHECK
        # =====================================================================
        print("\n=== PART 17: LEAKAGE CHECK ===")
        numeric_features = [
            "Start_Lat", "Start_Lng", "Distance(mi)", "Temperature(F)",
            "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
            "Precipitation(in)",
        ]
        categorical_features = [
            "State", "Timezone", "Weather_Condition", "Sunrise_Sunset",
            "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
        ]
        boolean_features = [
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
            "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
        ]
        time_features = ["Year", "Month", "DayOfWeek", "Hour"]

        candidate_features = numeric_features + categorical_features + boolean_features + time_features
        forbidden_features = [
            "Severity", "ID", "Description", "Street", "City", "County",
            "Zipcode", "Airport_Code", "End_Time", "End_Lat", "End_Lng",
            "Weather_Timestamp", "Turning_Loop", "Source", "Country", "Start_Time",
        ]

        leakage_detected = [f for f in candidate_features if f in forbidden_features]
        if leakage_detected:
            raise RuntimeError(f"FATAL: Forbidden leakage features detected: {leakage_detected}")

        print("Features Used (Total 28):")
        print(f"  - Numeric ({len(numeric_features)}): {numeric_features}")
        print(f"  - Categorical ({len(categorical_features)}): {categorical_features}")
        print(f"  - Infrastructure Boolean ({len(boolean_features)}): {boolean_features}")
        print(f"  - Derived Temporal ({len(time_features)}): {time_features}")
        print(f"Features Excluded ({len(forbidden_features)}): {forbidden_features}")
        print("Forbidden features check: PASSED (Zero leakage detected)")

        # =====================================================================
        # PART 4 & 5 — PREPARE DATA & TIME FEATURES
        # =====================================================================
        print("\n=== PART 4 & 5: PREPROCESSING PREPARATION ===")
        # Derive time features safely with coalesce to prevent any nulls
        prep_df = df_raw.withColumn("label", (F.col("Severity") - 1).cast("double"))
        prep_df = (
            prep_df.withColumn("Year", F.coalesce(F.year("Start_Time").cast("double"), F.lit(2020.0)))
            .withColumn("Month", F.coalesce(F.month("Start_Time").cast("double"), F.lit(6.0)))
            .withColumn("DayOfWeek", F.coalesce(F.dayofweek("Start_Time").cast("double"), F.lit(4.0)))
            .withColumn("Hour", F.coalesce(F.hour("Start_Time").cast("double"), F.lit(12.0)))
        )

        for b in boolean_features:
            prep_df = prep_df.withColumn(b, F.when(F.col(b) == True, 1.0).otherwise(0.0))

        for c in categorical_features:
            prep_df = prep_df.withColumn(c, F.coalesce(F.col(c), F.lit("Unknown")))

        # Drop raw Start_Time and unused columns to conserve memory
        columns_to_keep = ["label", "Severity"] + candidate_features
        prep_df = prep_df.select(columns_to_keep)

        # =====================================================================
        # PART 8 — TRAIN / TEST SPLIT (80/20)
        # =====================================================================
        print("\n=== PART 8: TRAIN / TEST SPLIT ===")
        train_raw, test_raw = prep_df.randomSplit([0.8, 0.2], seed=42)

        train_count = train_raw.count()
        test_count = test_raw.count()
        print(f"Training rows: {train_count:,} ({train_count / total_rows * 100:.2f}%)")
        print(f"Testing rows: {test_count:,} ({test_count / total_rows * 100:.2f}%)")

        train_sev_rows = train_raw.groupBy("Severity").agg(F.count(F.lit(1)).alias("count")).orderBy("Severity").collect()
        test_sev_rows = test_raw.groupBy("Severity").agg(F.count(F.lit(1)).alias("count")).orderBy("Severity").collect()

        train_sev_dict = {r["Severity"]: r["count"] for r in train_sev_rows}
        test_sev_dict = {r["Severity"]: r["count"] for r in test_sev_rows}

        print("Training severity distribution:")
        for s in [1, 2, 3, 4]:
            c = train_sev_dict.get(s, 0)
            print(f"  Severity {s}: {c:,} ({c / train_count * 100:.2f}%)")

        print("Testing severity distribution:")
        for s in [1, 2, 3, 4]:
            c = test_sev_dict.get(s, 0)
            print(f"  Severity {s}: {c:,} ({c / test_count * 100:.2f}%)")

        # Visualization: Train vs Test Class Distribution
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(4)
        width = 0.35
        train_p = [train_sev_dict[s] / train_count * 100 for s in [1, 2, 3, 4]]
        test_p = [test_sev_dict[s] / test_count * 100 for s in [1, 2, 3, 4]]
        ax.bar(x - width / 2, train_p, width, label="Train Set (80%)", color="#2b5c8f", edgecolor="black")
        ax.bar(x + width / 2, test_p, width, label="Test Set (20%)", color="#e28743", edgecolor="black")
        ax.set_title("Severity Class Distribution: Train vs Test Split (Seed 42)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Severity Level", fontsize=11)
        ax.set_ylabel("Percentage of Partition (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["Severity 1\n(0.87%)", "Severity 2\n(79.67%)", "Severity 3\n(16.81%)", "Severity 4\n(2.65%)"])
        ax.legend()
        plt.tight_layout()
        chart_split_path = CHARTS_DIR / "train_test_class_distribution.png"
        plt.savefig(chart_split_path, dpi=150)
        plt.close()
        print(f"Saved chart: {chart_split_path.name}")

        # =====================================================================
        # PART 7 — CLASS IMBALANCE STRATEGY (TRAINING SET ONLY)
        # =====================================================================
        print("\n=== PART 7: CLASS WEIGHTING (TRAIN SET ONLY) ===")
        # Formula: w_c = N_train / (4 * N_train_c)
        class_weights = {}
        for s in [1, 2, 3, 4]:
            n_c = train_sev_dict.get(s, 1)
            class_weights[float(s - 1)] = float(train_count / (4.0 * n_c))
            print(f"  Class Severity {s} (label {s - 1}.0): N={n_c:,}, weight={class_weights[float(s - 1)]:.4f}")

        weight_expr = (
            F.when(F.col("label") == 0.0, class_weights[0.0])
            .when(F.col("label") == 1.0, class_weights[1.0])
            .when(F.col("label") == 2.0, class_weights[2.0])
            .when(F.col("label") == 3.0, class_weights[3.0])
            .otherwise(1.0)
        )
        train_raw = train_raw.withColumn("class_weight", weight_expr)
        test_raw = test_raw.withColumn("class_weight", F.lit(1.0))

        # =====================================================================
        # PART 6 — BUILD & FIT PREPROCESSING PIPELINE
        # =====================================================================
        print("\n=== PART 6: PIPELINE FITTING (TRAIN SET ONLY) ===")
        # Imputer for numerics
        imputer_out = [c + "_imp" for c in numeric_features]
        imputer = Imputer(inputCols=numeric_features, outputCols=imputer_out, strategy="median")

        # StringIndexer and OneHotEncoder for categoricals
        idx_out = [c + "_idx" for c in categorical_features]
        vec_out = [c + "_vec" for c in categorical_features]
        indexers = [
            StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
            for c in categorical_features
        ]
        encoder = OneHotEncoder(inputCols=idx_out, outputCols=vec_out, handleInvalid="keep")

        # VectorAssembler with handleInvalid="keep" to guarantee zero rows skipped
        assembler_inputs = imputer_out + time_features + boolean_features + vec_out
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

        pipeline_stages = [imputer] + indexers + [encoder, assembler]
        pipeline = Pipeline(stages=pipeline_stages)

        print("Fitting Preprocessing Pipeline on training data...")
        t_pipe = time.perf_counter()
        pipeline_model = pipeline.fit(train_raw)
        print(f"Pipeline fitted in {time.perf_counter() - t_pipe:.2f} s")

        print("Transforming train and test datasets...")
        train_df = pipeline_model.transform(train_raw).select("features", "label", "Severity", "class_weight")
        test_df = pipeline_model.transform(test_raw).select("features", "label", "Severity", "class_weight")

        # =====================================================================
        # PART 10, 11, 12, 13 — TRAIN THREE MODELS SEQUENTIALLY
        # =====================================================================
        models_config = [
            (
                "Logistic Regression",
                "logistic_regression",
                LogisticRegression(
                    featuresCol="features",
                    labelCol="label",
                    weightCol="class_weight",
                    maxIter=20,
                    regParam=0.01,
                    family="multinomial",
                ),
            ),
            (
                "Decision Tree",
                "decision_tree",
                DecisionTreeClassifier(
                    featuresCol="features",
                    labelCol="label",
                    weightCol="class_weight",
                    maxDepth=8,
                    minInstancesPerNode=20,
                ),
            ),
            (
                "Random Forest",
                "random_forest",
                RandomForestClassifier(
                    featuresCol="features",
                    labelCol="label",
                    weightCol="class_weight",
                    numTrees=25,
                    maxDepth=8,
                    seed=42,
                    minInstancesPerNode=20,
                ),
            ),
        ]

        comparison_results = []
        all_per_class_results = []

        for model_title, model_dirname, estimator in models_config:
            print(f"\n{'=' * 60}")
            print(f"TRAINING MODEL: {model_title}")
            print(f"{'=' * 60}")
            t_m0 = time.perf_counter()

            model = estimator.fit(train_df)
            train_duration = time.perf_counter() - t_m0
            print(f"{model_title} training completed in {train_duration:.2f} s")

            # Save model
            save_path = MODELS_DIR / model_dirname
            if save_path.exists():
                shutil.rmtree(save_path)
            model.save(str(save_path))
            print(f"Saved model to: {save_path.relative_to(PROJECT_ROOT)}")

            # Predictions on test set
            print(f"Generating predictions on test set ({test_count:,} rows)...")
            preds = model.transform(test_df)
            pred_count = preds.count()
            print(f"Prediction row count: {pred_count:,} (Expected: {test_count:,})")
            if pred_count != test_count:
                raise RuntimeError(f"Prediction count mismatch for {model_title}: {pred_count:,} vs {test_count:,}")

            # Print 10 prediction examples (Part 11)
            print(f"\n=== 10 PREDICTION EXAMPLES ({model_title}) ===")
            preds_sample = preds.select("Severity", (F.col("prediction") + 1).cast("int").alias("Predicted_Severity"), "probability").limit(10).collect()
            for i, p in enumerate(preds_sample, 1):
                prob_vec = p["probability"]
                prob_str = "[" + ", ".join(f"{v:.3f}" for v in prob_vec.toArray()) + "]" if prob_vec is not None else "N/A"
                print(f"  Example {i}: True Severity={p['Severity']}, Predicted={p['Predicted_Severity']}, Probabilities={prob_str}")

            # Compute exact metrics and confusion matrix
            summary_metrics, per_class, cm = compute_metrics_and_cm(preds, num_classes=4)
            summary_metrics["Model"] = model_title
            summary_metrics["Training_Time_s"] = round(train_duration, 2)
            comparison_results.append(summary_metrics)

            for pc in per_class:
                pc["Model"] = model_title
                all_per_class_results.append(pc)

            print(f"\n{model_title} Test Set Metrics:")
            print(f"  Accuracy:           {summary_metrics['Accuracy']}")
            print(f"  Weighted Precision: {summary_metrics['Weighted_Precision']}")
            print(f"  Weighted Recall:    {summary_metrics['Weighted_Recall']}")
            print(f"  Weighted F1:        {summary_metrics['Weighted_F1']}")
            print(f"  Macro F1:           {summary_metrics['Macro_F1']}")

            # Save Confusion Matrix Chart (Part 13)
            cm_chart_path = CHARTS_DIR / f"confusion_matrix_{model_dirname}.png"
            plot_confusion_matrix(cm, model_title, cm_chart_path)

        # =====================================================================
        # PART 12 — SAVE METRIC TABLES
        # =====================================================================
        print("\n=== PART 12: SAVING MODEL EVALUATION TABLES ===")
        comp_df = pd.DataFrame(comparison_results)[["Model", "Accuracy", "Weighted_Precision", "Weighted_Recall", "Weighted_F1", "Macro_F1", "Training_Time_s"]]
        comp_path = TABLES_DIR / "model_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"Saved: {comp_path.relative_to(PROJECT_ROOT)}")

        per_class_df = pd.DataFrame(all_per_class_results)[["Model", "Severity", "Precision", "Recall", "F1", "Support"]]
        per_class_path = TABLES_DIR / "per_class_metrics.csv"
        per_class_df.to_csv(per_class_path, index=False)
        print(f"Saved: {per_class_path.relative_to(PROJECT_ROOT)}")

        # =====================================================================
        # PART 15 — MODEL COMPARISON CHART
        # =====================================================================
        print("\n=== PART 15: MODEL COMPARISON CHART ===")
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(comp_df))
        w = 0.25
        ax.bar(x - w, comp_df["Accuracy"], w, label="Accuracy", color="#4e79a7", edgecolor="black")
        ax.bar(x, comp_df["Weighted_F1"], w, label="Weighted F1", color="#59a14f", edgecolor="black")
        ax.bar(x + w, comp_df["Macro_F1"], w, label="Macro F1", color="#f28e2b", edgecolor="black")
        ax.set_title("Multiclass Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df["Model"], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        for i in range(len(comp_df)):
            ax.text(i - w, comp_df.loc[i, "Accuracy"] + 0.02, f"{comp_df.loc[i, 'Accuracy']:.2f}", ha="center", fontsize=8)
            ax.text(i, comp_df.loc[i, "Weighted_F1"] + 0.02, f"{comp_df.loc[i, 'Weighted_F1']:.2f}", ha="center", fontsize=8)
            ax.text(i + w, comp_df.loc[i, "Macro_F1"] + 0.02, f"{comp_df.loc[i, 'Macro_F1']:.2f}", ha="center", fontsize=8)

        plt.tight_layout()
        chart_comp_path = CHARTS_DIR / "model_comparison_metrics.png"
        plt.savefig(chart_comp_path, dpi=150)
        plt.close()
        print(f"Saved chart: {chart_comp_path.name}")

        # =====================================================================
        # PART 18 & 19 — FINAL VERIFICATION AND SUMMARY
        # =====================================================================
        csv_size_after = RAW_CSV.stat().st_size
        csv_unchanged = RAW_CSV.exists() and csv_size_before == EXPECTED_CSV_BYTES and csv_size_after == EXPECTED_CSV_BYTES
        print(f"\nRAW CSV SIZE AFTER: {csv_size_after} bytes")
        print(f"RAW CSV UNCHANGED: {csv_unchanged}")

        elapsed_total = time.perf_counter() - t_start

        print("\n============================================================")
        print("STAGE 7 FINAL SUMMARY")
        print("============================================================")
        print(f"ML Dataset Rows: {total_rows}")
        print(f"Training Rows: {train_count}")
        print(f"Testing Rows: {test_count}")
        print("\nTarget: Severity")
        print("Number of Classes: 4 (Severity 1, 2, 3, 4)")
        print(f"\nFeatures Used: {len(candidate_features)} ({candidate_features})")
        print(f"Features Excluded: {len(forbidden_features)} ({forbidden_features})")
        print("\nClass Weighting: Balanced Class Weights w_c = N_train / (4 * N_train_c) on training partition")
        print("Train/Test Split: 80% Train, 20% Test (Fixed Random Seed 42)")
        print("\nLogistic Regression: Trained & Evaluated (outputs/models/logistic_regression)")
        print("Decision Tree: Trained & Evaluated (outputs/models/decision_tree)")
        print("Random Forest: Trained & Evaluated (outputs/models/random_forest)")
        print("\nModel Comparison Table: outputs/tables/model_comparison.csv")
        print("Per-Class Metrics: outputs/tables/per_class_metrics.csv")
        print("Confusion Matrices:")
        print("  - outputs/charts/confusion_matrix_logistic_regression.png")
        print("  - outputs/charts/confusion_matrix_decision_tree.png")
        print("  - outputs/charts/confusion_matrix_random_forest.png")
        print("Model Comparison Chart: outputs/charts/model_comparison_metrics.png")
        print(f"\nRaw CSV Unchanged: {csv_unchanged}")

        status = "PASS" if (train_count + test_count == EXPECTED_ROWS and csv_unchanged and len(comparison_results) == 3) else "FAIL"
        print(f"Status: {status}")

    except Exception as exc:
        print("\n=== ERROR DURING ML PIPELINE ===")
        print(f"Error Type: {type(exc).__name__}")
        print(f"Error Message: {exc}")
        print("Status: FAIL")
        raise
    finally:
        spark.stop()
        print("Spark Stopped: True")


if __name__ == "__main__":
    main()
