"""
DATASET QUALITY / CLASS DISTRIBUTION PROFILE ONLY.

Full-CSV Spark aggregations. Does NOT:
  - use Pandas
  - collect() the full dataset
  - cache/persist the full dataset
  - write Parquet, charts, models
  - modify the raw CSV
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, lit, max as smax, min as smin, sum as ssum, when

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingestion_smoke_test import build_explicit_schema  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "US_Accidents_March23.csv"
EXPECTED_ROWS = 7_728_394

IMPORTANT_COLS = [
    "ID",
    "Severity",
    "Start_Time",
    "End_Time",
    "Start_Lat",
    "Start_Lng",
    "End_Lat",
    "End_Lng",
    "Distance(mi)",
    "City",
    "State",
    "Temperature(F)",
    "Wind_Chill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",
    "Sunrise_Sunset",
    "Traffic_Signal",
    "Junction",
    "Crossing",
]

BLANK_STRING_COLS = ["City", "State", "Weather_Condition", "Sunrise_Sunset"]

NUMERIC_RANGE_COLS = [
    "Start_Lat",
    "Start_Lng",
    "End_Lat",
    "End_Lng",
    "Distance(mi)",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
]


def safe_alias(prefix: str, name: str) -> str:
    cleaned = (
        name.replace("(", "_")
        .replace(")", "_")
        .replace("%", "pct")
        .replace(" ", "_")
    )
    return f"{prefix}_{cleaned}"


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    schema = build_explicit_schema()
    if len(schema.fields) != 46:
        raise RuntimeError(f"Expected 46 schema fields, got {len(schema.fields)}")

    spark = (
        SparkSession.builder.appName("us_accidents_dataset_profile")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.maxResultSize", "64m")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    conf = spark.sparkContext.getConf()

    print("=== DATASET PROFILE ===")
    print(f"input_file={RAW_CSV}")
    print(f"spark_version={spark.version}")
    print(f"spark_master={conf.get('spark.master')}")
    print(f"spark_driver_memory={conf.get('spark.driver.memory', '<not explicitly set>')}")

    status = "FAIL"
    t0 = time.perf_counter()

    try:
        df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
            .schema(schema)
            .load(str(RAW_CSV))
        )

        print("\n=== 3. VERIFY ROW COUNT ===")
        total_rows = df.count()
        print(f"total_rows={total_rows}")
        if total_rows != EXPECTED_ROWS:
            print(
                f"DISCREPANCY: expected {EXPECTED_ROWS:,} parsed rows from Stage 3B, "
                f"got {total_rows:,} (difference={total_rows - EXPECTED_ROWS})"
            )
        else:
            print("row_count_matches_stage_3b=True")

        print("\n=== 4. SEVERITY DISTRIBUTION ===")
        sev_rows = (
            df.groupBy("Severity")
            .agg(count(lit(1)).alias("cnt"))
            .orderBy(col("Severity").asc_nulls_last())
            .collect()
        )
        print("SEVERITY DISTRIBUTION")
        print("Severity | Count | Percentage")
        for r in sev_rows:
            sev = r["Severity"]
            cnt = r["cnt"]
            pct = (cnt / total_rows * 100.0) if total_rows else 0.0
            print(f"{sev} | {cnt} | {pct:.6f}")
        distinct_severity = len(sev_rows)
        sev_vals = [r["Severity"] for r in sev_rows if r["Severity"] is not None]
        min_sev = min(sev_vals) if sev_vals else None
        max_sev = max(sev_vals) if sev_vals else None
        print(f"distinct_Severity_values={distinct_severity}")
        print(f"minimum_Severity={min_sev}")
        print(f"maximum_Severity={max_sev}")
        print(f"Severity_values_present={[r['Severity'] for r in sev_rows]}")

        print("\n=== 5. NULL / MISSING VALUE PROFILE ===")
        agg_exprs = []
        for c in IMPORTANT_COLS:
            agg_exprs.append(
                ssum(when(col(c).isNull(), 1).otherwise(0)).alias(safe_alias("null", c))
            )
        for c in BLANK_STRING_COLS:
            agg_exprs.append(
                ssum(when(col(c) == "", 1).otherwise(0)).alias(safe_alias("blank", c))
            )
        agg_exprs.extend(
            [
                smin("Start_Time").alias("min_start"),
                smax("Start_Time").alias("max_start"),
                smin("End_Time").alias("min_end"),
                smax("End_Time").alias("max_end"),
                ssum(
                    when(col("End_Time") < col("Start_Time"), 1).otherwise(0)
                ).alias("end_before_start"),
            ]
        )
        for c in NUMERIC_RANGE_COLS:
            agg_exprs.append(smin(c).alias(safe_alias("min", c)))
            agg_exprs.append(smax(c).alias(safe_alias("max", c)))

        stats = df.agg(*agg_exprs).collect()[0]

        null_pairs = []
        for c in IMPORTANT_COLS:
            nnull = int(stats[safe_alias("null", c)])
            pct = (nnull / total_rows * 100.0) if total_rows else 0.0
            null_pairs.append((c, nnull, pct))
        null_pairs.sort(key=lambda x: x[1], reverse=True)

        print("IMPORTANT COLUMN NULL COUNTS")
        print("Column | Null Count | Null Percentage")
        for c, nnull, pct in null_pairs:
            print(f"{c} | {nnull} | {pct:.6f}")

        zero_null_cols = sum(1 for _, n, _ in null_pairs if n == 0)
        some_null_cols = sum(1 for _, n, _ in null_pairs if n > 0)
        print(f"total_rows={total_rows}")
        print(f"important_columns_with_zero_nulls={zero_null_cols}")
        print(f"important_columns_with_at_least_one_null={some_null_cols}")

        print("\nBLANK STRING COUNTS (string columns only; empty string '', not SQL null)")
        print("Column | Blank Count | Blank Percentage")
        for c in BLANK_STRING_COLS:
            nblank = int(stats[safe_alias("blank", c)])
            pct = (nblank / total_rows * 100.0) if total_rows else 0.0
            print(f"{c} | {nblank} | {pct:.6f}")

        print("\n=== 6. DATE/TIME RANGE ===")
        print("DATE/TIME RANGE")
        print(f"Start_Time min: {stats['min_start']}")
        print(f"Start_Time max: {stats['max_start']}")
        print(f"End_Time min: {stats['min_end']}")
        print(f"End_Time max: {stats['max_end']}")
        end_before_start = int(stats["end_before_start"])
        print(f"records_end_before_start={end_before_start}")

        print("\n=== 7. STATE COVERAGE ===")
        state_counts = df.groupBy("State").agg(count(lit(1)).alias("cnt"))
        # Small grouped result only (~dozens of rows), not the full dataset.
        state_counts.cache()
        distinct_states = state_counts.count()
        top10 = state_counts.orderBy(col("cnt").desc()).limit(10).collect()
        state_counts.unpersist()

        print("STATE COVERAGE")
        print(f"distinct_states={distinct_states}")
        print("TOP 10 STATES BY ACCIDENT COUNT")
        print("Rank | State | Count | Percentage")
        for i, r in enumerate(top10, 1):
            pct = (r["cnt"] / total_rows * 100.0) if total_rows else 0.0
            print(f"{i} | {r['State']} | {r['cnt']} | {pct:.6f}")

        print("\n=== 8. BASIC NUMERICAL SANITY CHECK ===")
        print("NUMERICAL RANGE CHECKS")
        print("Column | Min | Max")
        for c in NUMERIC_RANGE_COLS:
            print(f"{c} | {stats[safe_alias('min', c)]} | {stats[safe_alias('max', c)]}")

        print("\n=== 9. DUPLICATE ID CHECK ===")
        distinct_ids = df.agg(countDistinct("ID").alias("n")).collect()[0]["n"]
        duplicate_id_count = total_rows - int(distinct_ids)
        print("ID UNIQUENESS")
        print(f"total_rows={total_rows}")
        print(f"distinct_ids={distinct_ids}")
        print(f"duplicate_id_count={duplicate_id_count}")

        elapsed_s = time.perf_counter() - t0
        status = "PASS"

        sev_dist_text = ", ".join(
            f"{r['Severity']}={r['cnt']}" for r in sev_rows
        )
        top_nulls = ", ".join(
            f"{c}={n}" for c, n, _ in null_pairs if n > 0
        ) or "none"

        print("\n==================================================")
        print("DATASET PROFILE SUMMARY")
        print("==================================================")
        print(f"Total Rows: {total_rows}")
        print(f"Number of Columns: {len(df.columns)}")
        print(f"Severity Classes: {distinct_severity} (min={min_sev}, max={max_sev})")
        print(f"Severity Distribution: {sev_dist_text}")
        print(f"Important Null Findings: {top_nulls}")
        print(f"Start Time Range: {stats['min_start']} to {stats['max_start']}")
        print(f"End Time Range: {stats['min_end']} to {stats['max_end']}")
        print(f"Records With End_Time < Start_Time: {end_before_start}")
        print(f"Distinct States: {distinct_states}")
        print(f"Duplicate ID Count: {duplicate_id_count}")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print(f"Status: {status}")

    except Exception as exc:
        elapsed_s = time.perf_counter() - t0
        print("\n=== ERROR / ABORT ===")
        print(f"exception_type={type(exc).__name__}")
        print(f"exception_message={exc}")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print("Status: FAIL")
        raise
    finally:
        spark.stop()
        print("spark_stopped=True")


if __name__ == "__main__":
    main()
