"""
FULL DATASET ROW-COUNT VALIDATION ONLY.

Performs a single Spark scan + df.count() on US_Accidents_March23.csv
using the same explicit 46-column schema from the ingestion smoke test.

Does NOT:
  - cache/persist
  - collect() the full dataset
  - convert to Pandas
  - write Parquet / outputs
  - clean, transform, analyze, or train models
  - modify/move/copy/delete the raw CSV
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession

# Allow importing sibling module when run as: python src/count_full_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingestion_smoke_test import build_explicit_schema  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "US_Accidents_March23.csv"

# From prior memory-safe newline scan (approximate structural data rows)
LINE_BASED_APPROX_DATA_ROWS = 7_728_394


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    file_size = RAW_CSV.stat().st_size
    schema = build_explicit_schema()
    if len(schema.fields) != 46:
        raise RuntimeError(f"Expected 46 schema fields, got {len(schema.fields)}")

    print("=== FULL DATASET COUNT (single scan, no cache) ===")
    print(f"input_file={RAW_CSV}")
    print(f"file_size_bytes={file_size}")
    print(f"line_based_approximate_data_rows={LINE_BASED_APPROX_DATA_ROWS}")
    print("inferSchema=False (explicit schema reused from smoke test)")

    spark = (
        SparkSession.builder.appName("us_accidents_full_row_count")
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
    spark_version = spark.version
    spark_master = conf.get("spark.master")
    driver_memory = conf.get("spark.driver.memory", "<not explicitly set>")

    print("\n=== SPARK SESSION ===")
    print(f"spark_version={spark_version}")
    print(f"spark_master={spark_master}")
    print(f"spark_driver_memory={driver_memory}")

    status = "UNKNOWN"
    parsed_count = None
    elapsed_s = None
    col_count = None
    sev_dtype = None
    start_dtype = None
    end_dtype = None

    try:
        df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
            .schema(schema)  # EXPLICIT — inferSchema NOT used
            .load(str(RAW_CSV))
        )

        # Lightweight schema validation only (no full-data ops beyond metadata)
        type_map = {f.name: f.dataType.simpleString() for f in df.schema.fields}
        col_count = len(df.columns)
        sev_dtype = type_map["Severity"]
        start_dtype = type_map["Start_Time"]
        end_dtype = type_map["End_Time"]

        print("\n=== LIGHTWEIGHT SCHEMA VALIDATION ===")
        print(f"number_of_columns={col_count}")
        print(f"Severity_dtype={sev_dtype}")
        print(f"Start_Time_dtype={start_dtype}")
        print(f"End_Time_dtype={end_dtype}")

        print("\n=== df.count() START (full scan; no cache/persist/collect) ===")
        t0 = time.perf_counter()
        parsed_count = df.count()
        elapsed_s = time.perf_counter() - t0
        print("=== df.count() COMPLETE ===")
        print(f"spark_parsed_row_count={parsed_count}")
        print(f"elapsed_seconds={elapsed_s:.2f}")

        difference = parsed_count - LINE_BASED_APPROX_DATA_ROWS
        print("\n=== COMPARISON ===")
        print(f"Spark parsed row count: {parsed_count}")
        print(f"Line-based approximate data-row count: {LINE_BASED_APPROX_DATA_ROWS}")
        print(f"Difference: {difference}")
        if difference == 0:
            print(
                "Difference explanation: ZERO — Spark parsed record count matches "
                "the prior line-based approximate data-row count exactly."
            )
        else:
            print(
                "Difference explanation: NON-ZERO — counts differ. Possible causes include "
                "embedded newlines inside quoted CSV fields, malformed rows dropped/added "
                "under PERMISSIVE mode, or header/line-boundary edge cases. "
                "The Spark count is the authoritative parsed record count for this schema."
            )

        schema_ok = (
            col_count == 46
            and sev_dtype == "int"
            and start_dtype == "timestamp"
            and end_dtype == "timestamp"
        )
        status = "PASS" if schema_ok and parsed_count is not None else "FAIL"

        print("\nFULL DATASET COUNT SUMMARY")
        print(f"Input File: {RAW_CSV}")
        print(f"File Size: {file_size} bytes")
        print(f"Spark Version: {spark_version}")
        print(f"Spark Master: {spark_master}")
        print(f"Driver Memory: {driver_memory}")
        print(f"Spark Parsed Row Count: {parsed_count}")
        print(f"Line-Based Approximate Row Count: {LINE_BASED_APPROX_DATA_ROWS}")
        print(f"Difference: {difference}")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print(
            f"Schema Validation: columns={col_count}, "
            f"Severity={sev_dtype}, Start_Time={start_dtype}, End_Time={end_dtype}"
        )
        print(f"Status: {status}")

    except Exception as exc:
        status = "FAIL"
        print("\n=== ERROR / ABORT ===")
        print(f"exception_type={type(exc).__name__}")
        print(f"exception_message={exc}")
        print(
            "Job stopped safely after failure. Not retrying automatically "
            "(resource-safety policy)."
        )
        print("\nFULL DATASET COUNT SUMMARY")
        print(f"Input File: {RAW_CSV}")
        print(f"File Size: {file_size} bytes")
        print(f"Spark Version: {spark_version}")
        print(f"Spark Master: {spark_master}")
        print(f"Driver Memory: {driver_memory}")
        print(f"Spark Parsed Row Count: {parsed_count}")
        print(f"Line-Based Approximate Row Count: {LINE_BASED_APPROX_DATA_ROWS}")
        print("Difference: N/A")
        print(f"Elapsed Time: {elapsed_s}")
        print(
            f"Schema Validation: columns={col_count}, "
            f"Severity={sev_dtype}, Start_Time={start_dtype}, End_Time={end_dtype}"
        )
        print(f"Status: {status}")
        raise
    finally:
        spark.stop()
        print("spark_stopped=True")


if __name__ == "__main__":
    main()
