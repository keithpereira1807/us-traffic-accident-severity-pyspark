"""
FULL DATASET PARQUET CONVERSION

Converts raw US_Accidents_March23.csv to Parquet using the explicit 46-column schema.
Preserves raw CSV untouched.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# Ensure JAVA_HOME and HADOOP_HOME are set
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

from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingestion_smoke_test import build_explicit_schema  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "US_Accidents_March23.csv"
PARQUET_DIR = PROJECT_ROOT / "data" / "processed" / "us_accidents_parquet"
EXPECTED_ROWS = 7_728_394
EXPECTED_CSV_BYTES = 3_058_183_727


def dir_size_bytes(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    csv_size_before = RAW_CSV.stat().st_size
    if csv_size_before != EXPECTED_CSV_BYTES:
        raise ValueError(f"Raw CSV size mismatch: expected {EXPECTED_CSV_BYTES}, got {csv_size_before}")

    schema = build_explicit_schema()

    spark = (
        SparkSession.builder.appName("us_accidents_full_parquet_conversion")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.maxResultSize", "64m")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.hadoop.hadoop.home.dir", os.environ.get("HADOOP_HOME", r"C:\hadoop"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("=== FULL PARQUET PREPARATION ===")
    print(f"Raw CSV: {RAW_CSV}")
    print(f"Raw CSV Size: {csv_size_before} bytes")
    print(f"Expected Rows: {EXPECTED_ROWS}")
    print(f"Spark Version: {spark.version}")
    print("Hadoop Version: 3.5.0")
    print(f"Output Path: {PARQUET_DIR}")

    t0 = time.perf_counter()

    try:
        # Check if already exists and valid
        if PARQUET_DIR.exists():
            try:
                existing_df = spark.read.parquet(str(PARQUET_DIR))
                existing_count = existing_df.count()
                existing_cols = len(existing_df.columns)
                if existing_count == EXPECTED_ROWS and existing_cols == 46:
                    print(f"Valid full Parquet dataset already exists ({existing_count} rows, {existing_cols} cols). Reusing.")
                    print(f"Parquet Row Count: {existing_count}")
                    print(f"Parquet Column Count: {existing_cols}")
                    csv_size_after = RAW_CSV.stat().st_size
                    print(f"Raw CSV Unchanged: {RAW_CSV.exists() and csv_size_after == EXPECTED_CSV_BYTES}")
                    return
            except Exception as e:
                print(f"Existing directory invalid ({e}). Recreating...")
                shutil.rmtree(PARQUET_DIR)

        print("\nReading raw CSV with explicit 46-column schema...")
        df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
            .schema(schema)
            .load(str(RAW_CSV))
        )

        print(f"Writing full dataset to Parquet: {PARQUET_DIR}...")
        PARQUET_DIR.parent.mkdir(parents=True, exist_ok=True)
        df.write.mode("overwrite").parquet(str(PARQUET_DIR))

        elapsed_write = time.perf_counter() - t0
        parquet_bytes = dir_size_bytes(PARQUET_DIR)
        parquet_mb = parquet_bytes / (1024 * 1024)
        print(f"Parquet write finished in {elapsed_write:.2f} s ({parquet_bytes} bytes / {parquet_mb:.2f} MB)")

        print("\nVerifying written Parquet dataset...")
        pq_df = spark.read.parquet(str(PARQUET_DIR))
        pq_rows = pq_df.count()
        pq_cols = len(pq_df.columns)

        csv_size_after = RAW_CSV.stat().st_size
        csv_unchanged = RAW_CSV.exists() and csv_size_before == EXPECTED_CSV_BYTES and csv_size_after == EXPECTED_CSV_BYTES

        print(f"Parquet Row Count: {pq_rows}")
        print(f"Parquet Column Count: {pq_cols}")
        print(f"Raw CSV Unchanged: {csv_unchanged}")

        if pq_rows != EXPECTED_ROWS:
            raise RuntimeError(f"Parquet row count mismatch: expected {EXPECTED_ROWS}, got {pq_rows}")
        if pq_cols != 46:
            raise RuntimeError(f"Parquet column count mismatch: expected 46, got {pq_cols}")
        if not csv_unchanged:
            raise RuntimeError(f"Raw CSV was modified!")

        print("\n=== PARQUET CONVERSION VERIFIED SUCCESSFULLY ===")

    finally:
        spark.stop()
        print("spark_stopped=True")


if __name__ == "__main__":
    main()
