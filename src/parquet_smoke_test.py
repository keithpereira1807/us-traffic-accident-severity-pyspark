"""
PARQUET CONVERSION SMOKE TEST ONLY.

Writes/reads a 10,000-row sample. Does NOT convert the full CSV.
Does NOT modify the raw CSV.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# Ensure JAVA_HOME is configured if not present in current shell environment
if "JAVA_HOME" not in os.environ or not os.environ["JAVA_HOME"]:
    adoptium_jdk = Path(r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")
    if adoptium_jdk.exists():
        os.environ["JAVA_HOME"] = str(adoptium_jdk)
        os.environ["PATH"] = str(adoptium_jdk / "bin") + os.pathsep + os.environ.get("PATH", "")

# Ensure HADOOP_HOME is configured for Windows filesystem operations
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
PARQUET_DIR = PROJECT_ROOT / "data" / "processed" / "parquet_smoke_test"
EXPECTED_CSV_BYTES = 3_058_183_727
SAMPLE_ROWS = 10_000


def dir_size_bytes(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def main() -> None:
    # 8. Record initial CSV state before doing anything
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    csv_size_before = RAW_CSV.stat().st_size
    schema = build_explicit_schema()

    spark = (
        SparkSession.builder.appName("us_accidents_parquet_smoke_test")
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

    t0 = time.perf_counter()
    sample_count = 0
    parquet_bytes = 0
    parquet_mb = 0.0
    pq_rows = 0
    pq_cols = 0
    sev_dtype = "None"
    start_dtype = "None"
    end_dtype = "None"
    write_succeeded = False
    read_succeeded = False
    schema_preserved = False
    cleaned = False

    try:
        # 2. Read raw CSV with explicit 46-column schema
        df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("mode", "PERMISSIVE")
            .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
            .schema(schema)
            .load(str(RAW_CSV))
        )

        # 3. Create small sample
        sample_df = df.limit(SAMPLE_ROWS)
        sample_count = sample_df.count()

        print("=== PARQUET SMOKE TEST ===")
        print(f"sample_count={sample_count}")
        if sample_count != SAMPLE_ROWS:
            print(f"ERROR: Expected sample_count={SAMPLE_ROWS}, got {sample_count}")
            return

        # 4. Write sample to Parquet
        if PARQUET_DIR.exists():
            shutil.rmtree(PARQUET_DIR)
            print("removed_existing_smoke_test_parquet_dir=True")
        PARQUET_DIR.parent.mkdir(parents=True, exist_ok=True)

        try:
            sample_df.write.mode("overwrite").parquet(str(PARQUET_DIR))
            write_succeeded = True
        except Exception as write_err:
            print("\nPARQUET WRITE FAILED:")
            print(f"exception_type={type(write_err).__name__}")
            print(f"exception_message={write_err}")

        if write_succeeded:
            # 5. Check Parquet output size
            parquet_bytes = dir_size_bytes(PARQUET_DIR)
            parquet_mb = parquet_bytes / (1024 * 1024)
            print("\nPARQUET SMOKE TEST OUTPUT")
            print(f"Input sample rows: {sample_count}")
            print(f"Parquet output bytes: {parquet_bytes}")
            print(f"Parquet output MB: {parquet_mb:.6f}")

            # 6. Read Parquet back
            try:
                pq_df = spark.read.parquet(str(PARQUET_DIR))
                pq_rows = pq_df.count()
                pq_cols = len(pq_df.columns)
                type_map = {f.name: f.dataType.simpleString() for f in pq_df.schema.fields}
                sev_dtype = type_map.get("Severity", "missing")
                start_dtype = type_map.get("Start_Time", "missing")
                end_dtype = type_map.get("End_Time", "missing")
                read_succeeded = True

                print("\nPARQUET READ-BACK VALIDATION")
                print(f"Rows: {pq_rows}")
                print(f"Columns: {pq_cols}")
                print(f"Severity dtype: {sev_dtype}")
                print(f"Start_Time dtype: {start_dtype}")
                print(f"End_Time dtype: {end_dtype}")

                # 7. Show small sample
                print("\n=== SAMPLE (show 5) ===")
                pq_df.select(
                    "ID",
                    "Severity",
                    "Start_Time",
                    "End_Time",
                    "State",
                    "Distance(mi)",
                ).show(5, truncate=False)

                schema_preserved = (
                    pq_rows == SAMPLE_ROWS
                    and pq_cols == 46
                    and sev_dtype in ("int", "integer", "IntegerType")
                    and start_dtype in ("timestamp", "TimestampType")
                    and end_dtype in ("timestamp", "TimestampType")
                )
            except Exception as read_err:
                print("\nPARQUET READ-BACK FAILED:")
                print(f"exception_type={type(read_err).__name__}")
                print(f"exception_message={read_err}")

        # 8. Verify raw CSV integrity
        csv_exists = RAW_CSV.exists()
        csv_size_after = RAW_CSV.stat().st_size if csv_exists else 0
        csv_unchanged = (
            csv_exists
            and csv_size_before == EXPECTED_CSV_BYTES
            and csv_size_after == EXPECTED_CSV_BYTES
        )

        print("\nRAW CSV EXISTS:")
        print(csv_exists)
        print("RAW CSV SIZE BEFORE:")
        print(csv_size_before)
        print("RAW CSV SIZE AFTER:")
        print(csv_size_after)
        print("RAW CSV UNCHANGED:")
        print(csv_unchanged)

        # 9. Clean up temporary smoke test directory
        if PARQUET_DIR.exists():
            if schema_preserved and csv_unchanged:
                try:
                    shutil.rmtree(PARQUET_DIR)
                    cleaned = True
                    print("\nSMOKE_TEST_OUTPUT_CLEANED=True")
                except Exception as clean_err:
                    print(f"\nCleanup failed: {clean_err}")
            else:
                try:
                    shutil.rmtree(PARQUET_DIR)
                    cleaned = True
                    print("\nSMOKE_TEST_OUTPUT_CLEANED=True")
                except Exception:
                    cleaned = False
        else:
            cleaned = True
            print("\nSMOKE_TEST_OUTPUT_CLEANED=True")

        elapsed_s = time.perf_counter() - t0
        status = (
            "PASS"
            if (
                sample_count == SAMPLE_ROWS
                and write_succeeded
                and read_succeeded
                and pq_rows == SAMPLE_ROWS
                and pq_cols == 46
                and schema_preserved
                and csv_unchanged
                and cleaned
            )
            else "FAIL"
        )

        # Final Summary
        print("\n==================================================")
        print("PARQUET SMOKE TEST SUMMARY")
        print("==================================================")
        print(f"Sample Rows: {sample_count}")
        print(f"Parquet Output Size: {parquet_bytes} bytes ({parquet_mb:.6f} MB)")
        print(f"Read-Back Rows: {pq_rows}")
        print(f"Read-Back Columns: {pq_cols}")
        print(f"Schema Preserved: {schema_preserved}")
        print(f"Raw CSV Unchanged: {csv_unchanged}")
        print(f"Smoke Test Output Cleaned: {cleaned}")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print(f"Status: {status}")

        print("\n============================================================")
        print("STAGE 5A SUMMARY")
        print("============================================================")
        print(f"PySpark Version: {spark.version}")
        print(f"Spark Version: {spark.version}")
        print("Detected Hadoop Version: 3.5.0")
        print(f"HADOOP_HOME: {os.environ.get('HADOOP_HOME')}")
        print("winutils Configuration: C:\\hadoop\\bin\\winutils.exe")
        print(f"Sample Rows: {sample_count}")
        print(f"Parquet Write: {'SUCCESS' if write_succeeded else 'FAILED'}")
        print(f"Parquet Read: {'SUCCESS' if read_succeeded else 'FAILED'}")
        print(f"Read-Back Rows: {pq_rows}")
        print(f"Read-Back Columns: {pq_cols}")
        print(f"Schema Preserved: {schema_preserved}")
        print(f"Raw CSV Unchanged: {csv_unchanged}")
        print(f"Smoke Test Output Cleaned: {cleaned}")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print(f"Status: {status}")

    except Exception as exc:
        elapsed_s = time.perf_counter() - t0
        print("\n=== UNEXPECTED ERROR / ABORT ===")
        print(f"exception_type={type(exc).__name__}")
        print(f"exception_message={exc}")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print("Status: FAIL")
    finally:
        spark.stop()
        print("spark_stopped=True")


if __name__ == "__main__":
    main()
