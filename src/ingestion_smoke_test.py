"""
INGESTION SMOKE TEST ONLY — NOT full-dataset ingestion.

Purpose:
  Prove PySpark can read a SMALL sample of US_Accidents_March23.csv
  using an EXPLICIT schema (inferSchema is NOT used).

Hard limits for this script:
  - Does NOT process the full ~2.85 GB / ~7.7M-row file in Spark
  - Does NOT write a full Parquet dataset
  - Does NOT modify/move/delete the raw CSV
  - Uses modest local Spark memory settings (~7.37 GB machine RAM)
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "US_Accidents_March23.csv"
SAMPLE_CSV = PROJECT_ROOT / "data" / "processed" / "ingestion_smoke_sample.csv"
SAMPLE_DATA_ROWS = 100  # header + 100 data rows only


def build_explicit_schema() -> StructType:
    """Explicit schema for all 46 verified header columns (no inference)."""
    return StructType(
        [
            StructField("ID", StringType(), True),
            StructField("Source", StringType(), True),
            StructField("Severity", IntegerType(), True),
            StructField("Start_Time", TimestampType(), True),
            StructField("End_Time", TimestampType(), True),
            StructField("Start_Lat", DoubleType(), True),
            StructField("Start_Lng", DoubleType(), True),
            StructField("End_Lat", DoubleType(), True),
            StructField("End_Lng", DoubleType(), True),
            StructField("Distance(mi)", DoubleType(), True),
            StructField("Description", StringType(), True),
            StructField("Street", StringType(), True),
            StructField("City", StringType(), True),
            StructField("County", StringType(), True),
            StructField("State", StringType(), True),
            StructField("Zipcode", StringType(), True),
            StructField("Country", StringType(), True),
            StructField("Timezone", StringType(), True),
            StructField("Airport_Code", StringType(), True),
            StructField("Weather_Timestamp", TimestampType(), True),
            StructField("Temperature(F)", DoubleType(), True),
            StructField("Wind_Chill(F)", DoubleType(), True),
            StructField("Humidity(%)", DoubleType(), True),
            StructField("Pressure(in)", DoubleType(), True),
            StructField("Visibility(mi)", DoubleType(), True),
            StructField("Wind_Direction", StringType(), True),
            StructField("Wind_Speed(mph)", DoubleType(), True),
            StructField("Precipitation(in)", DoubleType(), True),
            StructField("Weather_Condition", StringType(), True),
            StructField("Amenity", BooleanType(), True),
            StructField("Bump", BooleanType(), True),
            StructField("Crossing", BooleanType(), True),
            StructField("Give_Way", BooleanType(), True),
            StructField("Junction", BooleanType(), True),
            StructField("No_Exit", BooleanType(), True),
            StructField("Railway", BooleanType(), True),
            StructField("Roundabout", BooleanType(), True),
            StructField("Station", BooleanType(), True),
            StructField("Stop", BooleanType(), True),
            StructField("Traffic_Calming", BooleanType(), True),
            StructField("Traffic_Signal", BooleanType(), True),
            StructField("Turning_Loop", BooleanType(), True),
            StructField("Sunrise_Sunset", StringType(), True),
            StructField("Civil_Twilight", StringType(), True),
            StructField("Nautical_Twilight", StringType(), True),
            StructField("Astronomical_Twilight", StringType(), True),
        ]
    )


def write_small_sample(src: Path, dst: Path, data_rows: int) -> int:
    """Copy header + first N data lines only (streaming; does not load full CSV)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8", newline="") as fin, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        for i, line in enumerate(fin):
            fout.write(line)
            written += 1
            # line 0 = header; stop after header + data_rows
            if i >= data_rows:
                break
    return written


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    print("=== INGESTION SMOKE TEST (small sample only) ===")
    print(f"raw_csv={RAW_CSV}")
    print(f"raw_size_bytes={RAW_CSV.stat().st_size}")
    print(f"sample_csv={SAMPLE_CSV}")
    print(f"sample_data_rows_requested={SAMPLE_DATA_ROWS}")

    n_lines = write_small_sample(RAW_CSV, SAMPLE_CSV, SAMPLE_DATA_ROWS)
    print(f"sample_lines_written_including_header={n_lines}")
    print(f"sample_size_bytes={SAMPLE_CSV.stat().st_size}")

    schema = build_explicit_schema()
    assert len(schema.fields) == 46, f"Expected 46 fields, got {len(schema.fields)}"

    spark = (
        SparkSession.builder.appName("us_accidents_ingestion_smoke_test")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    conf = spark.sparkContext.getConf()
    print("\n=== 11. SPARK CONFIG (memory / local) ===")
    print(f"spark.version={spark.version}")
    print(f"spark.master={conf.get('spark.master')}")
    print(f"spark.driver.memory={conf.get('spark.driver.memory', '<not explicitly set>')}")
    print(
        f"spark.executor.memory={conf.get('spark.executor.memory', '<not explicitly set / local mode>')}"
    )

    print("\n=== 1. SPARK VERSION ===")
    print(spark.version)

    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .schema(schema)  # EXPLICIT — inferSchema NOT used
        .load(str(SAMPLE_CSV))
    )

    print("\n=== 2. DATAFRAME SCHEMA ===")
    df.printSchema()

    print("\n=== 3. COLUMN COUNT ===")
    print(len(df.columns))

    print("\n=== 4. COLUMN LIST ===")
    print(df.columns)

    focus_cols = [
        "ID",
        "Severity",
        "Start_Time",
        "End_Time",
        "Start_Lat",
        "Start_Lng",
        "State",
        "Distance(mi)",
        "Temperature(F)",
        "Weather_Condition",
    ]

    print("\n=== 5. SAMPLE ROWS (focus columns, limit 5) ===")
    df.select(*focus_cols).show(5, truncate=False)

    print("\n=== 6. SELECTED SPARK DATA TYPES ===")
    type_map = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    for c in [
        "Severity",
        "Start_Time",
        "End_Time",
        "Start_Lat",
        "Start_Lng",
        "Distance(mi)",
        "Temperature(F)",
    ]:
        print(f"{c}: {type_map[c]}")

    print("\n=== 7. SMALL VALIDATION ===")
    row_count = df.count()
    sev_distinct = df.select("Severity").distinct().orderBy("Severity")
    sev_vals = [r["Severity"] for r in sev_distinct.collect()]
    sev_n = len(sev_vals)
    print(f"smoke_test_row_count={row_count}")
    print(f"distinct_Severity_count={sev_n}")
    print(f"Severity_values_observed={sev_vals}")

    print("\n=== 8. TIMESTAMP PARSING CHECK ===")
    df.select("ID", "Start_Time", "End_Time").show(5, truncate=False)
    ts_ok = type_map["Start_Time"] == "timestamp" and type_map["End_Time"] == "timestamp"
    null_start = df.filter(F.col("Start_Time").isNull()).count()
    null_end = df.filter(F.col("End_Time").isNull()).count()
    print(f"Start_Time_nulls_in_sample={null_start}")
    print(f"End_Time_nulls_in_sample={null_end}")
    print(f"timestamp_types_ok={ts_ok}")

    print("\n=== 9. NUMERIC WEATHER FIELD CHECK (Temperature(F)) ===")
    print(f"Temperature(F)_dtype={type_map['Temperature(F)']}")
    df.select("ID", "Temperature(F)").show(5, truncate=False)
    temp_non_null = df.filter(F.col("Temperature(F)").isNotNull()).count()
    print(f"Temperature(F)_non_null_rows={temp_non_null}")
    numeric_ok = type_map["Temperature(F)"] == "double" and temp_non_null > 0

    print("\n=== 10. BOOLEAN / INFRASTRUCTURE FIELD CHECK (Traffic_Signal) ===")
    print(f"Traffic_Signal_dtype={type_map['Traffic_Signal']}")
    df.select("ID", "Traffic_Signal", "Amenity", "Crossing").show(5, truncate=False)
    bool_vals = [
        r["Traffic_Signal"]
        for r in df.select("Traffic_Signal").distinct().collect()
    ]
    print(f"Traffic_Signal_distinct_values={bool_vals}")
    boolean_ok = type_map["Traffic_Signal"] == "boolean" and any(
        v is True for v in bool_vals
    ) and any(v is False for v in bool_vals)

    # Final summary (exact labels requested)
    if (
        row_count > 0
        and len(df.columns) == 46
        and type_map["Severity"] == "int"
        and ts_ok
        and null_start == 0
        and null_end == 0
        and numeric_ok
        and boolean_ok
    ):
        status = "PASS"
    else:
        status = "FAIL"

    print("\nINGESTION SMOKE TEST SUMMARY")
    print(f"Spark Version: {spark.version}")
    print(f"Test Rows: {row_count}")
    print(f"Column Count: {len(df.columns)}")
    print(f"Severity Data Type: {type_map['Severity']}")
    print(f"Start_Time Data Type: {type_map['Start_Time']}")
    print(f"End_Time Data Type: {type_map['End_Time']}")
    print(
        f"Numeric Parsing: {'OK' if numeric_ok else 'FAIL'} "
        f"(Temperature(F) as {type_map['Temperature(F)']}, non-null={temp_non_null})"
    )
    print(
        f"Boolean Parsing: {'OK' if boolean_ok else 'FAIL'} "
        f"(Traffic_Signal as {type_map['Traffic_Signal']}, values={bool_vals})"
    )
    print(
        f"Timestamp Parsing: {'OK' if ts_ok and null_start == 0 and null_end == 0 else 'FAIL'} "
        f"(null Start_Time={null_start}, null End_Time={null_end})"
    )
    print(f"Smoke Test Status: {status}")

    spark.stop()


if __name__ == "__main__":
    main()
