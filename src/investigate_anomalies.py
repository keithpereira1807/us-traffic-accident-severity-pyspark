"""
ANOMALY AND MISSINGNESS INVESTIGATION ONLY.

Does NOT clean, impute, cap, or modify data.
Does NOT collect the full dataset, cache it, write Parquet, or train models.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, max as smax, min as smin, sum as ssum, when

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingestion_smoke_test import build_explicit_schema  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "US_Accidents_March23.csv"


def pct(n: int, total: int) -> float:
    return (n / total * 100.0) if total else 0.0


def top_n(df, column: str, n: int = 5):
    return (
        df.groupBy(column)
        .agg(count(lit(1)).alias("cnt"))
        .orderBy(col("cnt").desc())
        .limit(n)
        .collect()
    )


def fmt_top(rows, value_col: str) -> str:
    if not rows:
        return "(none)"
    return ", ".join(f"{r[value_col]}={r['cnt']}" for r in rows)


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")

    schema = build_explicit_schema()
    spark = (
        SparkSession.builder.appName("us_accidents_anomaly_investigation")
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

    print("=== ANOMALY INVESTIGATION ===")
    print(f"input_file={RAW_CSV}")
    print(f"spark_version={spark.version}")
    print(f"spark_master={conf.get('spark.master')}")
    print(f"spark_driver_memory={conf.get('spark.driver.memory', '<not explicitly set>')}")

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

        ws = col("Wind_Speed(mph)")
        temp = col("Temperature(F)")
        pres = col("Pressure(in)")
        vis = col("Visibility(mi)")
        precip = col("Precipitation(in)")
        elat = col("End_Lat")
        elng = col("End_Lng")
        st = col("Start_Time")
        et = col("End_Time")

        stats = df.agg(
            count(lit(1)).alias("total_rows"),
            ssum(when(ws.isNull(), 1).otherwise(0)).alias("ws_null"),
            ssum(when(ws == 0, 1).otherwise(0)).alias("ws_zero"),
            ssum(when(ws > 100, 1).otherwise(0)).alias("ws_gt100"),
            ssum(when(ws > 200, 1).otherwise(0)).alias("ws_gt200"),
            ssum(when(ws > 500, 1).otherwise(0)).alias("ws_gt500"),
            ssum(when(ws > 1000, 1).otherwise(0)).alias("ws_gt1000"),
            smax(ws).alias("ws_max"),
            ssum(when(temp.isNull(), 1).otherwise(0)).alias("temp_null"),
            ssum(when(temp < -50, 1).otherwise(0)).alias("temp_lt_m50"),
            ssum(when(temp < -20, 1).otherwise(0)).alias("temp_lt_m20"),
            ssum(when(temp > 120, 1).otherwise(0)).alias("temp_gt120"),
            ssum(when(temp > 150, 1).otherwise(0)).alias("temp_gt150"),
            ssum(when(temp > 180, 1).otherwise(0)).alias("temp_gt180"),
            ssum(when(temp > 200, 1).otherwise(0)).alias("temp_gt200"),
            smin(temp).alias("temp_min"),
            smax(temp).alias("temp_max"),
            ssum(when(pres.isNull(), 1).otherwise(0)).alias("pres_null"),
            ssum(when(pres == 0, 1).otherwise(0)).alias("pres_zero"),
            ssum(when(pres < 20, 1).otherwise(0)).alias("pres_lt20"),
            ssum(when(pres > 35, 1).otherwise(0)).alias("pres_gt35"),
            ssum(when(pres > 40, 1).otherwise(0)).alias("pres_gt40"),
            ssum(when(pres > 50, 1).otherwise(0)).alias("pres_gt50"),
            smin(pres).alias("pres_min"),
            smax(pres).alias("pres_max"),
            ssum(when(vis.isNull(), 1).otherwise(0)).alias("vis_null"),
            ssum(when(vis == 0, 1).otherwise(0)).alias("vis_zero"),
            ssum(when(vis > 20, 1).otherwise(0)).alias("vis_gt20"),
            ssum(when(vis > 50, 1).otherwise(0)).alias("vis_gt50"),
            ssum(when(vis > 100, 1).otherwise(0)).alias("vis_gt100"),
            smin(vis).alias("vis_min"),
            smax(vis).alias("vis_max"),
            ssum(when(precip.isNull(), 1).otherwise(0)).alias("precip_null"),
            ssum(when(precip == 0, 1).otherwise(0)).alias("precip_zero"),
            ssum(when(precip > 1, 1).otherwise(0)).alias("precip_gt1"),
            ssum(when(precip > 5, 1).otherwise(0)).alias("precip_gt5"),
            ssum(when(precip > 10, 1).otherwise(0)).alias("precip_gt10"),
            ssum(when(precip > 20, 1).otherwise(0)).alias("precip_gt20"),
            ssum(when(precip > 30, 1).otherwise(0)).alias("precip_gt30"),
            smin(precip).alias("precip_min"),
            smax(precip).alias("precip_max"),
            ssum(when(elat.isNull() & elng.isNull(), 1).otherwise(0)).alias("end_both_null"),
            ssum(when(elat.isNull() & elng.isNotNull(), 1).otherwise(0)).alias("end_lat_only"),
            ssum(when(elat.isNotNull() & elng.isNull(), 1).otherwise(0)).alias("end_lng_only"),
            ssum(when(elat.isNotNull() & elng.isNotNull(), 1).otherwise(0)).alias("end_both_present"),
            ssum(when(st.isNull() & et.isNull(), 1).otherwise(0)).alias("ts_both_null"),
            ssum(when(st.isNull() & et.isNotNull(), 1).otherwise(0)).alias("ts_start_only"),
            ssum(when(st.isNotNull() & et.isNull(), 1).otherwise(0)).alias("ts_end_only"),
            ssum(when(st.isNotNull() & et.isNotNull(), 1).otherwise(0)).alias("ts_both_present"),
        ).collect()[0]

        total = int(stats["total_rows"])

        print("\nWIND SPEED INVESTIGATION")
        print(f"Null: {int(stats['ws_null'])}")
        print(f"Zero: {int(stats['ws_zero'])}")
        print(f">100: {int(stats['ws_gt100'])}")
        print(f">200: {int(stats['ws_gt200'])}")
        print(f">500: {int(stats['ws_gt500'])}")
        print(f">1000: {int(stats['ws_gt1000'])}")
        print(f"Maximum: {stats['ws_max']}")

        print("\nTOP 10 DISTINCT HIGHEST Wind_Speed(mph) VALUES")
        print("Wind_Speed(mph) | Count")
        top_ws = (
            df.filter(ws.isNotNull())
            .groupBy("Wind_Speed(mph)")
            .agg(count(lit(1)).alias("cnt"))
            .orderBy(col("Wind_Speed(mph)").desc())
            .limit(10)
            .collect()
        )
        for r in top_ws:
            print(f"{r['Wind_Speed(mph)']} | {r['cnt']}")

        print("\nTEMPERATURE INVESTIGATION")
        print(f"Null: {int(stats['temp_null'])}")
        print(f"< -50: {int(stats['temp_lt_m50'])}")
        print(f"< -20: {int(stats['temp_lt_m20'])}")
        print(f">120: {int(stats['temp_gt120'])}")
        print(f">150: {int(stats['temp_gt150'])}")
        print(f">180: {int(stats['temp_gt180'])}")
        print(f">200: {int(stats['temp_gt200'])}")
        print(f"Minimum: {stats['temp_min']}")
        print(f"Maximum: {stats['temp_max']}")

        print("\nPRESSURE INVESTIGATION")
        print(f"Null: {int(stats['pres_null'])}")
        print(f"Zero: {int(stats['pres_zero'])}")
        print(f"<20: {int(stats['pres_lt20'])}")
        print(f">35: {int(stats['pres_gt35'])}")
        print(f">40: {int(stats['pres_gt40'])}")
        print(f">50: {int(stats['pres_gt50'])}")
        print(f"Minimum: {stats['pres_min']}")
        print(f"Maximum: {stats['pres_max']}")

        print("\nVISIBILITY INVESTIGATION")
        print(f"Null: {int(stats['vis_null'])}")
        print(f"Zero: {int(stats['vis_zero'])}")
        print(f">20: {int(stats['vis_gt20'])}")
        print(f">50: {int(stats['vis_gt50'])}")
        print(f">100: {int(stats['vis_gt100'])}")
        print(f"Minimum: {stats['vis_min']}")
        print(f"Maximum: {stats['vis_max']}")

        print("\nPRECIPITATION INVESTIGATION")
        print(f"Null: {int(stats['precip_null'])}")
        print(f"Zero: {int(stats['precip_zero'])}")
        print(f">1: {int(stats['precip_gt1'])}")
        print(f">5: {int(stats['precip_gt5'])}")
        print(f">10: {int(stats['precip_gt10'])}")
        print(f">20: {int(stats['precip_gt20'])}")
        print(f">30: {int(stats['precip_gt30'])}")
        print(f"Minimum: {stats['precip_min']}")
        print(f"Maximum: {stats['precip_max']}")

        print("\nEND COORDINATE MISSINGNESS")
        print(
            f"Both null: {int(stats['end_both_null'])} "
            f"({pct(int(stats['end_both_null']), total):.6f}%)"
        )
        print(
            f"Only End_Lat null: {int(stats['end_lat_only'])} "
            f"({pct(int(stats['end_lat_only']), total):.6f}%)"
        )
        print(
            f"Only End_Lng null: {int(stats['end_lng_only'])} "
            f"({pct(int(stats['end_lng_only']), total):.6f}%)"
        )
        print(
            f"Both present: {int(stats['end_both_present'])} "
            f"({pct(int(stats['end_both_present']), total):.6f}%)"
        )

        print("\nTIMESTAMP MISSINGNESS")
        print(
            f"Both null: {int(stats['ts_both_null'])} "
            f"({pct(int(stats['ts_both_null']), total):.6f}%)"
        )
        print(
            f"Only Start_Time null: {int(stats['ts_start_only'])} "
            f"({pct(int(stats['ts_start_only']), total):.6f}%)"
        )
        print(
            f"Only End_Time null: {int(stats['ts_end_only'])} "
            f"({pct(int(stats['ts_end_only']), total):.6f}%)"
        )
        print(
            f"Both present: {int(stats['ts_both_present'])} "
            f"({pct(int(stats['ts_both_present']), total):.6f}%)"
        )

        print("\n=== 9. EXTREME VALUE CONTEXT ===")
        ws200 = df.filter(ws > 200)
        ws200_count = int(stats["ws_gt200"])
        print("\nWind_Speed(mph) > 200")
        print(f"count={ws200_count}")
        print(f"top_5_Weather_Condition: {fmt_top(top_n(ws200, 'Weather_Condition'), 'Weather_Condition')}")
        print(f"top_5_State: {fmt_top(top_n(ws200, 'State'), 'State')}")

        temp150 = df.filter(temp > 150)
        temp150_count = int(stats["temp_gt150"])
        print("\nTemperature(F) > 150")
        print(f"count={temp150_count}")
        print(f"top_5_Weather_Condition: {fmt_top(top_n(temp150, 'Weather_Condition'), 'Weather_Condition')}")
        print(f"top_5_State: {fmt_top(top_n(temp150, 'State'), 'State')}")

        vis50 = df.filter(vis > 50)
        vis50_count = int(stats["vis_gt50"])
        print("\nVisibility(mi) > 50")
        print(f"count={vis50_count}")
        print(f"top_5_Weather_Condition: {fmt_top(top_n(vis50, 'Weather_Condition'), 'Weather_Condition')}")
        print(f"top_5_State: {fmt_top(top_n(vis50, 'State'), 'State')}")

        elapsed_s = time.perf_counter() - t0

        print("\n==================================================")
        print("ANOMALY INVESTIGATION SUMMARY")
        print("==================================================")
        print(
            "Wind Speed Findings: "
            f"null={int(stats['ws_null'])}, zero={int(stats['ws_zero'])}, "
            f">100={int(stats['ws_gt100'])}, >200={int(stats['ws_gt200'])}, "
            f">500={int(stats['ws_gt500'])}, >1000={int(stats['ws_gt1000'])}, "
            f"max={stats['ws_max']}"
        )
        print(
            "Temperature Findings: "
            f"null={int(stats['temp_null'])}, <-50={int(stats['temp_lt_m50'])}, "
            f"<-20={int(stats['temp_lt_m20'])}, >120={int(stats['temp_gt120'])}, "
            f">150={int(stats['temp_gt150'])}, >180={int(stats['temp_gt180'])}, "
            f">200={int(stats['temp_gt200'])}, min={stats['temp_min']}, max={stats['temp_max']}"
        )
        print(
            "Pressure Findings: "
            f"null={int(stats['pres_null'])}, zero={int(stats['pres_zero'])}, "
            f"<20={int(stats['pres_lt20'])}, >35={int(stats['pres_gt35'])}, "
            f">40={int(stats['pres_gt40'])}, >50={int(stats['pres_gt50'])}, "
            f"min={stats['pres_min']}, max={stats['pres_max']}"
        )
        print(
            "Visibility Findings: "
            f"null={int(stats['vis_null'])}, zero={int(stats['vis_zero'])}, "
            f">20={int(stats['vis_gt20'])}, >50={int(stats['vis_gt50'])}, "
            f">100={int(stats['vis_gt100'])}, min={stats['vis_min']}, max={stats['vis_max']}"
        )
        print(
            "Precipitation Findings: "
            f"null={int(stats['precip_null'])}, zero={int(stats['precip_zero'])}, "
            f">1={int(stats['precip_gt1'])}, >5={int(stats['precip_gt5'])}, "
            f">10={int(stats['precip_gt10'])}, >20={int(stats['precip_gt20'])}, "
            f">30={int(stats['precip_gt30'])}, min={stats['precip_min']}, max={stats['precip_max']}"
        )
        print(
            "End Coordinate Missingness: "
            f"both_null={int(stats['end_both_null'])} ({pct(int(stats['end_both_null']), total):.4f}%), "
            f"only_End_Lat_null={int(stats['end_lat_only'])}, "
            f"only_End_Lng_null={int(stats['end_lng_only'])}, "
            f"both_present={int(stats['end_both_present'])}"
        )
        print(
            "Timestamp Missingness: "
            f"both_null={int(stats['ts_both_null'])} ({pct(int(stats['ts_both_null']), total):.4f}%), "
            f"only_Start_Time_null={int(stats['ts_start_only'])}, "
            f"only_End_Time_null={int(stats['ts_end_only'])}, "
            f"both_present={int(stats['ts_both_present'])}"
        )
        print(
            "Key Extreme-Value Context: "
            f"wind>200 count={ws200_count}; temp>150 count={temp150_count}; "
            f"visibility>50 count={vis50_count}"
        )
        print("Cleaning Decisions Made: NONE")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print("Status: PASS")

    except Exception as exc:
        elapsed_s = time.perf_counter() - t0
        print("\n=== ERROR / ABORT ===")
        print(f"exception_type={type(exc).__name__}")
        print(f"exception_message={exc}")
        print(f"Elapsed Time: {elapsed_s:.2f} seconds")
        print("Cleaning Decisions Made: NONE")
        print("Status: FAIL")
        raise
    finally:
        spark.stop()
        print("spark_stopped=True")


if __name__ == "__main__":
    main()
