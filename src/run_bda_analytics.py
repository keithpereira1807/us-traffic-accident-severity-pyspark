"""
STAGE 6 — BIG DATA ANALYTICS, VISUALIZATIONS, AND SQL ANALYTICS
US Traffic Accident Severity Analytics Using PySpark

Processes 7,728,394 records from verified Parquet storage without collecting
the full dataset to driver or loading full dataset into Pandas.
Generates publication-quality charts and CSV summary tables.
"""

from __future__ import annotations

import csv
import os
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
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "US_Accidents_March23.csv"
PARQUET_DIR = PROJECT_ROOT / "data" / "processed" / "us_accidents_parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
DOCS_DIR = PROJECT_ROOT / "docs"

EXPECTED_ROWS = 7_728_394
EXPECTED_CSV_BYTES = 3_058_183_727


def save_df_to_csv(df_or_rows, col_names, output_path: Path):
    """Save small aggregated rows or Pandas dataframe to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_rows, pd.DataFrame):
        df_or_rows.to_csv(output_path, index=False)
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            for row in df_or_rows:
                writer.writerow([row[c] if hasattr(row, "__getitem__") else getattr(row, c) for c in col_names])
    print(f"Saved table: {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    t_start = time.perf_counter()

    # Step 13: Verify raw CSV before analytics
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")
    csv_size_before = RAW_CSV.stat().st_size
    print(f"RAW CSV SIZE BEFORE: {csv_size_before} bytes")
    if csv_size_before != EXPECTED_CSV_BYTES:
        raise ValueError(f"Raw CSV size mismatch: expected {EXPECTED_CSV_BYTES}, got {csv_size_before}")

    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f"Processed Parquet dataset not found at {PARQUET_DIR}. Run convert_to_parquet.py first.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("us_accidents_bda_analytics")
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

    generated_tables = []
    generated_charts = []

    try:
        print("\nLoading full dataset from Parquet...")
        df = spark.read.parquet(str(PARQUET_DIR))
        total_rows = df.count()
        total_cols = len(df.columns)
        print(f"Loaded Parquet: {total_rows} rows, {total_cols} columns")
        if total_rows != EXPECTED_ROWS:
            raise RuntimeError(f"Row count mismatch: expected {EXPECTED_ROWS}, got {total_rows}")

        # Register temp view for Spark SQL
        df.createOrReplaceTempView("accidents")

        # =====================================================================
        # PART 3 — DATASET OVERVIEW
        # =====================================================================
        print("\n=== PART 3: DATASET OVERVIEW ===")
        overview_row = df.select(
            F.count("ID").alias("total_rows"),
            F.countDistinct("ID").alias("distinct_ids"),
            F.min("Start_Time").alias("min_start_time"),
            F.max("Start_Time").alias("max_start_time"),
            F.countDistinct("State").alias("num_states"),
            F.countDistinct("City").alias("num_cities"),
            F.countDistinct("County").alias("num_counties"),
        ).collect()[0]

        overview_data = [
            ("Total Rows", overview_row["total_rows"]),
            ("Total Columns", total_cols),
            ("Distinct Accident IDs", overview_row["distinct_ids"]),
            ("Minimum Start Time", str(overview_row["min_start_time"])),
            ("Maximum Start Time", str(overview_row["max_start_time"])),
            ("Number of States", overview_row["num_states"]),
            ("Number of Cities", overview_row["num_cities"]),
            ("Number of Counties", overview_row["num_counties"]),
        ]
        overview_df = pd.DataFrame(overview_data, columns=["Metric", "Value"])
        overview_path = TABLES_DIR / "dataset_overview.csv"
        overview_df.to_csv(overview_path, index=False)
        generated_tables.append(overview_path.name)
        print(f"Saved: {overview_path}")
        for k, v in overview_data:
            print(f"  {k}: {v}")

        if overview_row["total_rows"] != overview_row["distinct_ids"]:
            raise RuntimeError(f"Unique ID check failed: {overview_row['total_rows']} vs {overview_row['distinct_ids']}")

        # =====================================================================
        # PART 4 — SEVERITY ANALYSIS
        # =====================================================================
        print("\n=== PART 4: SEVERITY ANALYSIS ===")
        sev_agg = (
            df.groupBy("Severity")
            .agg(F.count(F.lit(1)).alias("Count"))
            .orderBy("Severity")
            .collect()
        )
        sev_data = []
        sev_sum = sum(r["Count"] for r in sev_agg)
        for r in sev_agg:
            sev = r["Severity"]
            cnt = r["Count"]
            pct = (cnt / sev_sum) * 100.0
            sev_data.append({"Severity": sev, "Count": cnt, "Percentage": round(pct, 2)})
            print(f"  Severity {sev}: {cnt:,} ({pct:.2f}%)")

        if sev_sum != EXPECTED_ROWS:
            raise RuntimeError(f"Severity sum {sev_sum} != {EXPECTED_ROWS}")

        sev_df = pd.DataFrame(sev_data)
        sev_path = TABLES_DIR / "severity_distribution.csv"
        sev_df.to_csv(sev_path, index=False)
        generated_tables.append(sev_path.name)

        # Chart: Severity Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([str(s) for s in sev_df["Severity"]], sev_df["Count"], color="#2b5c8f", edgecolor="black", width=0.55)
        ax.set_title("US Traffic Accident Severity Distribution (2016-2023)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Severity Level (1 = Least Severe, 4 = Most Severe)", fontsize=11)
        ax.set_ylabel("Number of Accidents", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        for bar in bars:
            yval = bar.get_height()
            pct_val = (yval / sev_sum) * 100
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 50000, f"{yval:,}\n({pct_val:.1f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(sev_df["Count"]) * 1.15)
        plt.tight_layout()
        chart_sev_path = CHARTS_DIR / "severity_distribution.png"
        plt.savefig(chart_sev_path, dpi=150)
        plt.close()
        generated_charts.append(chart_sev_path.name)
        print(f"Saved chart: {chart_sev_path}")

        # Severity by State
        sev_state_spark = (
            df.groupBy("State", "Severity")
            .agg(F.count(F.lit(1)).alias("Count"))
            .orderBy("State", "Severity")
        )
        sev_state_rows = sev_state_spark.collect()
        sev_state_df = pd.DataFrame([{"State": r["State"], "Severity": r["Severity"], "Count": r["Count"]} for r in sev_state_rows])
        sev_state_pivot = sev_state_df.pivot(index="State", columns="Severity", values="Count").fillna(0).astype(int)
        sev_state_pivot["Total"] = sev_state_pivot.sum(axis=1)
        sev_state_pivot = sev_state_pivot.sort_values(by="Total", ascending=False).reset_index()
        sev_state_path = TABLES_DIR / "severity_by_state.csv"
        sev_state_pivot.to_csv(sev_state_path, index=False)
        generated_tables.append(sev_state_path.name)

        # Chart: Severity by State (Top 15 states stacked)
        top15_state_sev = sev_state_pivot.head(15).copy()
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = pd.Series([0] * len(top15_state_sev))
        colors = {1: "#72b7b2", 2: "#4e79a7", 3: "#f28e2b", 4: "#e15759"}
        for s in [1, 2, 3, 4]:
            if s in top15_state_sev.columns:
                ax.bar(top15_state_sev["State"], top15_state_sev[s], bottom=bottom, label=f"Severity {s}", color=colors[s], edgecolor="white", linewidth=0.5)
                bottom += top15_state_sev[s]
        ax.set_title("Accident Severity Composition in Top 15 States by Accident Volume", fontsize=13, fontweight="bold")
        ax.set_xlabel("State", fontsize=11)
        ax.set_ylabel("Total Number of Accidents", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.legend(title="Severity")
        plt.tight_layout()
        chart_sev_state_path = CHARTS_DIR / "severity_by_state.png"
        plt.savefig(chart_sev_state_path, dpi=150)
        plt.close()
        generated_charts.append(chart_sev_state_path.name)
        print(f"Saved chart: {chart_sev_state_path}")

        # =====================================================================
        # PART 5 — TEMPORAL ANALYSIS
        # =====================================================================
        print("\n=== PART 5: TEMPORAL ANALYSIS ===")
        # Derive time fields
        df_time = (
            df.withColumn("Year", F.year("Start_Time"))
            .withColumn("Month", F.month("Start_Time"))
            .withColumn("DayOfWeek", F.dayofweek("Start_Time"))  # 1=Sunday, 7=Saturday
            .withColumn("Hour", F.hour("Start_Time"))
        )

        # 1. Accidents by Year
        year_rows = df_time.groupBy("Year").agg(F.count(F.lit(1)).alias("Count")).orderBy("Year").collect()
        year_df = pd.DataFrame([{"Year": r["Year"], "Count": r["Count"]} for r in year_rows if r["Year"] is not None])
        year_df["Percentage"] = (year_df["Count"] / total_rows * 100).round(2)
        year_path = TABLES_DIR / "accidents_by_year.csv"
        year_df.to_csv(year_path, index=False)
        generated_tables.append(year_path.name)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(year_df["Year"].astype(str), year_df["Count"], color="#3b6978", edgecolor="black")
        ax.set_title("Recorded Accidents by Year (2016 - 2023)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Number of Accidents", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.tight_layout()
        chart_year_path = CHARTS_DIR / "accidents_by_year.png"
        plt.savefig(chart_year_path, dpi=150)
        plt.close()
        generated_charts.append(chart_year_path.name)

        # 2. Accidents by Month
        month_rows = df_time.groupBy("Month").agg(F.count(F.lit(1)).alias("Count")).orderBy("Month").collect()
        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        month_df = pd.DataFrame([{"Month": r["Month"], "Month_Name": month_names.get(r["Month"], str(r["Month"])), "Count": r["Count"]} for r in month_rows if r["Month"] is not None])
        month_df["Percentage"] = (month_df["Count"] / total_rows * 100).round(2)
        month_path = TABLES_DIR / "accidents_by_month.csv"
        month_df.to_csv(month_path, index=False)
        generated_tables.append(month_path.name)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(month_df["Month_Name"], month_df["Count"], marker="o", color="#d95f02", linewidth=2.5, markersize=6)
        ax.set_title("Aggregated Accidents by Month of Year (2016-2023)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("Number of Accidents", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        chart_month_path = CHARTS_DIR / "accidents_by_month.png"
        plt.savefig(chart_month_path, dpi=150)
        plt.close()
        generated_charts.append(chart_month_path.name)

        # 3. Accidents by Day of Week
        dow_rows = df_time.groupBy("DayOfWeek").agg(F.count(F.lit(1)).alias("Count")).orderBy("DayOfWeek").collect()
        dow_names = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
        dow_df = pd.DataFrame([{"DayOfWeek": r["DayOfWeek"], "Day_Name": dow_names.get(r["DayOfWeek"], str(r["DayOfWeek"])), "Count": r["Count"]} for r in dow_rows if r["DayOfWeek"] is not None])
        dow_df["Percentage"] = (dow_df["Count"] / total_rows * 100).round(2)
        dow_path = TABLES_DIR / "accidents_by_day_of_week.csv"
        dow_df.to_csv(dow_path, index=False)
        generated_tables.append(dow_path.name)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(dow_df["Day_Name"], dow_df["Count"], color="#7570b3", edgecolor="black")
        ax.set_title("Accidents by Day of the Week", fontsize=13, fontweight="bold")
        ax.set_xlabel("Day of Week", fontsize=11)
        ax.set_ylabel("Number of Accidents", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.tight_layout()
        chart_dow_path = CHARTS_DIR / "accidents_by_day_of_week.png"
        plt.savefig(chart_dow_path, dpi=150)
        plt.close()
        generated_charts.append(chart_dow_path.name)

        # 4. Accidents by Hour
        hour_rows = df_time.groupBy("Hour").agg(F.count(F.lit(1)).alias("Count")).orderBy("Hour").collect()
        hour_df = pd.DataFrame([{"Hour": r["Hour"], "Count": r["Count"]} for r in hour_rows if r["Hour"] is not None])
        hour_df["Percentage"] = (hour_df["Count"] / total_rows * 100).round(2)
        hour_path = TABLES_DIR / "accidents_by_hour.csv"
        hour_df.to_csv(hour_path, index=False)
        generated_tables.append(hour_path.name)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(hour_df["Hour"], hour_df["Count"], marker="s", color="#1b9e77", linewidth=2)
        ax.set_title("Accidents by Hour of Day (0-23 Local Time)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour (24-Hour Format)", fontsize=11)
        ax.set_ylabel("Number of Accidents", fontsize=11)
        ax.set_xticks(range(0, 24))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        chart_hour_path = CHARTS_DIR / "accidents_by_hour.png"
        plt.savefig(chart_hour_path, dpi=150)
        plt.close()
        generated_charts.append(chart_hour_path.name)

        # Severity distribution across years
        year_sev_rows = df_time.groupBy("Year", "Severity").agg(F.count(F.lit(1)).alias("Count")).orderBy("Year", "Severity").collect()
        year_sev_df = pd.DataFrame([{"Year": r["Year"], "Severity": r["Severity"], "Count": r["Count"]} for r in year_sev_rows if r["Year"] is not None])
        year_sev_pivot = year_sev_df.pivot(index="Year", columns="Severity", values="Count").fillna(0).astype(int)
        fig, ax = plt.subplots(figsize=(10, 5))
        year_sev_pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="black", linewidth=0.5)
        ax.set_title("Yearly Accident Counts by Severity Level (2016 - 2023)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Accident Count", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.legend(title="Severity")
        plt.xticks(rotation=0)
        plt.tight_layout()
        chart_year_sev_path = CHARTS_DIR / "accidents_by_year_severity.png"
        plt.savefig(chart_year_sev_path, dpi=150)
        plt.close()
        generated_charts.append(chart_year_sev_path.name)

        # =====================================================================
        # PART 6 — GEOGRAPHIC ANALYSIS
        # =====================================================================
        print("\n=== PART 6: GEOGRAPHIC ANALYSIS ===")
        # Top 10 States
        state_rows = df.groupBy("State").agg(F.count(F.lit(1)).alias("Count")).orderBy(F.desc("Count")).limit(10).collect()
        top_states_df = pd.DataFrame([{"State": r["State"], "Count": r["Count"]} for r in state_rows])
        top_states_df["Percentage_of_Total"] = (top_states_df["Count"] / total_rows * 100).round(2)
        top_states_path = TABLES_DIR / "top_10_states.csv"
        top_states_df.to_csv(top_states_path, index=False)
        generated_tables.append(top_states_path.name)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(top_states_df["State"], top_states_df["Count"], color="#2c7fb8", edgecolor="black")
        ax.set_title("Top 10 US States by Recorded Accident Count (Note: Denominators not adjusted)", fontsize=12, fontweight="bold")
        ax.set_xlabel("State", fontsize=11)
        ax.set_ylabel("Accidents", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        for idx, row in top_states_df.iterrows():
            ax.text(idx, row["Count"] + 30000, f"{row['Percentage_of_Total']}%", ha="center", fontsize=9)
        plt.tight_layout()
        chart_states_path = CHARTS_DIR / "top_10_states.png"
        plt.savefig(chart_states_path, dpi=150)
        plt.close()
        generated_charts.append(chart_states_path.name)

        # Top 20 Cities
        city_rows = df.groupBy("City", "State").agg(F.count(F.lit(1)).alias("Count")).orderBy(F.desc("Count")).limit(20).collect()
        top_cities_df = pd.DataFrame([{"City": r["City"], "State": r["State"], "Count": r["Count"]} for r in city_rows])
        top_cities_df["Percentage_of_Total"] = (top_cities_df["Count"] / total_rows * 100).round(2)
        top_cities_path = TABLES_DIR / "top_20_cities.csv"
        top_cities_df.to_csv(top_cities_path, index=False)
        generated_tables.append(top_cities_path.name)

        fig, ax = plt.subplots(figsize=(10, 7))
        city_labels = [f"{r['City']} ({r['State']})" for _, r in top_cities_df.iterrows()]
        ax.barh(city_labels[::-1], top_cities_df["Count"][::-1], color="#41b6c4", edgecolor="black")
        ax.set_title("Top 20 US Cities by Recorded Accidents", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Accidents", fontsize=11)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.tight_layout()
        chart_cities_path = CHARTS_DIR / "top_20_cities.png"
        plt.savefig(chart_cities_path, dpi=150)
        plt.close()
        generated_charts.append(chart_cities_path.name)

        # Top 20 Counties
        county_rows = df.groupBy("County", "State").agg(F.count(F.lit(1)).alias("Count")).orderBy(F.desc("Count")).limit(20).collect()
        top_counties_df = pd.DataFrame([{"County": r["County"], "State": r["State"], "Count": r["Count"]} for r in county_rows])
        top_counties_df["Percentage_of_Total"] = (top_counties_df["Count"] / total_rows * 100).round(2)
        top_counties_path = TABLES_DIR / "top_20_counties.csv"
        top_counties_df.to_csv(top_counties_path, index=False)
        generated_tables.append(top_counties_path.name)

        # =====================================================================
        # PART 7 — WEATHER ANALYSIS
        # =====================================================================
        print("\n=== PART 7: WEATHER ANALYSIS ===")
        # Top 15 Weather Conditions
        weather_rows = (
            df.groupBy("Weather_Condition")
            .agg(F.count(F.lit(1)).alias("Count"))
            .filter(F.col("Weather_Condition").isNotNull())
            .orderBy(F.desc("Count"))
            .limit(15)
            .collect()
        )
        weather_df = pd.DataFrame([{"Weather_Condition": r["Weather_Condition"], "Count": r["Count"]} for r in weather_rows])
        weather_df["Percentage_of_Total"] = (weather_df["Count"] / total_rows * 100).round(2)
        weather_path = TABLES_DIR / "top_weather_conditions.csv"
        weather_df.to_csv(weather_path, index=False)
        generated_tables.append(weather_path.name)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(weather_df["Weather_Condition"][::-1], weather_df["Count"][::-1], color="#6baed6", edgecolor="black")
        ax.set_title("Top 15 Weather Conditions Associated with Recorded Accidents", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Accidents", fontsize=11)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.tight_layout()
        chart_weather_path = CHARTS_DIR / "top_weather_conditions.png"
        plt.savefig(chart_weather_path, dpi=150)
        plt.close()
        generated_charts.append(chart_weather_path.name)

        # Severity by Major Weather Condition (Top 10 conditions)
        top10_weather_list = [r["Weather_Condition"] for r in weather_rows[:10]]
        w_sev_rows = (
            df.filter(F.col("Weather_Condition").isin(top10_weather_list))
            .groupBy("Weather_Condition", "Severity")
            .agg(F.count(F.lit(1)).alias("Count"))
            .collect()
        )
        w_sev_df = pd.DataFrame([{"Weather_Condition": r["Weather_Condition"], "Severity": r["Severity"], "Count": r["Count"]} for r in w_sev_rows])
        w_sev_pivot = w_sev_df.pivot(index="Weather_Condition", columns="Severity", values="Count").fillna(0).astype(int)
        w_sev_pivot["Total"] = w_sev_pivot.sum(axis=1)
        w_sev_pivot = w_sev_pivot.sort_values(by="Total", ascending=False).reset_index()
        w_sev_path = TABLES_DIR / "weather_severity_summary.csv"
        w_sev_pivot.to_csv(w_sev_path, index=False)
        generated_tables.append(w_sev_path.name)

        # Chart: Severity percentage within major weather conditions
        w_sev_pct = w_sev_pivot.set_index("Weather_Condition")[[1, 2, 3, 4]]
        w_sev_pct = w_sev_pct.div(w_sev_pct.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(11, 5.5))
        w_sev_pct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="black", linewidth=0.5)
        ax.set_title("Severity Proportion Across Top 10 Weather Conditions", fontsize=13, fontweight="bold")
        ax.set_xlabel("Weather Condition", fontsize=11)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        chart_w_sev_path = CHARTS_DIR / "weather_severity_summary.png"
        plt.savefig(chart_w_sev_path, dpi=150)
        plt.close()
        generated_charts.append(chart_w_sev_path.name)

        # Weather Numeric Variables Summary
        weather_numeric_cols = [
            "Temperature(F)",
            "Humidity(%)",
            "Pressure(in)",
            "Visibility(mi)",
            "Wind_Speed(mph)",
            "Precipitation(in)",
        ]
        numeric_summary_rows = []
        for col_name in weather_numeric_cols:
            agg_res = df.select(
                F.count(F.col(col_name)).alias("count"),
                F.mean(F.col(col_name)).alias("mean"),
                F.min(F.col(col_name)).alias("min"),
                F.max(F.col(col_name)).alias("max"),
                F.expr(f"percentile_approx(`{col_name}`, 0.5)").alias("median"),
            ).collect()[0]
            numeric_summary_rows.append({
                "Variable": col_name,
                "Count": agg_res["count"],
                "Missing": total_rows - agg_res["count"],
                "Missing_Pct": round((total_rows - agg_res["count"]) / total_rows * 100, 2),
                "Mean": round(agg_res["mean"], 2) if agg_res["mean"] is not None else None,
                "Median": round(float(agg_res["median"]), 2) if agg_res["median"] is not None else None,
                "Min": round(agg_res["min"], 2) if agg_res["min"] is not None else None,
                "Max": round(agg_res["max"], 2) if agg_res["max"] is not None else None,
            })
        w_num_df = pd.DataFrame(numeric_summary_rows)
        w_num_path = TABLES_DIR / "weather_numeric_summary.csv"
        w_num_df.to_csv(w_num_path, index=False)
        generated_tables.append(w_num_path.name)

        # =====================================================================
        # PART 8 — ROAD / INFRASTRUCTURE ANALYSIS
        # =====================================================================
        print("\n=== PART 8: ROAD / INFRASTRUCTURE ANALYSIS ===")
        infra_cols = [
            "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
            "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
            "Traffic_Signal", "Turning_Loop",
        ]
        infra_rows = []
        for c in infra_cols:
            res = df.select(
                F.count(F.when(F.col(c) == True, 1)).alias("true_cnt"),
                F.count(F.when(F.col(c) == False, 1)).alias("false_cnt"),
            ).collect()[0]
            t_cnt = res["true_cnt"]
            f_cnt = res["false_cnt"]
            infra_rows.append({
                "Infrastructure_Feature": c,
                "True_Count": t_cnt,
                "False_Count": f_cnt,
                "True_Percentage": round((t_cnt / total_rows) * 100.0, 3),
            })
        infra_df = pd.DataFrame(infra_rows).sort_values(by="True_Percentage", ascending=False).reset_index(drop=True)
        infra_path = TABLES_DIR / "infrastructure_features.csv"
        infra_df.to_csv(infra_path, index=False)
        generated_tables.append(infra_path.name)

        # Chart: Infrastructure Features True Percentage
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(infra_df["Infrastructure_Feature"], infra_df["True_Percentage"], color="#3182bd", edgecolor="black")
        ax.set_title("Prevalence of Infrastructure Indicators in Recorded Accidents", fontsize=13, fontweight="bold")
        ax.set_xlabel("Infrastructure Feature", fontsize=11)
        ax.set_ylabel("Presence Percentage (%)", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        for idx, row in infra_df.iterrows():
            if row["True_Percentage"] > 0.05:
                ax.text(idx, row["True_Percentage"] + 0.3, f"{row['True_Percentage']:.1f}%", ha="center", fontsize=8)
        plt.tight_layout()
        chart_infra_path = CHARTS_DIR / "infrastructure_features.png"
        plt.savefig(chart_infra_path, dpi=150)
        plt.close()
        generated_charts.append(chart_infra_path.name)

        # Severity distribution for selected features (Junction, Crossing, Traffic_Signal, Railway, Stop)
        focus_features = ["Junction", "Crossing", "Traffic_Signal", "Railway", "Stop"]
        focus_rows = []
        for feat in focus_features:
            res = (
                df.filter(F.col(feat) == True)
                .groupBy("Severity")
                .agg(F.count(F.lit(1)).alias("Count"))
                .collect()
            )
            feat_tot = sum(r["Count"] for r in res)
            for r in res:
                s = r["Severity"]
                c = r["Count"]
                focus_rows.append({
                    "Feature": feat,
                    "Severity": s,
                    "Count": c,
                    "Feature_Total": feat_tot,
                    "Severity_Pct": round(c / feat_tot * 100, 2),
                })
        infra_sev_df = pd.DataFrame(focus_rows)
        infra_sev_path = TABLES_DIR / "infrastructure_severity_summary.csv"
        infra_sev_df.to_csv(infra_sev_path, index=False)
        generated_tables.append(infra_sev_path.name)

        # =====================================================================
        # PART 9 — DISTANCE ANALYSIS
        # =====================================================================
        print("\n=== PART 9: DISTANCE ANALYSIS ===")
        dist_res = df.select(
            F.count(F.col("Distance(mi)")).alias("count"),
            F.mean(F.col("Distance(mi)")).alias("mean"),
            F.min(F.col("Distance(mi)")).alias("min"),
            F.max(F.col("Distance(mi)")).alias("max"),
            F.expr("percentile_approx(`Distance(mi)`, 0.5)").alias("median"),
        ).collect()[0]

        dist_sum_df = pd.DataFrame([{
            "Metric": "Distance(mi)",
            "Count": dist_res["count"],
            "Mean": round(dist_res["mean"], 4),
            "Median": round(float(dist_res["median"]), 4),
            "Min": round(dist_res["min"], 4),
            "Max": round(dist_res["max"], 4),
        }])
        dist_sum_path = TABLES_DIR / "distance_summary.csv"
        dist_sum_df.to_csv(dist_sum_path, index=False)
        generated_tables.append(dist_sum_path.name)

        # Distance Buckets: 0, 0-0.1, 0.1-0.5, 0.5-1, 1-5, 5-10, 10-25, 25+
        dist_bucket_expr = (
            F.when(F.col("Distance(mi)") == 0.0, "0")
            .when((F.col("Distance(mi)") > 0.0) & (F.col("Distance(mi)") <= 0.1), "0 - 0.1")
            .when((F.col("Distance(mi)") > 0.1) & (F.col("Distance(mi)") <= 0.5), "0.1 - 0.5")
            .when((F.col("Distance(mi)") > 0.5) & (F.col("Distance(mi)") <= 1.0), "0.5 - 1.0")
            .when((F.col("Distance(mi)") > 1.0) & (F.col("Distance(mi)") <= 5.0), "1.0 - 5.0")
            .when((F.col("Distance(mi)") > 5.0) & (F.col("Distance(mi)") <= 10.0), "5.0 - 10.0")
            .when((F.col("Distance(mi)") > 10.0) & (F.col("Distance(mi)") <= 25.0), "10.0 - 25.0")
            .when(F.col("Distance(mi)") > 25.0, "25+")
            .otherwise("Unknown")
        )
        dist_bucket_rows = (
            df.withColumn("Distance_Bucket", dist_bucket_expr)
            .groupBy("Distance_Bucket")
            .agg(F.count(F.lit(1)).alias("Count"))
            .collect()
        )
        bucket_order = ["0", "0 - 0.1", "0.1 - 0.5", "0.5 - 1.0", "1.0 - 5.0", "5.0 - 10.0", "10.0 - 25.0", "25+"]
        bucket_dict = {r["Distance_Bucket"]: r["Count"] for r in dist_bucket_rows}
        bucket_rows_ordered = []
        for b in bucket_order:
            cnt = bucket_dict.get(b, 0)
            bucket_rows_ordered.append({"Distance_Bucket": b, "Count": cnt, "Percentage": round(cnt / total_rows * 100, 2)})
        dist_bucket_df = pd.DataFrame(bucket_rows_ordered)
        dist_bucket_path = TABLES_DIR / "distance_buckets.csv"
        dist_bucket_df.to_csv(dist_bucket_path, index=False)
        generated_tables.append(dist_bucket_path.name)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(dist_bucket_df["Distance_Bucket"], dist_bucket_df["Count"], color="#e7298a", edgecolor="black")
        ax.set_title("Accident Length Extent Distribution (Distance in Miles)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Distance Bucket (miles)", fontsize=11)
        ax.set_ylabel("Number of Accidents", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.tight_layout()
        chart_dist_path = CHARTS_DIR / "distance_buckets.png"
        plt.savefig(chart_dist_path, dpi=150)
        plt.close()
        generated_charts.append(chart_dist_path.name)

        # Distance by Severity
        dist_sev_res = (
            df.groupBy("Severity")
            .agg(
                F.count(F.col("Distance(mi)")).alias("Count"),
                F.mean(F.col("Distance(mi)")).alias("Mean_Distance"),
                F.expr("percentile_approx(`Distance(mi)`, 0.5)").alias("Median_Distance"),
                F.min(F.col("Distance(mi)")).alias("Min_Distance"),
                F.max(F.col("Distance(mi)")).alias("Max_Distance"),
            )
            .orderBy("Severity")
            .collect()
        )
        dist_sev_df = pd.DataFrame([{
            "Severity": r["Severity"],
            "Count": r["Count"],
            "Mean_Distance": round(r["Mean_Distance"], 4),
            "Median_Distance": round(float(r["Median_Distance"]), 4),
            "Min_Distance": round(r["Min_Distance"], 4),
            "Max_Distance": round(r["Max_Distance"], 4),
        } for r in dist_sev_res])
        dist_sev_path = TABLES_DIR / "distance_by_severity.csv"
        dist_sev_df.to_csv(dist_sev_path, index=False)
        generated_tables.append(dist_sev_path.name)

        # =====================================================================
        # PART 10 — SPARK SQL
        # =====================================================================
        print("\n=== PART 10: SPARK SQL DEMONSTRATION ===")

        # SQL 1: Count by state
        q1 = "SELECT State, COUNT(*) as Accident_Count FROM accidents GROUP BY State ORDER BY Accident_Count DESC"
        print(f"\nSQL Query 1: {q1}")
        sql_state_df = spark.sql(q1).toPandas()
        sql_state_path = TABLES_DIR / "sql_state_summary.csv"
        sql_state_df.to_csv(sql_state_path, index=False)
        generated_tables.append(sql_state_path.name)

        # SQL 2: Count by severity
        q2 = "SELECT Severity, COUNT(*) as Accident_Count, ROUND(COUNT(*) * 100.0 / 7728394, 2) as Percentage FROM accidents GROUP BY Severity ORDER BY Severity"
        print(f"\nSQL Query 2: {q2}")
        sql_sev_df = spark.sql(q2).toPandas()
        sql_sev_path = TABLES_DIR / "sql_severity_summary.csv"
        sql_sev_df.to_csv(sql_sev_path, index=False)
        generated_tables.append(sql_sev_path.name)

        # SQL 3: Top 10 cities
        q3 = "SELECT City, State, COUNT(*) as Accident_Count FROM accidents WHERE City IS NOT NULL GROUP BY City, State ORDER BY Accident_Count DESC LIMIT 10"
        print(f"\nSQL Query 3: {q3}")
        sql_city_df = spark.sql(q3).toPandas()
        sql_city_path = TABLES_DIR / "sql_top_cities.csv"
        sql_city_df.to_csv(sql_city_path, index=False)
        generated_tables.append(sql_city_path.name)

        # SQL 4: Count by year
        q4 = "SELECT YEAR(Start_Time) as Year, COUNT(*) as Accident_Count FROM accidents WHERE Start_Time IS NOT NULL GROUP BY YEAR(Start_Time) ORDER BY Year"
        print(f"\nSQL Query 4: {q4}")
        sql_year_df = spark.sql(q4).toPandas()
        sql_year_path = TABLES_DIR / "sql_year_summary.csv"
        sql_year_df.to_csv(sql_year_path, index=False)
        generated_tables.append(sql_year_path.name)

        # SQL 5: Severity distribution by state
        q5 = """
        SELECT State,
               COUNT(CASE WHEN Severity = 1 THEN 1 END) as Sev_1,
               COUNT(CASE WHEN Severity = 2 THEN 1 END) as Sev_2,
               COUNT(CASE WHEN Severity = 3 THEN 1 END) as Sev_3,
               COUNT(CASE WHEN Severity = 4 THEN 1 END) as Sev_4,
               COUNT(*) as Total
        FROM accidents
        GROUP BY State
        ORDER BY Total DESC
        """
        print(f"\nSQL Query 5: {q5}")
        sql_sev_state_df = spark.sql(q5).toPandas()
        sql_sev_state_path = TABLES_DIR / "sql_severity_by_state.csv"
        sql_sev_state_df.to_csv(sql_sev_state_path, index=False)
        generated_tables.append(sql_sev_state_path.name)

        # =====================================================================
        # PART 13 & 15 — VERIFICATION AND SUMMARY
        # =====================================================================
        print("\n=== VERIFICATION AND SUMMARY ===")
        csv_size_after = RAW_CSV.stat().st_size
        csv_unchanged = RAW_CSV.exists() and csv_size_before == EXPECTED_CSV_BYTES and csv_size_after == EXPECTED_CSV_BYTES
        print(f"RAW CSV SIZE BEFORE: {csv_size_before}")
        print(f"RAW CSV SIZE AFTER: {csv_size_after}")
        print(f"RAW CSV UNCHANGED: {csv_unchanged}")

        elapsed_total = time.perf_counter() - t_start

        print("\n============================================================")
        print("STAGE 6 FINAL SUMMARY")
        print("============================================================")
        print(f"Full Parquet Created/Reused: Reused ({PARQUET_DIR})")
        print(f"Parquet Row Count: {total_rows}")
        print(f"Parquet Columns: {total_cols}")
        print(f"Analytics Script: src/run_bda_analytics.py")
        print(f"Tables Generated: {len(generated_tables)} ({', '.join(sorted(generated_tables))})")
        print(f"Charts Generated: {len(generated_charts)} ({', '.join(sorted(generated_charts))})")
        print("Findings Document: docs/analytics_findings.md")
        print("Spark SQL Completed: True (5 queries executed)")
        print(f"Raw CSV Unchanged: {csv_unchanged}")
        print("ML Performed: False")

        status = "PASS" if (total_rows == EXPECTED_ROWS and csv_unchanged and len(generated_tables) >= 15 and len(generated_charts) >= 12) else "FAIL"
        print(f"\nStatus: {status}")

    except Exception as exc:
        print("\n=== ERROR DURING ANALYTICS ===")
        print(f"Error Type: {type(exc).__name__}")
        print(f"Error Message: {exc}")
        print("Status: FAIL")
        raise
    finally:
        spark.stop()
        print("spark_stopped=True")


if __name__ == "__main__":
    main()
