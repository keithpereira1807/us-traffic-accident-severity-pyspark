"""
US Traffic Accident Severity Analytics — Interactive Web Dashboard
Big Data Analytics and Machine Learning Using PySpark

A professional, local Streamlit dashboard showcasing the verified 7.7M-row
big data analytics, class imbalance handling, PySpark MLlib models,
and real-time severity classification.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Add project root to sys.path for local imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import predictor, TOP_STATES, TOP_WEATHER, INFRASTRUCTURE_FEATURES

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="US Accident Severity Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Executive CSS Styling (Clean Blue / White Analytics Theme)
st.markdown(
    """
    <style>
    /* Main container and typography */
    .main {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Executive Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 15px;
        opacity: 0.9;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-val {
        font-size: 26px;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
    }

    /* Pipeline Banner */
    .pipeline-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        font-size: 13px;
        font-weight: 600;
        color: #334155;
        background: #f1f5f9;
        padding: 8px 14px;
        border-radius: 6px;
        border-left: 3px solid #2563eb;
    }
    .pipeline-arrow {
        color: #94a3b8;
        font-weight: bold;
    }

    /* Prediction Result Cards */
    .pred-card {
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .pred-sev-1 { background: linear-gradient(135deg, #15803d 0%, #22c55e 100%); }
    .pred-sev-2 { background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); }
    .pred-sev-3 { background: linear-gradient(135deg, #c2410c 0%, #f97316 100%); }
    .pred-sev-4 { background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%); }

    /* Alert Banner */
    .info-banner {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 14px;
        color: #1e40af;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg",
    width=160,
)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "1. Overview",
        "2. Temporal Analysis",
        "3. Geographic Analysis",
        "4. Weather & Road Conditions",
        "5. Severity Analysis",
        "6. Model Comparison",
        "7. Predict Severity",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Specs**
    - **Engine**: PySpark 4.2.0
    - **Dataset**: 7,728,394 Rows
    - **Target**: Severity (Classes 1–4)
    - **Memory**: ~7.37 GB Host RAM
    - **Status**: Complete & Verified
    """
)


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page.startswith("1."):
    st.markdown(
        """
        <div class="main-header">
            <h1>US Traffic Accident Severity Analytics</h1>
            <p>Big Data Analytics and Machine Learning Using PySpark • 7.728M Records Analyzed</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Project Pipeline Visual
    st.markdown(
        """
        <div class="pipeline-container">
            <div class="pipeline-step">1. Raw CSV (2.85 GB)</div>
            <span class="pipeline-arrow">➔</span>
            <div class="pipeline-step">2. PySpark Explicit Schema</div>
            <span class="pipeline-arrow">➔</span>
            <div class="pipeline-step">3. Snappy Parquet (~726 MB)</div>
            <span class="pipeline-arrow">➔</span>
            <div class="pipeline-step">4. Distributed BDA & SQL</div>
            <span class="pipeline-arrow">➔</span>
            <div class="pipeline-step">5. Class-Weighted MLlib</div>
            <span class="pipeline-arrow">➔</span>
            <div class="pipeline-step">6. Severity Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Metrics Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-val">7,728,394</div><div class="kpi-label">Accident Records</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-val">46</div><div class="kpi-label">Attributes</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-val">49</div><div class="kpi-label">US States</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-val">2.85 GB</div><div class="kpi-label">Raw Dataset</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-val">~726 MB</div><div class="kpi-label">Snappy Parquet</div></div>',
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            '<div class="kpi-card"><div class="kpi-val">PySpark 4.2</div><div class="kpi-label">Compute Engine</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Big Data Processing Evidence
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("Big Data Processing Evidence")
        evidence_data = {
            "Dimension / Requirement": [
                "Dataset Row Count",
                "Dataset Column Count",
                "Raw CSV File Size",
                "Columnar Parquet Size",
                "PySpark Version",
                "Hadoop Integration",
                "Host RAM Constraint",
                "Analytical Artifacts",
                "Trained ML Models",
            ],
            "Verified Project Metric": [
                "7,728,394 verified rows (100% unique IDs)",
                "46 attributes (explicit StructType schema)",
                "3,058,183,727 bytes (2.85 GB uncompressed)",
                "~726 MB Snappy compressed (74% compression savings)",
                "PySpark 4.2.0 on Apache Spark",
                "Hadoop 3.5.0 with native Windows winutils",
                "~7.37 GB RAM (memory-conscious local[1] execution)",
                "23 analytical CSV tables & 13 high-res charts",
                "Logistic Regression, Decision Tree, Random Forest",
            ],
        }
        st.dataframe(pd.DataFrame(evidence_data), hide_index=True, use_container_width=True)

    with col_r:
        st.subheader("Severity Target Distribution")
        sev_df = pd.read_csv(TABLES_DIR / "severity_distribution.csv")
        st.dataframe(
            sev_df.rename(
                columns={
                    "Severity": "Severity Level",
                    "Count": "Incident Count",
                    "Percentage": "Percentage (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.image(str(CHARTS_DIR / "severity_distribution.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Key Macro Trends at a Glance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Accidents by Year (2016–2023)")
        st.image(str(CHARTS_DIR / "accidents_by_year.png"), use_container_width=True)
    with col2:
        st.markdown("#### Top 10 States by Recorded Volume")
        st.image(str(CHARTS_DIR / "top_10_states.png"), use_container_width=True)


# ==============================================================================
# PAGE 2: TEMPORAL ANALYSIS
# ==============================================================================
elif page.startswith("2."):
    st.markdown(
        """
        <div class="main-header">
            <h1>Temporal Analytics</h1>
            <p>Diurnal, Day-of-Week, Monthly, and Annual Accident Patterns</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-banner">
            <strong>Observational Note:</strong> Temporal accident concentrations align with high-exposure commuter travel periods. 
            These statistics reflect historical event volumes and do not imply that commuting inherently causes accidents.
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2 = st.tabs(["Diurnal (Hourly) & Weekly", "Annual & Monthly Seasonality"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Diurnal Hourly Distribution")
            st.image(str(CHARTS_DIR / "accidents_by_hour.png"), use_container_width=True)
            st.markdown(
                """
                **Hourly Insights:**
                - **Morning Commute Peak**: **7:00 AM – 8:00 AM** accounts for **14.09%** of all daily accidents.
                - **Evening Commute Peak**: **4:00 PM – 5:00 PM** accounts for **13.41%** of daily accidents.
                - **Overnight Trough**: Drops to **0.96%** at **3:00 AM**, correlating with minimal traffic density.
                """
            )

        with c2:
            st.markdown("### Day of Week Distribution")
            st.image(str(CHARTS_DIR / "accidents_by_day_of_week.png"), use_container_width=True)
            st.markdown(
                """
                **Weekly Insights:**
                - **Weekday Volume**: **76.55%** of all accidents occur Monday through Friday.
                - **Peak Day**: **Friday** records the highest volume (**1,237,229 incidents / 16.01%**).
                - **Weekends**: Saturday (12.28%) and Sunday (11.17%) represent substantially lower volumes.
                """
            )

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Multi-Year Trend & Reporting Evolution")
            st.image(str(CHARTS_DIR / "accidents_by_year.png"), use_container_width=True)
            st.markdown(
                """
                **Annual Trends:**
                - Recorded accident counts grew from 2016 through 2021 as additional state sensor APIs and municipal data sources were integrated.
                - Year 2023 represents a partial-year observation window ending April 1, 2023.
                """
            )

        with c2:
            st.markdown("### Severity Trends Across Years")
            st.image(str(CHARTS_DIR / "accidents_by_year_severity.png"), use_container_width=True)
            st.markdown(
                """
                **Severity Cross-Tabulation:**
                - Severity 2 remains the dominant operational classification across all observation years.
                - Severe disruption classes (Severity 3 & 4) represent consistent baseline shares.
                """
            )


# ==============================================================================
# PAGE 3: GEOGRAPHIC ANALYSIS
# ==============================================================================
elif page.startswith("3."):
    st.markdown(
        """
        <div class="main-header">
            <h1>Geographic Analytics</h1>
            <p>Spatial Distribution Across 49 States, 13,678 Cities, and 1,871 Counties</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-banner">
            <strong>Important Population Caveat:</strong> Accident counts reflect raw event volumes driven by extensive road network mileage and high populations.
            A state with higher counts is not necessarily "more dangerous" on a per-capita or per-VMT (Vehicle Miles Traveled) basis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Top 10 States by Volume")
        st.image(str(CHARTS_DIR / "top_10_states.png"), use_container_width=True)

        states_df = pd.read_csv(TABLES_DIR / "top_10_states.csv")
        st.dataframe(
            states_df.rename(
                columns={
                    "State": "State Code",
                    "Count": "Accident Count",
                    "Percentage_of_Total": "Share (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with c2:
        st.subheader("Top 20 Urban Metros (Cities)")
        st.image(str(CHARTS_DIR / "top_20_cities.png"), use_container_width=True)

        cities_df = pd.read_csv(TABLES_DIR / "top_20_cities.csv")
        st.dataframe(
            cities_df.head(10).rename(
                columns={
                    "City": "City Name",
                    "State": "State",
                    "Count": "Accident Count",
                    "Percentage_of_Total": "Share (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("State-Wise Severity Proportions")
    st.image(str(CHARTS_DIR / "severity_by_state.png"), use_container_width=True)


# ==============================================================================
# PAGE 4: WEATHER & ROAD CONDITIONS
# ==============================================================================
elif page.startswith("4."):
    st.markdown(
        """
        <div class="main-header">
            <h1>Weather & Road Infrastructure Analytics</h1>
            <p>Ambient Meteorological Telemetry & Physical Traffic Management Controls</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    w1, w2 = st.tabs(["Meteorological Conditions", "Road Infrastructure Indicators"])

    with w1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Top Recorded Weather Conditions")
            st.image(str(CHARTS_DIR / "top_weather_conditions.png"), use_container_width=True)
            st.markdown(
                """
                **Prevalence of Fair Weather:**
                - Over **81%** of all accidents occur during benign conditions (*Fair: 32.84%*, *Mostly Cloudy: 13.11%*, *Clear: 11.34%*).
                - *Interpretation*: Most driving occurs during good weather, naturally generating the majority of incident exposure.
                """
            )

        with c2:
            st.markdown("### Weather vs. Severity Cross-Tabulation")
            st.image(str(CHARTS_DIR / "weather_severity_summary.png"), use_container_width=True)
            st.markdown(
                """
                **Adverse Weather Observations:**
                - Precipitation, fog, and snow correlate with localized visibility reductions and lower surface friction.
                - However, severity distributions within rainy/foggy conditions maintain proportions comparable to fair conditions.
                """
            )

        st.markdown("### Meteorological Numerical Sensor Summary")
        weather_num_df = pd.read_csv(TABLES_DIR / "weather_numeric_summary.csv")
        st.dataframe(weather_num_df, hide_index=True, use_container_width=True)

    with w2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Infrastructure Control Frequencies")
            st.image(str(CHARTS_DIR / "infrastructure_features.png"), use_container_width=True)
            st.markdown(
                """
                **Common Roadway Fixtures:**
                - `Traffic_Signal`: Present in **14.80%** of records (1,143,485 incidents).
                - `Crossing`: Present in **11.31%** of records (874,019 incidents).
                - `Junction`: Present in **7.28%** of records (562,818 incidents).
                """
            )

        with c2:
            st.markdown("### Infrastructure Feature Breakdown")
            infra_df = pd.read_csv(TABLES_DIR / "infrastructure_features.csv")
            st.dataframe(
                infra_df.rename(
                    columns={
                        "Feature": "Infrastructure Feature",
                        "True_Count": "Records Present",
                        "True_Percentage": "Frequency (%)",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown(
                """
                **Junction Severity Association:**
                - Incidents occurring at highway interchanges and merge ramps (`Junction=True`) exhibit a higher proportion of **Severity 3 & 4** events compared to surface intersections.
                """
            )


# ==============================================================================
# PAGE 5: SEVERITY ANALYSIS
# ==============================================================================
elif page.startswith("5."):
    st.markdown(
        """
        <div class="main-header">
            <h1>Accident Severity Analysis & Impact Extent</h1>
            <p>Target Class Distributions, Operational Definitions, and Corridor Queue Lengths</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Severity Class Breakdown")
        sev_classes = {
            "Severity Level": ["Severity 1", "Severity 2", "Severity 3", "Severity 4"],
            "Operational Definition": [
                "Short delay / minor traffic disruption / rapid clearance",
                "Moderate traffic delay / standard partial lane slowdown",
                "Significant traffic backup / multi-lane corridor impact",
                "Critical road closure / long delay / major corridor blockage",
            ],
            "Record Count": ["67,366", "6,156,981", "1,299,337", "204,710"],
            "Share (%)": ["0.87%", "79.67%", "16.81%", "2.65%"],
        }
        st.dataframe(pd.DataFrame(sev_classes), hide_index=True, use_container_width=True)

        st.markdown(
            """
            **The Class Imbalance Problem:**
            - Severity 2 represents almost four out of five records (**79.67%**).
            - Minor accidents (**0.87%**) and critical closures (**2.65%**) are rare minority events.
            - Unweighted classification models default to predicting only Severity 2; hence, **balanced class weighting** was implemented.
            """
        )

    with c2:
        st.subheader("Train vs. Test Split Imbalance (Seed 42)")
        st.image(str(CHARTS_DIR / "train_test_class_distribution.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Physical Corridor Impact: Distance(mi) by Severity")
    cd1, cd2 = st.columns([1, 1])
    with cd1:
        st.image(str(CHARTS_DIR / "distance_buckets.png"), use_container_width=True)
    with cd2:
        st.markdown("#### Distance Statistics by Severity Class")
        dist_df = pd.read_csv(TABLES_DIR / "distance_by_severity.csv")
        st.dataframe(dist_df, hide_index=True, use_container_width=True)
        st.markdown(
            """
            **Physical Extent Takeaway:**
            - **42.73%** of all records have `Distance = 0.0` (point-source accidents).
            - **Severity 4** incidents impact an average corridor extent of **1.50 miles** (median 0.47 mi), substantially larger than Severity 2 (mean 0.30 mi).
            """
        )


# ==============================================================================
# PAGE 6: MODEL COMPARISON
# ==============================================================================
elif page.startswith("6."):
    st.markdown(
        """
        <div class="main-header">
            <h1>Machine Learning Model Comparison</h1>
            <p>Evaluation of PySpark MLlib Classifiers on 1,545,478 Test Records</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Summary Performance Table")
    comp_df = pd.read_csv(TABLES_DIR / "model_comparison.csv")
    st.dataframe(
        comp_df.rename(
            columns={
                "Model": "Model Architecture",
                "Accuracy": "Accuracy",
                "Weighted_Precision": "Weighted Precision",
                "Weighted_Recall": "Weighted Recall",
                "Weighted_F1": "Weighted F1",
                "Macro_F1": "Macro F1",
                "Training_Time_s": "Training Time (s)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.image(str(CHARTS_DIR / "model_comparison_metrics.png"), use_container_width=True)

    st.markdown(
        """
        <div class="info-banner">
            <strong>Key Finding:</strong> Within this experiment and on the held-out test partition, the 
            <strong>Decision Tree Classifier</strong> achieved the highest measured 
            <strong>Accuracy (0.4738)</strong>, <strong>Weighted F1 (0.5462)</strong>, and 
            <strong>Macro F1 (0.3656)</strong> among the three evaluated PySpark models.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Per-Class Precision, Recall, and F1-Score")
    per_class_df = pd.read_csv(TABLES_DIR / "per_class_metrics.csv")
    st.dataframe(per_class_df, hide_index=True, use_container_width=True)

    st.markdown(
        """
        **Impact of Balanced Class Loss Weighting ($w_c = N_{\\text{train}} / (4 \\times N_{\\text{train}, c})$):**
        - **Severity 1 Recall**: Decision Tree achieved **93.57%**, capturing 12,498 out of 13,357 rare minor incidents.
        - **Severity 4 Recall**: Decision Tree achieved **86.85%** (and Random Forest achieved **85.37%**), successfully flagging critical road closures.
        - *Operational Tradeoff*: Increased minority recall reduces majority precision, which is advantageous for emergency dispatch where missing a major closure carries catastrophic consequences.
        """
    )

    st.markdown("---")
    st.subheader("Confusion Matrices (Test Set Predictions)")
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.markdown("#### Decision Tree (Best)")
        st.image(str(CHARTS_DIR / "confusion_matrix_decision_tree.png"), use_container_width=True)
    with cm2:
        st.markdown("#### Logistic Regression")
        st.image(str(CHARTS_DIR / "confusion_matrix_logistic_regression.png"), use_container_width=True)
    with cm3:
        st.markdown("#### Random Forest")
        st.image(str(CHARTS_DIR / "confusion_matrix_random_forest.png"), use_container_width=True)


# ==============================================================================
# PAGE 7: PREDICT SEVERITY
# ==============================================================================
elif page.startswith("7."):
    st.markdown(
        """
        <div class="main-header">
            <h1>What will be the predicted accident severity?</h1>
            <p>Enter accident characteristics to classify the severity of an accident using the trained PySpark ML model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-banner">
            <strong>Important Operational Notice:</strong> This is an accident severity classification model. 
            It predicts the severity class for an accident record based on the supplied characteristics. 
            It does <strong>not</strong> predict whether an accident will occur.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Form layout
    with st.form("prediction_form"):
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            st.markdown("#### 1. Location & Physical Corridor Extent")
        with col_m2:
            model_choice = st.selectbox(
                "Model Architecture",
                ["Decision Tree (Best Model)", "Logistic Regression"],
                index=0,
            )

        cl1, cl2, cl3, cl4 = st.columns(4)
        with cl1:
            input_state = st.selectbox("State", TOP_STATES, index=0)
        with cl2:
            input_lat = st.number_input("Start Latitude", value=34.0522, format="%.4f")
        with cl3:
            input_lng = st.number_input("Start Longitude", value=-118.2437, format="%.4f")
        with cl4:
            input_dist = st.number_input(
                "Corridor Impact Distance (mi)",
                value=0.5,
                min_value=0.0,
                max_value=100.0,
                step=0.1,
            )

        st.markdown("#### 2. Temporal Characteristics")
        ct1, ct2, ct3, ct4 = st.columns(4)
        with ct1:
            input_year = st.slider("Year", min_value=2016, max_value=2023, value=2022)
        with ct2:
            input_month = st.selectbox(
                "Month",
                list(range(1, 13)),
                index=5,
                format_func=lambda m: pd.to_datetime(f"2022-{m}-01").strftime("%B"),
            )
        with ct3:
            input_dow = st.selectbox(
                "Day of Week",
                [1, 2, 3, 4, 5, 6, 7],
                index=5,
                format_func=lambda d: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d - 1],
            )
        with ct4:
            input_hour = st.slider("Hour of Day (0–23)", min_value=0, max_value=23, value=17)

        st.markdown("#### 3. Ambient Meteorological Conditions")
        cw1, cw2, cw3, cw4 = st.columns(4)
        with cw1:
            input_temp = st.number_input("Temperature (°F)", value=65.0, step=1.0)
            input_hum = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0)
        with cw2:
            input_press = st.number_input("Pressure (in)", value=29.90, format="%.2f")
            input_vis = st.number_input("Visibility (mi)", value=10.0, min_value=0.0, max_value=140.0)
        with cw3:
            input_wind = st.number_input("Wind Speed (mph)", value=8.0, min_value=0.0)
            input_precip = st.number_input(
                "Precipitation (in)", value=0.0, min_value=0.0, format="%.2f"
            )
        with cw4:
            input_weather = st.selectbox("Weather Condition", TOP_WEATHER, index=0)
            input_daynight = st.selectbox("Daylight Indicator", ["Day", "Night"], index=0)

        st.markdown("#### 4. Nearby Roadway & Infrastructure Indicators")
        infra_cols = st.columns(6)
        infra_dict = {}
        for idx, feat in enumerate(INFRASTRUCTURE_FEATURES):
            col_target = infra_cols[idx % 6]
            with col_target:
                infra_dict[feat] = st.checkbox(
                    feat.replace("_", " "),
                    value=(feat in ["Traffic_Signal", "Crossing"]),
                )

        submit_btn = st.form_submit_button("⚡ Predict Severity", use_container_width=True)

    if submit_btn:
        st.markdown("---")
        # Run prediction via the lightweight exact inference engine
        with st.spinner("Classifying accident severity via trained PySpark model..."):
            pred_res = predictor.predict(
                model_name=model_choice,
                start_lat=input_lat,
                start_lng=input_lng,
                distance=input_dist,
                temperature=input_temp,
                humidity=input_hum,
                pressure=input_press,
                visibility=input_vis,
                wind_speed=input_wind,
                precipitation=input_precip,
                year=input_year,
                month=input_month,
                day_of_week=input_dow,
                hour=input_hour,
                state=input_state,
                weather_condition=input_weather,
                sunrise_sunset=input_daynight,
                civil_twilight=input_daynight,
                infrastructure=infra_dict,
            )

        sev = pred_res["severity"]
        sev_class_name = f"pred-sev-{sev}"

        # Prediction Display Card
        st.markdown(
            f"""
            <div class="pred-card {sev_class_name}">
                <div style="font-size: 13px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; opacity: 0.9;">
                    Predicted Severity Class
                </div>
                <div style="font-size: 38px; font-weight: 800; margin: 4px 0 8px 0;">
                    {pred_res['label']}
                </div>
                <div style="font-size: 16px; font-weight: 500; opacity: 0.95;">
                    {pred_res['description']}
                </div>
                <div style="margin-top: 12px; font-size: 13px; opacity: 0.85;">
                    Model Used: {pred_res['model_used']} • Classification Confidence: {pred_res['confidence']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probabilities & Submitted Characteristics
        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            st.subheader("Model Class Probabilities")
            prob_df = pd.DataFrame(
                list(pred_res["probabilities"].items()),
                columns=["Severity Class", "Probability"],
            )
            prob_df["Confidence (%)"] = (prob_df["Probability"] * 100).round(2)
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

            # Bar chart of probabilities
            st.bar_chart(prob_df.set_index("Severity Class")["Probability"])

        with p_col2:
            st.subheader("Submitted Incident Characteristics")
            active_infra = [k.replace("_", " ") for k, v in infra_dict.items() if v]
            char_data = {
                "Attribute Group": [
                    "Location",
                    "Impact Distance",
                    "Timing",
                    "Ambient Weather",
                    "Active Infrastructure",
                ],
                "Submitted Value": [
                    f"{input_state} ({input_lat:.4f}, {input_lng:.4f})",
                    f"{input_dist:.2f} miles",
                    f"{input_year}, Month {input_month}, Hour {input_hour}:00",
                    f"{input_weather}, {input_temp}°F, {input_hum}% Hum, {input_daynight}",
                    ", ".join(active_infra) if active_infra else "None reported",
                ],
            }
            st.dataframe(pd.DataFrame(char_data), hide_index=True, use_container_width=True)
