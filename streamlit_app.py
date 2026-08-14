"""
Phase 4: Early-Warning Dashboard
Decision-support tool for educators - upload a cohort's weeks 1-4 engagement
data and get back flagged at-risk students, per assessment format, with
fairness transparency and a before/after mitigation comparison.

Run with: streamlit run streamlit_app.py
Expects a 'dashboard_models/' folder (from export_models_for_dashboard.py)
in the same directory.
"""

import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score

st.set_page_config(page_title="Early-Warning Dashboard", layout="wide", page_icon="🎓")

# ---------- Light visual polish ----------
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #1A1F2B;
    border: 1px solid #2A3040;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { font-size: 0.9em; opacity: 0.8; }
h1 { border-bottom: 3px solid #4A6FA5; padding-bottom: 12px; }
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Early-Warning Dashboard")
st.markdown("""
Upload a class's engagement data from **weeks 1-4** to identify students who may be
at risk of scoring below the pass mark, per assessment format. This tool is a
**decision-support aid** — flagged students should be reviewed by a teacher,
not treated as an automated pass/fail decision.
""")

MODEL_DIR = "dashboard_models"
FORMATS = ["TMA", "CMA", "Exam"]

# ---------- Known fairness limitations (from Phase 2 EOD results, Table II) ----------
KNOWN_FAIRNESS_GAPS = {
    "TMA": {"Age Band": 0.107, "IMD Band": 0.089, "Disability": 0.026, "Gender": 0.005},
    "CMA": {"IMD Band": 0.210, "Age Band": 0.152, "Gender": 0.114, "Disability": 0.095},
    "Exam": {"Age Band": 0.864, "IMD Band": 0.400, "Disability": 0.183, "Gender": 0.057},
}

# ---------- Before/after Reweighting mitigation (from notebook cells 34 and 40) ----------
MITIGATION_COMPARISON = {
    "TMA": {
        "baseline": {"recall": 0.185, "gender_eod": 0.0048, "imd_eod": 0.0891,
                      "age_eod": 0.1068, "disability_eod": 0.0256},
        "reweighted": {"recall": 0.200, "gender_eod": 0.0213, "imd_eod": 0.0824,
                        "age_eod": 0.1041, "disability_eod": 0.0247},
    },
    "CMA": {
        "baseline": {"recall": 0.189, "gender_eod": 0.1144, "imd_eod": 0.2096,
                      "age_eod": 0.1518, "disability_eod": 0.0946},
        "reweighted": {"recall": 0.3234, "gender_eod": 0.2551, "imd_eod": 0.2602,
                        "age_eod": 0.0559, "disability_eod": 0.0365},
    },
    "Exam": {
        "baseline": {"recall": 0.158, "gender_eod": 0.0568, "imd_eod": 0.4000,
                      "age_eod": 0.8636, "disability_eod": 0.1826},
        "reweighted": {"recall": 0.2632, "gender_eod": 0.1847, "imd_eod": 0.4000,
                        "age_eod": 0.8636, "disability_eod": 0.0731},
    },
}

@st.cache_resource
def load_models():
    models = {}
    for fmt in FORMATS:
        path = os.path.join(MODEL_DIR, f"{fmt.lower()}_model.pkl")
        if os.path.exists(path):
            models[fmt] = joblib.load(path)
    return models

models = load_models()

if not models:
    st.error(
        f"No models found in '{MODEL_DIR}/'. Run export_models_for_dashboard.py "
        "in your notebook first, then copy the resulting folder next to this app."
    )
    st.stop()

# =====================================================================
# Cross-format fairness comparison (all 3 formats, always visible)
# =====================================================================
st.divider()
st.subheader("📊 Fairness Comparison Across All Formats")
st.markdown(
    "This model's accuracy is not equally reliable across formats or demographic "
    "groups. The chart below shows all three assessment formats side by side."
)
all_formats_data = []
for fmt, gaps in KNOWN_FAIRNESS_GAPS.items():
    for attr, eod in gaps.items():
        all_formats_data.append({"Format": fmt, "Attribute": attr, "EOD": eod})
cross_format_df = pd.DataFrame(all_formats_data)
pivot_df = cross_format_df.pivot(index="Attribute", columns="Format", values="EOD")
st.bar_chart(pivot_df)
st.caption(
    "Final Examinations shows the largest fairness gaps overall, particularly for "
    "Age Band - consistent with the project's core finding that written exams are "
    "the riskiest format for equitable prediction."
)
st.divider()

# =====================================================================
# Sidebar controls
# =====================================================================
st.sidebar.header("1. Upload Cohort Data")
st.sidebar.markdown("""
CSV should contain one row per student with these columns:
`gender, imd_band, age_band, disability, region, num_of_prev_attempts,
studied_credits, date_submitted, is_banked, weight,
weeks_1_4_total_clicks, weeks_1_4_days_active, weeks_1_4_avg_weekly_clicks`
""")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

st.sidebar.header("2. Select Format")
selected_format = st.sidebar.selectbox("Assessment format", FORMATS)

st.sidebar.header("3. (Optional) Validate Against Real Outcomes")
st.sidebar.markdown(
    "If you have the actual outcome for each student (whether they scored "
    "below 40), upload it here to see how accurate the predictions were."
)
outcomes_file = st.sidebar.file_uploader(
    "Upload actual outcomes CSV (optional)", type="csv", key="outcomes"
)
st.sidebar.caption("Should contain a column named 'actual_at_risk' (1 = at risk, 0 = not), "
                    "same row order as your main upload.")

# =====================================================================
# Main logic - runs once a cohort CSV is uploaded
# =====================================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    bundle = models[selected_format]
    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_cols = bundle["feature_cols"]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded file is missing required columns: {missing_cols}")
        st.stop()

    df_encoded = df.copy()
    encoding_errors = []
    for col, le in encoders.items():
        try:
            df_encoded[col] = le.transform(df_encoded[col])
        except ValueError as e:
            encoding_errors.append(f"{col}: {e}")

    if encoding_errors:
        st.error("Some values in your file don't match the model's training categories:")
        for err in encoding_errors:
            st.text(f"  - {err}")
        st.stop()

    X = df_encoded[feature_cols]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = df.copy()
    results["Risk Prediction"] = ["At Risk" if p == 1 else "Not At Risk" for p in predictions]
    results["Risk Probability"] = probabilities.round(3)
    results = results.sort_values("Risk Probability", ascending=False)
    n_flagged = (predictions == 1).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Students Analyzed", len(df))
    col2.metric("Flagged At-Risk", int(n_flagged))
    col3.metric("Flag Rate", f"{n_flagged/len(df):.1%}")

    st.subheader(f"Flagged Students — {selected_format}")
    flagged = results[results["Risk Prediction"] == "At Risk"]
    if len(flagged) > 0:
        st.dataframe(
            flagged[["Risk Prediction", "Risk Probability"] +
                    [c for c in df.columns if c in ["gender", "imd_band", "age_band", "disability"]]],
            use_container_width=True
        )
        csv_out = flagged.to_csv(index=False)
        st.download_button("📥 Download flagged students (CSV)", csv_out, "flagged_students.csv")
    else:
        st.info("No students flagged as at-risk in this cohort.")

    # ---------- Real-outcomes validation (only shown if outcomes file uploaded) ----------
    if outcomes_file is not None:
        outcomes_df = pd.read_csv(outcomes_file)
        if 'actual_at_risk' not in outcomes_df.columns:
            st.sidebar.error("Outcomes file must contain an 'actual_at_risk' column.")
        elif len(outcomes_df) != len(df):
            st.sidebar.error(f"Outcomes file has {len(outcomes_df)} rows but cohort file has {len(df)}.")
        else:
            actual = outcomes_df['actual_at_risk'].values
            st.divider()
            st.subheader("✅ Live Validation Against Real Outcomes")
            vcol1, vcol2, vcol3 = st.columns(3)
            vcol1.metric("Accuracy", f"{accuracy_score(actual, predictions):.1%}")
            vcol2.metric("Recall (caught at-risk students)", f"{recall_score(actual, predictions):.1%}")
            vcol3.metric("Precision", f"{precision_score(actual, predictions, zero_division=0):.1%}")
            st.caption(
                "Recall being well below 100% is expected and consistent with the "
                "project's documented findings - this model catches a minority of "
                "at-risk students, which is why it should be used as an early flag "
                "alongside teacher judgment, not a sole decision-maker."
            )

    with st.expander("View all students"):
        st.dataframe(results, use_container_width=True)

    st.divider()

    # ---------- Fairness transparency (single format, detailed) ----------
    st.subheader("⚠️ Known Model Fairness Limitations")
    st.markdown(f"""
    This model's accuracy is **not equally reliable across all student groups** for
    **{selected_format}**. Based on validated testing (Equalized Odds Difference,
    0 = fully equal, higher = less equal):
    """)
    gaps = KNOWN_FAIRNESS_GAPS[selected_format]
    gap_df = pd.DataFrame(list(gaps.items()), columns=["Demographic Attribute", "Fairness Gap (EOD)"])
    gap_df = gap_df.sort_values("Fairness Gap (EOD)", ascending=False)
    st.bar_chart(gap_df.set_index("Demographic Attribute"), color="#4A6FA5")

    worst_attr = gap_df.iloc[0]["Demographic Attribute"]
    st.warning(
        f"The largest known gap for {selected_format} is by **{worst_attr}** "
        f"(EOD={gap_df.iloc[0]['Fairness Gap (EOD)']:.3f}). Predictions for students in "
        f"underrepresented subgroups of this attribute should be reviewed with extra care."
    )

    # ---------- Before/after mitigation comparison ----------
    if selected_format in MITIGATION_COMPARISON:
        st.divider()
        st.subheader("🔧 Impact of Fairness Mitigation (Reweighting)")
        st.markdown(
            "This project tested three bias-mitigation techniques; **Reweighting** gave "
            "the most balanced improvement overall. Here's how it changes model "
            "behaviour for this format:"
        )
        comp = MITIGATION_COMPARISON[selected_format]
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown("**Before Mitigation (Baseline)**")
            st.metric("Recall (at-risk students caught)", f"{comp['baseline']['recall']:.1%}")
            st.metric("Gender EOD", f"{comp['baseline']['gender_eod']:.4f}")
            st.metric("IMD Band EOD", f"{comp['baseline']['imd_eod']:.4f}")
            st.metric("Age Band EOD", f"{comp['baseline']['age_eod']:.4f}")
            st.metric("Disability EOD", f"{comp['baseline']['disability_eod']:.4f}")
        with mcol2:
            st.markdown("**After Reweighting**")
            st.metric("Recall (at-risk students caught)",
                       f"{comp['reweighted']['recall']:.1%}",
                       delta=f"{(comp['reweighted']['recall']-comp['baseline']['recall'])*100:+.1f}pp")
            for label, key in [("Gender EOD", "gender_eod"), ("IMD Band EOD", "imd_eod"),
                                ("Age Band EOD", "age_eod"), ("Disability EOD", "disability_eod")]:
                delta = comp['reweighted'][key] - comp['baseline'][key]
                st.metric(label, f"{comp['reweighted'][key]:.4f}",
                           delta=f"{delta:+.4f}", delta_color="inverse")
        st.caption(
            "Note: the model currently deployed above uses the baseline (non-mitigated) "
            "version. This comparison is shown for transparency about the tradeoffs "
            "involved - note Reweighting does not uniformly improve every metric (e.g. "
            "CMA's Gender EOD worsens under Reweighting), consistent with the project's "
            "core finding that no single technique improves fairness across all groups "
            "simultaneously."
        )

else:
    st.info("👈 Upload a CSV file using the sidebar to get started.")
    with st.expander("See example CSV format"):
        example = pd.DataFrame({
            "gender": ["M", "F"], "imd_band": ["20-30%", "80-90%"], "age_band": ["0-35", "35-55"],
            "disability": ["N", "Y"], "region": ["London Region", "Scotland"],
            "num_of_prev_attempts": [0, 1], "studied_credits": [60, 120],
            "date_submitted": [5, 12], "is_banked": [0, 0], "weight": [20, 30],
            "weeks_1_4_total_clicks": [150, 45], "weeks_1_4_days_active": [12, 4],
            "weeks_1_4_avg_weekly_clicks": [37.5, 11.25],
        })
        st.dataframe(example)