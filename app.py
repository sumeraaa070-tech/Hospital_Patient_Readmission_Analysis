import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Hospital Readmission Analytics",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Patient Readmission Analytics")
st.caption("Exploratory healthcare analytics using historical hospital encounters")

DATA_PATH = "/content/diabetic_data.csv"
OUT = "/content/outputs"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.replace(["?", "Unknown/Invalid", ""], pd.NA)
    data["readmitted_binary"] = data["readmitted"].isin(["<30", ">30"]).astype(int)
    return data

df = load_data()

st.sidebar.header("Dashboard Filters")

ages = sorted(df["age"].dropna().unique().tolist())
selected_ages = st.sidebar.multiselect(
    "Select age groups",
    ages,
    default=ages
)

stay_min = int(df["time_in_hospital"].min())
stay_max = int(df["time_in_hospital"].max())

stay_range = st.sidebar.slider(
    "Days in hospital",
    stay_min,
    stay_max,
    (stay_min, stay_max)
)

filtered = df[
    df["age"].isin(selected_ages)
    & df["time_in_hospital"].between(stay_range[0], stay_range[1])
].copy()

total = len(filtered)
rate = (
    filtered["readmitted_binary"].mean() * 100
    if total else 0
)

early_rate = (
    filtered["readmitted"].eq("<30").mean() * 100
    if total else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hospital Encounters", f"{total:,}")
c2.metric("Any Recorded Readmission", f"{rate:.1f}%")
c3.metric("Readmission Within 30 Days", f"{early_rate:.1f}%")
c4.metric("Average Hospital Stay",
          f"{filtered['time_in_hospital'].mean():.1f} days"
          if total else "N/A")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Readmission Categories")
    category_counts = (
        filtered["readmitted"]
        .value_counts()
        .reindex(["NO", ">30", "<30"], fill_value=0)
        .rename_axis("Category")
        .reset_index(name="Encounters")
    )
    fig = px.bar(
        category_counts,
        x="Category",
        y="Encounters",
        color="Category",
        title="Encounter Count by Readmission Category"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Readmission by Age")
    age_chart = (
        filtered.groupby("age", observed=True)["readmitted_binary"]
        .mean()
        .mul(100)
        .reset_index(name="Readmission Rate (%)")
    )
    fig = px.bar(
        age_chart,
        x="age",
        y="Readmission Rate (%)",
        title="Readmission Rate Across Age Groups"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Hospital Stay Analysis")
stay_chart = (
    filtered.groupby("time_in_hospital")["readmitted_binary"]
    .mean()
    .mul(100)
    .reset_index(name="Readmission Rate (%)")
)
fig = px.line(
    stay_chart,
    x="time_in_hospital",
    y="Readmission Rate (%)",
    markers=True,
    title="Readmission Rate by Length of Stay"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model Comparison")
comparison_path = os.path.join(OUT, "model_comparison.csv")
if os.path.exists(comparison_path):
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True)
    available_metrics = [
        c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        if c in comparison.columns
    ]
    if available_metrics:
        chart_data = comparison.set_index("Model")[available_metrics]
        st.bar_chart(chart_data)
else:
    st.info("Run the model comparison notebook cells to create this table.")

st.subheader("Feature Importance")
importance_path = os.path.join(OUT, "feature_importance.csv")
if os.path.exists(importance_path):
    importance = pd.read_csv(importance_path).head(15)
    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features by Permutation Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered Encounter Sample")
st.dataframe(
    filtered[
        ["age", "gender", "time_in_hospital", "num_lab_procedures",
         "num_medications", "readmitted"]
    ].head(100),
    use_container_width=True
)

st.caption(
    "Educational analysis only. These historical associations are not "
    "causal findings or clinical recommendations. Encounters may include "
    "multiple visits by the same patient."
)
