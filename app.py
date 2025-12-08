import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------- PAGE CONFIG --------------------

st.set_page_config(page_title="Global Attention Shift Predictor", layout="wide")

st.title("🌍 Global Attention Shift Predictor")
st.subheader("Predicting where human digital attention will move next")

st.markdown("""
This project analyzes:
- 🔍 Search Trends  
- 📺 Video Consumption  
- 📱 Screen Time Usage  

To forecast future digital attention using Machine Learning.
""")

st.success("✅ Streamlit App is running successfully")

st.markdown("---")

# -------------------- LOAD DATA --------------------

st.markdown("## 📂 Dataset Loading")

try:
    df = pd.read_csv("data/trending_searches_in_us.csv")
    st.success("✅ Dataset loaded successfully!")

    st.markdown("### 🔍 Raw Data Preview")
    st.dataframe(df.head())

except Exception as e:
    st.error("❌ Dataset not found or incorrect file format.")
    st.write(e)
    st.stop()

st.markdown("---")

# ================== ✅ ADVANCED DATA PREPROCESSING ==================

st.markdown("## 🧪 Advanced Data Preprocessing")

# 1️⃣ Dataset Shape Before
st.write("### 📌 Dataset Shape (Before Cleaning)")
st.write(df.shape)

# 2️⃣ Missing Values Before
st.write("### 🔎 Missing Values Before Cleaning")
st.dataframe(df.isnull().sum())

# 3️⃣ Remove Duplicate Rows
dup_count = df.duplicated().sum()
df = df.drop_duplicates()
st.write(f"### 🗑️ Duplicate Rows Removed: {dup_count}")

#DROP UNSTRUCTURED TEXT COLUMN (NOT USED IN CURRENT ML)
if "trend_breakdown" in df.columns:
    df = df.drop(columns=["trend_breakdown"])
    st.info("🗑️ 'trend_breakdown' column dropped (unstructured text not used for ML)")


# 4️⃣ Clean Date Column (If Exists)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    st.success("✅ Date column cleaned and sorted")

# 5️⃣ Handle Missing Numeric Values → MEAN
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

st.success("✅ Missing numeric values filled using mean")

# 6️⃣ Handle Missing Categorical Values → MODE
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

st.success("✅ Missing categorical values filled using mode")

# 7️⃣ Outlier Handling Using IQR
st.markdown("### 📉 Outlier Handling Using IQR Method")

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    original_count = df.shape[0]
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    removed = original_count - df.shape[0]

    st.write(f"Outliers removed from `{col}`: {removed}")

st.success("✅ Outliers handled using IQR method")

# 8️⃣ Missing Values After Cleaning
st.write("### ✅ Missing Values After Cleaning")
st.dataframe(df.isnull().sum())

# 9️⃣ Dataset Shape After Cleaning
st.write("### 📌 Dataset Shape (After Cleaning)")
st.write(df.shape)

st.success("✅ Dataset is now fully CLEANED and ML-ready")

st.markdown("---")

# -------------------- FIRST REAL VISUALIZATION --------------------

# -------------------- FIRST REAL VISUALIZATION --------------------

st.markdown("## 📈 Digital Attention Trend")

# Use start_date instead of date
if "start_date" in df.columns and len(numeric_cols) > 0:

    # Convert to datetime safely
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"])

    # ✅ Create daily trend instead of noisy minute-level spikes
    df["date_only"] = df["start_date"].dt.date

    daily_trend = df.groupby("date_only")[numeric_cols[0]].mean().reset_index()

    fig = px.line(
    daily_trend,
    x="date_only",
    y=numeric_cols[0],
    title="📈 Daily Average Digital Attention Trend",
    labels={
        "date_only": "Date",
        numeric_cols[0]: "Average Daily Attention"
    },
    markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Not enough data for time-series visualization")

st.markdown("---")

st.success("✅ Phase 2 Complete: Data Loaded, Cleaned & Visualized")
