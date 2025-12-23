# app.py
"""
Global Attention Shift Predictor - unified Streamlit app
Implements:
- Multi-dataset backend preprocessing (hidden)
- Builds Human Attention Index (HAI)
- Dashboard (HAI charts + KPI cards)
- Trend Analysis (HAI time series)
- Forecast (Linear Regression) with evaluation (MAE, RMSE, MAPE)
- Trend Classification (Logistic) with evaluation (Accuracy, Precision, Recall, F1, Confusion matrix)
- Predict page: One-click predict using latest data + optional manual input
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

def safe_rerun():
    """
    Robust rerun function that works across Streamlit versions.
    It prioritizes the modern st.rerun() and st.query_params,
    but falls back to older methods if necessary.
    """
    # 1. Try the modern standard st.rerun() (Streamlit >= 1.27)
    if hasattr(st, "rerun"):
        st.rerun()
        return

    # 2. Try the older st.experimental_rerun()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
        return

    # 3. Fallback: Trigger a reload via query params
    try:
        # Modern: st.query_params (Streamlit >= 1.30)
        # Uses dictionary syntax
        st.query_params["_rerun"] = str(time.time())
    except (AttributeError, TypeError):
        # Legacy: st.experimental_set_query_params (Deprecation warning source)
        try:
            st.experimental_set_query_params(_rerun=str(time.time()))
        except Exception:
            pass


# -------------------------
# Page config & CSS (dark theme, compact)
# -------------------------
st.set_page_config(page_title="Global Attention Shift Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .main { background: #0b0f14; color: #e6eef6; }
    .reportview-container .main .block-container{padding-top:1rem;}
    .kpi {background:#071225; padding:16px; border-radius:8px; text-align:center; color:#e6eef6;}
    .small { color:#bcd5ea; font-size:13px; }
    .hero { background: linear-gradient(180deg,#0f1724,#081025); padding:30px; border-radius:12px; margin-bottom:18px; }
    .navbtn{ background:#14293f; color:#eaf5ff; padding:10px 18px; border-radius:8px; border: none; }
    .feature { background:#071225; padding:12px; border-radius:8px; color:#dbeeff; }
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# Utility helpers
# -------------------------
def zscore(series):
    # Check for zero variance to avoid divide-by-zero
    if series.std(ddof=0) == 0:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / series.std(ddof=0)

def safe_parse_dates(df, candidates):
    """Return a parsed date column name or None"""
    for c in candidates:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                if df[c].notna().sum() > 0:
                    return c
            except Exception:
                continue
    return None

# -------------------------
# Cleaning functions for each dataset
# (adjust column-name arrays here if your files differ)
# -------------------------
def clean_search(df):
    """Expect: date-like col (start_date or date) and numeric search_volume"""
    df = df.copy()
    date_col = safe_parse_dates(df, ["start_date", "date", "timestamp"])
    if date_col is None:
        raise ValueError("Search dataset missing a recognizable date column")
    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    # candidate volume columns
    vol_cols = [c for c in ["search_volume", "volume", "searches"] if c in df.columns]
    if len(vol_cols) == 0:
        df["search_volume"] = 0.0
    else:
        df["search_volume"] = pd.to_numeric(df[vol_cols[0]], errors='coerce').fillna(0)
    daily = df.groupby("date")[["search_volume"]].mean().reset_index()
    return daily

def clean_youtube(df):
    """Expect columns: publish_date (or upload_date), views, likes"""
    df = df.copy()
    date_col = safe_parse_dates(df, ["publish_date", "upload_date", "date"])
    if date_col is None:
        raise ValueError("YouTube file missing a date column (publish_date/upload_date/date)")
    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    # views
    vcol = next((c for c in ["views", "view_count", "total_views"] if c in df.columns), None)
    lcol = next((c for c in ["likes", "like_count"] if c in df.columns), None)
    df['views'] = pd.to_numeric(df[vcol], errors='coerce').fillna(0) if vcol else 0.0
    df['likes'] = pd.to_numeric(df[lcol], errors='coerce').fillna(0) if lcol else 0.0
    daily = df.groupby("date")[["views", "likes"]].mean().reset_index().rename(columns={"views": "youtube_views", "likes": "youtube_likes"})
    return daily

def clean_tiktok(df):
    """Expect upload_date (or date) and views"""
    df = df.copy()
    date_col = safe_parse_dates(df, ["upload_date", "publish_date", "date"])
    if date_col is None:
        raise ValueError("TikTok file missing date column (upload_date/upload_date/date)")
    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    vcol = next((c for c in ["views","view_count","plays"] if c in df.columns), None)
    df['tiktok_views'] = pd.to_numeric(df[vcol], errors='coerce').fillna(0) if vcol else 0.0
    daily = df.groupby("date")[["tiktok_views"]].mean().reset_index()
    return daily

def clean_screen_time(df):
    """Expect date and screen_time_min (or total_minutes)"""
    df = df.copy()
    date_col = safe_parse_dates(df, ["date", "start_date", "timestamp"])
    if date_col is None:
        raise ValueError("Screen time file missing 'date' column")
    # attempt to parse a time-like string to a numeric minute (if provided as HH:MM:SS or mm:ss)
    def parse_time_cell(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        # try HH:MM:SS or MM:SS
        parts = s.split(':')
        try:
            if len(parts) == 3:
                h,m,sec = map(float, parts)
                return h*60 + m + sec/60.0
            if len(parts) == 2:
                m,sec = map(float, parts)
                return m + sec/60.0
        except Exception:
            pass
        # fallback numeric
        try:
            return float(s)
        except Exception:
            return np.nan

    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    # candidate column names
    scols = [c for c in ["screen_time_min", "screen_time", "minutes", "screen_time_minutes", "duration"] if c in df.columns]
    if len(scols) == 0:
        # maybe time strings under 'date' -> not expected; set to 0
        df['screen_time_min'] = 0.0
    else:
        df['screen_time_min'] = df[scols[0]].apply(parse_time_cell).fillna(0)
    daily = df.groupby("date")[["screen_time_min"]].mean().reset_index()
    return daily

# -------------------------
# Load & preprocess ALL datasets (cached)
# -------------------------
@st.cache_data(ttl=3600)
def load_and_build_master():
    # Load each file if present; missing files will be skipped but app will still run.
    import os
    data_path = "data"
    # placeholders
    search_df = youtube_df = tiktok_df = screen_df = None
    msgs = []

    # SEARCH
    try:
        p = os.path.join(data_path, "trending_searches_in_us.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            search_df = clean_search(df)
            msgs.append("search OK")
        else:
            msgs.append("search missing")
    except Exception as e:
        msgs.append(f"search err: {e}")

    # YOUTUBE
    try:
        p = os.path.join(data_path, "youtube.xlsx")
        if os.path.exists(p):
            df = pd.read_excel(p)
            youtube_df = clean_youtube(df)
            msgs.append("youtube OK")
        else:
            msgs.append("youtube missing")
    except Exception as e:
        msgs.append(f"youtube err: {e}")

    # TIKTOK
    try:
        p = os.path.join(data_path, "tiktok_data.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            tiktok_df = clean_tiktok(df)
            msgs.append("tiktok OK")
        else:
            msgs.append("tiktok missing")
    except Exception as e:
        msgs.append(f"tiktok err: {e}")

    # SCREEN TIME
    try:
        p = os.path.join(data_path, "screen_time_app_usage_dataset.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            screen_df = clean_screen_time(df)
            msgs.append("screen OK")
        else:
            msgs.append("screen missing")
    except Exception as e:
        msgs.append(f"screen err: {e}")

    # Combine (outer join on date)
    # Start with search if present, else create empty df with date range from available datasets
    frames = []
    for x in [search_df, youtube_df, tiktok_df, screen_df]:
        if isinstance(x, pd.DataFrame):
            frames.append(x)

    if len(frames) == 0:
        raise RuntimeError("No input datasets found in ./data/. Please add at least one dataset.")

    # Merge progressively
    master = frames[0]
    for f in frames[1:]:
        master = master.merge(f, on="date", how="outer")

    # Sort & fill missing numeric with 0 (we want zscore across actual numbers)
    master = master.sort_values("date").reset_index(drop=True)
    numeric_cols = [c for c in master.columns if c != "date"]
    for c in numeric_cols:
        master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0.0)

    # Compute z-scores per platform column
    zcols = {}
    for c in numeric_cols:
        zcols[c + "_z"] = zscore(master[c])

    zdf = pd.DataFrame(zcols)
    zdf["date"] = master["date"].values
    # Compute HAI = mean of available z-scores (row-wise)
    zonly = [col for col in zdf.columns if col.endswith("_z")]
    zdf["HAI"] = zdf[zonly].mean(axis=1)
    # For user-friendly HAI 0-100 index:
    # scale HAI to 0-100 using min-max on historical HAI
    hai = zdf["HAI"].values
    if len(hai) > 1 and np.nanstd(hai) > 0:
        hai_min, hai_max = np.nanmin(hai), np.nanmax(hai)
        if hai_max - hai_min == 0:
            zdf["HAI_0_100"] = 50 + 10 * zdf["HAI"]  # fallback
        else:
            zdf["HAI_0_100"] = 100 * (zdf["HAI"] - hai_min) / (hai_max - hai_min)
    else:
        zdf["HAI_0_100"] = 50 + 10 * zdf["HAI"]

    # Combine final master with z-scores
    final = master.merge(zdf, on="date", how="left")
    # Keep only relevant columns
    keep_cols = ["date"] + numeric_cols + [c for c in final.columns if c.endswith("_z")] + ["HAI", "HAI_0_100"]
    final = final[keep_cols]
    # Final cleaning: ensure no NaN in HAI
    final["HAI"] = final["HAI"].fillna(0.0)
    final["HAI_0_100"] = final["HAI_0_100"].fillna(50.0)
    return final, msgs

# -------------------------
# Try to build master; display only backend failure messages if any (UI shows friendly messages)
# -------------------------
try:
    master_df, backend_msgs = load_and_build_master()
    backend_ok = True
except Exception as e:
    master_df = None
    backend_msgs = []
    backend_ok = False
    backend_error = e

# If backend failed, show an error and stop
if not backend_ok:
    st.error(f"Backend load failed: {backend_error}")
    st.stop()

# -------------------------
# Navigation (simple top buttons)
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

cols = st.columns([1,1,1,1,6])
with cols[0]:
    if st.button("Home", key="btn_home"):
        st.session_state.page = "Home"
with cols[1]:
    if st.button("Dashboard", key="btn_dashboard"):
        st.session_state.page = "Dashboard"
with cols[2]:
    if st.button("Trend Analysis", key="btn_trend"):
        st.session_state.page = "Trend"
with cols[3]:
    if st.button("Forecast", key="btn_forecast"):
        st.session_state.page = "Forecast"
with cols[4]:
    if st.button("Predict", key="btn_predict"):
        st.session_state.page = "Predict"

page = st.session_state.page

# -------------------------
# HOME PAGE: objective and HOW IT WORKS (concise)
# -------------------------
if page == "Home":
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("<h1 style='color:white; margin:0;'>Global Attention Shift Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='small' style='margin-top:6px; color:#bcd5ea;'>A compact research-grade demo that combines search, video, and device usage signals into a single Human Attention Index (HAI), forecasts short-term attention movement, and classifies trend direction.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Data sources used (platform signals)")
    st.markdown("- Search volume (daily)\n- YouTube views & likes (daily)\n- TikTok plays/views (daily)\n- Screen time (average daily minutes)\n")

    st.markdown("### How the prediction works (simple)")
    st.info("""
    1. Each platform's numeric signal is standardized (z-score).  
    2. We compute HAI = average of platform z-scores (positive = above long-term average, negative = below).  
    3. Forecast: Linear Regression on recent HAI to project next 7 days.  
    4. Classification: Logistic Regression to predict whether attention will RISE or FALL tomorrow.  
    """)
    st.markdown("Click **Predict** in the top menu to run a one-click active prediction on the latest available day.")

# -------------------------
# DASHBOARD
# -------------------------
if page == "Dashboard":
    st.title("Dashboard")

    df = master_df.copy().sort_values("date")
    df['date_dt'] = pd.to_datetime(df['date'])

    # KPI cards: HAI latest, HAI 7-day % change, Average HAI (you asked to skip the last one)
    latest_row = df.iloc[-1]
    hai_latest = latest_row["HAI"]
    hai_pct7 = None
    if len(df) >= 8:
        prev = df["HAI"].iloc[-8:-1].mean()
        if prev != 0:
            hai_pct7 = 100.0 * (hai_latest - prev) / abs(prev)
    avg_hai = df["HAI"].mean()

    k1, k2, k3 = st.columns([1,1,1])
    with k1:
        st.markdown('<div class="kpi"><h3 style="margin:6px;">HAI (latest)</h3><div style="font-size:28px;">{:.3f}</div><div class="small">raw z-score mean</div></div>'.format(hai_latest), unsafe_allow_html=True)
    with k2:
        pct_text = f"{hai_pct7:.1f}%" if hai_pct7 is not None else "N/A"
        st.markdown('<div class="kpi"><h3 style="margin:6px;">HAI 7-day change</h3><div style="font-size:28px;">{}</div><div class="small">relative to previous week</div></div>'.format(pct_text), unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi"><h3 style="margin:6px;">Avg HAI (history)</h3><div style="font-size:28px;">{:.3f}</div><div class="small">mean HAI</div></div>'.format(avg_hai), unsafe_allow_html=True)

    st.markdown("---")

    # HAI over time chart (last 180 days if many)
    show_df = df.copy()
    if len(show_df) > 180:
        show_df = show_df.iloc[-180:]
    fig = px.line(show_df, x='date_dt', y='HAI', title='HAI over time (normalized z-score)', markers=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(yaxis_title="HAI (z-score)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Platform contribution (normalized z-scores) — stacked idea (shows relative influence)")
    zcols = [c for c in df.columns if c.endswith("_z")]
    if len(zcols) >= 1:
        stacked = df[['date_dt'] + zcols].copy().set_index('date_dt')
        # show last 90 days for clarity
        if len(stacked) > 90:
            stacked = stacked.iloc[-90:]
        stacked_reset = stacked.reset_index().melt(id_vars='date_dt', var_name='platform', value_name='z')
        fig2 = px.area(stacked_reset, x='date_dt', y='z', color='platform', title='Platform z-score contributions (last days)')
        fig2.update_layout(yaxis_title='z-score contribution')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No platform z-score columns available to show contribution.")

    st.markdown("### Correlation between platform signals (Pearson on z-scores)")
    corr_cols = [c for c in df.columns if c.endswith("_z")]
    if len(corr_cols) >= 2:
        # Check for constant columns (zero variance)
        zero_var_cols = [c for c in corr_cols if df[c].std(ddof=0) == 0]
        if len(zero_var_cols) > 0:
            st.warning(f"Note: The following columns have constant values (0 variance), so their correlation is undefined (shown as 0): {', '.join(zero_var_cols)}")

        # Compute correlation and fill NaNs with 0 to avoid broken heatmap
        corr_mat = df[corr_cols].corr().fillna(0)
        
        # enforce -1..1
        fig3 = px.imshow(corr_mat, text_auto=".2f", zmin=-1, zmax=1, title="Correlation matrix (z-scores)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Not enough platform columns to compute correlation.")

# -------------------------
# TREND ANALYSIS
# -------------------------
if page == "Trend":
    st.title("Daily HAI Trend")
    df = master_df.copy().sort_values("date")
    df['date_dt'] = pd.to_datetime(df['date'])
    window = st.slider("Smoothing window (days)", 1, 21, 7)
    df['HAI_smooth'] = df['HAI'].rolling(window=window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date_dt'], y=df['HAI'], mode='lines+markers', name='HAI'))
    fig.add_trace(go.Scatter(x=df['date_dt'], y=df['HAI_smooth'], mode='lines', name=f'{window}-day rolling mean'))
    fig.update_layout(title="HAI: raw vs smoothed", yaxis_title="HAI (z-score)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Data (last 50 rows):")
    st.dataframe(df[['date', 'HAI', 'HAI_0_100']].tail(50))

# -------------------------
# FORECAST
# -------------------------
if page == "Forecast":
    st.title("HAI Forecast (Linear Regression)")
    df = master_df.copy().sort_values("date").reset_index(drop=True)
    df['date_dt'] = pd.to_datetime(df['date'])
    # Use last N days for training (make it configurable)
    use_days = st.number_input("Use how many recent days to train (min 7)", min_value=7, max_value=len(df), value=min(90, len(df)), step=1)
    train_df = df.iloc[-use_days:].reset_index(drop=True)
    X = np.arange(len(train_df)).reshape(-1,1)
    y = train_df['HAI'].values
    model = LinearRegression()
    model.fit(X, y)
    # Forecast next 7 days
    future_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7)
    future_idx = np.arange(len(train_df), len(train_df) + future_days).reshape(-1,1)
    future_pred = model.predict(future_idx)
    # Build combined frame for plotting
    train_df = train_df.copy()
    train_df['type'] = 'historical'
    future_dates = pd.date_range(start=pd.to_datetime(train_df['date'].iloc[-1]) + pd.Timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({'date': future_dates.date, 'date_dt': future_dates, 'HAI': future_pred, 'type': 'forecast'})
    combined = pd.concat([train_df[['date','date_dt','HAI','type']], future_df], ignore_index=True)
    combined = combined.sort_values('date_dt')
    # Plot
    fig = px.line(combined, x='date_dt', y='HAI', color='type', title=f"HAI: last {use_days} days + {future_days}-day forecast", markers=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    # Forecast table
    st.markdown("### Forecast values")
    st.dataframe(future_df[['date','HAI']].rename(columns={'HAI':'Predicted_HAI'}))

    # Evaluation: use last 14 days as holdout if possible
    if len(train_df) >= 21:
        holdout_days = min(14, len(train_df)//3)
        train_X = np.arange(len(train_df) - holdout_days).reshape(-1,1)
        train_y = train_df['HAI'].iloc[:len(train_df)-holdout_days].values
        test_X = np.arange(len(train_df) - holdout_days, len(train_df)).reshape(-1,1)
        test_y = train_df['HAI'].iloc[len(train_df)-holdout_days:].values
        eval_model = LinearRegression().fit(train_X, train_y)
        ypred = eval_model.predict(test_X)
        mae = mean_absolute_error(test_y, ypred)
        rmse = np.sqrt(mean_squared_error(test_y, ypred))
        # Avoid division by zero in MAPE
        denom = np.where(np.abs(test_y) < 1e-8, 1e-8, np.abs(test_y))
        mape = np.mean(np.abs((test_y - ypred) / denom)) * 100.0
        # Show as small KPI cards
        e1, e2, e3 = st.columns(3)
        e1.metric("MAE", f"{mae:.3f}")
        e2.metric("RMSE", f"{rmse:.3f}")
        e3.metric("MAPE", f"{mape:.2f}%")
        st.markdown("**Model evaluation**: these numbers are computed on a small holdout from the recent training window — lower is better.")
    else:
        st.info("Not enough history to compute a robust holdout evaluation (need >=21 days).")

# -------------------------
# TREND CLASSIFICATION
# -------------------------
if page == "Trend" or page == "Predict" or page == "Dashboard":
    pass  # keep page definitions separate below

if page == "Trend" and False:
    # placeholder if needed later
    pass

if page == "Predict":
    st.title("Predict (one-click + optional manual)")
    df = master_df.copy().sort_values("date").reset_index(drop=True)
    latest = df.iloc[-1]
    st.markdown("#### One-click active prediction (uses latest available data)")
    if st.button("Run prediction (latest data)"):
        # Build features from latest: use the z-score HAI value as numeric and also platform z-scores
        latest_hai = latest['HAI']
        # Classifier: build using historical HAI -> tomorrow direction
        trend = df[['date','HAI']].copy()
        trend['next'] = trend['HAI'].shift(-1)
        trend = trend.dropna()
        trend['trend_target'] = (trend['next'] > trend['HAI']).astype(int)
        X = trend[['HAI']].values
        y = trend['trend_target'].values
        if len(X) < 10:
            st.warning("Not enough history to train classification model (need >=10 days).")
        else:
            clf = LogisticRegression()
            clf.fit(X, y)
            pred_class = clf.predict([[latest_hai]])[0]
            # Predict HAI for next day using simple linear fit on recent HAI
            L = min(30, len(df))
            X_lr = np.arange(L).reshape(-1,1)
            y_lr = df['HAI'].iloc[-L:].values
            lr = LinearRegression().fit(X_lr, y_lr)
            next_pred = lr.predict([[L]])[0]
            # Interpret HAI_0_100
            hai_pct = float(latest['HAI_0_100'])
            if next_pred < -0.5:
                band = "Low attention"
            elif next_pred < 0.5:
                band = "Moderate"
            elif next_pred < 1.5:
                band = "High"
            else:
                band = "Viral-level"
            st.success(f"Predicted next-day HAI (z-score): {next_pred:.3f} — interpreted as **{band}**")
            st.markdown(f"- Classifier: Attention is likely **{'RISING' if pred_class==1 else 'FALLING'}** tomorrow.")
    # Optional manual input
    st.markdown("---")
    st.markdown("#### Manual input (optional)")
    # Let user supply the 4 platform raw values; if they leave blanks we'll use latest values
    c1, c2, c3, c4 = st.columns(4)
    df = master_df.copy().sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    with c1:
        s_in = st.number_input("Search volume (raw)", value=float(last.get("search_volume", 0.0)))
    with c2:
        y_in = st.number_input("YouTube views (raw)", value=float(last.get("youtube_views", 0.0)))
    with c3:
        t_in = st.number_input("TikTok views (raw)", value=float(last.get("tiktok_views", 0.0)))
    with c4:
        st_in = st.number_input("Screen time (min)", value=float(last.get("screen_time_min", 0.0)))
    if st.button("Predict from manual input"):
        # Build a temporary vector, compute z-scores using historical mean/std (from master_df)
        features = []
        zcols = []
        hist = master_df.copy()
        # For each input, use column stats if available, else assume zero mean/std=1
        def to_z(val, colname):
            if colname in hist.columns:
                mu = hist[colname].mean()
                sd = hist[colname].std(ddof=0) if hist[colname].std(ddof=0) != 0 else 1.0
                return (val - mu) / sd
            return 0.0
        zvals = []
        zvals.append(to_z(s_in, "search_volume"))
        zvals.append(to_z(y_in, "youtube_views"))
        zvals.append(to_z(t_in, "tiktok_views"))
        zvals.append(to_z(st_in, "screen_time_min"))
        hai_val = np.mean(zvals)
        # Build classifier similar to above
        trend = df[['date','HAI']].copy()
        trend['next'] = trend['HAI'].shift(-1)
        trend = trend.dropna()
        trend['trend_target'] = (trend['next'] > trend['HAI']).astype(int)
        X = trend[['HAI']].values
        y = trend['trend_target'].values
        if len(X) < 10:
            st.warning("Not enough history to train classification model (need >=10 days).")
        else:
            clf = LogisticRegression()
            clf.fit(X, y)
            pred_class = clf.predict([[hai_val]])[0]
            # linear regression forecast via recent history
            L = min(30, len(df))
            lr = LinearRegression().fit(np.arange(L).reshape(-1,1), df['HAI'].iloc[-L:].values)
            next_pred = lr.predict([[L]])[0]
            if next_pred < -0.5:
                band = "Low attention"
            elif next_pred < 0.5:
                band = "Moderate"
            elif next_pred < 1.5:
                band = "High"
            else:
                band = "Viral-level"
            st.success(f"Manual-pred HAI (z-score): {next_pred:.3f} -> {band}")
            st.markdown(f"- Classifier says: **{'RISING' if pred_class==1 else 'FALLING'}** tomorrow.")

# -------------------------
# Trend Classification page (separate page)
# -------------------------
if page == "Dashboard" and False:
    pass  # placeholder

if st.session_state.page == "Trend Classification":
    st.title("Trend Classification (Rising / Falling)")
    df = master_df.copy().sort_values("date").reset_index(drop=True)
    trend = df[['date','HAI']].copy()
    trend['next'] = trend['HAI'].shift(-1)
    trend = trend.dropna()
    trend['target'] = (trend['next'] > trend['HAI']).astype(int)
    X = trend[['HAI']].values
    y = trend['target'].values
    if len(X) < 10:
        st.warning("Not enough data for classification (need >=10 samples).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LogisticRegression().fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        st.success(f"Accuracy: {acc:.2f}")
        st.markdown("### Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.markdown("### Detailed metrics")
        st.write(f"Precision: {prec:.2f}  |  Recall: {rec:.2f}  |  F1: {f1:.2f}")

# -------------------------
# Ensure pages available in top nav for user's chosen flow
# The above logic uses page strings; to allow direct Trend Classification page,
# expose a small menu button in sidebar to go there.
# -------------------------
st.sidebar.title("Advanced")
if st.sidebar.button("Trend Classification (advanced)"):
    st.session_state.page = "Trend Classification"
    safe_rerun()

# -------------------------
# End of app
# -------------------------