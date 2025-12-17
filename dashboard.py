# src/dashboard.py
"""
Professional, clean & aligned Trend Dashboard
- Redesigned layout: aligned charts & tables, consistent KPIs, tidy spacing
- Works with combined_trends.csv produced by src/main.py
"""

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Trend Analysis", page_icon="📊", layout="wide")

# -------------------------
# Styling: clean, minimal
# -------------------------
st.markdown(
    """
    <style>
      /* Page background & default text */
      .stApp { background-color: #f6f8fb; color: #0b1220; }

      /* Header */
      .header { font-size:24px; font-weight:700; color: #0b1220; margin-bottom: 0px; }
      .subheader { color: #4b5563; margin-top: 4px; margin-bottom: 12px; font-size:14px; }

      /* KPI card */
      .kpi {
          background: #ffffff;
          padding: 14px 16px;
          border-radius: 10px;
          box-shadow: 0 6px 18px rgba(13, 30, 42, 0.06);
          color: #0b1220;
      }
      .kpi-value { font-size:20px; font-weight:700; color:#0b1220; }
      .kpi-label { font-size:12px; color:#64748b; margin-top:4px; }

      /* Table container */
      .table-box {
          background: #ffffff;
          padding: 10px;
          border-radius: 10px;
          box-shadow: 0 6px 18px rgba(13, 30, 42, 0.04);
      }

      /* Small caption */
      .small { color:#64748b; font-size:12px; }

      /* Make dataframe heights consistent */
      .stDataFrame > div { background: white !important; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Data load & normalization
# -------------------------
@st.cache_data(ttl=300)
def load_data(path="combined_trends.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp", "ingested_at"], infer_datetime_format=True)
    expected = ["source", "subsource", "title", "text", "metric", "comments", "timestamp", "url", "id", "score", "zscore", "is_anomaly", "ingested_at"]
    for c in expected:
        if c not in df.columns:
            df[c] = None
    df["metric"] = pd.to_numeric(df["metric"].fillna(0), errors="coerce").fillna(0)
    df["score"] = pd.to_numeric(df["score"].fillna(df["metric"]), errors="coerce").fillna(0)
    df["is_anomaly"] = df["is_anomaly"].astype(bool, errors="ignore")
    # derive category
    def _cat(src):
        if isinstance(src, str):
            s = src.lower()
            if "youtube" in s: return "Video"
            if "reddit" in s: return "Post"
        return "Other"
    df["category"] = df["source"].apply(_cat)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True, errors="coerce")
    return df

# load data
try:
    df_raw = load_data("combined_trends.csv")
except FileNotFoundError:
    st.error("combined_trends.csv not found. Run src/main.py first.")
    st.stop()

# -------------------------
# Header & filters
# -------------------------
last_run = df_raw["ingested_at"].dropna().max() if not df_raw["ingested_at"].isna().all() else None

left_h, right_h = st.columns([3, 1])
with left_h:
    st.markdown('<div class="header">Multi-Source Trend Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Reddit posts + YouTube videos — clean, aligned, professional dashboard</div>', unsafe_allow_html=True)
with right_h:
    if last_run is not None and not pd.isna(last_run):
        st.markdown(f"**Last ETL:**  \n{pd.to_datetime(last_run).strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        st.markdown("**Last ETL:**  \nN/A")

st.markdown("---")

# sidebar filters (slicers)
st.sidebar.header("Filters")
sources = st.sidebar.multiselect("Source", options=sorted(df_raw["source"].dropna().unique()), default=sorted(df_raw["source"].dropna().unique()))
categories = st.sidebar.multiselect("Category", options=sorted(df_raw["category"].dropna().unique()), default=sorted(df_raw["category"].dropna().unique()))
sublist = sorted(df_raw["subsource"].dropna().unique())
subsource = st.sidebar.multiselect("Subsource (channel/subreddit)", options=sublist, default=None)
keyword = st.sidebar.text_input("Title contains (keyword)")
hours = st.sidebar.slider("Limit to last N hours (0 = all)", min_value=0, max_value=7*24, value=72)
top_n = st.sidebar.slider("Top N items", min_value=5, max_value=50, value=10)

# apply filters
df = df_raw.copy()
if sources:
    df = df[df["source"].isin(sources)]
if categories:
    df = df[df["category"].isin(categories)]
if subsource:
    df = df[df["subsource"].isin(subsource)]
if keyword:
    df = df[df["title"].str.contains(keyword, case=False, na=False)]
if hours > 0:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=int(hours))
    df = df[df["timestamp"].notna() & (df["timestamp"] >= cutoff)]

# -------------------------
# KPI row (aligned cards)
# -------------------------
total_rows = len(df)
total_metric = int(df["metric"].sum())
total_anoms = int(df["is_anomaly"].sum())

k1, k2, k3, k4 = st.columns([1.2, 1.2, 1.2, 1.0])
with k1:
    st.markdown(f'<div class="kpi"><div class="kpi-value">{total_rows:,}</div><div class="kpi-label">Total items</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="kpi-value">{total_metric:,}</div><div class="kpi-label">Sum metric (views / score)</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div class="kpi-value">{total_anoms:,}</div><div class="kpi-label">Anomalies</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi"><div class="kpi-value">Sample</div><div class="kpi-label">Showing filtered rows</div></div>', unsafe_allow_html=True)

st.markdown("")  # small spacer

# -------------------------
# Main layout: top section
# Left: Time series
# Right: Top trends table + Anomalies table (stacked)
# -------------------------
left_col, right_col = st.columns([2.2, 1.0])

with left_col:
    st.subheader("Time series — top series (aggregated)")
    if df.empty:
        st.info("No data for the current filters.")
    else:
        # robust aggregation: set index and resample grouped by title
        temp = df[["title", "metric", "timestamp"]].dropna(subset=["timestamp"])
        if not temp.empty:
            temp = temp.set_index("timestamp")
            agg_hour = temp.groupby("title")["metric"].resample("1H").sum().reset_index().rename(columns={"metric": "metric_sum"})
            top_titles = df.groupby("title")["metric"].sum().nlargest(top_n).index.tolist()
            plot_df = agg_hour[agg_hour["title"].isin(top_titles)]
            if not plot_df.empty:
                chart = (
                    alt.Chart(plot_df)
                    .mark_line(point=False)
                    .encode(
                        x=alt.X("timestamp:T", title="Time"),
                        y=alt.Y("metric_sum:Q", title="Metric (sum)"),
                        color=alt.Color("title:N", title="Title", legend=alt.Legend(columns=1, orient="right")),
                        tooltip=["title", "timestamp", "metric_sum"]
                    )
                    .interactive()
                    .properties(height=420)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Not enough time series rows to plot. Try increasing the timeframe.")
        else:
            st.info("No timestamped rows available.")

with right_col:
    st.subheader("Top trends (by score)")
    with st.container():
        st.markdown('<div class="table-box">', unsafe_allow_html=True)
        if df.empty:
            st.info("No trends to show.")
        else:
            top_table = df.sort_values("score", ascending=False)[["source", "subsource", "title", "score", "metric", "zscore", "is_anomaly", "timestamp"]].head(top_n)
            st.dataframe(top_table.reset_index(drop=True), height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")  # spacing
    st.subheader("Emerging anomalies")
    with st.container():
        st.markdown('<div class="table-box">', unsafe_allow_html=True)
        anom_table = df[df["is_anomaly"]].sort_values("zscore", key=lambda s: s.abs(), ascending=False)[["source", "subsource", "title", "score", "metric", "zscore", "timestamp"]]
        if anom_table.empty:
            st.info("No anomalies detected for current filters.")
        else:
            st.dataframe(anom_table.head(top_n).reset_index(drop=True), height=260)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Secondary row: Bar chart + Scatter (side-by-side, aligned)
# -------------------------
sec_left, sec_right = st.columns([1.6, 1.0])

with sec_left:
    st.subheader("Top subsources (channels / subreddits)")
    if not df.empty:
        top_subs = df.groupby("subsource")["score"].sum().nlargest(top_n).reset_index()
        bar = (
            alt.Chart(top_subs)
            .mark_bar()
            .encode(
                x=alt.X("score:Q", title="Total score"),
                y=alt.Y("subsource:N", sort="-x", title="Subsource"),
                tooltip=["subsource", "score"]
            ).properties(height=320)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("No data to plot.")

with sec_right:
    st.subheader("Metric over time (scatter)")
    if not df.empty:
        scatter = (
            alt.Chart(df)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("metric:Q", title="Metric"),
                color=alt.Color("source:N"),
                size=alt.Size("metric:Q", scale=alt.Scale(range=[20, 300]), legend=None),
                tooltip=["source", "subsource", "title", "metric", "timestamp"],
            )
            .properties(height=320)
        )
        st.altair_chart(scatter, use_container_width=True)
    else:
        st.info("No data to plot.")

# -------------------------
# Tabs: Raw data + Export
# -------------------------
st.markdown("---")
tab1, tab2 = st.tabs(["Raw data", "Export"])

with tab1:
    st.write("Filtered data sample (first 500 rows)")
    st.dataframe(df.sort_values("ingested_at", ascending=False).head(500), height=420)

with tab2:
    if df.empty:
        st.info("No data to export")
    else:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (filtered)", csv, "trends_filtered.csv", "text/csv")

st.markdown("---")
st.caption("Tip: Use the left-side filters to focus the dashboard. Use Top N to control chart/table density.")
