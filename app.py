import io
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==============
# Helper functions
# ==============

def snake_case(name: str) -> str:
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    return name.strip("_").lower()


def parse_date_from_name(name: str):
    """Try to extract a DD-MM-YYYY date from a filename."""
    m = re.search(r"(\d{2}-\d{2}-\d{4})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d-%m-%Y").date()
    except ValueError:
        return None


def load_uploaded_data(uploaded_files):
    """
    Accepts uploaded_files from st.file_uploader.
    Supports:
      - A single ZIP that contains CSVs
      - Or multiple CSV/Excel files directly
    Returns:
      dfs: dict[name -> DataFrame]
      report_date: date or None
    """
    dfs = {}
    report_date = None

    if not uploaded_files:
        return dfs, report_date

    # Streamlit gives a single file if accept_multiple_files=False
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for upl in uploaded_files:
        filename = upl.name

        # Try to extract report date from filename if not set yet
        if report_date is None:
            rd = parse_date_from_name(filename)
            if rd:
                report_date = rd

        if filename.lower().endswith(".zip"):
            # Read all CSV files inside the zip
            file_bytes = upl.read()
            zf = zipfile.ZipFile(io.BytesIO(file_bytes))
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                raw = zf.read(member)
                df = pd.read_csv(io.BytesIO(raw))

                base = member.split("/")[-1]
                base = base.rsplit(".", 1)[0]
                key = snake_case(base)
                dfs[key] = df
        else:
            # Single CSV or Excel uploaded directly
            if filename.lower().endswith(".csv"):
                df = pd.read_csv(upl)
            else:
                df = pd.read_excel(upl)

            base = filename.rsplit(".", 1)[0]
            key = snake_case(base)
            dfs[key] = df

    return dfs, report_date


# ==============
# Enrichment/modeling
# ==============

def enrich_shop_clicks(df, report_date):
    if df is None:
        return None
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]

    # Expected columns include: itemid, description, totalimpressions, totalclicks, totalctr, usersctr, etc.
    # Split description "SA - muvi Cinemas" -> country, shop_name
    if "description" in df.columns:
        df["country"] = df["description"].str.split(" - ", n=1).str[0]
        df["shop_name"] = df["description"].str.split(" - ", n=1).str[1]

    if report_date is not None:
        df["report_date"] = pd.to_datetime(report_date)

    for col in df.columns:
        if col not in ["itemid", "description", "country", "shop_name", "report_date"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Normalized metric column names
    if "totalimpressions" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalimpressions": "total_impressions"}, inplace=True)
    if "totalclicks" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalclicks": "total_clicks"}, inplace=True)
    if "totalctr" in df.columns and "total_ctr" not in df.columns:
        df.rename(columns={"totalctr": "total_ctr"}, inplace=True)
    if "usersctr" in df.columns and "users_ctr" not in df.columns:
        df.rename(columns={"usersctr": "users_ctr"}, inplace=True)

    # Recompute CTR just in case
    if "total_impressions" in df.columns and "total_clicks" in df.columns:
        df["ctr"] = np.where(
            df["total_impressions"] > 0,
            df["total_clicks"] / df["total_impressions"] * 100,
            np.nan,
        )

    return df


def enrich_collection_clicks(df, report_date):
    if df is None:
        return None
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]

    if report_date is not None:
        df["report_date"] = pd.to_datetime(report_date)

    for col in df.columns:
        if col not in ["country", "city", "location", "listid", "report_date"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "totalimpressions" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalimpressions": "total_impressions"}, inplace=True)
    if "totalclicks" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalclicks": "total_clicks"}, inplace=True)
    if "totalctr" in df.columns and "total_ctr" not in df.columns:
        df.rename(columns={"totalctr": "total_ctr"}, inplace=True)

    if "total_impressions" in df.columns and "total_clicks" in df.columns:
        df["ctr"] = np.where(
            df["total_impressions"] > 0,
            df["total_clicks"] / df["total_impressions"] * 100,
            np.nan,
        )

    return df


def enrich_content_clicks(df, report_date):
    if df is None:
        return None
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]

    if report_date is not None:
        df["report_date"] = pd.to_datetime(report_date)

    for col in df.columns:
        if col not in ["location", "itemtype", "itemid", "description", "report_date"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "totalimpressions" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalimpressions": "total_impressions"}, inplace=True)
    if "totalclicks" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalclicks": "total_clicks"}, inplace=True)
    if "totalctr" in df.columns and "total_ctr" not in df.columns:
        df.rename(columns={"totalctr": "total_ctr"}, inplace=True)

    if "total_impressions" in df.columns and "total_clicks" in df.columns:
        df["ctr"] = np.where(
            df["total_impressions"] > 0,
            df["total_clicks"] / df["total_impressions"] * 100,
            np.nan,
        )
    return df


def enrich_shop_clicks_per_collection(df, report_date):
    if df is None:
        return None
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]

    if report_date is not None:
        df["report_date"] = pd.to_datetime(report_date)

    if "description" in df.columns:
        df["country"] = df["description"].str.split(" - ", n=1).str[0]
        df["shop_name"] = df["description"].str.split(" - ", n=1).str[1]

    for col in df.columns:
        if col not in ["location", "listid", "itemid", "description", "country", "shop_name", "report_date"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "totalimpressions" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalimpressions": "total_impressions"}, inplace=True)
    if "totalclicks" in df.columns and "total_clicks" not in df.columns:
        df.rename(columns={"totalclicks": "total_clicks"}, inplace=True)
    if "totalctr" in df.columns and "total_ctr" not in df.columns:
        df.rename(columns={"totalctr": "total_ctr"}, inplace=True)

    if "total_impressions" in df.columns and "total_clicks" in df.columns:
        df["ctr"] = np.where(
            df["total_impressions"] > 0,
            df["total_clicks"] / df["total_impressions"] * 100,
            np.nan,
        )

    return df


def enrich_search_terms(df, report_date):
    if df is None:
        return None
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]

    if "itemid" in df.columns and "search_term" not in df.columns:
        df.rename(columns={"itemid": "search_term"}, inplace=True)

    if report_date is not None:
        df["report_date"] = pd.to_datetime(report_date)

    for col in df.columns:
        if col not in ["country", "city", "search_term", "report_date"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def enrich_push_notifications(df, report_date):
    if df is None:
        return None
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]

    if "itemid" in df.columns and "campaign" not in df.columns:
        df.rename(columns={"itemid": "campaign"}, inplace=True)

    if report_date is not None:
        df["report_date"] = pd.to_datetime(report_date)

    for col in df.columns:
        if col not in ["action", "campaign", "report_date"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def build_model(dfs, report_date):
    model = {}
    model["shop_clicks"] = enrich_shop_clicks(dfs.get("shop_clicks"), report_date)
    model["collection_clicks"] = enrich_collection_clicks(dfs.get("collection_clicks"), report_date)
    model["content_clicks"] = enrich_content_clicks(dfs.get("content_clicks"), report_date)
    model["shop_clicks_per_collection"] = enrich_shop_clicks_per_collection(
        dfs.get("shop_clicks_per_collection"), report_date
    )
    model["search_terms"] = enrich_search_terms(dfs.get("search_terms"), report_date)
    model["push_notifications"] = enrich_push_notifications(dfs.get("push_notifications"), report_date)
    return model


def compute_shop_kpis(shop_df):
    if shop_df is None or shop_df.empty:
        return {"total_impressions": 0, "total_clicks": 0, "ctr": np.nan}
    total_impressions = shop_df["total_impressions"].sum()
    total_clicks = shop_df["total_clicks"].sum()
    ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else np.nan
    return {
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "ctr": ctr,
    }


# ==============
# Streamlit UI
# ==============

st.set_page_config(
    page_title="Unipal Analytics Dashboard",
    layout="wide",
    page_icon="📊",
)

st.title("Unipal Analytics Dashboard")

st.write(
    "Upload a **daily or monthly ZIP export** from the Unipal analytics system "
    "(e.g. `📊 Daily Analytics Report 30-11-2025.zip`). "
    "This app will parse the raw CSVs inside and build an interactive dashboard."
)

uploaded = st.file_uploader(
    "Upload analytics ZIP or CSV/Excel files",
    type=["zip", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("⬆️ Upload at least one file to begin.")
    st.stop()

with st.spinner("Loading & modeling data..."):
    dfs_raw, report_date = load_uploaded_data(uploaded)
    model = build_model(dfs_raw, report_date)

shop_df = model["shop_clicks"]
collection_df = model["collection_clicks"]
content_df = model["content_clicks"]
shop_coll_df = model["shop_clicks_per_collection"]
search_df = model["search_terms"]
push_df = model["push_notifications"]

if report_date:
    st.caption(f"Report date detected: **{report_date}**")

# ==============
# Sidebar filters
# ==============

st.sidebar.header("Filters")

# Date range (if multiple dates in future)
date_min, date_max = None, None
if shop_df is not None and "report_date" in shop_df.columns:
    date_min = shop_df["report_date"].min()
    date_max = shop_df["report_date"].max()

if date_min is not None and date_max is not None:
    date_sel = st.sidebar.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(date_sel, tuple):
        start_date, end_date = date_sel
    else:
        start_date = end_date = date_sel
else:
    start_date = end_date = None


def filter_by_date(df):
    if df is None or df.empty or start_date is None or end_date is None:
        return df
    if "report_date" not in df.columns:
        return df
    mask = (df["report_date"].dt.date >= start_date) & (df["report_date"].dt.date <= end_date)
    return df[mask]


# Countries / shops

countries = []
shops = []
if shop_df is not None:
    if "country" in shop_df.columns:
        countries = sorted(shop_df["country"].dropna().unique().tolist())
    if "shop_name" in shop_df.columns:
        shops = sorted(shop_df["shop_name"].dropna().unique().tolist())

country_filter = st.sidebar.multiselect("Country", countries)
shop_filter = st.sidebar.multiselect("Shop", shops)

# Locations / collections / content types / search / push
locations = []
collections_list = []
content_types = []
search_terms_list = []
push_actions = []
push_campaigns = []

if collection_df is not None and "location" in collection_df.columns:
    locations = sorted(collection_df["location"].dropna().unique().tolist())
if collection_df is not None and "listid" in collection_df.columns:
    collections_list = sorted(collection_df["listid"].dropna().unique().tolist())
if content_df is not None and "itemtype" in content_df.columns:
    content_types = sorted(content_df["itemtype"].dropna().unique().tolist())
if search_df is not None and "search_term" in search_df.columns:
    search_terms_list = sorted(search_df["search_term"].dropna().unique().tolist())
if push_df is not None and "action" in push_df.columns:
    push_actions = sorted(push_df["action"].dropna().unique().tolist())
if push_df is not None and "campaign" in push_df.columns:
    push_campaigns = sorted(push_df["campaign"].dropna().unique().tolist())

location_filter = st.sidebar.multiselect("Page Location", locations)
collection_filter = st.sidebar.multiselect("Collection / List", collections_list)
content_type_filter = st.sidebar.multiselect("Content Type", content_types)
search_term_filter = st.sidebar.multiselect("Search Term", search_terms_list)
push_action_filter = st.sidebar.multiselect("Push Action", push_actions)
push_campaign_filter = st.sidebar.multiselect("Push Campaign", push_campaigns)


def filter_shop(df):
    if df is None:
        return None
    df = filter_by_date(df)
    if df is None:
        return None
    if country_filter:
        df = df[df["country"].isin(country_filter)]
    if shop_filter:
        df = df[df["shop_name"].isin(shop_filter)]
    return df


def filter_collection(df):
    if df is None:
        return None
    df = filter_by_date(df)
    if df is None:
        return None
    if country_filter and "country" in df.columns:
        df = df[df["country"].isin(country_filter)]
    if location_filter:
        df = df[df["location"].isin(location_filter)]
    if collection_filter:
        df = df[df["listid"].isin(collection_filter)]
    return df


def filter_content(df):
    if df is None:
        return None
    df = filter_by_date(df)
    if df is None:
        return None
    if location_filter and "location" in df.columns:
        df = df[df["location"].isin(location_filter)]
    if content_type_filter and "itemtype" in df.columns:
        df = df[df["itemtype"].isin(content_type_filter)]
    return df


def filter_search(df):
    if df is None:
        return None
    df = filter_by_date(df)
    if df is None:
        return None
    if country_filter and "country" in df.columns:
        df = df[df["country"].isin(country_filter)]
    if search_term_filter:
        df = df[df["search_term"].isin(search_term_filter)]
    return df


def filter_push(df):
    if df is None:
        return None
    df = filter_by_date(df)
    if df is None:
        return None
    if push_action_filter:
        df = df[df["action"].isin(push_action_filter)]
    if push_campaign_filter:
        df = df[df["campaign"].isin(push_campaign_filter)]
    return df


shop_f = filter_shop(shop_df)
collection_f = filter_collection(collection_df)
content_f = filter_content(content_df)
search_f = filter_search(search_df)
push_f = filter_push(push_df)

# ==============
# KPI cards
# ==============

st.subheader("Top-level KPIs (Shop Clicks)")

kpis = compute_shop_kpis(shop_f)
c1, c2, c3 = st.columns(3)
c1.metric("Total Impressions", f"{kpis['total_impressions']:,.0f}")
c2.metric("Total Clicks", f"{kpis['total_clicks']:,.0f}")
ctr_val = kpis["ctr"]
c3.metric("CTR (%)", f"{ctr_val:,.2f}" if not np.isnan(ctr_val) else "N/A")

st.caption("KPIs are based on Shop Clicks after applying the filters on the left.")


# ==============
# Charts
# ==============

# Time series (if multiple dates)
if shop_f is not None and not shop_f.empty and "report_date" in shop_f.columns:
    ts = (
        shop_f.groupby("report_date")[["total_impressions", "total_clicks"]]
        .sum()
        .reset_index()
        .sort_values("report_date")
    )
    st.markdown("### Impressions & Clicks over time (Shop level)")
    fig_ts = px.line(
        ts,
        x="report_date",
        y=["total_impressions", "total_clicks"],
        labels={"value": "Count", "report_date": "Date"},
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# Funnel
if shop_f is not None and not shop_f.empty:
    st.markdown("### Funnel: Impressions → Clicks (Shop level)")
    total_impressions = shop_f["total_impressions"].sum()
    total_clicks = shop_f["total_clicks"].sum()
    funnel_df = pd.DataFrame(
        {"stage": ["Impressions", "Clicks"], "value": [total_impressions, total_clicks]}
    )
    fig_funnel = px.funnel(funnel_df, x="value", y="stage")
    st.plotly_chart(fig_funnel, use_container_width=True)

# Top shops
if shop_f is not None and not shop_f.empty:
    st.markdown("### Top Shops by Clicks")
    top_shops = (
        shop_f.sort_values("total_clicks", ascending=False)
        .head(15)
    )
    fig_shops = px.bar(
        top_shops,
        x="shop_name",
        y="total_clicks",
        color="country" if "country" in top_shops.columns else None,
        hover_data=["total_impressions", "ctr"],
        labels={"shop_name": "Shop", "total_clicks": "Clicks"},
    )
    fig_shops.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_shops, use_container_width=True)

# Collections
if collection_f is not None and not collection_f.empty:
    st.markdown("### Top Collections / Lists by Clicks")
    top_collections = (
        collection_f.sort_values("total_clicks", ascending=False)
        .head(15)
    )
    fig_col = px.bar(
        top_collections,
        x="listid",
        y="total_clicks",
        color="location",
        hover_data=["total_impressions", "ctr"],
        labels={"listid": "Collection / List", "total_clicks": "Clicks"},
    )
    fig_col.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_col, use_container_width=True)

# Content widgets
if content_f is not None and not content_f.empty:
    st.markdown("### Content Widgets / Banners")
    top_content = (
        content_f.sort_values("total_clicks", ascending=False)
        .head(15)
    )
    fig_content = px.bar(
        top_content,
        x="description",
        y="total_clicks",
        color="itemtype" if "itemtype" in top_content.columns else None,
        hover_data=["total_impressions", "ctr"],
        labels={"description": "Content", "total_clicks": "Clicks"},
    )
    fig_content.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_content, use_container_width=True)

# Search terms
if search_f is not None and not search_f.empty:
    st.markdown("### Top Search Terms")
    top_search = (
        search_f.sort_values("total", ascending=False)
        .head(20)
    )
    fig_search = px.bar(
        top_search,
        x="search_term",
        y="total",
        color="country" if "country" in top_search.columns else None,
        labels={"search_term": "Search Term", "total": "Search Count"},
    )
    fig_search.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_search, use_container_width=True)

# Push notifications
if push_f is not None and not push_f.empty:
    st.markdown("### Push Notifications Performance")
    push_agg = (
        push_f.groupby(["action", "campaign"], dropna=False)[["total", "users", "guests", "clients"]]
        .sum()
        .reset_index()
    )
    fig_push = px.bar(
        push_agg,
        x="campaign",
        y="total",
        color="action",
        labels={"campaign": "Campaign", "total": "Events"},
    )
    fig_push.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_push, use_container_width=True)

# Detailed table
st.markdown("### Detailed Shop Data (filtered)")
if shop_f is not None and not shop_f.empty:
    cols_to_show = [c for c in shop_f.columns if c != "itemid"]
    st.dataframe(
        shop_f[cols_to_show].sort_values("total_clicks", ascending=False),
        use_container_width=True,
    )

    csv_bytes = shop_f[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered shop data as CSV",
        data=csv_bytes,
        file_name="unipal_shop_data_filtered.csv",
        mime="text/csv",
    )
else:
    st.write("No shop data after applying filters.")
