# app.py
# -*- coding: utf-8 -*-
"""
CPCM — Ticket Source Analysis (Streamlit)
Run:
  pip install streamlit pandas numpy odfpy openpyxl seaborn plotly matplotlib
  streamlit run app.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import re
import streamlit as st

# ======== Page config ========
st.set_page_config(
    page_title="CPCM — Ticket Source Analysis",
    page_icon="🎟️",
    layout="wide",
)

# ======== Styles (light touch) ========
st.markdown(
    """
    <style>
      .metric-small .stMetric { padding: 0.25rem 0.5rem; }
      .stDownloadButton, .stButton>button { border-radius: 10px; }
      .tablesmall table td, .tablesmall table th { font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======== Default column keys (can be remapped in the sidebar) ========
DEFAULT_COLS = {
    "Date of Activity": "Date of Activity",
    "Amount": "Amount",
    "Tickets Earned": "Tickets Earned",
    "Redemption Currency Loaded": "Redemption Currency Loaded",
    "Tickets Manually Loaded": "Tickets Manually Loaded",
    "Tickets Loaded Via TicketReceipts": "Tickets Loaded Via TicketReceipts",
    "Tickets Loaded Via Transaction": "Tickets Loaded Via Transaction",
    "Tickets Redeemed": "Tickets Redeemed",
    "ActivityType": "ActivityType",
    "Product / Game Name": "Product / Game Name",
    "Loyalty Points": "Loyalty Points",
    "Username": "Username",
}

# ======== Core functions ========


def get_username_col(
    df: pd.DataFrame, guesses: list[str] | None = None
) -> Optional[str]:
    guesses = guesses or ["Username", "username"]
    for c in guesses:
        if c in df.columns:
            return c
    return None


def _to_datetime_safe(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def _to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


def normalize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Ensure the DataFrame has columns exactly as our internal keys expect:
      - keys in `mapping` are expected names; values are current names in the file
    """
    current = df.copy()
    rename_map = {}
    for expected, existing in mapping.items():
        if existing in current.columns:
            rename_map[existing] = expected
    current = current.rename(columns=rename_map)
    return current


def build_ticket_summary(
    df: pd.DataFrame,
    COL_TICKETS_EARNED: str,
    COL_REDEEM_LOADED: str,
    COL_MANUAL_LOADED: str,
    COL_AMOUNT: str,
) -> Tuple[pd.DataFrame, float]:
    TICKET_COLS_MAIN = [COL_TICKETS_EARNED, COL_REDEEM_LOADED, COL_MANUAL_LOADED]
    missing = [c for c in TICKET_COLS_MAIN + [COL_AMOUNT] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    ticket_summary = (
        df[TICKET_COLS_MAIN]
        .sum()
        .rename_axis("Source")
        .reset_index(name="Total Tickets")
    )

    total_topup = float(df[COL_AMOUNT].sum())
    total_tickets_all = float(ticket_summary["Total Tickets"].sum())

    ticket_summary["Share (%)"] = np.where(
        total_tickets_all > 0,
        ticket_summary["Total Tickets"] / total_tickets_all * 100,
        0.0,
    ).round(2)

    ticket_summary["Tickets per 1K IDR (overall)"] = np.where(
        total_topup > 0,
        (ticket_summary["Total Tickets"] / total_topup) * 1000,
        np.nan,
    ).round(2)

    src_map = {
        COL_TICKETS_EARNED: "Tickets Earned",
        COL_REDEEM_LOADED: "Redemption Currency Loaded",
        COL_MANUAL_LOADED: "Tickets Manually Loaded",
    }
    ticket_summary["Source"] = (
        ticket_summary["Source"].map(src_map).fillna(ticket_summary["Source"])
    )

    return ticket_summary, total_topup


def extract_set_name(product_name: str) -> Optional[str]:
    if not isinstance(product_name, str):
        return None
    m = re.search(
        r"Load\s*Tickets\s*-\s*Redemption\s*Currency-(.+)",
        product_name,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def build_loadtickets_set_aggregation(
    df: pd.DataFrame,
    COL_ACTIVITYTYPE: str,
    COL_PRODUCT: str,
    COL_REDEEM_LOADED: str,
) -> pd.DataFrame:
    required = [COL_ACTIVITYTYPE, COL_PRODUCT, COL_REDEEM_LOADED]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for LOADTICKETS aggregation: {missing}")

    tmp = df[
        df[COL_ACTIVITYTYPE].astype(str).str.upper().str.strip().eq("LOADTICKETS")
    ].copy()
    tmp["SetName"] = tmp[COL_PRODUCT].apply(extract_set_name)
    tmp = tmp[tmp["SetName"].notna()].copy()

    agg = (
        tmp.groupby("SetName", dropna=False)
        .agg(
            Total_Redemption_Currency_Loaded=(COL_REDEEM_LOADED, "sum"),
            Appearances=("SetName", "size"),
        )
        .reset_index()
        .sort_values(
            ["Total_Redemption_Currency_Loaded", "Appearances"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    return agg


def build_cashier_activity_table(
    df: pd.DataFrame,
    COL_REDEEM_LOADED: str,
    COL_LOYALTY: str,
    COL_TICKETS_REDEEMED: str,
    username_col_guess: Optional[str] = None,
    add_total: bool = True,
) -> pd.DataFrame:
    uname_col = username_col_guess or get_username_col(df)
    if uname_col is None or uname_col not in df.columns:
        raise ValueError("No username column found (try 'Username' or 'username').")

    required = [uname_col, COL_REDEEM_LOADED, COL_LOYALTY, COL_TICKETS_REDEEMED]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for cashier activity: {missing}")

    agg_cashier = (
        df.groupby(uname_col, dropna=False)
        .agg(
            **{
                "Total_Redemption_Currency_Loaded": (COL_REDEEM_LOADED, "sum"),
                "Total_Loyalty_Points": (COL_LOYALTY, "sum"),
                "Total_Tickets_Redeemed": (COL_TICKETS_REDEEMED, "sum"),
            }
        )
        .reset_index()
        .rename(columns={uname_col: "Username"})
        .sort_values(
            [
                "Total_Redemption_Currency_Loaded",
                "Total_Loyalty_Points",
                "Total_Tickets_Redeemed",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if add_total:
        totals = {
            "Username": "TOTAL",
            "Total_Redemption_Currency_Loaded": agg_cashier[
                "Total_Redemption_Currency_Loaded"
            ].sum(),
            "Total_Loyalty_Points": agg_cashier["Total_Loyalty_Points"].sum(),
            "Total_Tickets_Redeemed": agg_cashier["Total_Tickets_Redeemed"].sum(),
        }
        agg_cashier = pd.concat(
            [agg_cashier, pd.DataFrame([totals])], ignore_index=True
        )

    return agg_cashier


def _decide_flag(idr_per_ticket: float, th_fraud: float, th_potential: float) -> str:
    if pd.isna(idr_per_ticket):
        return "N/A"
    if idr_per_ticket < th_fraud:
        return "Fraud"
    if th_fraud <= idr_per_ticket <= th_potential:
        return "Potential Fraud"
    return "Normal"


def compute_overall_log(
    df: pd.DataFrame,
    COL_AMOUNT: str,
    TICKET_COLS_ALL: list[str],
    COL_TICKETS_REDEEMED: str,
    COL_ACTIVITYTYPE: str,
    COL_TICKETS_EARNED: str,
    COL_REDEEM_LOADED: str,
    COL_MANUAL_LOADED: str,
    th_fraud: float,
    th_potential: float,
) -> Dict[str, Any]:
    total_topup = float(df[COL_AMOUNT].sum()) if COL_AMOUNT in df.columns else 0.0
    total_tickets = float(df[TICKET_COLS_ALL].sum().sum())

    # -- normalize redeemed: always positive (if source stores negatives)
    redeemed_raw = (
        float(df[COL_TICKETS_REDEEMED].sum())
        if COL_TICKETS_REDEEMED in df.columns
        else 0.0
    )
    redeemed_sign_fixed = redeemed_raw < 0
    total_redeemed = abs(redeemed_raw)

    # Possible ticket left uses the normalized redeemed
    ticket_left = total_tickets - total_redeemed

    # top up count
    if COL_ACTIVITYTYPE in df.columns and COL_AMOUNT in df.columns:
        topup_count = int(
            df[
                (
                    df[COL_ACTIVITYTYPE].astype(str).str.upper().str.strip()
                    == "TRANSACTION"
                )
                & (df[COL_AMOUNT] > 0)
            ].shape[0]
        )
    else:
        topup_count = 0

    idr_per_ticket = (total_topup / total_tickets) if total_tickets > 0 else np.nan
    flag = _decide_flag(idr_per_ticket, th_fraud, th_potential)

    earned_total = (
        float(df[COL_TICKETS_EARNED].sum()) if COL_TICKETS_EARNED in df.columns else 0.0
    )
    redeem_loaded_total = (
        float(df[COL_REDEEM_LOADED].sum()) if COL_REDEEM_LOADED in df.columns else 0.0
    )
    manual_loaded_total = (
        float(df[COL_MANUAL_LOADED].sum()) if COL_MANUAL_LOADED in df.columns else 0.0
    )

    assumption = None
    if flag in ("Fraud", "Potential Fraud"):
        if earned_total > redeem_loaded_total:
            assumption = (
                "Assumption: Customer likely acquires tickets from third-party sellers "
                "(e.g., marketplaces). Example reference: ~10,000 tickets for IDR 77,000 "
                "(~IDR 7.7 per ticket), far below on-site cost."
            )
        elif redeem_loaded_total > earned_total:
            assumption = (
                "Assumption: Customer likely stockpiles/trades collection card sets and redeems them "
                "in bulk, yielding very low effective IDR per ticket."
            )
        elif (
            manual_loaded_total >= max(earned_total, redeem_loaded_total)
        ) and manual_loaded_total > 0:
            assumption = (
                "Assumption: Manual ticket loads dominate; potential staff/cashier intervention. "
                "Audit access logs and receipts."
            )
        else:
            assumption = (
                "Assumption: Ticket inflow diverges from typical gameplay pattern; review load channels "
                "and cashier actions."
            )

    return {
        "total_topup": total_topup,
        "total_tickets": total_tickets,
        "total_redeemed": total_redeemed,  # normalized to positive
        "ticket_left": ticket_left,
        "topup_count": topup_count,
        "idr_per_ticket": idr_per_ticket,
        "flag": flag,
        "assumption": assumption,
        "redeemed_sign_fixed": redeemed_sign_fixed,  # True if source used negatives
    }


# ======== Cached loaders ========


@st.cache_data(show_spinner=False)
def read_excel_like(
    upload: bytes, engine_hint: Optional[str]
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of sheet_name -> DataFrame (like pd.read_excel(sheet_name=None)).
    engine_hint: 'odf' for .ods, otherwise None lets pandas decide.
    """
    bio = BytesIO(upload)
    if engine_hint == "odf":
        xls = pd.read_excel(bio, sheet_name=None, engine="odf")
    else:
        # openpyxl for xlsx/xlsm; xlrd for old xls (if installed)
        xls = pd.read_excel(bio, sheet_name=None)
    return xls


@st.cache_data(show_spinner=False)
def prepare_dataframe(
    df: pd.DataFrame,
    colmap: Dict[str, str],
) -> pd.DataFrame:
    # normalize columns to expected keys
    dfx = normalize_columns(df, colmap)

    # dtypes
    _to_datetime_safe(dfx, "Date of Activity")
    for c in ["ActivityType", "Product / Game Name"]:
        if c in dfx.columns:
            dfx[c] = dfx[c].astype(str)

    numeric_cols = [
        "Amount",
        "Tickets Redeemed",
        "Loyalty Points",
        "Tickets Earned",
        "Redemption Currency Loaded",
        "Tickets Manually Loaded",
        "Tickets Loaded Via TicketReceipts",
        "Tickets Loaded Via Transaction",
    ]
    _to_numeric_safe(dfx, numeric_cols)

    return dfx


# ======== Sidebar: controls ========
st.sidebar.title("⚙️ Controls")

uploaded = st.sidebar.file_uploader(
    "Upload spreadsheet (.ods, .xlsx, .xlsm, .xls)",
    type=["ods", "xlsx", "xlsm", "xls"],
)

sheet_name_choice = None
colmap = DEFAULT_COLS.copy()

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    engine_hint = "odf" if suffix == ".ods" else None
    try:
        all_sheets = read_excel_like(uploaded.getvalue(), engine_hint)
        sheet_names = list(all_sheets.keys())
        sheet_name_choice = st.sidebar.selectbox("Choose sheet", sheet_names, index=0)
        # Column mapper (optional): show only columns present in chosen sheet
        if sheet_name_choice:
            df_preview = all_sheets[sheet_name_choice]
            # Let user map if headers differ
            with st.sidebar.expander("🔁 Column Mapping (optional)"):
                for expected in DEFAULT_COLS.keys():
                    options = [c for c in df_preview.columns if isinstance(c, str)]
                    # keep current mapping if present, else fallback to expected
                    current = colmap.get(expected, expected)
                    default_idx = (
                        options.index(current)
                        if current in options
                        else 0 if options else 0
                    )
                    if options:
                        chosen = st.selectbox(
                            f"{expected}",
                            options,
                            index=default_idx,
                            key=f"map_{expected}",
                        )
                        colmap[expected] = chosen
                    else:
                        st.write(
                            f"⚠️ No string-like headers found for `{expected}`; keep default."
                        )
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")

# Fraud threshold tuners
with st.sidebar.expander("🚨 Fraud Thresholds (IDR / ticket)"):
    th_fraud = st.number_input(
        "Fraud if < this value", min_value=0.0, value=20.0, step=1.0
    )
    th_potential = st.number_input(
        "Potential Fraud upper bound", min_value=0.0, value=40.0, step=1.0
    )

top_k_sets = st.sidebar.slider(
    "Top-K Sets for charts", min_value=5, max_value=50, value=20, step=1
)

st.title("🎟️ CPCM — Ticket Source Analysis")

st.markdown(
    "Upload file transaksi arcade (ODS/Excel), pilih sheet & (opsional) sesuaikan mapping kolom. "
    "App ini menampilkan **Overall Summary**, **Ticket Summary**, **Redemption Set Aggregation**, "
    "dan **Cashier Activity** lengkap dengan **tabel** dan **chart** (static & interactive),"
    "."
)

# ======== Main flow ========
if uploaded is None:
    st.info("⬆️ Upload a file to begin.")
    st.stop()

if sheet_name_choice is None:
    st.warning("Please choose a sheet.")
    st.stop()

raw_df = read_excel_like(
    uploaded.getvalue(), "odf" if uploaded.name.endswith(".ods") else None
)[sheet_name_choice]
df = prepare_dataframe(raw_df, colmap)

# Useful local aliases after normalization
COL_DATE = "Date of Activity"
COL_AMOUNT = "Amount"
COL_TICKETS_EARNED = "Tickets Earned"
COL_REDEEM_LOADED = "Redemption Currency Loaded"
COL_MANUAL_LOADED = "Tickets Manually Loaded"
COL_TKT_RECEIPTS = "Tickets Loaded Via TicketReceipts"
COL_TKT_VIA_TX = "Tickets Loaded Via Transaction"
COL_TICKETS_REDEEMED = "Tickets Redeemed"
COL_ACTIVITYTYPE = "ActivityType"
COL_PRODUCT = "Product / Game Name"
COL_LOYALTY = "Loyalty Points"

TICKET_COLS_ALL = [
    COL_TICKETS_EARNED,
    COL_REDEEM_LOADED,
    COL_MANUAL_LOADED,
    COL_TKT_RECEIPTS,
    COL_TKT_VIA_TX,
]

# Username pick (auto-detect + allow override)
detected_username = get_username_col(df, guesses=["Username", "username"])
username_col = (
    st.selectbox(
        "Username column",
        options=[detected_username]
        + [c for c in df.columns if c not in (detected_username or [])],
        index=0 if detected_username else 0,
    )
    if detected_username
    else st.selectbox("Username column", options=[c for c in df.columns])
)

# ======== Overall Summary (top) ========
overall = compute_overall_log(
    df=df,
    COL_AMOUNT=COL_AMOUNT,
    TICKET_COLS_ALL=TICKET_COLS_ALL,
    COL_TICKETS_REDEEMED=COL_TICKETS_REDEEMED,
    COL_ACTIVITYTYPE=COL_ACTIVITYTYPE,
    COL_TICKETS_EARNED=COL_TICKETS_EARNED,
    COL_REDEEM_LOADED=COL_REDEEM_LOADED,
    COL_MANUAL_LOADED=COL_MANUAL_LOADED,
    th_fraud=th_fraud,
    th_potential=th_potential,
)

st.subheader("📌 Overall Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Total Top Up (IDR)", f"{overall['total_topup']:,.0f}")
c2.metric("Total Tickets Inflow", f"{overall['total_tickets']:,.0f}")

# Tickets Redeemed (always positive) + tooltip
c3.metric(
    "Tickets Redeemed",
    f"{overall['total_redeemed']:,.0f}",
    help="Total Ticket Redeemed.",
)

# Possible left ditempatkan di kanan langsung dari Tickets Redeemed
c4.metric(
    "Possible Customer Ticket left",
    f"{overall['ticket_left']:,.0f}",
    help="Total Tickets Inflow − Tickets Redeemed.",
)

c5.metric("Top Up Count", f"{overall['topup_count']:,d}")
c6.metric(
    "IDR per Ticket",
    (
        "NaN"
        if pd.isna(overall["idr_per_ticket"])
        else f"{overall['idr_per_ticket']:,.2f}"
    ),
    help="Total Top Up ÷ Total Tickets Inflow",
)

# Optional warnings/info
if overall["ticket_left"] < 0:
    st.warning(
        "Possible Customer Ticket left bernilai negatif. Cek kembali data (mis. double counting inflow/redeemed)."
    )

flag_col = {
    "Fraud": "🔴 Fraud",
    "Potential Fraud": "🟠 Potential Fraud",
    "Normal": "🟢 Normal",
    "N/A": "⚪ N/A",
}[overall["flag"]]
st.markdown(f"**Flagging:** {flag_col}")
if overall["assumption"]:
    st.info(overall["assumption"])

# ======== Ticket Summary ========
st.subheader("🎯 Ticket Summary (Overall)")
ts_df, total_topup = build_ticket_summary(
    df,
    COL_TICKETS_EARNED=COL_TICKETS_EARNED,
    COL_REDEEM_LOADED=COL_REDEEM_LOADED,
    COL_MANUAL_LOADED=COL_MANUAL_LOADED,
    COL_AMOUNT=COL_AMOUNT,
)

st.caption(f"Total Top Up (IDR): **{total_topup:,.0f}**")
st.dataframe(ts_df, use_container_width=True, hide_index=True)

col_dl1, col_dl2 = st.columns(2)
csv_ts = ts_df.to_csv(index=False).encode("utf-8")
col_dl1.download_button(
    "⬇️ Download Ticket Summary (CSV)",
    data=csv_ts,
    file_name="ticket_summary.csv",
    mime="text/csv",
)

# Static (matplotlib + seaborn)
with st.expander("📊 Static Charts (Matplotlib + Seaborn)"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ts_df, x="Source", y="Total Tickets", edgecolor="black", ax=ax1)
    ax1.set_title("Total Tickets by Source")
    ax1.set_xlabel("")
    ax1.set_ylabel("Total Tickets")
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")
    st.pyplot(fig1, use_container_width=False)

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.pie(
        ts_df["Total Tickets"],
        labels=ts_df["Source"],
        autopct="%1.1f%%",
        startangle=140,
    )
    ax2.set_title("Ticket Source Distribution")
    st.pyplot(fig2, use_container_width=False)

# Interactive (Plotly)
with st.expander("🧭 Interactive Charts (Plotly)"):
    import plotly.express as px

    fig_bar = px.bar(
        ts_df,
        x="Source",
        y="Total Tickets",
        hover_data=["Share (%)", "Tickets per 1K IDR (overall)"],
        title="Total Tickets by Source (Interactive)",
        text="Total Tickets",
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="Total Tickets")
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(
        ts_df,
        names="Source",
        values="Total Tickets",
        title="Ticket Source Distribution (Interactive)",
        hover_data=["Share (%)"],
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# ======== Redemption Set Aggregation (LOADTICKETS) ========
st.subheader("🧩 Redemption Set Aggregation (LOADTICKETS)")
try:
    set_agg = build_loadtickets_set_aggregation(
        df,
        COL_ACTIVITYTYPE=COL_ACTIVITYTYPE,
        COL_PRODUCT=COL_PRODUCT,
        COL_REDEEM_LOADED=COL_REDEEM_LOADED,
    )
    st.dataframe(set_agg, use_container_width=True, hide_index=True)
    csv_set = set_agg.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Set Aggregation (CSV)",
        data=csv_set,
        file_name="aggregasi_set_kartu.csv",
        mime="text/csv",
    )

    # Static
    with st.expander("📊 Static Chart — Top-K Sets"):
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = set_agg.head(top_k_sets).copy()
        fig3, ax3 = plt.subplots(figsize=(10, max(5, int(0.4 * len(data)))))
        sns.barplot(
            data=data,
            y="SetName",
            x="Total_Redemption_Currency_Loaded",
            edgecolor="black",
            ax=ax3,
        )
        ax3.set_title(f"Top {top_k_sets} Sets — Total Redemption Currency Loaded")
        ax3.set_xlabel("Total Redemption Currency Loaded")
        ax3.set_ylabel("Set Name")
        st.pyplot(fig3, use_container_width=True)

    # Interactive
    with st.expander("🧭 Interactive Chart — Top-K Sets"):
        import plotly.express as px

        data = set_agg.head(top_k_sets).copy()
        fig = px.bar(
            data,
            y="SetName",
            x="Total_Redemption_Currency_Loaded",
            orientation="h",
            title=f"Top {top_k_sets} Sets — Total Redemption Currency Loaded (Interactive)",
            hover_data=["Appearances"],
        )
        fig.update_layout(
            xaxis_title="Total Redemption Currency Loaded", yaxis_title="Set Name"
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Set aggregation not available: {e}")

# ======== Cashier Activity (with TOTAL) ========
st.subheader("👤 Cashier Activity Summary (with TOTAL)")
try:
    cashier_tbl = build_cashier_activity_table(
        df,
        COL_REDEEM_LOADED=COL_REDEEM_LOADED,
        COL_LOYALTY=COL_LOYALTY,
        COL_TICKETS_REDEEMED=COL_TICKETS_REDEEMED,
        username_col_guess=username_col,
        add_total=True,
    )
    st.dataframe(
        cashier_tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Username": st.column_config.TextColumn("Username", width="medium"),
        },
    )
    csv_cashier = cashier_tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Cashier Activity (CSV)",
        data=csv_cashier,
        file_name="cashier_activity_summary.csv",
        mime="text/csv",
    )
except Exception as e:
    st.error(f"Cashier activity failed: {e}")

# ======== Raw data (optional) ========
with st.expander("📄 Raw Data Preview"):
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)

# ======== Footer ========
st.markdown("---")
st.caption(
    "CPCM Fraud report - To be update by William "
    "and cashier activity. Adjust fraud thresholds in the sidebar."
)
