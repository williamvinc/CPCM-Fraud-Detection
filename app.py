# app.py
# -*- coding: utf-8 -*-
"""
CPCM — Ticket Source/Fraud Analysis (Streamlit, multi-sheet overview)

Run:
  pip install streamlit pandas numpy odfpy openpyxl plotly
  streamlit run app.py
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import base64
import numpy as np
import pandas as pd
import re
import streamlit as st

# ===========================
# Page & basic styling
# ===========================
st.set_page_config(
    page_title="CPCM — Ticket Fraud Analysis",
    page_icon="🎟️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .metric-small .stMetric { padding: 0.25rem 0.5rem; }
      .stDownloadButton, .stButton>button { border-radius: 10px; }
      .tablesmall table td, .tablesmall table th { font-size: 0.9rem; }
      .open-col button { width: 100%; }
      .logo-wrapper { display: flex; justify-content: center; margin: 1.5rem 0; }
      .logo-wrapper.logo-sidebar { justify-content: flex-start; margin: 0 0 1.5rem; }
      .logo-circle { width: 110px; height: 110px; border-radius: 50%; object-fit: cover; border: 2px solid #f0f0f0; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================
# Branding helpers
# ===========================
def render_logo(sidebar: bool = False) -> None:
    logo_path = Path(__file__).with_name("logo.png")
    if not logo_path.exists():
        return
    try:
        encoded_logo = base64.b64encode(logo_path.read_bytes()).decode()
    except Exception:
        return

    container = st.sidebar if sidebar else st
    wrapper_class = "logo-wrapper logo-sidebar" if sidebar else "logo-wrapper"

    container.markdown(
        f"""
        <div class="{wrapper_class}">
            <img src="data:image/png;base64,{encoded_logo}" alt="CPCM logo" class="logo-circle" />
        </div>
        """,
        unsafe_allow_html=True,
    )


render_logo(sidebar=True)


# ===========================
# Simple authentication
# ===========================
VALID_USER_ID = "cpcm"
VALID_PASSWORD = "cpcmcpcm"
AUTH_SESSION_KEY = "authenticated"


def _trigger_rerun() -> bool:
    """Try to rerun the app using any available Streamlit API."""
    for fn_name in ("experimental_rerun", "rerun"):
        rerun_fn = getattr(st, fn_name, None)
        if callable(rerun_fn):
            rerun_fn()
            return True
    return False


def require_login() -> None:
    if st.session_state.get(AUTH_SESSION_KEY, False):
        return

    title_box = st.empty()
    title_box.title("Please Login")
    form_box = st.empty()
    with form_box.form("login_form"):
        user_id = st.text_input("ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if user_id == VALID_USER_ID and password == VALID_PASSWORD:
            st.session_state[AUTH_SESSION_KEY] = True
            form_box.empty()
            title_box.empty()
            st.success("Login Success.")
            if not _trigger_rerun():
                return
        else:
            st.error("Wrong ID or password.")

    st.stop()


# ===========================
# Default column keys (can be remapped in the sidebar)
# ===========================
DEFAULT_COLS = {
    "Date of Activity": "Date of Activity",
    "Activity Site": "Activity Site",
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
    "Customer Name": "Customer Name",
    "Phone Number": "Phone Number",
    "Card Issued Store": "Card Issued Store",
    "Card Number": "Card Number",  # <— added
}


# ===========================
# Formatting helpers (thousand separators everywhere)
# ===========================
def format_number(val: Any, decimals: int = 0) -> str:
    """Return a string with thousands separators; NaN becomes 'NaN'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NaN"
    try:
        fmt = f"{{:,.{decimals}f}}"
        s = fmt.format(float(val))
        return s if decimals > 0 else s.split(".")[0]
    except Exception:
        return str(val)


def format_dataframe_columns(
    df: pd.DataFrame, decimals_map: Dict[str, int]
) -> pd.DataFrame:
    """Return a copy of df where listed columns are formatted as strings with separators."""
    out = df.copy()
    for col, dec in decimals_map.items():
        if col in out.columns:
            out[col] = out[col].apply(lambda x: format_number(x, dec))
    return out


# ===========================
# Core helpers
# ===========================
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
    current = df.copy()
    rename_map = {}
    for expected, existing in mapping.items():
        if existing in current.columns:
            rename_map[existing] = expected
    current = current.rename(columns=rename_map)
    return current


def pick_mode_nonempty(df: pd.DataFrame, col: str) -> Optional[str]:
    """Return most frequent non-empty string for a column (if exists)."""
    if col in df.columns and df[col].notna().any():
        s = df[col].astype(str).str.strip()
        s = s[s != ""]
        if not s.empty:
            try:
                return s.mode(dropna=True).iloc[0]
            except Exception:
                return s.iloc[0]
    return None


# ===========================
# Business logic
# ===========================
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
                "Count": (uname_col, "size"),
            }
        )
        .reset_index()
        .rename(columns={uname_col: "Username"})
        .sort_values(
            [
                "Total_Redemption_Currency_Loaded",
                "Total_Loyalty_Points",
                "Total_Tickets_Redeemed",
                "Count",
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
            "Count": int(agg_cashier["Count"].sum()),
        }
        agg_cashier = pd.concat(
            [agg_cashier, pd.DataFrame([totals])], ignore_index=True
        )
    return agg_cashier


def _decide_flag(value: float, th_fraud: float, th_potential: float) -> str:
    if pd.isna(value):
        return "N/A"
    if value < th_fraud:
        return "Fraud"
    if th_fraud <= value <= th_potential:
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
    th_eff_fraud: float,
    th_eff_potential: float,
) -> Dict[str, Any]:
    total_topup = float(df[COL_AMOUNT].sum()) if COL_AMOUNT in df.columns else 0.0
    total_tickets = float(df[TICKET_COLS_ALL].sum().sum())

    redeemed_raw = (
        float(df[COL_TICKETS_REDEEMED].sum())
        if COL_TICKETS_REDEEMED in df.columns
        else 0.0
    )
    redeemed_sign_fixed = redeemed_raw < 0
    total_redeemed = abs(redeemed_raw)

    ticket_left = total_tickets - total_redeemed

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

    if total_topup > 0:
        value_eff_pct = (total_tickets / total_topup) * 100
    else:
        value_eff_pct = np.nan
    flag_eff = _decide_flag(value_eff_pct, th_eff_fraud, th_eff_potential)

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
                "(e.g., marketplaces). Example: ~10,000 tickets for IDR 77,000 (~IDR 7.7/ticket)."
            )
        elif redeem_loaded_total > earned_total:
            assumption = "Assumption: Customer stockpiles/trades collection card sets and redeems them in bulk."
        elif (
            manual_loaded_total >= max(earned_total, redeem_loaded_total)
        ) and manual_loaded_total > 0:
            assumption = "Assumption: Manual ticket loads dominate; potential staff/cashier intervention."
        else:
            assumption = "Assumption: Ticket inflow diverges from typical gameplay pattern; review load channels."

    return {
        "total_topup": total_topup,
        "total_tickets": total_tickets,
        "total_redeemed": total_redeemed,
        "ticket_left": ticket_left,
        "topup_count": topup_count,
        "idr_per_ticket": idr_per_ticket,
        "value_eff_pct": value_eff_pct,
        "flag": flag,
        "flag_eff": flag_eff,
        "assumption": assumption,
        "redeemed_sign_fixed": redeemed_sign_fixed,
        "th_fraud": th_fraud,
        "th_potential": th_potential,
        "th_eff_fraud": th_eff_fraud,
        "th_eff_potential": th_eff_potential,
    }


# ===========================
# Header detection helpers
# ===========================
TARGET_COL = "Card Issued Store"


def _norm_text(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\u00a0", " ")
    s = " ".join(s.split())
    return s


def _find_header_index(df: pd.DataFrame, target_col: str) -> Optional[int]:
    t = _norm_text(target_col).lower()
    for i in range(len(df)):
        row = df.iloc[i].apply(_norm_text).str.lower()
        if (row == t).any():
            return i
    return None


def _reheader(df_raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    header_idx = _find_header_index(df_raw, target_col)
    if header_idx is None:
        return df_raw
    cols = df_raw.iloc[header_idx].map(_norm_text)
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = cols
    return df.dropna(how="all", axis=1)


# ===========================
# Totals helper
# ===========================
def append_total_row(
    df: pd.DataFrame,
    label_col: Optional[str],
    numeric_only: bool = True,
    label_value: str = "TOTAL",
) -> pd.DataFrame:
    if df.empty:
        return df
    df2 = df.copy()
    if numeric_only:
        nums = df2.select_dtypes(include=[np.number]).columns.tolist()
    else:
        nums = [c for c in df2.columns if (label_col is None or c != label_col)]
        for c in nums:
            df2[c] = pd.to_numeric(df2[c], errors="ignore")
        nums = df2.select_dtypes(include=[np.number]).columns.tolist()

    total_row = {c: (df2[c].sum() if c in nums else "") for c in df2.columns}
    if label_col:
        total_row[label_col] = label_value
    df_total = pd.DataFrame([total_row])
    out = pd.concat([df2, df_total], ignore_index=True)
    return out


# ===========================
# Cached loaders & preparation
# ===========================
@st.cache_data(show_spinner=False)
def read_excel_like(
    upload: bytes, engine_hint: Optional[str]
) -> Dict[str, pd.DataFrame]:
    """
    Read a workbook with no header row, detect the header row (the one containing 'Card Issued Store'),
    then drop rows where that column is empty (NaN or whitespace).
    """
    bio = BytesIO(upload)
    if engine_hint == "odf":
        raw = pd.read_excel(bio, sheet_name=None, header=None, engine="odf")
    else:
        raw = pd.read_excel(bio, sheet_name=None, header=None)

    cleaned: Dict[str, pd.DataFrame] = {}
    for sheet_name, df_raw in raw.items():
        df = _reheader(df_raw, TARGET_COL)
        df = df.dropna(how="all")
        if TARGET_COL in df.columns:
            df = df.dropna(subset=[TARGET_COL])
            mask_nonempty = df[TARGET_COL].astype(str).map(_norm_text).str.len() > 0
            df = df[mask_nonempty]
        cleaned[sheet_name] = df.reset_index(drop=True)
    return cleaned


@st.cache_data(show_spinner=False)
def prepare_dataframe(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """Normalize column names, coerce types, and drop suspicious rows."""
    dfx = normalize_columns(df, colmap)

    _to_datetime_safe(dfx, "Date of Activity")
    for c in [
        "ActivityType",
        "Product / Game Name",
        "Customer Name",
        "Phone Number",
        "Card Issued Store",
        "Activity Site",
        "Card Number",  # <— ensure str
    ]:
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

    # Rule: if Amount has value but Card Issued Store is empty -> drop
    if ("Amount" in dfx.columns) and ("Card Issued Store" in dfx.columns):
        mask_bad = dfx["Amount"].notna() & dfx["Card Issued Store"].fillna(
            ""
        ).str.strip().eq("")
        dfx = dfx.loc[not mask_bad if isinstance(mask_bad, bool) else ~mask_bad].copy()

    return dfx


# ===========================
# Sidebar controls
# ===========================
require_login()

st.sidebar.title("Controls")

if st.sidebar.button("Log out"):
    st.session_state.pop(AUTH_SESSION_KEY, None)
    if not _trigger_rerun():
        st.stop()

if st.session_state.get(AUTH_SESSION_KEY, False):
    st.sidebar.caption("Hi There!")

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
        all_sheets_preview = read_excel_like(uploaded.getvalue(), engine_hint)
        sheet_names_preview = list(all_sheets_preview.keys())

        # Show one sample sheet to help map columns
        sample_sheet = sheet_names_preview[0] if sheet_names_preview else None
        if sample_sheet:
            df_preview = all_sheets_preview[sample_sheet]
            with st.sidebar.expander("🔁 Column Mapping (optional)"):
                for expected in DEFAULT_COLS.keys():
                    options = [c for c in df_preview.columns if isinstance(c, str)]
                    current = colmap.get(expected, expected)
                    default_idx = (
                        options.index(current)
                        if current in options
                        else (0 if options else 0)
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
                            f"⚠️ No string-like headers detected for {expected}; keeping default."
                        )
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")

# Fraud threshold tuners
with st.sidebar.expander("🚨 Fraud Thresholds (IDR / ticket & Cost Index %)"):
    th_fraud = st.number_input(
        "IDR/ticket: mark as Fraud if < this value", min_value=0.0, value=20.0, step=1.0
    )
    th_potential = st.number_input(
        "IDR/ticket: Potential Fraud upper bound", min_value=0.0, value=40.0, step=1.0
    )
    th_eff_fraud = st.number_input(
        "Cost Index (%): mark as Fraud if < this value",
        min_value=0.0,
        value=20.0,
        step=50.0,
        help="Cost Index = (Total Tickets / Total Top Up ) × 100",
    )
    th_eff_potential = st.number_input(
        "Cost Index (%): Potential Fraud upper bound",
        min_value=0.0,
        value=40.0,
        step=50.0,
    )
    top_k_sets = st.slider(
        "Top-K Sets for charts", min_value=5, max_value=50, value=20, step=1
    )

# Session state for drilldown
if "selected_sheet" not in st.session_state:
    st.session_state.selected_sheet = None

# ===========================
# UI Head
# ===========================
st.title("CPCM — Ticket Fraud Analysis")
st.markdown(
    "Upload your arcade transaction file (ODS/Excel). The app will:\n"
    "1) **Auto-scan all sheets whose names start with a digit** (e.g., `1CZ`, `8.1 CZ`, `1PV`).\n"
    "2) **Build a summary per sheet** → click **Open ▶** to see **details** identical to the single-sheet view.\n"
)

# ===========================
# Main flow
# ===========================
if uploaded is None:
    st.info("⬆️ Please upload a file to begin.")
    st.stop()

# Read all sheets (cached)
all_sheets = read_excel_like(
    uploaded.getvalue(), "odf" if uploaded.name.lower().endswith(".ods") else None
)

# Keep only sheet names that start with a digit
digit_sheet_names = [sn for sn in all_sheets.keys() if re.match(r"^\s*\d", str(sn))]

if not digit_sheet_names:
    st.warning("No sheets found that start with a digit. Please check the sheet names.")
    st.stop()

# ===========================
# Aliases
# ===========================
COL_DATE = "Date of Activity"
COL_SITE = "Activity Site"
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
COL_CUST_NAME = "Customer Name"
COL_PHONE = "Phone Number"
COL_STORE = "Card Issued Store"
COL_CARDNO = "Card Number"  # <— new alias

TICKET_COLS_ALL = [
    COL_TICKETS_EARNED,
    COL_REDEEM_LOADED,
    COL_MANUAL_LOADED,
    COL_TKT_RECEIPTS,
    COL_TKT_VIA_TX,
]


def _flag_emoji(flag: str) -> str:
    return {
        "Fraud": "🔴 Fraud",
        "Potential Fraud": "🟠 Potential Fraud",
        "Normal": "🟢 Normal",
        "N/A": "⚪ N/A",
    }.get(flag, "⚪ N/A")


# ===========================
# Build per-sheet summary table
# ===========================
prepared_per_sheet: Dict[str, pd.DataFrame] = {}
summary_rows: List[Dict[str, Any]] = []

for sn in digit_sheet_names:
    raw_df = all_sheets[sn]
    try:
        dfn = prepare_dataframe(raw_df, colmap)
        prepared_per_sheet[sn] = dfn

        # pick representative identity values per sheet
        card_no = pick_mode_nonempty(dfn, COL_CARDNO)
        cust_name = pick_mode_nonempty(dfn, COL_CUST_NAME)
        phone_num = pick_mode_nonempty(dfn, COL_PHONE)
        card_store = pick_mode_nonempty(dfn, COL_STORE)

        overall = compute_overall_log(
            df=dfn,
            COL_AMOUNT=COL_AMOUNT,
            TICKET_COLS_ALL=TICKET_COLS_ALL,
            COL_TICKETS_REDEEMED=COL_TICKETS_REDEEMED,
            COL_ACTIVITYTYPE=COL_ACTIVITYTYPE,
            COL_TICKETS_EARNED=COL_TICKETS_EARNED,
            COL_REDEEM_LOADED=COL_REDEEM_LOADED,
            COL_MANUAL_LOADED=COL_MANUAL_LOADED,
            th_fraud=th_fraud,
            th_potential=th_potential,
            th_eff_fraud=th_eff_fraud,
            th_eff_potential=th_eff_potential,
        )

        summary_rows.append(
            {
                "Card Number": card_no or "—",
                "Customer Name": cust_name or "—",
                "Card Issued Store": card_store or "—",
                "IDR per Ticket": overall["idr_per_ticket"],
                "Cost Index (%)": overall["value_eff_pct"],
                "Flagging (IDR/Ticket)": _flag_emoji(overall["flag"]),
                "Cost Index Flag": _flag_emoji(overall["flag_eff"]),
                "Phone Number": phone_num or "—",
                "Sheet": sn,
                "Total Top Up (IDR)": overall["total_topup"],
                "Top Up Count": overall["topup_count"],
                "Total Tickets Inflow": overall["total_tickets"],
                "Tickets Redeemed": overall["total_redeemed"],
                "Possible Customer Ticket Left": overall["ticket_left"],
            }
        )
    except Exception as e:
        summary_rows.append(
            {
                "Card Number": "—",
                "Customer Name": "—",
                "Card Issued Store": "—",
                "IDR per Ticket": np.nan,
                "Cost Index (%)": np.nan,
                "Flagging (IDR/Ticket)": "⚪ N/A",
                "Cost Index Flag": "⚪ N/A",
                "Phone Number": "—",
                "Sheet": sn,
                "Total Top Up (IDR)": np.nan,
                "Top Up Count": np.nan,
                "Total Tickets Inflow": np.nan,
                "Tickets Redeemed": np.nan,
                "Possible Customer Ticket Left": np.nan,
            }
        )
        st.warning(f"Sheet '{sn}' failed to compute: {e}")

summary_df_raw = pd.DataFrame(summary_rows)

# Ensure column order exactly as requested
desired_order = [
    "Card Number",
    "Customer Name",
    "Card Issued Store",
    "IDR per Ticket",
    "Cost Index (%)",
    "Flagging (IDR/Ticket)",
    "Cost Index Flag",
    "Phone Number",
    "Sheet",
    "Total Top Up (IDR)",
    "Top Up Count",
    "Total Tickets Inflow",
    "Tickets Redeemed",
    "Possible Customer Ticket Left",
]
# Reindex with any missing columns appended (defensive)
cols = [c for c in desired_order if c in summary_df_raw.columns] + [
    c for c in summary_df_raw.columns if c not in desired_order
]
summary_df_raw = summary_df_raw.reindex(columns=cols)

# Apply display formatting for numeric columns
summary_df_display = format_dataframe_columns(
    summary_df_raw,
    {
        "IDR per Ticket": 2,
        "Cost Index (%)": 2,
        "Total Top Up (IDR)": 0,
        "Top Up Count": 0,
        "Total Tickets Inflow": 0,
        "Tickets Redeemed": 0,
        "Possible Customer Ticket Left": 0,
    },
)

st.subheader("🧾 Multi-Sheet Summary")
st.caption("Click **Open ▶** on the row you want to drill into.")
st.dataframe(
    summary_df_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Card Number": st.column_config.TextColumn("Card Number", width="medium"),
        "Customer Name": st.column_config.TextColumn("Customer Name", width="medium"),
        "Card Issued Store": st.column_config.TextColumn("Card Issued Store", width="medium"),
        "IDR per Ticket": st.column_config.TextColumn("IDR per Ticket"),
        "Cost Index (%)": st.column_config.TextColumn("Cost Index (%)"),
        "Flagging (IDR/Ticket)": st.column_config.TextColumn("Flagging (IDR/Ticket)"),
        "Cost Index Flag": st.column_config.TextColumn("Cost Index Flag"),
        "Phone Number": st.column_config.TextColumn("Phone Number", width="medium"),
        "Sheet": st.column_config.TextColumn("Sheet", width="small"),
        "Total Top Up (IDR)": st.column_config.TextColumn("Total Top Up (IDR)"),
        "Top Up Count": st.column_config.TextColumn("Top Up Count", width="small"),
        "Total Tickets Inflow": st.column_config.TextColumn("Total Tickets Inflow"),
        "Tickets Redeemed": st.column_config.TextColumn("Tickets Redeemed"),
        "Possible Customer Ticket Left": st.column_config.TextColumn("Possible Customer Ticket Left"),
    },
)

# ===========================
# Multi-sheet: Top metrics per Activity Site (aggregate across all digit sheets)
# ===========================
st.markdown("---")
st.subheader("📊 Cross-Sheet — Top Activity Site Metrics")
st.caption("Aggregate semua sheet (yang diproses di Multi-Sheet Summary).")

# Build per-sheet site-level aggregates and then combine
site_topup_frames = []
site_tickets_frames = []
site_redeemed_frames = []

for sn, dfn in prepared_per_sheet.items():
    if dfn is None or dfn.empty:
        continue

    # Topup (Amount)
    if (COL_SITE in dfn.columns) and (COL_AMOUNT in dfn.columns):
        try:
            s = (
                dfn.groupby(COL_SITE, dropna=False)[COL_AMOUNT]
                .sum()
                .reset_index()
                .rename(columns={COL_AMOUNT: "Total Top Up (IDR)"})
            )
            s["Sheet"] = sn
            site_topup_frames.append(s)
        except Exception:
            pass

    # Tickets Inflow (sum of ticket inflow columns)
    ticket_cols_present_local = [c for c in TICKET_COLS_ALL if c in dfn.columns]
    if (COL_SITE in dfn.columns) and ticket_cols_present_local:
        try:
            t = (
                dfn.groupby(COL_SITE, dropna=False)[ticket_cols_present_local]
                .sum()
                .reset_index()
            )
            # compute total tickets inflow per site
            t["Total Tickets Inflow"] = t[ticket_cols_present_local].sum(axis=1)
            t = t[[COL_SITE, "Total Tickets Inflow"]]
            t["Sheet"] = sn
            site_tickets_frames.append(t)
        except Exception:
            pass

    # Tickets Redeemed (absolute sum)
    if (COL_SITE in dfn.columns) and (COL_TICKETS_REDEEMED in dfn.columns):
        try:
            r = dfn.copy()
            r["Tickets Redeemed (Abs)"] = r[COL_TICKETS_REDEEMED].abs()
            r_agg = (
                r.groupby(COL_SITE, dropna=False)["Tickets Redeemed (Abs)"]
                .sum()
                .reset_index()
                .rename(columns={"Tickets Redeemed (Abs)": "Total Tickets Redeemed"})
            )
            r_agg["Sheet"] = sn
            site_redeemed_frames.append(r_agg)
        except Exception:
            pass

# Combine frames across sheets
def combine_and_sum(frames: List[pd.DataFrame], key_col: str, sum_col: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=[key_col, sum_col])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby(key_col, dropna=False)[sum_col].sum().reset_index()
    combined = combined.sort_values(sum_col, ascending=False).reset_index(drop=True)
    return combined

combined_topup = combine_and_sum(site_topup_frames, COL_SITE, "Total Top Up (IDR)")
combined_tickets = combine_and_sum(site_tickets_frames, COL_SITE, "Total Tickets Inflow")
combined_redeemed = combine_and_sum(site_redeemed_frames, COL_SITE, "Total Tickets Redeemed")

# Display results (tables + charts) — REPLACE the existing expander block with this
with st.expander(f"📈 Cross-Sheet Charts — Top {top_k_sets} Activity Sites"):
    import plotly.express as px

    # tooltips text for cross-sheet charts
    topup_tooltip_cross = "Cabang tempat pelanggan melakukan top-up, walaupun kartu mereka dibuat di cabang lain."
    tickets_tooltip_cross = "Cabang tempat pelanggan menerima tiket, meski kartu mereka dibuat di cabang lain."
    redeem_tooltip_cross = "Cabang tempat pelanggan menukarkan tiket, walaupun kartu mereka berasal dari cabang lain."



    # Helper to show table + chart + download for one metric WITH tooltip header
    def show_metric_table_and_chart_with_tooltip(df_metric: pd.DataFrame, site_col: str, metric_col: str, title: str, tooltip: str, key: str = None):
        if df_metric.empty:
            st.info(f"No data available for **{title}**.")
            return

        # header with tooltip
        st.markdown(
            f'<div style="font-weight:600; display:flex; align-items:center; gap:6px;">'
            f'{title} <span title="{tooltip}" style="cursor:help;">ℹ️</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Safe copy and ensure metric_col exists
        if metric_col not in df_metric.columns:
            st.warning(f"Metric column '{metric_col}' not found for {title}.")
            return

        df_display = format_dataframe_columns(df_metric.head(top_k_sets), {metric_col: 0})
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Bar chart (horizontal)
        fig = px.bar(
            df_metric.head(top_k_sets).copy(),
            x=metric_col,
            y=site_col,
            orientation="h",
            title="",
            text=metric_col,
        )
        fig.update_layout(xaxis_title=metric_col, yaxis_title="Activity Site", yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        # Download CSV
        csv_bytes = df_metric.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download {title} (all sites)",
            data=csv_bytes,
            file_name=f"{title.lower().replace(' ','_')}_all_sites.csv",
            mime="text/csv",
            key=key,
        )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        show_metric_table_and_chart_with_tooltip(
            combined_topup,
            COL_SITE,
            "Total Top Up (IDR)",
            f"Top Topup per Activity Site — Top {top_k_sets} (cross-branch)",
            topup_tooltip_cross,
            key="cross_sheet_topup",
        )

    with col_b:
        show_metric_table_and_chart_with_tooltip(
            combined_tickets,
            COL_SITE,
            "Total Tickets Inflow",
            f"Top Tickets Inflow per Activity Site — Top {top_k_sets} (cross-branch)",
            tickets_tooltip_cross,
            key="cross_sheet_tickets",
        )

    with col_c:
        show_metric_table_and_chart_with_tooltip(
            combined_redeemed,
            COL_SITE,
            "Total Tickets Redeemed",
            f"Tickets Redeemed per Activity Site — Top {top_k_sets} (cross-branch)",
            redeem_tooltip_cross,
            key="cross_sheet_redeemed",
        )


# --- Cross-branch analysis (all sheets combined)
# concat semua prepared sheets menjadi satu DF global
all_df = pd.concat([df for df in prepared_per_sheet.values() if df is not None], ignore_index=True) if prepared_per_sheet else pd.DataFrame()

# normalisasi nama kolom (pastikan tersedia)
for c in [COL_STORE, COL_SITE, COL_AMOUNT, COL_TICKETS_EARNED, COL_REDEEM_LOADED, COL_MANUAL_LOADED, COL_TICKETS_REDEEMED, COL_DATE, COL_CARDNO]:
    if c not in all_df.columns:
        # create empty col to avoid KeyErrors later
        all_df[c] = np.nan

# convert to proper dtypes (prepare_dataframe already did per sheet, but safe)
_to_numeric_safe(all_df, ["Amount", "Tickets Redeemed", "Loyalty Points",
                          "Tickets Earned", "Redemption Currency Loaded",
                          "Tickets Manually Loaded", "Tickets Loaded Via TicketReceipts",
                          "Tickets Loaded Via Transaction"])

# Define cross-branch flag (ignore case + trim)
all_df["Card_Issued_Store_clean"] = all_df[COL_STORE].astype(str).str.strip()
all_df["Activity_Site_clean"] = all_df[COL_SITE].astype(str).str.strip()
all_df["is_cross_branch"] = (
    (~all_df["Card_Issued_Store_clean"].isna())
    & (~all_df["Activity_Site_clean"].isna())
    & (all_df["Card_Issued_Store_clean"] != all_df["Activity_Site_clean"])
)

# Aggregate metrics for cross-branch only
cross_df = all_df[all_df["is_cross_branch"] == True].copy()

# ticket cols present (global)
ticket_cols_present = [c for c in TICKET_COLS_ALL if c in cross_df.columns]

# KPI summary
total_cards = int(all_df[COL_CARDNO].nunique()) if COL_CARDNO in all_df.columns else int(all_df["Card Number"].nunique())
cards_with_cross = int(cross_df[COL_CARDNO].nunique()) if COL_CARDNO in cross_df.columns else int(cross_df["Card Number"].nunique())
pct_cards_cross = 100.0 * cards_with_cross / total_cards if total_cards > 0 else np.nan
total_topup_cross = float(cross_df["Amount"].sum()) if "Amount" in cross_df.columns else 0.0
total_tickets_cross = float(cross_df[ticket_cols_present].sum().sum()) if ticket_cols_present else 0.0

# Top origin stores by outbound cross-branch topup (Top 20)
outbound = (
    cross_df.groupby("Card_Issued_Store_clean", dropna=False)["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Amount": "Total Outbound TopUp (IDR)", "Card_Issued_Store_clean": "Origin Store"})
    .sort_values("Total Outbound TopUp (IDR)", ascending=False)
)
# Top destinations (where non-local cards topup)
inbound_sites = (
    cross_df.groupby("Activity_Site_clean", dropna=False)["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Amount": "Total Inbound TopUp (IDR)", "Activity_Site_clean": "Activity Site"})
    .sort_values("Total Inbound TopUp (IDR)", ascending=False)
)

# Top tickets inflow cross-branch by destination
if ticket_cols_present:
    cross_df["Total Tickets Inflow"] = cross_df[ticket_cols_present].sum(axis=1)
else:
    cross_df["Total Tickets Inflow"] = 0
tickets_by_site = (
    cross_df.groupby("Activity_Site_clean", dropna=False)["Total Tickets Inflow"]
    .sum()
    .reset_index()
    .rename(columns={"Total Tickets Inflow": "Total Tickets Inflow (cross-branch)", "Activity_Site_clean": "Activity Site"})
    .sort_values("Total Tickets Inflow (cross-branch)", ascending=False)
)

# Tickets redeemed by destination (abs)
cross_df["Tickets Redeemed Abs"] = cross_df[COL_TICKETS_REDEEMED].abs()
redeemed_by_site = (
    cross_df.groupby("Activity_Site_clean", dropna=False)["Tickets Redeemed Abs"]
    .sum()
    .reset_index()
    .rename(columns={"Tickets Redeemed Abs": "Total Tickets Redeemed (cross-branch)", "Activity_Site_clean": "Activity Site"})
    .sort_values("Total Tickets Redeemed (cross-branch)", ascending=False)
)

# ---------------------------
# Cross-branch flows, top sites and Sankey (replacement block)
# ---------------------------

# safety: ensure cross_df exists
if "cross_df" not in locals():
    cross_df = pd.DataFrame()

# Top N for flows
topN = int(top_k_sets) if "top_k_sets" in locals() else 20

# Build flow dataframe (Origin -> Destination) by Amount (cross-branch)
flow_df = (
    cross_df.groupby(["Card_Issued_Store_clean", "Activity_Site_clean"], dropna=False)["Amount"]
    .sum()
    .reset_index()
    .rename(columns={
        "Card_Issued_Store_clean": "Origin",
        "Activity_Site_clean": "Destination",
        "Amount": "Value",
    })
)
flow_df = flow_df.sort_values("Value", ascending=False).head(500)  # cap to keep chart responsive

# Top destinations (inbound topup)
inbound_sites = (
    cross_df.groupby("Activity_Site_clean", dropna=False)["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Activity_Site_clean": "Activity Site", "Amount": "Total Inbound TopUp (IDR)"})
    .sort_values("Total Inbound TopUp (IDR)", ascending=False)
)

# Top tickets inflow per destination (cross-branch)
ticket_cols_present = [c for c in TICKET_COLS_ALL if c in cross_df.columns]
if ticket_cols_present:
    cross_df["Total Tickets Inflow (cross-branch)"] = cross_df[ticket_cols_present].sum(axis=1)
else:
    cross_df["Total Tickets Inflow (cross-branch)"] = 0

tickets_by_site = (
    cross_df.groupby("Activity_Site_clean", dropna=False)["Total Tickets Inflow (cross-branch)"]
    .sum()
    .reset_index()
    .rename(columns={"Activity_Site_clean": "Activity Site"})
    .sort_values("Total Tickets Inflow (cross-branch)", ascending=False)
)

# Tickets redeemed by destination (abs)
if COL_TICKETS_REDEEMED in cross_df.columns:
    cross_df["Tickets Redeemed Abs"] = cross_df[COL_TICKETS_REDEEMED].abs()
else:
    cross_df["Tickets Redeemed Abs"] = 0

redeemed_by_site = (
    cross_df.groupby("Activity_Site_clean", dropna=False)["Tickets Redeemed Abs"]
    .sum()
    .reset_index()
    .rename(columns={"Activity_Site_clean": "Activity Site", "Tickets Redeemed Abs": "Total Tickets Redeemed (cross-branch)"})
    .sort_values("Total Tickets Redeemed (cross-branch)", ascending=False)
)

# ---------------------------
# KPIs (reuse existing tooltip renderer if present)
# ---------------------------
st.markdown("---")
st.subheader("🌐 Cross-Branch Overview (All Sheets)")

# if helper _label_with_tooltip exists use it; otherwise define local fallback
if "_label_with_tooltip" in globals():
    _lab = _label_with_tooltip
else:
    def _lab(label: str, tooltip: str):
        html = (
            f'<div style="font-weight:600; display:flex; align-items:center; gap:6px;">'
            f'{label} <span title="{tooltip}" style="cursor:help;">ℹ️</span>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

tt_unique = "Jumlah kartu berbeda yang ditemukan di semua data yang diupload."
tt_cross_cards = "Jumlah kartu yang pernah beraktivitas di cabang lain, bukan di cabang tempat kartu dibuat."
tt_topup = "Total uang top-up yang dilakukan pelanggan di cabang lain, bukan cabang tempat kartu dibuat."
tt_tickets = "Total tiket yang diterima pelanggan di cabang lain, bukan cabang tempat kartu dibuat."

k1, k2, k3, k4 = st.columns(4)
with k1:
    _lab("Unique cards processed", tt_unique)
    st.metric(label="", value=f"{total_cards:,}")
with k2:
    _lab("Cards with any cross-branch activity", tt_cross_cards)
    pct_text = f" ({pct_cards_cross:.2f}%)" if not pd.isna(pct_cards_cross) else ""
    st.metric(label="", value=f"{cards_with_cross:,}{pct_text}")
with k3:
    _lab("Total Top Up (cross-branch)", tt_topup)
    st.metric(label="", value=f"Rp {total_topup_cross:,.0f}")
with k4:
    _lab("Total Tickets Inflow (cross-branch)", tt_tickets)
    st.metric(label="", value=f"{total_tickets_cross:,.0f}")

# ---------------------------
# Cross-Sheet Top-N charts with tooltips
# ---------------------------
with st.expander(f"📊 Cross-Branch Top {topN} (Topup / Tickets / Redeem)"):
    import plotly.express as px

    # helper
    def _show_top_table_chart(df_metric: pd.DataFrame, metric_col: str, title: str, tooltip: str, key: str = None):
        if df_metric is None or df_metric.empty:
            st.info(f"No data for {title}")
            return
        # header + tooltip
        st.markdown(
            f'<div style="font-weight:600; display:flex; align-items:center; gap:6px;">'
            f'{title} <span title="{tooltip}" style="cursor:help;">ℹ️</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        df_display = df_metric.head(topN).copy()
        # format numeric column if present
        if metric_col in df_display.columns:
            df_display = format_dataframe_columns(df_display, {metric_col: 0})
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # chart
        fig = px.bar(
            df_metric.head(topN).copy(),
            x=metric_col,
            y="Activity Site",
            orientation="h",
            text=metric_col,
            title="",
        )
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        # download
        csv_bytes = df_metric.to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇️ Download {title} (all sites)", data=csv_bytes, file_name=f"{title.lower().replace(' ','_')}_all_sites.csv", mime="text/csv", key=key)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        _show_top_table_chart(inbound_sites.rename(columns={"Activity Site": "Activity Site"}), "Total Inbound TopUp (IDR)", f"Top Topup per Activity Site — Top {topN} (cross-branch)", "Cabang tujuan (Activity Site) yang menerima top-up terbesar dari kartu non-local.", key="cross_branch_topup")
    with col_b:
        _show_top_table_chart(tickets_by_site.rename(columns={"Activity Site": "Activity Site"}), "Total Tickets Inflow (cross-branch)", f"Top Tickets Inflow per Activity Site — Top {topN} (cross-branch)", "Cabang tujuan dengan Tickets Inflow teratas dari kartu non-local.", key="cross_branch_tickets")
    with col_c:
        _show_top_table_chart(redeemed_by_site.rename(columns={"Activity Site": "Activity Site"}), "Total Tickets Redeemed (cross-branch)", f"Tickets Redeemed per Activity Site — Top {topN} (cross-branch)", "Cabang tujuan dengan redeem tiket terbanyak oleh kartu non-local.", key="cross_branch_redeemed")

# quick summary
st.write(f"- Unique cards processed: **{total_cards:,}**")
st.write(f"- Cards with any cross-branch activity: **{cards_with_cross:,}** ({pct_cards_cross:.2f}%)")
st.write(f"- Total Top Up (cross-branch): **{total_topup_cross:,.0f} IDR**")
st.write(f"- Total Tickets Inflow (cross-branch): **{total_tickets_cross:,.0f}**")

# ---------------------------
# Sankey (Flow) with tooltip header
# ---------------------------
if not flow_df.empty:
    import plotly.graph_objects as go

    sankey_tooltip = "Diagram ini menunjukkan aliran top-up dari cabang tempat kartu dibuat ke cabang tempat pelanggan melakukan top-up."

    st.markdown(
        f'<div style="font-weight:600; display:flex; align-items:center; gap:6px;">'
        f'Flow: Card Issued Store → Activity Site (by Topup IDR) <span title="{sankey_tooltip}" style="cursor:help;">ℹ️</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    labels = list(pd.unique(flow_df[["Origin", "Destination"]].values.ravel()))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    sankey_source = flow_df["Origin"].map(label_to_idx).tolist()
    sankey_target = flow_df["Destination"].map(label_to_idx).tolist()
    sankey_value = flow_df["Value"].tolist()

    fig_sankey = go.Figure(
        data=go.Sankey(
            node=dict(label=labels, pad=15, thickness=15),
            link=dict(source=sankey_source, target=sankey_target, value=sankey_value),
        )
    )
    fig_sankey.update_layout(title_text="Flow: Card Issued Store → Activity Site (by Topup IDR)", height=600)
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.info("No cross-branch flow data available for Sankey.")


# Per-card sample (top offenders)
# st.markdown("---")
# st.markdown("**Per-card sample (top offenders by amount_cross_branch)**")
# if not card_agg.empty:
#     st.dataframe(
#         format_dataframe_columns(
#             card_agg[["Card Number", "n_tx_total", "n_tx_cross_branch", "amount_total", "amount_cross_branch", "tickets_total", "pct_cross_branch"]].fillna(0),
#             {"amount_total": 0, "amount_cross_branch": 0, "pct_cross_branch": 2, "tickets_total": 0},
#         ),
#         use_container_width=True,
#         hide_index=True,
#     )
# else:
#     st.info("No per-card data available.")

# ===========================
# Faux "clickable row": render Open ▶ buttons
# ===========================
with st.container():
    st.markdown("**Open a sheet:**")
    btn_cols = st.columns(6)
    per_row = 6
    for i, sn in enumerate(summary_df_raw["Sheet"].astype(str).tolist()):
        col = btn_cols[i % per_row]
        if col.button(f"Open ▶ {sn}", key=f"open_{sn}"):
            st.session_state.selected_sheet = sn

# Default to first sheet if none selected
selected_sheet = st.session_state.selected_sheet or summary_df_raw["Sheet"].iloc[0]
st.markdown(f"---\n### 🔎 Detail — **{selected_sheet}**")

# ===========================
# Detail view (single-sheet section)
# ===========================
df = prepared_per_sheet[selected_sheet]

# Username selection (auto-detect + allow override)
detected_username = get_username_col(df, guesses=["Username", "username"])
if detected_username:
    other_cols = [c for c in df.columns if c != detected_username]
    username_col = st.selectbox(
        "Username column",
        options=[detected_username] + other_cols,
        index=0,
        key=f"usercol_{selected_sheet}",
    )
else:
    username_col = st.selectbox(
        "Username column",
        options=[c for c in df.columns],
        key=f"usercol_{selected_sheet}",
    )

# Overall metrics
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
    th_eff_fraud=th_eff_fraud,
    th_eff_potential=th_eff_potential,
)

st.subheader("📌 Overall Summary")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Total Top Up (IDR)", f"{overall['total_topup']:,.0f}")
c2.metric("Total Tickets Inflow", f"{overall['total_tickets']:,.0f}")
c3.metric(
    "Tickets Redeemed",
    f"{overall['total_redeemed']:,.0f}",
    help="Total Ticket Redeemed.",
)
c4.metric(
    "Possible Customer Ticket Left",
    f"{overall['ticket_left']:,.0f}",
    help="Total Tickets Inflow − Tickets Redeemed.",
)
c5.metric("Top Up Count", f"{overall['topup_count']:,.0f}")
c6.metric(
    "IDR per Ticket",
    (
        "NaN"
        if pd.isna(overall["idr_per_ticket"])
        else f"{overall['idr_per_ticket']:,.2f}"
    ),
    help="Total Top Up ÷ Total Tickets Inflow",
)
c7.metric(
    "Cost Index (%)",
    (
        "NaN"
        if pd.isna(overall["value_eff_pct"])
        else f"{overall['value_eff_pct']:,.2f}%"
    ),
    help="(Total Top Up ÷ Total Tickets Inflow) × 100",
)

if overall["ticket_left"] < 0:
    st.warning(
        "Possible Customer Ticket Left is negative. Check for double counting or data issues."
    )

flag_col = _flag_emoji(overall["flag"])
st.markdown(f"**Flagging (IDR/ticket):** {flag_col}")

flag_eff_col = _flag_emoji(overall["flag_eff"])
st.caption(f"Cost Index Flag: {flag_eff_col}")

if overall["assumption"]:
    st.info(overall["assumption"])

# ===========================
# Customer Identity
# ===========================
with st.container():
    st.markdown("**👤 Customer Identity**")
    cust_name = pick_mode_nonempty(df, COL_CUST_NAME)
    phone_num = pick_mode_nonempty(df, COL_PHONE)
    card_store = pick_mode_nonempty(df, COL_STORE)
    st.write(
        f"- **Customer Name:** {cust_name or '—'}  \n"
        f"- **Phone Number:** {phone_num or '—'}  \n"
        f"- **Card Issued Store:** {card_store or '—'}"
    )

# ===========================
# Why is it flagged?
# ===========================
with st.expander("Why is it flagged? (numbers vs thresholds)"):
    st.write(
        f"- **IDR/ticket (actual):** "
        f"{'NaN' if pd.isna(overall['idr_per_ticket']) else format_number(overall['idr_per_ticket'], 2)}  "
        f"→ Fraud if **< {overall['th_fraud']:,.2f}**, Potential if **≤ {overall['th_potential']:,.2f}**"
    )
    st.write(
        f"- **Cost Index % (actual):** "
        f"{'NaN' if pd.isna(overall['value_eff_pct']) else format_number(overall['value_eff_pct'], 2) + '%'}  "
        f"→ Fraud if **< {overall['th_eff_fraud']:,.0f}%**, Potential if **≤ {overall['th_eff_potential']:,.0f}%**"
    )
    st.caption("Tip: Tweak thresholds in the sidebar if the flagging feels too strict.")

# ===========================
# Customer Top-Up History
# ===========================
st.subheader("💳 Customer Top-Up History")
try:
    mask_amount = df[COL_AMOUNT] > 0
    mask_tx = (
        df[COL_ACTIVITYTYPE].astype(str).str.upper().str.strip().eq("TRANSACTION")
        if COL_ACTIVITYTYPE in df.columns
        else True
    )
    df_topup = df[mask_amount & mask_tx].copy()

    keep_cols = [
        c
        for c in [COL_DATE, COL_SITE, COL_AMOUNT, COL_CUST_NAME, COL_PHONE]
        if c in df_topup.columns
    ]
    if not keep_cols:
        raise ValueError("No expected columns for top-up history were found.")

    df_topup = df_topup[keep_cols].sort_values(COL_DATE, ascending=False)
    if COL_AMOUNT in df_topup.columns:
        df_topup[COL_AMOUNT] = df_topup[COL_AMOUNT].round(0)

    if COL_AMOUNT in df_topup.columns:
        label_col = keep_cols[1] if len(keep_cols) > 1 else keep_cols[0]
        df_topup_total = append_total_row(df_topup, label_col=label_col)
        first_text = next((c for c in keep_cols if c != COL_AMOUNT), keep_cols[0])
        df_topup_total.loc[df_topup_total.index[-1], first_text] = "TOTAL"
    else:
        df_topup_total = df_topup

    df_topup_display = format_dataframe_columns(
        df_topup_total, {COL_AMOUNT: 0} if COL_AMOUNT in df_topup_total.columns else {}
    )
    st.dataframe(df_topup_display, use_container_width=True, hide_index=True)

    csv_topup = df_topup_total.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Top-Up History ({selected_sheet})",
        data=csv_topup,
        file_name=f"topup_history_{selected_sheet}.csv",
        mime="text/csv",
    )
except Exception as e:
    st.warning(
        f"Top-Up History unavailable: {e}. Map 'Activity Site', 'Customer Name' & 'Phone Number' in the sidebar if present."
    )

# ===========================
# Ticket Summary (Overall)
# ===========================
st.subheader("🎯 Ticket Summary (Overall)")
ts_df, total_topup = build_ticket_summary(
    df,
    COL_TICKETS_EARNED=COL_TICKETS_EARNED,
    COL_REDEEM_LOADED=COL_REDEEM_LOADED,
    COL_MANUAL_LOADED=COL_MANUAL_LOADED,
    COL_AMOUNT=COL_AMOUNT,
)
overall_tickets = (
    float(ts_df["Total Tickets"].sum()) if "Total Tickets" in ts_df.columns else 0.0
)
overall_tickets_per_1k = (
    ((overall_tickets / total_topup) * 1000) if total_topup > 0 else np.nan
)

ts_total = {
    "Source": "TOTAL",
    "Total Tickets": overall_tickets,
    "Share (%)": 100.0 if overall_tickets > 0 else 0.0,
    "Tickets per 1K IDR (overall)": (
        round(overall_tickets_per_1k, 2)
        if not pd.isna(overall_tickets_per_1k)
        else np.nan
    ),
}
ts_df_total = pd.concat([ts_df, pd.DataFrame([ts_total])], ignore_index=True)

st.caption(f"Total Top Up (IDR): **{total_topup:,.0f}**")

ts_df_display = format_dataframe_columns(
    ts_df_total,
    {"Total Tickets": 0, "Share (%)": 2, "Tickets per 1K IDR (overall)": 2},
)
st.dataframe(ts_df_display, use_container_width=True, hide_index=True)

col_dl1, col_dl2 = st.columns(2)
csv_ts = ts_df_total.to_csv(index=False).encode("utf-8")
col_dl1.download_button(
    f"⬇️ Download Ticket Summary ({selected_sheet})",
    data=csv_ts,
    file_name=f"ticket_summary_{selected_sheet}.csv",
    mime="text/csv",
)

# ===========================
# Interactive charts (Plotly)
# ===========================
with st.expander("🧭 Interactive Charts (Plotly)"):
    import plotly.express as px

    ts_plot = ts_df_total[ts_df_total["Source"] != "TOTAL"].copy()

    fig_bar = px.bar(
        ts_plot,
        x="Source",
        y="Total Tickets",
        hover_data=["Share (%)", "Tickets per 1K IDR (overall)"],
        title=f"Total Tickets by Source — {selected_sheet}",
        text="Total Tickets",
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="Total Tickets")
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(
        ts_plot,
        names="Source",
        values="Total Tickets",
        title=f"Ticket Source Distribution — {selected_sheet}",
        hover_data=["Share (%)"],
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

# ===========================
# Top Tickets Inflow per Activity Site (single sheet) — with tooltip
# ===========================
st.subheader("🏆 Top Tickets Inflow per Activity Site")

_tt_top_tickets = (
    "Cabang yang menerima jumlah Tickets Inflow terbesar untuk kartu pada sheet ini. "
    "Tickets Inflow dihitung sebagai penjumlahan kolom Tickets Earned, Redemption Currency Loaded, "
    "Tickets Manually Loaded, Tickets Loaded Via TicketReceipts, Tickets Loaded Via Transaction."
)

# small helper for inline tooltip label (kept local to this block)
def _inline_label_with_tooltip(text: str, tooltip: str):
    st.markdown(
        f'<div style="font-weight:600; display:flex; align-items:center; gap:8px;">'
        f'{text} <span title="{tooltip}" style="cursor:help;">ℹ️</span></div>',
        unsafe_allow_html=True,
    )

_inline_label_with_tooltip("Top Tickets Inflow per Activity Site", _tt_top_tickets)
st.caption(f"**Card Issued Store:** {card_store or '—'}")

if COL_SITE in df.columns:
    site_agg = (
        df.groupby(COL_SITE, dropna=False)[TICKET_COLS_ALL]
        .sum()
        .reset_index()
    )

    site_agg["Total Tickets Inflow"] = site_agg[TICKET_COLS_ALL].sum(axis=1)
    site_agg = site_agg.sort_values("Total Tickets Inflow", ascending=False)

    format_map = {col: 0 for col in TICKET_COLS_ALL + ["Total Tickets Inflow"]}
    site_agg_display = format_dataframe_columns(site_agg, format_map)

    st.dataframe(
        site_agg_display,
        use_container_width=True,
        hide_index=True,
    )

    # Download CSV
    csv_site = site_agg.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Top Tickets per Activity Site ({selected_sheet})",
        data=csv_site,
        file_name=f"top_tickets_site_{selected_sheet}.csv",
        mime="text/csv",
    )
else:
    st.info("Column 'Activity Site' not found.")


# ===========================
# Top Topup per Activity Site (single sheet) — with tooltip
# ===========================
st.subheader("💸 Top Topup per Activity Site")

_tt_top_topup = (
    "Cabang yang menerima nilai top-up (Amount) terbesar untuk kartu pada sheet ini. "
    "Dihitung SUM(Amount) per Activity Site."
)

_inline_label_with_tooltip("Top Topup per Activity Site", _tt_top_topup)
st.caption(f"**Card Issued Store:** {card_store or '—'}")

if COL_SITE in df.columns and COL_AMOUNT in df.columns:
    try:
        site_topup = (
            df.groupby(COL_SITE, dropna=False)[COL_AMOUNT]
            .sum()
            .reset_index()
            .rename(columns={COL_AMOUNT: "Total Top Up (IDR)"})
        )
        site_topup = site_topup.sort_values("Total Top Up (IDR)", ascending=False)

        site_topup_display = format_dataframe_columns(
            site_topup, {"Total Top Up (IDR)": 0}
        )

        st.dataframe(
            site_topup_display,
            use_container_width=True,
            hide_index=True,
        )

        csv_site_topup = site_topup.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download Top Topup per Activity Site ({selected_sheet})",
            data=csv_site_topup,
            file_name=f"top_topup_site_{selected_sheet}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Top Topup per Activity Site unavailable: {e}")
else:
    st.info("Column 'Activity Site' or 'Amount' not found.")



# ===========================
# Tickets Redeemed per Activity Site (single sheet) — with tooltip
# ===========================
st.subheader("🎟️ Tickets Redeemed per Activity Site")

_tt_redeem = (
    "Cabang tempat redeem (penukaran tiket) terbesar terjadi untuk kartu pada sheet ini. "
    "Dihitung SUM(abs(Tickets Redeemed)) per Activity Site."
)

_inline_label_with_tooltip("Tickets Redeemed per Activity Site", _tt_redeem)
st.caption(f"**Card Issued Store:** {card_store or '—'}")

if COL_SITE in df.columns and COL_TICKETS_REDEEMED in df.columns:
    df_red = df.copy()
    df_red["Tickets Redeemed (Abs)"] = df_red[COL_TICKETS_REDEEMED].abs()

    site_red = (
        df_red.groupby(COL_SITE, dropna=False)["Tickets Redeemed (Abs)"]
        .sum()
        .reset_index()
        .rename(columns={"Tickets Redeemed (Abs)": "Total Tickets Redeemed"})
    )

    site_red = site_red.sort_values("Total Tickets Redeemed", ascending=False)

    site_red_display = format_dataframe_columns(
        site_red,
        {"Total Tickets Redeemed": 0},
    )

    st.dataframe(
        site_red_display,
        use_container_width=True,
        hide_index=True,
    )

    csv_site_red = site_red.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Tickets Redeemed per Activity Site ({selected_sheet})",
        data=csv_site_red,
        file_name=f"tickets_redeemed_site_{selected_sheet}.csv",
        mime="text/csv",
    )
else:
    st.info("Column 'Activity Site' not found.")



# ===========================
# Redemption Set Aggregation (LOADTICKETS)
# ===========================
st.subheader("🧩 Redemption Set Aggregation (LOADTICKETS)")
try:
    set_agg = build_loadtickets_set_aggregation(
        df,
        COL_ACTIVITYTYPE=COL_ACTIVITYTYPE,
        COL_PRODUCT=COL_PRODUCT,
        COL_REDEEM_LOADED=COL_REDEEM_LOADED,
    )

    if not set_agg.empty:
        set_total = {
            "SetName": "TOTAL",
            "Total_Redemption_Currency_Loaded": set_agg[
                "Total_Redemption_Currency_Loaded"
            ].sum(),
            "Appearances": int(set_agg["Appearances"].sum()),
        }
        set_agg_total = pd.concat(
            [set_agg, pd.DataFrame([set_total])], ignore_index=True
        )
    else:
        set_agg_total = set_agg

    set_agg_display = format_dataframe_columns(
        set_agg_total,
        {"Total_Redemption_Currency_Loaded": 0, "Appearances": 0},
    )
    st.dataframe(set_agg_display, use_container_width=True, hide_index=True)

    csv_set = set_agg_total.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Set Aggregation ({selected_sheet})",
        data=csv_set,
        file_name=f"aggregated_sets_{selected_sheet}.csv",
        mime="text/csv",
    )

    with st.expander(f"🧭 Interactive Chart — Top-{top_k_sets} Sets"):
        import plotly.express as px

        data = (
            set_agg[set_agg["SetName"] != "TOTAL"].head(top_k_sets).copy()
            if not set_agg.empty
            else set_agg.copy()
        )
        fig = px.bar(
            data,
            y="SetName",
            x=(
                "Total_Redemption_Currency Loaded"
                if "Total_Redemption_Currency Loaded" in data.columns
                else "Total_Redemption_Currency_Loaded"
            ),
            orientation="h",
            title=f"Top {top_k_sets} Sets — {selected_sheet}",
            hover_data=["Appearances"],
        )
        fig.update_layout(
            xaxis_title="Total Redemption Currency Loaded", yaxis_title="Set Name"
        )
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Set aggregation not available: {e}")

# ===========================
# Cashier Activity (with TOTAL)
# ===========================
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

    cashier_display = format_dataframe_columns(
        cashier_tbl,
        {
            "Total_Redemption_Currency_Loaded": 0,
            "Total_Loyalty_Points": 0,
            "Total_Tickets_Redeemed": 0,
            "Count": 0,
        },
    )

    st.dataframe(
        cashier_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Username": st.column_config.TextColumn("Username", width="medium"),
            "Count": st.column_config.TextColumn("Count"),
        },
    )

    csv_cashier = cashier_tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Cashier Activity ({selected_sheet})",
        data=csv_cashier,
        file_name=f"cashier_activity_summary_{selected_sheet}.csv",
        mime="text/csv",
    )
except Exception as e:
    st.error(f"Cashier activity failed: {e}")

# ===========================
# Raw data (optional)
# ===========================
with st.expander("📄 Raw Data Preview"):
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)

# ===========================
# Footer
# ===========================
st.markdown("---")
st.caption(
    "CPCM Fraud report — created and maintained by William. Adjust thresholds in the sidebar."
)
