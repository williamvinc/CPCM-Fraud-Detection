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
        "Card Issued Store": st.column_config.TextColumn(
            "Card Issued Store", width="medium"
        ),
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
        "Possible Customer Ticket Left": st.column_config.TextColumn(
            "Possible Customer Ticket Left"
        ),
    },
)

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
# Top Tickets Inflow per Activity Site
# ===========================
st.subheader("🏆 Top Tickets Inflow per Activity Site")
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
    st.info(
        "Column 'Activity Site' not found."
    )

# ===========================
# Tickets Redeemed per Activity Site
# ===========================
st.subheader("🎟️ Tickets Redeemed per Activity Site")
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
    "Column 'Activity Site' not found."


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
