# -*- coding: utf-8 -*-
"""
CPCM — Ticket Source Analysis
Requirements:
  pip install pandas odfpy seaborn plotly
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import re

# --- Config ---
FILE_PATH = "cpcm_logs.ods"  # ganti ke path file kamu
SHEET_NAME = 0  # bisa pakai "Sheet1" atau index 0

# --- Columns mapping (pastikan sesuai dengan header file) ---
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
# Username kolom bisa "Username" atau "username"
POSSIBLE_USERNAME_COLS = ["Username", "username"]

TICKET_COLS = [
    COL_TICKETS_EARNED,
    COL_REDEEM_LOADED,
    COL_MANUAL_LOADED,
    COL_TKT_RECEIPTS,
    COL_TKT_VIA_TX,
]


# =========================
# Data loading & cleansing
# =========================
def load_cpcm(path: str | Path, sheet_name=0) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in [".ods", ".odf", ".odt"]:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="odf")
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)  # auto engine

    # Normalize dtypes
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

    for c in [COL_ACTIVITYTYPE, COL_PRODUCT]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Numeric columns
    numeric_cols = [COL_AMOUNT, COL_TICKETS_REDEEMED, COL_LOYALTY] + TICKET_COLS
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def get_username_col(df: pd.DataFrame) -> Optional[str]:
    for c in POSSIBLE_USERNAME_COLS:
        if c in df.columns:
            return c
    return None


# =========================
# Core aggregations
# =========================
def build_ticket_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    # Safety
    missing = [c for c in TICKET_COLS[:3] + [COL_AMOUNT] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    ticket_summary = (
        df[TICKET_COLS[:3]]  # hanya 3 sumber utama untuk tabel ini
        .sum()
        .rename_axis("Source")
        .reset_index(name="Total Tickets")
    )

    total_topup = float(df[COL_AMOUNT].sum())
    ticket_summary["Total Top Up (IDR)"] = total_topup

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


# =========================
# Static plots (matplotlib + seaborn)
# =========================
def plot_static(ticket_summary: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 5))
    sns.barplot(data=ticket_summary, x="Source", y="Total Tickets", edgecolor="black")
    plt.title("Total Tiket Berdasarkan Sumber")
    plt.ylabel("Total Tickets")
    plt.xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.pie(
        ticket_summary["Total Tickets"],
        labels=ticket_summary["Source"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Distribusi Sumber Tiket Customer")
    plt.tight_layout()
    plt.show()


# =========================
# Interactive plots (plotly)
# =========================
def plot_interactive(ticket_summary: pd.DataFrame):
    import plotly.express as px

    fig_bar = px.bar(
        ticket_summary,
        x="Source",
        y="Total Tickets",
        hover_data=["Share (%)", "Tickets per 1K IDR (overall)", "Total Top Up (IDR)"],
        title="Total Tiket Berdasarkan Sumber (Interactive)",
        text="Total Tickets",
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="Total Tickets")
    fig_bar.show()

    fig_pie = px.pie(
        ticket_summary,
        names="Source",
        values="Total Tickets",
        title="Distribusi Sumber Tiket Customer (Interactive)",
        hover_data=["Share (%)"],
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.show()


# =========================
# Ticket Summary table exporter
# =========================
def export_summary(ticket_summary: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ticket_summary.to_csv(out_path, index=False)
    return out_path


# ============================================================
# ==== LOADTICKETS Set Aggregation (Redemption Currency)  ====
# ============================================================
def extract_set_name(product_name: str) -> str | None:
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


def build_loadtickets_set_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    required = [COL_ACTIVITYTYPE, COL_PRODUCT, COL_REDEEM_LOADED]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for LOADTICKETS aggregation: {missing}")

    tmp = df[df[COL_ACTIVITYTYPE].str.upper().str.strip().eq("LOADTICKETS")].copy()
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


def export_set_aggregation(agg_df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)
    return out_path


def plot_set_aggregation_static(agg_df: pd.DataFrame, top_k: int = 20):
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = agg_df.head(top_k).copy()
    plt.figure(figsize=(10, max(5, int(0.4 * len(data)))))
    sns.barplot(
        data=data, y="SetName", x="Total_Redemption_Currency_Loaded", edgecolor="black"
    )
    plt.title(f"Top {top_k} Set — Total Redemption Currency Loaded")
    plt.xlabel("Total Redemption Currency Loaded")
    plt.ylabel("Set Name")
    plt.tight_layout()
    plt.show()


def plot_set_aggregation_interactive(agg_df: pd.DataFrame, top_k: int = 20):
    import plotly.express as px

    data = agg_df.head(top_k).copy()
    fig = px.bar(
        data,
        y="SetName",
        x="Total_Redemption_Currency_Loaded",
        orientation="h",
        title=f"Top {top_k} Set — Total Redemption Currency Loaded (Interactive)",
        hover_data=["Appearances"],
    )
    fig.update_layout(
        xaxis_title="Total Redemption Currency Loaded", yaxis_title="Set Name"
    )
    fig.show()


# ======================================
# ==== CASHIER ACTIVITY (with TOTAL) ====
# ======================================
def build_cashier_activity_table(
    df: pd.DataFrame, add_total: bool = True
) -> pd.DataFrame:
    """
    Aggregates by Username:
      - Total_Redemption_Currency_Loaded
      - Total_Loyalty_Points
      - Total_Tickets_Redeemed
    Adds a TOTAL row if add_total=True.
    """
    uname_col = get_username_col(df)
    if uname_col is None:
        raise ValueError(
            "No username column found (expected one of: %s)" % POSSIBLE_USERNAME_COLS
        )

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


def export_cashier_activity(agg_df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)
    return out_path


# ======================================
# ==== OVERALL LOG (fraud flagging)  ====
# ======================================
def _decide_flag(idr_per_ticket: float) -> str:
    if pd.isna(idr_per_ticket):
        return "N/A"
    if idr_per_ticket < 20:
        return "Fraud"
    if 20 <= idr_per_ticket <= 40:
        return "Potential Fraud"
    return "Normal"


def print_overall_log(df: pd.DataFrame) -> dict:
    """
    Prints:
      - total top up (sum Amount)
      - total ticket earned (sum of all ticket columns)
      - total ticket redeemed (sum Tickets Redeemed)
      - top up count (count of TRANSACTION with Amount > 0)
      - Flagging based on IDR per ticket (topup / tickets)
      - Assumption (EN) if Flagging is Fraud or Potential Fraud
    Returns dict with the computed metrics.
    """
    # totals
    total_topup = float(df[COL_AMOUNT].sum()) if COL_AMOUNT in df.columns else 0.0
    total_tickets = float(
        df[TICKET_COLS].sum().sum()
    )  # sum across all 5 ticket columns
    total_redeemed = (
        float(df[COL_TICKETS_REDEEMED].sum())
        if COL_TICKETS_REDEEMED in df.columns
        else 0.0
    )

    # top up count: TRANSACTION & Amount > 0
    if COL_ACTIVITYTYPE in df.columns and COL_AMOUNT in df.columns:
        topup_count = int(
            df[
                (df[COL_ACTIVITYTYPE].str.upper().str.strip() == "TRANSACTION")
                & (df[COL_AMOUNT] > 0)
            ].shape[0]
        )
    else:
        topup_count = 0

    idr_per_ticket = (total_topup / total_tickets) if total_tickets > 0 else np.nan
    flag = _decide_flag(idr_per_ticket)

    # for assumption routing
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
                "(e.g., online marketplaces). A commonly referenced price point is about "
                "10,000 tickets for IDR 77,000 (~IDR 7.7 per ticket), implying an abnormally "
                "low effective cost compared with on-site play/top-ups."
            )
        elif redeem_loaded_total > earned_total:
            assumption = (
                "Assumption: Customer likely stockpiles or trades collection card sets and redeems "
                "them in bulk, resulting in a very low effective IDR per ticket."
            )
        elif (
            manual_loaded_total >= max(earned_total, redeem_loaded_total)
            and manual_loaded_total > 0
        ):
            assumption = (
                "Assumption: Manual ticket loads dominate, which can indicate staff/cashier intervention. "
                "Recommend auditing cashier/user access logs and receipt trails for irregularities."
            )
        else:
            assumption = (
                "Assumption: Ticket inflow pattern diverges from typical gameplay behavior. "
                "Further review of load channels and cashier actions is recommended."
            )

    # pretty print
    def fmt_num(x):
        return f"{x:,.0f}" if pd.notna(x) else "NaN"

    print("======================================================")
    print("Overall Summary")
    print(f"total top up           : {fmt_num(total_topup)}")
    print(f"total ticket earned    : {fmt_num(total_tickets)}")
    print(f"total ticket redeemed  : {fmt_num(total_redeemed)}")
    print(f"top up count           : {topup_count}")
    if pd.notna(idr_per_ticket):
        print(f"IDR per ticket         : {idr_per_ticket:,.2f}")
    else:
        print("IDR per ticket         : NaN")
    print(f"Flagging               : {flag}")
    if assumption:
        print(f"Assumption             : {assumption}")
    print("======================================================")

    return {
        "total_topup": total_topup,
        "total_tickets": total_tickets,
        "total_redeemed": total_redeemed,
        "topup_count": topup_count,
        "idr_per_ticket": idr_per_ticket,
        "flag": flag,
        "assumption": assumption,
    }


# =========================
# Main demo
# =========================
if __name__ == "__main__":
    df = load_cpcm(FILE_PATH, sheet_name=SHEET_NAME)

    # ---- OVERALL LOG (muncul PALING ATAS) ----
    _ = print_overall_log(df)

    # ---- Ticket summary (overall) ----
    ticket_summary, total_topup = build_ticket_summary(df)
    print("\n=== Ticket Summary (with Total Top Up) ===")
    print(ticket_summary)

    # Static charts (matplotlib + seaborn)
    plot_static(ticket_summary)

    # Interactive charts (plotly)
    plot_interactive(ticket_summary)

    # (Optional) export table
    export_path = export_summary(ticket_summary, "outputs/ticket_summary.csv")
    print(f"\nSaved: {export_path.resolve()}")

    # ---- LOADTICKETS set aggregation ----
    set_agg = build_loadtickets_set_aggregation(df)
    print("\n=== Aggregasi Set Kartu (Redemption Currency) ===")
    print(set_agg.head(20))  # preview

    plot_set_aggregation_static(set_agg, top_k=20)
    plot_set_aggregation_interactive(set_agg, top_k=20)

    set_export_path = export_set_aggregation(set_agg, "outputs/aggregasi_set_kartu.csv")
    print(f"Saved: {set_export_path.resolve()}")

    # ---- CASHIER ACTIVITY (with TOTAL row) ----
    cashier_tbl = build_cashier_activity_table(df, add_total=True)
    print("\n=== Cashier Activity Summary (with TOTAL) ===")
    print(cashier_tbl)

    cashier_export = export_cashier_activity(
        cashier_tbl, "outputs/cashier_activity_summary.csv"
    )
    print(f"Saved: {cashier_export.resolve()}")
