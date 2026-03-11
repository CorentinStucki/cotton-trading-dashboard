# ============================================================
# app/Home.py
# Cotton dashboard - simulated preview version
# ============================================================
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import streamlit.components.v1 as components

from data.scoring import (
    classify_score,
    normalize_change_to_signal,
    weighted_composite_score,
)

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Cotton Trading Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0b1020;
    }

    .block-container {
        max-width: 1650px;
        padding-top: 3.5rem;
        padding-bottom: 1.8rem;
    }

    .dashboard-title {
        font-size: 4.2rem;
        font-weight: 800;
        color: #f4f7ff;
        margin-bottom: 0.2rem;
        line-height: 1.05;
        white-space: normal;
        overflow: visible;
        word-break: normal;
    }

    .dashboard-subtitle {
        color: #98a5c3;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f2f5ff;
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
    }

    .section-subtitle {
        color: #99a7c2;
        font-size: 0.88rem;
        margin-bottom: 0.55rem;
    }

    .top-card {
        border-radius: 16px;
        padding: 16px 18px;
        background: linear-gradient(180deg, rgba(21,29,48,0.95), rgba(12,18,34,0.95));
        border: 1px solid rgba(110,130,170,0.18);
        min-height: 108px;
    }

    .top-card-label {
        color: #9ba8c5;
        font-size: 0.84rem;
        margin-bottom: 0.25rem;
    }

    .top-card-value {
        color: #f4f7ff;
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .top-card-sub {
        color: #c9d2e6;
        font-size: 0.92rem;
    }

    .score-bullish {
        color: #59d98e !important;
    }

    .score-bearish {
        color: #ff6b6b !important;
    }

    .score-neutral {
        color: #f1c75b !important;
    }

    .table-card {
        border-radius: 16px;
        padding: 14px 14px 10px 14px;
        background: linear-gradient(180deg, rgba(21,29,48,0.95), rgba(12,18,34,0.95));
        border: 1px solid rgba(110,130,170,0.18);
        margin-bottom: 1rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(110,130,170,0.10);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(255,255,255,0.10);
        margin-top: 1.1rem !important;
        margin-bottom: 1.2rem !important;
    }

    .header-spacer {
        height: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def seeded_rng() -> np.random.Generator:
    """
    Create a deterministic random generator that changes with time,
    so the demo data feels alive but remains stable enough per refresh.
    """
    seed = int(datetime.now().strftime("%Y%m%d%H%M"))
    return np.random.default_rng(seed)


def make_series(
    start: float,
    n: int,
    drift: float,
    vol: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Creates a simple mock time series using a noisy random walk.
    Good enough for a realistic prototype.
    """
    values = [start]
    for _ in range(n - 1):
        step = drift + rng.normal(0, vol)
        values.append(max(0.01, values[-1] + step))
    return np.array(values)


def make_time_index(n: int, freq_minutes: int = 60) -> pd.DatetimeIndex:
    """
    Creates a UTC datetime index.
    """
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=freq_minutes * (n - 1))
    return pd.date_range(start=start, periods=n, freq=f"{freq_minutes}min")


def pct_change(current: float, previous: float) -> float:
    """
    Standard percentage change helper.
    """
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100.0


def signed_arrow(value: float) -> str:
    """
    Arrow helper for directional labels.
    """
    if value > 0:
        return "↑"
    if value < 0:
        return "↓"
    return "→"


def bias_label(value: float) -> str:
    """
    Basic directional label.
    """
    if value > 0:
        return "Bullish"
    if value < 0:
        return "Bearish"
    return "Neutral"


def score_to_intensity(score: float) -> float:
    """
    Maps a score from [-3, +3] into [0, 100].
    """
    clipped = max(-3.0, min(3.0, score))
    return round(((clipped + 3.0) / 6.0) * 100.0, 1)


def format_last(value: float, decimals: int = 2) -> str:
    """
    Clean numeric formatting.
    """
    return f"{value:,.{decimals}f}"


def build_quote_rows(raw_rows: list[dict]) -> pd.DataFrame:
    """
    Creates a standard market monitor table with display columns
    plus hidden helper numeric columns used for styling.
    """
    df = pd.DataFrame(raw_rows)
    return df[["Ticker", "Last Price", "Net", "%1D", "_net_num", "_pct_num"]]

def color_ticker(val):
    """
    Bloomberg-like orange for tickers.
    """
    return "color: #f5a623; font-weight: 600;"


def color_pos_neg(val):
    """
    Green if positive, red if negative, neutral grey if zero / missing.
    """
    try:
        v = float(val)
    except Exception:
        return "color: #dfe6f5;"

    if v > 0:
        return "color: #59d98e; font-weight: 600;"
    elif v < 0:
        return "color: #ff6b6b; font-weight: 600;"
    else:
        return "color: #dfe6f5;"

def color_bias(val):
    """
    Color textual bias values.
    """
    if str(val).lower() == "bullish":
        return "color: #59d98e; font-weight: 600;"
    elif str(val).lower() == "bearish":
        return "color: #ff6b6b; font-weight: 600;"
    else:
        return "color: #f1c75b; font-weight: 600;"


def style_market_table(df: pd.DataFrame):
    """
    Apply Bloomberg-like styling to market monitor tables.
    - Ticker in orange
    - Net / %1D green or red depending on sign
    """
    display_df = df[["Ticker", "Last Price", "Net", "%1D"]].copy()

    styler = display_df.style

    # Bloomberg orange for tickers
    styler = styler.applymap(color_ticker, subset=["Ticker"])

    # Net coloring
    if "_net_num" in df.columns:
        net_colors = [
            color_pos_neg(v) for v in df["_net_num"]
        ]
        styler = styler.apply(
            lambda _: net_colors,
            subset=["Net"],
            axis=0,
        )

    # %1D coloring
    if "_pct_num" in df.columns:
        pct_colors = [
            color_pos_neg(v) for v in df["_pct_num"]
        ]
        styler = styler.apply(
            lambda _: pct_colors,
            subset=["%1D"],
            axis=0,
        )

    styler = styler.format(
        {
            "Last Price": "{:,.1f}",
        },
        na_rep=""
    )

    return styler


def style_market_table_int(df: pd.DataFrame):
    """
    Variant for mostly integer-like market tables.
    """
    display_df = df[["Ticker", "Last Price", "Net", "%1D"]].copy()

    styler = display_df.style

    styler = styler.applymap(color_ticker, subset=["Ticker"])

    if "_net_num" in df.columns:
        net_colors = [
            color_pos_neg(v) for v in df["_net_num"]
        ]
        styler = styler.apply(
            lambda _: net_colors,
            subset=["Net"],
            axis=0,
        )

    if "_pct_num" in df.columns:
        pct_colors = [
            color_pos_neg(v) for v in df["_pct_num"]
        ]
        styler = styler.apply(
            lambda _: pct_colors,
            subset=["%1D"],
            axis=0,
        )

    styler = styler.format(
        {
            "Last Price": "{:,.0f}",
        },
        na_rep=""
    )

    return styler

def format_arrow_value(value: float, decimals: int = 2) -> str:
    """
    Formats numeric variations with an arrow.
    Example: ↑ 1.25 / ↓ -0.84 / → 0.00
    """
    arrow = signed_arrow(value)
    return f"{arrow} {value:.{decimals}f}"


# ------------------------------------------------------------
# STYLE FOR INDICATOR TABLES
# ------------------------------------------------------------
def style_indicator_table(df: pd.DataFrame):
    """
    Styling for Cotton #2 Indicators
    Uses hidden numeric helper columns for color styling while keeping
    display columns with Bloomberg-like arrows.
    """
    display_df = df[["Variable", "Last", "Intensity", "Vs Last Day", "Vs Last Week", "Bias"]].copy()
    styler = display_df.style

    # Color daily variation column using hidden numeric values
    if "_daily_num" in df.columns:
        daily_colors = [color_pos_neg(v) for v in df["_daily_num"]]
        styler = styler.apply(lambda _: daily_colors, subset=["Vs Last Day"], axis=0)

    # Color weekly variation column using hidden numeric values
    if "_weekly_num" in df.columns:
        weekly_colors = [color_pos_neg(v) for v in df["_weekly_num"]]
        styler = styler.apply(lambda _: weekly_colors, subset=["Vs Last Week"], axis=0)

    # Color bias text using hidden bias label
    if "_bias_text" in df.columns:
        bias_colors = [color_bias(v) for v in df["_bias_text"]]
        styler = styler.apply(lambda _: bias_colors, subset=["Bias"], axis=0)

    # Format numeric columns
    styler = styler.format(
        {
            "Last": "{:,.2f}",
            "Intensity": "{:,.0f}",
        }
    )

    return styler

# ------------------------------------------------------------
# STYLE FOR SIGNAL BREAKDOWN
# ------------------------------------------------------------
def style_signal_table(df: pd.DataFrame):
    """
    Styling for Signal Breakdown table
    - green/red signals
    - proper decimal formatting
    """

    styler = (
        df.style
        .applymap(color_pos_neg, subset=["Signal", "Contribution"])
        .format({
            "Signal": "{:+.2f}",
            "Weight": "{:.2f}",
            "Contribution": "{:+.2f}",
        })
    )

    return styler


# ============================================================
# BUILD THE CORE SIMULATED DATASET
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def build_demo_core_dataset():
    rng = seeded_rng()
    idx = make_time_index(72, freq_minutes=60)

    # Core cotton / macro / commodity series used by the model
    cotton = make_series(67.2, len(idx), drift=0.03, vol=0.45, rng=rng)
    dxy = make_series(103.8, len(idx), drift=-0.005, vol=0.08, rng=rng)
    bcom = make_series(135.0, len(idx), drift=0.02, vol=0.18, rng=rng)
    bcomag = make_series(57.0, len(idx), drift=0.01, vol=0.08, rng=rng)
    oil = make_series(100.0, len(idx), drift=0.01, vol=0.55, rng=rng)

    sugar = make_series(15.2, len(idx), drift=0.01, vol=0.10, rng=rng)
    coffee = make_series(300.0, len(idx), drift=0.03, vol=1.0, rng=rng)
    cocoa = make_series(3138.0, len(idx), drift=2.0, vol=45.0, rng=rng)

    corn = make_series(451.0, len(idx), drift=0.04, vol=1.8, rng=rng)
    soy = make_series(1200.0, len(idx), drift=0.07, vol=3.0, rng=rng)
    wheat = make_series(630.0, len(idx), drift=0.05, vol=2.0, rng=rng)

    # Kept internally for score logic
    may_jul = make_series(-0.65, len(idx), drift=0.005, vol=0.06, rng=rng)
    jul_dec = make_series(1.15, len(idx), drift=-0.003, vol=0.07, rng=rng)

    open_int = make_series(99277, len(idx), drift=8, vol=140, rng=rng)
    volume = np.abs(make_series(23000, len(idx), drift=20, vol=1800, rng=rng))

    df = pd.DataFrame(
        {
            "CT1": cotton,
            "DXY": dxy,
            "BCOM": bcom,
            "BCOMAG": bcomag,
            "CL1": oil,
            "SB1": sugar,
            "KC1": coffee,
            "CC1": cocoa,
            "C1": corn,
            "S1": soy,
            "W1": wheat,
            "MAY_JUL": may_jul,
            "JUL_DEC": jul_dec,
            "OPEN_INT": open_int,
            "VOLUME": volume,
        },
        index=idx,
    )

    return df


df = build_demo_core_dataset()

latest = df.iloc[-1]
prev = df.iloc[-2]
week_ago = df.iloc[-6] if len(df) >= 6 else df.iloc[0]

# ============================================================
# COMPOSITE SCORE LOGIC
# ============================================================

signals = {
    "cotton_momentum": normalize_change_to_signal(
        pct_change(latest["CT1"], df["CT1"].iloc[-6]),
        scale=1.8,
    ),
    "spread_structure": normalize_change_to_signal(
        latest["MAY_JUL"] * -12.0,
        scale=1.0,
    ),
    "soft_complex": np.mean(
        [
            normalize_change_to_signal(pct_change(latest["SB1"], df["SB1"].iloc[-6]), scale=2.2),
            normalize_change_to_signal(pct_change(latest["KC1"], df["KC1"].iloc[-6]), scale=2.2),
            normalize_change_to_signal(pct_change(latest["CC1"], df["CC1"].iloc[-6]), scale=2.2),
        ]
    ),
    "agri_complex": np.mean(
        [
            normalize_change_to_signal(pct_change(latest["C1"], df["C1"].iloc[-6]), scale=2.0),
            normalize_change_to_signal(pct_change(latest["S1"], df["S1"].iloc[-6]), scale=2.0),
            normalize_change_to_signal(pct_change(latest["W1"], df["W1"].iloc[-6]), scale=2.0),
        ]
    ),
    "energy": normalize_change_to_signal(
        pct_change(latest["CL1"], df["CL1"].iloc[-6]),
        scale=2.0,
    ),
    "macro": np.mean(
        [
            # Rising DXY is bearish for cotton
            -normalize_change_to_signal(pct_change(latest["DXY"], df["DXY"].iloc[-6]), scale=0.8),
            normalize_change_to_signal(pct_change(latest["BCOM"], df["BCOM"].iloc[-6]), scale=1.5),
            normalize_change_to_signal(pct_change(latest["BCOMAG"], df["BCOMAG"].iloc[-6]), scale=1.2),
        ]
    ),
}

weights = {
    "cotton_momentum": 0.28,
    "spread_structure": 0.20,
    "soft_complex": 0.16,
    "agri_complex": 0.14,
    "energy": 0.12,
    "macro": 0.10,
}

composite_score = weighted_composite_score(signals, weights)
signal_label = classify_score(composite_score)

# ============================================================
# COMPOSITE SCORE VISUAL DIRECTION / COLOR
# ============================================================

if composite_score > 0:
    score_css_class = "score-bullish"
    score_direction = "↑ Bullish"
elif composite_score < 0:
    score_css_class = "score-bearish"
    score_direction = "↓ Bearish"
else:
    score_css_class = "score-neutral"
    score_direction = "→ Neutral"

# ============================================================
# MARKET SNAPSHOT FOR "COTTON #2 INDICATORS"
# ============================================================

softs_snapshot = {
    "Sugar": {
        "last": latest["SB1"],
        "daily_delta": latest["SB1"] - prev["SB1"],
        "weekly_delta": latest["SB1"] - week_ago["SB1"],
    },
    "Coffee": {
        "last": latest["KC1"],
        "daily_delta": latest["KC1"] - prev["KC1"],
        "weekly_delta": latest["KC1"] - week_ago["KC1"],
    },
    "Cocoa": {
        "last": latest["CC1"],
        "daily_delta": latest["CC1"] - prev["CC1"],
        "weekly_delta": latest["CC1"] - week_ago["CC1"],
    },
}

ags_snapshot = {
    "Corn": {
        "last": latest["C1"],
        "daily_delta": latest["C1"] - prev["C1"],
        "weekly_delta": latest["C1"] - week_ago["C1"],
    },
    "Soybeans": {
        "last": latest["S1"],
        "daily_delta": latest["S1"] - prev["S1"],
        "weekly_delta": latest["S1"] - week_ago["S1"],
    },
    "Wheat": {
        "last": latest["W1"],
        "daily_delta": latest["W1"] - prev["W1"],
        "weekly_delta": latest["W1"] - week_ago["W1"],
    },
}


def build_indicator_df(snapshot_dict: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for name, item in snapshot_dict.items():
        direction = "Bullish" if item["weekly_delta"] > 0 else "Bearish" if item["weekly_delta"] < 0 else "Neutral"
        bias_arrow = "↑" if direction == "Bullish" else "↓" if direction == "Bearish" else "→"

        intensity = score_to_intensity(
            normalize_change_to_signal(
                item["weekly_delta"],
                scale=max(abs(item["last"]) * 0.01, 0.5),
            )
        )

        rows.append(
            {
                "Variable": name,
                "Last": round(item["last"], 2),
                "Intensity": intensity,

                # Display versions with arrows
                "Vs Last Day": format_arrow_value(item["daily_delta"], 2),
                "Vs Last Week": format_arrow_value(item["weekly_delta"], 2),
                "Bias": f"{bias_arrow} {direction}",

                # Hidden numeric helper columns for styling
                "_daily_num": round(item["daily_delta"], 2),
                "_weekly_num": round(item["weekly_delta"], 2),
                "_bias_text": direction,
            }
        )

    return pd.DataFrame(rows).sort_values("Intensity", ascending=False)


softs_indicator_df = build_indicator_df(softs_snapshot)
ags_indicator_df = build_indicator_df(ags_snapshot)

# ============================================================
# HELPERS FOR MARKET MONITOR TABLES
# ============================================================

def make_quote(ticker: str, last: float, prev_value: float | None = None, decimals: int = 1) -> dict:
    """
    Standard quote row used in all Bloomberg-style monitor tables.
    Adds arrows for Net and %1D, while keeping numeric helper columns
    for styling.
    """
    if prev_value is None:
        prev_value = last * 0.99

    net = last - prev_value
    pct = pct_change(last, prev_value)

    net_arrow = "↑" if net > 0 else "↓" if net < 0 else "→"
    pct_arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"

    return {
        "Ticker": ticker,
        "Last Price": round(last, decimals),

        # display columns
        "Net": f"{net_arrow} {net:,.{decimals}f}",
        "%1D": f"{pct_arrow} {pct:,.1f}",

        # helper numeric columns for styling
        "_net_num": net,
        "_pct_num": pct,
    }


def simulated_quote_from_base(
    rng: np.random.Generator,
    ticker: str,
    base: float,
    pct_vol: float = 1.8,
    decimals: int = 1,
) -> dict:
    """
    Creates a simulated quote around a base value.
    """
    prev_val = base * (1 + rng.normal(0, pct_vol / 100))
    last_val = base * (1 + rng.normal(0, pct_vol / 100))
    return make_quote(ticker, last_val, prev_val, decimals=decimals)


@st.cache_data(ttl=120, show_spinner=False)
def build_market_monitor_tables():
    """
    Creates Bloomberg-style monitor tables using simulated but coherent data.
    We reuse some values from the core dataset where relevant.
    """
    rng = seeded_rng()

    # --------------------------------------------------------
    # EUROPE / US - broad commodity block
    # --------------------------------------------------------
    broad_rows = [
        make_quote("BCOM", latest["BCOM"], prev["BCOM"], decimals=0),
        make_quote("BCOMAG", latest["BCOMAG"], prev["BCOMAG"], decimals=0),
        simulated_quote_from_base(rng, "XBTUSD", 68787, pct_vol=2.5, decimals=0),
    ]

    # --------------------------------------------------------
    # EUROPE / US - energy
    # --------------------------------------------------------
    energy_rows = [
        make_quote("CL1", latest["CL1"], prev["CL1"], decimals=0),
        simulated_quote_from_base(rng, "CO1", 102, pct_vol=2.0, decimals=0),
        simulated_quote_from_base(rng, "XB1", 294, pct_vol=2.2, decimals=0),
        simulated_quote_from_base(rng, "HO1", 390, pct_vol=2.0, decimals=0),
        simulated_quote_from_base(rng, "QS1", 1233, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "NG1", 3.0, pct_vol=3.0, decimals=1),
        simulated_quote_from_base(rng, "FN1", 153, pct_vol=2.5, decimals=0),
        simulated_quote_from_base(rng, "TZT1", 160, pct_vol=2.5, decimals=0),
    ]

    # --------------------------------------------------------
    # EUROPE / US - metals
    # --------------------------------------------------------
    metals_rows = [
        simulated_quote_from_base(rng, "XAU", 5102, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "XAG", 85, pct_vol=1.7, decimals=0),
        simulated_quote_from_base(rng, "LMAHDS03", 3446, pct_vol=1.8, decimals=0),
        simulated_quote_from_base(rng, "LMCADS03", 12862, pct_vol=1.6, decimals=0),
        simulated_quote_from_base(rng, "LMZSDS03", 3298, pct_vol=1.6, decimals=0),
        simulated_quote_from_base(rng, "LMNIDS03", 17469, pct_vol=1.7, decimals=0),
        simulated_quote_from_base(rng, "LMPBDS03", 1953, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "LMSNDS03", 50065, pct_vol=1.3, decimals=0),
        simulated_quote_from_base(rng, "RBT1", 3115, pct_vol=1.9, decimals=0),
        simulated_quote_from_base(rng, "IOE1", 816, pct_vol=1.7, decimals=0),
        simulated_quote_from_base(rng, "SC01", 103, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "HG1", 569, pct_vol=1.8, decimals=0),
    ]

    # --------------------------------------------------------
    # EUROPE / US - agriculture / softs
    # --------------------------------------------------------
    ag_rows = [
        make_quote("W1", latest["W1"], prev["W1"], decimals=0),
        make_quote("C1", latest["C1"], prev["C1"], decimals=0),
        make_quote("S1", latest["S1"], prev["S1"], decimals=0),
        make_quote("SB1", latest["SB1"], prev["SB1"], decimals=1),
        make_quote("CC1", latest["CC1"], prev["CC1"], decimals=0),
        simulated_quote_from_base(rng, "KO1", 4454, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "LC1", 235, pct_vol=1.4, decimals=0),
        make_quote("KC1", latest["KC1"], prev["KC1"], decimals=0),
        make_quote("CT1", latest["CT1"], prev["CT1"], decimals=0),
    ]

    # --------------------------------------------------------
    # CHINA commodities
    # --------------------------------------------------------
    china_energy_rows = [
        simulated_quote_from_base(rng, "SCP1", 796, pct_vol=2.0, decimals=0),
        simulated_quote_from_base(rng, "F01", 4790, pct_vol=2.0, decimals=0),
        simulated_quote_from_base(rng, "SLS1", 5384, pct_vol=2.0, decimals=0),
    ]

    china_metals_rows = [
        simulated_quote_from_base(rng, "IOE1", 816, pct_vol=1.8, decimals=0),
        simulated_quote_from_base(rng, "RBT1", 3115, pct_vol=1.8, decimals=0),
        simulated_quote_from_base(rng, "ROC1", 3249, pct_vol=1.7, decimals=0),
        simulated_quote_from_base(rng, "SAI1", 21915, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "AUA1", 1139, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "CU1", 199990, pct_vol=1.3, decimals=0),
        simulated_quote_from_base(rng, "AA1", 24560, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "ZNA1", 24355, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "XI1", 137410, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "PBL1", 16640, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "X001", 385750, pct_vol=1.4, decimals=0),
    ]

    china_ag_rows = [
        simulated_quote_from_base(rng, "V1", 15575, pct_vol=1.6, decimals=0),
        simulated_quote_from_base(rng, "V001", 3122, pct_vol=1.6, decimals=0),
        simulated_quote_from_base(rng, "AC1", 2382, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "AK1", 4802, pct_vol=1.6, decimals=0),
        simulated_quote_from_base(rng, "AE1", 3112, pct_vol=1.6, decimals=0),
        simulated_quote_from_base(rng, "SH1", 8508, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "ZRR1", 2291, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "ZR01", 10054, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "PA1", 8824, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "CB1", 5421, pct_vol=1.6, decimals=0),
        simulated_quote_from_base(rng, "RT1", 17025, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "SRB1", 13550, pct_vol=1.6, decimals=0),
    ]

    # --------------------------------------------------------
    # OVERVIEW INDICES
    # --------------------------------------------------------
    asia_rows = [
        simulated_quote_from_base(rng, "STI", 4757, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "HSI", 25408, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "SHSZ300", 4615, pct_vol=1.1, decimals=0),
        simulated_quote_from_base(rng, "SHCOMP", 4097, pct_vol=1.1, decimals=0),
        simulated_quote_from_base(rng, "SZCOMP", 2681, pct_vol=1.2, decimals=0),
        simulated_quote_from_base(rng, "TPX", 3576, pct_vol=1.2, decimals=0),
        simulated_quote_from_base(rng, "NKY", 52729, pct_vol=1.5, decimals=0),
        simulated_quote_from_base(rng, "KOSPI", 5252, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "TWSE", 32110, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "NIFTY", 24028, pct_vol=1.2, decimals=0),
    ]

    america_rows = [
        simulated_quote_from_base(rng, "INDU", 47502, pct_vol=1.1, decimals=0),
        simulated_quote_from_base(rng, "SPX", 6740, pct_vol=1.0, decimals=0),
        simulated_quote_from_base(rng, "CCMP", 22388, pct_vol=1.1, decimals=0),
        simulated_quote_from_base(rng, "SPTSX", 33084, pct_vol=1.0, decimals=0),
        simulated_quote_from_base(rng, "MEXBOL", 67314, pct_vol=1.4, decimals=0),
        simulated_quote_from_base(rng, "IBOV", 178556, pct_vol=1.4, decimals=0),
    ]

    europe_rows = [
        simulated_quote_from_base(rng, "SX5E", 5621, pct_vol=1.1, decimals=0),
        simulated_quote_from_base(rng, "UKX", 10170, pct_vol=1.0, decimals=0),
        simulated_quote_from_base(rng, "CAC", 7837, pct_vol=1.0, decimals=0),
        simulated_quote_from_base(rng, "DAX", 23269, pct_vol=1.0, decimals=0),
        simulated_quote_from_base(rng, "IBEX", 16785, pct_vol=1.0, decimals=0),
        simulated_quote_from_base(rng, "FTSEMIB", 43544, pct_vol=1.1, decimals=0),
        simulated_quote_from_base(rng, "OMX", 2977, pct_vol=1.2, decimals=0),
        simulated_quote_from_base(rng, "SMI", 12858, pct_vol=1.0, decimals=0),
    ]

    return {
        "broad": build_quote_rows(broad_rows),
        "energy": build_quote_rows(energy_rows),
        "metals": build_quote_rows(metals_rows),
        "agriculture": build_quote_rows(ag_rows),
        "china_energy": build_quote_rows(china_energy_rows),
        "china_metals": build_quote_rows(china_metals_rows),
        "china_agriculture": build_quote_rows(china_ag_rows),
        "indices_asia": build_quote_rows(asia_rows),
        "indices_america": build_quote_rows(america_rows),
        "indices_europe": build_quote_rows(europe_rows),
    }


market_tables = build_market_monitor_tables()

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="dashboard-title">Cotton Trading Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Preview version — simulated market data, structure aligned to final Bloomberg-based architecture.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="header-spacer"></div>', unsafe_allow_html=True)

# ============================================================
# TOP SUMMARY CARDS
# ============================================================

top1, top2, top3, top4 = st.columns(4)

with top1:
    cotton_intraday = latest["CT1"] - prev["CT1"]
    cotton_intraday_pct = pct_change(latest["CT1"], prev["CT1"])
    cotton_intraday_arrow = "↑" if cotton_intraday > 0 else "↓" if cotton_intraday < 0 else "→"
    cotton_intraday_class = "score-bullish" if cotton_intraday > 0 else "score-bearish" if cotton_intraday < 0 else "score-neutral"

    st.markdown(
        f"""
        <div class="top-card">
            <div class="top-card-label">Cotton Spot</div>
            <div class="top-card-value">{latest["CT1"]:.2f}</div>
            <div class="top-card-sub {cotton_intraday_class}">
                {cotton_intraday_arrow} {cotton_intraday:+.2f} ({cotton_intraday_pct:+.2f}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top2:
    st.markdown(
        f"""
        <div class="top-card">
            <div class="top-card-label">Composite Score</div>
            <div class="top-card-value {score_css_class}">{composite_score:+.2f}</div>
            <div class="top-card-sub {score_css_class}">{score_direction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top3:
    cotton_move = pct_change(latest["CT1"], prev["CT1"])
    cotton_direction = signed_arrow(cotton_move)
    cotton_class = "score-bullish" if cotton_move > 0 else "score-bearish" if cotton_move < 0 else "score-neutral"

    st.markdown(
        f"""
        <div class="top-card">
            <div class="top-card-label">Cotton Bias</div>
            <div class="top-card-value {cotton_class}">{cotton_direction} {bias_label(cotton_move)}</div>
            <div class="top-card-sub">{format_last(latest["CT1"], 2)} c/lb</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top4:
    st.markdown(
        """
        <div class="top-card">
            <div class="top-card-label">Last Update</div>
            <div class="top-card-sub" style="margin-bottom: 0.35rem;">Singapore time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        """
        <div style="
            margin-top:-88px;
            padding-left:18px;
            padding-right:18px;
            font-family: sans-serif;
            color:#f4f7ff;
        ">
            <div id="sg-clock" style="
                font-size: 1.75rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 0.30rem;
            ">--:--:--</div>
            <div style="
                color:#c9d2e6;
                font-size:0.92rem;
            ">Live clock</div>
        </div>

        <script>
        function updateSingaporeClock() {
            const now = new Date();
            const formatter = new Intl.DateTimeFormat('en-GB', {
                timeZone: 'Asia/Singapore',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
            });
            const time = formatter.format(now);
            const el = document.getElementById("sg-clock");
            if (el) {
                el.textContent = time;
            }
        }

        updateSingaporeClock();
        setInterval(updateSingaporeClock, 1000);
        </script>
        """,
        height=70,
    )

# ============================================================
# MAIN MARKET MONITOR LAYOUT
# ============================================================

# ------------------------------------------------------------
# FIRST ROW:
# Cotton #2 Indicators + Signal Breakdown side by side
# ------------------------------------------------------------
top_left, top_right = st.columns([0.62, 0.38], gap="large")

with top_left:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cotton #2 Indicators</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Main cross-market drivers relevant for cotton direction.</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Softs section
    # --------------------------------------------------------
    softs_avg_intensity = softs_indicator_df["Intensity"].mean()
    softs_weekly_mean = softs_indicator_df["_weekly_num"].mean()
    softs_avg_bias = "Bullish" if softs_weekly_mean > 0 else "Bearish" if softs_weekly_mean < 0 else "Neutral"
    softs_avg_class = "score-bullish" if softs_avg_bias == "Bullish" else "score-bearish" if softs_avg_bias == "Bearish" else "score-neutral"

    st.markdown("**Softs**")
    st.markdown(
        f"""
        <div style="margin-bottom:0.45rem;">
            <span style="color:#9ba8c5; font-size:0.88rem;">Average Softs Intensity:</span>
            <span class="{softs_avg_class}" style="font-weight:700; margin-left:8px;">{softs_avg_intensity:.0f}</span>
            <span class="{softs_avg_class}" style="font-weight:700; margin-left:10px;">{softs_avg_bias}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        style_indicator_table(softs_indicator_df),
        use_container_width=True,
        hide_index=True,
        height=140,
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Ags & Oilseeds section
    # --------------------------------------------------------
    ags_avg_intensity = ags_indicator_df["Intensity"].mean()
    ags_weekly_mean = ags_indicator_df["_weekly_num"].mean()
    ags_avg_bias = "Bullish" if ags_weekly_mean > 0 else "Bearish" if ags_weekly_mean < 0 else "Neutral"
    ags_avg_class = "score-bullish" if ags_avg_bias == "Bullish" else "score-bearish" if ags_avg_bias == "Bearish" else "score-neutral"

    st.markdown("**Ags & Oilseeds**")
    st.markdown(
        f"""
        <div style="margin-bottom:0.45rem;">
            <span style="color:#9ba8c5; font-size:0.88rem;">Average Ags Intensity:</span>
            <span class="{ags_avg_class}" style="font-weight:700; margin-left:8px;">{ags_avg_intensity:.0f}</span>
            <span class="{ags_avg_class}" style="font-weight:700; margin-left:10px;">{ags_avg_bias}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        style_indicator_table(ags_indicator_df),
        use_container_width=True,
        hide_index=True,
        height=140,
    )

    st.markdown('</div>', unsafe_allow_html=True)

with top_right:
    breakdown_rows = [
        {
            "Driver": "Cotton Momentum",
            "Signal": round(signals["cotton_momentum"], 2),
            "Weight": weights["cotton_momentum"],
            "Contribution": round(signals["cotton_momentum"] * weights["cotton_momentum"], 2),
        },
        {
            "Driver": "Spread Structure",
            "Signal": round(signals["spread_structure"], 2),
            "Weight": weights["spread_structure"],
            "Contribution": round(signals["spread_structure"] * weights["spread_structure"], 2),
        },
        {
            "Driver": "Soft Complex",
            "Signal": round(signals["soft_complex"], 2),
            "Weight": weights["soft_complex"],
            "Contribution": round(signals["soft_complex"] * weights["soft_complex"], 2),
        },
        {
            "Driver": "Agri Complex",
            "Signal": round(signals["agri_complex"], 2),
            "Weight": weights["agri_complex"],
            "Contribution": round(signals["agri_complex"] * weights["agri_complex"], 2),
        },
        {
            "Driver": "Energy",
            "Signal": round(signals["energy"], 2),
            "Weight": weights["energy"],
            "Contribution": round(signals["energy"] * weights["energy"], 2),
        },
        {
            "Driver": "Macro",
            "Signal": round(signals["macro"], 2),
            "Weight": weights["macro"],
            "Contribution": round(signals["macro"] * weights["macro"], 2),
        },
    ]
    breakdown_df = pd.DataFrame(breakdown_rows)

    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Signal Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">How the composite score is built.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
    style_signal_table(breakdown_df),
    use_container_width=True,
    hide_index=True,
    height=210,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# SECOND ROW:
# Europe / US Commodities + China Commodities side by side
# ------------------------------------------------------------
mid_left, mid_right = st.columns([0.5, 0.5], gap="large")

with mid_left:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Overview Commodity</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Bloomberg-style monitor for broad commodities, energy, metals and agriculture/softs.</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Top layout inside Overview Commodity:
    # LEFT  = Broad
    # RIGHT = Energy (top) + Metals (bottom)
    # --------------------------------------------------------
    overview_left, overview_right = st.columns([0.48, 0.52], gap="large")

    with overview_left:
        st.markdown("**Broad**")
        st.dataframe(
            style_market_table(market_tables["broad"]),
            use_container_width=True,
            hide_index=True,
            height=165,
        )

    with overview_right:
        st.markdown("**Energy**")
        st.dataframe(
            style_market_table(market_tables["energy"]),
            use_container_width=True,
            hide_index=True,
            height=315,
        )

        st.markdown("**Metals**")
        st.dataframe(
            style_market_table(market_tables["metals"]),
            use_container_width=True,
            hide_index=True,
            height=455,
        )

    # --------------------------------------------------------
    # Bottom full-width table
    # --------------------------------------------------------
    st.markdown("**Agriculture / Softs**")
    st.dataframe(
        style_market_table(market_tables["agriculture"]),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    st.markdown('</div>', unsafe_allow_html=True)

with mid_right:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">China Commodities</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Separate view for Chinese commodity markets, as requested.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Energy**")
    st.dataframe(
        style_market_table_int(market_tables["china_energy"]),
        use_container_width=True,
        hide_index=True,
        height=155,
    )

    st.markdown("**Metals**")
    st.dataframe(
        style_market_table_int(market_tables["china_metals"]),
        use_container_width=True,
        hide_index=True,
        height=455,
    )

    st.markdown("**Agriculture / Softs / Oilseeds**")
    st.dataframe(
        style_market_table_int(market_tables["china_agriculture"]),
        use_container_width=True,
        hide_index=True,
        height=470,
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# OVERVIEW INDICES SECTION
# ============================================================

st.divider()

st.markdown('<div class="table-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Overview Indices</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Global equity index monitor inspired by Bloomberg Overview Indices.</div>',
    unsafe_allow_html=True,
)

idx_col1, idx_col2, idx_col3 = st.columns(3)

with idx_col1:
    st.markdown("**Asia / Pacific**")
    st.dataframe(
        style_market_table_int(market_tables["indices_asia"]),
        use_container_width=True,
        hide_index=True,
        height=390,
    )

with idx_col2:
    st.markdown("**America**")
    st.dataframe(
        style_market_table_int(market_tables["indices_america"]),
        use_container_width=True,
        hide_index=True,
        height=390,
    )

with idx_col3:
    st.markdown("**Europe**")
    st.dataframe(
        style_market_table_int(market_tables["indices_europe"]),
        use_container_width=True,
        hide_index=True,
        height=390,
    )

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# BOTTOM DRIVER SUMMARY
# ============================================================

st.divider()

bullish = sorted(
    [(k, v * weights[k]) for k, v in signals.items()],
    key=lambda x: x[1],
    reverse=True,
)
bearish = sorted(
    [(k, v * weights[k]) for k, v in signals.items()],
    key=lambda x: x[1],
)

pretty_names = {
    "cotton_momentum": "Cotton Momentum",
    "spread_structure": "Spread Structure",
    "soft_complex": "Soft Complex",
    "agri_complex": "Agri Complex",
    "energy": "Energy",
    "macro": "Macro",
}

bottom1, bottom2, bottom3 = st.columns(3)

with bottom1:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top Bullish Drivers</div>', unsafe_allow_html=True)
    for k, v in bullish[:3]:
        st.metric(pretty_names[k], f"{v:+.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with bottom2:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top Bearish Drivers</div>', unsafe_allow_html=True)
    for k, v in bearish[:3]:
        st.metric(pretty_names[k], f"{v:+.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with bottom3:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Market Context</div>', unsafe_allow_html=True)
    st.metric("Open Interest", f"{int(latest['OPEN_INT']):,}", f"{int(latest['OPEN_INT'] - prev['OPEN_INT']):+d}")
    st.metric("Volume", f"{int(latest['VOLUME']):,}", f"{int(latest['VOLUME'] - prev['VOLUME']):+d}")
    st.metric("BCOM", f"{latest['BCOM']:.2f}", f"{pct_change(latest['BCOM'], prev['BCOM']):+.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)
