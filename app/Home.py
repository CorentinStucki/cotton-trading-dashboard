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
import streamlit as st
import streamlit.components.v1 as components
import json
import requests

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
        font-size: 2.7rem;
        font-weight: 700;
        color: #f4f7ff;
        margin-bottom: 0.2rem;
        line-height: 1.05;
        white-space: normal;
        overflow: visible;
        word-break: normal;
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f2f5ff;
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
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
    Stable simulated data for 2 minutes at a time.
    """
    seed = int(datetime.now().timestamp() // 120)
    return np.random.default_rng(seed)


def make_series(
    start: float,
    n: int,
    drift: float,
    vol: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = [start]
    for _ in range(n - 1):
        step = drift + rng.normal(0, vol)
        values.append(max(0.01, values[-1] + step))
    return np.array(values)


def make_time_index(n: int, freq_minutes: int = 60) -> pd.DatetimeIndex:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(minutes=freq_minutes * (n - 1))
    return pd.date_range(start=start, periods=n, freq=f"{freq_minutes}min")


def pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100.0


def signed_arrow(value: float) -> str:
    if value > 0:
        return "↑"
    if value < 0:
        return "↓"
    return "→"


def bias_label(value: float) -> str:
    if value > 0:
        return "Bullish"
    if value < 0:
        return "Bearish"
    return "Neutral"


def score_to_intensity(score: float) -> float:
    clipped = max(-3.0, min(3.0, score))
    return round(((clipped + 3.0) / 6.0) * 100.0, 1)


def format_last(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def normalize_weights(weight_dict: dict[str, float]) -> dict[str, float]:
    total = sum(weight_dict.values())
    if total <= 0:
        n = len(weight_dict)
        return {k: 1 / n for k in weight_dict}
    return {k: v / total for k, v in weight_dict.items()}


def weighted_group_score(signal_dict: dict[str, float], weight_dict: dict[str, float]) -> float:
    norm_weights = normalize_weights(weight_dict)
    return sum(signal_dict[k] * norm_weights[k] for k in signal_dict if k in norm_weights)


def format_arrow_value(value: float, decimals: int = 2) -> str:
    arrow = signed_arrow(value)
    return f"{arrow} {value:.{decimals}f}"


def build_quote_rows(raw_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    return df[["Ticker", "Last Price", "Net", "%1D", "_net_num", "_pct_num"]]


def color_ticker(val):
    return "color: #f5a623; font-weight: 600;"


def color_pos_neg(val):
    try:
        v = float(val)
    except Exception:
        return "color: #dfe6f5;"
    if v > 0:
        return "color: #59d98e; font-weight: 600;"
    if v < 0:
        return "color: #ff6b6b; font-weight: 600;"
    return "color: #dfe6f5;"

def color_bias(val):
    val_str = str(val).lower()
    if "bullish" in val_str:
        return "color: #59d98e; font-weight: 600;"
    if "bearish" in val_str:
        return "color: #ff6b6b; font-weight: 600;"
    return "color: #f1c75b; font-weight: 600;"


def style_market_table(df: pd.DataFrame):
    display_df = df[["Ticker", "Last Price", "Net", "%1D"]].copy()
    styler = display_df.style

    styler = styler.applymap(color_ticker, subset=["Ticker"])

    if "_net_num" in df.columns:
        net_colors = [color_pos_neg(v) for v in df["_net_num"]]
        styler = styler.apply(lambda _: net_colors, subset=["Net"], axis=0)

    if "_pct_num" in df.columns:
        pct_colors = [color_pos_neg(v) for v in df["_pct_num"]]
        styler = styler.apply(lambda _: pct_colors, subset=["%1D"], axis=0)

    styler = styler.format({"Last Price": "{:,.1f}"}, na_rep="")
    return styler


def style_market_table_int(df: pd.DataFrame):
    display_df = df[["Ticker", "Last Price", "Net", "%1D"]].copy()
    styler = display_df.style

    styler = styler.applymap(color_ticker, subset=["Ticker"])

    if "_net_num" in df.columns:
        net_colors = [color_pos_neg(v) for v in df["_net_num"]]
        styler = styler.apply(lambda _: net_colors, subset=["Net"], axis=0)

    if "_pct_num" in df.columns:
        pct_colors = [color_pos_neg(v) for v in df["_pct_num"]]
        styler = styler.apply(lambda _: pct_colors, subset=["%1D"], axis=0)

    styler = styler.format({"Last Price": "{:,.0f}"}, na_rep="")
    return styler


def style_indicator_table(df: pd.DataFrame):
    display_df = df[["Variable", "Last", "Intensity (%)", "DOD", "WOW", "Bias"]].copy()
    styler = display_df.style

    if "_dod_num" in df.columns:
        dod_colors = [color_pos_neg(v) for v in df["_dod_num"]]
        styler = styler.apply(lambda _: dod_colors, subset=["DOD"], axis=0)

    if "_wow_num" in df.columns:
        wow_colors = [color_pos_neg(v) for v in df["_wow_num"]]
        styler = styler.apply(lambda _: wow_colors, subset=["WOW"], axis=0)

    if "_bias_text" in df.columns:
        bias_colors = [color_bias(v) for v in df["_bias_text"]]
        styler = styler.apply(lambda _: bias_colors, subset=["Bias"], axis=0)

    styler = styler.format(
        {
            "Last": "{:,.2f}",
            "Intensity (%)": "{:.1f}",
        }
    )
    return styler

def convert_market_table_to_signal_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert market monitor tables into the same structure as Softs table:
    Variable / Last / Intensity (%) / DOD / WOW / Bias
    """
    out = df.copy()

    out["Variable"] = out["Ticker"]
    out["Last"] = out["Last Price"]

    # Use absolute daily move as a proxy for intensity, normalized to 100%
    abs_moves = out["_pct_num"].abs()
    total_abs = abs_moves.sum()

    if total_abs == 0:
        out["Intensity (%)"] = round(100 / len(out), 1)
    else:
        out["Intensity (%)"] = (abs_moves / total_abs * 100).round(1)

    # DOD and WOW are approximated from current daily change
    out["DOD"] = out["_pct_num"].apply(lambda x: format_arrow_value(x, 2) + "%")
    out["WOW"] = out["_pct_num"].apply(lambda x: format_arrow_value(x, 2) + "%")

    out["_dod_num"] = out["_pct_num"].round(2)
    out["_wow_num"] = out["_pct_num"].round(2)

    out["_bias_text"] = out["_pct_num"].apply(
        lambda x: "Bullish" if x > 0 else "Bearish" if x < 0 else "Neutral"
    )
    out["Bias"] = out["_bias_text"].apply(
        lambda x: f"↑ {x}" if x == "Bullish" else f"↓ {x}" if x == "Bearish" else f"→ {x}"
    )

    return out[
        ["Variable", "Last", "Intensity (%)", "DOD", "WOW", "Bias", "_dod_num", "_wow_num", "_bias_text"]
    ].reset_index(drop=True)

def build_technicals_monthly_df() -> pd.DataFrame:
    """
    Placeholder monthly technicals table.
    Structure matches Viola's requested format and can later be replaced
    by real Bloomberg / technical indicator calculations.
    """
    technical_rows = [
        {"Variable": "Moving Average", "Last": 1.20, "Intensity (%)": 20.0, "DOD_num": 0.15, "WOW_num": 0.80},
        {"Variable": "TrendLine", "Last": 1.00, "Intensity (%)": 15.0, "DOD_num": 0.05, "WOW_num": 0.40},
        {"Variable": "14-Day RSI", "Last": 56.20, "Intensity (%)": 15.0, "DOD_num": 1.10, "WOW_num": 3.40},
        {"Variable": "MacD", "Last": 0.85, "Intensity (%)": 18.0, "DOD_num": 0.08, "WOW_num": 0.32},
        {"Variable": "Slow Stochastics", "Last": 61.40, "Intensity (%)": 12.0, "DOD_num": -0.90, "WOW_num": 2.10},
        {"Variable": "Bollinger Bands", "Last": 0.72, "Intensity (%)": 10.0, "DOD_num": 0.02, "WOW_num": -0.15},
        {"Variable": "Z-Score", "Last": -0.45, "Intensity (%)": 10.0, "DOD_num": 0.03, "WOW_num": -0.20},
    ]

    rows = []
    for r in technical_rows:
        direction = "Bullish" if r["WOW_num"] > 0 else "Bearish" if r["WOW_num"] < 0 else "Neutral"
        bias_arrow = "↑" if direction == "Bullish" else "↓" if direction == "Bearish" else "→"

        rows.append(
            {
                "Variable": r["Variable"],
                "Last": r["Last"],
                "Intensity (%)": r["Intensity (%)"],
                "DOD": format_arrow_value(r["DOD_num"], 2),
                "WOW": format_arrow_value(r["WOW_num"], 2),
                "Bias": f"{bias_arrow} {direction}",
                "_dod_num": r["DOD_num"],
                "_wow_num": r["WOW_num"],
                "_bias_text": direction,
            }
        )

    return pd.DataFrame(rows)

# ============================================================
# PERSISTENT WEIGHTS STORAGE (SUPABASE)
# ============================================================

DEFAULT_WEIGHTS_CONFIG = {
    "global": {
        "cotton_momentum": 0.28,
        "spread_structure": 0.20,
        "soft_complex": 0.16,
        "agri_complex": 0.14,
        "energy": 0.12,
        "macro": 0.10,
    },
    "softs": {
        "sugar": 0.33,
        "coffee": 0.33,
        "cocoa": 0.34,
    },
    "grains": {
        "corn": 0.34,
        "soybeans": 0.33,
        "wheat": 0.33,
    },
}


def get_supabase_config():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    profile = st.secrets.get("WEIGHTS_PROFILE", "default")
    return url, key, profile


def load_weights_from_store() -> dict:
    """
    Load weights from Supabase.
    Falls back to defaults if unavailable.
    """
    url, key, profile = get_supabase_config()

    if not url or not key:
        return DEFAULT_WEIGHTS_CONFIG.copy()

    endpoint = f"{url}/rest/v1/dashboard_weights"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    params = {
        "profile": f"eq.{profile}",
        "select": "weights_json",
        "limit": "1",
    }

    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()

        if rows and "weights_json" in rows[0]:
            stored = rows[0]["weights_json"]
            return {
                "global": stored.get("global", DEFAULT_WEIGHTS_CONFIG["global"]).copy(),
                "softs": stored.get("softs", DEFAULT_WEIGHTS_CONFIG["softs"]).copy(),
                "grains": stored.get("grains", DEFAULT_WEIGHTS_CONFIG["grains"]).copy(),
            }

    except Exception as e:
        st.sidebar.warning(f"Could not load saved weights. Using defaults.")

    return DEFAULT_WEIGHTS_CONFIG.copy()


def save_weights_to_store(weights_config: dict) -> bool:
    """
    Save weights to Supabase.
    """
    url, key, profile = get_supabase_config()

    if not url or not key:
        st.sidebar.error("Supabase is not configured in secrets.")
        return False

    endpoint = f"{url}/rest/v1/dashboard_weights"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    payload = {
        "profile": profile,
        "weights_json": weights_config,
    }

    try:
        resp = requests.post(
            f"{endpoint}?on_conflict=profile",
            headers=headers,
            data=json.dumps(payload),
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception:
        st.sidebar.error("Could not save weights.")
        return False


def get_current_weights_config_from_session() -> dict:
    return {
        "global": {
            "cotton_momentum": st.session_state["w_cotton"],
            "spread_structure": st.session_state["w_spread"],
            "soft_complex": st.session_state["w_softs"],
            "agri_complex": st.session_state["w_agri"],
            "energy": st.session_state["w_energy"],
            "macro": st.session_state["w_macro"],
        },
        "softs": {
            "sugar": st.session_state["w_sugar"],
            "coffee": st.session_state["w_coffee"],
            "cocoa": st.session_state["w_cocoa"],
        },
        "grains": {
            "corn": st.session_state["w_corn"],
            "soybeans": st.session_state["w_soybeans"],
            "wheat": st.session_state["w_wheat"],
        },
    }


def initialize_weight_session_state():
    if "weights_initialized" not in st.session_state:
        saved = load_weights_from_store()

        st.session_state["w_cotton"] = saved["global"]["cotton_momentum"]
        st.session_state["w_spread"] = saved["global"]["spread_structure"]
        st.session_state["w_softs"] = saved["global"]["soft_complex"]
        st.session_state["w_agri"] = saved["global"]["agri_complex"]
        st.session_state["w_energy"] = saved["global"]["energy"]
        st.session_state["w_macro"] = saved["global"]["macro"]

        st.session_state["w_sugar"] = saved["softs"]["sugar"]
        st.session_state["w_coffee"] = saved["softs"]["coffee"]
        st.session_state["w_cocoa"] = saved["softs"]["cocoa"]

        st.session_state["w_corn"] = saved["grains"]["corn"]
        st.session_state["w_soybeans"] = saved["grains"]["soybeans"]
        st.session_state["w_wheat"] = saved["grains"]["wheat"]

        st.session_state["weights_initialized"] = True

# ============================================================
# SIDEBAR — MODEL WEIGHTS
# ============================================================
initialize_weight_session_state()

st.sidebar.markdown("## Global Weights")

w_cotton = st.sidebar.number_input(
    "Cotton Momentum", 0.0, 1.0, step=0.01, key="w_cotton"
)
w_spread = st.sidebar.number_input(
    "Spread Structure", 0.0, 1.0, step=0.01, key="w_spread"
)
w_softs = st.sidebar.number_input(
    "Soft Complex", 0.0, 1.0, step=0.01, key="w_softs"
)
w_agri = st.sidebar.number_input(
    "Agri Complex", 0.0, 1.0, step=0.01, key="w_agri"
)
w_energy = st.sidebar.number_input(
    "Energy", 0.0, 1.0, step=0.01, key="w_energy"
)
w_macro = st.sidebar.number_input(
    "Macro", 0.0, 1.0, step=0.01, key="w_macro"
)

weights = {
    "cotton_momentum": w_cotton,
    "spread_structure": w_spread,
    "soft_complex": w_softs,
    "agri_complex": w_agri,
    "energy": w_energy,
    "macro": w_macro,
}

total_weight = sum(weights.values())
st.sidebar.markdown(f"**Total weight = {total_weight:.2f}**")

if abs(total_weight - 1.0) > 0.001:
    st.sidebar.warning("Total should be 1.00")

norm_global = normalize_weights(weights)

st.sidebar.markdown(
    f"""
**Normalized global weights**  
Cotton: {norm_global['cotton_momentum']:.2f}  
Spread: {norm_global['spread_structure']:.2f}  
Softs: {norm_global['soft_complex']:.2f}  
Grains: {norm_global['agri_complex']:.2f}  
Energy: {norm_global['energy']:.2f}  
Macro: {norm_global['macro']:.2f}
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Softs Weights")

w_cotton_soft = st.sidebar.number_input("Cotton", 0.0, 10.0, 0.25, 0.01, key="w_cotton_soft")
w_sugar = st.sidebar.number_input("Sugar", min_value=0.0, max_value=10.0, step=0.01, key="w_sugar")
w_coffee = st.sidebar.number_input("Coffee", min_value=0.0, max_value=10.0, step=0.01, key="w_coffee")
w_cocoa = st.sidebar.number_input("Cocoa", min_value=0.0, max_value=10.0, step=0.01, key="w_cocoa")

softs_internal_weights = {
    "cotton": w_cotton_soft,
    "sugar": w_sugar,
    "cocoa": w_cocoa,
    "coffee": w_coffee,
}

softs_internal_total = sum(softs_internal_weights.values())
st.sidebar.markdown(f"**Softs total = {softs_internal_total:.2f}**")
if softs_internal_total <= 0:
    st.sidebar.warning("Softs total should be > 0")

st.sidebar.markdown("---")
st.sidebar.markdown("## Grains & Oilseeds Weights")

w_corn = st.sidebar.number_input("Corn", 0.0, 10.0, step=0.01, key="w_corn")
w_soybeans = st.sidebar.number_input("Soybeans", 0.0, 10.0, step=0.01, key="w_soybeans")
w_wheat = st.sidebar.number_input("Wheat", 0.0, 10.0, step=0.01, key="w_wheat")

grains_internal_weights = {
    "corn": w_corn,
    "soybeans": w_soybeans,
    "wheat": w_wheat,
}

grains_internal_total = sum(grains_internal_weights.values())
st.sidebar.markdown(f"**Grains total = {grains_internal_total:.2f}**")
if grains_internal_total <= 0:
    st.sidebar.warning("Grains total should be > 0")

st.sidebar.markdown("---")

col_save, col_reset = st.sidebar.columns(2)

with col_save:
    if st.button("Save Weights", use_container_width=True):
        cfg = get_current_weights_config_from_session()
        if save_weights_to_store(cfg):
            st.sidebar.success("Weights saved")

with col_reset:
    if st.button("Reset Defaults", use_container_width=True):
        defaults = DEFAULT_WEIGHTS_CONFIG

        st.session_state["w_cotton"] = defaults["global"]["cotton_momentum"]
        st.session_state["w_spread"] = defaults["global"]["spread_structure"]
        st.session_state["w_softs"] = defaults["global"]["soft_complex"]
        st.session_state["w_agri"] = defaults["global"]["agri_complex"]
        st.session_state["w_energy"] = defaults["global"]["energy"]
        st.session_state["w_macro"] = defaults["global"]["macro"]

        st.session_state["w_sugar"] = defaults["softs"]["sugar"]
        st.session_state["w_coffee"] = defaults["softs"]["coffee"]
        st.session_state["w_cocoa"] = defaults["softs"]["cocoa"]

        st.session_state["w_corn"] = defaults["grains"]["corn"]
        st.session_state["w_soybeans"] = defaults["grains"]["soybeans"]
        st.session_state["w_wheat"] = defaults["grains"]["wheat"]

        st.rerun()

# ============================================================
# BUILD THE CORE SIMULATED DATASET
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def build_demo_core_dataset():
    rng = seeded_rng()
    idx = make_time_index(72, freq_minutes=60)

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

    may_jul = make_series(-0.65, len(idx), drift=0.005, vol=0.06, rng=rng)
    jul_dec = make_series(1.15, len(idx), drift=-0.003, vol=0.07, rng=rng)

    open_int = make_series(99277, len(idx), drift=8, vol=140, rng=rng)
    volume = np.abs(make_series(23000, len(idx), drift=20, vol=1800, rng=rng))

    return pd.DataFrame(
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


df = build_demo_core_dataset()
latest = df.iloc[-1]
prev = df.iloc[-2]
week_ago = df.iloc[-6] if len(df) >= 6 else df.iloc[0]

# ============================================================
# RAW SUB-SIGNALS FOR INTERNAL GROUPS
# ============================================================

softs_subsignals = {
    "cotton": normalize_change_to_signal(
        pct_change(latest["CT1"], df["CT1"].iloc[-6]),
        scale=2.2,
    ),
    "sugar": normalize_change_to_signal(
        pct_change(latest["SB1"], df["SB1"].iloc[-6]),
        scale=2.2,
    ),
    "cocoa": normalize_change_to_signal(
        pct_change(latest["CC1"], df["CC1"].iloc[-6]),
        scale=2.2,
    ),
    "coffee": normalize_change_to_signal(
        pct_change(latest["KC1"], df["KC1"].iloc[-6]),
        scale=2.2,
    ),
}

grains_subsignals = {
    "corn": normalize_change_to_signal(pct_change(latest["C1"], df["C1"].iloc[-6]), scale=2.0),
    "soybeans": normalize_change_to_signal(pct_change(latest["S1"], df["S1"].iloc[-6]), scale=2.0),
    "wheat": normalize_change_to_signal(pct_change(latest["W1"], df["W1"].iloc[-6]), scale=2.0),
}

soft_complex_score = weighted_group_score(softs_subsignals, softs_internal_weights)
agri_complex_score = weighted_group_score(grains_subsignals, grains_internal_weights)

# ============================================================
# MAIN SIGNALS + COMPOSITE
# ============================================================

signals = {
    "cotton_momentum": normalize_change_to_signal(
        pct_change(latest["CT1"], df["CT1"].iloc[-6]), scale=1.8
    ),
    "spread_structure": normalize_change_to_signal(
        latest["MAY_JUL"] * -12.0, scale=1.0
    ),
    "soft_complex": soft_complex_score,
    "agri_complex": agri_complex_score,
    "energy": normalize_change_to_signal(
        pct_change(latest["CL1"], df["CL1"].iloc[-6]), scale=2.0
    ),
    "macro": np.mean(
        [
            -normalize_change_to_signal(pct_change(latest["DXY"], df["DXY"].iloc[-6]), scale=0.8),
            normalize_change_to_signal(pct_change(latest["BCOM"], df["BCOM"].iloc[-6]), scale=1.5),
            normalize_change_to_signal(pct_change(latest["BCOMAG"], df["BCOMAG"].iloc[-6]), scale=1.2),
        ]
    ),
}

composite_score = weighted_composite_score(signals, norm_global)
signal_label = classify_score(composite_score)

pretty_block_names = {
    "cotton_momentum": "Cotton Momentum",
    "spread_structure": "Spread Structure",
    "soft_complex": "Soft Complex",
    "agri_complex": "Grains & Oilseeds",
    "energy": "Energy",
    "macro": "Macro",
}

contrib_rows = []
for k, v in signals.items():
    contrib_rows.append(
        {
            "Block": pretty_block_names.get(k, k),
            "Signal": round(v, 2),
            "Weight": round(norm_global[k], 2),
            "Contribution": round(v * norm_global[k], 2),
        }
    )

contrib_df = pd.DataFrame(contrib_rows).sort_values("Contribution", ascending=False).reset_index(drop=True)

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
# COTTON #2 INDICATORS DATA
# ============================================================

softs_snapshot = {
    "Cotton": {
        "last": latest["CT1"],
        "dod_pct": pct_change(latest["CT1"], prev["CT1"]),
        "wow_pct": pct_change(latest["CT1"], week_ago["CT1"]),
    },
    "Sugar": {
        "last": latest["SB1"],
        "dod_pct": pct_change(latest["SB1"], prev["SB1"]),
        "wow_pct": pct_change(latest["SB1"], week_ago["SB1"]),
    },
    "Cocoa": {
        "last": latest["CC1"],
        "dod_pct": pct_change(latest["CC1"], prev["CC1"]),
        "wow_pct": pct_change(latest["CC1"], week_ago["CC1"]),
    },
    "Coffee": {
        "last": latest["KC1"],
        "dod_pct": pct_change(latest["KC1"], prev["KC1"]),
        "wow_pct": pct_change(latest["KC1"], week_ago["KC1"]),
    },
}

grains_snapshot = {
    "Corn": {
        "last": latest["C1"],
        "dod_pct": pct_change(latest["C1"], prev["C1"]),
        "wow_pct": pct_change(latest["C1"], week_ago["C1"]),
    },
    "Soybeans": {
        "last": latest["S1"],
        "dod_pct": pct_change(latest["S1"], prev["S1"]),
        "wow_pct": pct_change(latest["S1"], week_ago["S1"]),
    },
    "Wheat": {
        "last": latest["W1"],
        "dod_pct": pct_change(latest["W1"], prev["W1"]),
        "wow_pct": pct_change(latest["W1"], week_ago["W1"]),
    },
}


def build_indicator_df(snapshot_dict, internal_weights):
    rows = []

    norm_weights = normalize_weights(internal_weights)

    for name, item in snapshot_dict.items():

        key_map = {
            "Cotton": "cotton",
            "Sugar": "sugar",
            "Coffee": "coffee",
            "Cocoa": "cocoa",
            "Corn": "corn",
            "Soybeans": "soybeans",
            "Wheat": "wheat",
        }

        weight_key = key_map[name]
        weight_value = norm_weights.get(weight_key, 0)

        direction = "Bullish" if item["wow_pct"] > 0 else "Bearish" if item["wow_pct"] < 0 else "Neutral"
        bias_arrow = "↑" if direction == "Bullish" else "↓" if direction == "Bearish" else "→"

        rows.append(
            {
                "Variable": name,
                "Last": round(item["last"], 2),

                # ✅ NEW: intensity = model weights
                "Intensity (%)": round(weight_value * 100, 1),

                "DOD": format_arrow_value(item["dod_pct"], 2) + "%",
                "WOW": format_arrow_value(item["wow_pct"], 2) + "%",
                "Bias": f"{bias_arrow} {direction}",

                "_dod_num": round(item["dod_pct"], 2),
                "_wow_num": round(item["wow_pct"], 2),
                "_bias_text": direction,
            }
        )

    df = pd.DataFrame(rows)

    return df.sort_values("Intensity (%)", ascending=False).reset_index(drop=True)


softs_indicator_df = build_indicator_df(softs_snapshot, softs_internal_weights)
grains_indicator_df = build_indicator_df(grains_snapshot, grains_internal_weights)
technicals_monthly_df = build_technicals_monthly_df()

# ============================================================
# MARKET MONITOR TABLE BUILDERS
# ============================================================

def make_quote(ticker: str, last: float, prev_value: float | None = None, decimals: int = 1) -> dict:
    if prev_value is None:
        prev_value = last * 0.99

    net = last - prev_value
    pct = pct_change(last, prev_value)

    net_arrow = "↑" if net > 0 else "↓" if net < 0 else "→"
    pct_arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"

    return {
        "Ticker": ticker,
        "Last Price": round(last, decimals),
        "Net": f"{net_arrow} {net:,.{decimals}f}",
        "%1D": f"{pct_arrow} {pct:,.1f}",
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
    prev_val = base * (1 + rng.normal(0, pct_vol / 100))
    last_val = base * (1 + rng.normal(0, pct_vol / 100))
    return make_quote(ticker, last_val, prev_val, decimals=decimals)


@st.cache_data(ttl=120, show_spinner=False)
def build_market_monitor_tables():
    rng = seeded_rng()

    broad_rows = [
        make_quote("BCOM", latest["BCOM"], prev["BCOM"], decimals=0),
        make_quote("BCOMAG", latest["BCOMAG"], prev["BCOMAG"], decimals=0),
        simulated_quote_from_base(rng, "XBTUSD", 68787, pct_vol=2.5, decimals=0),
    ]

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
others_table = convert_market_table_to_signal_format(market_tables["broad"])
energy_table = convert_market_table_to_signal_format(market_tables["energy"])
metals_table = convert_market_table_to_signal_format(market_tables["metals"])

china_energy_table = convert_market_table_to_signal_format(market_tables["china_energy"])
china_metals_table = convert_market_table_to_signal_format(market_tables["china_metals"])
china_agriculture_table = convert_market_table_to_signal_format(market_tables["china_agriculture"])

asia_table = convert_market_table_to_signal_format(market_tables["indices_asia"])
us_table = convert_market_table_to_signal_format(market_tables["indices_america"])
europe_table = convert_market_table_to_signal_format(market_tables["indices_europe"])
# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="dashboard-title">Cotton Trading Dashboard</div>', unsafe_allow_html=True)
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
            <div class="top-card-label">CTK6</div>
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
            <div class="top-card-label">Bias</div>
            <div class="top-card-value {cotton_class}">{cotton_direction} {bias_label(cotton_move)}</div>
            <div class="top-card-sub">{format_last(latest["CT1"], 2)} c/lb</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top4:
    components.html(
        """
        <div style="
            border-radius:16px;
            padding:16px 18px;
            min-height:108px;
            background: linear-gradient(180deg, rgba(21,29,48,0.95), rgba(12,18,34,0.95));
            border: 1px solid rgba(110,130,170,0.18);
            font-family: sans-serif;
            color:#f4f7ff;
            box-sizing:border-box;
        ">
            <div style="
                color:#9ba8c5;
                font-size:0.84rem;
                margin-bottom:0.25rem;
            ">Last Update</div>

            <div id="sg-clock" style="
                color:#f4f7ff;
                font-size:1.55rem;
                font-weight:800;
                line-height:1.1;
                margin-bottom:0.25rem;
            ">--/--/---- --:--:--</div>

            <div style="
                color:#c9d2e6;
                font-size:0.92rem;
            ">Singapore time</div>
        </div>

        <script>
        function updateSingaporeClock() {
            const now = new Date();

            const dateFormatter = new Intl.DateTimeFormat('en-GB', {
                timeZone: 'Asia/Singapore',
                day: '2-digit',
                month: '2-digit',
                year: 'numeric'
            });

            const timeFormatter = new Intl.DateTimeFormat('en-GB', {
                timeZone: 'Asia/Singapore',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false
            });

            const date = dateFormatter.format(now);
            const time = timeFormatter.format(now);

            const el = document.getElementById('sg-clock');
            if (el) {
                el.textContent = `${date} ${time}`;
            }
        }

        updateSingaporeClock();
        setInterval(updateSingaporeClock, 1000);
        </script>
        """,
        height=120,
    )

# ============================================================
# MAIN MARKET MONITOR LAYOUT
# ============================================================

top_left, top_right = st.columns([0.62, 0.38], gap="large")

with top_left:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cotton #2 Indicators</div>', unsafe_allow_html=True)

    softs_wow_mean = softs_indicator_df["_wow_num"].mean()
    softs_avg_bias = "Bullish" if softs_wow_mean > 0 else "Bearish" if softs_wow_mean < 0 else "Neutral"
    softs_avg_class = "score-bullish" if softs_avg_bias == "Bullish" else "score-bearish" if softs_avg_bias == "Bearish" else "score-neutral"

    st.markdown(
        f"""
        <div style="margin-bottom:0.45rem;">
         <span style="font-weight:700; color:#f4f7ff;">Softs</span>
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

    grains_wow_mean = grains_indicator_df["_wow_num"].mean()
    grains_avg_bias = "Bullish" if grains_wow_mean > 0 else "Bearish" if grains_wow_mean < 0 else "Neutral"
    grains_avg_class = "score-bullish" if grains_avg_bias == "Bullish" else "score-bearish" if grains_avg_bias == "Bearish" else "score-neutral"

    st.markdown(
        f"""
     <div style="margin-bottom:0.45rem;">
          <span style="font-weight:700; color:#f4f7ff;">Grains & Oilseeds</span>
           <span class="{grains_avg_class}" style="font-weight:700; margin-left:10px;">{grains_avg_bias}</span>
        </div>
     """,
     unsafe_allow_html=True,
    )

    st.dataframe(
        style_indicator_table(grains_indicator_df),
        use_container_width=True,
        hide_index=True,
        height=140,
    )

    st.markdown('</div>', unsafe_allow_html=True)

with top_right:
    signal_rows = [
        {"Variable": "Cotton Momentum", "Last": signals["cotton_momentum"], "Weight": weights["cotton_momentum"]},
        {"Variable": "Spread Structure", "Last": signals["spread_structure"], "Weight": weights["spread_structure"]},
        {"Variable": "Soft Complex", "Last": signals["soft_complex"], "Weight": weights["soft_complex"]},
        {"Variable": "Agri Complex", "Last": signals["agri_complex"], "Weight": weights["agri_complex"]},
        {"Variable": "Energy", "Last": signals["energy"], "Weight": weights["energy"]},
        {"Variable": "Macro", "Last": signals["macro"], "Weight": weights["macro"]},
    ]
    signal_df = pd.DataFrame(signal_rows)

    signal_df["Intensity (%)"] = (signal_df["Weight"] / signal_df["Weight"].sum() * 100).round(1)
    signal_df["_dod_num"] = signal_df["Last"].round(2)
    signal_df["_wow_num"] = (signal_df["Last"] * signal_df["Weight"]).round(2)
    signal_df["DOD"] = signal_df["_dod_num"].apply(lambda x: format_arrow_value(x, 2))
    signal_df["WOW"] = signal_df["_wow_num"].apply(lambda x: format_arrow_value(x, 2))
    signal_df["_bias_text"] = signal_df["Last"].apply(
        lambda x: "Bullish" if x > 0 else "Bearish" if x < 0 else "Neutral"
    )
    signal_df["Bias"] = signal_df["_bias_text"].apply(
        lambda x: f"↑ {x}" if x == "Bullish" else f"↓ {x}" if x == "Bearish" else f"→ {x}"
    )

    signal_display_df = signal_df[
        ["Variable", "Last", "Intensity (%)", "DOD", "WOW", "Bias", "_dod_num", "_wow_num", "_bias_text"]
    ].copy()

    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Signal Breakdown</div>', unsafe_allow_html=True)
    st.dataframe(
        style_indicator_table(signal_display_df),
        use_container_width=True,
        hide_index=True,
        height=255,
    )
    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Technicals (Monthly Chart)</div>', unsafe_allow_html=True)

    st.dataframe(
        style_indicator_table(technicals_monthly_df),
        use_container_width=True,
        hide_index=True,
        height=290,
    )

    st.markdown('</div>', unsafe_allow_html=True)

mid_left, mid_right = st.columns([0.5, 0.5], gap="large")

with mid_left:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)

    overview_left, overview_right = st.columns([0.48, 0.52], gap="large")

    with overview_left:
        st.markdown("**Others**")
        st.dataframe(
            style_indicator_table(others_table),
            use_container_width=True,
            hide_index=True,
            height=165,
        )

    with overview_right:
        st.markdown("**Energy**")
        st.dataframe(
            style_indicator_table(energy_table),
            use_container_width=True,
            hide_index=True,
            height=315,
        )

        st.markdown("**Metals**")
        st.dataframe(
        style_indicator_table(metals_table),
        use_container_width=True,
        hide_index=True,
        height=455,
    )

    st.markdown('</div>', unsafe_allow_html=True)

with mid_right:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">China Commodities</div>', unsafe_allow_html=True)

    st.markdown("**Energy**")
    st.dataframe(
        style_indicator_table(china_energy_table),
        use_container_width=True,
        hide_index=True,
        height=155,
    )

    st.markdown("**Metals**")
    st.dataframe(
        style_indicator_table(china_metals_table),
        use_container_width=True,
        hide_index=True,
        height=455,
    )

    st.markdown("**Agriculture / Softs / Oilseeds**")
    st.dataframe(
        style_indicator_table(china_agriculture_table),
        use_container_width=True,
        hide_index=True,
        height=470,
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.markdown('<div class="table-card">', unsafe_allow_html=True)

idx_col1, idx_col2, idx_col3 = st.columns(3)

with idx_col1:
    st.markdown("**Asia**")
    st.dataframe(
        style_indicator_table(asia_table),
        use_container_width=True,
        hide_index=True,
        height=385,
    )

with idx_col2:
    st.markdown("**The US**")
    st.dataframe(
        style_indicator_table(us_table),
        use_container_width=True,
        hide_index=True,
        height=385,
    )

with idx_col3:
    st.markdown("**Europe**")
    st.dataframe(
        style_indicator_table(europe_table),
        use_container_width=True,
        hide_index=True,
        height=385,
    )

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

bullish = sorted(
    [(k, v * norm_global[k]) for k, v in signals.items()],
    key=lambda x: x[1],
    reverse=True,
)
bearish = sorted(
    [(k, v * norm_global[k]) for k, v in signals.items()],
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