import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.scoring import (
    classify_score,
    normalize_change_to_signal,
    weighted_composite_score,
)

st.set_page_config(
    page_title="Cotton Trading Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main {
        background-color: #0b1020;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1450px;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(21,29,48,0.95), rgba(12,18,34,0.95));
        border: 1px solid rgba(110,130,170,0.18);
        border-radius: 16px;
        padding: 14px 16px;
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f2f5ff;
        margin-bottom: 0.5rem;
    }
    .small-muted {
        color: #98a5c3;
        font-size: 0.9rem;
    }
    .signal-box {
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(180deg, rgba(21,29,48,0.95), rgba(12,18,34,0.95));
        border: 1px solid rgba(110,130,170,0.18);
        margin-bottom: 12px;
    }
    .signal-label {
        font-size: 0.9rem;
        color: #98a5c3;
        margin-bottom: 4px;
    }
    .signal-value {
        font-size: 2rem;
        font-weight: 800;
        color: #f2f5ff;
        margin-bottom: 4px;
    }
    .signal-sub {
        font-size: 0.95rem;
        color: #c5cde0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def seeded_rng() -> np.random.Generator:
    # stable enough within a minute, changes over time for demo realism
    seed = int(datetime.now().strftime("%Y%m%d%H%M"))
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


def score_to_intensity(score: float) -> float:
    # map [-3, +3] -> [0, 100]
    clipped = max(-3.0, min(3.0, score))
    return round(((clipped + 3.0) / 6.0) * 100.0, 1)


def sparkline_figure(values: pd.Series, line_color: str = "#7ad7f0") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines",
            line=dict(color=line_color, width=2),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=46,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def styled_line_chart(
    x,
    y,
    title: str,
    line_color: str = "#7ad7f0",
    fill: bool = False,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=line_color, width=2.5),
            fill="tozeroy" if fill else None,
            fillcolor="rgba(122,215,240,0.10)" if fill else None,
            name=title,
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16, color="#f2f5ff")),
        height=290,
        margin=dict(l=30, r=18, t=48, b=30),
        paper_bgcolor="#12192b",
        plot_bgcolor="#12192b",
        font=dict(color="#dfe6f5"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#9fb0d1"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(color="#9fb0d1"),
        ),
        showlegend=False,
    )
    return fig


# ---------- Mock market data ----------
@st.cache_data(ttl=120, show_spinner=False)
def build_demo_dataset():
    rng = seeded_rng()
    idx = make_time_index(72, freq_minutes=60)

    cotton = make_series(67.2, len(idx), drift=0.03, vol=0.45, rng=rng)
    dxy = make_series(103.8, len(idx), drift=-0.005, vol=0.08, rng=rng)
    bcom = make_series(126.0, len(idx), drift=0.01, vol=0.18, rng=rng)
    oil = make_series(77.5, len(idx), drift=0.01, vol=0.55, rng=rng)
    sugar = make_series(21.8, len(idx), drift=0.005, vol=0.08, rng=rng)
    coffee = make_series(188.0, len(idx), drift=0.02, vol=0.9, rng=rng)
    cocoa = make_series(9480, len(idx), drift=1.5, vol=90, rng=rng)
    corn = make_series(438, len(idx), drift=0.03, vol=1.9, rng=rng)
    soy = make_series(1160, len(idx), drift=0.08, vol=3.2, rng=rng)
    wheat = make_series(579, len(idx), drift=0.04, vol=2.3, rng=rng)

    may_jul = make_series(-0.65, len(idx), drift=0.005, vol=0.06, rng=rng)
    jul_dec = make_series(1.15, len(idx), drift=-0.003, vol=0.07, rng=rng)
    oi = make_series(99277, len(idx), drift=8, vol=120, rng=rng)
    vol = np.abs(make_series(23000, len(idx), drift=25, vol=1800, rng=rng))

    df = pd.DataFrame(
        {
            "CT1": cotton,
            "DXY": dxy,
            "BCOM": bcom,
            "CL1": oil,
            "SB1": sugar,
            "KC1": coffee,
            "CC1": cocoa,
            "C1": corn,
            "S1": soy,
            "W1": wheat,
            "MAY_JUL": may_jul,
            "JUL_DEC": jul_dec,
            "OPEN_INT": oi,
            "VOLUME": vol,
        },
        index=idx,
    )

    return df


df = build_demo_dataset()

# ---------- Compute current state ----------
latest = df.iloc[-1]
prev = df.iloc[-2]
week_ago = df.iloc[-6] if len(df) >= 6 else df.iloc[0]

def pct_change(a, b):
    if b == 0:
        return 0.0
    return ((a - b) / b) * 100.0

market_snapshot = {
    "Cotton Front Month": {
        "symbol": "CT1 Comdty",
        "last": latest["CT1"],
        "chg_pct": pct_change(latest["CT1"], prev["CT1"]),
        "weekly_delta": latest["CT1"] - week_ago["CT1"],
        "series": df["CT1"].tail(24),
    },
    "Dollar Index": {
        "symbol": "DXY Index",
        "last": latest["DXY"],
        "chg_pct": pct_change(latest["DXY"], prev["DXY"]),
        "weekly_delta": latest["DXY"] - week_ago["DXY"],
        "series": df["DXY"].tail(24),
    },
    "Bloomberg Commodity Index": {
        "symbol": "BCOM Index",
        "last": latest["BCOM"],
        "chg_pct": pct_change(latest["BCOM"], prev["BCOM"]),
        "weekly_delta": latest["BCOM"] - week_ago["BCOM"],
        "series": df["BCOM"].tail(24),
    },
    "Crude Oil": {
        "symbol": "CL1 Comdty",
        "last": latest["CL1"],
        "chg_pct": pct_change(latest["CL1"], prev["CL1"]),
        "weekly_delta": latest["CL1"] - week_ago["CL1"],
        "series": df["CL1"].tail(24),
    },
    "Sugar": {
        "symbol": "SB1 Comdty",
        "last": latest["SB1"],
        "chg_pct": pct_change(latest["SB1"], prev["SB1"]),
        "weekly_delta": latest["SB1"] - week_ago["SB1"],
        "series": df["SB1"].tail(24),
    },
    "Coffee": {
        "symbol": "KC1 Comdty",
        "last": latest["KC1"],
        "chg_pct": pct_change(latest["KC1"], prev["KC1"]),
        "weekly_delta": latest["KC1"] - week_ago["KC1"],
        "series": df["KC1"].tail(24),
    },
    "Cocoa": {
        "symbol": "CC1 Comdty",
        "last": latest["CC1"],
        "chg_pct": pct_change(latest["CC1"], prev["CC1"]),
        "weekly_delta": latest["CC1"] - week_ago["CC1"],
        "series": df["CC1"].tail(24),
    },
    "Corn": {
        "symbol": "C1 Comdty",
        "last": latest["C1"],
        "chg_pct": pct_change(latest["C1"], prev["C1"]),
        "weekly_delta": latest["C1"] - week_ago["C1"],
        "series": df["C1"].tail(24),
    },
    "Soybeans": {
        "symbol": "S1 Comdty",
        "last": latest["S1"],
        "chg_pct": pct_change(latest["S1"], prev["S1"]),
        "weekly_delta": latest["S1"] - week_ago["S1"],
        "series": df["S1"].tail(24),
    },
    "Wheat": {
        "symbol": "W1 Comdty",
        "last": latest["W1"],
        "chg_pct": pct_change(latest["W1"], prev["W1"]),
        "weekly_delta": latest["W1"] - week_ago["W1"],
        "series": df["W1"].tail(24),
    },
    "May/Jul Spread": {
        "symbol": "CTK/CTN",
        "last": latest["MAY_JUL"],
        "chg_pct": pct_change(latest["MAY_JUL"], prev["MAY_JUL"]) if prev["MAY_JUL"] != 0 else 0.0,
        "weekly_delta": latest["MAY_JUL"] - week_ago["MAY_JUL"],
        "series": df["MAY_JUL"].tail(24),
    },
    "Jul/Dec Spread": {
        "symbol": "CTN/CTZ",
        "last": latest["JUL_DEC"],
        "chg_pct": pct_change(latest["JUL_DEC"], prev["JUL_DEC"]) if prev["JUL_DEC"] != 0 else 0.0,
        "weekly_delta": latest["JUL_DEC"] - week_ago["JUL_DEC"],
        "series": df["JUL_DEC"].tail(24),
    },
}

# ---------- Signals ----------
signals = {
    "cotton_momentum": normalize_change_to_signal(pct_change(latest["CT1"], df["CT1"].iloc[-6]), scale=1.8),
    "spread_structure": normalize_change_to_signal((latest["MAY_JUL"] * -12.0), scale=1.0),
    "soft_complex": np.mean([
        normalize_change_to_signal(pct_change(latest["SB1"], df["SB1"].iloc[-6]), scale=2.2),
        normalize_change_to_signal(pct_change(latest["KC1"], df["KC1"].iloc[-6]), scale=2.2),
        normalize_change_to_signal(pct_change(latest["CC1"], df["CC1"].iloc[-6]), scale=2.2),
    ]),
    "agri_complex": np.mean([
        normalize_change_to_signal(pct_change(latest["C1"], df["C1"].iloc[-6]), scale=2.0),
        normalize_change_to_signal(pct_change(latest["S1"], df["S1"].iloc[-6]), scale=2.0),
        normalize_change_to_signal(pct_change(latest["W1"], df["W1"].iloc[-6]), scale=2.0),
    ]),
    "energy": normalize_change_to_signal(pct_change(latest["CL1"], df["CL1"].iloc[-6]), scale=2.0),
    "macro": np.mean([
        -normalize_change_to_signal(pct_change(latest["DXY"], df["DXY"].iloc[-6]), scale=0.8),
        normalize_change_to_signal(pct_change(latest["BCOM"], df["BCOM"].iloc[-6]), scale=1.5),
    ]),
}

weights = {
    "cotton_momentum": 0.30,
    "spread_structure": 0.22,
    "soft_complex": 0.14,
    "agri_complex": 0.12,
    "energy": 0.12,
    "macro": 0.10,
}

composite_score = weighted_composite_score(signals, weights)
signal_label = classify_score(composite_score)

# ---------- Header ----------
st.title("Cotton Trading Dashboard")
st.caption("Preview version — simulated market data, production structure, 120s refresh target.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active Contract", "CT1 Comdty", "Front Month")
c2.metric("Composite Score", f"{composite_score:+.2f}", signal_label)
c3.metric("Cotton Front Month", f"{latest['CT1']:.2f} c/lb", f"{pct_change(latest['CT1'], prev['CT1']):+.2f}%")
c4.metric("Last Update", datetime.now().strftime("%H:%M:%S"), "Demo mode")

st.divider()

# ---------- Main layout ----------
left, right = st.columns([0.44, 0.56], gap="large")

with left:
    st.markdown('<div class="panel-title">Cotton #2 Indicators</div>', unsafe_allow_html=True)
    st.caption("Prototype scoring table built around cotton, spreads, softs, agriculture, energy and macro drivers.")

    rows = []
    for name, item in market_snapshot.items():
        direction = "Bullish" if item["weekly_delta"] > 0 else "Bearish" if item["weekly_delta"] < 0 else "Neutral"
        # for DXY, rising is bearish cotton
        if name == "Dollar Index":
            direction = "Bearish" if item["weekly_delta"] > 0 else "Bullish" if item["weekly_delta"] < 0 else "Neutral"

        proxy_score = score_to_intensity(
            normalize_change_to_signal(item["weekly_delta"], scale=max(abs(item["last"]) * 0.01, 0.5))
        )
        rows.append(
            {
                "Variable": name,
                "Ticker": item["symbol"],
                "Last": round(item["last"], 2),
                "Intensity": proxy_score,
                "Vs Last Week": round(item["weekly_delta"], 2),
                "Bias": direction,
            }
        )

    indicator_df = pd.DataFrame(rows).sort_values("Intensity", ascending=False)
    st.dataframe(indicator_df, use_container_width=True, height=470)

    st.markdown('<div class="panel-title" style="margin-top:0.8rem;">Driver Breakdown</div>', unsafe_allow_html=True)

    driver_df = pd.DataFrame(
        [
            {"Driver": "Cotton Momentum", "Signal": signals["cotton_momentum"], "Weight": weights["cotton_momentum"]},
            {"Driver": "Spread Structure", "Signal": signals["spread_structure"], "Weight": weights["spread_structure"]},
            {"Driver": "Soft Complex", "Signal": signals["soft_complex"], "Weight": weights["soft_complex"]},
            {"Driver": "Agri Complex", "Signal": signals["agri_complex"], "Weight": weights["agri_complex"]},
            {"Driver": "Energy", "Signal": signals["energy"], "Weight": weights["energy"]},
            {"Driver": "Macro", "Signal": signals["macro"], "Weight": weights["macro"]},
        ]
    )
    driver_df["Contribution"] = driver_df["Signal"] * driver_df["Weight"]
    driver_df["Contribution"] = driver_df["Contribution"].round(2)
    driver_df["Signal"] = driver_df["Signal"].round(2)
    driver_df["Weight"] = driver_df["Weight"].round(2)

    st.dataframe(driver_df, use_container_width=True, hide_index=True)

with right:
    st.markdown(
        f"""
        <div class="signal-box">
            <div class="signal-label">Composite Directional Bias</div>
            <div class="signal-value">{signal_label}</div>
            <div class="signal-sub">Score {composite_score:+.2f} based on weighted cross-market drivers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top1, top2 = st.columns(2)
    with top1:
        st.plotly_chart(
            styled_line_chart(df.index, df["CT1"], "Cotton Front Month (CT1)", fill=True),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with top2:
        st.plotly_chart(
            styled_line_chart(df.index, df["DXY"], "Dollar Index (DXY)", line_color="#f3b463"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    bottom1, bottom2 = st.columns(2)
    with bottom1:
        st.plotly_chart(
            styled_line_chart(df.index, df["MAY_JUL"], "May / Jul Spread", line_color="#7ef0a9"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with bottom2:
        corr = df["CT1"].rolling(12).corr(df["DXY"]).fillna(0)
        st.plotly_chart(
            styled_line_chart(df.index, corr, "Rolling Corr: CT1 vs DXY", line_color="#e38cff"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

# ---------- Bottom section ----------
st.divider()
b1, b2, b3 = st.columns(3)

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

with b1:
    st.markdown('<div class="panel-title">Top Bullish Drivers</div>', unsafe_allow_html=True)
    for k, v in bullish[:3]:
        st.metric(pretty_names[k], f"{v:+.2f}")

with b2:
    st.markdown('<div class="panel-title">Top Bearish Drivers</div>', unsafe_allow_html=True)
    for k, v in bearish[:3]:
        st.metric(pretty_names[k], f"{v:+.2f}")

with b3:
    st.markdown('<div class="panel-title">Market Context</div>', unsafe_allow_html=True)
    st.metric("Open Interest", f"{int(latest['OPEN_INT']):,}", f"{int(latest['OPEN_INT'] - prev['OPEN_INT']):+d}")
    st.metric("Volume", f"{int(latest['VOLUME']):,}", f"{int(latest['VOLUME'] - prev['VOLUME']):+d}")
    st.metric("BCOM", f"{latest['BCOM']:.2f}", f"{pct_change(latest['BCOM'], prev['BCOM']):+.2f}%")