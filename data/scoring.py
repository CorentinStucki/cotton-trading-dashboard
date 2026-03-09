import pandas as pd


def clamp_score(x: float, low: float = -3.0, high: float = 3.0) -> float:
    return max(low, min(high, x))


def normalize_change_to_signal(change_pct: float, scale: float = 1.5) -> float:
    """
    Maps % change into a directional signal roughly in [-3, +3].
    """
    if pd.isna(change_pct):
        return 0.0
    return clamp_score(change_pct / scale)


def weighted_composite_score(signals: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values()) if weights else 1.0
    if total_weight == 0:
        return 0.0
    return sum(signals[k] * weights.get(k, 0.0) for k in signals) / total_weight


def classify_score(score: float) -> str:
    if score >= 1.5:
        return "Strong Bullish"
    if score >= 0.5:
        return "Bullish"
    if score <= -1.5:
        return "Strong Bearish"
    if score <= -0.5:
        return "Bearish"
    return "Neutral"