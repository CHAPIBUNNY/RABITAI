def calculate_trade_probability(
    indicators: dict,
    direction: str,
    tf_summary_votes: list[str] | None = None,
    weights: dict | None = None,
    bias: float = 0.1
) -> float:
    """
    Estimate trade success probability (0–100) via a logistic on normalized features.

    Args:
        indicators: must contain "ADX", "EMA9", "EMA21", "MACD.macd", "MACD.signal",
                    "AO", "CCI20", "RSI", and optionally "Entropy20" and "vix".
        direction: 'bullish' or 'bearish'
        tf_summary_votes: list of "BUY"/"SELL"/... from different timeframes
        weights: optional feature weights; keys: adx, ema_macd, ao, cci, rsi, tf, entropy
        bias: intercept term for the logit

    Returns:
        probability (edge score) in percent [0.0–100.0]
    """
    adx_raw = indicators.get("ADX", 0.0)
    adx = np.clip(adx_raw / 50.0, 0.0, 1.0)

    weights = weights or {}
    for feature in ["adx", "ema_macd", "ao", "cci", "rsi", "tf", "entropy"]:
        if feature not in weights:
            weights[feature] = 0.0

    default_weights = {
        "adx":      0.20,
        "ema_macd": 0.15,
        "ao":       0.10,
        "cci":      0.05,
        "rsi":      0.10,
        "tf":       0.10,
        "entropy":  0.10,
    }
    if sum(weights.get(f, 0.0) for f in ["adx", "ema_macd", "ao", "cci", "rsi", "tf", "entropy"]) == 0.0:
        weights = default_weights.copy()

    ema9, ema21 = indicators.get("EMA9", 0.0), indicators.get("EMA21", 0.0)
    macd, macd_sig = indicators.get("MACD.macd", 0.0), indicators.get("MACD.signal", 0.0)
    ema_macd = 1.0 if (direction == "bullish" and ema9 > ema21 and macd > macd_sig) or \
                       (direction == "bearish" and ema9 < ema21 and macd < macd_sig) else 0.0

    ao = indicators.get("AO", 0.0)
    ao_norm = 1.0 if (direction == "bullish" and ao > 0) or (direction == "bearish" and ao < 0) else 0.0

    cci_raw = indicators.get("CCI20", 0.0)
    cci = np.clip(abs(cci_raw) / 200.0, 0.0, 1.0)
    if adx > 0.5 and abs(cci_raw) > 100:
        cci = 1.0

    rsi_raw = indicators.get("RSI", 50.0)
    rsi = np.clip((rsi_raw - 30.0) / 40.0, 0.0, 1.0)
    if adx > 0.5 and rsi_raw > 70:
        rsi = 1.0

    entropy_raw = indicators.get("Entropy20", 0.0)
    entropy = np.clip(entropy_raw / 5.0, 0.0, 1.0)

    tf = 0.0
    if tf_summary_votes:
        votes = [v.upper() for v in tf_summary_votes]
        buys  = sum(1 for v in votes if v == "BUY") + 2 * sum(1 for v in votes if v == "STRONG_BUY")
        sells = sum(1 for v in votes if v == "SELL") + 2 * sum(1 for v in votes if v == "STRONG_SELL")
        tf = (buys - sells) / (len(votes) * 2)

    feats = {
        "adx":      adx,
        "ema_macd": ema_macd,
        "ao":       ao_norm,
        "cci":      cci,
        "rsi":      rsi,
        "tf":       tf,
        "entropy":  entropy,
    }

    scaling_factor = 1 + (adx - 0.5)
    scaling_factor = np.clip(scaling_factor, 0.5, 1.5)

    adx_component = weights["adx"] * feats["adx"]
    non_adx_sum = (
        weights["ema_macd"] * feats["ema_macd"] +
        weights["ao"] * feats["ao"] +
        weights["cci"] * feats["cci"] +
        weights["rsi"] * feats["rsi"] +
        weights["tf"] * feats["tf"] +
        weights["entropy"] * feats["entropy"]
    )
    scaled_non_adx = scaling_factor * non_adx_sum

    print(f"[DEBUG] Bias: {bias}")
    print(f"[DEBUG] ADX component: {adx_component} (weights['adx'] * feats['adx'])")
    print(f"[DEBUG] Scaling factor: {scaling_factor}")
    print(f"[DEBUG] Non-ADX sum: {non_adx_sum}")
    print(f"[DEBUG] Scaled non-ADX contribution: {scaled_non_adx}")

    logit = bias + adx_component + scaled_non_adx
    print(f"[DEBUG] Final logit: {logit}")

    prob = 1.0 / (1.0 + math.exp(-logit))
    return round(float(np.clip(prob * 100, 0, 100)), 2)