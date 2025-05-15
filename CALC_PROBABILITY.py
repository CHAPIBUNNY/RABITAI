#Turns indicator values into a probability using a logistic regression model.

def calculate_trade_probability(
    indicators: Dict[str, float],
    direction: str,
    tf_summary_votes: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
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
        probability in percent [0.0–100.0]
    """
    # --- Defaults & normalizations ---
    adx_raw = indicators.get("ADX", 0.0)  # Default to 0.0 if missing
    adx = np.clip(adx_raw / 50.0, 0.0, 1.0)

    weights = weights or {}
    adx_weight = weights.get("adx", 0.2)  # Default weight for ADX

    required_features = ["adx", "ema_macd", "ao", "cci", "rsi", "tf", "entropy"]
    for feature in required_features:
        if feature not in weights:
            weights[feature] = 0.0  # Ensure all required features are present

    # EMA+MACD alignment: 1 if both align with direction, else 0
    ema9, ema21 = indicators.get("EMA9", 0.0), indicators.get("EMA21", 0.0)
    macd, macd_sig = indicators.get("MACD.macd", 0.0), indicators.get("MACD.signal", 0.0)
    ema_macd = 0.0
    if direction == "bullish" and ema9 > ema21 and macd > macd_sig:
        ema_macd = 1.0
    elif direction == "bearish" and ema9 < ema21 and macd < macd_sig:
        ema_macd = 1.0

    # Awesome Oscillator
    ao = indicators.get("AO", 0.0)
    ao_norm = 1.0 if (direction == "bullish" and ao > 0) or (direction == "bearish" and ao < 0) else 0.0

    # CCI20 scaled to [0,1], with strong-trend override
    cci_raw = indicators.get("CCI20", 0.0)
    cci = np.clip(abs(cci_raw) / 200.0, 0.0, 1.0)
    if adx > 0.5 and abs(cci_raw) > 100:
        cci = 1.0

    # RSI strength mapped from 30–70 → 0–1, clipped; strong-trend override
    rsi_raw = indicators.get("RSI", 50.0)
    rsi = np.clip((rsi_raw - 30.0) / 40.0, 0.0, 1.0)
    if adx > 0.5 and rsi_raw > 70:
        rsi = 1.0

    # Entropy20 (normalize roughly to 0‑1 range by /5.0, clamp)
    entropy_raw = indicators.get("Entropy20", 0.0)
    entropy = np.clip(entropy_raw / 5.0, 0.0, 1.0)

    # Multi‑TF buy ratio
    tf = 0.0
    if tf_summary_votes:
        votes = [v.upper() for v in tf_summary_votes]
        # count STRONG votes as double
        buys  = sum(1 for v in votes if v == "BUY") + 2 * sum(1 for v in votes if v == "STRONG_BUY")
        sells = sum(1 for v in votes if v == "SELL") + 2 * sum(1 for v in votes if v == "STRONG_SELL")
        tf = (buys - sells) / (len(votes) * 2)  # normalize between -1 and +1, then shift to 0–1 if needed