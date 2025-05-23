def local_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].squeeze()  # ensure 1D series
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    # Moving averages
    ema9   = _ema(close, 9).iloc[-1].item()
    ema21  = _ema(close, 21).iloc[-1].item()
    sma50  = _sma(close, 50).iloc[-1].item()
    sma200 = _sma(close, 200).iloc[-1].item()

    # MACD (12-26-9)
    macd_line   = _ema(close, 12) - _ema(close, 26)
    macd_sig    = _ema(macd_line, 9)
    macd        = macd_line.iloc[-1].item()
    macd_signal = macd_sig.iloc[-1].item()

    # Awesome Oscillator (median-price SMA5 – SMA34)
    median = (high + low) / 2
    ao = _sma(median, 5).iloc[-1].item() - _sma(median, 34).iloc[-1].item()

    # RSI-14
    delta = close.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs  = _sma(up, 14) / _sma(dn, 14)
    rsi = 100 - 100 / (1 + rs.iloc[-1].item())

    # ATR-14 (Wilder)
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1].item()

    # CCI-20
    tp  = (high + low + close) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    cci = cci.iloc[-1].item()

    # ADX-14
    plus_dm  = np.where((high.diff() > 0) & (high.diff() > -low.diff()), high.diff(), 0)
    minus_dm = np.where((-low.diff() > 0) & (-low.diff() > high.diff()), -low.diff(), 0)
    tr14  = tr.rolling(14).sum()
    plus_di  = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean().iloc[-1].item()

    # Bollinger Bands (20-period)
    bb_basis = close.rolling(20).mean().iloc[-1].item()
    bb_std   = close.rolling(20).std().iloc[-1].item()
    bb_upper = bb_basis + 2 * bb_std
    bb_lower = bb_basis - 2 * bb_std

    return {
        "price":        close.iloc[-1].item(),
        "EMA9":         ema9,
        "EMA21":        ema21,
        "SMA50":        sma50,
        "SMA200":       sma200,
        "MACD.macd":    macd,
        "MACD.signal":  macd_signal,
        "AO":           ao,
        "RSI":          rsi,
        "CCI20":        cci,
        "ATR":          atr,
        "ADX":          adx,
        "BB.upper":     bb_upper,
        "BB.lower":     bb_lower,
        "volume":       df["Volume"].iloc[-1].item(),
    }