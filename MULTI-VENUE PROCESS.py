def get_ta(tv_sym: str, exchange: str, screener: str, interval):
    """
    Try TradingView_TA first. If that returns no indicators,
    switch to the local indicator engine (via Yahoo) and return
    a drop-in dummy TA object.
    """
    try:
        data = TA_Handler(symbol=tv_sym,
                          exchange=exchange,
                          screener=screener,
                          interval=interval).get_analysis()
        if data and data.indicators:
            return data
        logging.info(f"TV-TA had no indicators for {tv_sym}@{exchange}/{screener}")
    except Exception as e:
        logging.info(f"TV-TA failed for {tv_sym}@{exchange}/{screener}: {e}")

    # Fallback: use Yahoo data for continuous futures ('=F') if needed
    yf_sym = {
        "COPPER": "HG=F",
        "GOLD":   "GC=F",
        "SILVER": "SI=F",
        "PLATINUM":  "PL=F",
        "PALLADIUM": "PA=F",
        "ALUMINIUM": "ALI=F",
        "WHEAT":  "ZW=F",
    }.get(tv_sym.upper(), tv_sym)

    try:
        df   = local_fetch_ohlcv(yf_sym, _YF_INTERVAL[interval])
        ind  = local_indicators(df)
        reco = local_recommend(ind)
        logging.info(f"↻ Local TA generated for {tv_sym} ({yf_sym}) [{reco}]")
        return _DummyTA(ind, reco)
    except Exception as e:
        raise RuntimeError(f"Local TA failed for {tv_sym}: {e}")

def local_fetch_ohlcv(yf_symbol: str, interval: str, lookback=200):
    """
    Returns a DataFrame with OHLCV data as used by the local indicator engine.
    Uses a shorter period for equities versus futures/commodities.
    """
    period = "30d" if asset_class(yf_symbol).lower() == "equity" else "60d"
    hist = yf.download(
        yf_symbol,
        period   = period,
        interval = interval,
        progress = False,
        auto_adjust = False
    )
    if hist.empty:
        raise RuntimeError(f"no Yahoo data for {yf_symbol}/{interval}")
    hist.rename(columns=str.capitalize, inplace=True)  # e.g. Open, High, etc.
    return hist.tail(lookback)

def local_indicators(df: pd.DataFrame) -> dict:
    """
    Computes key technical indicators from OHLCV data.
    Uses moving averages, MACD, RSI, ATR, CCI, ADX, and Bollinger Bands
    to build a full indicator pack.
    """
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

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

    # Awesome Oscillator
    median = (high + low) / 2
    ao = _sma(median, 5).iloc[-1].item() - _sma(median, 34).iloc[-1].item()

    # RSI-14
    delta = close.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = _sma(up, 14) / _sma(dn, 14)
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
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean().iloc[-1].item()

    # Bollinger bands
    bb_basis = close.rolling(20).mean().iloc[-1].item()
    bb_std   = close.rolling(20).std().iloc[-1].item()
    bb_upper = bb_basis + 2 * bb_std
    bb_lower = bb_basis - 2 * bb_std

    return {
        "price":       close.iloc[-1].item(),
        "EMA9":        ema9,
        "EMA21":       ema21,
        "SMA50":       sma50,
        "SMA200":      sma200,
        "MACD.macd":   macd,
        "MACD.signal": macd_signal,
        "AO":          ao,
        "RSI":         rsi,
        "CCI20":       cci,
        "ATR":         atr,
        "ADX":         adx,
        "BB.upper":    bb_upper,
        "BB.lower":    bb_lower,
        "volume":      df["Volume"].iloc[-1].item(),
    }

def convert_to_yf_symbol(symbol: str) -> str:
    """
    Convert an internal symbol to a Yahoo Finance–compatible ticker.
    Follows explicit alias mapping, FX pair formatting, crypto conversion,
    and a generic USD rule.
    """
    if not symbol:
        return ""
    sym_up = str(symbol).upper()
    if sym_up in YF_ALIAS:
        return YF_ALIAS[sym_up]
    if len(sym_up) == 6 and sym_up.isalpha():
        return f"{sym_up}=X"
    crypto_syms = {"BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD"}
    if sym_up in crypto_syms:
        return sym_up.replace("USD", "-USD")
    if sym_up.endswith("USD") and sym_up not in {"AVGO", "AMD", "CRM"}:
        return sym_up.replace("USD", "-USD")
    return symbol

def fetch_ohlcv(symbol, resolution="HOUR"):
    """
    Fetch OHLCV data for a given symbol.
    Attempts to retrieve data via Capital.com (omitted here) and falls back to yfinance.
    """
    symbol = str(symbol)
    epic = lookup_epic(symbol)
    if not epic:
        print(f"❌ No EPIC for {symbol}")
        return pd.DataFrame()  # Return empty DataFrame if no EPIC found

    # Fallback to yfinance:
    print(f"⚠️ Falling back to yfinance for {symbol} with {resolution} resolution")
    try:
        interval_map = {
            "MINUTE_15": "15m",
            "MINUTE_30": "30m",
            "HOUR": "1h",
            "HOUR_4": "4h"
        }
        yf_interval = interval_map.get(resolution, "1h")
        ohlcv_df = yf.download(
            convert_to_yf_symbol(symbol),
            period="5d",
            interval=yf_interval,
            progress=False,
            auto_adjust=False
        )
        if ohlcv_df.empty:
            raise RuntimeError(f"no data from yfinance for {symbol}")
        ohlcv_df.rename(columns=lambda x: x.capitalize(), inplace=True)
        return ohlcv_df
    except Exception as e:
        print(f"❌ yfinance OHLCV error for {symbol}: {e}")
        raise RuntimeError(f"Unable to fetch OHLCV data for {symbol}")