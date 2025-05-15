# Extracts features, computes confidence, and generates the core signal for each ticker.

def analyze_ticker(symbol: str) -> Tuple[float, float, str, str, float, float, float, float]:
    """
    Full AI/TA analysis for one ticker.
    Returns (confidence, probability).  Skips gracefully on bad feeds.
    """
    try:
        global dynamic_weights, top_headline, charts, confidence_history
        sym = symbol
        asset_cls = asset_class(sym)                # <-- NEW
        rules = _rules_for(sym)                     # <-- NEW
        print(f"   ↳ asset class = {asset_cls}")

        # ---------- ROUTING ----------
        tv_sym, exchange, screener = tv_symbol_info(symbol)

        print(f"\n📈 Analyzing ticker: {symbol}")
        print(f"   ↳ routed to {tv_sym}@{exchange}/{screener}")

        # ---------- TradingView TA -----------
        def safe_ta(sym, exch, scrn, ivl):
            try:
                h = TA_Handler(symbol=sym, exchange=exch,
                            screener=scrn, interval=ivl)
                data = h.get_analysis()
                return data if data and data.indicators else None
            except Exception as e:
                logging.warning(f"TV-TA fail {sym}@{exch}/{scrn}/{ivl}: {e}")
                return None

        try:
            tf_15m = get_ta(tv_sym, exchange, screener, TVI.INTERVAL_15_MINUTES)
            tf_30m = get_ta(tv_sym, exchange, screener, TVI.INTERVAL_30_MINUTES)
            tf_1h  = get_ta(tv_sym, exchange, screener, TVI.INTERVAL_1_HOUR)
            tf_4h  = get_ta(tv_sym, exchange, screener, TVI.INTERVAL_4_HOURS)
        except RuntimeError as e:
            print(f"⚠️ Skipping {symbol}: {e}")
            return 0.0, 50.0, None, None, 0.0, 0.0, 0.0, 0.0

        # ---------- INDICATORS ----------
        indicators = {k: ensure_scalar(v) for k, v in tf_1h.indicators.items()}
        print(f"   ↳ indicators pulled: {list(indicators)[:8]}…")

        # ---------- PRICE FEED ----------
        _epic_cache: Dict[str, str] = {}
        def get_epic(sym):
            if sym not in _epic_cache:
                _epic_cache[sym] = lookup_epic(sym)
            return _epic_cache[sym]

        realtime_price = get_realtime_price(symbol, epic=get_epic(symbol))
        if realtime_price is None:
            print(f"⚠️ Skipping {symbol}: real-time price unavailable.")
            return 0.0, 50.0, None, None, 0.0, 0.0, 0.0, 0.0

        price = realtime_price
        indicators["price"] = realtime_price
        print(f"   ↳ real-time price = {realtime_price:.2f}")

        # ---------- OHLCV ----------
        ohlcv_df = fetch_ohlcv(symbol)
        if ohlcv_df is None or ohlcv_df.empty:
            print(f"⚠️ Skipping {symbol}: OHLCV feed missing.")
            return 0.0, 50.0, None, None, 0.0, 0.0, 0.0, 0.0

        # Compute EMA9 and EMA21 directly from real OHLCV data.
        if "Close" not in ohlcv_df.columns:
            raise ValueError("OHLCV does not contain 'Close' values; cannot compute EMAs.")
        indicators["EMA9"] = ohlcv_df["Close"].ewm(span=9, adjust=False).mean().iloc[-1].item()
        indicators["EMA21"] = ohlcv_df["Close"].ewm(span=21, adjust=False).mean().iloc[-1].item()

        if "ADX" not in indicators or not indicators["ADX"]: 
            adx_calc = calculate_adx(ohlcv_df) 
            if hasattr(adx_calc, "iloc"): 
                adx_calc = adx_calc.iloc[-1] 
            indicators["ADX"] = float(adx_calc)  # Ensure ADX is a scalar float
            print(f" ↳ ADX computed locally = {adx_calc:.1f}")

        vbp_stats = calc_vbp(ohlcv_df, bins=20)
        indicators.update(vbp_stats)  

        latest_volume = int(ensure_scalar(ohlcv_df['Volume'].iloc[-1]))  # use most recent bar
        indicators["volume"] = latest_volume

        # --- ATR Calculation: ALWAYS assign atr (even if already in indicators) ---
        atr = to_scalar(indicators.get("ATR", price * 0.01))
        if 'ATR' not in indicators or not indicators["ATR"]:
            indicators["ATR"] = atr
        print(f"ATR for {symbol}: {atr}")

        # --- Log indicator data into rl_trainingsheet.csv for RL retraining ---
        data_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": parse_number(indicators.get("price", 0)),
            "SMA50": parse_number(indicators.get("SMA50", 0)),
            "SMA200": parse_number(indicators.get("SMA200", 0)),
            "EMA9": parse_number(indicators.get("EMA9", 0)),
            "EMA21": parse_number(indicators.get("EMA21", 0)),
            "MACD.macd": parse_number(indicators.get("MACD.macd", 0)),
            "MACD.signal": parse_number(indicators.get("MACD.signal", 0)),
            "AO": parse_number(indicators.get("AO", 0)),
            "CCI20": parse_number(indicators.get("CCI20", 0)),
            "RSI": parse_number(indicators.get("RSI", 50)),
            "Stoch.RSI": parse_number(indicators.get("Stoch.RSI", 0)),
            "ATR": parse_number(indicators.get("ATR", 0)),
            "BB.lower": parse_number(indicators.get("BB.lower", 0)),
            "volume": parse_number(indicators.get("volume", 0)),
            "vix": parse_number(indicators.get("vix", 0)),
            "success": 0
        }

        print(f"Logging data for {symbol} into rl_trainingsheet.csv...")
        csv_path = Path("rl_trainingsheet.csv")
        is_new_file = not csv_path.exists()
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data_row.keys()))
            if is_new_file:
                writer.writeheader()
            writer.writerow(data_row)
            print(f"✅ Successfully appended data for {symbol} to rl_trainingsheet.csv")

        multi_summary = f"15m: {tf_15m.summary.get('RECOMMENDATION', 'N/A')}, 30m: {tf_30m.summary.get('RECOMMENDATION', 'N/A')}, 1H: {tf_1h.summary.get('RECOMMENDATION', 'N/A')}, 4H: {tf_4h.summary.get('RECOMMENDATION', 'N/A')}"
        print(f"Multi-timeframe summary for {symbol}: {multi_summary}")
        
        if not all([
            realtime_price,
            ohlcv_df is not None and not ohlcv_df.empty
        ]):
            print(f"⚠️ Missing data for {symbol}. Skipping analysis.")
            return confidence, 50  

        sma_50 = float(indicators.get("SMA50", 0))
        sma_200 = float(indicators.get("SMA200", 0))
        macd = float(indicators.get("MACD.macd", 0))
        macd_sig = float(indicators.get("MACD.signal", 0))
        ao = float(indicators.get("AO", 0))
        cci = float(indicators.get("CCI20", 0))
        rsi = float(indicators.get("RSI", 50))
        bb_percent = float(indicators.get("BB.lower", 0))
        atr_val = float(indicators.get("ATR", realtime_price * 0.01))
        ema_9 = float(indicators.get("EMA9", 0))
        ema_21 = float(indicators.get("EMA21", 0))
        stoch_rsi = float(indicators.get("Stoch.RSI", 0))
        confidence, direction = 0, None

        # --- ENTROPY (20‑bar rolling) ---------------------------------------
        # log‑returns → last 20 bars → Shannon entropy
        returns_20 = np.log(ohlcv_df["Close"]).diff().dropna().tail(20)
        ent_20     = calc_entropy(returns_20)        # nats
        indicators["Entropy20"] = ent_20
        print(f"Entropy(20) for {symbol}: {ent_20:.3f} nats")

        .......