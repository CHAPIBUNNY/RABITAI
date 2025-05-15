# **signals.py – Internal Design Reference**

> *Scope –* formulation, signal generation, modelling loop, LLM layer, news‑sentiment, weights & probabilities.

---

## 1 · High‑level flow

```text
raw market feeds  ─┐
TA indicators (TV) ─┼─►  analyze_ticker()                           ←── dynamic RF weights (optional retrain)
OHLCV history      ─┘          │
                               ├─► confidence builder  ─┐
                               │                       ├─► size / risk layer (not covered here)
  GPT‑4 meta‑signal ◄──────────┘
                               │
                               └─► calculate_trade_probability() ─► P(success)
                                                                     (0–100)
```

All downstream e‑mails / webhooks take **`tech_conf`**, **`probability`**, the LLM verdict and an English rationale.

---

## 2 · Feature formulation & normalisation

| Feature        | Raw source                                     | Normalisation *f(x)* → \[0, 1] | Notes                      |               |                                   |
| -------------- | ---------------------------------------------- | ------------------------------ | -------------------------- | ------------- | --------------------------------- |
| **ADX**        | `indicators["ADX"] / 50`                       | clipped 0‑1                    | Trend‑strength scaler      |               |                                   |
| **EMA / MACD** | `½·sign(EMA9 – EMA21) + ½·sign(MACD – signal)` | {‑1, 0, +1} → {0, 0.5, 1}      | Momentum consensus         |               |                                   |
| **AO**         | \`tanh(                                        | AO                             |  / 50)\`                   | symmetric 0‑1 | —                                 |
| **CCI20**      | \`                                             | CCI                            | / 200\` (cap)              | capped        | Cycle extremes (=1 if strong ADX) |
| **RSI**        | `(RSI – 30) / 40` (cap)                        | capped                         | Band widens when ADX > 0.5 |               |                                   |
| **TF vote**    | `(BUY – SELL) / (2·n)`  (STRONG = 2 votes)     | –1…+1 → 0‑1                    | Sentiment of 4 TFs         |               |                                   |
| **Entropy20**  | `Entropy20 / 5`                                | clipped                        | Regime predictability      |               |                                   |

### Entropy bias

$\displaystyle\;\Delta C_{\text{entropy}}\;=\;\frac{E_{\text{ref}}-E_{20}}{E_{\text{ref}}}\times0.13$

`E_ref` is asset‑class‑specific (1.60 … 2.30).

---

## 3 · Confidence pipeline *(variable `tech_conf`)*

1. **Start at 0**
2. Add *entropy bias* (formula above).
3. **Momentum gate** – EMA + MACD agreement ± 0.20.
4. **RSI gate** – abort if outside neutral band (class‑dependent).
5. **Multi‑TF vote**
   $\Delta C_{\text{tf}} = 0.19\,(N_{BUY}-N_{SELL})$
6. **Client‑sentiment bias**
   $\Delta C_{\text{sent}} = 0.15\,\frac{\text{score}-50}{50}$
7. Volume, ATR break‑out, VBP gaps, etc. (±0.05 … 0.15 each).
8. GPT‑4 meta‑signal **+0.20** for *BUY*, **‑0.20** for *SELL*.
9. Clamp to **\[‑1 … +1]**.

**Early aborts**: missing feeds · RSI out‑of‑band · 3‑vs‑1 TF mismatch.

---

## 4 · Probability model – `calculate_trade_probability()`

### 4.1 Feature vector

$\mathbf f = \bigl[f_{ADX},\;f_{EMA/MACD},\;f_{AO},\;f_{CCI},\;f_{RSI},\;f_{TF},\;f_{Entropy}\bigr]$

### 4.2 Weight vector *(static / RF‑learned)*

```python
weights = {
    "adx": 0.20, "ema_macd": 0.15, "ao": 0.10,
    "cci": 0.05, "rsi": 0.10, "tf": 0.10, "entropy": 0.10,
}
```

### 4.3 Dynamic trend scaling

$s = \operatorname{clip}\bigl(1 + (f_{ADX}-0.5),\;0.5,\;1.5\bigr)$

### 4.4 Logit & sigmoid

$\text{logit} = b + w_{ADX}f_{ADX} + s\sum_{i\neq ADX} w_i f_i$

$P(\text{success}) = \sigma(\text{logit}) = \frac{1}{1+e^{-\text{logit}}}$

Probability is mapped to **0–100 %** and nudged ±10 % by an ATR sanity factor (if ATR/price deviates from 2 %).

---

## 5 · LLM layer

### 5.1 `generate_meta_signal()`

* Builds compact JSON prompt with **symbol, price, SMA50/200, MACD, AO, CCI20, RSI, EMA9/21, ADX, ATR, VIX, PoC, HVN band…**
* GPT‑4 must respond:

```json
{"signal":"BUY|SELL|HOLD","price":2568.11,"reason":"…"}
```

* `signal` feeds back ±0.20 into confidence.
* `reason` is logged / emailed.

### 5.2 `generate_llm_rationale()`

Lightweight call that turns the final decision into a plain‑English blurb for dashboards & reports.

---

## 6 · News & sentiment inputs

| Source                       | Function                   | Usage                                                      |
| ---------------------------- | -------------------------- | ---------------------------------------------------------- |
| Capital.com client positions | `fetch_client_sentiment()` | Produces **`sentiment_score`** (0–100) → ΔC<sub>sent</sub> |
| RSS headline                 | `get_top_headline()`       | Added to GPT prompt (context only)                         |
| VIX                          | `get_vix_value()`          | Narrative in LLM rationale                                 |

---

## 7 · Weight management & retraining

* Each tick logs **17 features + success flag** to `rl_trainingsheet.csv`.
* Nightly batch fits a **Random Forest**, extracts feature importances → writes **`dynamic_weights`**.
* If RF fails or data is sparse, engine falls back to static weights (see §4.2).
* `dynamic_weights` are injected into `calculate_trade_probability()` on the next tick.

---

## 8 · Signal‑generation pseudo‑code

```python
for symbol in watchlist:
    indicators = fetch_tv_indicators(symbol)  # TV TA + OHLCV
    if rsi_outside_band(indicators):
        continue

    confidence = build_confidence(indicators)
    if tf_vote_mismatch(indicators):
        continue

    probability = calculate_trade_probability(indicators)
    llm_signal, llm_reason = generate_meta_signal(symbol, indicators, probability)
    confidence += 0.20 * sign(llm_signal)

    send_webhook(
        symbol=symbol,
        confidence=confidence,
        probability=probability,
        reason=llm_reason
    )
```

Developers can swap indicators, adjust weight learning, or disable gates without touching the outer orchestration.

---

## Appendix A · Formula quick‑grab

| Name                 | Formula                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| Entropy bias         | \$\displaystyle \Delta C = \frac{E\_{ref}-E\_{20}}{E\_{ref}}\times0.13\$ |
| TF vote ΔC           | \$0.19,(N\_{BUY}-N\_{SELL})\$                                            |
| Sentiment ΔC         | \$0.15,\dfrac{\text{score}-50}{50}\$                                     |
| Scaling factor \$s\$ | \$s = \text{clip}\bigl(1 + (f\_{ADX}-0.5),,0.5,,1.5\bigr)\$              |
| Logit                | \$b + w\_{ADX}f\_{ADX} + s\sum w\_i f\_i\$                               |
| Probability          | \$\sigma(\text{logit}) = \dfrac{1}{1+e^{-\text{logit}}}\$                |

---
