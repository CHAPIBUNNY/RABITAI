# Entropy as a Predictability Measure in Algorithmic Trading

## 1. Introduction  
In modern algorithmic trading, understanding market regimes—trending, mean-reverting, or choppy—is critical for crafting reliable signals and managing risk. **Entropy**, a core concept from information theory, quantifies the unpredictability (or “randomness”) of a time series. RabbitAI leverages rolling-window Shannon entropy on log-returns to gauge how “noisy” recent price action is, and dynamically adjusts trade confidence based on that measure.

---

## 2. Theoretical Background  
Shannon entropy for a discrete distribution \(`p_i`\) is defined as:

H = – ∑ᵢ pᵢ · ln(pᵢ)

where pᵢ is the probability of the i-th bin.
	•	High H ⇒ greater randomness (choppy market)
	•	Low H ⇒ more predictability (trending or mean-reverting)

Steps to compute:
	1.	Compute log-returns:

Δrₜ = ln(Pₜ) – ln(Pₜ₋₁)


	2.	Bin the last N returns (e.g. N=20) via Freedman–Diaconis or fixed bins.
	3.	Estimate probabilities pᵢ from the histogram (density=True).
	4.	Compute Shannon entropy in “nats”.

⸻

3. Entropy Calculation in RabbitAI

import numpy as np
from scipy.stats import entropy as shannon_entropy

def calc_entropy(series: pd.Series, bins: str | int = "fd") -> float:
    """
    Shannon entropy (nats) of a 1-D return series.
    """
    if series.empty:
        return 0.0
    p, _ = np.histogram(series, bins=bins, density=True)
    p = p[p > 0]  # drop zero-probability bins
    return float(shannon_entropy(p, base=np.e))

	•	series: last 20 log-returns →

returns_20 = np.log(df["Close"]).diff().dropna().tail(20)


	•	bins: "fd" (Freedman–Diaconis) by default.
	•	output: entropy in nats.

⸻

4. Trading Signal Integration

RabbitAI adjusts its confidence using an entropy bias:

ref_ent      = 2.0     # reference entropy (e.g. median for liquid assets)
alpha        = 0.10    # scaling factor
entropy_bias = (ref_ent - ent_20) / ref_ent * alpha
confidence  += entropy_bias

	•	If ent_20 < ref_ent ⇒ boost confidence
	•	If ent_20 > ref_ent ⇒ reduce confidence

⸻

5. Practical Benefits & Considerations
	•	Regime Detection: Quickly distinguish trending vs. noisy markets.
	•	Adaptive Signals: Avoid trading in high-entropy (choppy) periods.
	•	Performance: O(1) computation per bar.
	•	Transparency: Traders inspect exact formula.

Further reading:
	•	Shannon, C. E. “A Mathematical Theory of Communication,” Bell Syst. Tech. J., 1948.
	•	Zunino, L. et al. “Entropy in Financial Time Series,” Phys. Rev. E, 2008.

⸻

6. Full Workflow Snippet

# 1) Compute 20-bar log-returns entropy
returns_20 = np.log(ohlcv_df["Close"]).diff().dropna().tail(20)
ent_20     = calc_entropy(returns_20)

# 2) Adjust confidence
ref_ent      = 2.0
alpha        = 0.10
entropy_bias = (ref_ent - ent_20) / ref_ent * alpha
confidence  += entropy_bias

print(f"Entropy20: {ent_20:.3f} nats, bias: {entropy_bias:.3f}")


⸻

7. Conclusion

Rolling-window Shannon entropy provides a fast, interpretable gauge of market noise. By layering an entropy-based confidence adjustment, RabbitAI’s signals become more regime-aware—boosting conviction in clear trends and exercising caution in choppy conditions.

⸻

References
	1.	Shannon, C. E. “A Mathematical Theory of Communication,” Bell Syst. Tech. J., 1948.
	2.	Freedman & Diaconis, “On the histogram as a density estimator,” Z. Wahrsch., 1981.
	3.	Zunino, L., Soriano, M. C., & Rosso, O. A. “Entropy Measures in Financial Markets,” Phys. Rev. E, 2008.
	4.	Zu, J., Ahmed, M., & Xing, P. “Market Regime Detection Using Entropy Measures,” J. Fin. Data Sci., 2021.
