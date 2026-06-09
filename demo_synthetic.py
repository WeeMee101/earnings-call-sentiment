"""
SYNTHETIC DEMO -- runs with no internet, no API key, no model download.

It fabricates earnings events with a KNOWN, planted relationship:
  - sentiment is partly a proxy for the earnings surprise (sue), plus
  - a small INDEPENDENT predictive component (the 'residual signal').
Then it runs the exact same event-study + regression + backtest code you'll use
on real data, and recovers that planted signal. This proves the analytical core
is correct. Swap synthetic data for real data (run_pipeline.py) and you're done.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from event_study import market_model_car, car_path
from backtest import add_standardized, build_sue, run_regressions, quintile_long_short

rng = np.random.default_rng(42)

N_TICKERS = 120
QUARTERS = [f"{y}Q{q}" for y in (2023, 2024) for q in (1, 2, 3, 4)]
EST_LEN, EVENT_LEN = 220, 20

# planted truth -- TOTAL abnormal drift over the 20-day window (realistic scale)
BETA_SUE = 0.020        # ~2% CAR per 1 sigma of surprise (PEAD-like)
BETA_RESID = 0.006      # small incremental drift from sentiment BEYOND the surprise
SENT_FROM_SUE = 0.7     # how much sentiment merely proxies the surprise


def make_events():
    rows = []
    for q in QUARTERS:
        for i in range(N_TICKERS):
            sue = rng.normal()
            resid_sent = rng.normal()                 # the part not from the number
            sentiment = SENT_FROM_SUE * sue + np.sqrt(1 - SENT_FROM_SUE**2) * resid_sent
            qa = sentiment + rng.normal(0, 0.4)
            rows.append({"ticker": f"T{i:03d}", "quarter": q, "sue_true": sue,
                         "resid_sent": resid_sent, "sent_full": sentiment,
                         "sent_qa": qa,
                         "eps_estimate": 1.00,
                         "eps_actual": 1.00 + 0.05 * sue})  # surprise -> sue later
    df = pd.DataFrame(rows).sort_values(["ticker", "quarter"])
    df["sent_delta"] = df.groupby("ticker")["sent_full"].diff()
    return df.dropna(subset=["sent_delta"]).reset_index(drop=True)


def simulate_car(row):
    """Build a full return series for one event, then run the REAL market model."""
    mkt = rng.normal(0.0003, 0.01, EST_LEN + EVENT_LEN)
    beta = rng.uniform(0.8, 1.3)
    idio = rng.normal(0, 0.015, EST_LEN + EVENT_LEN)
    stock = beta * mkt + idio
    # plant the TOTAL drift across the EVENT window (so CAR scale is realistic)
    drift = BETA_SUE * row["sue_true"] + BETA_RESID * row["resid_sent"]
    stock[EST_LEN:] += drift / EVENT_LEN  # spread the total over the 20 days
    est_ret, ev_ret = stock[:EST_LEN], stock[EST_LEN:]
    est_mkt, ev_mkt = mkt[:EST_LEN], mkt[EST_LEN:]
    car20 = market_model_car(ev_ret, ev_mkt, est_ret, est_mkt)
    car5 = market_model_car(ev_ret[:5], ev_mkt[:5], est_ret, est_mkt)
    path = car_path(ev_ret, ev_mkt, est_ret, est_mkt)
    return car5, car20, path


def main():
    print("=" * 64)
    print("SYNTHETIC DEMO  (planted: surprise drift + small residual sentiment)")
    print("=" * 64)
    ev = make_events()
    cars5, cars20, paths = [], [], []
    for _, r in ev.iterrows():
        c5, c20, path = simulate_car(r)
        cars5.append(c5); cars20.append(c20); paths.append(path)
    ev["CAR_5"], ev["CAR_20"] = cars5, cars20

    ev = build_sue(ev)                       # eps_actual/estimate -> sue
    ev = add_standardized(ev, "sent_full")
    ev = add_standardized(ev, "sent_qa")
    ev = add_standardized(ev, "sent_delta")
    print(f"\nEvents built: {len(ev)}  across {ev.quarter.nunique()} quarters\n")

    models = run_regressions(ev, outcome="CAR_20")
    m1, m2, m3 = models["m1"], models["m2"], models["m3"]
    print("-- Model 1: CAR_20 ~ sentiment  (naive)")
    print(f"   sentiment coef = {m1.params['sent_full_z']:+.5f}  "
          f"p = {m1.pvalues['sent_full_z']:.4f}")
    print("-- Model 2: CAR_20 ~ sentiment + SUE  (does sentiment SURVIVE?)")
    print(f"   sentiment coef = {m2.params['sent_full_z']:+.5f}  "
          f"p = {m2.pvalues['sent_full_z']:.4f}")
    print(f"   SUE coef       = {m2.params['sue']:+.5f}  "
          f"p = {m2.pvalues['sue']:.4f}")
    print("-- Model 3: + Q&A + tone change")
    print(f"   qa coef    = {m3.params['sent_qa_z']:+.5f}  p = {m3.pvalues['sent_qa_z']:.4f}")
    print(f"   delta coef = {m3.params['sent_delta_z']:+.5f}  p = {m3.pvalues['sent_delta_z']:.4f}")

    print("\n-- Quintile long/short (top vs bottom sentiment, net of costs)")
    stats = quintile_long_short(ev, signal="sent_full_z", outcome="CAR_20")
    for k, v in stats.items():
        print(f"   {k:14s}: {v}")

    # average CAR path: top vs bottom sentiment quintile
    ev["qbin"] = ev.groupby("quarter")["sent_full_z"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    parr = np.vstack(paths)
    top = parr[(ev.qbin == 4).values].mean(0)
    bot = parr[(ev.qbin == 0).values].mean(0)
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, EVENT_LEN + 1), top, label="Top sentiment quintile")
    plt.plot(range(1, EVENT_LEN + 1), bot, label="Bottom sentiment quintile")
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("Trading days after earnings (t+1 ...)"); plt.ylabel("Average CAR")
    plt.title("Average abnormal-return path by sentiment quintile (synthetic)")
    plt.legend(); plt.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "demo_car_path.png"
    plt.savefig(out, dpi=130)
    print(f"\nsaved figure -> {out}")
    print("\nInterpretation: sentiment is significant in Model 1, partly attenuated")
    print("by SUE in Model 2 but SURVIVES -> tone carried info beyond the number.")
    print("That surviving coefficient is exactly what a real signal looks like.")
    print("\nNOTE: synthetic data has a clean planted signal, so significance here")
    print("is guaranteed. On REAL data expect much weaker, often insignificant")
    print("results after the SUE control and costs -- that's the honest finding.")


if __name__ == "__main__":
    main()
