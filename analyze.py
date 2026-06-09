"""
Session 4 analysis + publishable chart.

The surprise-residual test: the announcement-day price reaction is the market's
instant read of the earnings numbers. We ask whether call sentiment predicts the
post-reaction DRIFT after controlling for that reaction -- i.e. does tone add
anything beyond the number itself?

Produces:
  - a printed comparison of regressions
  - figures/earnings_drift_by_signal.png  (two-panel event-study chart)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from backtest import add_standardized

ev = pd.read_parquet(ROOT / "data" / "processed" / "events.parquet")
for c in ["sent_full", "sent_qa", "sent_delta", "reaction"]:
    ev = add_standardized(ev, c)


def reg(formula):
    return smf.ols(formula, ev.dropna(subset=["reaction_z", "sent_full_z"])).fit(
        cov_type="cluster", cov_kwds={"groups": ev.dropna(
            subset=["reaction_z", "sent_full_z"])["quarter"]})


def line(label, m, var):
    print(f"  {label:<46}coef={m.params[var]:+.4f}  p={m.pvalues[var]:.3f}")


print(f"\nEvents: {len(ev)} | companies: {ev.ticker.nunique()} | quarters: {ev.quarter.nunique()}")
print("\nDoes the earnings-day reaction predict the next 20 days of drift? (PEAD)")
line("reaction -> CAR_20", reg("CAR_20 ~ reaction_z"), "reaction_z")
print("\nDoes call tone predict drift on its own?")
line("sentiment -> CAR_20", reg("CAR_20 ~ sent_full_z"), "sent_full_z")
print("\nDoes call tone add ANYTHING beyond the reaction?")
both = reg("CAR_20 ~ sent_full_z + reaction_z")
line("sentiment | controlling for reaction", both, "sent_full_z")
line("reaction  | controlling for sentiment", both, "reaction_z")

# ---------- chart ----------
paths = np.vstack(ev["car_path"].values)            # (n_events, 20) in %
days = np.arange(1, paths.shape[1] + 1)


def top_bottom(col):
    q = pd.qcut(ev[col].rank(method="first"), 5, labels=False)
    return paths[(q == 4).values].mean(0) * 100, paths[(q == 0).values].mean(0) * 100


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
C_HI, C_LO = "#1b6ca8", "#d1495b"

for ax, col, title, hi_lbl, lo_lbl in [
    (ax1, "reaction_z", "Grouped by earnings-day price reaction",
     "Biggest positive reaction (top 20%)", "Biggest negative reaction (bottom 20%)"),
    (ax2, "sent_full_z", "Grouped by FinBERT call sentiment",
     "Most positive tone (top 20%)", "Most negative tone (bottom 20%)")]:
    hi, lo = top_bottom(col)
    ax.plot(days, hi, color=C_HI, lw=2.4, label=hi_lbl)
    ax.plot(days, lo, color=C_LO, lw=2.4, label=lo_lbl)
    ax.axhline(0, color="#888", lw=0.8, zorder=0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Trading days after earnings")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
ax1.set_ylabel("Average cumulative abnormal return (%)")

fig.suptitle("Do earnings calls predict post-earnings stock drift?",
             fontsize=15, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
         f"Event study of {len(ev):,} S&P 500 earnings calls, {ev.ticker.nunique()} companies, 2021–2023.  "
         "Left: the price reaction predicts further drift (a real, known effect).  "
         "Right: call tone does not.",
         ha="center", fontsize=9.5, color="#444")
plt.tight_layout()
out = ROOT / "figures" / "earnings_drift_by_signal.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"\nsaved chart -> {out}")


if __name__ == "__main__":
    pass
