"""
One command runs the whole study end-to-end and regenerates results + chart:

    python run_pipeline.py

Steps:
  1. fetch_data       -> data/raw/      (transcripts + prices; cached)
  2. score_sentiment  -> data/processed/sentiment.parquet  (FinBERT)
  3. align_returns    -> data/processed/events.parquet
  4. analyze          -> printed results + figures/earnings_drift_by_signal.png

Cached files are reused, so a re-run after a change to one stage is fast.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def run(name, path):
    print(f"\n{'='*60}\nSTEP: {name}\n{'='*60}")
    if subprocess.run([sys.executable, str(path)]).returncode != 0:
        sys.exit(f"step '{name}' failed; fix and re-run")


if __name__ == "__main__":
    run("1. fetch data", SRC / "fetch_data.py")
    run("2. score sentiment", SRC / "score_sentiment.py")
    run("3. align returns", SRC / "align_returns.py")
    run("4. analyze + chart", HERE / "analyze.py")
