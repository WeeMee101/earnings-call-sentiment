# RUNBOOK — every micro-step to run this

You have two ways to run it. **Start with Path A (Colab).** It needs zero
installation, gives you a free GPU, and works the same on Mac or Windows. Use
Path B later when you want the polished local repo to push to GitHub.

---

## What software is involved (the whole list)

- **Python** — the language that runs the `.py` files. (Colab already has it.)
- **pip** — installs the libraries below. (Comes with Python.)
- **Libraries** (installed by one command): `transformers` + `torch` (the FinBERT
  model), `yfinance` (prices), `pandas`/`numpy`/`statsmodels`/`scipy` (the math),
  `matplotlib` (charts), `requests` (API calls).
- **An FMP API key** — free, from financialmodelingprep.com. This is the only
  account you need to create.
- (Path B only) **VS Code** — a free editor to view/edit the files. Optional.

That's it. No database, no server, nothing to host.

---

## Path A — Google Colab (recommended, ~10 minutes, no install)

### Step A1 — See the demo run first (proves the code works)
1. Go to **colab.research.google.com** → sign in with Google → **New notebook**.
2. Upload the project: click the **folder icon** on the left sidebar → the
   **upload icon** → select the `earnings-sentiment` folder's files, OR upload
   the whole `.zip` and unzip it in a cell:
   ```python
   !unzip earnings-sentiment.zip
   ```
3. In a cell, install the math libraries and run the synthetic demo:
   ```python
   !pip install statsmodels pyarrow -q
   !cd earnings-sentiment && python demo_synthetic.py
   ```
4. You'll see the regression output and a saved figure. **This is the same code
   that runs on real data** — you've now confirmed the engine works.

### Step A2 — Run the real pipeline (no API key needed)
1. Turn on the GPU (optional but faster): **Runtime → Change runtime type →
   T4 GPU → Save**.
2. Install everything:
   ```python
   !pip install -r earnings-sentiment/requirements.txt -q
   ```
3. Run it:
   ```python
   !cd earnings-sentiment && python run_pipeline.py
   ```
4. First run downloads the transcript dataset (one-time, a few minutes) and
   FinBERT (~440 MB, one time). Then it scores the calls, aligns returns, and
   prints the regression + backtest.

### Step A3 — Start tiny, then scale
Open `src/fetch_data.py`, find `TICKERS = [...]` and `YEARS = [...]`. Keep ~5
names for the first run so the whole chain finishes in a couple of minutes. Then
add more tickers / years and re-run. Transcripts come from a free Hugging Face
dataset and prices from yfinance — no key, no rate limits.

---

## Path B — Local on your laptop (for the GitHub repo)

### Step B1 — Install Python
- **Mac:** Python 3 is likely already there. Check in **Terminal** (Cmd+Space →
  "Terminal"): type `python3 --version`. If missing, install from python.org or
  `brew install python`.
- **Windows:** download from **python.org/downloads** → run installer → **CHECK
  the box "Add Python to PATH"** → Install. Open **PowerShell** (Start menu →
  "PowerShell") and type `python --version`.

### Step B2 — Get the project onto your machine
Unzip the `earnings-sentiment.zip` somewhere you'll find it (e.g. Documents).

### Step B3 — Open a terminal IN the project folder
- **Mac:** in Terminal: `cd ~/Documents/earnings-sentiment`
- **Windows:** in PowerShell: `cd C:\Users\YOURNAME\Documents\earnings-sentiment`
(Or open the folder in **VS Code** and use its built-in terminal: Terminal → New Terminal.)

### Step B4 — Create an isolated environment and install
```bash
# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
(If Windows blocks the activate script: run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry.)

### Step B5 — Run the demo, then the real thing
```bash
python demo_synthetic.py            # confirm the engine works

# Mac/Linux: set key and run
export FMP_KEY=PASTE_YOUR_KEY_HERE
python run_pipeline.py
```
```powershell
# Windows
$env:FMP_KEY="PASTE_YOUR_KEY_HERE"
python run_pipeline.py
```

---

## The order things happen (so you know where you are)

```
fetch_data.py    -> data/raw/transcripts.parquet, surprises.parquet, prices.parquet
score_sentiment.py -> data/processed/sentiment.parquet      (FinBERT scores)
align_returns.py -> data/processed/events.parquet           (CARs + signal + surprise)
backtest / run_pipeline.py -> regression tables + figures/  (the answer)
```

You are "building it" the moment Step A1 / B5 prints output. There's no separate
build/compile — Python runs the files directly.

---

## If something breaks
- `ModuleNotFoundError` → you skipped the `pip install`, or your venv isn't active.
- FMP returns empty `[]` → that ticker/quarter has no transcript on the free tier,
  or you hit the rate limit. Reduce tickers, add `time.sleep`, or upgrade.
- FinBERT labels look inverted → print `model.config.id2label` and check the order.
- Everything looks too good (huge Sharpe) → you're probably capturing the t=0 gap.
  Re-read the timing logic in `align_returns.py`.
