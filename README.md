# Tactical Entry Screener

A single-file Streamlit app that screens tickers using tactical “value-entry” signals (RSI, Bollinger Bands, 52-week low proximity, 50-day SMA, MACD histogram) and computes a weighted Entry Score. Reads tickers from `stocks.csv`, refreshes automatically, and highlights high-scoring ideas.

---

## Features

* Login gate (username/password in code)
* Reads tickers from `stocks.csv` (one or two columns)
* Auto-refresh every 5 minute without logging you out
* Indicator calculations:

  * RSI (14)
  * Bollinger Bands (20, 2σ)
  * 50-day SMA
  * MACD (12/26/9) histogram
* Weighted Entry Score and row highlight for Score ≥ 60
* Screener links to company pages on Screener.in
* CSV download of current results
* Scoring reference table built into the UI

---

## Scoring Model

| Criteria                  | Description                                                | Weight |
| ------------------------- | ---------------------------------------------------------- | ------ |
| RSI < 30                  | Indicates oversold condition (lower RSI gets higher score) | 30     |
| Price near/below Lower BB | Proximity to lower Bollinger Band suggests support         | 35     |
| Near 52-Week Low          | Scaled within 10% of 52-week low                           | 20     |
| Price < 50D SMA           | Price below 50-day moving average                          | 10     |
| MACD Histogram > 0        | Bullish crossover indication (binary/graded)               | 5      |

**Entry Score** is the weighted sum. Higher scores indicate better alignment with tactical entry signals.

You can change thresholds/weights at the top of `app.py`:

```python
RSI_THRESHOLD = 30
SMA_PERIOD = 50
BB_PERIOD = 20
BB_STDDEV = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

WEIGHT_RSI = 30
WEIGHT_BB = 35
WEIGHT_52W_LOW = 20
WEIGHT_SMA = 10
WEIGHT_MACD = 5
```

---

## Requirements

* Python 3.12 (recommended)
* Packages pinned in `requirements.txt`:

  ```
  streamlit==1.38.0
  streamlit-autorefresh==0.0.2
  pandas==2.2.2
  numpy==1.26.4
  yfinance==0.2.43
  st-aggrid==0.3.4.post3
  ```

---

## Setup

From the project root:

```bash
# ./reset_env.sh
```

Create a `stocks.csv` next to `app.py`. Either:

Named (two columns):

```csv
name,ticker
Tata Motors,TATAMOTORS.NS
ITC,ITC.NS
Reliance,RELIANCE.NS
HDFC Bank,HDFCBANK.NS
```

---

## Run

```bash
streamlit run app.py
```

Login with:

* Username: `admin`
* Password: `Bct1bnco#`

Change these in `app.py`:

```python
USERNAME = "admin"
PASSWORD = "Bct1bnco#"
```

---

## Auto-Refresh and Caching

* Auto-refresh is handled by `streamlit-autorefresh` every 60 seconds:

  ```python
  st_autorefresh(interval=60_000, key="data-refresh")
  ```
* Historical price data is cached for 5 minutes per ticker:

  ```python
  @st.cache_data(ttl=300)
  def fetch_history(ticker): ...
  ```

  This avoids rate-limits and speeds up the app. If you need fresh pulls on demand, click **Refresh now** (which clears the cache) or reduce the TTL.

To force a new fetch each minute regardless of cache, set `ttl=0` or remove the decorator (expect more API calls and slower runs).

---

## UI Notes

* Rows with `Score ≥ 60` are highlighted (light yellow, bold) for quick scanning.
* Indicator columns get green/red/grey backgrounds to show pass/fail/error.
* A built-in “Scoring Criteria & Weights” expander shows the model rules.
* “Screener” column links to Screener.in for each ticker.

## Security

Credentials are hard-coded for personal use. Do not deploy publicly with hard-coded secrets. If you need persistence across restarts without risking secrets in code, use environment variables or a secrets manager.

---

## Project Layout

Single-file app:

```
app.py
requirements.txt
stocks.csv           # you provide this
README.md
```

---

## License

Personal/internal use. Add a license if you plan to share.
