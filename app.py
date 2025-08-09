import io
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# -------------------- App config --------------------
st.set_page_config(page_title="Tactical Entry Screener", layout="wide")

APP_TITLE = "Tactical Entry Screener"
USERNAME = "admin"
PASSWORD = "Bct1bnco#"

# Indicator thresholds/periods
RSI_THRESHOLD = 30
SMA_PERIOD = 50
BB_PERIOD = 20
BB_STDDEV = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Weights
WEIGHT_RSI = 30
WEIGHT_BB = 35
WEIGHT_52W_LOW = 20
WEIGHT_SMA = 10
WEIGHT_MACD = 5

# -------------------- Session init --------------------
if "auth" not in st.session_state:
    st.session_state.auth = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "scored_df" not in st.session_state:
    st.session_state.scored_df = pd.DataFrame()

# -------------------- Auth --------------------
def login_view():
    st.title(APP_TITLE)
    st.subheader("Login")
    with st.form("login-form", clear_on_submit=False):
        u = st.text_input("Username", value="", key="username")
        p = st.text_input("Password", type="password", value="", key="password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if u == USERNAME and p == PASSWORD:
                st.session_state.auth = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

def logout_button():
    if st.sidebar.button("Log out"):
        st.session_state.auth = False
        st.rerun()

# -------------------- Indicators --------------------
def compute_rsi(df, period=14):
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    close = df["Close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_bollinger_bands(df, period=BB_PERIOD, num_std=BB_STDDEV):
    close = df["Close"]
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band, lower_band, sma

# -------------------- Data + Scoring --------------------
@st.cache_data(show_spinner=False, ttl=300)  # cache 5 min
def fetch_history(ticker: str):
    t = yf.Ticker(ticker)
    return t.history(period="1y")

def calculate_score(row):
    score = 0
    if isinstance(row["RSI"], float) and np.isfinite(row["RSI"]) and row["RSI"] < RSI_THRESHOLD:
        score += WEIGHT_RSI * ((RSI_THRESHOLD - row["RSI"]) / RSI_THRESHOLD)
    if isinstance(row["Price"], float) and isinstance(row["50D SMA"], float) and np.isfinite(row["50D SMA"]) and row["Price"] < row["50D SMA"]:
        pct_below = (row["50D SMA"] - row["Price"]) / row["50D SMA"]
        score += min(WEIGHT_SMA, WEIGHT_SMA * pct_below)
    if isinstance(row["MACD Hist"], float) and np.isfinite(row["MACD Hist"]):
        proximity = max(min((row["MACD Hist"] + 0.05) / 0.05, 1), 0)
        score += round(WEIGHT_MACD * proximity, 2)
    if isinstance(row["Price"], float) and isinstance(row["Lower BB"], float) and np.isfinite(row["Lower BB"]):
        proximity = row["Lower BB"] / row["Price"]
        score += round(min(WEIGHT_BB, WEIGHT_BB * proximity), 2)
    if isinstance(row["Price"], float) and isinstance(row["52W Low"], float) and np.isfinite(row["52W Low"]):
        proximity = row["52W Low"] / row["Price"]
        score += round(min(WEIGHT_52W_LOW, WEIGHT_52W_LOW * proximity), 2)
    return round(score, 2)

def make_screener_link(ticker: str):
    return f"https://www.screener.in/company/{ticker.split('.')[0]}/"

def load_watchlist_from_csv(path: str = "stocks.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"`{path}` not found in the app folder. Add it and reload.")
        return pd.DataFrame(columns=["name", "ticker"])
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "ticker" not in df.columns:
            st.error("CSV must contain at least a `ticker` column.")
            return pd.DataFrame(columns=["name", "ticker"])
        if "name" not in df.columns:
            df["name"] = df["ticker"]
        df = df[["name", "ticker"]].fillna("").astype(str)
        df = df[(df["name"] != "") & (df["ticker"] != "")]
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Failed to read `{path}`: {e}")
        return pd.DataFrame(columns=["name", "ticker"])

def process_watchlist(df_in: pd.DataFrame):
    rows, latest_ts = [], []
    for _, r in df_in.iterrows():
        name = str(r["name"]).strip()
        ticker = str(r["ticker"]).strip()
        if not ticker:
            continue
        try:
            df = fetch_history(ticker)
            if df.empty:
                raise RuntimeError("No data")
            close = df["Close"]
            price = float(close.iloc[-1])

            upper_bb, lower_bb, sma50_series = compute_bollinger_bands(df)
            sma50 = float(sma50_series.iloc[-1]) if not np.isnan(sma50_series.iloc[-1]) else np.nan
            rsi_series = compute_rsi(df)
            rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else np.nan
            _, _, hist = compute_macd(df)
            macd_hist = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else np.nan
            bb_lower = float(lower_bb.iloc[-1]) if not np.isnan(lower_bb.iloc[-1]) else np.nan

            try:
                pe = yf.Ticker(ticker).info.get("trailingPE", None)
                pe_val = round(float(pe), 2) if isinstance(pe, (int, float)) else "N/A"
            except Exception:
                pe_val = "N/A"

            week52_low = float(close.min())
            latest_ts.append(df.index[-1])

            rsi_ok = rsi < RSI_THRESHOLD if np.isfinite(rsi) else False
            sma_ok = price < sma50 if np.isfinite(sma50) else False
            macd_ok = macd_hist > 0 if np.isfinite(macd_hist) else False
            bb_ok = price <= bb_lower * 1.02 if np.isfinite(bb_lower) else False
            near_52w_low_ok = price <= week52_low * 1.1 if np.isfinite(week52_low) else False

            rows.append({
                "Stock": name or ticker,
                "Ticker": ticker,
                "Price": round(price, 2),
                "Lower BB": round(bb_lower, 2) if np.isfinite(bb_lower) else np.nan,
                "RSI": round(rsi, 2) if np.isfinite(rsi) else np.nan,
                "52W Low": round(week52_low, 2) if np.isfinite(week52_low) else np.nan,
                "50D SMA": round(sma50, 2) if np.isfinite(sma50) else np.nan,
                "MACD Hist": round(macd_hist, 4) if np.isfinite(macd_hist) else np.nan,
                "Score": None,
                "RSI_OK": rsi_ok, "SMA_OK": sma_ok, "MACD_OK": macd_ok,
                "BB_OK": bb_ok, "Near52wLow_OK": near_52w_low_ok,
                "Screener": make_screener_link(ticker),
                "Current PE": pe_val,
            })
        except Exception:
            rows.append({
                "Stock": name or ticker, "Ticker": ticker,
                "Price": "Error", "Lower BB": "Error", "RSI": "Error", "52W Low": "Error",
                "50D SMA": "Error", "MACD Hist": "Error", "Score": None,
                "RSI_OK": False, "SMA_OK": False, "MACD_OK": False, "BB_OK": False, "Near52wLow_OK": False,
                "Screener": make_screener_link(ticker), "Current PE": "Error",
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["Score"] = out.apply(calculate_score, axis=1)
        out.sort_values(by="Score", ascending=False, inplace=True, na_position="last")
        out.reset_index(drop=True, inplace=True)

    latest = max(latest_ts).strftime('%Y-%m-%d %H:%M:%S') if latest_ts else None
    return out, latest

# -------------------- UI helpers --------------------
def scoring_reference():
    with st.expander("Scoring Criteria & Weights", expanded=False):
        st.markdown(
            """
| **Criteria** | **Description** | **Weightage** |
|---|---|---|
| RSI < 30 | Indicates oversold condition (lower RSI gets higher score) | 30 |
| Price near/below Lower BB | Proximity to lower Bollinger Band suggests support levels | 35 |
| Near 52-Week Low | Stock is close to its 52-week low (scaled within 10%) | 20 |
| Price < 50D SMA | Price trading below 50-day moving average | 10 |
| MACD Histogram > 0 | Bullish crossover indication (binary) | 5 |

**Entry Score:** Weighted sum based on how strongly a stock meets each condition. Higher scores indicate better alignment with value-entry signals.
            """
        )

def legend_bar(last_refresh_str: str | None):
    st.markdown(
        """
<style>
.legend { display: flex; gap: 18px; align-items: center; font-size: 14px; color: #444; margin-bottom: 8px; }
.badge { padding: 2px 8px; border-radius: 6px; border: 1px solid #bbb; background: #f3f4f6; font-weight: 600; }
</style>
""",
        unsafe_allow_html=True,
    )
    last_txt = last_refresh_str if last_refresh_str else "—"
    html = f"""
<div class="legend">
  <div><b>Legend:</b></div>
  <div class="badge">Green = OK</div>
  <div class="badge">Red = Fails criterion</div>
  <div class="badge">Grey = Error/No data</div>
  <div style="margin-left:auto;">Last refresh: <b>{last_txt}</b> (auto every 5 min)</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)

def aggrid_table(df: pd.DataFrame):
    view_cols = ["Stock","Price","Lower BB","RSI","52W Low","50D SMA","MACD Hist","Score","Screener"]
    gb = GridOptionsBuilder.from_dataframe(df[view_cols])

    gb.configure_column("Screener", header_name="Screener", cellRenderer=JsCode("""
        class UrlCellRenderer {
          init(params) { this.eGui = document.createElement('a'); this.eGui.href = params.value; this.eGui.innerText = 'Open'; this.eGui.target = '_blank'; }
          getGui() { return this.eGui; }
        }
    """))

    style_fn = JsCode("""
function(params){
  const row = params.data || {}; const val = params.value;
  const isErr = (v) => String(v).toLowerCase() === 'error';
  const green = {'backgroundColor':'#86efac', 'fontWeight':'600'};
  const red   = {'backgroundColor':'#fca5a5', 'fontWeight':'600'};
  const grey  = {'backgroundColor':'#e5e7eb', 'fontWeight':'600'};
  const col = params.colDef.field;
  if (['RSI','50D SMA','MACD Hist','Lower BB','52W Low'].includes(col)){
    if (isErr(val)) return grey;
    if (col==='RSI') return row['RSI_OK'] ? green : red;
    if (col==='50D SMA') return row['SMA_OK'] ? green : red;
    if (col==='MACD Hist') return row['MACD_OK'] ? green : red;
    if (col==='Lower BB') return row['BB_OK'] ? green : red;
    if (col==='52W Low') return row['Near52wLow_OK'] ? green : red;
  }
  return null;
}
""")
    for c in ["RSI","50D SMA","MACD Hist","Lower BB","52W Low"]:
        gb.configure_column(c, cellStyle=style_fn)

    # Row demarcation for high-score ideas (>= 60)
    row_style_fn = JsCode("""
function(params) {
  const s = Number(params.data && params.data.Score);
  if (!isNaN(s) && s >= 60) {
    return {'backgroundColor': '#fff3cd', 'fontWeight': '700'}; // light yellow + bold
  }
  return {};
}
""")

    gb.configure_grid_options(domLayout="autoHeight", enableCellTextSelection=True, getRowStyle=row_style_fn)
    grid_options = gb.build()

    AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme="alpine",
        height=460
    )

# -------------------- Page --------------------
def screener_page():
    st.header("Screener")
    scoring_reference()

    # Auto-refresh every 1 minute; session is preserved (no logout)
    st_autorefresh(interval=300*1000, key="data-refresh")

    watchlist = load_watchlist_from_csv("stocks.csv")
    if watchlist.empty:
        st.stop()

    if st.button("Refresh now"):
        fetch_history.clear()

    start_time = time.time()
    with st.spinner("Scoring… this uses cached data (5 min TTL)"):
        df, latest = process_watchlist(watchlist)
    elapsed = time.time() - start_time
    st.info(f"Scoring completed in {elapsed:.2f} seconds.")

    st.session_state.scored_df = df
    st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    legend_bar(st.session_state.last_refresh or latest)
    aggrid_table(df)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download results CSV", data=csv_buf.getvalue(), file_name="screener_results.csv", mime="text/csv")

# -------------------- Main --------------------
def main():
    if not st.session_state.auth:
        login_view()
        return

    st.sidebar.title(APP_TITLE)
    logout_button()
    screener_page()

if __name__ == "__main__":
    main()
