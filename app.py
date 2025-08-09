import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import yfinance as yf

st.set_page_config(page_title="Tactical Entry Screener", layout="wide")

APP_TITLE = "Tactical Entry Screener"
USERNAME = "admin"
PASSWORD = "Bct1bnco#"

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

if "auth" not in st.session_state:
    st.session_state.auth = False
if "watchlist" not in st.session_state:
    st.session_state.watchlist = pd.DataFrame({
        "name": ["Tata Motors","ITC","Reliance","HDFC Bank"],
        "ticker": ["TATAMOTORS.NS","ITC.NS","RELIANCE.NS","HDFCBANK.NS"]
    })
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "scored_df" not in st.session_state:
    st.session_state.scored_df = pd.DataFrame()

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

def compute_rsi(df, period=14):
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

@st.cache_data(show_spinner=False, ttl=300)
def fetch_history(ticker: str):
    t = yf.Ticker(ticker)
    df = t.history(period="1y")
    return df

def calculate_score(row):
    import numpy as np
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
    base = ticker.split(".")[0]
    return f"https://www.screener.in/company/{base}/"

def process_watchlist(df_in: pd.DataFrame):
    import numpy as np
    rows = []
    latest_ts = []

    for _, r in df_in.iterrows():
        name = str(r["name"]).strip()
        ticker = str(r["ticker"]).strip()
        if not name or not ticker:
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

            row = {
                "Stock": name,
                "Ticker": ticker,
                "Price": round(price, 2),
                "Lower BB": round(bb_lower, 2) if np.isfinite(bb_lower) else np.nan,
                "RSI": round(rsi, 2) if np.isfinite(rsi) else np.nan,
                "52W Low": round(week52_low, 2) if np.isfinite(week52_low) else np.nan,
                "50D SMA": round(sma50, 2) if np.isfinite(sma50) else np.nan,
                "MACD Hist": round(macd_hist, 4) if np.isfinite(macd_hist) else np.nan,
                "Score": None,
                "RSI_OK": rsi_ok,
                "SMA_OK": sma_ok,
                "MACD_OK": macd_ok,
                "BB_OK": bb_ok,
                "Near52wLow_OK": near_52w_low_ok,
                "Screener": make_screener_link(ticker),
            }
            rows.append(row)
        except Exception:
            rows.append({
                "Stock": name, "Ticker": ticker,
                "Price": "Error", "Lower BB": "Error", "RSI": "Error", "52W Low": "Error",
                "50D SMA": "Error", "MACD Hist": "Error", "Score": None,
                "RSI_OK": False, "SMA_OK": False, "MACD_OK": False, "BB_OK": False, "Near52wLow_OK": False,
                "Screener": make_screener_link(ticker),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["Score"] = out.apply(calculate_score, axis=1)
        out.sort_values(by="Score", ascending=False, inplace=True, na_position="last")
        out.reset_index(drop=True, inplace=True)

    latest = max(latest_ts).strftime('%Y-%m-%d %H:%M:%S') if latest_ts else None
    return out, latest

def legend_bar(last_refresh_str: str | None):
    st.markdown(
        """
<style>
.legend { display: flex; gap: 18px; align-items: center; font-size: 14px; color: #444; margin-bottom: 8px; }
.badge { padding: 2px 8px; border-radius: 6px; border: 1px solid #ddd; background: #fafafa; }
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
  const green = {'backgroundColor':'#e8f5e9'}; const red = {'backgroundColor':'#ffebee'}; const grey = {'backgroundColor':'#eeeeee'};
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
    gb.configure_grid_options(domLayout="autoHeight", enableCellTextSelection=True)
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

def manage_watchlist_page():
    st.header("Manage Watchlist")
    st.caption("Upload, view, and edit your stocks list (name, ticker). Save to persist in session and download as CSV.")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        file = st.file_uploader("Upload stocks.csv", type=["csv"])
    with col2:
        if st.button("Download current CSV"):
            csv_buf = io.StringIO()
            st.session_state.watchlist.to_csv(csv_buf, index=False)
            st.download_button("Click to download", data=csv_buf.getvalue(), file_name="stocks.csv", mime="text/csv", key="dlbtn")
    with col3:
        if st.button("Add blank row"):
            st.session_state.watchlist.loc[len(st.session_state.watchlist)] = {"name":"","ticker":""}

    st.write("Edit below and click **Save changes**")
    edited = st.data_editor(
        st.session_state.watchlist,
        num_rows="dynamic",
        use_container_width=True,
        key="editor"
    )

    save = st.button("Save changes")
    if save:
        edited = edited.fillna("")
        edited["name"] = edited["name"].astype(str).str.strip()
        edited["ticker"] = edited["ticker"].astype(str).str.strip()
        edited = edited[(edited["name"]!="") & (edited["ticker"]!="")]
        st.session_state.watchlist = edited.reset_index(drop=True)
        st.success("Watchlist updated.")

    st.markdown(
        """
**CSV format**
```
name,ticker
Tata Motors,TATAMOTORS.NS
ITC,ITC.NS
Reliance,RELIANCE.NS
HDFC Bank,HDFCBANK.NS
```
"""
    )

def screener_page():
    st.header("Screener")
    st.autorefresh(interval=300000, key="auto-refresh")  # 5 min

    run = st.button("Refresh now")
    if run:
        fetch_history.clear()

    with st.spinner("Scoring… this uses cached data (5 min TTL)"):
        df, latest = process_watchlist(st.session_state.watchlist)

    st.session_state.scored_df = df
    st.session_state.last_refresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    legend_bar(st.session_state.last_refresh if st.session_state.last_refresh else latest)
    aggrid_table(df)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download results CSV", data=csv_buf.getvalue(), file_name="screener_results.csv", mime="text/csv")

def main():
    if not st.session_state.auth:
        login_view()
        return

    if st.sidebar.button("Log out"):
        st.session_state.auth = False
        st.rerun()

    st.sidebar.title(APP_TITLE)
    page = st.sidebar.radio("Navigate", ["Screener","Manage Watchlist"])

    if page == "Screener":
        screener_page()
    elif page == "Manage Watchlist":
        manage_watchlist_page()

if __name__ == "__main__":
    main()