import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="TITAN X: Ultimate Analyst", page_icon="⚡", layout="wide")

# Custom CSS
st.markdown("""
<style>
.disclaimer {
    color: red;
    font-weight: bold;
    font-size: 14px;
    border: 2px solid red;
    padding: 10px;
    text-align: center;
    margin-top: 20px;
    background-color: #ffe6e6;
}
.metric-box {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ddd;
    text-align: center;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ TITAN X: Multi-Asset Techno-Fundamental Engine")
st.markdown("Universal Analyzer for: **Stocks, Crypto, Forex, Commodities & Indices.**")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Command Center")
    user_ticker = st.text_input("Asset Symbol", value="RELIANCE", help="Type any name: 'BTC', 'GOLD', 'NIFTY', 'EURUSD'")
    user_email = st.text_input("Email Report To (Optional)", placeholder="you@gmail.com")
    run_btn = st.button("🚀 Run Full Scan", type="primary")

# --- 3. SMART ASSET RESOLVER ---
def get_valid_ticker(symbol):
    symbol = symbol.upper().strip()
    # Priority List: 1. Direct | 2. Indian NSE | 3. Crypto USD | 4. Yahoo Futures | 5. Indices
    formats = [symbol, f"{symbol}.NS", f"{symbol}-USD", f"{symbol}=X", f"{symbol}=F", f"^{symbol}"]
    
    for fmt in formats:
        try:
            t = yf.Ticker(fmt)
            hist = t.history(period="5d")
            if not hist.empty:
                return fmt, t.info.get('longName', fmt)
        except:
            continue
    return None, None

# --- 4. ADVANCED MATH ENGINE (Fixed for Pandas 2.0+) ---
def calculate_indicators(df):
    # Ensure simple columns
    df.columns = [c.capitalize() for c in df.columns]
    
    # 1. TREND ------------------------------------------------
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # MACD
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ADX (Trend Strength)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr_list = [df['High'] - df['Low'], 
               (df['High'] - df['Close'].shift()).abs(), 
               (df['Low'] - df['Close'].shift()).abs()]
    tr = pd.concat(tr_list, axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    
    df['+DI'] = 100 * (df['+DM'].rolling(14).mean() / atr_14)
    df['-DI'] = 100 * (df['-DM'].rolling(14).mean() / atr_14)
    dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = dx.rolling(14).mean()

    # Ichimoku Cloud
    high_9 = df['High'].rolling(9).max()
    low_9 = df['Low'].rolling(9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2 

    high_26 = df['High'].rolling(26).max()
    low_26 = df['Low'].rolling(26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2 

    # 2. MOMENTUM ---------------------------------------------
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic
    lowest_14 = df['Low'].rolling(14).min()
    highest_14 = df['High'].
