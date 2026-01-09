import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="TITAN X: PULSE PRO", page_icon="🔴", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #c9d1d9; }
    h1, h2, h3, h4 { color: #c9d1d9; }
    .stTextInput>div>div>input { background-color: #21262d; color: white; border: 1px solid #30363d; }
    .stSelectbox>div>div>div { background-color: #21262d; color: white; }
    
    /* Live Pulse Dashboard Cards */
    .pulse-card {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 15px; text-align: center; margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .pulse-card:hover { transform: scale(1.02); border-color: #58a6ff; }
    .pulse-sym { font-size: 14px; color: #8b949e; font-weight: bold; }
    .pulse-price { font-size: 22px; font-weight: bold; color: #f0f6fc; }
    .pulse-vol { font-size: 11px; color: #8b949e; margin-top: 5px; }
    
    /* Verdicts */
    .verdict-box {
        padding: 20px; border-radius: 5px; text-align: center; font-weight: 900; 
        font-size: 24px; margin-bottom: 20px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #0d1117; }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d; color: #c9d1d9; border-radius: 4px 4px 0 0; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #238636; color: white; }
    
    /* News */
    .news-card {
        background-color: #161b22; padding: 10px; margin-bottom: 8px; border-radius: 4px; border-left: 3px solid #238636;
    }
    .news-link { color: #58a6ff; text-decoration: none; font-weight: bold; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET RESOLVER ---
ASSET_MAP = {
    "GOLD": {"sym": "GC=F", "c": "USD"}, "SILVER": {"sym": "SI=F", "c": "USD"},
    "CRUDE": {"sym": "CL=F", "c": "USD"}, "BITCOIN": {"sym": "BTC-USD", "c": "USD"},
    "ETHEREUM": {"sym": "ETH-USD", "c": "USD"}, "NIFTY": {"sym": "^NSEI", "c": "INR"},
    "BANKNIFTY": {"sym": "^NSEBANK", "c": "INR"}, "S&P500": {"sym": "^GSPC", "c": "USD"},
    "NASDAQ": {"sym": "^IXIC", "c": "USD"}, "EURUSD": {"sym": "EURUSD=X", "c": "USD"}
}

def resolve_ticker(inp):
    clean = inp.upper().strip()
    if clean in ASSET_MAP: return ASSET_MAP[clean]['sym'], clean, ASSET_MAP[clean]['c']
    curr = "INR"
    sym = f"{clean}.NS"
    if "-" in clean or "=" in clean: sym, curr = clean, "USD"
    elif len(clean) <= 5 and clean.isalpha():
        t = yf.Ticker(clean)
        if not t.history(period="5d").empty: return clean, clean, "USD"
    return sym, clean, curr

# --- 3. INDICATOR ENGINE ---
def calculate_indicators(df):
    df.columns = [c.capitalize() for c in df.columns]
    C, H, L, V = df['Close'], df['High'], df['Low'], df['Volume']
    
    # Heikin Ashi
    df['HA_Close'] = (df['Open'] + H + L + C) / 4
    df['HA_Open'] = (df['Open'].shift(1) + C.shift(1)) / 2
    df['HA_Open'].iloc[0] = df['Open'].iloc[0]
    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    # MAs
    for p in [9, 20, 50, 100, 200]: df[f'SMA_{p}'] = C.rolling(p).mean()
    for p in [12, 26]: df[f'EMA_{p}'] = C.ewm(span=p).mean()

    # BB
    bb_std = C.rolling(20).std()
    df['BB_Up'] = df['SMA_20'] + (2*bb_std)
    df['BB_Lo'] = df['SMA_20'] - (2*bb_std)
    df['BB_W'] = (df['BB_Up'] - df['BB_Lo']) / df['SMA_20']

    # RSI & MACD
    delta = C.diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1 + (gain/loss)))
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Sig'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Sig']

    # Stochastic
    l14, h14 = L.rolling(14).min(), H.rolling(14).max()
    df['Stoch_K'] = 100 * ((C - l14) / (h14 - l14))
    
    # ATR & Ichimoku
    tr = pd.concat([H-L, (H-C.shift()).abs(), (L-C.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    p9 = (H.rolling(9).max() + L.rolling(9).min())/2
    p26 = (H.rolling(26).max() + L.rolling(26).min())/2
    df['Senkou_A'] = ((p9 + p26)/2).shift(26)
    
    # VWAP
    df['VWAP'] = (C * V).cumsum() / V.cumsum()

    # Rare Tools
    r1 = C.diff(10).rolling(10).mean()
    r2 = C.diff(15).rolling(10).mean()
    r3 = C.diff(20).rolling(10).mean()
    r4 = C.diff(30).rolling(15).mean()
    df['KST'] = (r1*1) + (r2*2) + (r3*3) + (r4*4)
    df['Coppock'] = (((C - C.shift(14))/C.shift(14)) + ((C - C.shift(11))/C.shift(11))).ewm(span=10).mean()
    
    return df

# --- 4. SIGNAL LOGIC ---
def generate_signal(c):
    score = 0
    reasons = []
    
    if c['Close'] > c['SMA_200']: score += 15; reasons.append("✅ Price > 200 SMA (Bull Trend)")
    else: reasons.append("🔻 Price < 200 SMA (Bear Trend)")
    
    if 50 < c['RSI'] < 70: score += 10
    if c['RSI'] < 30: score += 10; reasons.append("✅ RSI Oversold (Bounce Likely)")
    if c['MACD'] > c['MACD_Sig']: score += 10
    if c['Close'] > c['Senkou_A']: score += 10
    if c['BB_W'] < 0.10: reasons.append("⚡ Volatility Squeeze Detect")
    
    return score, reasons

# --- 5. DASHBOARD HELPERS ---
def fetch_dashboard_data(tickers):
    data_list = []
    for symbol, name in tickers.items():
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                curr = hist.iloc[-1]['Close']
                prev = hist.iloc[-2]['Close']
                vol = hist.iloc[-1]['Volume']
                change = ((curr - prev) / prev) * 100
                data_list.append({
                    "Name": name, "Price": curr, "Change": change, "Vol": vol
                })
        except: pass
    return data_list

def render_dashboard_grid(data):
    cols = st.columns(4)
    for i, item in enumerate(data):
        with cols[i % 4]:
            color = "#28a745" if item['Change'] >= 0 else "#dc3545"
            arrow = "▲" if item['Change'] >= 0 else "▼"
            st.markdown(f"""
            <div class="pulse-card" style="border-top: 3px solid {color};">
                <div class="pulse-sym">{item['Name']}</div>
                <div class="pulse-price" style="color:{color}">{item['Price']:,.2f}</div>
                <div style="color:{color}; font-weight:bold; font-size:14px;">{arrow} {item['Change']:+.2f}%</div>
                <div class="pulse-vol">Vol: {item['Vol']/1000:.1f}K</div>
            </div>
            """, unsafe_allow_html=True)

# --- 6. APP UI ---
st.title("🔴 TITAN X: MARKET PULSE PRO")

with st.sidebar:
    st.header("Search Asset")
    asset_in = st.text_input("Symbol", "GOLD")
    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly"])
    
    st.divider()
    st.header("Strategy Lab")
    user_bias = st.radio("Your View:", ["Neutral", "Bullish", "Bearish"])
    run = st.button("🚀 CONNECT & ANALYZE")

if run:
    tick, name, curr_sym = resolve_ticker(asset_in)
    per = "5y"
    inter = "1d" if timeframe == "Daily" else "1wk"
    
    with st.spinner(f"Scanning Global Markets & {name}..."):
        # FETCH MAIN ASSET
        try:
            ticker_obj = yf.Ticker(tick)
            df = ticker_obj.history(period=per, interval=inter)
            
            if df.empty: st.error("No Data Found.")
            else:
                df = calculate_indicators(df)
                c = df.iloc[-1]
                
                # --- MASTER TABS ---
                t_pulse, t_chart, t_data, t_edu, t_verdict = st.tabs(["🔴 Live Pulse", "📊 TradingView", "🔢 Omniscient Data", "🎓 Academy", "🤖 Verdict"])

                # === TAB 1: LIVE PULSE DASHBOARD ===
                with t_pulse:
                    st.markdown("### 🌍 Global Market Heartbeat")
                    
                    # SUB TABS FOR DASHBOARD
                    dp_indices, dp_stocks, dp_crypto, dp_comm, dp_forex = st.tabs(["Indices", "Stocks", "Crypto", "Commodities", "Forex"])
                    
                    with dp_indices:
                        st.caption("Live Movement: Major Indices")
                        indices = {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", "^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^BSESN": "SENSEX", "^DJI": "DOW JONES", "^GDAXI": "DAX", "^FTSE": "FTSE 100"}
                        render_dashboard_grid(fetch_dashboard_data(indices))
                        
                    with dp_stocks:
                        st.caption("Live Movement: Top Actives (India/US)")
                        stocks = {"RELIANCE.NS": "Reliance", "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia", "MSFT": "Microsoft"}
                        render_dashboard_grid(fetch_dashboard_data(stocks))
                        
                    with dp_crypto:
                        st.caption("Live Movement: Crypto Assets")
                        cryptos = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "XRP", "DOGE-USD": "Dogecoin", "ADA-USD": "Cardano", "BNB-USD": "Binance Coin", "MATIC-USD": "Polygon"}
                        render_dashboard_grid(fetch_dashboard_data(cryptos))
                        
                    with dp_comm:
                        st.caption("Live Movement: Commodities")
                        comms = {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil", "NG=F": "Natural Gas", "HG=F": "Copper", "PL=F": "Platinum"}
                        render_dashboard_grid(fetch_dashboard_data(comms))
                        
                    with dp_forex:
                        st.caption("Live Movement: Currencies")
                        forex = {"INR=X": "USD/INR", "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "JPY=X": "USD/JPY", "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD"}
                        render_dashboard_grid(fetch_dashboard_data(forex))
                    
                    st.divider()
                    
                    # SELECTED ASSET NEWS
                    st.markdown(f"#### 📰 Live News: {name}")
                    try:
                        news = ticker_obj.news
                        if news:
                            for n in news[:4]:
                                st.markdown(f"""
                                <div class="news-card">
                                    <a href="{n['link']}" target="_blank" class="news-link">{n['title']}</a>
                                    <div style="font-size:11px; color:#8b949e;">{datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d %H:%M')}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else: st.info("No recent news found.")
                    except: st.info("News feed currently unavailable.")

                # === TAB 2: CHART ===
                with t_chart:
                    c_type, c_overlay, c_sub = st.columns([1, 2, 2])
                    with c_type: chart_mode = st.selectbox("Type", ["Candlestick", "Heikin Ashi", "Line", "Area"])
                    with c_overlay: overlays = st.multiselect("Overlays", ["SMA 50", "SMA 200", "Bollinger Bands", "Ichimoku", "VWAP"], default=["SMA 200"])
                    with c_sub: subplots = st.multiselect("Sub-Charts", ["Volume", "RSI", "MACD", "Stochastic"], default=["Volume", "RSI"])

                    rows = 1 + len(subplots)
                    row_heights = [0.6] + [0.4/len(subplots)] * len(subplots) if subplots else [1.0]
                    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

                    if chart_mode == "Candlestick": fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                    elif chart_mode == "Heikin Ashi": fig.add_trace(go.Candlestick(x=df.index, open=df['HA_Open'], high=df['HA_High'], low=df['HA_Low'], close=df['HA_Close'], name="Heikin"), row=1, col=1)
                    elif chart_mode == "Line": fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close"), row=1, col=1)
                    elif chart_mode == "Area": fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy', name="Close"), row=1, col=1)

                    if "SMA 200" in overlays: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=2), name="SMA 200"), row=1, col=1)
                    if "Bollinger Bands" in overlays:
                         fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray'), showlegend=False), row=1, col=1)
                         fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lo'], line=dict(color='gray'), fill='tonexty', name="BB"), row=1, col=1)
                    if "Ichimoku" in overlays:
                        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'], line=dict(color='green'), showlegend=False), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'], line=dict(color='red'), fill='tonexty', name="Cloud"), row=1, col=1)

                    curr_row = 2
                    for sub in subplots:
                        if sub == "Volume": fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Vol"), row=curr_row, col=1)
                        elif sub == "RSI": 
                            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=curr_row, col=1)
                            fig.add_hline(y=70, row=curr_row, col=1, line_dash="dot"); fig.add_hline(y=30, row=curr_row, col=1, line_dash="dot")
                        elif sub == "MACD":
                            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Hist"), row=curr_row, col=1)
                            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name="MACD"), row=curr_row, col=1)
                            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Sig'], line=dict(color='orange'), name="Sig"), row=curr_row, col=1)
                        curr_row += 1

                    fig.update_layout(height=700, plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font=dict(color='#c9d1d9'), xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                # === TAB 3: DATA ===
                with t_data:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f"<div class='metric-card'><span class='metric-label'>SMA 200</span><br><span class='metric-value'>{c['SMA_200']:.2f}</span></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='metric-card'><span class='metric-label'>RSI</span><br><span class='metric-value'>{c['RSI']:.2f}</span></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='metric-card'><span class='metric-label'>KST</span><br><span class='metric-value'>{c['KST']:.2f}</span></div>", unsafe_allow_html=True)
                    c4.markdown(f"<div class='metric-card'><span class='metric-label'>Coppock</span><br><span class='metric-value'>{c['Coppock']:.2f}</span></div>", unsafe_allow_html=True)

                # === TAB 4: EDUCATION ===
                with t_edu:
                    st.markdown("### 🎓 Market Academy")
                    e1, e2 = st.tabs(["🕯️ Patterns", "📈 Terms"])
                    with e1:
                        st.info("**Doji:** Represents indecision. Open and Close are virtually equal.")
                        st.info("**Hammer:** Bullish reversal pattern. Small body, long lower wick.")
                        st.info("**Engulfing:** One candle completely 'eats' the previous one.")
                    with e2:
                        st.info("**ATR:** Average True Range. Measures volatility (noise).")
                        st.info("**RSI:** Relative Strength Index. >70 Overbought, <30 Oversold.")

                # === TAB 5: VERDICT ===
                with t_verdict:
                    if user_bias == "Neutral":
                        st.warning("🔒 Verdict Locked. Select Bullish/Bearish in Sidebar.")
                    else:
                        score, reasons = generate_signal(c)
                        if score >= 60: txt, col = "BUY ✅", "#28a745"
                        elif score <= 20: txt, col = "SELL 🔻", "#dc3545"
                        else: txt, col = "HOLD 😐", "#d39e00"
                        st.markdown(f"<div class='verdict-box' style='background-color:{col}'>{txt}</div>", unsafe_allow_html=True)
                        for r in reasons: st.info(r)

        except Exception as e: st.error(f"Error: {e}")

st.markdown('<div style="text-align:center; margin-top:50px; color:#666">TITAN X: EDUCATION ONLY</div>', unsafe_allow_html=True)
