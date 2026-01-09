import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. GLOBAL SETTINGS & THEME ---
st.set_page_config(
    page_title="TITAN X: GENESIS TERMINAL",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force Dark Mode & Bloomberg Terminal Styling
st.markdown("""
<style>
    /* CORE THEME */
    .stApp { background-color: #000000; font-family: 'Roboto Mono', monospace; }
    h1, h2, h3, h4, h5, h6 { color: #e0e0e0 !important; font-family: 'Inter', sans-serif; }
    p, span, div { color: #b0b0b0; }
    
    /* REMOVE PADDING */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 98% !important; }
    
    /* SEARCH BARS */
    .stTextInput>div>div>input { 
        background-color: #111111; color: #00ff00; border: 1px solid #333; 
        font-weight: bold; letter-spacing: 1px;
    }
    
    /* METRIC CARDS */
    .metric-container {
        background: #0a0a0a; border: 1px solid #222; padding: 15px; 
        border-radius: 4px; text-align: center; margin-bottom: 10px;
    }
    .metric-val { font-size: 20px; font-weight: 700; color: #fff; }
    .metric-lbl { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] { background-color: #000; border-bottom: 1px solid #222; }
    .stTabs [data-baseweb="tab"] {
        background-color: #000; color: #666; border: none; font-size: 12px; font-weight: bold; text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] { color: #00ff00 !important; border-bottom: 2px solid #00ff00; }
    
    /* TABLE STYLING */
    [data-testid="stDataFrame"] { border: 1px solid #222; }
    
    /* VERDICT BANNER */
    .verdict-banner {
        padding: 15px; font-size: 24px; font-weight: 900; text-align: center; 
        text-transform: uppercase; letter-spacing: 4px; border: 1px solid;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE MANAGEMENT ---
if 'active_asset' not in st.session_state:
    st.session_state['active_asset'] = "RELIANCE.NS"

def update_asset(new_val):
    """Updates the global asset from any search bar"""
    clean = new_val.upper().strip()
    
    # SMART RESOLVER LOGIC
    # 1. Indian Stocks (Default)
    if not any(x in clean for x in ["-", "=", "^", "."]):
        clean = f"{clean}.NS"
        
    # 2. Commodities / Forex Proxies
    overrides = {
        "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE": "CL=F", "NIFTY": "^NSEI", 
        "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "USDINR": "INR=X",
        "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "ETH": "ETH-USD"
    }
    if clean in overrides: clean = overrides[clean]
    
    st.session_state['active_asset'] = clean

# --- 3. HEAVY DUTY CALCULATION ENGINE ---
def compute_indicators(df):
    if df.empty: return df
    df = df.copy()
    
    # 1. CANDLE MATH
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    
    # 2. TREND
    for p in [9, 20, 50, 200]: df[f'SMA_{p}'] = df['Close'].rolling(p).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    
    # 3. VOLATILITY
    df['STD_20'] = df['Close'].rolling(20).std()
    df['BB_Up'] = df['SMA_20'] + (2 * df['STD_20'])
    df['BB_Lo'] = df['SMA_20'] - (2 * df['STD_20'])
    df['BB_W'] = (df['BB_Up'] - df['BB_Lo']) / df['SMA_20']
    
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # 4. MOMENTUM
    delta = df['Close'].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1 + (gain/loss)))
    
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # 5. ICHIMOKU
    p9 = (df['High'].rolling(9).max() + df['Low'].rolling(9).min())/2
    p26 = (df['High'].rolling(26).max() + df['Low'].rolling(26).min())/2
    df['Tenkan'] = p9
    df['Kijun'] = p26
    df['SpanA'] = ((p9 + p26)/2).shift(26)
    
    return df

# --- 4. RENDERERS ---

def render_chart_ui():
    # SYNCED SEARCH
    col_search, col_tf, col_type = st.columns([2, 1, 1])
    with col_search:
        q = st.text_input("CMD > SEARCH ASSET", value=st.session_state['active_asset'], key="chart_search")
        if q != st.session_state['active_asset']: update_asset(q); st.rerun()
    with col_tf:
        tf = st.selectbox("TIMEFRAME", ["1m", "5m", "15m", "1h", "1d", "1wk"], index=4)
    with col_type:
        ctype = st.selectbox("STYLE", ["Candle", "Heikin Ashi", "Line"])

    # DATA MAP
    p_map = {"1m":"1d", "5m":"5d", "15m":"5d", "1h":"1mo", "1d":"2y", "1wk":"5y"}
    
    try:
        t = yf.Ticker(st.session_state['active_asset'])
        df = t.history(period=p_map[tf], interval=tf)
        
        if df.empty:
            st.error("NO DATA FEED. MARKET MAY BE CLOSED OR ASSET INVALID.")
            return
            
        df = compute_indicators(df)
        
        # PLOTLY CHARTING (TRADINGVIEW STYLE)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.75, 0.25])
        
        # MAIN PANEL
        if ctype == "Candle":
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Px"), row=1, col=1)
        elif ctype == "Heikin Ashi":
            fig.add_trace(go.Candlestick(x=df.index, open=df['HA_Open'], high=df['HA_High'], low=df['HA_Low'], close=df['HA_Close'], name="HA"), row=1, col=1)
        
        # OVERLAYS
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#2962ff', width=2), name="SMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='#666', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lo'], line=dict(color='#666', width=1), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="BB"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(color='rgba(0,255,0,0.3)'), name="Cloud"), row=1, col=1)
        
        # SUB PANEL (MACD & VOL)
        colors = ['#00ff00' if c >= o else '#ff0000' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.3, name="Vol"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00ffff'), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='#ff9900'), name="Sig"), row=2, col=1)
        
        # LAYOUT CONFIG
        fig.update_layout(
            height=700, plot_bgcolor='#000', paper_bgcolor='#000',
            font=dict(color='#e0e0e0'), xaxis_rangeslider_visible=False,
            dragmode='pan',
            grid=dict(rows=2, columns=1, pattern='independent'),
            xaxis=dict(showgrid=True, gridcolor='#222'),
            yaxis=dict(showgrid=True, gridcolor='#222', side='right')
        )
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
        
    except Exception as e: st.error(f"CHART ERROR: {e}")

def render_pulse_ui():
    st.markdown("### 📡 MARKET SCANNER")
    
    # 1. WATCHLIST GENERATOR
    assets = {
        "INDICES": ["^NSEI", "^NSEBANK", "^BSESN", "^GSPC", "^IXIC"],
        "STOCKS (IN)": ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "SBIN.NS"],
        "COMMODITIES": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"],
        "FOREX": ["INR=X", "EURINR=X", "GBPUSD=X", "EURUSD=X", "JPY=X"],
        "CRYPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    }
    
    # TABBED VIEW
    tabs = st.tabs(list(assets.keys()))
    
    for tab, category in zip(tabs, assets.keys()):
        with tab:
            # Create Dataframe for Pulse
            data = []
            tickers = assets[category]
            for t in tickers:
                try:
                    obj = yf.Ticker(t)
                    hist = obj.history(period="2d")
                    if len(hist) > 1:
                        curr = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        chg = ((curr-prev)/prev)*100
                        vol = hist['Volume'].iloc[-1]
                        data.append([t, curr, chg, vol])
                except: pass
            
            # Render Table
            if data:
                df_pulse = pd.DataFrame(data, columns=["ASSET", "PRICE", "CHANGE %", "VOLUME"])
                
                # Styling
                def color_chg(val):
                    color = '#00ff00' if val >= 0 else '#ff0000'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(
                    df_pulse.style.applymap(color_chg, subset=['CHANGE %'])
                    .format({"PRICE": "{:,.2f}", "CHANGE %": "{:+.2f}%", "VOLUME": "{:,.0f}"}),
                    use_container_width=True, height=300
                )
            else:
                st.warning("DATA FEED OFFLINE FOR THIS SECTOR")

def render_analysis_ui():
    # SYNCED SEARCH
    col_s, _ = st.columns([3, 1])
    with col_s:
        q = st.text_input("CMD > TARGET ASSET", value=st.session_state['active_asset'], key="ana_search")
        if q != st.session_state['active_asset']: update_asset(q); st.rerun()
        
    tick = st.session_state['active_asset']
    
    try:
        t = yf.Ticker(tick)
        df = t.history(period="1y")
        df = compute_indicators(df)
        c = df.iloc[-1]
        
        # SCORING
        score = 0
        reasons = []
        if c['Close'] > c['SMA_200']: score += 20; reasons.append("Price > 200 SMA (Bullish Trend)")
        if c['RSI'] > 50: score += 10
        if c['MACD'] > c['Signal']: score += 10; reasons.append("MACD Bullish Cross")
        if c['Close'] > c['SpanA']: score += 10
        if c['BB_W'] < 0.10: reasons.append("VOLATILITY SQUEEZE DETECTED")
        
        # DISPLAY
        col1, col2 = st.columns([1, 2])
        
        with col1:
            verdict = "BUY" if score >= 50 else "SELL"
            v_col = "#00ff00" if verdict == "BUY" else "#ff0000"
            st.markdown(f"<div class='verdict-banner' style='color:{v_col}; border-color:{v_col}'>{verdict}<br><span style='font-size:14px'>CONFIDENCE: {score}%</span></div>", unsafe_allow_html=True)
            
            st.markdown("#### 🎯 LEVELS")
            st.info(f"R1: {c['Close'] + (c['ATR']*2):,.2f}")
            st.error(f"S1: {c['Close'] - (c['ATR']*1.5):,.2f}")
            
        with col2:
            st.markdown("#### ⚡ TECHNICAL DRIVERS")
            for r in reasons:
                st.write(f"• {r}")
                
            st.markdown("#### 📊 METRIC MATRIX")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"<div class='metric-container'><div class='metric-lbl'>RSI</div><div class='metric-val'>{c['RSI']:.1f}</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-container'><div class='metric-lbl'>ATR</div><div class='metric-val'>{c['ATR']:.2f}</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-container'><div class='metric-lbl'>SMA 200</div><div class='metric-val'>{c['SMA_200']:.2f}</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-container'><div class='metric-lbl'>BB WIDTH</div><div class='metric-val'>{c['BB_W']:.2f}</div></div>", unsafe_allow_html=True)

    except Exception as e: st.error(e)

def render_strategy_ui():
    st.subheader("🧪 STRATEGY LAB")
    
    # SYNCED SEARCH
    q = st.text_input("TEST ASSET", value=st.session_state['active_asset'], key="strat_search")
    if q != st.session_state['active_asset']: update_asset(q); st.rerun()
    
    c1, c2, c3 = st.columns(3)
    with c1: bias = st.selectbox("YOUR BIAS", ["LONG", "SHORT"])
    with c2: entry = st.number_input("ENTRY PRICE", value=0.0)
    with c3: sl = st.number_input("STOP LOSS", value=0.0)
    
    if st.button("RUN SIMULATION"):
        t = yf.Ticker(st.session_state['active_asset'])
        df = t.history(period="6mo")
        df = compute_indicators(df)
        c = df.iloc[-1]
        
        # VALIDATION
        st.markdown("### 🛡️ RISK REPORT")
        risk = abs(entry - sl)
        if risk < c['ATR']:
            st.error(f"CRITICAL: Stop Loss is too tight ({risk:.2f}). Daily noise (ATR) is {c['ATR']:.2f}. High probability of stop hunt.")
        else:
            st.success("✅ Risk Management: Pass. Stop Loss handles daily volatility.")
            
        # TREND ALIGNMENT
        trend = "LONG" if c['Close'] > c['SMA_200'] else "SHORT"
        if bias == trend:
            st.success(f"✅ Trend Alignment: Pass. You are trading with the 200 SMA.")
        else:
            st.warning(f"⚠️ Trend Conflict: You are fading the trend. Major trend is {trend}.")

def render_education():
    st.markdown("### 🎓 TITAN ACADEMY: MASTER TRADING")
    
    with st.expander("📚 CHAPTER 1: CANDLESTICK PATTERNS (The Language of Price)"):
        st.markdown("""
        * **DOJI:** The 'Cross'. Open = Close. Means Indecision. Market is taking a breath.
        * **HAMMER:** Small body, long tail. Buyers rejected lower prices. Bullish reversal.
        * **SHOOTING STAR:** Small body, long upper wick. Sellers rejected higher prices. Bearish reversal.
        * **ENGULFING:** The second candle eats the first. Shows massive power shift.
        """)
        
    with st.expander("📚 CHAPTER 2: INDICATORS (The Dashboard)"):
        st.markdown("""
        * **RSI (Relative Strength Index):** Speedometer. >70 is Speeding (Overbought), <30 is Stalled (Oversold).
        * **MACD:** The GPS. Tells you the direction and strength of the trend.
        * **BOLLINGER BANDS:** The Road. Price usually stays on the road. If it goes off-road (outside bands), it snaps back.
        * **ATR (Average True Range):** The Suspension. Measures how bumpy the road is. Use this to set Stop Losses.
        """)
    
    with st.expander("📚 CHAPTER 3: FUNDAMENTALS (The Engine)"):
        st.markdown("""
        * **P/E Ratio:** How expensive the stock is compared to its profit. High P/E = Growth Expectation.
        * **EPS (Earnings Per Share):** Pure profit divided by number of shares. Higher is better.
        * **Dividend Yield:** Interest paid to you for holding the stock.
        """)

# --- 5. MAIN NAVIGATION CONTROLLER ---
t_pulse, t_chart, t_ana, t_strat, t_edu = st.tabs([
    "🔴 LIVE PULSE", "📈 CHART", "🤖 ANALYSIS", "🧪 STRATEGY", "🎓 EDUCATION"
])

with t_pulse: render_pulse_ui()
with t_chart: render_chart_ui()
with t_ana: render_analysis_ui()
with t_strat: render_strategy_ui()
with t_edu: render_education()

st.markdown("---")
st.markdown("<div style='text-align:center; color:#333;'>TITAN X GENESIS | POWERED BY PYTHON & YFINANCE</div>", unsafe_allow_html=True)
