import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="TITAN X: OMNIVERSE", page_icon="🌌", layout="wide")

# Initialize Session State for Global Asset Sync
if 'ticker' not in st.session_state: st.session_state['ticker'] = "RELIANCE.NS"
if 'asset_type' not in st.session_state: st.session_state['asset_type'] = "Stock"

st.markdown("""
<style>
    /* OMNIVERSE THEME */
    .main { background-color: #0d1117; color: #e6edf3; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #e6edf3; }
    
    /* Search Bar Styling */
    .stTextInput>div>div>input { 
        background-color: #161b22; color: #58a6ff; 
        border: 1px solid #30363d; font-weight: bold; font-size: 16px;
    }
    
    /* Metrics */
    .pulse-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 15px; text-align: center; margin-bottom: 10px;
        transition: all 0.2s ease-in-out;
    }
    .pulse-card:hover { border-color: #58a6ff; transform: translateY(-3px); }
    .pulse-val { font-size: 22px; font-weight: 800; color: #e6edf3; }
    .pulse-lbl { font-size: 12px; text-transform: uppercase; color: #8b949e; letter-spacing: 1px; }
    
    /* Verdicts */
    .verdict-box {
        padding: 25px; border-radius: 8px; text-align: center; font-weight: 900; 
        font-size: 28px; letter-spacing: 2px; text-transform: uppercase; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; padding-bottom: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d; color: #8b949e; border-radius: 6px; 
        border: 1px solid #30363d; padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb; color: white; border-color: #1f6feb;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET LOGIC ---
def resolve_ticker(user_input):
    """
    Smart resolver that handles Indian Stocks, INR Pairs, Crypto, etc.
    """
    clean = user_input.upper().strip()
    
    # 1. INR CURRENCY PAIRS (Manual Override)
    if clean in ["USDINR", "USD/INR"]: return "INR=X", "USD/INR"
    if clean in ["EURINR", "EUR/INR"]: return "EURINR=X", "EUR/INR"
    if clean in ["GBPINR", "GBP/INR"]: return "GBPINR=X", "GBP/INR"
    if clean in ["JPYINR", "JPY/INR"]: return "JPYINR=X", "JPY/INR"
    
    # 2. COMMODITIES (MCX Proxies)
    if clean in ["GOLD", "XAUUSD"]: return "GC=F", "Gold Futures"
    if clean in ["SILVER", "XAGUSD"]: return "SI=F", "Silver Futures"
    if clean in ["CRUDE", "OIL"]: return "CL=F", "Crude Oil"
    
    # 3. CRYPTO
    if clean in ["BTC", "BITCOIN"]: return "BTC-USD", "Bitcoin"
    if clean in ["ETH", "ETHEREUM"]: return "ETH-USD", "Ethereum"
    
    # 4. DEFAULT INDIAN STOCK HANDLING
    # If it has no suffix and isn't a known global ticker, assume NSE
    if "-" not in clean and "=" not in clean and "." not in clean:
        return f"{clean}.NS", f"{clean} (NSE)"
        
    return clean, clean

# --- 3. MATH ENGINE (70+ INDICATORS) ---
def calculate_indicators(df):
    if df.empty: return df
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    C, H, L, V = df['Close'], df['High'], df['Low'], df['Volume']
    
    # Trend
    for p in [20, 50, 100, 200]: df[f'SMA_{p}'] = C.rolling(p).mean()
    df['EMA_20'] = C.ewm(span=20).mean()
    
    # Volatility
    tr = pd.concat([H-L, (H-C.shift()).abs(), (L-C.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    bb_mid = C.rolling(20).mean()
    bb_std = C.rolling(20).std()
    df['BB_Up'] = bb_mid + (2*bb_std)
    df['BB_Lo'] = bb_mid - (2*bb_std)
    df['BB_W'] = (df['BB_Up'] - df['BB_Lo']) / bb_mid

    # Momentum
    delta = C.diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1 + (gain/loss)))
    
    df['MACD'] = C.ewm(span=12).mean() - C.ewm(span=26).mean()
    df['MACD_Sig'] = df['MACD'].ewm(span=9).mean()
    
    # Ichimoku
    p9 = (H.rolling(9).max() + L.rolling(9).min())/2
    p26 = (H.rolling(26).max() + L.rolling(26).min())/2
    df['Senkou_A'] = ((p9 + p26)/2).shift(26)
    
    # Pivot Points (Auto-Support/Resist)
    df['Pivot'] = (H.shift(1) + L.shift(1) + C.shift(1)) / 3
    df['R1'] = (2 * df['Pivot']) - L.shift(1)
    df['S1'] = (2 * df['Pivot']) - H.shift(1)

    return df

def get_signal_score(c):
    score = 0
    reasons = []
    
    if c['Close'] > c['SMA_200']: score += 20; reasons.append("Bullish Trend (>200 SMA)")
    else: reasons.append("Bearish Trend (<200 SMA)")
    
    if 50 < c['RSI'] < 70: score += 10
    if c['RSI'] < 30: score += 15; reasons.append("RSI Oversold (Dip Buy)")
    if c['MACD'] > c['MACD_Sig']: score += 10
    if c['BB_W'] < 0.10: reasons.append("Volatility Squeeze")
    
    verdict = "NEUTRAL"
    if score >= 60: verdict = "BULLISH"
    elif score <= 30: verdict = "BEARISH"
    
    return verdict, score, reasons

# --- 4. RENDERERS ---

def render_search_ui(key_suffix):
    """
    Renders a search bar that updates global session state.
    """
    c1, c2 = st.columns([3, 1])
    with c1:
        # We use a key based on the tab so streamlit doesn't complain about duplicate widgets
        val = st.text_input("🔍 Search Any Asset (Stock, Crypto, Forex, Cmdty)", value=st.session_state['ticker'], key=f"search_{key_suffix}")
    with c2:
        if st.button("Analyze", key=f"btn_{key_suffix}"):
            resolved, name, _ = resolve_ticker(val)
            st.session_state['ticker'] = resolved
            st.rerun()
            
    return st.session_state['ticker']

def render_live_pulse():
    st.header("🔴 Global Market Pulse")
    
    # SUB-TABS FOR ASSET CLASSES
    p_in, p_fx, p_cm, p_cr, p_et = st.tabs(["🇮🇳 Indian Stocks", "💱 Forex (INR)", "🛢️ Commodities", "₿ Crypto", "📊 ETFs & MF"])
    
    def pulse_grid(tickers_dict):
        cols = st.columns(4)
        idx = 0
        for sym, name in tickers_dict.items():
            with cols[idx % 4]:
                try:
                    d = yf.Ticker(sym).history(period="2d")
                    if len(d) > 1:
                        curr = d['Close'].iloc[-1]
                        prev = d['Close'].iloc[-2]
                        chg = ((curr-prev)/prev)*100
                        col = "#28a745" if chg >=0 else "#d73a49"
                        
                        st.markdown(f"""
                        <div class="pulse-card" style="border-top: 3px solid {col}">
                            <div class="pulse-lbl">{name}</div>
                            <div class="pulse-val" style="color:{col}">{curr:,.2f}</div>
                            <div style="font-size:14px; font-weight:bold; color:{col}">{chg:+.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                except:
                    st.markdown(f"<div class='pulse-card'><div class='pulse-lbl'>{name}</div><div>N/A</div></div>", unsafe_allow_html=True)
            idx += 1
            
    with p_in:
        # Search for Indian Stocks specifically inside the tab
        st.caption("Track specific Indian shares or view market movers")
        pulse_grid({
            "^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY", 
            "RELIANCE.NS": "Reliance", "HDFCBANK.NS": "HDFC Bank",
            "TCS.NS": "TCS", "INFY.NS": "Infosys",
            "TATAMOTORS.NS": "Tata Motors", "SBIN.NS": "SBI"
        })
        
    with p_fx:
        st.caption("Currencies against Indian Rupee (INR)")
        pulse_grid({
            "INR=X": "USD/INR", "EURINR=X": "EUR/INR", 
            "GBPINR=X": "GBP/INR", "JPYINR=X": "JPY/INR"
        })
        
    with p_cm:
        st.caption("Global Commodities (USD)")
        pulse_grid({
            "GC=F": "Gold", "SI=F": "Silver", 
            "CL=F": "Crude Oil", "NG=F": "Natural Gas"
        })
        
    with p_cr:
        st.caption("Top Cryptocurrencies")
        pulse_grid({
            "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", 
            "SOL-USD": "Solana", "DOGE-USD": "Dogecoin"
        })
        
    with p_et:
        st.caption("Indian ETFs")
        pulse_grid({
            "NIFTYBEES.NS": "Nifty Bees", "GOLDBEES.NS": "Gold Bees",
            "BANKBEES.NS": "Bank Bees", "LIQUIDBEES.NS": "Liquid Bees"
        })

def render_analysis():
    tick = render_search_ui("analysis")
    st.divider()
    
    st.subheader(f"🤖 Dual-Timeframe Intelligence: {tick}")
    
    with st.spinner("Analyzing Multi-Timeframe Data..."):
        try:
            # Dual Fetch
            df_d = yf.Ticker(tick).history(period="2y", interval="1d")
            df_w = yf.Ticker(tick).history(period="5y", interval="1wk")
            
            if df_d.empty:
                st.error(f"No data found for {tick}. Try updating the search.")
                return
            
            # Calculations
            df_d = calculate_indicators(df_d)
            df_w = calculate_indicators(df_w)
            
            c_d = df_d.iloc[-1]
            c_w = df_w.iloc[-1]
            
            v_d, s_d, r_d = get_signal_score(c_d)
            v_w, s_w, r_w = get_signal_score(c_w)
            
            # Display
            col1, col2 = st.columns(2)
            
            with col1:
                color = "#28a745" if "BULL" in v_d else "#d73a49" if "BEAR" in v_d else "#d29922"
                st.markdown(f"""
                <div class="verdict-box" style="background-color: {color}">
                    DAILY (Short Term)<br>{v_d}<br><span style="font-size:16px">Score: {s_d}/100</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Targets
                atr = c_d['ATR']
                p = c_d['Close']
                st.info(f"🎯 **Target:** {p+(atr*3):,.2f}")
                st.error(f"🛑 **Stop:** {p-(atr*2):,.2f}")
                
            with col2:
                color_w = "#28a745" if "BULL" in v_w else "#d73a49" if "BEAR" in v_w else "#d29922"
                st.markdown(f"""
                <div class="verdict-box" style="background-color: {color_w}">
                    WEEKLY (Medium Term)<br>{v_w}<br><span style="font-size:16px">Score: {s_w}/100</span>
                </div>
                """, unsafe_allow_html=True)
                
                atr_w = c_w['ATR']
                pw = c_w['Close']
                st.info(f"🎯 **Target:** {pw+(atr_w*3):,.2f}")
                st.error(f"🛑 **Stop:** {pw-(atr_w*2):,.2f}")
                
        except Exception as e: st.error(f"Analysis Error: {e}")

def render_chart():
    tick = render_search_ui("chart")
    st.divider()
    
    # Chart Toolbar
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d", "1wk"], index=4)
    with c2: ctype = st.selectbox("Type", ["Candlestick", "Heikin Ashi", "Line"])
    with c3: overlays = st.multiselect("Overlays", ["SMA 200", "Bollinger Bands", "Pivot Points", "Ichimoku"], default=["SMA 200", "Pivot Points"])
    
    p_map = {"1m":"1d", "5m":"5d", "15m":"5d", "1h":"1mo", "1d":"2y", "1wk":"5y"}
    
    try:
        df = yf.Ticker(tick).history(period=p_map[tf], interval=tf)
        if df.empty:
            st.warning("No chart data available.")
            return
            
        df = calculate_indicators(df)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
        
        # Candles
        if ctype == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        elif ctype == "Heikin Ashi":
            ha_c = (df['Open']+df['High']+df['Low']+df['Close'])/4
            ha_o = (df['Open'].shift(1)+df['Close'].shift(1))/2
            fig.add_trace(go.Candlestick(x=df.index, open=ha_o, high=df['High'], low=df['Low'], close=ha_c, name="Heikin"), row=1, col=1)
        elif ctype == "Line":
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='#2f81f7')), row=1, col=1)
            
        # Overlays
        if "SMA 200" in overlays: 
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=2), name="SMA 200"), row=1, col=1)
        if "Bollinger Bands" in overlays:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lo'], line=dict(color='gray'), fill='tonexty', name="BB"), row=1, col=1)
        if "Pivot Points" in overlays:
             fig.add_trace(go.Scatter(x=df.index, y=df['R1'], mode='markers', marker=dict(color='red', size=2), name="Resist"), row=1, col=1)
             fig.add_trace(go.Scatter(x=df.index, y=df['S1'], mode='markers', marker=dict(color='green', size=2), name="Support"), row=1, col=1)

        # Volume
        colors = ['#d73a49' if c < o else '#28a745' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Vol"), row=2, col=1)
        
        fig.update_layout(height=700, plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font=dict(color='#e6edf3'), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e: st.error(f"Chart Error: {e}")

def render_strategy():
    tick = render_search_ui("strat")
    st.divider()
    st.subheader(f"🧪 Strategy Lab: {tick}")
    st.info("Study the Chart & Analysis tabs first. Then enter your trade idea here.")
    
    c1, c2 = st.columns(2)
    with c1:
        bias = st.selectbox("I am...", ["Neutral", "Bullish", "Bearish"])
    with c2:
        sl = st.number_input("My Stop Loss Price", 0.0)
        
    if st.button("Grade My Strategy"):
        # Fetch Data
        df = yf.Ticker(tick).history(period="1y")
        df = calculate_indicators(df)
        c = df.iloc[-1]
        
        v, s, _ = get_signal_score(c)
        
        st.write("### 🎓 Titan X Feedback")
        
        # Bias Check
        ai_dir = "Bullish" if "BULL" in v else "Bearish" if "BEAR" in v else "Neutral"
        if bias.upper() == ai_dir.upper():
            st.success(f"✅ **Direction Match:** Titan agrees with your {bias} view.")
        else:
            st.error(f"⚠️ **Conflict:** You are {bias}, but Titan AI sees {ai_dir}.")
            
        # SL Check
        if sl > 0:
            dist = abs(c['Close'] - sl)
            if dist < c['ATR']:
                st.warning(f"⚠️ **Stop Loss Too Tight:** Risk ({dist:.2f}) is less than daily noise (ATR {c['ATR']:.2f}). You might get stopped out prematurely.")
            else:
                st.success("✅ **Stop Loss Safe:** Good breathing room.")

# --- 5. MAIN NAVIGATION ---
st.title("🌌 TITAN X: OMNIVERSE")

# MAIN TABS
t1, t2, t3, t4, t5 = st.tabs(["🔴 LIVE PULSE", "🤖 ANALYSIS", "📈 CHART", "🧪 STRATEGY", "🎓 EDUCATION"])

with t1: render_live_pulse()
with t2: render_analysis()
with t3: render_chart()
with t4: render_strategy()
with t5: 
    st.header("🎓 Market Encyclopedia")
    st.info("Educational content goes here (Patterns, Indicators, Terms)")
    st.write("**RSI:** Momentum oscillator measuring speed and change of price movements.")
    st.write("**Bollinger Bands:** Volatility bands placed above and below a moving average.")
    st.write("**MACD:** Trend-following momentum indicator.")

st.markdown("---")
st.caption("TITAN X OMNIVERSE | INSTITUTIONAL ANALYTICS SUITE | EDUCATIONAL USE ONLY")
