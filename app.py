import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- 1. TERMINAL CONFIGURATION ---
st.set_page_config(page_title="TITAN X: HEDGE FUND", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    /* TITAN X: PRO THEME */
    .stApp { background-color: #0e1117; }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 { color: #e6e6e6; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    p, div, span { color: #b0b3b8; }
    
    /* Inputs */
    .stTextInput>div>div>input { background-color: #1c2128; color: white; border: 1px solid #30363d; border-radius: 4px; }
    .stSelectbox>div>div>div { background-color: #1c2128; color: white; border: none; }
    
    /* Metrics & Cards */
    .pulse-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d; border-radius: 6px; padding: 12px;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .pulse-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
    
    /* Verdict Tags */
    .verdict-tag {
        font-size: 14px; font-weight: bold; padding: 4px 8px; border-radius: 4px; display: inline-block;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22; color: #8b949e; border-radius: 4px; 
        border: 1px solid #30363d; padding: 8px 20px; font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636; color: white; border-color: #238636;
    }
    
    /* Disclaimer */
    .disclaimer { 
        font-size: 11px; color: #484f58; text-align: center; margin-top: 50px; 
        border-top: 1px solid #30363d; padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ASSET RESOLVER ---
ASSET_MAP = {
    "GOLD": {"sym": "GC=F", "c": "USD"}, "SILVER": {"sym": "SI=F", "c": "USD"},
    "CRUDE": {"sym": "CL=F", "c": "USD"}, "BITCOIN": {"sym": "BTC-USD", "c": "USD"},
    "ETHEREUM": {"sym": "ETH-USD", "c": "USD"}, "SOLANA": {"sym": "SOL-USD", "c": "USD"},
    "NIFTY": {"sym": "^NSEI", "c": "INR"}, "BANKNIFTY": {"sym": "^NSEBANK", "c": "INR"},
    "SENSEX": {"sym": "^BSESN", "c": "INR"}, "S&P500": {"sym": "^GSPC", "c": "USD"},
    "NASDAQ": {"sym": "^IXIC", "c": "USD"}, "DOW": {"sym": "^DJI", "c": "USD"},
    "EURUSD": {"sym": "EURUSD=X", "c": "USD"}, "GBPUSD": {"sym": "GBPUSD=X", "c": "USD"},
    "RELIANCE": {"sym": "RELIANCE.NS", "c": "INR"}, "HDFCBANK": {"sym": "HDFCBANK.NS", "c": "INR"},
    "TCS": {"sym": "TCS.NS", "c": "INR"}, "INFOSYS": {"sym": "INFY.NS", "c": "INR"}
}

def resolve_ticker(inp):
    clean = inp.upper().strip()
    if clean in ASSET_MAP: return ASSET_MAP[clean]['sym'], clean, ASSET_MAP[clean]['c']
    curr = "INR"
    sym = f"{clean}.NS"
    if "-" in clean or "=" in clean: sym, curr = clean, "USD"
    elif len(clean) <= 5 and clean.isalpha():
        t = yf.Ticker(clean)
        try:
            if not t.history(period="5d").empty: return clean, clean, "USD"
        except: pass
    return sym, clean, curr

# --- 3. QUANT ENGINE ---
def calculate_metrics(df):
    if df.empty: return df
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    
    # Basic MAs
    for p in [20, 50, 100, 200]: df[f'SMA_{p}'] = df['Close'].rolling(p).mean()
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1 + (gain/loss)))
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Sig'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Sig']
    
    # Bollinger Bands
    std = df['Close'].rolling(20).std()
    df['BB_Up'] = df['SMA_20'] + (2*std)
    df['BB_Lo'] = df['SMA_20'] - (2*std)
    df['BB_W'] = (df['BB_Up'] - df['BB_Lo']) / df['SMA_20']
    
    # ATR (Volatility)
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # Ichimoku
    p9 = (df['High'].rolling(9).max() + df['Low'].rolling(9).min())/2
    p26 = (df['High'].rolling(26).max() + df['Low'].rolling(26).min())/2
    df['Tenkan'] = p9
    df['Kijun'] = p26
    df['Senkou_A'] = ((p9 + p26)/2).shift(26)
    
    # Pivot High/Low (For Charting)
    df['Pivot_High'] = df['High'].rolling(10, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(10, center=True).min()
    
    return df

def get_signal_score(c):
    score = 0
    reasons = []
    
    # Trend
    if c['Close'] > c['SMA_200']: score += 20; reasons.append("Price > 200 SMA (Bull Trend)")
    else: reasons.append("Price < 200 SMA (Bear Trend)")
    
    if c['EMA_9'] > c['EMA_20']: score += 10
    
    # Momentum
    if 50 < c['RSI'] < 70: score += 10
    if c['MACD'] > c['MACD_Sig']: score += 10
    if c['Close'] > c['Senkou_A']: score += 10
    
    # Reversal
    if c['RSI'] < 30: score += 15; reasons.append("RSI Oversold (Potential Bounce)")
    if c['BB_W'] < 0.10: reasons.append("Volatility Squeeze (Breakout Imminent)")
    
    verdict = "NEUTRAL"
    if score >= 60: verdict = "BULLISH"
    elif score <= 30: verdict = "BEARISH"
    
    return verdict, score, reasons

# --- 4. RENDERERS ---

def render_chart_tab(tick, name):
    # PROFESSIONAL TOOLBAR
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1: 
        tf = st.selectbox("Timeframe", ["1 Minute", "5 Minutes", "15 Minutes", "1 Hour", "4 Hours", "Daily", "Weekly"], index=5)
    with c2: 
        ctype = st.selectbox("Type", ["Candlestick", "Heikin Ashi", "Line", "Area"])
    with c3:
        scale = st.selectbox("Scale", ["Linear", "Logarithmic"])
    with c4:
        overlays = st.multiselect("Overlays", ["Smart Money Levels", "SMA 200", "Bollinger Bands", "Ichimoku", "EMA 20"], default=["SMA 200", "Smart Money Levels"])

    # MAP TIMEFRAME
    tf_map = {"1 Minute":"1m", "5 Minutes":"5m", "15 Minutes":"15m", "1 Hour":"1h", "4 Hours":"1h", "Daily":"1d", "Weekly":"1wk"}
    p_map = {"1 Minute":"1d", "5 Minutes":"5d", "15 Minutes":"5d", "1 Hour":"1mo", "4 Hours":"3mo", "Daily":"2y", "Weekly":"5y"}
    
    with st.spinner("Initializing Institutional Chart Engine..."):
        try:
            df = yf.Ticker(tick).history(period=p_map[tf], interval=tf_map[tf])
            if df.empty:
                st.error("No Data.")
                return
            
            df = calculate_metrics(df)
            
            # --- PLOTLY ENGINE ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
            
            # 1. PRICE ACTION
            if ctype == "Candlestick":
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            elif ctype == "Line":
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Close", line=dict(color='#2962ff')), row=1, col=1)
            elif ctype == "Area":
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], fill='tozeroy', name="Close", line=dict(color='#2962ff')), row=1, col=1)
            elif ctype == "Heikin Ashi":
                 ha_c = (df['Open']+df['High']+df['Low']+df['Close'])/4
                 ha_o = (df['Open'].shift(1)+df['Close'].shift(1))/2
                 fig.add_trace(go.Candlestick(x=df.index, open=ha_o, high=df['High'], low=df['Low'], close=ha_c, name="Heikin"), row=1, col=1)

            # 2. OVERLAYS
            if "SMA 200" in overlays: 
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#2962ff', width=2), name="SMA 200"), row=1, col=1)
            if "EMA 20" in overlays:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#f23645', width=1), name="EMA 20"), row=1, col=1)
            if "Bollinger Bands" in overlays:
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lo'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name="BB"), row=1, col=1)
            if "Ichimoku" in overlays:
                fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'], line=dict(color='#00ff00', width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'], line=dict(color='#ff0000', width=1), fill='tonexty', name="Cloud"), row=1, col=1)
            
            # 3. SMART MONEY OVERLAYS (Auto-S/R)
            if "Smart Money Levels" in overlays:
                # Detect Pivot Highs/Lows as S/R
                last_high = df['Pivot_High'].last_valid_index()
                last_low = df['Pivot_Low'].last_valid_index()
                if last_high:
                    val = df.loc[last_high]['High']
                    fig.add_hline(y=val, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Resistance")
                if last_low:
                    val = df.loc[last_low]['Low']
                    fig.add_hline(y=val, line_dash="dash", line_color="green", row=1, col=1, annotation_text="Support")

            # 4. VOLUME
            colors = ['#f23645' if c < o else '#2962ff' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Vol"), row=2, col=1)

            # 5. PRO LAYOUT
            fig.update_layout(
                height=750,
                plot_bgcolor='#131722', paper_bgcolor='#131722',
                font=dict(color='#b2b5be', family="Roboto"),
                grid=dict(rows=2, columns=1, pattern='independent'),
                showlegend=True,
                xaxis_rangeslider_visible=False,
                yaxis=dict(type="log" if scale == "Logarithmic" else "linear", showgrid=True, gridcolor='#2a2e39'),
                xaxis=dict(showgrid=True, gridcolor='#2a2e39'),
                yaxis2=dict(showgrid=False),
                # TRADINGVIEW MODE BAR
                dragmode='pan',
                modebar=dict(
                    bgcolor='#1c2128',
                    color='#b2b5be',
                    activecolor='#2962ff',
                    add=['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape', 'zoomIn2d', 'zoomOut2d', 'resetScale2d']
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
            st.caption("🖱️ **Pro Tip:** Hover over the top-right of the chart to access **Drawing Tools** (Trendlines, Rectangles, etc.). Scroll to Zoom.")
            
        except Exception as e: st.error(f"Chart Render Error: {e}")

def render_analysis_tab(tick, name):
    st.subheader(f"🧠 Titan X Intelligence: {name}")
    
    with st.spinner("Running Dual-Core Analysis Engine..."):
        try:
            # AUTO-RUN BOTH TIMEFRAMES (User has no choice)
            df_d = yf.Ticker(tick).history(period="2y", interval="1d")
            df_w = yf.Ticker(tick).history(period="5y", interval="1wk")
            
            if len(df_d) < 50: 
                st.error("Not enough historical data for deep analysis.")
                return

            df_d = calculate_metrics(df_d)
            df_w = calculate_metrics(df_w)
            
            c_d = df_d.iloc[-1]
            c_w = df_w.iloc[-1]
            
            v_d, s_d, r_d = get_signal_score(c_d)
            v_w, s_w, r_w = get_signal_score(c_w)
            
            # --- DISPLAY ENGINE ---
            col1, col2 = st.columns(2)
            
            # DAILY CARD
            with col1:
                color = "#28a745" if "BULL" in v_d else "#dc3545" if "BEAR" in v_d else "#d39e00"
                st.markdown(f"""
                <div class="pulse-card" style="border-top: 5px solid {color}">
                    <h3>SHORT-TERM (Daily)</h3>
                    <h1 style="color:{color}">{v_d}</h1>
                    <p>Confidence: {s_d}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Auto-Targets
                atr = c_d['ATR']
                price = c_d['Close']
                t1 = price + (atr * 2)
                t2 = price + (atr * 4)
                sl = price - (atr * 1.5)
                
                st.markdown("#### 🎯 Execution Levels")
                st.info(f"**Target 1:** {t1:,.2f} (+2 ATR)")
                st.success(f"**Target 2:** {t2:,.2f} (+4 ATR)")
                st.error(f"**Stop Loss:** {sl:,.2f} (-1.5 ATR)")
                
                with st.expander("Technical Drivers (Daily)"):
                    for r in r_d: st.write(f"• {r}")
            
            # WEEKLY CARD
            with col2:
                color_w = "#28a745" if "BULL" in v_w else "#dc3545" if "BEAR" in v_w else "#d39e00"
                st.markdown(f"""
                <div class="pulse-card" style="border-top: 5px solid {color_w}">
                    <h3>MEDIUM-TERM (Weekly)</h3>
                    <h1 style="color:{color_w}">{v_w}</h1>
                    <p>Confidence: {s_w}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                 # Auto-Targets
                atr_w = c_w['ATR']
                price_w = c_w['Close']
                t1_w = price_w + (atr_w * 3)
                sl_w = price_w - (atr_w * 2)
                
                st.markdown("#### 🎯 Execution Levels")
                st.info(f"**Major Target:** {t1_w:,.2f}")
                st.error(f"**Major Support:** {sl_w:,.2f}")
                
                with st.expander("Structural Drivers (Weekly)"):
                    for r in r_w: st.write(f"• {r}")

        except Exception as e: st.error(f"Analysis Error: {e}")

def render_pulse_tab():
    st.subheader("🌍 Global Market Dashboard")
    
    # SUB-TABS for Assets
    p1, p2, p3, p4 = st.tabs(["INDICES", "STOCKS", "CRYPTO", "FOREX"])
    
    def pulse_grid(tickers):
        cols = st.columns(4)
        for i, (sym, name) in enumerate(tickers.items()):
            try:
                d = yf.Ticker(sym).history(period="2d")
                if len(d) > 1:
                    curr = d['Close'].iloc[-1]
                    prev = d['Close'].iloc[-2]
                    chg = ((curr-prev)/prev)*100
                    color = "#28a745" if chg >=0 else "#dc3545"
                    with cols[i%4]:
                        st.markdown(f"""
                        <div class="pulse-card">
                            <div style="font-size:12px; color:#8b949e;">{name}</div>
                            <div style="font-size:20px; font-weight:bold; color:{color}">{curr:,.2f}</div>
                            <div style="font-size:14px; color:{color}">{chg:+.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
            except: pass

    with p1: pulse_grid({"^NSEI":"NIFTY 50", "^NSEBANK":"BANK NIFTY", "^GSPC":"S&P 500", "^IXIC":"NASDAQ"})
    with p2: pulse_grid({"RELIANCE.NS":"Reliance", "HDFCBANK.NS":"HDFC Bank", "AAPL":"Apple", "TSLA":"Tesla"})
    with p3: pulse_grid({"BTC-USD":"Bitcoin", "ETH-USD":"Ethereum", "SOL-USD":"Solana", "DOGE-USD":"Dogecoin"})
    with p4: pulse_grid({"EURUSD=X":"EUR/USD", "GBPUSD=X":"GBP/USD", "JPY=X":"USD/JPY", "INR=X":"USD/INR"})

# --- 5. MAIN APP ---
st.title("🖥️ TITAN X: HEDGE FUND TERMINAL")

with st.sidebar:
    st.header("Search")
    asset_input = st.text_input("Ticker", "GOLD")
    search_btn = st.button("LOAD TERMINAL", type="primary")

if search_btn or asset_input:
    tick, name, curr_sym = resolve_ticker(asset_input)
    
    # MAIN NAVIGATION
    tab_pulse, tab_analysis, tab_chart, tab_strategy, tab_edu = st.tabs([
        "🔴 LIVE PULSE", "🤖 AI ANALYSIS", "📈 PRO CHART", "🧪 STRATEGY", "🎓 ACADEMY"
    ])
    
    with tab_pulse: 
        render_pulse_tab()
        st.divider()
        st.write(f"### Active Asset: {name}")
        # Mini ticker for selected asset
        t = yf.Ticker(tick).history(period="2d")
        if not t.empty:
            c = t['Close'].iloc[-1]
            p = t['Close'].iloc[-2]
            ch = ((c-p)/p)*100
            col = "green" if ch>=0 else "red"
            st.markdown(f"# {c:,.2f} :{col}[{ch:+.2f}%]")

    with tab_analysis: render_analysis_tab(tick, name)
    
    with tab_chart: render_chart_tab(tick, name)
    
    with tab_strategy:
        st.subheader("🧪 Blind Strategy Lab")
        st.info("Don't check the AI Verdict yet! Analyze the chart and lock your view.")
        bias = st.radio("Your Verdict", ["Neutral", "Bullish", "Bearish"])
        if bias != "Neutral": st.success("View Locked. Check the AI Analysis tab to compare.")

    with tab_edu:
        st.subheader("🎓 Market Concepts")
        st.info("**Smart Money Levels:** These are automatically detected previous pivot highs and lows where institutional orders often rest.")
        st.info("**Heikin Ashi:** 'Average Bar' in Japanese. Smoothes trends to filter out noise.")

st.markdown('<div class="disclaimer">TITAN X TERMINAL | INSTITUTIONAL GRADE ANALYTICS | EDUCATIONAL USE ONLY</div>', unsafe_allow_html=True)
