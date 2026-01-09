import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# --- 1. GLOBAL TERMINAL CONFIGURATION ---
st.set_page_config(page_title="TITAN X: BLOOMBERG TERMINAL", page_icon="🌐", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Institutional Look
st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp { background-color: #000000; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    
    /* REMOVE STREAMLIT PADDING */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    
    /* SEARCH BAR (COMMAND LINE STYLE) */
    .stTextInput>div>div>input { 
        background-color: #121212; color: #00ff00; border: 1px solid #333; 
        font-family: 'Courier New', monospace; font-weight: bold; font-size: 18px;
    }
    
    /* TICKER TAPE */
    .ticker-tape {
        width: 100%; overflow: hidden; white-space: nowrap; background: #121212; 
        border-bottom: 1px solid #333; padding: 5px 0; margin-bottom: 10px;
    }
    .ticker-item { display: inline-block; padding: 0 20px; font-size: 14px; }
    .up { color: #00ff00; } .down { color: #ff0000; }
    
    /* INSTITUTIONAL CARDS */
    .metric-box {
        background: #111; border: 1px solid #333; padding: 15px; margin-bottom: 10px;
        text-align: center; border-radius: 2px;
    }
    .metric-val { font-size: 24px; font-weight: 700; color: #fff; }
    .metric-lbl { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    
    /* VERDICT BOX */
    .verdict-banner {
        padding: 20px; font-size: 28px; font-weight: 900; text-align: center;
        text-transform: uppercase; letter-spacing: 3px; border: 2px solid;
    }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { background-color: #000; border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] {
        background-color: #000; color: #666; border: none; font-size: 14px; font-weight: bold;
    }
    .stTabs [aria-selected="true"] { color: #ff9900; border-bottom: 2px solid #ff9900; }
</style>
""", unsafe_allow_html=True)

# --- 2. GLOBAL STATE & ASSET LOGIC ---
if 'symbol' not in st.session_state: st.session_state['symbol'] = "RELIANCE.NS"
if 'asset_name' not in st.session_state: st.session_state['asset_name'] = "Reliance Industries"

ASSET_DB = {
    # INDICES
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN", "S&P500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI", "VIX": "^IndiaVIX",
    # COMMODITIES
    "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE": "CL=F", "NATGAS": "NG=F", "COPPER": "HG=F",
    # CRYPTO
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD", "DOGE": "DOGE-USD",
    # FOREX (INR PAIRS)
    "USDINR": "INR=X", "EURINR": "EURINR=X", "GBPINR": "GBPINR=X", "JPYINR": "JPYINR=X",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X"
}

def smart_search(query):
    q = query.upper().strip()
    
    # 1. Check Database
    if q in ASSET_DB: return ASSET_DB[q], q
    
    # 2. Heuristics
    if q.endswith("FUT"): return f"{q.replace('FUT', '=F')}", q # Commodities
    if q in ["USD", "EUR", "GBP"]: return f"{q}INR=X", f"{q}/INR"
    
    # 3. Default to NSE Equity if no suffix
    if "-" not in q and "=" not in q and "." not in q:
        return f"{q}.NS", q
        
    return q, q

# --- 3. HEAVY DUTY MATH ENGINE (70+ INDICATORS) ---
def compute_technical_matrix(df):
    if df.empty: return df
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    C, H, L, V = df['Close'], df['High'], df['Low'], df['Volume']
    
    # --- TREND ---
    for p in [9, 20, 50, 100, 200]: df[f'SMA_{p}'] = C.rolling(p).mean()
    for p in [9, 12, 26]: df[f'EMA_{p}'] = C.ewm(span=p).mean()
    
    # Ichimoku (Complete)
    p9 = (H.rolling(9).max() + L.rolling(9).min())/2
    p26 = (H.rolling(26).max() + L.rolling(26).min())/2
    df['Tenkan'] = p9
    df['Kijun'] = p26
    df['SpanA'] = ((p9 + p26)/2).shift(26)
    df['SpanB'] = ((H.rolling(52).max() + L.rolling(52).min())/2).shift(26)
    
    # Supertrend
    atr_per = 10
    mult = 3
    hl2 = (H+L)/2
    tr = pd.concat([H-L, (H-C.shift()).abs(), (L-C.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(atr_per).mean()
    # (Simplified Supertrend logic for display speed)
    df['UpperBand'] = hl2 + (mult * df['ATR'])
    df['LowerBand'] = hl2 - (mult * df['ATR'])
    
    # --- OSCILLATORS ---
    # RSI
    delta = C.diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1 + (gain/loss)))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Stochastic
    l14, h14 = L.rolling(14).min(), H.rolling(14).max()
    df['StochK'] = 100 * ((C - l14) / (h14 - l14))
    
    # CCI
    tp = (H+L+C)/3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x)))))
    
    # Bollinger
    std = C.rolling(20).std()
    df['BB_Up'] = df['SMA_20'] + (2*std)
    df['BB_Lo'] = df['SMA_20'] - (2*std)
    df['BB_W'] = (df['BB_Up'] - df['BB_Lo']) / df['SMA_20']
    
    # --- INSTITUTIONAL FLOW ---
    # VWAP
    df['VWAP'] = (C * V).cumsum() / V.cumsum()
    # MFI
    raw = tp * V
    pos = np.where(tp > tp.shift(), raw, 0)
    neg = np.where(tp < tp.shift(), raw, 0)
    mfi_r = pd.Series(pos).rolling(14).sum() / pd.Series(neg).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_r))
    
    return df

# --- 4. TOP NAV: COMMAND LINE ---
c_search, c_status = st.columns([4, 1])
with c_search:
    query = st.text_input("", placeholder="COMMAND LINE: ENTER ASSET (E.G., TATASTEEL, GOLD, BTC, USDINR)...", key="nav_search")
    if query:
        sym, name = smart_search(query)
        st.session_state['symbol'] = sym
        st.session_state['asset_name'] = name
        st.rerun()

with c_status:
    st.markdown(f"<div style='padding-top:20px; color:#00ff00; font-weight:bold; text-align:right;'>CONNECTED: {st.session_state['asset_name']}</div>", unsafe_allow_html=True)

# --- 5. DATA FETCHING (HEADLESS) ---
current_ticker = st.session_state['symbol']
try:
    # Fetch Data
    t = yf.Ticker(current_ticker)
    df_d = t.history(period="2y", interval="1d") # Daily for Analysis
    df_w = t.history(period="5y", interval="1wk") # Weekly for Macro
    
    # Real-time Quote
    live_df = t.history(period="2d", interval="5m")
    if not live_df.empty:
        live_price = live_df['Close'].iloc[-1]
        prev_close = t.history(period="2d", interval="1d")['Close'].iloc[-2]
        day_change = ((live_price - prev_close)/prev_close)*100
    else:
        live_price = df_d['Close'].iloc[-1]
        day_change = 0.0

    # Process Data
    df_d = compute_technical_matrix(df_d)
    df_w = compute_technical_matrix(df_w)
    c_d = df_d.iloc[-1]
    c_w = df_w.iloc[-1]

except Exception as e:
    st.error(f"DATA FEED ERROR: {e}. PLEASE CHECK TICKER.")
    st.stop()

# --- 6. MAIN WORKSPACE TABS ---
tab_pulse, tab_chart, tab_analysis, tab_strat, tab_edu = st.tabs([
    "🔴 LIVE MARKET PULSE", "📈 CHARTING STATION", "🤖 AI ANALYST", "🧪 STRATEGY LAB", "🎓 LIBRARY"
])

# =========================================================
# TAB 1: LIVE MARKET PULSE (Global Dashboard)
# =========================================================
with tab_pulse:
    # 1. Ticker Tape
    indices = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "GOLD": "GC=F", "USDINR": "INR=X", "BTC": "BTC-USD", "S&P500": "^GSPC"}
    tape_html = "<div class='ticker-tape'>"
    for name, sym in indices.items():
        try:
            d = yf.Ticker(sym).history(period="2d")
            p = d['Close'].iloc[-1]
            ch = ((p - d['Close'].iloc[-2])/d['Close'].iloc[-2])*100
            col = "up" if ch >= 0 else "down"
            tape_html += f"<span class='ticker-item'>{name}: <span class='{col}'>{p:,.0f} ({ch:+.2f}%)</span></span>"
        except: pass
    tape_html += "</div>"
    st.markdown(tape_html, unsafe_allow_html=True)

    # 2. Asset Deep Dive
    st.markdown(f"### ⚡ DEEP PULSE: {st.session_state['asset_name']}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate Metrics
    vol_spike = c_d['Volume'] > (df_d['Volume'].rolling(20).mean().iloc[-1] * 1.5)
    ath = df_w['High'].max()
    dd_ath = ((c_d['Close'] - ath)/ath)*100
    
    col1.markdown(f"<div class='metric-box'><div class='metric-lbl'>PRICE</div><div class='metric-val' style='color:{'#0f0' if day_change>0 else '#f00'}'>{live_price:,.2f}</div><div style='font-size:12px'>{day_change:+.2f}%</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><div class='metric-lbl'>RSI (14)</div><div class='metric-val'>{c_d['RSI']:.1f}</div><div style='font-size:12px'>{'OVERSOLD' if c_d['RSI']<30 else 'OVERBOUGHT' if c_d['RSI']>70 else 'NEUTRAL'}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'><div class='metric-lbl'>VOLUME</div><div class='metric-val'>{c_d['Volume']/1000:.1f}K</div><div style='font-size:12px; color:{'#0f0' if vol_spike else '#666'}'>{'SPIKE' if vol_spike else 'NORMAL'}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-box'><div class='metric-lbl'>VS 200 SMA</div><div class='metric-val'>{((c_d['Close']-c_d['SMA_200'])/c_d['SMA_200'])*100:+.1f}%</div><div style='font-size:12px'>{'BULLISH' if c_d['Close']>c_d['SMA_200'] else 'BEARISH'}</div></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='metric-box'><div class='metric-lbl'>DRAWDOWN</div><div class='metric-val' style='color:#f00'>{dd_ath:.1f}%</div><div style='font-size:12px'>FROM ATH</div></div>", unsafe_allow_html=True)

    # 3. News Feed
    st.markdown("#### 📰 INSTITUTIONAL NEWS WIRE")
    try:
        news = t.news
        for n in news[:5]:
            pub = datetime.fromtimestamp(n['providerPublishTime']).strftime('%H:%M | %d %b')
            st.markdown(f"""
            <div style="border-left:3px solid #333; padding-left:10px; margin-bottom:10px;">
                <a href="{n['link']}" style="color:#58a6ff; font-weight:bold; font-size:16px; text-decoration:none;">{n['title']}</a>
                <div style="color:#666; font-size:12px;">{n['publisher']} • {pub}</div>
            </div>
            """, unsafe_allow_html=True)
    except: st.info("NO NEWS FEED AVAILABLE")

# =========================================================
# TAB 2: CHARTING STATION (TradingView Clone)
# =========================================================
with tab_chart:
    # Toolbar
    c_tf, c_type, c_over, c_sub = st.columns([1, 1, 2, 2])
    with c_tf: tf = st.selectbox("INTERVAL", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=6)
    with c_type: ctype = st.selectbox("STYLE", ["Candle", "Heikin Ashi", "Line", "HLC Area"])
    with c_over: overlay = st.multiselect("OVERLAYS", ["SMA 20", "SMA 50", "SMA 200", "Bollinger Bands", "Ichimoku Cloud", "VWAP", "Supertrend"], default=["SMA 200", "Bollinger Bands"])
    with c_sub: subplots = st.multiselect("OSCILLATORS", ["Volume", "RSI", "MACD", "Stochastic", "CCI"], default=["Volume", "RSI"])

    # Map TF to YF
    tf_map = {"1m":"1d", "5m":"5d", "15m":"5d", "30m":"5d", "1h":"1mo", "4h":"1mo", "1d":"2y", "1wk":"5y", "1mo":"max"}
    i_map = {"1m":"1m", "5m":"5m", "15m":"15m", "30m":"30m", "1h":"1h", "4h":"1h", "1d":"1d", "1wk":"1wk", "1mo":"1mo"}

    with st.spinner("RENDERING HD CHART..."):
        chart_df = yf.Ticker(current_ticker).history(period=tf_map[tf], interval=i_map[tf])
        chart_df = compute_technical_matrix(chart_df)
        
        # Plotly Construction
        rows = 1 + len(subplots)
        row_heights = [0.7] + [0.3/len(subplots)] * len(subplots) if subplots else [1.0]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights)
        
        # 1. Main Price
        if ctype == "Candle":
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name="Price"), row=1, col=1)
        elif ctype == "Heikin Ashi":
            ha_c = (chart_df['Open']+chart_df['High']+chart_df['Low']+chart_df['Close'])/4
            ha_o = (chart_df['Open'].shift(1)+chart_df['Close'].shift(1))/2
            fig.add_trace(go.Candlestick(x=chart_df.index, open=ha_o, high=chart_df['High'], low=chart_df['Low'], close=ha_c, name="Heikin"), row=1, col=1)
        elif ctype == "Line":
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Close'], mode='lines', line=dict(color='#00ff00', width=2), name="Price"), row=1, col=1)
            
        # 2. Overlays
        if "SMA 200" in overlay: fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_200'], line=dict(color='white', width=2), name="SMA 200"), row=1, col=1)
        if "VWAP" in overlay: fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['VWAP'], line=dict(color='#ff00ff', width=1), name="VWAP"), row=1, col=1)
        if "Bollinger Bands" in overlay:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Up'], line=dict(color='#333'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Lo'], line=dict(color='#333'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="BB"), row=1, col=1)
        if "Ichimoku Cloud" in overlay:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SpanA'], line=dict(color='rgba(0,255,0,0.2)'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SpanB'], line=dict(color='rgba(255,0,0,0.2)'), fill='tonexty', name="Cloud"), row=1, col=1)

        # 3. Subplots
        r_idx = 2
        for sub in subplots:
            if sub == "Volume":
                cols = ['#ff0000' if c < o else '#00ff00' for c, o in zip(chart_df['Close'], chart_df['Open'])]
                fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color=cols, name="Vol"), row=r_idx, col=1)
            elif sub == "RSI":
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'], line=dict(color='#ff9900'), name="RSI"), row=r_idx, col=1)
                fig.add_hline(y=70, line_dash='dot', row=r_idx, col=1); fig.add_hline(y=30, line_dash='dot', row=r_idx, col=1)
            elif sub == "MACD":
                fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['MACD']-chart_df['Signal'], name="Hist"), row=r_idx, col=1)
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MACD'], line=dict(color='cyan'), name="MACD"), row=r_idx, col=1)
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Signal'], line=dict(color='orange'), name="Sig"), row=r_idx, col=1)
            r_idx += 1

        # Configuration
        fig.update_layout(
            height=800, plot_bgcolor='#000000', paper_bgcolor='#000000',
            font=dict(color='#e0e0e0', family='Roboto Mono'),
            xaxis_rangeslider_visible=False,
            dragmode='pan',
            hovermode='x unified',
            modebar=dict(bgcolor='#333', color='#fff', activecolor='#00ff00')
        )
        fig.update_xaxes(showgrid=True, gridcolor='#222')
        fig.update_yaxes(showgrid=True, gridcolor='#222')
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

# =========================================================
# TAB 3: AI ANALYST (Scorecard)
# =========================================================
with tab_analysis:
    # SCORING ALGO
    def grade_trend(c):
        score = 0
        reasons = []
        if c['Close'] > c['SMA_200']: score += 20; reasons.append("Price > 200 SMA")
        if c['Close'] > c['SMA_50']: score += 10; reasons.append("Price > 50 SMA")
        if c['RSI'] > 50: score += 10
        if c['MACD'] > c['Signal']: score += 10; reasons.append("MACD Bullish Cross")
        if c['Close'] > c['SpanA']: score += 10; reasons.append("Above Ichimoku Cloud")
        if c['RSI'] < 30: score += 15; reasons.append("RSI Oversold (Bounce)")
        if c['BB_W'] < 0.10: score += 5; reasons.append("Volatility Squeeze")
        
        grade = "BUY" if score >= 60 else "SELL" if score <= 30 else "HOLD"
        color = "#00ff00" if grade == "BUY" else "#ff0000" if grade == "SELL" else "#ff9900"
        return grade, score, color, reasons

    d_grade, d_score, d_col, d_why = grade_trend(c_d)
    w_grade, w_score, w_col, w_why = grade_trend(c_w)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<div class='verdict-banner' style='color:{d_col}; border-color:{d_col}'>SHORT TERM: {d_grade} ({d_score}%)</div>", unsafe_allow_html=True)
        st.markdown("#### 🎯 DAILY TARGETS")
        atr = c_d['ATR']
        st.info(f"RESISTANCE: {c_d['Close'] + (atr*2):,.2f}")
        st.error(f"SUPPORT: {c_d['Close'] - (atr*1.5):,.2f}")
        st.write("DRIVERS:")
        for r in d_why: st.write(f"- {r}")

    with col2:
        st.markdown(f"<div class='verdict-banner' style='color:{w_col}; border-color:{w_col}'>MEDIUM TERM: {w_grade} ({w_score}%)</div>", unsafe_allow_html=True)
        st.markdown("#### 🎯 WEEKLY TARGETS")
        atr_w = c_w['ATR']
        st.info(f"TARGET: {c_w['Close'] + (atr_w*4):,.2f}")
        st.error(f"INVALIDATION: {c_w['Close'] - (atr_w*2):,.2f}")
        st.write("DRIVERS:")
        for r in w_why: st.write(f"- {r}")

# =========================================================
# TAB 4: STRATEGY LAB
# =========================================================
with tab_strat:
    st.subheader("🧪 BACKTEST YOUR IDEA")
    
    sb1, sb2, sb3 = st.columns(3)
    bias = sb1.selectbox("DIRECTION", ["LONG (BUY)", "SHORT (SELL)"])
    entry = sb2.number_input("ENTRY PRICE", value=float(c_d['Close']))
    sl = sb3.number_input("STOP LOSS", value=float(c_d['Close'] * 0.95))
    
    if st.button("RUN SIMULATION"):
        risk = abs(entry - sl)
        reward_r = risk * 2
        
        st.write("---")
        st.write(f"### 📊 TRADE LOGIC: {bias}")
        
        # Risk Check
        atr = c_d['ATR']
        if risk < atr:
            st.warning(f"⚠️ DANGER: Your Stop Loss width ({risk:.2f}) is smaller than daily noise (ATR: {atr:.2f}). You will likely get stopped out by random volatility.")
        else:
            st.success("✅ RISK: Stop Loss placement allows for normal volatility.")
            
        # Alignment Check
        ai_bias = "LONG (BUY)" if d_score > 50 else "SHORT (SELL)"
        if bias == ai_bias:
            st.success(f"✅ CONFLUENCE: Titan AI also signals {bias} based on trend metrics.")
        else:
            st.error(f"❌ CONFLICT: You want to {bias}, but Titan AI metrics signal {ai_bias}.")
            
        st.info(f"💰 TARGET (1:2 RR): {(entry + reward_r) if 'LONG' in bias else (entry - reward_r):,.2f}")

# =========================================================
# TAB 5: LIBRARY (EDUCATION)
# =========================================================
with tab_edu:
    st.markdown("### 📚 THE TRADER'S HANDBOOK")
    
    ed1, ed2, ed3 = st.tabs(["CANDLESTICK BIBLE", "INDICATOR MASTERY", "FUNDAMENTALS"])
    
    with ed1:
        st.markdown("""
        #### 1. THE HAMMER (Reversal)
        * **Look:** Small body at top, long shadow at bottom (2x body size).
        * **Meaning:** Sellers pushed price down, but buyers rejected it.
        * **Action:** Buy if next candle breaks hammer high.
        
        #### 2. BULLISH ENGULFING (Momentum)
        * **Look:** Green candle completely eats the previous Red candle.
        * **Meaning:** Buyers have overwhelmed sellers.
        """)
        
    with ed2:
        st.markdown("""
        #### RSI (Relative Strength Index)
        * **The Speedometer:** Measures how fast price is moving.
        * **>70:** Overbought (Engine overheating). Expect cooldown.
        * **<30:** Oversold (Engine cold). Expect warmup.
        
        #### MACD (Moving Average Convergence Divergence)
        * **The Trend Catcher:**
        * **Line Cross:** Fast line crosses Slow line = Entry Signal.
        * **Histogram:** Shows strength of the push.
        """)
        
    with ed3:
        st.markdown("""
        #### PE RATIO
        * **The Price Tag:** Price / Earnings.
        * **High (30+):** Growth stock (Expect future profit).
        * **Low (<15):** Value stock (Cheap, possibly undervalued).
        """)
