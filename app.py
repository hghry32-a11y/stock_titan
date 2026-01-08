import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="TITAN X: Ultimate Analyst", page_icon="⚡", layout="wide")

# Custom CSS for the RED DISCLAIMER & Metrics
st.markdown("""
<style>
.disclaimer {
    color: red;
    font-weight: bold;
    font-size: 16px;
    border: 2px solid red;
    padding: 15px;
    text-align: center;
    margin-top: 30px;
    background-color: #ffe6e6;
}
.metric-box {
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ddd;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ TITAN X: Multi-Asset Techno-Fundamental Engine")
st.markdown("""
**Universal Analyzer for:** Stocks, Commodities (Gold/Oil), Forex, Crypto, & Indices.
*Includes: Ichimoku, ADX, RSI, MACD, Stochastics, Bollinger Squeezes, MFI, and Fundamentals.*
""")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Command Center")
    user_ticker = st.text_input("Asset Symbol", value="RELIANCE", help="Type any name: 'BTC', 'GOLD', 'NIFTY', 'EURUSD'")
    user_email = st.text_input("Email Report To (Optional)", placeholder="you@gmail.com")
    run_btn = st.button("🚀 Run Full Scan", type="primary")

# --- 3. SMART ASSET RESOLVER ---
def get_valid_ticker(symbol):
    symbol = symbol.upper().strip()
    
    # List of suffixes to try automatically
    # 1. Direct (Crypto/US Stocks) | 2. Indian NSE | 3. Currencies | 4. Commodities | 5. Indices
    formats = [symbol, f"{symbol}.NS", f"{symbol}-USD", f"{symbol}=X", f"{symbol}=F", f"^{symbol}"]
    
    for fmt in formats:
        t = yf.Ticker(fmt)
        # Check if we can fetch history
        if not t.history(period="5d").empty:
            return fmt, t.info.get('longName', fmt)

    return None, None

# --- 4. ADVANCED MATH ENGINE (Manually Coded for Stability) ---
def calculate_indicators(df):
    # 1. TREND ------------------------------------------------
    # SMAs & EMAs
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ADX (Average Directional Index - Trend Strength)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([df['High'] - df['Low'], 
                    (df['High'] - df['Close'].shift()).abs(), 
                    (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df['+DI'] = 100 * (df['+DM'].rolling(14).mean() / atr_14)
    df['-DI'] = 100 * (df['-DM'].rolling(14).mean() / atr_14)
    dx = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = dx.rolling(14).mean()

    # Ichimoku Cloud (Baseline)
    high_9 = df['High'].rolling(9).max()
    low_9 = df['Low'].rolling(9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2 # Conversion Line

    high_26 = df['High'].rolling(26).max()
    low_26 = df['Low'].rolling(26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2 # Base Line

    # 2. MOMENTUM ---------------------------------------------
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    lowest_14 = df['Low'].rolling(14).min()
    highest_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - lowest_14) / (highest_14 - lowest_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: pd.Series(x).mad())
    df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp)

    # 3. VOLATILITY -------------------------------------------
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    std_dev = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Mid'] - (std_dev * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'] # Squeeze indicator

    # ATR (Average True Range)
    df['ATR'] = atr_14

    # 4. VOLUME -----------------------------------------------
    # MFI (Money Flow Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(14).sum()
    neg_mf = pd.Series(neg_flow).rolling(14).sum()
    mfi_ratio = pos_mf / neg_mf
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    return df

# --- 5. ANALYSIS LOGIC ---
def analyze_asset(symbol):
    try:
        ticker, long_name = get_valid_ticker(symbol)
        if not ticker: return None, "❌ Asset not found. Check spelling."

        stock = yf.Ticker(ticker)
        df = stock.history(period="2y") # Need 2y for 200 SMA
        
        if df.empty: return None, "❌ No Data Found."
        
        # Fundamental Check (Basic)
        info = stock.info
        fund_data = {
            "PE Ratio": info.get('trailingPE', 'N/A'),
            "Market Cap": info.get('marketCap', 'N/A'),
            "Sector": info.get('sector', 'N/A')
        }

        # Calculate ALL Indicators
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        
        # --- SCORING ALGORITHM (0-100) ---
        score = 0
        details = []

        # A. TREND (40 Points)
        if curr['Close'] > curr['SMA_200']:
            score += 15
            details.append("✅ PRICE > 200 SMA (Major Bull Trend)")
        if curr['Close'] > curr['EMA_20']:
            score += 10
            details.append("✅ PRICE > 20 EMA (Short Term Bullish)")
        if curr['ADX'] > 25:
            score += 5
            details.append(f"✅ ADX is {curr['ADX']:.1f} (Strong Trend)")
        if curr['Tenkan_sen'] > curr['Kijun_sen']:
            score += 10
            details.append("✅ Ichimoku Conversion > Base Line (Bullish)")

        # B. MOMENTUM (30 Points)
        if 40 < curr['RSI'] < 70:
            score += 10
            details.append(f"✅ RSI is {curr['RSI']:.1f} (Healthy)")
        elif curr['RSI'] > 70:
            details.append(f"⚠️ RSI is {curr['RSI']:.1f} (Overbought)")
        
        if curr['MACD'] > curr['MACD_Signal']:
            score += 10
            details.append("✅ MACD Bullish Crossover")
        
        if curr['Stoch_K'] < 80 and curr['Stoch_K'] > curr['Stoch_D']:
            score += 10
            details.append("✅ Stochastic Rising")

        # C. VOLUME & LIQUIDITY (20 Points)
        if curr['MFI'] > 50:
            score += 10
            details.append("✅ Money Flow Index > 50 (Buying Pressure)")
        
        # D. VOLATILITY (10 Points)
        if curr['BB_Width'] < 0.10:
            score += 10
            details.append("⚡ Bollinger Squeeze Detected (Expect Explosive Move)")

        # --- VERDICT ---
        rating = "HOLD 😐"
        color = "orange"
        if score >= 80: 
            rating = "STRONG BUY 🚀"
            color = "green"
        elif score >= 60: 
            rating = "BUY ✅"
            color = "lightgreen"
        elif score <= 30: 
            rating = "STRONG SELL 🔻"
            color = "red"
        elif score <= 45: 
            rating = "SELL 📉"
            color = "salmon"

        # --- LEVELS ---
        atr_val = curr['ATR']
        stop_loss = curr['Close'] - (atr_val * 2) # 2 ATR Stop
        target_1 = curr['Close'] + (atr_val * 3)  # 3 ATR Target
        
        result = {
            "Symbol": ticker,
            "Name": long_name,
            "Price": curr['Close'],
            "Score": score,
            "Rating": rating,
            "RatingColor": color,
            "Details": details,
            "Stop Loss": stop_loss,
            "Target": target_1,
            "Fundamentals": fund_data,
            "Indicators": {
                "RSI": curr['RSI'],
                "MACD": curr['MACD'],
                "ADX": curr['ADX'],
                "CCI": curr['CCI']
            }
        }
        return result, None

    except Exception as e:
        return None, f"Calculation Error: {str(e)}"

# --- 6. EMAIL ENGINE ---
def send_email_report(data, email_to):
    try:
        user = st.secrets["EMAIL_USER"]
        pwd = st.secrets["EMAIL_PASS"]
    except:
        st.error("⚠️ Email Secrets missing.")
        return

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = email_to
    msg['Subject'] = f"⚡ TITAN X Report: {data['Rating']} on {data['Name']}"
    
    # Generate Detail List HTML
    reasons_html = "".join([f"<li style='margin-bottom:5px;'>{d}</li>" for d in data['Details']])
    
    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; border: 1px solid #ddd; padding: 20px;">
        <h2 style="color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px;">TITAN X Analysis: {data['Name']}</h2>
        
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center;">
            <h1 style="color: {data['RatingColor']}; margin: 0;">{data['Rating']}</h1>
            <p style="font-size: 18px; margin-top: 5px;">Score: <b>{data['Score']}/100</b></p>
        </div>

        <h3>📊 Technical Signals</h3>
        <ul>{reasons_html}</ul>
        
        <h3>🎯 Strategic Levels</h3>
        <table style="width:100%; text-align: left;">
            <tr><td><b>Current Price:</b></td><td>{data['Price']:.2f}</td></tr>
            <tr><td><b>🛑 Stop Loss:</b></td><td style="color:red; font-weight:bold;">{data['Stop Loss']:.2f}</td></tr>
            <tr><td><b>✅ Target:</b></td><td style="color:green; font-weight:bold;">{data['Target']:.2f}</td></tr>
        </table>
        
        <br>
        <p style="color: red; font-weight: bold; border: 2px solid red; padding: 10px; text-align: center; font-size: 12px;">
            *** DISCLAIMER - THE ABOVE ANALYSIS IS STRICTLY FOR EDUCATION PURPOSE AND NO TIP OR INVESTMENT ADVICE. SO, KINDLY DO YOUR THOROUGH RESEARCH ***
        </p>
    </div>
    """
    msg.attach(MIMEText(html, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(user, pwd)
        server.send_message(msg)
        server.quit()
        st.success(f"✅ Detailed Report sent to {email_to}")
    except Exception as e:
        st.error(f"Email Failed: {e}")

# --- 7. MAIN UI ---
if run_btn:
    with st.spinner(f"Running TITAN X Algorithms on {user_ticker}..."):
        data, error = analyze_asset(user_ticker)
        
        if error:
            st.error(error)
        else:
            # HEADER
            st.markdown(f"## {data['Name']} ({data['Symbol']})")
            
            # TOP METRICS
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price", f"{data['Price']:.2f}")
            m2.metric("Total Score", f"{data['Score']}/100")
            m3.metric("RSI (14)", f"{data['Indicators']['RSI']:.1f}")
            m4.metric("ADX Strength", f"{data['Indicators']['ADX']:.1f}")
            
            st.divider()
            
            # MAIN ANALYSIS
            c_left, c_right = st.columns([2, 1])
            
            with c_left:
                st.subheader(f"Verdict: :{data['RatingColor']}[{data['Rating']}]")
                for item in data['Details']:
                    if "✅" in item: st.success(item)
                    elif "⚠️" in item: st.warning(item)
                    elif "⚡" in item: st.info(item)
                
                # Show Fundamental Snapshot if available
                if data['Fundamentals']['PE Ratio'] != 'N/A':
                    st.caption(f"**Fundamentals:** PE: {data['Fundamentals']['PE Ratio']} | Sector: {data['Fundamentals']['Sector']}")

            with c_right:
                st.subheader("🎯 Trade Plan")
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="color:green; margin:0;">{data['Target']:.2f}</h3>
                    <small>TARGET (3x ATR)</small>
                </div>
                <div style="height:10px"></div>
                <div class="metric-box">
                    <h3 style="color:red; margin:0;">{data['Stop Loss']:.2f}</h3>
                    <small>STOP LOSS (2x ATR)</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.write("**Key Indicators:**")
                st.write(f"- MACD: {data['Indicators']['MACD']:.2f}")
                st.write(f"- CCI: {data['Indicators']['CCI']:.2f}")

            # Send Email
            if user_email:
                send_email_report(data, user_email)

            # DISCLAIMER
            st.markdown("""
            <div class="disclaimer">
            *** DISCLAIMER - THE ABOVE ANALYSIS IS STRICTLY FOR EDUCATION PURPOSE AND NO TIP OR INVESTMENT ADVICE. SO, KINDLY DO YOUR THOROUGH RESEARCH ***
            </div>
            """, unsafe_allow_html=True)
