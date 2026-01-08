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
    highest_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - lowest_14) / (highest_14 - lowest_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # CCI (FIXED: Replaced deprecated .mad() with numpy calculation)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(20).mean()
    
    # Custom Mean Absolute Deviation function
    def get_mad(x):
        return np.mean(np.abs(x - np.mean(x)))
    
    mad_tp = tp.rolling(20).apply(get_mad)
    df['CCI'] = (tp - sma_tp) / (0.015 * mad_tp)

    # 3. VOLATILITY -------------------------------------------
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    std_dev = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Mid'] - (std_dev * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']

    # ATR
    df['ATR'] = atr_14

    # 4. VOLUME -----------------------------------------------
    # MFI
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
        df = stock.history(period="2y")
        
        if df.empty: return None, "❌ No Data Found. Market may be closed."
        
        # Calculate Indicators
        df = calculate_indicators(df)
        curr = df.iloc[-1]
        
        # Fundamentals (Try/Except block for safety)
        try:
            info = stock.info
            fund_data = {
                "PE Ratio": info.get('trailingPE', 'N/A'),
                "Market Cap": info.get('marketCap', 'N/A'),
                "Sector": info.get('sector', 'N/A')
            }
        except:
            fund_data = {"PE Ratio": "N/A", "Market Cap": "N/A", "Sector": "N/A"}

        # --- SCORING ALGORITHM (0-100) ---
        score = 0
        details = []

        # A. TREND
        if curr['Close'] > curr['SMA_200']:
            score += 15
            details.append("✅ PRICE > 200 SMA (Long Term Bull Trend)")
        if curr['Close'] > curr['EMA_20']:
            score += 10
            details.append("✅ PRICE > 20 EMA (Short Term Bullish)")
        if curr['ADX'] > 25:
            score += 5
            details.append(f"✅ ADX {curr['ADX']:.0f} (Strong Trend)")
        if curr['Tenkan_sen'] > curr['Kijun_sen']:
            score += 10
            details.append("✅ Ichimoku Bullish (Conversion > Base)")

        # B. MOMENTUM
        if 40 < curr['RSI'] < 70:
            score += 10
            details.append(f"✅ RSI {curr['RSI']:.0f} (Healthy)")
        elif curr['RSI'] > 70:
            details.append(f"⚠️ RSI {curr['RSI']:.0f} (Overbought)")
        
        if curr['MACD'] > curr['MACD_Signal']:
            score += 10
            details.append("✅ MACD Buy Signal")
        
        if curr['Stoch_K'] < 80 and curr['Stoch_K'] > curr['Stoch_D']:
            score += 10
            details.append("✅ Stochastic Rising")

        # C. VOLUME
        if curr['MFI'] > 50:
            score += 10
            details.append("✅ Money Flow > 50 (Buying Pressure)")
        
        # D. VOLATILITY
        if curr['BB_Width'] < 0.10:
            score += 10
            details.append("⚡ Bollinger Squeeze (Big Move Coming)")

        # VERDICT
        rating = "HOLD 😐"
        color = "orange"
        if score >= 80: rating, color = "STRONG BUY 🚀", "green"
        elif score >= 60: rating, color = "BUY ✅", "lightgreen"
        elif score <= 30: rating, color = "STRONG SELL 🔻", "red"
        elif score <= 45: rating, color = "SELL 📉", "salmon"

        # LEVELS
        atr_val = curr['ATR']
        stop_loss = curr['Close'] - (atr_val * 2)
        target = curr['Close'] + (atr_val * 3)
        
        return {
            "Symbol": ticker, "Name": long_name, "Price": curr['Close'],
            "Score": score, "Rating": rating, "Color": color,
            "Details": details, "Stop Loss": stop_loss, "Target": target,
            "Fundamentals": fund_data,
            "Indicators": {"RSI": curr['RSI'], "MACD": curr['MACD'], "ADX": curr['ADX'], "CCI": curr['CCI']}
        }, None

    except Exception as e:
        return None, f"Calculation Error: {str(e)}"

# --- 6. EMAIL ENGINE ---
def send_email_report(data, email_to):
    try:
        user = st.secrets["EMAIL_USER"]
        pwd = st.secrets["EMAIL_PASS"]
    except:
        st.error("⚠️ Secrets Missing in Streamlit Settings.")
        return

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = email_to
    msg['Subject'] = f"⚡ TITAN X: {data['Rating']} on {data['Name']}"
    
    reasons_html = "".join([f"<li>{d}</li>" for d in data['Details']])
    
    html = f"""
    <div style="font-family: Arial; padding: 20px; border: 1px solid #ddd;">
        <h2 style="color: #2c3e50;">TITAN X Analysis: {data['Name']}</h2>
        <h1 style="color: {data['Color']};">{data['Rating']}</h1>
        <p>Score: <b>{data['Score']}/100</b></p>
        <hr>
        <h3>📊 Signals</h3>
        <ul>{reasons_html}</ul>
        <h3>🎯 Levels</h3>
        <p><b>Price:</b> {data['Price']:.2f}</p>
        <p><b>🛑 Stop Loss:</b> {data['Stop Loss']:.2f}</p>
        <p><b>✅ Target:</b> {data['Target']:.2f}</p>
        <br>
        <p style="color: red; border: 2px solid red; padding: 10px; text-align: center; font-weight: bold;">
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
        st.success(f"✅ Report sent to {email_to}")
    except Exception as e:
        st.error(f"Email Failed: {e}")

# --- 7. MAIN UI ---
if run_btn:
    with st.spinner(f"Analyzing {user_ticker}..."):
        data, error = analyze_asset(user_ticker)
        
        if error:
            st.error(error)
        else:
            # HEADER
            st.markdown(f"## {data['Name']} ({data['Symbol']})")
            
            # METRICS ROW
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{data['Price']:.2f}")
            c2.metric("Score", f"{data['Score']}/100")
            c3.metric("RSI", f"{data['Indicators']['RSI']:.1f}")
            c4.metric("ADX", f"{data['Indicators']['ADX']:.1f}")
            
            st.divider()
            
            # SPLIT VIEW
            left, right = st.columns([2, 1])
            
            with left:
                st.subheader(f"Verdict: :{data['Color']}[{data['Rating']}]")
                for item in data['Details']:
                    if "✅" in item: st.success(item)
                    elif "⚠️" in item: st.warning(item)
                    elif "⚡" in item: st.info(item)
                
                if data['Fundamentals']['PE Ratio'] != 'N/A':
                    st.caption(f"Fundamentals: PE {data['Fundamentals']['PE Ratio']} | Sector: {data['Fundamentals']['Sector']}")

            with right:
                st.subheader("🎯 Plan")
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="color:green; margin:0;">{data['Target']:.2f}</h3>
                    <small>TARGET</small>
                </div>
                <div class="metric-box">
                    <h3 style="color:red; margin:0;">{data['Stop Loss']:.2f}</h3>
                    <small>STOP LOSS</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**MACD:** {data['Indicators']['MACD']:.2f}")
                st.write(f"**CCI:** {data['Indicators']['CCI']:.2f}")

            # Send Email
            if user_email:
                send_email_report(data, user_email)

            # DISCLAIMER
            st.markdown("""
            <div class="disclaimer">
            *** DISCLAIMER - THE ABOVE ANALYSIS IS STRICTLY FOR EDUCATION PURPOSE AND NO TIP OR INVESTMENT ADVICE. SO, KINDLY DO YOUR THOROUGH RESEARCH ***
            </div>
            """, unsafe_allow_html=True)
