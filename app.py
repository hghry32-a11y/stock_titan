import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# 1. PAGE SETUP
st.set_page_config(page_title="TITAN Strategist", page_icon="🛡️")
st.title("🛡️ TITAN: Stock Command Center")

# 2. SIDEBAR INPUTS
with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Stock Symbol (e.g. VBL.NS)", value="VBL.NS")
    email_input = st.text_input("Email Report To:", placeholder="you@gmail.com")
    run_btn = st.button("🚀 Run Analysis", type="primary")

# 3. ANALYSIS FUNCTION
def analyze(ticker):
    try:
        # Fetch Data (Using history to avoid errors)
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            st.error(f"Stock '{ticker}' not found. Did you forget .NS?")
            return None

        # Clean Data
        df.columns = [c.capitalize() for c in df.columns]
        curr = df.iloc[-1]

        # Calculate Indicators
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ATR
        tr = df['High'] - df['Low']
        atr = tr.rolling(14).mean().iloc[-1]

        # Signals
        trend = "UPTREND ✅" if sma50 > sma200 else "DOWNTREND 🔻"
        momentum = "BULLISH 🚀" if (curr['Close'] > sma50 and 50 < rsi < 70) else "NEUTRAL 😐"
        
        levels = {
            "Price": round(curr['Close'], 2),
            "Trend": trend,
            "Momentum": momentum,
            "Stop Loss": round(curr['Close'] - (atr * 2), 2),
            "Target": round(curr['Close'] + (atr * 3), 2)
        }
        return levels

    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return None

# 4. EMAIL FUNCTION
def send_email(data, recipient):
    # Fetch secrets from Streamlit settings
    try:
        user = st.secrets["EMAIL_USER"]
        password = st.secrets["EMAIL_PASS"]
    except:
        st.warning("⚠️ Secrets not found. Add EMAIL_USER and EMAIL_PASS in Streamlit settings.")
        return

    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = recipient
    msg['Subject'] = f"🛡️ TITAN Report: {ticker_input}"
    
    body = f"""
    <h3>TITAN Analysis for {ticker_input}</h3>
    <ul>
        <li><b>Price:</b> {data['Price']}</li>
        <li><b>Trend:</b> {data['Trend']}</li>
        <li><b>Stop Loss:</b> {data['Stop Loss']}</li>
        <li><b>Target:</b> {data['Target']}</li>
    </ul>
    """
    msg.attach(MIMEText(body, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        st.success(f"✅ Email sent to {recipient}!")
    except Exception as e:
        st.error(f"Email Failed: {e}")

# 5. MAIN LOGIC
if run_btn:
    with st.spinner("Analyzing markets..."):
        result = analyze(ticker_input)
        if result:
            # Display Metrics
            c1, c2 = st.columns(2)
            c1.metric("Trend", result['Trend'])
            c2.metric("Momentum", result['Momentum'])
            
            st.divider()
            st.subheader("Key Levels")
            st.write(f"🛑 **Stop Loss:** ₹{result['Stop Loss']}")
            st.write(f"🎯 **Target:** ₹{result['Target']}")
            
            # Send Email
            if email_input:
                send_email(result, email_input)
