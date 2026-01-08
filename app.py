import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import plotly.graph_objects as go
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="TITAN X: Layman Analyst", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .big-verdict {
        padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px;
        color: white; font-size: 28px; font-weight: 800; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .analogy-box {
        background-color: #ffffff; border-left: 6px solid #4CAF50; padding: 15px;
        margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .analogy-title { font-weight: bold; color: #2c3e50; font-size: 16px; display: flex; align-items: center; }
    .analogy-text { color: #555; font-size: 14px; margin-top: 5px; font-style: italic; }
    .disclaimer { 
        color: #D8000C; background-color: #FFBABA; border: 2px solid #D8000C; 
        padding: 15px; text-align: center; font-weight: bold; margin-top: 30px; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SMART DICTIONARY (Human Language) ---
ASSET_MAP = {
    "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE OIL": "CL=F", "OIL": "CL=F",
    "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW JONES": "^DJI",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDINR": "INR=X",
    "RELIANCE": "RELIANCE.NS", "TATASTEEL": "TATASTEEL.NS", "HDFC": "HDFCBANK.NS", "SBI": "SBIN.NS"
}

def resolve_ticker(user_input):
    clean = user_input.upper().strip()
    # 1. Check Dictionary
    if clean in ASSET_MAP: return ASSET_MAP[clean], clean
    # 2. Check Crypto/US
    if "-" not in clean and len(clean) <= 4:
        t = yf.Ticker(f"{clean}-USD")
        if not t.history(period="5d").empty: return f"{clean}-USD", clean
    # 3. Default to India
    if "." not in clean and "=" not in clean: return f"{clean}.NS", clean
    return clean, clean

# --- 3. LAYMAN LOGIC ENGINE ---
def analyze_market_layman(symbol):
    ticker, name = resolve_ticker(symbol)
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        if df.empty: return None, "❌ I couldn't find that asset. Check the spelling!"
        
        # --- CALCULATIONS ---
        df.columns = [c.capitalize() for c in df.columns]
        curr = df.iloc[-1]
        
        # 1. Moving Averages (The "Climate")
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        
        # 2. RSI (The "Sprinter")
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 3. Bollinger Bands (The "Rubber Band")
        bb_mid = df['Close'].rolling(20).mean().iloc[-1]
        std = df['Close'].rolling(20).std().iloc[-1]
        bb_upper = bb_mid + (std * 2)
        bb_lower = bb_mid - (std * 2)
        bb_width = (bb_upper - bb_lower) / bb_mid
        
        # 4. ATR (The "Ocean Waves" / Volatility)
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # --- GENERATING LAYMAN EXPLANATIONS ---
        score = 0
        insights = []

        # A. TREND CHECK (Weather Analogy)
        if curr['Close'] > sma200:
            score += 25
            insights.append({
                "title": "🌤️ Long-Term Weather: SUMMER (Bullish)",
                "analogy": f"The price is ABOVE the 200-day average. Imagine the market is in 'Summer'. It is generally warm and safe to go outside (Buy).",
                "color": "green"
            })
        else:
            insights.append({
                "title": "❄️ Long-Term Weather: WINTER (Bearish)",
                "analogy": f"The price is BELOW the 200-day average. It is 'Winter'. It is cold and risky. You should wear a jacket (Be careful) or stay inside (Cash).",
                "color": "red"
            })

        # B. MOMENTUM CHECK (Runner Analogy)
        if 50 < rsi < 70:
            score += 25
            insights.append({
                "title": "🏃 RSI Momentum: HEALTHY RUNNER",
                "analogy": f"The runner (RSI) is at {rsi:.0f} speed. They are running fast but not tired yet. They can keep going!",
                "color": "green"
            })
        elif rsi >= 70:
            insights.append({
                "title": "🥵 RSI Momentum: EXHAUSTED (Overbought)",
                "analogy": f"The runner is sprinting at {rsi:.0f} speed! They are out of breath and face red. They NEED to stop and rest (Price drop) before running again.",
                "color": "orange"
            })
        elif rsi <= 30:
            score += 10
            insights.append({
                "title": "💤 RSI Momentum: RESTED (Oversold)",
                "analogy": f"The runner is sitting on the bench (RSI {rsi:.0f}). They have lots of energy and are ready to start sprinting soon (Price bounce).",
                "color": "green"
            })

        # C. VOLATILITY CHECK (Ocean/Spring Analogy)
        if bb_width < 0.10:
            score += 25
            insights.append({
                "title": "🌀 Volatility: COILED SPRING",
                "analogy": "The market is squeezed tight like a spring. It is very quiet... too quiet. Get ready for a massive explosive jump soon!",
                "color": "blue"
            })
        
        # VERDICT
        if score >= 75: rating, color = "STRONG BUY 🚀", "#28a745"
        elif score >= 50: rating, color = "BUY ✅", "#5dbb63"
        elif score <= 25: rating, color = "STRONG SELL 🔻", "#dc3545"
        else: rating, color = "HOLD / WAIT 😐", "#ffc107"

        # LEVELS
        stop_loss = curr['Close'] - (atr * 2)
        target = curr['Close'] + (atr * 3)

        return {
            "Symbol": ticker, "Name": name, "Price": curr['Close'],
            "Rating": rating, "Color": color, "Score": score,
            "Insights": insights, "Stop Loss": stop_loss, "Target": target,
            "ATR": atr, "History": df
        }, None

    except Exception as e:
        return None, f"Ouch! Something broke: {e}"

# --- 4. UI LAYOUT ---
st.title("🧠 TITAN X: Strategy Coach")
st.markdown("I explain the market in **Plain English**. No complicated jargon.")

with st.sidebar:
    st.header("🔎 Search")
    user_input = st.text_input("Asset Name", value="Gold", help="Try: Bitcoin, Reliance, Euro, Nifty")
    user_email = st.text_input("Email Report (Optional)")
    
    st.divider()
    st.header("🧪 Strategy Lab")
    st.info("Test your own idea! Tell me what YOU think, and I'll grade your logic.")
    user_view = st.selectbox("I think the market will go...", ["No Opinion", "Up (Bullish)", "Down (Bearish)"])
    if user_view != "No Opinion":
        u_sl = st.number_input("My Stop Loss Price is:", value=0.0)
    else:
        u_sl = 0
        
    run_btn = st.button("🚀 EXPLAIN MARKET TO ME")

if run_btn:
    with st.spinner(f"Reading the charts for {user_input}..."):
        data, error = analyze_market_layman(user_input)
        
        if error:
            st.error(error)
        else:
            # TABS
            tab1, tab2, tab3 = st.tabs(["📖 The Story", "🧪 Grade My Strategy", "📈 Chart"])
            
            # TAB 1: THE STORY
            with tab1:
                st.markdown(f"""
                <div class="big-verdict" style="background-color: {data['Color']};">
                    VERDICT: {data['Rating']}
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    st.subheader("📝 Why? (The Explanation)")
                    for item in data['Insights']:
                        st.markdown(f"""
                        <div class="analogy-box" style="border-left-color: {item['color']};">
                            <div class="analogy-title">{item['title']}</div>
                            <div class="analogy-text">"{item['analogy']}"</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("🎯 Safe Levels")
                    st.info(f"**Current Price:** {data['Price']:.2f}")
                    
                    st.success(f"""
                    **✅ TARGET (Goal)**
                    **{data['Target']:.2f}**
                    *(This is where we take profit)*
                    """)
                    
                    st.error(f"""
                    **🛑 STOP LOSS (Safety Net)**
                    **{data['Stop Loss']:.2f}**
                    *(If price drops here, sell to be safe)*
                    """)

            # TAB 2: STRATEGY LAB
            with tab2:
                st.header("🧑‍🏫 Professor Titan's Feedback")
                if user_view == "No Opinion":
                    st.warning("Please tell me your view in the Sidebar first!")
                else:
                    # 1. Bias Check
                    titan_bias = "Up (Bullish)" if data['Score'] > 40 else "Down (Bearish)"
                    if user_view == titan_bias:
                        st.success(f"✅ **A+ Logic:** We agree! You see it going {user_view}, and the math agrees.")
                    else:
                        st.error(f"⚠️ **Careful:** You think it's going {user_view}, but the charts say it looks {titan_bias}.")
                    
                    # 2. Safety Check (Stop Loss)
                    if u_sl > 0:
                        risk_dist = abs(data['Price'] - u_sl)
                        safe_dist = data['ATR']
                        st.write(f"**Your Risk Distance:** {risk_dist:.2f} points")
                        st.write(f"**Safe Distance (ATR):** {safe_dist:.2f} points")
                        
                        if risk_dist < safe_dist:
                            st.warning("⚠️ **Your Stop Loss is too close!** It's like standing 1 inch from a moving train. You will get hit by normal noise. Move it further back.")
                        else:
                            st.success("✅ **Perfect Positioning:** You gave the trade enough room to breathe.")

            # TAB 3: CHART
            with tab3:
                st.subheader(f"{data['Name']} Price Action")
                fig = go.Figure(data=[go.Candlestick(x=data['History'].index,
                                open=data['History']['Open'], high=data['History']['High'],
                                low=data['History']['Low'], close=data['History']['Close'])])
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            # DISCLAIMER
            st.markdown("""
            <div class="disclaimer">
            *** DISCLAIMER - THE ABOVE ANALYSIS IS STRICTLY FOR EDUCATION PURPOSE AND NO TIP OR INVESTMENT ADVICE. SO, KINDLY DO YOUR THOROUGH RESEARCH ***
            </div>
            """, unsafe_allow_html=True)

            # EMAIL
            if user_email:
                try:
                    u = st.secrets["EMAIL_USER"]
                    p = st.secrets["EMAIL_PASS"]
                    msg = MIMEMultipart()
                    msg['From'] = u
                    msg['To'] = user_email
                    msg['Subject'] = f"🧠 Titan Story: {data['Name']}"
                    
                    # HTML Email Construction
                    story_html = ""
                    for item in data['Insights']:
                        story_html += f"<li><b>{item['title']}</b>: {item['analogy']}</li>"
                    
                    html = f"""
                    <h2>The Story of {data['Name']}</h2>
                    <h1>Verdict: {data['Rating']}</h1>
                    <p>Price: {data['Price']:.2f}</p>
                    <hr>
                    <h3>The Explanation:</h3>
                    <ul>{story_html}</ul>
                    <hr>
                    <h3>The Plan:</h3>
                    <p><b>Target:</b> {data['Target']:.2f}</p>
                    <p><b>Stop Loss:</b> {data['Stop Loss']:.2f}</p>
                    <br>
                    <p style="color:red; font-weight:bold;">*** EDUCATIONAL USE ONLY ***</p>
                    """
                    msg.attach(MIMEText(html, 'html'))
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(u, p)
                    server.send_message(msg)
                    server.quit()
                    st.toast("Email sent!", icon="📧")
                except:
                    pass
