import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import plotly.graph_objects as go
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- 1. PAGE CONFIGURATION ---
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

# --- 2. SMART DICTIONARY ---
ASSET_MAP = {
    "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE OIL": "CL=F", "OIL": "CL=F",
    "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "ETH": "ETH-USD",
    "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN",
    "S&P 500": "^GSPC", "NASDAQ": "^IXIC",
    "EURUSD": "EURUSD=X", "USDINR": "INR=X",
    "RELIANCE": "RELIANCE.NS", "TATASTEEL": "TATASTEEL.NS", "HDFC": "HDFCBANK.NS"
}

def resolve_ticker(user_input):
    clean = user_input.upper().strip()
    if clean in ASSET_MAP: return ASSET_MAP[clean], clean
    if "-" not in clean and len(clean) <= 4:
        # Check Crypto vs US Stock
        t = yf.Ticker(f"{clean}-USD")
        if not t.history(period="5d").empty: return f"{clean}-USD", clean
    if "." not in clean and "=" not in clean: return f"{clean}.NS", clean
    return clean, clean

# --- 3. LAYMAN LOGIC ENGINE ---
def analyze_market_layman(symbol):
    ticker, name = resolve_ticker(symbol)
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        if df.empty: return None, "❌ I couldn't find that asset. Check the spelling!"
        
        df.columns = [c.capitalize() for c in df.columns]
        curr = df.iloc[-1]
        
        # CALCULATIONS
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Bollinger Bands
        bb_mid = df['Close'].rolling(20).mean().iloc[-1]
        std = df['Close'].rolling(20).std().iloc[-1]
        bb_upper = bb_mid + (std * 2)
        bb_lower = bb_mid - (std * 2)
        bb_width = (bb_upper - bb_lower) / bb_mid
        
        # ATR (Volatility)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift()).abs()
        tr3 = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # EXPLANATIONS
        score = 0
        insights = []

        # Trend (Weather)
        if curr['Close'] > sma200:
            score += 25
            insights.append({
                "title": "🌤️ Long-Term Weather: SUMMER (Bullish)",
                "analogy": "Price is above the 200-day average. It is 'Summer'—warm and generally safe to be outside (Buy).",
                "color": "green"
            })
        else:
            insights.append({
                "title": "❄️ Long-Term Weather: WINTER (Bearish)",
                "analogy": "Price is below the 200-day average. It is 'Winter'—cold and risky. Wear a jacket (Stop Loss) or stay inside (Cash).",
                "color": "red"
            })

        # RSI (Runner)
        if 50 < rsi < 70:
            score += 25
            insights.append({
                "title": "🏃 RSI Momentum: HEALTHY RUNNER",
                "analogy": f"The runner is at {rsi:.0f} speed. Fast but not tired yet. They can keep going!",
                "color": "green"
            })
        elif rsi >= 70:
            insights.append({
                "title": "🥵 RSI Momentum: EXHAUSTED",
                "analogy": f"The runner is sprinting at {rsi:.0f}! They are out of breath and need to rest (Price drop) soon.",
                "color": "orange"
            })
        elif rsi <= 30:
            score += 10
            insights.append({
                "title": "💤 RSI Momentum: RESTED",
                "analogy": f"The runner is resting on the bench ({rsi:.0f}). They have energy to start running again soon (Bounce).",
                "color": "green"
            })

        # Volatility (Spring)
        if bb_width < 0.10:
            score += 25
            insights.append({
                "title": "🌀 Volatility: COILED SPRING",
                "analogy": "The market is quiet... too quiet. Like a coiled spring, it is about to jump aggressively.",
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
        return None, f"Analysis Error: {e}"

# --- 4. UI LAYOUT ---
st.title("🧠 TITAN X: Layman Analyst")
st.markdown("I explain the market in **Plain English**. No jargon.")

with st.sidebar:
    st.header("Search")
    user_input = st.text_input("Asset Name", value="Gold")
    user_email = st.text_input("Email (Optional)")
    
    st.divider()
    st.header("Strategy Lab")
    user_view = st.selectbox("My View", ["No Opinion", "Up (Bullish)", "Down (Bearish)"])
    if user_view != "No Opinion":
        u_sl = st.number_input("My Stop Loss", value=0.0)
    else:
        u_sl = 0
    
    run_btn = st.button("🚀 EXPLAIN MARKET")

if run_btn:
    with st.spinner(f"Reading charts for {user_input}..."):
        data, error = analyze_market_layman(user_input)
        
        if error:
            st.error(error)
        else:
            tab1, tab2, tab3 = st.tabs(["📖 The Story", "🧪 Grade Me", "📈 Chart"])
            
            # TAB 1: STORY
            with tab1:
                st.markdown(f"""
                <div class="big-verdict" style="background-color: {data['Color']};">
                    VERDICT: {data['Rating']}
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns([1.5, 1])
                with c1:
                    st.subheader("Why? (The Analogies)")
                    for item in data['Insights']:
                        st.markdown(f"""
                        <div class="analogy-box" style="border-left-color: {item['color']};">
                            <div class="analogy-title">{item['title']}</div>
                            <div class="analogy-text">"{item['analogy']}"</div>
                        </div>
                        """, unsafe_allow_html=True)
                with c2:
                    st.subheader("Safe Levels")
                    st.info(f"Price: {data['Price']:.2f}")
                    st.success(f"✅ TARGET: {data['Target']:.2f}")
                    st.error(f"🛑 STOP LOSS: {data['Stop Loss']:.2f}")

            # TAB 2: STRATEGY LAB
            with tab2:
                st.header("Professor Titan's Grade")
                if user_view != "No Opinion":
                    titan_bias = "Up (Bullish)" if data['Score'] > 40 else "Down (Bearish)"
                    if user_view == titan_bias:
                        st.success(f"✅ **A+ Logic:** We agree! You see {user_view}, and the math confirms it.")
                    else:
                        st.error(f"⚠️ **Careful:** You see {user_view}, but the market 'Weather' looks {titan_bias}.")
                    
                    if u_sl > 0:
                        risk = abs(data['Price'] - u_sl)
                        if risk < data['ATR']:
                            st.warning("⚠️ **Too Close:** Your Stop Loss is too tight. Give it more room.")
                        else:
                            st.success("✅ **Good Safety:** Your Stop Loss is safe from random noise.")
                else:
                    st.warning("Select your view in the sidebar to get graded!")

            # TAB 3: CHART
            with tab3:
                st.subheader("Price Chart")
                fig = go.Figure(data=[go.Candlestick(x=data['History'].index,
                                open=data['History']['Open'], high=data['History']['High'],
                                low=data['History']['Low'], close=data['History']['Close'])])
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="disclaimer">
            *** DISCLAIMER - EDUCATION ONLY. NO INVESTMENT ADVICE. ***
            </div>
            """, unsafe_allow_html=True)
