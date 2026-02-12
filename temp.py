# main.py - FINAL STABLE OFFLINE ANALYSIS ENGINE

import os
import urllib.parse
import pandas as pd
import numpy as np
import upstox_client
import csv
from math import floor

# --- Technical Analysis Libraries ---
import pandas_ta as ta

# --- Configuration & Credentials ---
try:
    from config.keys import API_KEY, API_SECRET, REDIRECT_URI
except ImportError:
    API_KEY = "YOUR_UPSTOX_API_KEY"
    API_SECRET = "YOUR_UPSTOX_API_SECRET"
    REDIRECT_URI = "http://127.0.0.1:3000"


# --- Global Variables ---
ACCESS_TOKEN = None
INSTRUMENTS_FILE = 'instruments.csv'
API_VERSION = '2.0' 
RISK_REWARD_RATIO = 2.0  # Enforce minimum 1:2 R:R
MIN_SL_DISTANCE_ABS = 1.00 # Minimum Rs 1.00 stop loss buffer


# -----------------------------------------------------------------
# 1 & 2. AUTHENTICATION (UNCHANGED)
# -----------------------------------------------------------------
def generate_login_url():
    """Generates the URL for manual login authorization."""
    auth_api = upstox_client.LoginApi()
    query_params = {'client_id': API_KEY, 'redirect_uri': REDIRECT_URI, 'response_type': 'code'}
    auth_url = "https://api.upstox.com/v2/login/authorization/dialog"
    login_url = f"{auth_url}?{urllib.parse.urlencode(query_params)}"
    
    print("\n" + "="*50)
    print("STEP 1: LOGIN REQUIRED")
    print("="*50)
    print("1. Visit the following URL in your web browser:")
    print(f"\n{login_url}\n")
    print("... (Follow remaining manual steps)")
    
    return input("Paste the final redirected URL here: ")

def get_access_token(full_redirect_url):
    """Exchanges the authorization code for an Access Token."""
    global ACCESS_TOKEN
    parsed_url = urllib.parse.urlparse(full_redirect_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    
    if 'code' not in query_params:
        print("\n[ERROR] Authorization Code not found.")
        return None

    auth_code = query_params['code'][0]
    token_api = upstox_client.LoginApi()
    
    try:
        response = token_api.token(
            code=auth_code, client_id=API_KEY, client_secret=API_SECRET,
            redirect_uri=REDIRECT_URI, grant_type='authorization_code',
            api_version=API_VERSION
        )
        ACCESS_TOKEN = response.access_token
        print("\n[SUCCESS] Access Token generated successfully!")
        print(f"Token (Expires daily): {ACCESS_TOKEN[:10]}...")
        return ACCESS_TOKEN

    except upstox_client.rest.ApiException as e:
        if "UDAPI100058" in str(e):
            print("\n[ERROR] Account Segments Inactive (UDAPI100058). Activate segments in Upstox app.")
        else:
            print(f"\n[ERROR] Failed to get Access Token:\n{e}")
        return None


# -----------------------------------------------------------------
# 3. SYNTHETIC DATA GENERATION (CORRECTED PERIODS)
# -----------------------------------------------------------------
# main.py (Change periods to 1000)

def create_synthetic_data(symbol='TEST_STOCK', periods=1000): # <-- INCREASED TO 1000
    """Generates a DataFrame resembling 5-minute Intraday data for testing."""
    print(f"\n[INFO] Creating synthetic data for {symbol} ({periods} candles)...")
    
    end_time = pd.Timestamp.now(tz='Asia/Kolkata').floor('5min') 
    index = pd.date_range(end=end_time, periods=periods, freq='5min') 
    
    np.random.seed(42)
    base_price = 1000
    prices = base_price + np.cumsum(np.random.randn(periods) * 0.5)
    
    df = pd.DataFrame(index=index)
    df['Open'] = prices
    df['High'] = df['Open'] + np.random.rand(periods) * 1.5
    df['Low'] = df['Open'] - np.random.rand(periods) * 1.5
    df['Close'] = prices + np.random.randn(periods) * 0.5
    df['Volume'] = np.random.randint(10000, 50000, periods)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    return df.dropna()


# -----------------------------------------------------------------
# 4. MANUAL PIVOT POINT CALCULATION & SMC LOGIC
# -----------------------------------------------------------------
def calculate_classic_pivots(df):
    """Calculates Classic Pivot Point (PP) and S/R levels."""
    # Note: Using the last candle's HLC for calculation simplicity in offline mode.
    H, L, C = df['High'].iloc[-1], df['Low'].iloc[-1], df['Close'].iloc[-1]
    PP = (H + L + C) / 3
    R1 = (2 * PP) - L
    S1 = (2 * PP) - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    return PP, R1, S1, R2, S2

def add_smc_levels(df):
    """Detects FVG and a recent Order Block for SL/TP placement."""
    print("[INFO] Detecting FVG and Order Block (SMC) levels...")
    
    # --- FVG Logic (Requires 2-row lookback, so first two rows will be NaN) ---
    df['FVG_BULLISH'] = (df['Low'].shift(2) > df['High'].shift(0)).astype(int) 
    df['FVG_BEARISH'] = (df['High'].shift(2) < df['Low'].shift(0)).astype(int)
    
    # --- Order Block (Simplified BUY OB) ---
    df['BODY_SIZE'] = abs(df['Open'] - df['Close'])
    atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else 1.0 
    
    # Simple OB: C1 is large bearish, followed by a bounce (BUY OB is C1 Low)
    # Use boolean indexing (where=...) instead of nested np.where for robustness
    is_red_c1 = (df['Close'].shift(1) < df['Open'].shift(1))
    is_green_c2 = (df['Close'] > df['Open'])
    is_impulsive_c1 = (df['BODY_SIZE'].shift(1) > (1.5 * atr_val))
    
    # Initialize OB column to NaN
    df['OB_BUY'] = np.nan
    
    # Set OB level where conditions are met
    df.loc[is_red_c1 & is_green_c2 & is_impulsive_c1, 'OB_BUY'] = df['Low'].shift(1)

    # Carry forward the last detected OB level
    df['LAST_OB_BUY'] = df['OB_BUY'].ffill()
    
    return df


# -----------------------------------------------------------------
# 5. ENHANCED TECHNICAL ANALYSIS ENGINE
# -----------------------------------------------------------------
def add_technical_indicators(df):
    """Applies a comprehensive set of indicators using pandas-ta."""
    print("[INFO] Adding Enhanced Technical Indicators...")
    
    # 1. Momentum and Trend
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.ema(length=20, append=True, col_names=('EMA_20',))
    df.ta.ema(length=50, append=True, col_names=('EMA_50',))
    
    # 2. Volatility and Risk
    df.ta.atr(append=True)
    df.ta.bbands(append=True)
    
    # 3. Institutional Flow (VWAP)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # NOTE: We return the full DF here to let the final dropna handle all NaN columns.
    return df


# -----------------------------------------------------------------
# 6. DECISION MODEL & SL/TP LOGIC (STRUCTURAL PLACEMENT)
# -----------------------------------------------------------------
def apply_decision_model(df, symbol):
    """Applies the structural decision logic (SMC + R:R) for SL/TP."""
    print(f"[INFO] Applying structural decision model for {symbol} (R:R 1:{RISK_REWARD_RATIO})...")
    
    last_row = df.iloc[-1]
    last_close = last_row['Close']
    atr_value = last_row['ATR']
    
    # --- 1. DETERMINE BIAS BASED ON CONFLUENCE (Simplified) ---
    # Confluence requires: 1) Price above EMA 50 (Trend) AND 2) FVG (Catalyst/Structure)
    if last_row['Close'] > last_row['EMA_50'] and last_row['FVG_BULLISH'] == 1:
        signal_type = "BUY"
    elif last_row['Close'] < last_row['EMA_50'] and last_row['FVG_BEARISH'] == 1:
        signal_type = "SELL"
    else:
        return "HOLD (Neutral)", last_close, last_close

    # --- 2. CALCULATE STRUCTURAL SL & TP ---
    
    if signal_type == "BUY":
        # SL: Place below the structural low (OB)
        structural_sl = last_row['LAST_OB_BUY'] 
        
        # If no OB is detected, fall back to a safer ATR multiple SL
        if np.isnan(structural_sl):
             structural_sl = last_close - (atr_value * 2.0)
        
        # SL Hunting Avoidance: Add ATR buffer and minimum distance enforcement
        sl_buffer = max(MIN_SL_DISTANCE_ABS, atr_value * 0.2)
        sl_level = structural_sl - sl_buffer
        
        # TP: Enforce R:R ratio based on the calculated risk
        risk_distance = last_close - sl_level
        tp_level = last_close + (risk_distance * RISK_REWARD_RATIO)
        recommendation = "BUY (SMC Confluence)"
        
    else: # SELL signal (Simplified symmetric logic)
        structural_sl = last_close + (atr_value * 2.0)
        
        sl_buffer = max(MIN_SL_DISTANCE_ABS, atr_value * 0.2)
        sl_level = structural_sl + sl_buffer
        
        risk_distance = sl_level - last_close
        tp_level = last_close - (risk_distance * RISK_REWARD_RATIO)
        recommendation = "SELL (SMC Confluence)"
    
    # Final Viability Check
    if (signal_type == "BUY" and tp_level <= sl_level) or (signal_type == "SELL" and tp_level >= sl_level):
        return "HOLD (Low R:R / Invalid)", last_close, last_close
        
    return recommendation, round(tp_level, 2), round(sl_level, 2)


# -----------------------------------------------------------------
# 7. DISPLAY RESULTS 
# -----------------------------------------------------------------
def display_results(df, symbol, recommendation, tp, sl, PP, R1, S1, R2, S2):
    """Prints the final report, including calculated levels."""
    latest_close = round(df.iloc[-1]['Close'], 2)
    latest_vwap = round(df.iloc[-1]['VWAP'], 2)
    latest_rsi = round(df.iloc[-1]['RSI_14'], 2)
    
    if recommendation.startswith("BUY"):
        risk = latest_close - sl
        reward = tp - latest_close
        final_rr = round(reward / risk, 2) if risk > 0 else "N/A"
    elif recommendation.startswith("SELL"):
        risk = sl - latest_close
        reward = latest_close - tp
        final_rr = round(reward / risk, 2) if risk > 0 else "N/A"
    else:
        final_rr = "N/A"

    print("\n" + "="*80)
    print(f"| ONE-CLICK INTRADAY ANALYSIS REPORT FOR: {symbol} |")
    print("="*80)
    print(f"| LATEST PRICE: {latest_close} | VWAP: {latest_vwap} | RSI: {latest_rsi}")
    print("-" * 80)
    print(f"| **FINAL RECOMMENDATION:** {recommendation}")
    print(f"| **TAKE PROFIT (TP):** {tp}")
    print(f"| **STOP LOSS (SL):** {sl}")
    print(f"| **RISK:REWARD (R:R):** 1:{final_rr}")
    print("-" * 80)
    print(f"| **CLASSIC PIVOT LEVELS**")
    print(f"| R2: {round(R2, 2)} | R1: {round(R1, 2)} | PP: {round(PP, 2)} | S1: {round(S1, 2)} | S2: {round(S2, 2)}") 
    print("-" * 80)
    print(f"| **Decision Basis:** Confluence of Price Action (FVG/OB) and Trend (EMA/RSI).")
    print("="*80)
    print("\n[TO DO NEXT]: Implement Live Chart Visualization and Real-Time Data Stream.")


# -----------------------------------------------------------------
# 8. MAIN EXECUTION FLOW (CORRECTED ORDER)
# -----------------------------------------------------------------
def start_analysis_flow():
    """The main analysis entry point, with corrected data flow order."""
    
    stock_symbol = "TATASTEEL" 
    data_df = create_synthetic_data(symbol=stock_symbol)
    
    if data_df.empty:
        print("[CRITICAL] Cannot generate initial data. Exiting.")
        return

    # --- PHASE 1: ADD ALL INDICATOR COLUMNS ---
    data_df = add_technical_indicators(data_df)
    
    # --- PHASE 2: ADD ALL SMC COLUMNS ---
    data_df = add_smc_levels(data_df)
    
    # --- PHASE 3: FINAL CLEANUP AND VALIDATION ---
    # Drop all rows that contain ANY NaN value (i.e., initial rows where indicators/SMC can't be calculated)
    data_df = data_df.dropna() 
    
    if data_df.empty:
        print("[CRITICAL] Dataframe is empty after analysis. Cannot proceed. Try increasing 'periods' in create_synthetic_data.")
        return
        
    # Manual Pivot Point Calculation (Uses the now-clean DataFrame)
    PP, R1, S1, R2, S2 = calculate_classic_pivots(data_df)
    
    # Phase 4: DECISION MODEL
    final_recommendation, tp, sl = apply_decision_model(data_df, stock_symbol)
    
    # Phase 5: OUTPUT
    display_results(data_df, stock_symbol, final_recommendation, tp, sl, PP, R1, S1, R2, S2)


def main():
    global ACCESS_TOKEN

    if not API_KEY or API_KEY == "YOUR_UPSTOX_API_KEY":
        print("[CRITICAL] Please update your API_KEY and API_SECRET in config/keys.py first.")
        return

    if os.path.exists("token.txt"):
        os.remove("token.txt")
        
    print("\n[NOTE] Skipping live data connection until Upstox segments are active.")
    
    start_analysis_flow()

if __name__ == "__main__":
    main()