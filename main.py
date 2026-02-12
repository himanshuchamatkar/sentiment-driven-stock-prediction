# main.py - FINAL LIVE ANALYSIS ENGINE

import os
import urllib.parse
import pandas as pd
import numpy as np
import upstox_client
import csv 
import json
from math import floor
import requests 
import gzip # New import utilized for compressed file reading

# --- NEW: AI Sentiment and Data Libraries ---
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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
API_VERSION = '2.0' 
RISK_REWARD_RATIO = 2.0  
MIN_SL_DISTANCE_ABS = 1.00

# --- CRITICAL FIX: Instrument Master Configuration ---
# TARGETING THE USER'S CONFIRMED FILE NAME: NSE.csv.gz
INSTRUMENT_FILE = "NSE.csv.gz" 
INSTRUMENT_MAP = {} # This will be populated dynamically

# Known liquid keys for benchmarking and fallback
FALLBACK_KEYS = {
    'RELIANCE': 'NSE_EQ|INE002A01018',
    'TCS': 'NSE_EQ|INE467B00010', # Using simpler fallback for robustness
    'HDFCBANK': 'NSE_EQ|INE040A01034',
    'INFY': 'NSE_EQ|INE009A01021',
    'MARUTI': 'NSE_EQ|INE585B01010',
    'HINDUNILVR': 'NSE_EQ|INE030A01027',
}


# -----------------------------------------------------------------
# 1 & 2. AUTHENTICATION & CLIENT SETUP
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
        
        # Save token for next run
        with open("token.txt", "w") as f:
            f.write(ACCESS_TOKEN)

        print("\n[SUCCESS] Access Token generated successfully!")
        return ACCESS_TOKEN

    except upstox_client.rest.ApiException as e:
        if "UDAPI100058" in str(e):
            print("\n[ERROR] Account Segments Inactive (UDAPI100058). Activate segments in Upstox app.")
            if e.body:
                print(f"API Response Body: {e.body}")
        else:
            print(f"\n[ERROR] Failed to get Access Token:\n{e}")
        return None

def get_api_client():
    """Configures and returns a functional API client using the saved token."""
    global ACCESS_TOKEN
    
    if not ACCESS_TOKEN:
        try:
            with open("token.txt", "r") as f:
                ACCESS_TOKEN = f.read().strip()
        except FileNotFoundError:
            print("[CRITICAL] Token file not found. Run authentication first.")
            return None

    configuration = upstox_client.Configuration()
    configuration.access_token = ACCESS_TOKEN
    
    api_client = upstox_client.ApiClient(configuration)
    return api_client


# -----------------------------------------------------------------
# 3. LIVE DATA FETCHING
# -----------------------------------------------------------------
def fetch_live_historical_data(api_client, stock_symbol):
    """Fetches real historical 1-min data from the Upstox API."""
    
    # Get the correct instrument key from the global map
    if stock_symbol not in INSTRUMENT_MAP:
        # If full map failed, this fallback will use the hardcoded FALLBACK_KEYS
        instrument_key = FALLBACK_KEYS.get(stock_symbol, f"NSE_EQ|{stock_symbol}")
        print(f"[WARNING] Symbol {stock_symbol} not found in map. Using fallback key: {instrument_key}")
    else:
        instrument_key = INSTRUMENT_MAP[stock_symbol]
    
    print(f"[INFO] Fetching historical 1-minute data for {stock_symbol} ({instrument_key})...")
    
    history_api = upstox_client.HistoryApi(api_client)
    
    try:
        # Fetch data using the fixed valid interval '1minute'
        response = history_api.get_historical_candle_data(
            instrument_key=instrument_key,
            interval='1minute', 
            to_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            api_version=API_VERSION 
        )
        
        data_list = response.data.candles
        # Columns: timestamp, Open, High, Low, Close, Volume, Open_Interest
        data_df = pd.DataFrame(data_list, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Interest'])
        
        # FIX 1: Removed unit='ms' since the API now returns ISO 8601 strings, not milliseconds integer
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        
        # NOTE: Keeping .dropna() here to clean initial data, but relying on indicator naming later.
        data_df = data_df.set_index('timestamp').drop(columns=['Open_Interest']).dropna()

        # FIX 2: Explicitly sort the index to satisfy pandas_ta and other indicator requirements
        data_df = data_df.sort_index()

        if data_df.empty:
            print(f"[CRITICAL] No historical data found for {stock_symbol}. Check symbol/market hours.")
            return pd.DataFrame()
            
        print(f"[SUCCESS] Fetched {len(data_df)} candles from Upstox.")
        return data_df

    except upstox_client.rest.ApiException as e:
        # Improved error output
        if e.body:
             print(f"[CRITICAL] Failed to fetch historical data: {e.status} {e.reason}\nResponse: {e.body}")
        else:
             print(f"[CRITICAL] Failed to fetch historical data: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------------
# 4. AI NEWS SENTIMENT
# -----------------------------------------------------------------
def get_news_sentiment(symbol):
    """Fetches stock news using yfinance and calculates VADER sentiment score."""
    print(f"[INFO] Running AI News Sentiment Analysis for {symbol}...")
    
    # Try fetching news using two common methods for Indian stocks
    symbols_to_try = [f"{symbol}.NS", symbol] 
    
    news_items = []
    for ticker_symbol in symbols_to_try:
        try:
            ticker = yf.Ticker(ticker_symbol) 
            news_items = ticker.news
            if news_items:
                break
        except Exception:
            # Continue trying the next symbol if yfinance fails 
            continue
    
    if not news_items:
        print("[INFO] No news items found via yfinance.")
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    compound_scores = []
    
    # Analyze the latest 5 news headlines
    for item in news_items[:5]:
        # FIX: Added try/except to prevent KeyError 'title'
        try:
            headline = item.get('title') # Use .get() for safer access
            if headline:
                vs = analyzer.polarity_scores(headline)
                compound_scores.append(vs['compound'])
            else:
                print("[INFO] Skipped a news item due to missing title.")
        except Exception:
            print("[INFO] Skipped a news item due to exception during processing.")
            continue
    
    if not compound_scores:
         return 0.0

    avg_sentiment = np.mean(compound_scores)
    
    print(f"[SUCCESS] Average Sentiment Score: {round(avg_sentiment, 3)}")
    return avg_sentiment


# -----------------------------------------------------------------
# 5. TECHNICAL AND SMC LOGIC (INTEGRATED)
# -----------------------------------------------------------------
def calculate_classic_pivots(df):
    """Calculates Classic Pivot Point (PP) and S/R levels."""
    H, L, C = df['High'].max(), df['Low'].min(), df['Close'].iloc[-1]
    PP = (H + L + C) / 3
    R1 = (2 * PP) - L
    S1 = (2 * PP) - H
    R2 = PP + (H - L)
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    return PP, R1, S1, R2, S2

def add_technical_indicators(df):
    """Applies a comprehensive set of indicators using pandas-ta."""
    print("[INFO] Adding Enhanced Technical Indicators...")
    
    # These functions add columns directly to df
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.ema(length=20, append=True, col_names=('EMA_20',))
    df.ta.ema(length=50, append=True, col_names=('EMA_50',))
    # FIX: Explicitly naming ATR column to avoid KeyError 
    df.ta.atr(append=True, col_names=('ATR',)) 
    df.ta.bbands(append=True)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    return df

def add_smc_levels(df):
    """Detects FVG and a recent Order Block for SL/TP placement."""
    print("[INFO] Detecting FVG and Order Block (SMC) levels...")
    
    # FVG BULLISH: C1 Low > C3 High
    df['FVG_BULLISH'] = (df['Low'].shift(2) > df['High'].shift(0)).astype(int) 
    # FVG BEARISH: C1 High < C3 Low
    df['FVG_BEARISH'] = (df['High'].shift(2) < df['Low'].shift(0)).astype(int)

    # --- Order Block (Simplified BUY OB) ---
    df['BODY_SIZE'] = abs(df['Open'] - df['Close'])
    # ATR is now guaranteed to exist due to explicit naming
    atr_val = df['ATR'].iloc[-1]
    
    df['OB_BUY'] = np.nan
    is_red_c1 = (df['Close'].shift(1) < df['Open'].shift(1))
    is_green_c2 = (df['Close'] > df['Open'])
    is_impulsive_c1 = (df['BODY_SIZE'].shift(1) > (1.5 * atr_val))
    
    # Set OB level where conditions are met (low of C1)
    df.loc[is_red_c1 & is_green_c2 & is_impulsive_c1, 'OB_BUY'] = df['Low'].shift(1)

    df['LAST_OB_BUY'] = df['OB_BUY'].ffill()
    
    return df


# -----------------------------------------------------------------
# 6. DECISION MODEL & SL/TP LOGIC
# -----------------------------------------------------------------
def apply_decision_model(df, symbol, sentiment_score):
    """Applies the structural decision logic (SMC + R:R + Sentiment) for SL/TP."""
    
    # CRITICAL SAFETY CHECK: Ensure DataFrame is not empty after cleanup/lag calculations
    if df.empty:
        return "HOLD (TA Failed - Empty Data)", 0.0, 0.0 
        
    print(f"[INFO] Applying structural decision model for {symbol} (R:R 1:{RISK_REWARD_RATIO})...")
    
    last_row = df.iloc[-1]
    last_close = last_row['Close']
    atr_value = last_row['ATR']
    
    # --- 1. DETERMINE BIAS BASED ON CONFLUENCE ---
    # Relaxed condition: require EMA trend OR structural FVG/OB
    is_ema_buy = (last_row['Close'] > last_row['EMA_50'])
    is_fvg_ob_buy = (last_row['FVG_BULLISH'] == 1) or (not np.isnan(last_row['LAST_OB_BUY']))

    is_ema_sell = (last_row['Close'] < last_row['EMA_50'])
    is_fvg_ob_sell = (last_row['FVG_BEARISH'] == 1)

    # Final Signal Confluence: (EMA OR SMC) AND (Confirmed by sentiment or neutral)
    if (is_ema_buy or is_fvg_ob_buy) and sentiment_score >= -0.1: # Tech Buy OR Structural Buy AND Sentiment is not heavily negative
        signal_type = "BUY"
    elif (is_ema_sell or is_fvg_ob_sell) and sentiment_score <= 0.1: # Tech Sell OR Structural Sell AND Sentiment is not heavily positive
        signal_type = "SELL"
    else:
        return "HOLD (Neutral/Conflicting)", last_close, last_close

    # --- 2. CALCULATE STRUCTURAL SL & TP ---
    if signal_type == "BUY":
        structural_sl = last_row['LAST_OB_BUY'] 
        
        # If no explicit OB is found, use ATR as a fallback stop loss reference
        if np.isnan(structural_sl):
             structural_sl = last_close - (atr_value * 2.0)
        
        sl_buffer = max(MIN_SL_DISTANCE_ABS, atr_value * 0.2)
        sl_level = structural_sl - sl_buffer
        
        risk_distance = last_close - sl_level
        tp_level = last_close + (risk_distance * RISK_REWARD_RATIO)
        recommendation = "BUY (SMC/AI Confluence)"
        
    else: # SELL signal (Symmetric logic)
        # Use ATR as a fallback stop loss reference for SELL side SL (above recent high)
        structural_sl = last_close + (atr_value * 2.0) 
        
        sl_buffer = max(MIN_SL_DISTANCE_ABS, atr_value * 0.2)
        sl_level = structural_sl + sl_buffer
        
        risk_distance = sl_level - last_close
        tp_level = last_close - (risk_distance * RISK_REWARD_RATIO)
        recommendation = "SELL (SMC/AI Confluence)"
    
    # Final Viability Check
    if (signal_type == "BUY" and tp_level <= sl_level) or (signal_type == "SELL" and tp_level >= sl_level):
        return "HOLD (Low R:R / Invalid)", last_close, last_close
        
    return recommendation, round(tp_level, 2), round(sl_level, 2)


# -----------------------------------------------------------------
# 7. DISPLAY RESULTS 
# -----------------------------------------------------------------
def display_results(df, symbol, recommendation, tp, sl, PP, R1, S1, R2, S2, sentiment_score):
    """Prints the final report, including calculated levels."""
    latest_close = round(df.iloc[-1]['Close'], 2)
    latest_vwap = round(df.iloc[-1]['VWAP'], 2)
    latest_rsi = round(df.iloc[-1]['RSI_14'], 2)
    
    risk = 0
    reward = 0

    if recommendation.startswith("BUY"):
        risk = latest_close - sl
        reward = tp - latest_close
    elif recommendation.startswith("SELL"):
        risk = sl - latest_close
        reward = latest_close - tp
    
    final_rr = round(reward / risk, 2) if risk > 0 and reward > 0 else "N/A"

    sentiment_label = 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'

    print("\n" + "="*80)
    print(f"| ONE-CLICK INTRADAY ANALYSIS REPORT FOR: {symbol} |")
    print("="*80)
    print(f"| LATEST PRICE: {latest_close} | VWAP: {latest_vwap} | RSI: {latest_rsi}")
    print(f"| **AI NEWS SENTIMENT:** {round(sentiment_score, 3)} ({sentiment_label})")
    print("-" * 80)
    print(f"| **FINAL RECOMMENDATION:** {recommendation}")
    print(f"| **TAKE PROFIT (TP):** {tp}")
    print(f"| **STOP LOSS (SL):** {sl}")
    print(f"| **RISK:REWARD (R:R):** 1:{final_rr}")
    print("-" * 80)
    print(f"| **CLASSIC PIVOT LEVELS**")
    print(f"| R2: {round(R2, 2)} | R1: {round(R1, 2)} | PP: {round(PP, 2)} | S1: {round(S1, 2)} | S2: {round(S2, 2)}") 
    print("-" * 80)
    print(f"| **Decision Basis:** Confluence of Price Action (FVG/OB), Trend (EMA/RSI), and AI Sentiment.")
    print("="*80)
    # The next step instruction is updated here
    print("\n[NEXT STEP]: If analysis succeeds, expand the list of symbols to cover the entire market.")


# -----------------------------------------------------------------
# 8. MAIN EXECUTION FLOW
# -----------------------------------------------------------------

def get_symbols_to_analyze():
    """
    Reads the full instrument master compressed CSV file (if available) and populates the global INSTRUMENT_MAP.
    Returns a list of symbols to analyze.
    """
    global INSTRUMENT_MAP
    
    if os.path.exists(INSTRUMENT_FILE):
        print(f"[INFO] Reading full market data from local file: {INSTRUMENT_FILE} (GZIP CSV Reader)...")
        
        try:
            # Use gzip.open with csv.DictReader for compressed CSV file
            with gzip.open(INSTRUMENT_FILE, 'rt', encoding='utf-8') as f:
                # Assuming the headers in the compressed CSV match the documentation fields:
                reader = csv.DictReader(f)
                
                new_map = {}
                for row in reader:
                    # Column names derived from the public CSV structure
                    exchange_col = row.get('exchange')
                    symbol_col = row.get('symbol') or row.get('tradingsymbol')
                    key_col = row.get('instrument_key')
                    
                    # We only care about NSE Equity (cash market stocks)
                    if exchange_col == 'NSE_EQ' and symbol_col and key_col:
                       new_map[symbol_col] = key_col

                if not new_map:
                     raise ValueError("CSV data loaded, but no instruments found matching NSE_EQ and containing a valid key.")
                     
                INSTRUMENT_MAP = new_map
                print(f"[SUCCESS] Loaded {len(INSTRUMENT_MAP)} NSE Equity instruments from local file.")
                
                # --- NEW FILTERING LOGIC ---
                # 1. Start with the full list of symbols
                filtered_symbols = list(INSTRUMENT_MAP.keys())
                
                # 2. Filter out non-tradable/illiquid instruments (e.g., RJ, D, C for bonds/securities)
                # Added numerical checks to filter out bond/security codes
                filtered_symbols = [
                    s for s in filtered_symbols if all(c not in s for c in ['RJ', 'D', 'C', 'ETF', 'IDX', 'IETF', 'GS'])
                    and not s.isdigit() and not s[0].isdigit() # Filter out symbols starting with numbers
                ]
                
                # 3. Filter for minimum length (typically 4+ chars for primary stocks)
                filtered_symbols = [s for s in filtered_symbols if len(s) >= 3]

                # 4. Add key benchmark stocks back, just in case the filter was too aggressive
                for benchmark in FALLBACK_KEYS.keys():
                    if benchmark not in filtered_symbols:
                        filtered_symbols.append(benchmark)
                
                print(f"[INFO] Filtered down to {len(filtered_symbols)} liquid/tradable equity symbols.")
                
                # *** REMOVED THE [:50] LIMIT FOR FULL MARKET ANALYSIS ***
                return filtered_symbols
        
        except Exception as e:
            print(f"[CRITICAL] Error reading local file {INSTRUMENT_FILE}: {e}")
            print("[FALLBACK] Using hardcoded list of 5 symbols.")
            INSTRUMENT_MAP = FALLBACK_KEYS
            return list(INSTRUMENT_MAP.keys())

    else:
        # Fallback if the file isn't found locally.
        print(f"[CRITICAL] {INSTRUMENT_FILE} not found locally. Using fallback.")
        
    # Final fallback if the local file is missing
    print("[FALLBACK] Using hardcoded list of 5 symbols.")
    INSTRUMENT_MAP = FALLBACK_KEYS
    return list(INSTRUMENT_MAP.keys())


def start_analysis_flow():
    """The main analysis entry point, fetching LIVE data and running pipeline."""
    
    # --- STEP 1: GET API CLIENT ---
    api_client = get_api_client()
    if not api_client:
        return
        
    # --- PHASE 0: Get the list of stocks to process ---
    symbols_list = get_symbols_to_analyze()
    print(f"[INFO] Starting analysis for {len(symbols_list)} symbols...")

    # --- MAIN LOOP: Iterate over all symbols ---
    for stock_symbol in symbols_list:
        print(f"\n{'='*20} STARTING ANALYSIS FOR {stock_symbol} {'='*20}")
        
        # --- PHASE 0: NEWS SENTIMENT (Runs first) ---
        sentiment_score = get_news_sentiment(stock_symbol) 

        # --- PHASE 1: FETCH LIVE HISTORICAL DATA (Uses INSTRUMENT_MAP) ---
        data_df = fetch_live_historical_data(api_client, stock_symbol)
        
        if data_df.empty:
            continue
            
        # --- PHASE 2: TECHNICAL ANALYSIS CORE ---
        data_df = add_technical_indicators(data_df)
        
        # --- PHASE 3: SMC and LEVEL CALCULATION ---
        # NOTE: data_df is NOT dropped here. ATR column is now present.
        data_df = add_smc_levels(data_df)
        
        # --- FINAL CLEANUP AND VALIDATION (MOVED HERE) ---
        # The .dropna() is now safe to run as all derived columns have been created.
        data_df = data_df.dropna() 
        
        if data_df.empty:
            print(f"[CRITICAL] Dataframe for {stock_symbol} is empty after cleanup/analysis. Skipping.")
            continue
            
        # Manual Pivot Point Calculation
        PP, R1, S1, R2, S2 = calculate_classic_pivots(data_df)
        
        # Phase 4: DECISION MODEL
        final_recommendation, tp, sl = apply_decision_model(data_df, stock_symbol, sentiment_score)
        
        # Phase 5: OUTPUT
        display_results(data_df, stock_symbol, final_recommendation, tp, sl, PP, R1, S1, R2, S2, sentiment_score)


def main():
    global ACCESS_TOKEN

    if not API_KEY or API_KEY == "YOUR_UPSTOX_API_KEY":
        print("[CRITICAL] Please update your API_KEY and API_SECRET in config/keys.py first.")
        return

    # --- Authentication Check/Flow ---
    if os.path.exists("token.txt"):
        print("[INFO] Found existing token. Skipping OAuth flow.")
        # Load token here to set the global ACCESS_TOKEN
        with open("token.txt", "r") as f:
            global ACCESS_TOKEN
            ACCESS_TOKEN = f.read().strip()
    else:
        print("[SETUP] Token not found. Starting authentication flow.")
        redirected_url = generate_login_url()
        token = get_access_token(redirected_url)
        if not token:
            print("[CRITICAL] Failed to authenticate. Exiting.")
            return
            
    # Start the main analysis flow
    start_analysis_flow()

if __name__ == "__main__":
    main()
