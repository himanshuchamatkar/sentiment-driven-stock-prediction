# api.py - All-in-one upgrade: Upstox news, FinBERT fallback (transformers/VADER),
# daily-only historical fetch, and a minimal vectorbt backtest endpoint.
#
# Optional dependencies: transformers, torch, vectorbt
# Install optional deps only if you want better sentiment/backtesting.

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, gzip, csv, traceback, time
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import requests
from math import floor

# Optional libraries (import if available)
use_transformers = False
finbert_pipe = None
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    # We'll lazy-load a FinBERT-like model when needed to avoid startup cost
    use_transformers = True
except Exception:
    use_transformers = False

use_vectorbt = False
try:
    import vectorbt as vbt
    use_vectorbt = True
except Exception:
    use_vectorbt = False

# NLP fallback (VADER)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
except Exception:
    SentimentIntensityAnalyzer = None
    nltk = None

# pandas_ta (technical indicators)
try:
    import pandas_ta as ta
except Exception:
    ta = None

# Upstox client - assumed present in your environment
try:
    import upstox_client
except Exception:
    upstox_client = None

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("api")

app = Flask(__name__)
CORS(app)

# -------------------------
# Config / keys (fallbacks)
# -------------------------
try:
    from config.keys import API_KEY, API_SECRET, REDIRECT_URI, NEWSAPI_KEY
except Exception:
    API_KEY = os.getenv('UPSTOX_API_KEY', "YOUR_UPSTOX_API_KEY")
    API_SECRET = os.getenv('UPSTOX_API_SECRET', "YOUR_UPSTOX_API_SECRET")
    REDIRECT_URI = os.getenv('UPSTOX_REDIRECT_URI', "http://127.0.0.1:3000")
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', None)

ACCESS_TOKEN = None
API_VERSION = '2.0'
RISK_REWARD_RATIO = 2.0
MIN_SL_DISTANCE_ABS = 1.00
INSTRUMENT_FILE = "NSE.csv.gz"
INSTRUMENT_MAP = {}

# Minimal fallback instrument keys
FALLBACK_KEYS = {
    'RELIANCE': 'NSE_EQ|INE002A01018',
    'TCS': 'NSE_EQ|INE467B01029',
    'BAJAJ-AUTO': 'NSE_EQ|INE917I01010',
    'HDFCBANK': 'NSE_EQ|INE040A01034',
    'INFY': 'NSE_EQ|INE009A01021',
    'MARUTI': 'NSE_EQ|INE585B01010',
}

# -------------------------
# Ensure NLTK VADER resource
# -------------------------
def ensure_vader():
    global SentimentIntensityAnalyzer
    if SentimentIntensityAnalyzer is None:
        logger.info("NLTK/VADER not installed. VADER fallback disabled.")
        return False
    try:
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading vader_lexicon...")
            nltk.download('vader_lexicon')
        return True
    except Exception as e:
        logger.exception("Failed to ensure VADER: %s", e)
        return False

vader_available = ensure_vader()
vader_analyzer = SentimentIntensityAnalyzer() if vader_available else None

# -------------------------
# Transformers (FinBERT) lazy loader
# -------------------------
def load_finbert_pipeline():
    global finbert_pipe
    if not use_transformers:
        logger.info("Transformers not available; FinBERT disabled.")
        return None
    if finbert_pipe is not None:
        return finbert_pipe
    # Use a pre-trained finance sentiment model known on HF hub; change if needed
    model_name = "yiyanghkust/finbert-tone"  # reasonable default
    try:
        logger.info("Loading transformer sentiment pipeline (%s). This may take time...", model_name)
        finbert_pipe = pipeline("sentiment-analysis", model=model_name, truncation=True, device=0 if os.getenv('CUDA_VISIBLE_DEVICES') else -1)
        logger.info("FinBERT pipeline loaded.")
        return finbert_pipe
    except Exception as e:
        logger.exception("Failed to load FinBERT pipeline: %s", e)
        finbert_pipe = None
        return None

# -------------------------
# Helper: Upstox API client
# -------------------------
def get_api_client():
    global ACCESS_TOKEN
    if not ACCESS_TOKEN:
        try:
            with open("token.txt", "r") as f:
                ACCESS_TOKEN = f.read().strip()
                logger.info("Loaded ACCESS_TOKEN from token.txt")
        except FileNotFoundError:
            logger.warning("token.txt not found — Upstox features will be disabled.")
            return None
    if upstox_client is None:
        logger.warning("upstox_client not installed; skipping broker features.")
        return None
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = ACCESS_TOKEN
        api_client = upstox_client.ApiClient(configuration)
        return api_client
    except Exception as e:
        logger.warning("Upstox client initialization failed: %s", e)
        return None

# -------------------------
# Instrument loader
# -------------------------
def get_symbols_to_analyze():
    global INSTRUMENT_MAP
    if INSTRUMENT_MAP:
        return list(INSTRUMENT_MAP.keys())
    if os.path.exists(INSTRUMENT_FILE):
        try:
            with gzip.open(INSTRUMENT_FILE, 'rt', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                new_map = {}
                for row in reader:
                    exchange_col = row.get('exchange')
                    symbol_col = row.get('symbol') or row.get('tradingsymbol')
                    key_col = row.get('instrument_key') or row.get('instrumentToken') or row.get('instrument_token')
                    if exchange_col == 'NSE_EQ' and symbol_col and key_col:
                        new_map[symbol_col] = key_col
                if not new_map:
                    logger.warning("Instrument CSV loaded but no NSE_EQ found.")
                INSTRUMENT_MAP = new_map
                logger.info("Loaded %d instruments from %s", len(INSTRUMENT_MAP), INSTRUMENT_FILE)
        except Exception as e:
            logger.warning("Failed to load instrument file: %s", e)
    INSTRUMENT_MAP.update(FALLBACK_KEYS)
    return list(INSTRUMENT_MAP.keys())

# -------------------------
# News sources
# Upstox -> NewsAPI -> yfinance
# -------------------------
def fetch_news_from_upstox(api_client, instrument_key, max_items=12):
    if not api_client or upstox_client is None:
        return []
    try:
        # SDK may differ - adapt if your SDK is different
        if hasattr(upstox_client, 'NewsApi'):
            news_api = upstox_client.NewsApi(api_client)
            resp = news_api.get_news(instrument_key=instrument_key, api_version=API_VERSION)
            items = []
            if isinstance(resp, dict):
                data = resp.get('data') or resp
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, list):
                            items.extend(v)
            else:
                items = getattr(resp, 'data', []) or getattr(resp, 'items', []) or []
            normalized = []
            for it in items[:max_items]:
                title = it.get('title') if isinstance(it, dict) else getattr(it, 'title', None)
                summary = it.get('summary') if isinstance(it, dict) else getattr(it, 'summary', None)
                pub = it.get('publishedAt') if isinstance(it, dict) else getattr(it, 'publishedAt', None)
                if title or summary:
                    normalized.append({'title': title or '', 'summary': summary or '', 'publishedAt': pub})
            logger.info("Upstox news fetched %d items for %s", len(normalized), instrument_key)
            return normalized
        else:
            logger.debug("Upstox SDK has no NewsApi in this version.")
            return []
    except Exception as e:
        logger.warning("Upstox news fetch failed: %s", e)
        return []

def fetch_news_from_newsapi(symbol, max_items=12):
    key = NEWSAPI_KEY
    if not key:
        return []
    try:
        # Query tuned for company mentions
        q = f"{symbol} stock OR {symbol} share OR {symbol} company"
        url = "https://newsapi.org/v2/everything"
        params = {'q': q, 'pageSize': max_items, 'sortBy': 'publishedAt', 'language': 'en', 'apiKey': key}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            logger.warning("NewsAPI status %s: %s", r.status_code, r.text[:200])
            return []
        data = r.json()
        articles = data.get('articles', []) or []
        normalized = []
        for a in articles[:max_items]:
            normalized.append({'title': a.get('title') or '', 'summary': a.get('description') or a.get('content') or '', 'publishedAt': a.get('publishedAt')})
        logger.info("NewsAPI returned %d items for %s", len(normalized), symbol)
        return normalized
    except Exception as e:
        logger.warning("NewsAPI fetch failed for %s: %s", symbol, e)
        return []

def fetch_news_from_yfinance(symbol, max_items=12):
    try:
        ticker = yf.Ticker(symbol if '.' in symbol else f"{symbol}.NS")
        news_items = getattr(ticker, 'news', []) or []
        normalized = []
        for it in news_items[:max_items]:
            title = it.get('title') if isinstance(it, dict) else None
            summary = it.get('summary') if isinstance(it, dict) else None
            pub = it.get('providerPublishTime') or it.get('pubDate') or it.get('publishedAt')
            if title or summary:
                normalized.append({'title': title or '', 'summary': summary or '', 'publishedAt': pub})
        logger.info("yfinance news returned %d items for %s", len(normalized), symbol)
        return normalized
    except Exception as e:
        logger.warning("yfinance news fetch failed for %s: %s", symbol, e)
        return []

# -------------------------
# Sentiment function: prefer FinBERT -> VADER
# Returns float in [-1,1] or None if no news
# -------------------------
def get_news_sentiment(api_client, symbol, max_items=12):
    if not INSTRUMENT_MAP:
        get_symbols_to_analyze()
    instrument_key = INSTRUMENT_MAP.get(symbol) or FALLBACK_KEYS.get(symbol)
    news_items = []

    # 1) Upstox
    if instrument_key and api_client:
        news_items = fetch_news_from_upstox(api_client, instrument_key, max_items=max_items)
        if news_items:
            logger.info("Using Upstox news (%d) for %s", len(news_items), symbol)

    # 2) NewsAPI
    if not news_items:
        news_items = fetch_news_from_newsapi(symbol, max_items=max_items)
        if news_items:
            logger.info("Using NewsAPI (%d) for %s", len(news_items), symbol)

    # 3) yfinance fallback
    if not news_items:
        news_items = fetch_news_from_yfinance(symbol, max_items=max_items)
        if news_items:
            logger.info("Using yfinance news (%d) for %s", len(news_items), symbol)

    if not news_items:
        logger.info("No news found for %s across sources.", symbol)
        return None

    texts = []
    for it in news_items[:max_items]:
        t = (it.get('title') or '').strip()
        s = (it.get('summary') or '').strip()
        txt = (t + ". " + s).strip()
        if txt:
            texts.append(txt)

    if not texts:
        logger.info("News items present but no usable text for %s.", symbol)
        return None

    # Prefer FinBERT if available
    if use_transformers:
        pipe = load_finbert_pipeline()
        if pipe is not None:
            try:
                # Batch classify in small chunks
                results = []
                batch = 8
                for i in range(0, len(texts), batch):
                    chunk = texts[i:i+batch]
                    out = pipe(chunk)
                    # mapping depends on model labels. For yiyanghkust/finbert-tone: labels like Positive/Negative/Neutral
                    for res in out:
                        lab = res.get('label', '').lower()
                        score = res.get('score', 0.0)
                        if 'positive' in lab:
                            results.append(score)
                        elif 'negative' in lab:
                            results.append(-score)
                        else:
                            results.append(0.0)
                if results:
                    mean_score = float(np.mean(results))
                    logger.debug("FinBERT mean score for %s = %s", symbol, mean_score)
                    return mean_score
            except Exception as e:
                logger.exception("FinBERT pipeline failed, falling back to VADER: %s", e)

    # Fallback: VADER if available
    if vader_analyzer is not None:
        try:
            scores = []
            for txt in texts:
                vs = vader_analyzer.polarity_scores(txt)
                scores.append(vs['compound'])
            if scores:
                mean_score = float(np.mean(scores))
                logger.debug("VADER mean score for %s = %s", symbol, mean_score)
                return mean_score
        except Exception as e:
            logger.exception("VADER scoring failed: %s", e)

    logger.info("Could not compute sentiment for %s.", symbol)
    return None

# -------------------------
# Indicators / SMC / SL/TP / Decision model (kept from your code)
# -------------------------
def calculate_classic_pivots(df):
    H, L, C = float(df['High'].max()), float(df['Low'].min()), float(df['Close'].iloc[-1])
    PP = (H + L + C) / 3.0
    R1 = (2 * PP) - L
    S1 = (2 * PP) - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    return PP, R1, S1, R2, S2

def add_technical_indicators(df):
    df = df.copy()
    try:
        if ta is not None:
            df.ta.rsi(append=True)
        else:
            df['RSI_14'] = df['Close'].diff().fillna(0)  # placeholder safe column
    except Exception:
        if 'RSI_14' not in df.columns:
            df['RSI_14'] = df['Close'].diff().fillna(0)
    try:
        if ta is not None:
            df.ta.ema(length=50, append=True, col_names=('EMA_50',))
        else:
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
    except Exception:
        if 'EMA_50' not in df.columns:
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
    try:
        if ta is not None:
            df.ta.atr(append=True, col_names=('ATR',))
        else:
            df['ATR'] = (df['High'] - df['Low']).rolling(14).mean().fillna(method='bfill')
    except Exception:
        if 'ATR' not in df.columns:
            df['ATR'] = (df['High'] - df['Low']).rolling(14).mean().fillna(method='bfill')
    try:
        if ta is not None:
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        else:
            df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3.0
    except Exception:
        df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3.0
    return df

def add_smc_levels(df):
    df = df.copy()
    df['FVG_BULLISH'] = ((df['Low'].shift(2) > df['High'].shift(0))).astype(int)
    df['FVG_BEARISH'] = ((df['High'].shift(2) < df['Low'].shift(0))).astype(int)
    df['BODY_SIZE'] = (df['Open'] - df['Close']).abs()
    atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns and not df['ATR'].isna().all() else (df['High'] - df['Low']).mean()
    df['OB_BUY'] = np.nan
    is_red_c1 = (df['Close'].shift(1) < df['Open'].shift(1))
    is_green_c2 = (df['Close'] > df['Open'])
    is_impulsive_c1 = (df['BODY_SIZE'].shift(1) > (1.5 * atr_val))
    df.loc[is_red_c1 & is_green_c2 & is_impulsive_c1, 'OB_BUY'] = df['Low'].shift(1)
    df['LAST_OB_BUY'] = df['OB_BUY'].ffill()
    return df

def compute_adaptive_sl_tp(last_price, atr, df, signal_type,
                           PP=None, R1=None, S1=None, timeframe='SWING',
                           min_sl_abs=1.0, rr_target=RISK_REWARD_RATIO):
    cfg = {
        'INTRADAY':  {'sl_mult': 1.0, 'max_sl_pct': 0.03, 'max_tp_pct': 0.10},
        'SWING':     {'sl_mult': 1.5, 'max_sl_pct': 0.12, 'max_tp_pct': 0.5},
        'POSITIONAL':{'sl_mult': 2.0, 'max_sl_pct': 0.20, 'max_tp_pct': 1.0},
    }
    if timeframe not in cfg: timeframe = 'SWING'
    conf = cfg[timeframe]

    atr = float(atr or 0.0)
    last = float(last_price)
    if last <= 0:
        return last, last, "Invalid last price."

    candidates = []
    try:
        last_row = df.iloc[-1]
        ob_buy = last_row.get('LAST_OB_BUY', np.nan) if 'LAST_OB_BUY' in df.columns else np.nan
        if not np.isnan(ob_buy):
            candidates.append(float(ob_buy))
    except Exception:
        pass
    if S1 is not None: candidates.append(float(S1))
    if R1 is not None: candidates.append(float(R1))
    try:
        lookback = 20 if timeframe == 'INTRADAY' else 50 if timeframe == 'SWING' else 120
        sub = df.iloc[-lookback:]
        candidates.append(float(sub['Low'].min()))
        candidates.append(float(sub['High'].max()))
    except Exception:
        pass

    support_price = None
    if signal_type == 'BUY':
        below = [c for c in candidates if c < last]
        support_price = max(below) if below else None
    else:
        above = [c for c in candidates if c > last]
        support_price = min(above) if above else None

    explanation_parts = []
    if support_price:
        explanation_parts.append(f"Structural level at {support_price:.2f}.")
    else:
        explanation_parts.append("No structural level; using ATR-based stop.")

    sl_atr_distance = conf['sl_mult'] * atr if atr > 0 else max(min_sl_abs, last * 0.01)
    if signal_type == 'BUY':
        sl_from_atr = last - sl_atr_distance
    else:
        sl_from_atr = last + sl_atr_distance

    buffer = max(min_sl_abs, 0.2 * atr if atr>0 else last * 0.005)
    sl_from_struct = (support_price - buffer) if (support_price and signal_type == 'BUY') else ((support_price + buffer) if support_price else None)

    max_sl_abs = last * conf['max_sl_pct']
    if last < 200:
        min_pct = 0.08 if timeframe == 'SWING' else 0.12 if timeframe == 'POSITIONAL' else 0.03
        pct_sl_price = last * (1 - min_pct) if signal_type == 'BUY' else last * (1 + min_pct)
        explanation_parts.append(f"Low-price rule applied ({min_pct*100:.0f}%).")
    else:
        pct_sl_price = None

    candidates_sl = [x for x in [sl_from_struct, sl_from_atr, pct_sl_price] if x is not None]
    valid = []
    for s in candidates_sl:
        if signal_type == 'BUY' and s < last: valid.append(s)
        if signal_type == 'SELL' and s > last: valid.append(s)

    if not valid:
        return last, last, "No valid SL candidate; HOLD recommended."

    final_sl = max(valid) if signal_type == 'BUY' else min(valid)
    dist = abs(last - final_sl)
    if dist < min_sl_abs:
        final_sl = last - min_sl_abs if signal_type == 'BUY' else last + min_sl_abs
        explanation_parts.append(f"Enforced minimum SL {min_sl_abs}.")

    dist = abs(last - final_sl)
    if dist > max_sl_abs:
        final_sl = last - max_sl_abs if signal_type == 'BUY' else last + max_sl_abs
        explanation_parts.append(f"SL capped to {conf['max_sl_pct']*100:.0f}% of price.")

    risk_distance = abs(last - final_sl)
    if risk_distance <= 0:
        return last, last, "Computed risk distance <= 0."

    tp_raw = last + (risk_distance * rr_target) if signal_type == 'BUY' else last - (risk_distance * rr_target)
    max_tp_abs = last * conf['max_tp_pct']
    if abs(tp_raw - last) > max_tp_abs:
        tp = last + max_tp_abs if signal_type == 'BUY' else last - max_tp_abs
        explanation_parts.append(f"TP capped to {conf['max_tp_pct']*100:.0f}% of price.")
    else:
        tp = tp_raw

    explanation_parts.append(f"SL {final_sl:.2f} (dist {risk_distance:.2f}), TP {tp:.2f}.")
    return float(final_sl), float(tp), " ".join(explanation_parts)

def apply_decision_model(df, sentiment_score, current_price, timeframe='SWING'):
    if df.empty:
        return "HOLD (TA Failed - Empty Data)", current_price, current_price, "No data"

    last_row = df.iloc[-1]
    atr_value = float(last_row.get('ATR', 0.0) or 0.0)
    last_close = float(current_price)

    is_ema_buy = (last_close > last_row.get('EMA_50', last_row.get('Close')))
    is_fvg_ob_buy = (last_row.get('FVG_BULLISH', 0) == 1) or (not np.isnan(last_row.get('LAST_OB_BUY', np.nan)))
    is_ema_sell = (last_close < last_row.get('EMA_50', last_row.get('Close')))
    is_fvg_ob_sell = (last_row.get('FVG_BEARISH', 0) == 1)

    signal_type = None
    s_score = sentiment_score if sentiment_score is not None else 0.0

    if (is_ema_buy or is_fvg_ob_buy) and s_score >= -0.1:
        signal_type = "BUY"
    elif (is_ema_sell or is_fvg_ob_sell) and s_score <= 0.1:
        signal_type = "SELL"
    else:
        return "HOLD (Neutral/Conflicting)", last_close, last_close, "Signals conflicted or sentiment not supportive."

    PP, R1, S1, R2, S2 = calculate_classic_pivots(df)
    sl_level, tp_level, sltp_expl = compute_adaptive_sl_tp(last_close, atr_value, df, signal_type, PP=PP, R1=R1, S1=S1, timeframe=timeframe)
    if (signal_type == "BUY" and tp_level <= sl_level) or (signal_type == "SELL" and tp_level >= sl_level):
        return "HOLD (Low R:R / Invalid)", last_close, last_close, "Invalid TP/SL."
    recommendation = "BUY (Hybrid Confluence)" if signal_type == "BUY" else "SELL (Hybrid Confluence)"
    return recommendation, float(tp_level), float(sl_level), sltp_expl

def generate_explanation(symbol, recommendation, df, sentiment_score, rr, sltp_expl):
    parts = []
    last = float(df['Close'].iloc[-1])
    ema50 = float(df['EMA_50'].iloc[-1]) if 'EMA_50' in df.columns else None
    rsi = float(df['RSI_14'].iloc[-1]) if 'RSI_14' in df.columns else None
    if ema50 is not None:
        parts.append(f"Price {'above' if last>ema50 else 'below'} 50-EMA ({ema50:.2f}).")
    if rsi is not None:
        if rsi >= 60: parts.append(f"Momentum strong (RSI {rsi:.1f}).")
        elif rsi <= 40: parts.append(f"Momentum weak (RSI {rsi:.1f}).")
        else: parts.append(f"Momentum neutral (RSI {rsi:.1f}).")
    sent_text = "neutral"
    if sentiment_score is not None:
        if sentiment_score > 0.2: sent_text = "positive"
        elif sentiment_score < -0.2: sent_text = "negative"
    else:
        sent_text = "not available"
    parts.append(f"News sentiment {sent_text} (score {0.0 if sentiment_score is None else sentiment_score:.2f}).")
    parts.append(f"R:R ≈ {rr:.2f}. {sltp_expl}")
    parts.append(f"Recommendation: {recommendation}.")
    return " ".join(parts)

def compile_results(df, symbol, recommendation, tp, sl, PP, R1, S1, R2, S2, sentiment_score, ltp, sltp_expl, sentiment_present=True):
    latest_close = float(ltp)
    if recommendation.startswith("BUY"):
        risk = latest_close - sl
        reward = tp - latest_close
    elif recommendation.startswith("SELL"):
        risk = sl - latest_close
        reward = latest_close - tp
    else:
        risk = reward = 0.0
    rr = round((reward / risk) if (risk>0 and reward>0) else 0.0, 2)
    reason_map = {
        "BUY (Hybrid Confluence)": "Bullish: trend + structure + sentiment confluence.",
        "SELL (Hybrid Confluence)": "Bearish: trend + structure + sentiment confluence.",
    }
    rsi_val = float(df.iloc[-1]['RSI_14']) if 'RSI_14' in df.columns else float(df['Close'].iloc[-1])
    explanation_text = generate_explanation(symbol, recommendation, df, sentiment_score, rr, sltp_expl)
    model_confidence = min(99, max(10, 50 + (rr * 10)))
    if not sentiment_present:
        model_confidence = max(5, model_confidence * 0.7)
    return {
        "symbol": symbol,
        "latestPrice": latest_close,
        "vwap": float(df.iloc[-1]['VWAP']) if 'VWAP' in df.columns else float(df['Close'].iloc[-1]),
        "rsi": rsi_val,
        "sentiment": float(0.0 if sentiment_score is None else sentiment_score),
        "sentiment_present": bool(sentiment_present),
        "recommendation": recommendation,
        "tp": float(tp),
        "sl": float(sl),
        "rr": float(rr),
        "risk": float(risk),
        "reward": float(reward),
        "reason": reason_map.get(recommendation, "Detailed analysis executed."),
        "explanation": explanation_text,
        "sltp_explanation": sltp_expl,
        "pivots": {"R2": round(R2,2), "R1": round(R1,2), "PP": round(PP,2), "S1": round(S1,2), "S2": round(S2,2)},
        "latest_index": df.index[-1].isoformat(),
        "model_confidence": round(model_confidence,1)
    }

# -------------------------
# Historical fetch: DAILY ONLY (long-term / swing)
# -------------------------
def fetch_live_historical_data(api_client, stock_symbol, interval='1d', period='5y', timeframe_hint='SWING'):
    """
    Simplified: daily data only for long-term/swing focus.
    Falls back to Upstox daily candles (if available) or yfinance daily history.
    """
    if not INSTRUMENT_MAP:
        get_symbols_to_analyze()
    instrument_key = INSTRUMENT_MAP.get(stock_symbol) or FALLBACK_KEYS.get(stock_symbol)
    if not instrument_key:
        raise ValueError(f"Symbol {stock_symbol} not found in instrument list.")

    # Try Upstox daily first
    if api_client and upstox_client is not None:
        try:
            history_api = upstox_client.HistoryApi(api_client)
            resp = history_api.get_historical_candle_data(
                instrument_key=instrument_key,
                interval='1day',
                to_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
                api_version=API_VERSION
            )
            data_list = getattr(resp, 'data', {}).get('candles', None)
            if data_list is None and isinstance(resp, dict):
                # attempt different shapes
                data_list = resp.get('data', {}).get(instrument_key, {}).get('candles')
            if data_list:
                df = pd.DataFrame(data_list, columns=['timestamp','Open','High','Low','Close','Volume','Open_Interest'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').drop(columns=['Open_Interest'], errors='ignore').dropna()
                df = df.sort_index()
                if not df.empty:
                    return df
        except Exception as e:
            logger.info("Upstox daily history failed: %s (falling back to yfinance)", e)

    # yfinance daily fallback
    yf_symbol = f"{stock_symbol}.NS"
    ticker = yf.Ticker(yf_symbol)
    try:
        hist = ticker.history(period=period, interval=interval, actions=False)
        if hist is None or hist.empty:
            raise ValueError("yfinance returned no daily history.")
        hist.index = pd.to_datetime(hist.index)
        hist = hist.rename(columns={'Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'})
        hist = hist[['Open','High','Low','Close','Volume']].dropna()
        if not hist.empty:
            return hist
    except Exception as e:
        logger.exception("yfinance daily fetch failed for %s: %s", yf_symbol, e)
        raise ValueError(f"No usable historical data for {stock_symbol}. Error: {e}")

# -------------------------
# Latest quote fetch (Upstox preferred, else last close)
# -------------------------
def fetch_latest_quote(api_client, stock_symbol):
    if not INSTRUMENT_MAP:
        get_symbols_to_analyze()
    instrument_key = INSTRUMENT_MAP.get(stock_symbol) or FALLBACK_KEYS.get(stock_symbol)
    if not instrument_key:
        raise ValueError(f"Symbol {stock_symbol} not found in instrument list.")
    if api_client and upstox_client is not None:
        try:
            quote_api = upstox_client.MarketQuoteApi(api_client)
            resp = quote_api.get_full_market_quote(instrument_key=instrument_key, api_version=API_VERSION)
            quote_data = resp.get('data', {}).get(instrument_key, {})
            if isinstance(quote_data, dict):
                ltp = None
                if 'ltp' in quote_data:
                    if isinstance(quote_data['ltp'], dict):
                        ltp = quote_data['ltp'].get('last_price') or quote_data['ltp'].get('ltp')
                    else:
                        ltp = quote_data['ltp']
                ltp = ltp or quote_data.get('last_price') or quote_data.get('lastTradedPrice')
                return float(ltp) if ltp is not None else None
        except Exception as e:
            logger.info("Upstox quote fetch failed: %s", e)
    return None

# -------------------------
# Flask endpoints
#  - /analyze : main analysis
#  - /backtest : minimal historical backtest using vectorbt if available
# -------------------------
@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    try:
        data = request.get_json() or {}
        symbol = (data.get('symbol') or '').upper()
        timeframe = (data.get('timeframe') or 'SWING').upper()
        if not symbol:
            return jsonify({"status":"error","message":"Missing stock symbol."}), 400

        api_client = get_api_client()
        get_symbols_to_analyze()
        if symbol not in INSTRUMENT_MAP and symbol not in FALLBACK_KEYS:
            return jsonify({"status":"error","message":f"Symbol {symbol} not found."}), 404

        # Daily-only fetch for long-term
        try:
            data_df = fetch_live_historical_data(api_client, symbol, interval='1d', period='5y', timeframe_hint=timeframe)
        except ValueError as e:
            return jsonify({"status":"error","message":str(e)}), 400

        ltp = fetch_latest_quote(api_client, symbol)
        if not ltp:
            ltp = float(data_df['Close'].iloc[-1])
            logger.info("Using historical close %s for %s as latest price.", ltp, symbol)

        # Sentiment: pass api_client so Upstox news can be used
        try:
            sentiment_score = get_news_sentiment(api_client, symbol)
            sentiment_present = sentiment_score is not None
        except Exception as e:
            logger.exception("Sentiment extraction failed: %s", e)
            sentiment_score = None
            sentiment_present = False

        # Indicators
        data_df = add_technical_indicators(data_df)
        data_df = add_smc_levels(data_df)
        # keep recent window to avoid NaNs from indicator warmup
        data_df = data_df.dropna(subset=['Close'])
        if 'EMA_50' in data_df.columns:
            data_df = data_df.iloc[-max(60, len(data_df)):]
        data_df = data_df.dropna()
        if data_df.empty:
            return jsonify({"status":"error","message":f"Data empty after indicators for {symbol}."}), 400

        # Decision
        PP, R1, S1, R2, S2 = calculate_classic_pivots(data_df)
        final_recommendation, tp, sl, sltp_expl = apply_decision_model(data_df, sentiment_score, ltp, timeframe=timeframe)
        analysis_result = compile_results(data_df, symbol, final_recommendation, tp, sl, PP, R1, S1, R2, S2, sentiment_score, ltp, sltp_expl, sentiment_present=sentiment_present)

        # prepare series
        N = min(360, len(data_df))
        sub = data_df.iloc[-N:]
        series = []
        for idx, row in sub.iterrows():
            series.append({'t': idx.isoformat(), 'o': float(row['Open']), 'h': float(row['High']), 'l': float(row['Low']), 'c': float(row['Close']), 'v': float(row['Volume'])})
        ema50 = [float(x) for x in sub['EMA_50'].tolist()] if 'EMA_50' in sub.columns else []
        vwap_series = [float(x) for x in sub['VWAP'].tolist()] if 'VWAP' in sub.columns else []
        signals = {'type': analysis_result['recommendation'], 'price': float(analysis_result['latestPrice']), 't': sub.index[-1].isoformat()}

        return jsonify({"status":"success","analysis":analysis_result,"series":series,"ema50":ema50,"vwap_series":vwap_series,"signals":signals})
    except Exception as e:
        traceback.print_exc()
        logger.exception("Unexpected server error: %s", e)
        return jsonify({"status":"error","message":"Unexpected server error: " + str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest_endpoint():
    """
    Minimal backtest endpoint to test SL/TP rules across historical daily data.
    Accepts JSON:
      { "symbol": "RELIANCE", "start": "2019-01-01", "end":"2024-12-31", "initial_cash":100000 }
    Requires vectorbt installed. This is a simple, fast prototype to get rough stats.
    """
    if not use_vectorbt:
        return jsonify({"status":"error","message":"vectorbt not installed. Install with `pip install vectorbt` to enable backtesting."}), 400
    try:
        payload = request.get_json() or {}
        symbol = (payload.get('symbol') or '').upper()
        if not symbol:
            return jsonify({"status":"error","message":"Missing symbol."}), 400
        start = payload.get('start', '2018-01-01')
        end = payload.get('end', None)
        cash = float(payload.get('initial_cash', 100000))
        api_client = get_api_client()
        # get daily history (5-10y)
        df = fetch_live_historical_data(api_client, symbol, interval='1d', period='10y')
        close = df['Close']
        ema50 = close.ewm(span=50).mean()
        sma200 = close.rolling(200).mean()
        # Simple rule: buy when close > ema50 and ema50 > sma200
        entries = (close > ema50) & (ema50 > sma200)
        exits = (close < ema50)  # simple exit rule
        pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=cash, fees=0.0003, slippage=0.0005)
        stats = {
            "total_return": float(pf.total_return()),
            "max_drawdown": float(pf.max_drawdown()),
            "sharpe": float(pf.sharpe_ratio()) if hasattr(pf, 'sharpe_ratio') else None,
            "trades": int(pf.total_trades),
            "win_rate": float(pf.win_rate) if hasattr(pf, 'win_rate') else None
        }
        return jsonify({"status":"success","symbol":symbol,"stats":stats})
    except Exception as e:
        traceback.print_exc()
        logger.exception("Backtest failed: %s", e)
        return jsonify({"status":"error","message":"Backtest error: " + str(e)}), 500

@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    try:
        get_symbols_to_analyze()
    except Exception as e:
        logger.warning("get_symbols_to_analyze() failed: %s", e)
    logger.info("--- API Server Initialized ---")
    logger.info("Serving analysis on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
