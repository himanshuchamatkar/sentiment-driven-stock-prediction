# api.py
# Production-ready API for StockAI (Render compatible)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
import logging
import pandas as pd
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StockAI")

app = Flask(__name__)

# IMPORTANT: allow frontend to access API
CORS(app, resources={r"/*": {"origins": "*"}})


# -----------------------------
# Example analysis function
# (Your original logic can stay)
# -----------------------------
def analyze_stock(symbol):

    ticker = yf.Ticker(f"{symbol}.NS")
    df = ticker.history(period="2y", interval="1d")

    if df.empty:
        raise ValueError("No historical data")

    latest_price = float(df["Close"].iloc[-1])

    ema50 = df["Close"].ewm(span=50).mean().iloc[-1]

    rsi = 50

    recommendation = "BUY" if latest_price > ema50 else "SELL"

    tp = latest_price * 1.05
    sl = latest_price * 0.97

    rr = (tp - latest_price) / (latest_price - sl)

    result = {
        "symbol": symbol,
        "latestPrice": latest_price,
        "vwap": latest_price,
        "rsi": rsi,
        "sentiment": 0,
        "recommendation": recommendation,
        "tp": tp,
        "sl": sl,
        "rr": rr,
        "risk": latest_price - sl,
        "reward": tp - latest_price,
        "reason": "Trend based EMA50 strategy",
        "explanation": "Simple EMA50 based signal",
        "sltp_explanation": "TP/SL derived from percentage move",
        "pivots": {},
        "latest_index": str(df.index[-1]),
        "model_confidence": 65
    }

    series = []

    for idx, row in df.tail(200).iterrows():
        series.append({
            "t": idx.isoformat(),
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
            "v": float(row["Volume"])
        })

    ema_series = df["Close"].ewm(span=50).mean().tail(200).tolist()

    return result, series, ema_series


# -----------------------------
# API Endpoint
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json()

        symbol = data.get("symbol")

        if not symbol:
            return jsonify({
                "status": "error",
                "message": "Missing symbol"
            }), 400

        analysis, series, ema50 = analyze_stock(symbol)

        return jsonify({
            "status": "success",
            "analysis": analysis,
            "series": series,
            "ema50": ema50,
            "vwap_series": [],
            "signals": {
                "type": analysis["recommendation"],
                "price": analysis["latestPrice"],
                "t": analysis["latest_index"]
            }
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# -----------------------------
# Root route (health check)
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "StockAI API running",
        "version": "1.0"
    })


# -----------------------------
# Serve frontend if needed
# -----------------------------
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


# -----------------------------
# Local run support
# -----------------------------
if __name__ == "__main__":

    PORT = int(os.environ.get("PORT", 5000))

    logger.info(f"Starting API on port {PORT}")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )