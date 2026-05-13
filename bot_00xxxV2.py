#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bot_00xxxV2.py — GLOBAL ENGINE v2.4
"""

import os
import sys
import math
import csv
import warnings
import datetime as dt
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG
START_CAPITAL = 50000.0
MAX_POSITIONS = 10
MIN_CASH_RATIO = 0.10
RISICO_PCT_PER_TRADE = 0.05
ATR_STOP_MULT = 2.0
SLIPPAGE_PCT = 0.001
TRADE_COST_FIXED = 15.0
TRADE_COST_PCT = 0.0035
TAX_RATE = 0.10
MAX_HOLD_DAYS = 20
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
LIVE_TRADES_FILE = "trades_live.csv"
LIVE_POSITIONS_FILE = "positions_live.csv"
LIVE_PORTFOLIO_FILE = "portfolio_live.csv"

EXCHANGES = {
    "041 Benelux": "tickers_041x.txt",
    "042 Parijs": "tickers_042x.txt",
    "043 Frankfurt": "tickers_043x.txt",
    "044 Spanje/Port": "tickers_044x.txt",
    "045 Londen": "tickers_045x.txt",
    "046 Milaan": "tickers_046x.txt",
    "047 Toronto": "tickers_047x.txt",
    "048 Nasdaq/NYSE": "tickers_048x.txt",
}

BACKTEST_START = "2019-01-01"
BACKTEST_END = dt.date.today().isoformat()

STRAT_CONFIG = {
    "Traag": {"sl_mult": 2.0, "tp_mult": 4.0},
    "Snel": {"sl_mult": 2.0, "tp_mult": 3.0},
    "Hyper Trend": {"sl_mult": 2.5, "tp_mult": 5.0},
    "Hyper Scalp": {"sl_mult": 1.5, "tp_mult": 2.5},
    "MRA Snel": {"sl_mult": 2.0, "tp_mult": 3.0},
    "MRA Traag": {"sl_mult": 2.5, "tp_mult": 4.0},
}

def trade_cost(amount):
    return TRADE_COST_FIXED + amount * TRADE_COST_PCT

def today_str():
    return dt.date.today().strftime("%Y-%m-%d")

def safe_float(val, default=float("nan")):
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except Exception:
        return default

def load_tickers_from_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace(",", "\n").replace("$", "")
        result = []
        for line in raw.splitlines():
            t = line.strip().upper()
            if t and not t.startswith("#"):
                result.append(t)
        return sorted(list(set(result)))

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram fout: {e}")
        print(text)

def bereken_atr_positie(portfolio_waarde, entry_prijs, atr, sl_mult=ATR_STOP_MULT, risico_pct=RISICO_PCT_PER_TRADE):
    risico_eur = portfolio_waarde * risico_pct
    stop_afstand = sl_mult * atr
    if stop_afstand <= 0 or entry_prijs <= 0:
        return 0, entry_prijs, 0.0
    aandelen = max(1, int(risico_eur / stop_afstand))
    stop_loss = entry_prijs - stop_afstand
    max_verlies = round(stop_afstand * aandelen, 2)
    return aandelen, stop_loss, max_verlies

if __name__ == "__main__":
    print("Bot start met backtest...")
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "backtest"
    if mode == "backtest":
        print("Backtest modus - vereist volledige implementatie")
    elif mode == "live":
        print("Live modus - vereist volledige implementatie")
    else:
        print("Gebruik: python bot_00xxxV2.py [backtest|live]")
