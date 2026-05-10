#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLOBAL SIGNAL ENGINE v2.0 — FIXED
- Automatische detectie van tickerbestanden (041–060)
- Compatibel met GitHub Actions
- Telegram-rapportage
- MRA + Trendstrategieën
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# TELEGRAM
# -------------------------------------------------------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def telegram_send(msg: str):
    if not TOKEN or not CHAT_ID:
        return
    chunks = [msg[i:i+3500] for i in range(0, len(msg), 3500)]
    for c in chunks:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={
                    "chat_id": CHAT_ID,
                    "text": c,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                },
                timeout=20
            )
            time.sleep(1)
        except Exception as e:
            logger.error(f"Telegram fout: {e}")

# -------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------
INZET = 2500.0
KOSTEN = 15.0 + (INZET * 0.0035)

# Traag (50/200)
T_ADX_MIN = 15
T_RSI_MIN = 45
T_RSI_MAX = 65
T_SL_ATR = 2.5
T_TP_ATR = 3.0

# Snel (20/50)
S_ADX_MIN = 18
S_RSI_MIN = 50
S_RSI_MAX = 70
S_SL_ATR = 2.0
S_TP_ATR = 3.0

# Hyper Trend
HT_ADX_MIN = 20
HT_RSI_MIN = 55
HT_RSI_MAX = 75
HT_SL_ATR = 2.0
HT_TP_ATR = 4.0

# Hyper Scalp
HS_IBS_MAX = 0.35
HS_RSI_MAX = 35
HS_SL_ATR = 2.0
HS_TP_ATR = 1.5

# MRA Snel
MRAS_IBS_MAX = 0.40
MRAS_RSI_MAX = 35
MRAS_SL_ATR = 3.0
MRAS_TP_PCT = 0.06

# MRA Traag
MRAT_IBS_MAX = 0.45
MRAT_RSI_MAX = 40
MRAT_SL_ATR = 3.5
MRAT_TP_PCT = 0.08

# -------------------------------------------------------------------------
# MARKT-DETECTIE
# -------------------------------------------------------------------------
def detect_market(ticker: str):
    t = ticker.upper()

    if t.endswith(".TO") or t.endswith(".V"):
        return "CA"

    EU_SUFFIXES = [
        ".AS", ".BR", ".PA", ".DE", ".MI", ".L", ".MC", ".BE",
        ".SW", ".HE", ".OL", ".ST", ".CO"
    ]
    if any(t.endswith(s) for s in EU_SUFFIXES):
        return "EU"

    return "US"

# -------------------------------------------------------------------------
# DOWNLOAD ENGINE
# -------------------------------------------------------------------------
def download_ticker(ticker: str):
    try:
        df = yf.download(
            ticker,
            period="5y",
            auto_adjust=False,
            progress=False
        )
        if df is None or len(df) < 260:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        for col in ["Close", "High", "Low", "Open", "Volume"]:
            if col in df.columns and isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        df = df.dropna()
        if len(df) < 260:
            return None

        return df

    except Exception as e:
        logger.error(f"Download fout {ticker}: {e}")
        return None

# -------------------------------------------------------------------------
# INDICATOREN
# -------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame):
    p = df["Close"].ffill()
    h = df["High"].ffill()
    l = df["Low"].ffill()
    v = df["Volume"].ffill()

    ema20  = p.ewm(span=20, adjust=False).mean()
    ema50  = p.ewm(span=50, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()

    ma10 = p.rolling(10).mean()
    ma20 = p.rolling(20).mean()
    vol_ma20 = v.rolling(20).mean()

    std20 = p.rolling(20).std()
    lower_bb = ma20 - 2.5 * std20

    ibs = (p - l) / (h - l + 1e-10)

    delta = p.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

    tr = pd.concat([
        h - l,
        (h - p.shift()).abs(),
        (l - p.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()

    up = h.diff().clip(lower=0)
    dn = (-l.diff()).clip(lower=0)
    plus_di = 100 * (up.ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (dn.ewm(alpha=1/14, adjust=False).mean() / (atr + 1e-10))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/14, adjust=False).mean() * 100

    return {
        "p": p, "h": h, "l": l, "v": v,
        "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "ma10": ma10, "ma20": ma20, "vol_ma20": vol_ma20,
        "lower_bb": lower_bb,
        "ibs": ibs, "rsi": rsi, "atr": atr, "adx": adx
    }

# -------------------------------------------------------------------------
# STRATEGIEËN
# -------------------------------------------------------------------------
# (strategieën blijven identiek — ingekort voor leesbaarheid)
# Jij hebt deze al, ze blijven 100% hetzelfde.
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# RUN STRATEGIES PER TICKER
# -------------------------------------------------------------------------
# (blijft identiek — ingekort)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# RAPPORTAGE
# -------------------------------------------------------------------------
def rapport(label, naam, nu, res, trades, sig):
    def fmt(n): return f"€{100000 + n:,.0f}"
    def block(lst): return "\n".join(lst) if lst else "Geen signalen"

    lijnen = [
        f"📊 {label} {naam} — GLOBAL v2.0",
        f"{nu}",
        "----------------------------------",
        f"🐢 Traag (50/200): {fmt(res['T'])} ({trades['T']} trades)",
        f"⚡ Snel (20/50): {fmt(res['S'])} ({trades['S']} trades)",
        f"🚀 Hyper Trend: {fmt(res['HT'])} ({trades['HT']} trades)",
        f"🔥 Hyper Scalp: {fmt(res['HS'])} ({trades['HS']} trades)",
        f"🛡️ MRA Snel: {fmt(res['MRAS'])} ({trades['MRAS']} trades)",
        f"🐢 MRA Traag: {fmt(res['MRAT'])} ({trades['MRAT']} trades)",
        "",
        "🛡️ SIGNALEN MRA Snel:",
        block(sig["MRAS"]),
        "",
        "🐢 SIGNALEN MRA Traag:",
        block(sig["MRAT"]),
        "",
        "🐢 SIGNALEN Traag:",
        block(sig["T"]),
        "",
        "⚡ SIGNALEN Snel:",
        block(sig["S"]),
        "",
        "🚀 SIGNALEN Hyper Trend:",
        block(sig["HT"]),
        "",
        "🔥 SIGNALEN Hyper Scalp:",
        block(sig["HS"]),
    ]

    telegram_send("\n".join(lijnen))

# -------------------------------------------------------------------------
# MAIN — **AANGEPAST MET AUTO-FILE-DETECTIE**
# -------------------------------------------------------------------------
def main():
    sectoren = {
        "041": "Benelux",
        "042": "Parijs",
        "043": "Frankfurt",
        "044": "Spanje/Portugal",
        "045": "Londen",
        "046": "Milaan",
        "047": "Toronto",
        "048": "Nasdaq/NYSE",
        "049": "Overig 1",
        "050": "Overig 2",
        "051": "Overig 3",
        "052": "Overig 4",
        "053": "Overig 5",
        "054": "Overig 6",
        "055": "Overig 7",
        "056": "Overig 8",
        "057": "Overig 9",
        "058": "Overig 10",
        "059": "Overig 11",
        "060": "Overig 12",
    }

    # Zoek ALLE tickerbestanden in de volledige repo
    alle_bestanden = []
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.startswith("tickers_") and f.endswith("x.txt"):
                alle_bestanden.append(os.path.join(root, f))

    if not alle_bestanden:
        logger.error("❌ Geen tickerbestanden gevonden in de repository!")
        return

    # Verwerk 041 t/m 060
    for i in range(41, 61):
        nr = f"0{i}"
        naam = sectoren.get(nr, f"Lijst {nr}")

        match = [p for p in alle_bestanden if f"tickers_{nr}x.txt" in p]

        if not match:
            logger.warning(f"⚠ Geen bestand gevonden voor {nr} ({naam})")
            continue

        bestandsnaam = match[0]
        logger.info(f"📁 Bestand gevonden: {bestandsnaam}")

        # Lijst uitvoeren
        with open(bestandsnaam, "r") as f:
            raw = f.read().replace("\n", ",").replace("$", "")
            tickers = sorted(set(t.strip().upper() for t in raw.split(",") if t.strip()))

        if not tickers:
            continue

        nu = datetime.now().strftime("%d/%m/%Y %H:%M")

        res = {k: 0.0 for k in ["T","S","HT","HS","MRAS","MRAT"]}
        trades = {k: 0 for k in res.keys()}
        sig = {k: [] for k in res.keys()}

        for t in tickers:
            df = download_ticker(t)
            if df is None:
                continue

            r, tr, sg = run_strategies(df, t)
            for k in res.keys():
                res[k] += r[k]
                trades[k] += tr[k]
                sig[k].extend(sg[k])

            time.sleep(0.2)

        rapport(nr, naam, nu, res, trades, sig)
        time.sleep(2)

if __name__ == "__main__":
    main()
