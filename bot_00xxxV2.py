#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLOBAL TRADING ENGINE v1.0
- 1 script voor ALLE markten
- Automatische markt-detectie (EU / US / CA)
- Meerdere strategieën per markt
- yfinance multi-index fix
- Telegram rapportage
- Backtest per strategie
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

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# TELEGRAM
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
INZET = 2500.0
KOSTEN = 15.0 + (INZET * 0.0035)

# -------------------------
# EU STRATEGIEËN
# -------------------------
# ET (EU Trend)
ET_ADX_MIN = 12
ET_RSI_MIN = 45
ET_RSI_MAX = 65
ET_VOL_MIN = 0.2
ET_SL_ATR = 2.5
ET_TP_ATR = 2.0

# EMR (EU Mean Reversion)
EMR_BB_STD = 2.5
EMR_IBS_MAX = 0.40
EMR_RSI_MAX = 35
EMR_SL_ATR = 3.0
EMR_TP_PCT = 0.06
EMR_MA_EXIT = 10

# EB (EU Breakout)
EB_LOOKBACK = 50
EB_ADX_MIN = 15
EB_RSI_MAX = 70
EB_VOL_SPIKE = 1.5
EB_SL_ATR = 2.0
EB_TP_ATR = 3.0

# -------------------------
# US STRATEGIEËN
# -------------------------
# UST (US Trend Momentum)
UST_ADX_MIN = 18
UST_RSI_MIN = 50
UST_RSI_MAX = 70
UST_SL_ATR = 2.0
UST_TP_ATR = 3.0

# USMR (US Mean Reversion)
USMR_IBS_MAX = 0.35
USMR_RSI_MAX = 30
USMR_SL_ATR = 2.5
USMR_TP_ATR = 2.0

# USB (US Breakout)
USB_LOOKBACK = 20
USB_VOL_SPIKE = 1.8
USB_ADX_MIN = 20
USB_SL_ATR = 2.0
USB_TP_ATR = 4.0

# -------------------------
# CA STRATEGIEËN
# -------------------------
# CAT (CA Trend)
CAT_ADX_MIN = 14
CAT_SL_ATR = 3.0
CAT_TP_ATR = 2.5

# CAMR (CA Mean Reversion)
CAMR_IBS_MAX = 0.45
CAMR_RSI_MAX = 40
CAMR_SL_ATR = 3.5
CAMR_TP_ATR = 2.0

# -----------------------------------------------------------------------------
# MARKT-DETECTIE
# -----------------------------------------------------------------------------
def detect_market(ticker: str):
    t = ticker.upper()

    # Canada
    if t.endswith(".TO") or t.endswith(".V"):
        return "CA"

    # Europa
    EU_SUFFIXES = [
        ".AS", ".BR", ".PA", ".DE", ".MI", ".L", ".MC", ".BE",
        ".SW", ".HE", ".OL", ".ST", ".CO"
    ]
    if any(t.endswith(s) for s in EU_SUFFIXES):
        return "EU"

    # US (default)
    return "US"

# -----------------------------------------------------------------------------
# DOWNLOAD ENGINE + MULTI-INDEX FIX
# -----------------------------------------------------------------------------
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

        # Multi-index flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Forceer kolommen naar Series
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

# -----------------------------------------------------------------------------
# INDICATOREN
# -----------------------------------------------------------------------------
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
    upper_bb = ma20 + EMR_BB_STD * std20
    lower_bb = ma20 - EMR_BB_STD * std20

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
        "upper_bb": upper_bb, "lower_bb": lower_bb,
        "ibs": ibs, "rsi": rsi, "atr": atr, "adx": adx
    }

# -----------------------------------------------------------------------------
# STRATEGIEËN — EU
# -----------------------------------------------------------------------------
def backtest_ET(ind):
    """EU Trend: pullback naar EMA20 binnen uptrend."""
    p = ind["p"]
    ema20 = ind["ema20"]
    ema50 = ind["ema50"]
    ema200 = ind["ema200"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    adx = ind["adx"]
    vol = ind["v"]
    vol_ma20 = ind["vol_ma20"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    for i in range(50, len(p)):
        cp = p.iloc[i]

        if not pos:
            if (
                ema50.iloc[i] > ema200.iloc[i] and
                ET_RSI_MIN <= rsi.iloc[i] <= ET_RSI_MAX and
                adx.iloc[i] > ET_ADX_MIN and
                vol.iloc[i] > vol_ma20.iloc[i] * ET_VOL_MIN and
                cp <= ema20.iloc[i] * 1.01 and
                cp >= ema20.iloc[i] * 0.97
            ):
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            sl = ins - ET_SL_ATR * atr.iloc[i]
            tp = ins + ET_TP_ATR * atr.iloc[i]

            if cp <= sl or cp >= tp:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = p.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades


def backtest_EMR(ind):
    """EU Mean Reversion: BB + IBS + RSI."""
    p = ind["p"]
    lower_bb = ind["lower_bb"]
    ibs = ind["ibs"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    ma10 = ind["ma10"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    pb = p.iloc[-252:]
    lbb = lower_bb.iloc[-252:]
    ibs_s = ibs.iloc[-252:]
    rsi_s = rsi.iloc[-252:]
    atr_s = atr.iloc[-252:]
    ma10_s = ma10.iloc[-252:]

    for i in range(20, len(pb)):
        cp = pb.iloc[i]

        if not pos:
            if (
                cp < lbb.iloc[i] and
                ibs_s.iloc[i] < EMR_IBS_MAX and
                rsi_s.iloc[i] < EMR_RSI_MAX
            ):
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            tp = ins * (1 + EMR_TP_PCT)
            sl = ins - EMR_SL_ATR * atr_s.iloc[i]

            if cp >= tp or cp <= sl or cp >= ma10_s.iloc[i]:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = pb.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades


def backtest_EB(ind):
    """EU Breakout: high + volume spike + ADX."""
    p = ind["p"]
    h = ind["h"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    adx = ind["adx"]
    vol = ind["v"]
    vol_ma20 = ind["vol_ma20"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    for i in range(EB_LOOKBACK, len(p)):
        cp = p.iloc[i]
        hh = h.iloc[i-EB_LOOKBACK:i].max()

        if not pos:
            if (
                cp > hh and
                vol.iloc[i] > vol_ma20.iloc[i] * EB_VOL_SPIKE and
                adx.iloc[i] > EB_ADX_MIN and
                rsi.iloc[i] < EB_RSI_MAX
            ):
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            sl = ins - EB_SL_ATR * atr.iloc[i]
            tp = ins + EB_TP_ATR * atr.iloc[i]

            if cp <= sl or cp >= tp:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = p.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades

# -----------------------------------------------------------------------------
# STRATEGIEËN — US
# -----------------------------------------------------------------------------
def backtest_UST(ind):
    """US Trend Momentum: sterkere trend, meer momentum."""
    p = ind["p"]
    ema20 = ind["ema20"]
    ema50 = ind["ema50"]
    ema200 = ind["ema200"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    adx = ind["adx"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    for i in range(60, len(p)):
        cp = p.iloc[i]

        if not pos:
            if (
                ema50.iloc[i] > ema200.iloc[i] and
                ema20.iloc[i] > ema50.iloc[i] and
                UST_RSI_MIN <= rsi.iloc[i] <= UST_RSI_MAX and
                adx.iloc[i] > UST_ADX_MIN
            ):
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            sl = ins - UST_SL_ATR * atr.iloc[i]
            tp = ins + UST_TP_ATR * atr.iloc[i]

            if cp <= sl or cp >= tp:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = p.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades


def backtest_USMR(ind):
    """US Mean Reversion: agressiever, kortere swings."""
    p = ind["p"]
    ibs = ind["ibs"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    ma10 = ind["ma10"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    pb = p.iloc[-252:]
    ibs_s = ibs.iloc[-252:]
    rsi_s = rsi.iloc[-252:]
    atr_s = atr.iloc[-252:]
    ma10_s = ma10.iloc[-252:]

    for i in range(20, len(pb)):
        cp = pb.iloc[i]

        if not pos:
            if ibs_s.iloc[i] < USMR_IBS_MAX and rsi_s.iloc[i] < USMR_RSI_MAX:
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            sl = ins - USMR_SL_ATR * atr_s.iloc[i]
            tp = ins + USMR_TP_ATR * atr_s.iloc[i]

            if cp <= sl or cp >= tp or cp >= ma10_s.iloc[i]:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = pb.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades


def backtest_USB(ind):
    """US Breakout: kortere lookback, sterk volume."""
    p = ind["p"]
    h = ind["h"]
    atr = ind["atr"]
    adx = ind["adx"]
    rsi = ind["rsi"]
    vol = ind["v"]
    vol_ma20 = ind["vol_ma20"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    for i in range(USB_LOOKBACK, len(p)):
        cp = p.iloc[i]
        hh = h.iloc[i-USB_LOOKBACK:i].max()

        if not pos:
            if (
                cp > hh and
                vol.iloc[i] > vol_ma20.iloc[i] * USB_VOL_SPIKE and
                adx.iloc[i] > USB_ADX_MIN and
                rsi.iloc[i] < 75
            ):
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            sl = ins - USB_SL_ATR * atr.iloc[i]
            tp = ins + USB_TP_ATR * atr.iloc[i]

            if cp <= sl or cp >= tp:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = p.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades

# -----------------------------------------------------------------------------
# STRATEGIEËN — CA
# -----------------------------------------------------------------------------
def backtest_CAT(ind):
    """CA Trend: grondstoffen, hogere ATR, bredere stops."""
    p = ind["p"]
    ema50 = ind["ema50"]
    ema200 = ind["ema200"]
    atr = ind["atr"]
    adx = ind["adx"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    for i in range(60, len(p)):
        cp = p.iloc[i]

        if not pos:
            if ema50.iloc[i] > ema200.iloc[i] and adx.iloc[i] > CAT_ADX_MIN:
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            sl = ins - CAT_SL_ATR * atr.iloc[i]
            tp = ins + CAT_TP_ATR * atr.iloc[i]

            if cp <= sl or cp >= tp:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = p.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades


def backtest_CAMR(ind):
    """CA Mean Reversion: iets ruimer dan EU, hogere ATR."""
    p = ind["p"]
    ibs = ind["ibs"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    ma10 = ind["ma10"]

    pr = 0.0
    trades = 0
    pos = False
    ins = 0.0

    pb = p.iloc[-252:]
    ibs_s = ibs.iloc[-252:]
    rsi_s = rsi.iloc[-252:]
    atr_s = atr.iloc[-252:]
    ma10_s = ma10.iloc[-252:]

    for i in range(20, len(pb)):
        cp = pb.iloc[i]

        if not pos:
            if ibs_s.iloc[i] < CAMR_IBS_MAX and rsi_s.iloc[i] < CAMR_RSI_MAX:
                pos = True
                ins = cp
                pr -= KOSTEN
                trades += 1
        else:
            sl = ins - CAMR_SL_ATR * atr_s.iloc[i]
            tp = ins + CAMR_TP_ATR * atr_s.iloc[i]

            if cp <= sl or cp >= tp or cp >= ma10_s.iloc[i]:
                pr += (INZET * (cp / ins) - INZET) - KOSTEN
                pos = False

    if pos:
        cp = pb.iloc[-1]
        pr += (INZET * (cp / ins) - INZET) - KOSTEN

    return pr, trades

# -----------------------------------------------------------------------------
# RUN STRATEGIES PER TICKER
# -----------------------------------------------------------------------------
def run_strategies(df: pd.DataFrame, ticker: str):
    ind = compute_indicators(df)
    market = detect_market(ticker)

    res = {
        "ET": 0.0, "EMR": 0.0, "EB": 0.0,
        "UST": 0.0, "USMR": 0.0, "USB": 0.0,
        "CAT": 0.0, "CAMR": 0.0
    }
    trades = {k: 0 for k in res.keys()}
    sig = {k: [] for k in res.keys()}

    # Backtests per markt-set
    if market == "EU":
        et_pnl, et_tr = backtest_ET(ind)
        emr_pnl, emr_tr = backtest_EMR(ind)
        eb_pnl, eb_tr = backtest_EB(ind)
        res["ET"] += et_pnl; trades["ET"] += et_tr
        res["EMR"] += emr_pnl; trades["EMR"] += emr_tr
        res["EB"] += eb_pnl; trades["EB"] += eb_tr

    elif market == "US":
        ust_pnl, ust_tr = backtest_UST(ind)
        usmr_pnl, usmr_tr = backtest_USMR(ind)
        usb_pnl, usb_tr = backtest_USB(ind)
        res["UST"] += ust_pnl; trades["UST"] += ust_tr
        res["USMR"] += usmr_pnl; trades["USMR"] += usmr_tr
        res["USB"] += usb_pnl; trades["USB"] += usb_tr

    elif market == "CA":
        cat_pnl, cat_tr = backtest_CAT(ind)
        camr_pnl, camr_tr = backtest_CAMR(ind)
        res["CAT"] += cat_pnl; trades["CAT"] += cat_tr
        res["CAMR"] += camr_pnl; trades["CAMR"] += camr_tr

    # Signalen (laatste bar)
    p = ind["p"]
    ema20 = ind["ema20"]
    ema50 = ind["ema50"]
    ema200 = ind["ema200"]
    rsi = ind["rsi"]
    atr = ind["atr"]
    adx = ind["adx"]
    vol = ind["v"]
    vol_ma20 = ind["vol_ma20"]
    lower_bb = ind["lower_bb"]
    ibs = ind["ibs"]
    h = ind["h"]

    # EU-signalen
    if market == "EU":
        if (
            ema50.iloc[-1] > ema200.iloc[-1] and
            abs(p.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1] < 0.02 and
            adx.iloc[-1] > ET_ADX_MIN
        ):
            sig["ET"].append(
                f"• `{ticker}`: 🇪🇺 🟢 ET pullback | €{p.iloc[-1]:.2f} | ATR {atr.iloc[-1]:.2f}"
            )

        if (
            p.iloc[-1] < lower_bb.iloc[-1] and
            ibs.iloc[-1] < EMR_IBS_MAX and
            rsi.iloc[-1] < EMR_RSI_MAX
        ):
            sig["EMR"].append(
                f"• `{ticker}`: 🇪🇺 🔵 EMR mean reversion | €{p.iloc[-1]:.2f}"
            )

        hh50 = h.iloc[-EB_LOOKBACK:].max()
        if (
            p.iloc[-1] > hh50 and
            vol.iloc[-1] > vol_ma20.iloc[-1] * EB_VOL_SPIKE and
            adx.iloc[-1] > EB_ADX_MIN
        ):
            sig["EB"].append(
                f"• `{ticker}`: 🇪🇺 🔴 EB breakout | €{p.iloc[-1]:.2f}"
            )

    # US-signalen
    if market == "US":
        if (
            ema50.iloc[-1] > ema200.iloc[-1] and
            ema20.iloc[-1] > ema50.iloc[-1] and
            UST_RSI_MIN <= rsi.iloc[-1] <= UST_RSI_MAX and
            adx.iloc[-1] > UST_ADX_MIN
        ):
            sig["UST"].append(
                f"• `{ticker}`: 🇺🇸 🟢 UST trend | €{p.iloc[-1]:.2f}"
            )

        if ibs.iloc[-1] < USMR_IBS_MAX and rsi.iloc[-1] < USMR_RSI_MAX:
            sig["USMR"].append(
                f"• `{ticker}`: 🇺🇸 🔵 USMR mean reversion | €{p.iloc[-1]:.2f}"
            )

        hh20 = h.iloc[-USB_LOOKBACK:].max()
        if (
            p.iloc[-1] > hh20 and
            vol.iloc[-1] > vol_ma20.iloc[-1] * USB_VOL_SPIKE and
            adx.iloc[-1] > USB_ADX_MIN
        ):
            sig["USB"].append(
                f"• `{ticker}`: 🇺🇸 🔴 USB breakout | €{p.iloc[-1]:.2f}"
            )

    # CA-signalen
    if market == "CA":
        if ema50.iloc[-1] > ema200.iloc[-1] and adx.iloc[-1] > CAT_ADX_MIN:
            sig["CAT"].append(
                f"• `{ticker}`: 🇨🇦 🟢 CAT trend | €{p.iloc[-1]:.2f}"
            )

        if ibs.iloc[-1] < CAMR_IBS_MAX and rsi.iloc[-1] < CAMR_RSI_MAX:
            sig["CAMR"].append(
                f"• `{ticker}`: 🇨🇦 🔵 CAMR mean reversion | €{p.iloc[-1]:.2f}"
            )

    return res, trades, sig

# -----------------------------------------------------------------------------
# RAPPORTAGE
# -----------------------------------------------------------------------------
def rapport(label, naam, nu, res, trades, sig):
    def fmt(n): return f"€{100000 + n:,.0f}"
    def block(lst): return "\n".join(lst) if lst else "Geen signalen"

    lijnen = [
        f"📊 *{label} {naam} — GLOBAL ENGINE*",
        f"_{nu}_",
        "----------------------------------",
        "🇪🇺 *EU STRATEGIEËN*",
        f"🟢 ET (Trend): {fmt(res['ET'])} ({trades['ET']} trades)",
        f"🔵 EMR (Mean Reversion): {fmt(res['EMR'])} ({trades['EMR']} trades)",
        f"🔴 EB (Breakout): {fmt(res['EB'])} ({trades['EB']} trades)",
        "",
        "🇺🇸 *US STRATEGIEËN*",
        f"🟢 UST (Trend): {fmt(res['UST'])} ({trades['UST']} trades)",
        f"🔵 USMR (Mean Reversion): {fmt(res['USMR'])} ({trades['USMR']} trades)",
        f"🔴 USB (Breakout): {fmt(res['USB'])} ({trades['USB']} trades)",
        "",
        "🇨🇦 *CA STRATEGIEËN*",
        f"🟢 CAT (Trend): {fmt(res['CAT'])} ({trades['CAT']} trades)",
        f"🔵 CAMR (Mean Reversion): {fmt(res['CAMR'])} ({trades['CAMR']} trades)",
        "",
        "📈 SIGNALEN EU (ET):",
        block(sig["ET"]),
        "",
        "📈 SIGNALEN EU (EMR):",
        block(sig["EMR"]),
        "",
        "📈 SIGNALEN EU (EB):",
        block(sig["EB"]),
        "",
        "📈 SIGNALEN US (UST):",
        block(sig["UST"]),
        "",
        "📈 SIGNALEN US (USMR):",
        block(sig["USMR"]),
        "",
        "📈 SIGNALEN US (USB):",
        block(sig["USB"]),
        "",
        "📈 SIGNALEN CA (CAT):",
        block(sig["CAT"]),
        "",
        "📈 SIGNALEN CA (CAMR):",
        block(sig["CAMR"]),
        "",
        "⚙️ PARAMETERS (kort):",
        "_EU: ET/EMR/EB — trend, mean reversion, breakout met EU-volatiliteit_",
        "_US: UST/USMR/USB — sterkere trends, kortere breakouts_",
        "_CA: CAT/CAMR — hogere ATR, bredere stops_",
    ]

    telegram_send("\n".join(lijnen))

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
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

    for i in range(41, 61):
        nr = f"0{i}"
        fname = f"tickers_{nr}x.txt"
        naam = sectoren.get(nr, f"Lijst {nr}")

        if not os.path.exists(fname):
            continue

        with open(fname, "r") as f:
            raw = f.read().replace("\n", ",").replace("$", "")
            tickers = sorted(set(t.strip().upper() for t in raw.split(",") if t.strip()))

        if not tickers:
            continue

        logger.info(f"Start analyse {nr} ({naam}) met {len(tickers)} tickers.")
        nu = datetime.now().strftime("%d/%m/%Y %H:%M")

        res = {
            "ET": 0.0, "EMR": 0.0, "EB": 0.0,
            "UST": 0.0, "USMR": 0.0, "USB": 0.0,
            "CAT": 0.0, "CAMR": 0.0
        }
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
