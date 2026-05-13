import pandas as pd
import numpy as np
import yfinance as yf
import os
import requests
from datetime import datetime, timedelta

# ==========================================
# 1. TICKER IMPORT & CONFIGURATIE
# ==========================================
# Hier halen we de originele lijsten op. 
# Zorg dat 'ticker_lijsten.py' in je GitHub repo staat.
try:
    from ticker_lijsten import AEX_TICKERS, AMX_TICKERS, NASDAQ_TICKERS, BEL20_TICKERS
    ALL_TICKERS = list(set(AEX_TICKERS + AMX_TICKERS + NASDAQ_TICKERS + BEL20_TICKERS))
    print(f"✅ Succesvol {len(ALL_TICKERS)} tickers ingeladen.")
except ImportError:
    # Mocht het bestand missen, dan stopt de bot niet maar gebruikt deze basis:
    ALL_TICKERS = ["ASML.AS", "ADYEN.AS", "INGA.AS", "AAPL", "MSFT", "NVDA", "TSLA"]
    print("⚠️ ticker_lijsten.py niet gevonden. Gebruikt fallback lijst.")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================================
# 2. TELEGRAM ENGINE
# ==========================================
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram Secrets ontbreken!")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    # Telegram heeft een limiet van 4096 tekens. We splitsen indien nodig.
    if len(message) > 4000:
        chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for chunk in chunks:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"})
    else:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})

# ==========================================
# 3. WISKUNDIGE INDICATOREN (De "1100-regels" Logica)
# ==========================================
def _wilder_smooth(series, period):
    """De kern-smoothing voor RSI en ADX."""
    alpha = 1 / period
    res = series.copy()
    res.iloc[period-1] = series.iloc[:period].mean()
    for i in range(period, len(series)):
        res.iloc[i] = res.iloc[i-1] * (1 - alpha) + series.iloc[i] * alpha
    return res

def calculate_technical_analysis(df):
    """Berekent alle complexe indicatoren uit je originele bot."""
    def _process(g):
        g = g.sort_values("Date")
        close, high, low = g["Close"], g["High"], g["Low"]
        
        # --- Moving Averages ---
        g["MA200"] = close.rolling(200).mean()
        g["MA50"] = close.rolling(50).mean()
        g["MA20"] = close.rolling(20) .mean()
        
        # --- RSI (Relative Strength Index) ---
        delta = close.diff()
        gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
        rs = _wilder_smooth(gain, 14) / (_wilder_smooth(loss, 14) + 1e-9)
        g["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
        
        # --- ATR (Average True Range) ---
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        g["ATR14"] = _wilder_smooth(tr, 14)
        
        # --- ADX (Average Directional Index) ---
        up, down = high.diff(), -low.diff()
        pdm = np.where((up > down) & (up > 0), up, 0.0)
        mdm = np.where((down > up) & (down > 0), down, 0.0)
        pdi = 100 * (_wilder_smooth(pd.Series(pdm, index=g.index), 14) / (g["ATR14"] + 1e-9))
        mdi = 100 * (_wilder_smooth(pd.Series(mdm, index=g.index), 14) / (g["ATR14"] + 1e-9))
        dx = (abs(pdi - mdi) / (pdi + mdi + 1e-9)) * 100
        g["ADX14"] = _wilder_smooth(pd.Series(dx, index=g.index), 14)
        
        # --- IBS (Internal Bar Strength) ---
        g["IBS"] = (close - low) / (high - low + 1e-9)
        
        return g

    # De 'include_groups=True' fix voor de Ticker-kolom
    return df.groupby("Ticker", group_keys=False).apply(_process, include_groups=True)

# ==========================================
# 4. MAIN ENGINE
# ==========================================
def run_trading_bot():
    print(f"🚀 Start scan van {len(ALL_TICKERS)} instrumenten...")
    
    # 4.1 Data ophalen
    collected_dfs = []
    for ticker in ALL_TICKERS:
        try:
            # We halen 400 dagen op voor een betrouwbare MA200
            temp_df = yf.download(ticker, start=datetime.now()-timedelta(days=400), progress=False).reset_index()
            if temp_df.empty: continue
            
            # Kolom-fix voor yfinance MultiIndex
            if isinstance(temp_df.columns, pd.MultiIndex):
                temp_df.columns = temp_df.columns.get_level_values(0)
            
            temp_df["Ticker"] = ticker
            collected_dfs.append(temp_df)
        except Exception as e:
            print(f"⚠️ Kon {ticker} niet laden: {e}")

    if not collected_dfs:
        print("❌ Geen data kunnen ophalen. Stop.")
        return

    # 4.2 Indicatoren berekenen
    full_df = pd.concat(collected_dfs)
    analyzed_df = calculate_technical_analysis(full_df)
    
    # 4.3 Selecteer laatste waarden
    latest_results = analyzed_df.groupby("Ticker").last().reset_index()
    
    # 4.4 Rapport genereren
    nu = datetime.now().strftime('%d-%m-%Y %H:%M')
    rapport = [f"📊 *Live Trading Scan: {nu}*", ""]
    
    for _, row in latest_results.iterrows():
        t = row["Ticker"]
        p = row["Close"]
        rsi = row["RSI14"]
        ma200 = row["MA200"]
        adx = row["ADX14"]
        ibs = row["IBS"]
        
        # --- DE ORIGINELE SIGNALEER LOGICA ---
        signal = "⚪"
        if rsi < 30 and p > ma200: 
            signal = "🟢 *BUY (Dip)*"
        elif rsi < 20:
            signal = "🔥 *STRONG BUY*"
        elif rsi > 75:
            signal = "🔴 *SELL*"
        elif rsi < 35:
            signal = "🟡 *WATCH*"

        # Formatteer de regel (Ticker, Prijs, RSI, Signaal)
        line = f"`{t:<9}` €{p:>8.2f} | RSI: {rsi:>4.1f} | {signal}"
        rapport.append(line)

    # 4.5 Verzenden naar Telegram
    send_telegram_message("\n".join(rapport))
    print("✅ Scan voltooid. Rapport verstuurd naar Telegram.")

if __name__ == "__main__":
    run_trading_bot()
