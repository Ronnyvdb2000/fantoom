import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import os
import csv
from datetime import datetime

# ===========================
# 📦 DATA ENGINE (v3.1)
# ===========================

def load_data(tickers):
    data = {}
    for t in tickers:
        try:
            df = yf.download(
                t,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False
            )
            if df.empty:
                print(f"[WARN] Geen data voor {t}")
                continue

            df.dropna(inplace=True)
            data[t] = df

            time.sleep(0.25)  # sequentieel, veilig
        except Exception as e:
            print(f"[ERROR] Download mislukt voor {t}: {e}")

    return data

# ===========================
# ⚡ INDICATORS
# ===========================

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def ADX(df, period=14):
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

    df["+DM"] = np.where(df["High"] > df["High"].shift(1),
                         df["High"] - df["High"].shift(1), 0)
    df["-DM"] = np.where(df["Low"] < df["Low"].shift(1),
                         df["Low"].shift(1) - df["Low"], 0)

    df["+DI"] = 100 * (df["+DM"].rolling(period).sum() / df["TR"].rolling(period).sum())
    df["-DI"] = 100 * (df["-DM"].rolling(period).sum() / df["TR"].rolling(period).sum())
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100
    return df["DX"].rolling(period).mean()

def ATR(df, period=14):
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    return tr.rolling(period).mean()

# ===========================
# ⚡ SIGNAL ENGINE (v3.1)
# ===========================

def strategy_slow(df):
    df = df.copy()
    df["EMA50"] = EMA(df["Close"], 50)
    df["EMA200"] = EMA(df["Close"], 200)
    df["ADX"] = ADX(df)
    df["RSI"] = RSI(df["Close"])

    last = df.iloc[-1]

    return (
        last["EMA50"] > last["EMA200"] and
        last["ADX"] > 18 and
        45 < last["RSI"] < 60 and
        df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1]
    )

def strategy_fast(df):
    df = df.copy()
    df["EMA20"] = EMA(df["Close"], 20)
    df["EMA50"] = EMA(df["Close"], 50)
    df["RSI"] = RSI(df["Close"])

    last = df.iloc[-1]

    return (
        last["EMA20"] > last["EMA50"] and
        last["RSI"] > 55 and
        df["Volume"].iloc[-1] > df["Volume"].rolling(10).mean().iloc[-1]
    )

def strategy_hypertrend(df):
    df = df.copy()
    df["EMA10"] = EMA(df["Close"], 10)
    df["EMA20"] = EMA(df["Close"], 20)
    df["EMA50"] = EMA(df["Close"], 50)
    df["ADX"] = ADX(df)
    df["RSI"] = RSI(df["Close"])

    last = df.iloc[-1]

    return (
        last["EMA10"] > last["EMA20"] > last["EMA50"] and
        last["ADX"] > 22 and
        last["RSI"] > 60 and
        df["Close"].iloc[-1] > df["Close"].iloc[-2]
    )

def strategy_scalp(df):
    df = df.copy()
    df["RSI"] = RSI(df["Close"])
    df["MA20"] = df["Close"].rolling(20).mean()

    last = df.iloc[-1]

    return (
        last["RSI"] < 35 and
        last["Close"] > last["MA20"] and
        df["Volume"].iloc[-1] > df["Volume"].rolling(5).mean().iloc[-1]
    )

def generate_buy_signals(data):
    signals = []

    for ticker, df in data.items():
        try:
            if strategy_slow(df):
                signals.append((ticker, "🐢 Traag"))

            if strategy_fast(df):
                signals.append((ticker, "⚡ Snel"))

            if strategy_hypertrend(df):
                signals.append((ticker, "🚀 Hyper Trend"))

            if strategy_scalp(df):
                signals.append((ticker, "🔥 Hyper Scalp"))

        except Exception as e:
            print(f"[ERROR] Strategie fout bij {ticker}: {e}")

    return signals

# ===========================
# 🔥 EXIT ENGINE (v3.1)
# ===========================

def generate_sell_signals(data, positions):
    sells = []

    for ticker, pos in positions.items():
        df = data.get(ticker)
        if df is None or df.empty:
            continue

        df = df.copy()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["RSI"] = RSI(df["Close"])
        df["ATR"] = ATR(df)

        last = df.iloc[-1]
        entry_price = pos.get("entry_price", last["Close"])
        max_price = pos.get("max_price", entry_price)
        days_in_pos = pos.get("days", 0)

        if last["Close"] > max_price:
            max_price = last["Close"]
            pos["max_price"] = max_price

        stop_level = entry_price - 2 * df["ATR"].iloc[-1]
        if last["Close"] < stop_level:
            sells.append((ticker, "Stoploss"))
            continue

        gain_pct = (last["Close"] / entry_price - 1) * 100
        if gain_pct >= 12:
            sells.append((ticker, "Take Profit 12%"))
            continue
        if gain_pct >= 8 and last["RSI"] > 65:
            sells.append((ticker, "Take Profit 8%+RSI"))
            continue

        if last["Close"] < last["MA20"]:
            sells.append((ticker, "MA20 Exit"))
            continue

        if last["RSI"] > 70:
            sells.append((ticker, "RSI > 70"))
            continue

        if days_in_pos > 20:
            sells.append((ticker, "Time Exit > 20d"))
            continue

    return sells

# ===========================
# 💼 PORTFOLIO ENGINE (v3.1)
# ===========================

PORTFOLIO_FILE = "portfolio_live.csv"

def load_portfolio():
    portfolio = {
        "cash": 50000.0,
        "positions": {}
    }

    if not os.path.exists(PORTFOLIO_FILE):
        return portfolio

    with open(PORTFOLIO_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "META":
                portfolio["cash"] = float(row["cash"])
            elif row["type"] == "POS":
                ticker = row["ticker"]
                portfolio["positions"][ticker] = {
                    "entry_price": float(row["entry_price"]),
                    "size": float(row["size"]),
                    "strategy": row["strategy"],
                    "days": int(row["days"]),
                    "max_price": float(row["max_price"])
                }

    return portfolio

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["type", "ticker", "entry_price", "size", "strategy", "days", "max_price", "cash"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            "type": "META",
            "ticker": "",
            "entry_price": "",
            "size": "",
            "strategy": "",
            "days": "",
            "max_price": "",
            "cash": portfolio["cash"]
        })

        for ticker, pos in portfolio["positions"].items():
            writer.writerow({
                "type": "POS",
                "ticker": ticker,
                "entry_price": pos["entry_price"],
                "size": pos["size"],
                "strategy": pos["strategy"],
                "days": pos["days"],
                "max_price": pos.get("max_price", pos["entry_price"]),
                "cash": ""
            })

def update_portfolio(buys, sells, portfolio, data):
    for ticker, reason in sells:
        if ticker in portfolio["positions"]:
            pos = portfolio["positions"][ticker]
            last_price = data[ticker]["Close"].iloc[-1]
            value = pos["size"] * last_price
            portfolio["cash"] += value
            del portfolio["positions"][ticker]

    max_positions = 10
    current_positions = len(portfolio["positions"])

    for ticker, strategy in buys:
        if current_positions >= max_positions:
            break

        if ticker in portfolio["positions"]:
            continue

        base_size = 2500
        if portfolio["cash"] > 60000:
            base_size = 3000

        if portfolio["cash"] < base_size:
            continue

        price = data[ticker]["Close"].iloc[-1]
        size = base_size / price

        portfolio["cash"] -= base_size
        portfolio["positions"][ticker] = {
            "entry_price": price,
            "size": size,
            "strategy": strategy,
            "days": 0,
            "max_price": price
        }
        current_positions += 1

    for pos in portfolio["positions"].values():
        pos["days"] += 1

    return portfolio

# ===========================
# 📲 TELEGRAM ENGINE
# ===========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Geen TELEGRAM_TOKEN of TELEGRAM_CHAT_ID ingesteld.")
        print(text)
        return

    try:
        requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            params={"chat_id": TELEGRAM_CHAT_ID, "text": text}
        )
    except Exception as e:
        print(f"[ERROR] Telegram fout: {e}")
        print(text)

def build_report(buys, sells, portfolio):
    lines = []
    lines.append("📊 GLOBAL ENGINE v3.1")
    lines.append(datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("")

    if buys:
        lines.append("🟢 BUY SIGNALEN:")
        for t, s in buys:
            lines.append(f"- {t} ({s})")
        lines.append("")
    else:
        lines.append("🟢 Geen nieuwe BUY signalen.")
        lines.append("")

    if sells:
        lines.append("🔴 SELL SIGNALEN:")
        for t, r in sells:
            lines.append(f"- {t} ({r})")
        lines.append("")
    else:
        lines.append("🔴 Geen SELL signalen.")
        lines.append("")

    lines.append("📂 PORTFOLIO:")
    lines.append(f"Cash: {portfolio['cash']:.2f}")
    if portfolio["positions"]:
        for t, pos in portfolio["positions"].items():
            lines.append(
                f"- {t}: {pos['size']:.2f} @ {pos['entry_price']:.2f} "
                f"({pos['strategy']}, {pos['days']}d)"
            )
    else:
        lines.append("- Geen open posities.")

    return "\n".join(lines)

# ===========================
# 🧠 MAIN WORKFLOW
# ===========================

def load_ticker_list():
    # Voor nu hardcoded; later kun je dit uit een bestand halen
    return ["AAPL", "MSFT", "NVDA", "META", "GOOGL"]

def main():
    print("🚀 GLOBAL ENGINE v3.1 start...")

    tickers = load_ticker_list()
    if not tickers:
        print("Geen tickers gevonden.")
        return

    data = load_data(tickers)
    if not data:
        print("Geen marktdata beschikbaar.")
        return

    portfolio = load_portfolio()

    buys = generate_buy_signals(data)
    sells = generate_sell_signals(data, portfolio["positions"])

    portfolio = update_portfolio(buys, sells, portfolio, data)
    save_portfolio(portfolio)

    report = build_report(buys, sells, portfolio)
    send_telegram_message(report)

    print("Klaar.")

if __name__ == "__main__":
    main()
