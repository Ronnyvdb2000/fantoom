#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH v2.1 voor bot_00xxxV2.py
Fixes:
  1. TypeError: 'NoneType' object is not subscriptable  (MU, OVV.TO e.a.)
  2. FutureWarning: DataFrameGroupBy.apply grouping columns
  3. Batch download MultiIndex detectie robuuster gemaakt

Vervang in bot_00xxxV2.py de functies:
  - download_history()   → zie hieronder
  - add_indicators()     → zie hieronder

De rest van de bot blijft ongewijzigd.
"""

# ──────────────────────────────────────────────────────────────────
# PATCH 1 + 3:  download_history()
# Wijzigingen:
#   - _normalise() vangt nu NoneType/lege DataFrame op vóór .dropna()
#   - Extra check: df_raw is None guard
#   - Batch MultiIndex level-check robuuster
#   - 1-voor-1 fallback: extra try/except rond xs() vervangen
# ──────────────────────────────────────────────────────────────────

def download_history(
    tickers,
    start=None,
    end=None,
    period="5y",
):
    import time
    import yfinance as yf
    import pandas as pd

    if not tickers:
        return pd.DataFrame()

    kwargs = dict(
        tickers=tickers,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if start and end:
        kwargs["start"] = start
        kwargs["end"]   = end
    else:
        kwargs["period"] = period

    frames = []

    def _normalise(df_raw, ticker):
        """
        FIX: vangt None en lege DataFrame op VOOR elke bewerking.
        Oorzaak TypeError was: df_raw = None → None.dropna() crash.
        """
        # ── Guard: None of geen DataFrame
        if df_raw is None:
            return None
        if not isinstance(df_raw, pd.DataFrame):
            return None
        if df_raw.empty:
            return None

        df = df_raw.copy()

        # Verwijder rijen waar alles NaN is
        df = df.dropna(how="all")
        if df.empty:
            return None

        # Date uit index
        if df.index.name in ("Date", "Datetime") or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # Dubbele kolommen verwijderen
        if "Date" in df.columns:
            df = df.loc[:, ~df.columns.duplicated()]
        if "Datetime" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        # MultiIndex platslaan
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Minimale check: Close kolom aanwezig?
        if "Close" not in df.columns:
            return None

        df["Ticker"] = ticker
        return df

    # ── Batch download
    try:
        data = yf.download(**kwargs)
    except Exception as e:
        print(f"[WARN] Batch download mislukt ({e}), probeer 1-voor-1...")
        data = pd.DataFrame()

    if data is not None and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            # Robuuste level-detectie: zoek level met ticker-namen
            levels_with_tickers = []
            for lvl in range(data.columns.nlevels):
                vals = set(data.columns.get_level_values(lvl))
                if any(t in vals for t in tickers):
                    levels_with_tickers.append(lvl)

            ticker_level = levels_with_tickers[-1] if levels_with_tickers else 1
            available    = set(data.columns.get_level_values(ticker_level))

            for t in tickers:
                if t not in available:
                    print(f"[WARN] {t}: geen data in batch (mogelijk delisted), overgeslagen.")
                    continue
                try:
                    raw  = data.xs(t, axis=1, level=ticker_level).copy()
                    norm = _normalise(raw, t)
                    if norm is not None:
                        frames.append(norm)
                    else:
                        print(f"[WARN] {t}: lege data na normalisatie, overgeslagen.")
                except Exception as e:
                    print(f"[WARN] {t}: fout bij verwerken ({e}), overgeslagen.")
        else:
            # Batch met 1 ticker → platte DataFrame
            norm = _normalise(data, tickers[0])
            if norm is not None:
                frames.append(norm)

    # ── Fallback 1-voor-1 (alleen als batch volledig leeg was)
    if not frames:
        print(f"[INFO] Probeer {len(tickers)} tickers 1-voor-1...")
        for t in tickers:
            try:
                kw = dict(tickers=t, auto_adjust=True, progress=False)
                if start and end:
                    kw["start"] = start
                    kw["end"]   = end
                else:
                    kw["period"] = period

                raw = yf.download(**kw)

                # FIX: raw kan None zijn of een lege DataFrame
                if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                    print(f"[WARN] {t}: geen data, overgeslagen.")
                    continue

                # Enkelticker MultiIndex platslaan
                if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                norm = _normalise(raw, t)
                if norm is not None:
                    frames.append(norm)
                else:
                    print(f"[WARN] {t}: geen bruikbare data, overgeslagen.")

                time.sleep(0.2)

            except TypeError as e:
                # FIX: vangt 'NoneType' object is not subscriptable
                print(f"[WARN] {t}: TypeError ondervangen ({e}), overgeslagen.")
            except Exception as e:
                print(f"[WARN] {t}: download mislukt ({e}), overgeslagen.")

    if not frames:
        return pd.DataFrame()

    import pandas as pd as pd2  # noqa – al geïmporteerd, hergebruik
    df = pd.concat(frames, ignore_index=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    df.sort_values(["Ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_ok    = df["Ticker"].nunique()
    n_total = len(tickers)
    if n_ok < n_total:
        print(f"[WARN] Data beschikbaar voor {n_ok}/{n_total} tickers. Mogelijk delisted tickers overgeslagen.")

    return df


# ──────────────────────────────────────────────────────────────────
# PATCH 2:  add_indicators()
# Wijziging: include_groups=False toegevoegd aan groupby.apply()
# Hiermee verdwijnt de FutureWarning van pandas 2.x
# ──────────────────────────────────────────────────────────────────

def add_indicators(df):
    import pandas as pd
    import numpy as np

    def _wilder_smooth(series, period):
        result    = pd.Series(index=series.index, dtype=float)
        valid     = series.dropna()
        if len(valid) < period:
            return result
        first_idx         = valid.index[period - 1]
        result[first_idx] = valid.iloc[:period].mean()
        for i in range(period, len(valid)):
            idx           = valid.index[i]
            prev_idx      = valid.index[i - 1]
            result[idx]   = result[prev_idx] * (period - 1) / period + valid.iloc[i] / period
        return result

    def _calc(group):
        # FIX FutureWarning: drop Ticker vóór berekeningen, voeg achteraf terug
        ticker_val = group["Ticker"].iloc[0] if "Ticker" in group.columns else None
        g = group.drop(columns=["Ticker"], errors="ignore").copy().reset_index(drop=True)

        close = g["Close"]
        high  = g["High"]
        low   = g["Low"]

        g["MA20"]  = close.rolling(20).mean()
        g["MA50"]  = close.rolling(50).mean()
        g["MA200"] = close.rolling(200).mean()

        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = _wilder_smooth(gain, 14)
        avg_loss = _wilder_smooth(loss, 14)
        rs       = avg_gain / (avg_loss + 1e-9)
        g["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

        hl  = high - low
        hcp = (high - close.shift()).abs()
        lcp = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        g["ATR14"] = _wilder_smooth(tr, 14)

        g["IBS"] = (close - low) / (high - low + 1e-9)

        up_move   = high.diff()
        down_move = (-low.diff())
        plus_dm   = np.where((up_move  > down_move) & (up_move  > 0), up_move,   0.0)
        minus_dm  = np.where((down_move > up_move)  & (down_move > 0), down_move, 0.0)

        s_plus_dm  = _wilder_smooth(pd.Series(plus_dm,  index=g.index), 14)
        s_minus_dm = _wilder_smooth(pd.Series(minus_dm, index=g.index), 14)
        s_tr       = _wilder_smooth(tr, 14)

        plus_di  = 100 * s_plus_dm  / (s_tr + 1e-9)
        minus_di = 100 * s_minus_dm / (s_tr + 1e-9)
        dx       = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        g["ADX14"] = _wilder_smooth(dx, 14)

        if ticker_val is not None:
            g["Ticker"] = ticker_val
        return g

    # FIX FutureWarning: include_groups=False
    return df.groupby("Ticker", group_keys=False).apply(
        _calc, include_groups=False
    )


# ──────────────────────────────────────────────────────────────────
# HOE TOEPASSEN IN BOT_00XXXV2.PY:
#
# Optie A — directe vervanging (aanbevolen):
#   Vervang de bestaande download_history() en add_indicators()
#   functies in bot_00xxxV2.py volledig door bovenstaande code.
#
# Optie B — import patch (snel testen):
#   Voeg onderaan bot_00xxxV2.py toe:
#
#     from patch_v21 import download_history, add_indicators
#
#   Python gebruikt dan de gepatchte versies.
#
# VOOR GITHUB ACTIONS:
#   Zorg dat patch_v21.py in dezelfde map staat als bot_00xxxV2.py
#   en voeg de import toe aan het einde van bot_00xxxV2.py.
# ──────────────────────────────────────────────────────────────────
