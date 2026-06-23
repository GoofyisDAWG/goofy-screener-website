"""
app.py — Goofy Screener  Public Website  v1
Stage 1 MVP: Rankings · Track Record · About

Run locally:   streamlit run app.py --server.port 8503
Deploy:        push to GitHub → share.streamlit.io
"""

import json, re, math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
SCREENER_DIR = BASE / "screener_output"
TRADES_DIR   = BASE / "paper_trades"

# ── constants ─────────────────────────────────────────────────────────────────
MARKETS = {"US": "🇺🇸 United States", "ASX": "🇦🇺 Australia", "JPX": "🇯🇵 Japan"}

VERDICT_ORDER = {"TRADE": 0, "ML HOLD": 1, "STAND DOWN": 2, "STAND DOWN (VOL SPIKE)": 3}

TIER_LABELS = {
    "S": ("S — Elite", "#e879f9"),
    "A": ("A — Strong", "#58a6ff"),
    "B": ("B — Good",   "#3fb950"),
    "C": ("C — Weak",   "#8b949e"),
}

STRATEGY_PLAIN = {
    "MA Crossover":      "Trend following — buy when short-term average crosses above long-term",
    "RSI":               "Mean reversion — buy when stock is oversold (RSI < 30)",
    "RSI Divergence":    "Momentum — buy when price drops but RSI stays high",
    "MACD":              "Trend momentum — buy when fast MA crosses above slow MA",
    "Bollinger Bands":   "Mean reversion — buy when price touches the lower band",
    "Mean Reversion":    "Statistical — buy when price is unusually far below its average",
    "Stochastic":        "Momentum — buy when Stochastic oscillator signals oversold",
    "Donchian":          "Breakout — buy when price breaks above its recent high range",
    "ADX":               "Trend strength — only trades when trend is strong and clear",
    "Supertrend":        "Trend following — uses volatility to define trend direction",
    "Ichimoku":          "Multi-signal — Japanese system combining price, time and momentum",
    "Volume Breakout":   "Volume-confirmed breakout — price and volume surge together",
    "Relative Strength": "Relative momentum — stocks outperforming their market index",
    "BB Squeeze":        "Volatility expansion — trades the breakout after a low-vol squeeze",
    "Gap Momentum":      "Gap + volume — buys stocks that gap up on high volume",
}


# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_screener_universe() -> tuple[pd.DataFrame, str, str]:
    """Load all stocks from the latest screener run (US + ASX + JPX sheets)."""
    files = sorted([f for f in SCREENER_DIR.glob("Goofy_Phase8*.xlsx")
                    if not f.name.startswith("~")])
    if not files:
        return pd.DataFrame(), "", ""
    latest = files[-1]

    frames = []
    for sheet_key in ["🇺🇸 US", "🇦🇺 ASX", "🇯🇵 JPX"]:
        try:
            df = pd.read_excel(latest, sheet_name=sheet_key, header=0)
            df = df.dropna(how="all")
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(), latest.name, ""

    combined = pd.concat(frames, ignore_index=True)

    # extract date from filename
    m = re.search(r"(\d{4}-\d{2}-\d{2})", latest.name)
    run_date = m.group(1) if m else "unknown"

    # hours ago
    try:
        dt = datetime.strptime(run_date, "%Y-%m-%d")
        diff = datetime.now() - dt
        hours_ago = f"{int(diff.total_seconds() / 3600)}h ago"
    except Exception:
        hours_ago = run_date

    return combined, run_date, hours_ago


@st.cache_data(ttl=600)
def load_trade_history() -> pd.DataFrame:
    """Load all closed trades across all runs for track record."""
    def fix_nan(s): return re.sub(r':\s*NaN', ': null', s)
    rows = []
    for run in range(1, 12):
        p = TRADES_DIR / f"run{run}_trades_log.json"
        if not p.exists():
            continue
        try:
            data = json.loads(fix_nan(p.read_text()))
        except Exception:
            continue
        for t in data.get("closed", []):
            pnl = t.get("pnl_pct")
            if pnl is None:
                continue
            rows.append({
                "run":         run,
                "asset":       t.get("asset", "?"),
                "market":      t.get("market", "?"),
                "strategy":    t.get("strategy", "?"),
                "pnl_pct":     float(pnl),
                "win":         float(pnl) > 0,
                "days_held":   t.get("days_held", 0),
                "exit_reason": t.get("exit_reason", "?"),
                "exit_date":   t.get("exit_date", ""),
                "entry_date":  t.get("entry_date", ""),
                "ml_score":    t.get("ml_score"),
                "tier":        t.get("tier", "?"),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_chart(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="6mo", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


# ── helpers ───────────────────────────────────────────────────────────────────

def signal_badge(verdict: str, simple: bool) -> str:
    if "TRADE" in verdict and "STAND" not in verdict:
        if simple:
            return "🟢 BUY"
        return "🟢 TRADE"
    if "ML HOLD" in verdict:
        return "🟡 WATCH" if simple else "🟡 ML HOLD"
    return "🔴 PASS" if simple else "🔴 STAND DOWN"


def ml_bar(score) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "—"
    filled = int(s / 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{bar} {s:.0f}%"


def tier_badge(tier: str) -> str:
    label, color = TIER_LABELS.get(str(tier), (f"Tier {tier}", "#8b949e"))
    return f"<span style='background:{color}20;color:{color};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold'>{label}</span>"


def trend_icon(trend: str) -> str:
    t = str(trend).lower()
    if "bull" in t: return "↑ Bull"
    if "bear" in t: return "↓ Bear"
    return "→ Sideways"


# ── indicator calculators ─────────────────────────────────────────────────────

def _rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def _macd(close, f=12, s=26, sig=9):
    ef  = close.ewm(span=f, adjust=False).mean()
    es  = close.ewm(span=s, adjust=False).mean()
    ml  = ef - es
    sl  = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl

def _bb(close, w=20, n=2.0):
    m  = close.rolling(w).mean()
    sd = close.rolling(w).std()
    return m + n*sd, m, m - n*sd

def _stoch(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    k_line = (close - lo) / (hi - lo + 1e-9) * 100
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def _donchian(high, low, period=20):
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    mid   = (upper + lower) / 2
    return upper, mid, lower

def _adx(high, low, close, period=14):
    tr  = pd.concat([high - low,
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    up  = high.diff().clip(lower=0)
    dn  = (-low.diff()).clip(lower=0)
    pdi = (up.rolling(period).mean() / atr.replace(0, np.nan)) * 100
    ndi = (dn.rolling(period).mean() / atr.replace(0, np.nan)) * 100
    dx  = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)) * 100
    adx = dx.rolling(period).mean()
    return adx, pdi, ndi

def _supertrend(high, low, close, period=10, mult=3.0):
    tr  = pd.concat([high - low,
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)
    for i in range(1, len(close)):
        prev_st  = st.iloc[i-1]
        prev_dir = direction.iloc[i-1]
        u = upper.iloc[i]; l = lower.iloc[i]
        if prev_dir == 1:
            st.iloc[i]        = max(l, prev_st) if close.iloc[i] > (prev_st if not np.isnan(prev_st) else l) else u
            direction.iloc[i] = 1 if close.iloc[i] > st.iloc[i] else -1
        else:
            st.iloc[i]        = min(u, prev_st) if close.iloc[i] < (prev_st if not np.isnan(prev_st) else u) else l
            direction.iloc[i] = -1 if close.iloc[i] < st.iloc[i] else 1
    return st, direction

def _keltner(close, high, low, ema_period=20, atr_period=10, mult=1.5):
    ema = close.ewm(span=ema_period, adjust=False).mean()
    tr  = pd.concat([high - low,
                     (high - close.shift()).abs(),
                     (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    return ema + mult*atr, ema, ema - mult*atr


# ── strategy chart builder ────────────────────────────────────────────────────

# Maps each strategy to its subplot layout
STRATEGY_SUBPLOTS = {
    "MA Crossover":      ["price"],
    "RSI":               ["price", "rsi"],
    "RSI Divergence":    ["price", "rsi"],
    "MACD":              ["price", "macd"],
    "Bollinger Bands":   ["price"],
    "Mean Reversion":    ["price", "rsi"],
    "Stochastic":        ["price", "stoch"],
    "Donchian":          ["price"],
    "ADX":               ["price", "adx"],
    "Supertrend":        ["price"],
    "Ichimoku":          ["price"],
    "Volume Breakout":   ["price", "volume"],
    "Relative Strength": ["price", "rs"],
    "BB Squeeze":        ["price"],
    "Gap Momentum":      ["price", "volume"],
}

STRATEGY_WHAT_TO_LOOK_FOR = {
    "MA Crossover":
        "The **blue line (MA 20)** crosses above the **orange line (MA 50)**. "
        "When the short-term average rises above the long-term average, momentum is shifting upward.",
    "RSI":
        "The **RSI line** (purple, bottom panel) drops below the **green dashed line (30)** — "
        "this means the stock is oversold. A signal fires when it bounces back above 30.",
    "RSI Divergence":
        "Price makes a **lower low** but the **RSI makes a higher low** (bottom panel). "
        "This divergence suggests sellers are losing strength even as price falls.",
    "MACD":
        "The **MACD line** (blue, bottom) crosses above the **signal line** (orange). "
        "Green histogram bars mean buying momentum is building.",
    "Bollinger Bands":
        "Price touches or crosses below the **lower purple band**. "
        "Statistically, price tends to revert toward the middle band from extremes.",
    "Mean Reversion":
        "Similar to Bollinger Bands — price is **unusually far below** its 20-day average. "
        "RSI below 30 (bottom panel) confirms the oversold condition.",
    "Stochastic":
        "The **%K line** (blue, bottom) crosses above the **%D line** (orange) "
        "while both are below 20. This is a classic oversold reversal signal.",
    "Donchian":
        "Price breaks **above the upper green band** — the highest it's been in 20 days. "
        "Donchian channels capture breakouts from consolidation ranges.",
    "ADX":
        "The **ADX line** (bottom panel) rises above 25, confirming a strong trend. "
        "At the same time +DI (green) crosses above -DI (red) — bulls are in control.",
    "Supertrend":
        "Price crosses **above the Supertrend line**. When the line flips from red to green "
        "and price is above it, the trend has turned bullish.",
    "Ichimoku":
        "Price breaks above the **Ichimoku cloud** (shaded area). The cloud acts as "
        "dynamic support/resistance — being above it means the trend is positive.",
    "Volume Breakout":
        "Price rises AND **volume** (bottom bars) spikes well above its 20-day average. "
        "High volume confirms buyers are serious, not just a thin-market move.",
    "Relative Strength":
        "The **relative strength line** (bottom panel) slopes upward — this stock is "
        "outperforming its market index (SPY/STW/Nikkei). Relative leaders tend to keep leading.",
    "BB Squeeze":
        "The **Bollinger Bands** (purple) narrow inside the **Keltner Channels** (orange) — "
        "this is the 'squeeze'. When BB expands back out, a breakout is starting.",
    "Gap Momentum":
        "The stock **gaps up** at open (price jumps overnight) on **high volume** (bottom). "
        "A strong gap with volume confirmation often continues higher intraday.",
}


def build_strategy_chart(price_df: pd.DataFrame, strategy: str,
                         asset: str, market: str,
                         benchmark_ticker: str = "SPY") -> go.Figure:
    """Build a Plotly chart with the right indicators for the given strategy."""
    if price_df.empty or "Close" not in price_df.columns:
        return go.Figure()

    close  = price_df["Close"].squeeze().dropna()
    high   = price_df["High"].squeeze().dropna() if "High"   in price_df.columns else close
    low    = price_df["Low"].squeeze().dropna()  if "Low"    in price_df.columns else close
    open_  = price_df["Open"].squeeze().dropna() if "Open"   in price_df.columns else close
    volume = price_df["Volume"].squeeze()        if "Volume" in price_df.columns else None

    layout = STRATEGY_SUBPLOTS.get(strategy, ["price"])
    n_rows = len(layout)
    heights = [0.6] + [0.2] * (n_rows - 1) if n_rows > 1 else [1.0]

    subplot_titles = []
    for p in layout:
        t = {"price": asset, "rsi": "RSI (14)",
             "macd": "MACD", "stoch": "Stochastic",
             "adx": "ADX", "volume": "Volume", "rs": "Relative Strength"}.get(p, p)
        subplot_titles.append(t)

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=heights,
                        subplot_titles=subplot_titles)

    # ── Row 1: candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=price_df.index, open=open_, high=high, low=low, close=close,
        name=asset,
        increasing=dict(line_color="#3fb950", fillcolor="#3fb950"),
        decreasing=dict(line_color="#f85149", fillcolor="#f85149"),
        showlegend=False,
    ), row=1, col=1)

    # ── Strategy-specific overlays ────────────────────────────────────────────
    if strategy == "MA Crossover":
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(x=price_df.index, y=ma20, name="MA 20",
            line=dict(color="#58a6ff", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=ma50, name="MA 50",
            line=dict(color="#f0883e", width=2, dash="dot")), row=1, col=1)
        # mark last crossover
        crosses = ((ma20 > ma50) & (ma20.shift() <= ma50.shift()))
        for idx in price_df.index[crosses][-3:]:
            fig.add_vline(x=idx, line_dash="dot", line_color="#3fb950",
                          opacity=0.5, row=1, col=1)

    elif strategy in ("Bollinger Bands", "BB Squeeze", "Mean Reversion"):
        bb_u, bb_m, bb_l = _bb(close)
        fig.add_trace(go.Scatter(x=price_df.index, y=bb_u, name="BB Upper",
            line=dict(color="#bc8cff", width=1, dash="dash"), opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=bb_m, name="BB Mid",
            line=dict(color="#bc8cff", width=1), opacity=0.4), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=bb_l, name="BB Lower",
            line=dict(color="#bc8cff", width=1, dash="dash"), opacity=0.7,
            fill="tonexty", fillcolor="rgba(188,140,255,0.04)"), row=1, col=1)
        if strategy == "BB Squeeze":
            kc_u, kc_m, kc_l = _keltner(close, high, low)
            fig.add_trace(go.Scatter(x=price_df.index, y=kc_u, name="KC Upper",
                line=dict(color="#f0883e", width=1, dash="dot"), opacity=0.6), row=1, col=1)
            fig.add_trace(go.Scatter(x=price_df.index, y=kc_l, name="KC Lower",
                line=dict(color="#f0883e", width=1, dash="dot"), opacity=0.6), row=1, col=1)

    elif strategy == "Donchian":
        dc_u, dc_m, dc_l = _donchian(high, low)
        fig.add_trace(go.Scatter(x=price_df.index, y=dc_u, name="DC Upper",
            line=dict(color="#3fb950", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=dc_m, name="DC Mid",
            line=dict(color="#8b949e", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=dc_l, name="DC Lower",
            line=dict(color="#f85149", width=1.5),
            fill="tonexty", fillcolor="rgba(63,185,80,0.05)"), row=1, col=1)

    elif strategy == "Supertrend":
        st_line, st_dir = _supertrend(high, low, close)
        bull_mask = st_dir == 1
        bear_mask = st_dir == -1
        idx = price_df.index
        if bull_mask.any():
            fig.add_trace(go.Scatter(x=idx[bull_mask], y=st_line[bull_mask],
                mode="lines", name="Supertrend (Bull)",
                line=dict(color="#3fb950", width=2)), row=1, col=1)
        if bear_mask.any():
            fig.add_trace(go.Scatter(x=idx[bear_mask], y=st_line[bear_mask],
                mode="lines", name="Supertrend (Bear)",
                line=dict(color="#f85149", width=2)), row=1, col=1)

    elif strategy == "Ichimoku":
        conv   = (high.rolling(9).max() + low.rolling(9).min()) / 2
        base   = (high.rolling(26).max() + low.rolling(26).min()) / 2
        span_a = ((conv + base) / 2).shift(26)
        span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        fig.add_trace(go.Scatter(x=price_df.index, y=conv, name="Conversion",
            line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=base, name="Base",
            line=dict(color="#f0883e", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=span_a, name="Span A",
            line=dict(color="#3fb950", width=0.8, dash="dot"),
            fill=None), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=span_b, name="Span B",
            line=dict(color="#f85149", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(63,185,80,0.08)"), row=1, col=1)

    # ── Subpanels ─────────────────────────────────────────────────────────────
    cur_row = 2

    if "rsi" in layout:
        rsi = _rsi(close)
        fig.add_trace(go.Scatter(x=price_df.index, y=rsi, name="RSI 14",
            line=dict(color="#e879f9", width=1.5)), row=cur_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f85149", opacity=0.5,
                      row=cur_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", opacity=0.5,
                      row=cur_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(63,185,80,0.06)",
                      line_width=0, row=cur_row, col=1)
        fig.update_yaxes(range=[0, 100], row=cur_row, col=1, tickfont_size=9)

        if strategy == "RSI Divergence":
            # annotate the most recent local min in RSI
            rsi_smooth = rsi.rolling(3).mean()
            if len(rsi_smooth.dropna()) > 10:
                recent = rsi_smooth.iloc[-40:]
                local_min_idx = recent.idxmin()
                fig.add_annotation(
                    x=local_min_idx, y=float(rsi_smooth[local_min_idx]),
                    text="Divergence<br>zone",
                    showarrow=True, arrowhead=2, arrowcolor="#d29922",
                    font=dict(color="#d29922", size=10),
                    row=cur_row, col=1,
                )
        cur_row += 1

    if "macd" in layout:
        ml, sl, mh = _macd(close)
        hc = ["#3fb950" if v >= 0 else "#f85149" for v in mh.fillna(0)]
        fig.add_trace(go.Bar(x=price_df.index, y=mh, name="Histogram",
            marker_color=hc, opacity=0.6), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=ml, name="MACD",
            line=dict(color="#58a6ff", width=1.5)), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=sl, name="Signal",
            line=dict(color="#f0883e", width=1.5)), row=cur_row, col=1)
        fig.update_yaxes(row=cur_row, col=1, tickfont_size=9)
        cur_row += 1

    if "stoch" in layout:
        k, d = _stoch(high, low, close)
        fig.add_trace(go.Scatter(x=price_df.index, y=k, name="%K",
            line=dict(color="#58a6ff", width=1.5)), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=d, name="%D",
            line=dict(color="#f0883e", width=1.5, dash="dot")), row=cur_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="#f85149", opacity=0.4,
                      row=cur_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#3fb950", opacity=0.4,
                      row=cur_row, col=1)
        fig.update_yaxes(range=[0, 100], row=cur_row, col=1, tickfont_size=9)
        cur_row += 1

    if "adx" in layout:
        adx, pdi, ndi = _adx(high, low, close)
        fig.add_trace(go.Scatter(x=price_df.index, y=adx, name="ADX",
            line=dict(color="#f0883e", width=2)), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=pdi, name="+DI",
            line=dict(color="#3fb950", width=1.5)), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=ndi, name="-DI",
            line=dict(color="#f85149", width=1.5)), row=cur_row, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="#8b949e", opacity=0.5,
                      annotation_text="Trend threshold (25)",
                      annotation_font_color="#8b949e",
                      row=cur_row, col=1)
        fig.update_yaxes(row=cur_row, col=1, tickfont_size=9)
        cur_row += 1

    if "volume" in layout and volume is not None:
        vol_clean = volume.reindex(price_df.index).fillna(0)
        vol_avg   = vol_clean.rolling(20).mean()
        vc = ["#3fb950" if c >= o else "#f85149"
              for c, o in zip(close.reindex(price_df.index),
                              open_.reindex(price_df.index))]
        fig.add_trace(go.Bar(x=price_df.index, y=vol_clean, name="Volume",
            marker_color=vc, opacity=0.6), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=price_df.index, y=vol_avg, name="Vol MA 20",
            line=dict(color="#d29922", width=1.5)), row=cur_row, col=1)
        fig.update_yaxes(row=cur_row, col=1, tickfont_size=9)
        cur_row += 1

    if "rs" in layout:
        bench_map = {"US": "SPY", "ASX": "STW.AX", "JPX": "1321.T"}
        bench_t   = bench_map.get(market, benchmark_ticker)
        try:
            bench_df = yf.download(bench_t, start=price_df.index[0],
                                   auto_adjust=True, progress=False)
            if isinstance(bench_df.columns, pd.MultiIndex):
                bench_df.columns = bench_df.columns.droplevel(1)
            bench_close = bench_df["Close"].squeeze().reindex(price_df.index, method="ffill")
            rs = (close / close.iloc[0]) / (bench_close / bench_close.iloc[0])
            fig.add_trace(go.Scatter(x=price_df.index, y=rs, name=f"RS vs {bench_t}",
                line=dict(color="#58a6ff", width=2)), row=cur_row, col=1)
            fig.add_hline(y=1.0, line_dash="dash", line_color="#8b949e",
                          annotation_text="benchmark", row=cur_row, col=1)
        except Exception:
            pass
        cur_row += 1

    # ── signal marker — vertical line at today ────────────────────────────────
    last_date = price_df.index[-1]
    last_date_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)
    fig.add_shape(type="line", x0=last_date_str, x1=last_date_str,
                  y0=0, y1=1, yref="paper",
                  line=dict(dash="dash", color="#d29922", width=1.5), opacity=0.8)
    fig.add_annotation(x=last_date_str, y=0.98, yref="paper",
                       text="Signal today", showarrow=False,
                       font=dict(color="#d29922", size=11),
                       xanchor="left", yanchor="top")

    # ── layout ────────────────────────────────────────────────────────────────
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
    chg    = last_close - prev_close
    chgp   = chg / prev_close * 100 if prev_close else 0
    clr    = "#3fb950" if chg >= 0 else "#f85149"

    fig.update_layout(
        height=160 + 200 * n_rows,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font_size=10),
        margin=dict(l=60, r=20, t=60, b=40),
        title=dict(
            text=(f"<b>{asset}</b> &nbsp; "
                  f"<span style='color:{clr}'>{last_close:.2f} "
                  f"{chg:+.2f} ({chgp:+.2f}%)</span> &nbsp; "
                  f"<span style='color:#8b949e;font-size:13px'>"
                  f"Strategy: {strategy}</span>"),
            font_size=16,
        ),
    )
    for r in range(1, n_rows + 1):
        fig.update_xaxes(gridcolor="#21262d", row=r, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=r, col=1)

    return fig


# ── rankings table renderer ───────────────────────────────────────────────────

def _render_table(df: pd.DataFrame, verdict_col: str, simple: bool):
    """Render a stock rankings table in simple or advanced mode."""
    if simple:
        for _, row in df.iterrows():
            asset   = row.get("Asset", "?")
            market  = row.get("Market", "?")
            strat   = row.get("Best Strategy", "?")
            tier    = str(row.get("Tier", "?"))
            ml      = row.get("ML Score", None)
            trend   = str(row.get("Current Trend", "?"))
            size    = row.get("Adj Size %", row.get("Recommended Size %", None))
            verdict = str(row.get(verdict_col, ""))

            badge = signal_badge(verdict, simple=True)
            ml_str = f"{float(ml):.0f}%" if pd.notna(ml) and ml not in ("—", None, "") else "—"
            try:
                ml_num = float(ml_str.replace("%",""))
                ml_color = "#3fb950" if ml_num >= 80 else ("#d29922" if ml_num >= 60 else "#8b949e")
            except Exception:
                ml_color = "#8b949e"
            size_str   = f"{float(size):.1f}%" if pd.notna(size) and size not in ("—", None, "") else "—"
            tier_color = TIER_LABELS.get(tier, ("", "#8b949e"))[1]
            trend_str  = trend_icon(trend)

            st.markdown(
                f"<div class='stock-row'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div>"
                f"<b style='font-size:16px'>{asset}</b> "
                f"<span style='color:#8b949e;font-size:12px'>{market}</span> &nbsp; "
                f"<span style='background:{tier_color}20;color:{tier_color};"
                f"padding:1px 7px;border-radius:4px;font-size:11px;font-weight:bold'>Tier {tier}</span>"
                f"</div>"
                f"<div style='font-size:18px;font-weight:bold'>{badge}</div>"
                f"</div>"
                f"<div style='margin-top:6px;font-size:13px;color:#8b949e'>"
                f"Strategy: <span style='color:#e6edf3'>{strat}</span> &nbsp;·&nbsp; "
                f"Confidence: <span style='color:{ml_color};font-weight:bold'>{ml_str}</span> &nbsp;·&nbsp; "
                f"Trend: {trend_str} &nbsp;·&nbsp; "
                f"Suggested size: <b>{size_str}</b>"
                f"</div>"
                f"<div style='margin-top:4px;font-size:12px;color:#6e7681'>"
                f"{STRATEGY_PLAIN.get(strat, '')}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        cols_to_show = [c for c in [
            "Market", "Asset", "Tier", "Best Strategy", "Score",
            "OUT Sharpe", "OUT Win Rate %", "Current Trend",
            "ML Score", "ML Gate", verdict_col, "Adj Size %",
        ] if c in df.columns]
        adv = df[cols_to_show].copy().rename(columns={verdict_col: "Verdict"})

        def _c_verdict(v):
            if "TRADE" in str(v) and "STAND" not in str(v): return "color:#3fb950;font-weight:bold"
            if "HOLD"  in str(v): return "color:#d29922;font-weight:bold"
            return "color:#8b949e"

        def _c_score(v):
            try:
                f = float(v)
                return "color:#3fb950" if f >= 70 else ("color:#d29922" if f >= 50 else "color:#f85149")
            except Exception:
                return ""

        fmt = {}
        if "Score"       in adv.columns: fmt["Score"]      = "{:.1f}"
        if "OUT Sharpe"  in adv.columns: fmt["OUT Sharpe"] = "{:.3f}"
        if "ML Score"    in adv.columns: fmt["ML Score"]   = lambda v: f"{float(v):.0f}%" if pd.notna(v) else "—"
        if "Adj Size %"  in adv.columns: fmt["Adj Size %"] = lambda v: f"{float(v):.1f}%" if pd.notna(v) else "—"

        styled = adv.style
        if "Verdict" in adv.columns:
            styled = styled.applymap(_c_verdict, subset=["Verdict"])
        if "Score" in adv.columns:
            styled = styled.applymap(_c_score, subset=["Score"])
        if fmt:
            styled = styled.format(fmt, na_rep="—")
        st.dataframe(styled, use_container_width=True, hide_index=True, height=450)


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Goofy Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]               { background: #161b22; border-right: 1px solid #30363d; }
.stTabs [data-baseweb="tab"]            { font-weight: 600; padding: 6px 20px; }
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; text-align: center;
}
.metric-card .label { color: #8b949e; font-size: 12px; margin-bottom: 4px; }
.metric-card .value { color: #e6edf3; font-size: 26px; font-weight: 700; }
.metric-card .sub   { color: #8b949e; font-size: 11px; margin-top: 2px; }
.stock-row {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; margin: 6px 0;
}
.section-header { color: #e6edf3; font-size: 20px; font-weight: 700; margin: 24px 0 12px; }
.disclaimer-box {
    background: #3d2600; border: 1px solid #d29922; border-radius: 8px;
    padding: 12px 16px; font-size: 12px; color: #d29922; margin-top: 8px;
}
a { color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ── sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Goofy Screener")
    st.caption("Free quantitative stock screener\nUS · ASX · JPX")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Home", "Screener Rankings", "Stock Chart", "Track Record", "About & Disclaimer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # simple/advanced toggle (shown on rankings pages)
    simple_mode = st.toggle("Simple mode", value=True,
                             help="ON = plain English. OFF = full quant data.")
    st.caption("Simple mode hides quant jargon.")
    st.markdown("---")
    st.caption("Data: yfinance · Methodology: walk-forward backtested strategies")
    st.caption("Not financial advice.")

# ── load data ─────────────────────────────────────────────────────────────────
df_universe, run_date, hours_ago = load_screener_universe()
df_history = load_trade_history()


# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
<div style='padding: 32px 0 16px'>
  <div style='font-size:36px;font-weight:800;color:#e6edf3'>Goofy Screener</div>
  <div style='font-size:18px;color:#8b949e;margin-top:6px'>
    Free quantitative stock screener — United States · Australia · Japan
  </div>
</div>
""", unsafe_allow_html=True)

    # last updated badge
    if run_date:
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
            f"padding:8px 14px;display:inline-block;font-size:13px;color:#8b949e'>"
            f"Last updated: <b style='color:#e6edf3'>{run_date}</b> ({hours_ago})"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("")

    # quick stats
    if not df_universe.empty:
        verdict_col = "P7 Verdict" if "P7 Verdict" in df_universe.columns else "Today's Verdict"
        total_stocks  = len(df_universe)
        trade_signals = df_universe[df_universe[verdict_col].str.contains("TRADE", na=False) &
                                    ~df_universe[verdict_col].str.contains("STAND", na=False)]
        n_signals = len(trade_signals)
        n_markets = df_universe["Market"].nunique() if "Market" in df_universe.columns else 3

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, sub in [
            (c1, "Stocks screened", str(total_stocks), "US + ASX + JPX"),
            (c2, "Buy signals today", str(n_signals), f"across {n_markets} markets"),
            (c3, "Markets covered", "3", "US · ASX · JPX"),
            (c4, "Strategies used", "15", "Walk-forward validated"),
        ]:
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value'>{value}</div>"
                f"<div class='sub'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # how it works
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### How it works")
        st.markdown("""
The Goofy Screener runs 15 quantitative trading strategies across 113 stocks every day.
For each stock, it finds the strategy with the strongest historical performance — tested on
data the model never saw during training.

**The pipeline:**
1. **Strategy selection** — 15 strategies are backtested and scored. The best one is chosen per stock.
2. **Regime filter** — signals are suppressed in markets where the chosen strategy historically underperforms (Bull / Bear / Sideways).
3. **ML gate** — an XGBoost model trained on hundreds of historical signals gives a confidence score. Low-confidence signals are held back.
4. **Position sizing** — Kelly criterion + volatility scaling sets a recommended allocation.

Signals are labelled **BUY**, **WATCH**, or **PASS**.
""")

    with col_r:
        st.markdown("### Today's top picks")
        if not df_universe.empty and not trade_signals.empty:
            top = trade_signals.sort_values("ML Score", ascending=False).head(6)
            for _, row in top.iterrows():
                mkt   = row.get("Market", "?")
                asset = row.get("Asset", "?")
                strat = row.get("Best Strategy", "?")
                ml    = row.get("ML Score", "—")
                tier  = row.get("Tier", "?")
                ml_str = f"{float(ml):.0f}%" if pd.notna(ml) and ml != "—" else "—"
                ml_color = "#3fb950" if ml_str != "—" and float(ml_str[:-1]) >= 80 else "#d29922"
                st.markdown(
                    f"<div class='stock-row'>"
                    f"<b>{asset}</b> &nbsp; <span style='color:#8b949e;font-size:12px'>{mkt}</span><br>"
                    f"<span style='font-size:12px;color:#8b949e'>{strat}</span> &nbsp;|&nbsp; "
                    f"ML: <span style='color:{ml_color};font-weight:bold'>{ml_str}</span> &nbsp;|&nbsp; "
                    f"Tier {tier}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No screener data found. Run goofy_screener_phase8.py first.")

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer-box'>"
        "This website is for educational purposes only and does not constitute financial advice. "
        "Past performance does not guarantee future results. Always do your own research before investing."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENER RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Screener Rankings":
    st.markdown("### Screener Rankings")
    if run_date:
        st.caption(f"Data from {run_date} ({hours_ago})  |  "
                   f"{'Simple mode — BUY/WATCH/PASS' if simple_mode else 'Advanced mode — full quant data'}")

    if df_universe.empty:
        st.warning("No screener data found. Run the screener first.")
    else:
        verdict_col = "P7 Verdict" if "P7 Verdict" in df_universe.columns else "Today's Verdict"

        # ── filters ──
        f1, f2, f3 = st.columns(3)
        mkt_filter  = f1.multiselect("Market", ["US", "ASX", "JPX"],
                                      default=["US", "ASX", "JPX"])
        tier_filter = f2.multiselect("Tier", ["S", "A", "B"],
                                      default=["S", "A", "B"])
        show_all    = f3.toggle("Show all (incl. PASS)", value=False)

        df_filtered = df_universe[df_universe["Market"].isin(mkt_filter)].copy()
        if "Tier" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Tier"].isin(tier_filter)]
        if not show_all:
            df_filtered = df_filtered[
                df_filtered[verdict_col].str.contains("TRADE", na=False) |
                df_filtered[verdict_col].str.contains("HOLD", na=False)
            ]

        if df_filtered.empty:
            st.info("No signals match the current filters.")
        else:
            # ── BUY signals first ──
            trade_mask = (df_filtered[verdict_col].str.contains("TRADE", na=False) &
                          ~df_filtered[verdict_col].str.contains("STAND", na=False))
            df_trade   = df_filtered[trade_mask].sort_values("ML Score", ascending=False)
            df_rest    = df_filtered[~trade_mask].sort_values("Score", ascending=False)

            if not df_trade.empty:
                st.markdown(f"#### BUY signals ({len(df_trade)})")
                _render_table(df_trade, verdict_col, simple_mode)

            if not df_rest.empty and show_all:
                st.markdown(f"#### Watch / Pass ({len(df_rest)})")
                _render_table(df_rest, verdict_col, simple_mode)
            elif not df_rest.empty:
                st.caption(f"+ {len(df_rest)} WATCH/PASS signals hidden. Toggle 'Show all' to see them.")

    st.markdown(
        "<div class='disclaimer-box' style='margin-top:24px'>"
        "Signals are generated by algorithmic models and are NOT investment advice. "
        "Always conduct your own research."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TRACK RECORD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Track Record":
    st.markdown("### Track Record")
    st.caption("Real paper trading results — all trades shown including losses. No cherry-picking.")

    if df_history.empty:
        st.info("No closed trades yet.")
    else:
        dc = df_history.copy()
        n_total  = len(dc)
        n_wins   = int(dc["win"].sum())
        win_rate = n_wins / n_total * 100
        avg_pnl  = dc["pnl_pct"].mean()
        avg_win  = dc[dc["win"]]["pnl_pct"].mean() if n_wins > 0 else 0
        avg_loss = dc[~dc["win"]]["pnl_pct"].mean() if n_total - n_wins > 0 else 0
        pf       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        n_stops  = int((dc["exit_reason"] == "STOP_LOSS").sum())

        # ── summary metrics ──
        cols = st.columns(6)
        for col, label, val, sub in [
            (cols[0], "Total trades",    str(n_total),          "all markets"),
            (cols[1], "Win rate",        f"{win_rate:.1f}%",    f"{n_wins}W / {n_total-n_wins}L"),
            (cols[2], "Avg P&L",         f"{avg_pnl:+.2f}%",   "per trade"),
            (cols[3], "Avg winner",      f"{avg_win:+.2f}%",    "when right"),
            (cols[4], "Avg loser",       f"{avg_loss:+.2f}%",   "when wrong"),
            (cols[5], "Stop-losses hit", str(n_stops),          f"{n_stops/n_total*100:.0f}% of trades"),
        ]:
            color = "#3fb950" if "+" in str(val) else ("#f85149" if "-" in str(val) else "#e6edf3")
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value' style='color:{color}'>{val}</div>"
                f"<div class='sub'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_l, col_r = st.columns([2, 1])

        with col_l:
            # ── equity curve ──
            st.markdown("#### Cumulative P&L over time")
            st.caption("Each step = one closed trade, sorted by exit date.")
            dc_sorted = dc.dropna(subset=["exit_date"]).copy()
            dc_sorted["exit_date"] = pd.to_datetime(dc_sorted["exit_date"], errors="coerce")
            dc_sorted = dc_sorted.dropna(subset=["exit_date"]).sort_values("exit_date")
            dc_sorted["cumulative"] = dc_sorted["pnl_pct"].cumsum()
            dc_sorted["trade_num"]  = range(1, len(dc_sorted) + 1)

            fig_eq = go.Figure()
            # colour line segments by direction
            for i in range(1, len(dc_sorted)):
                x0 = dc_sorted["exit_date"].iloc[i-1]
                x1 = dc_sorted["exit_date"].iloc[i]
                y0 = dc_sorted["cumulative"].iloc[i-1]
                y1 = dc_sorted["cumulative"].iloc[i]
                color = "#3fb950" if y1 >= y0 else "#f85149"
                fig_eq.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            # hover dots
            fig_eq.add_trace(go.Scatter(
                x=dc_sorted["exit_date"],
                y=dc_sorted["cumulative"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=dc_sorted["pnl_pct"],
                    colorscale=[[0,"#f85149"],[0.5,"#8b949e"],[1,"#3fb950"]],
                ),
                text=[f"{a} | {s} | {p:+.1f}% | {d}"
                      for a, s, p, d in zip(dc_sorted["asset"], dc_sorted["strategy"],
                                            dc_sorted["pnl_pct"], dc_sorted["exit_date"].dt.strftime("%b %d"))],
                hoverinfo="text",
                showlegend=False,
            ))

            fig_eq.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
            fig_eq.update_layout(
                height=320, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3", margin=dict(l=50,r=20,t=20,b=40),
                xaxis=dict(gridcolor="#21262d", title=""),
                yaxis=dict(gridcolor="#21262d", title="Cumulative P&L %"),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # ── strategy table ──
            st.markdown("#### By strategy")
            ss = (dc.groupby("strategy")
                  .agg(trades=("pnl_pct","count"),
                       wins=("win","sum"),
                       avg_pnl=("pnl_pct","mean"))
                  .reset_index()
                  .sort_values("avg_pnl", ascending=False))
            ss["win_rate"] = ss["wins"] / ss["trades"] * 100
            ss["Signal"] = ss["avg_pnl"].apply(
                lambda v: "🟢 Positive" if v > 0 else "🔴 Negative"
            )
            ss_display = ss[["strategy","trades","win_rate","avg_pnl","Signal"]].copy()
            ss_display.columns = ["Strategy","Trades","Win rate %","Avg P&L %","Result"]
            ss_display["Win rate %"] = ss_display["Win rate %"].round(1)
            ss_display["Avg P&L %"]  = ss_display["Avg P&L %"].round(2)

            def _color_pnl(v):
                if pd.isna(v): return ""
                return "color:#3fb950;font-weight:bold" if v > 0 else "color:#f85149;font-weight:bold"

            st.dataframe(
                ss_display.style
                    .applymap(_color_pnl, subset=["Avg P&L %"])
                    .format({"Avg P&L %": "{:+.2f}%", "Win rate %": "{:.1f}%"}),
                use_container_width=True, hide_index=True,
            )

        with col_r:
            # ── exit breakdown ──
            st.markdown("#### How trades ended")
            exit_counts = dc["exit_reason"].value_counts()
            exit_colors = {
                "STOP_LOSS":       "#f85149",
                "HOLD_COMPLETE":   "#3fb950",
                "SIGNAL_REVERSAL": "#58a6ff",
                "TAKE_PROFIT":     "#39d353",
            }
            fig_pie = go.Figure(go.Pie(
                labels=[l.replace("_"," ").title() for l in exit_counts.index],
                values=exit_counts.values,
                hole=0.45,
                marker_colors=[exit_colors.get(l,"#8b949e") for l in exit_counts.index],
                textfont_size=11,
            ))
            fig_pie.update_layout(
                height=240, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#e6edf3", margin=dict(l=0,r=0,t=10,b=10),
                legend=dict(font_size=10, orientation="v"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # ── market breakdown ──
            st.markdown("#### By market")
            ms = (dc.groupby("market")
                  .agg(trades=("pnl_pct","count"),
                       wins=("win","sum"),
                       avg_pnl=("pnl_pct","mean"))
                  .reset_index())
            ms["win_rate"] = ms["wins"] / ms["trades"] * 100
            for _, row in ms.iterrows():
                color = "#3fb950" if row["avg_pnl"] > 0 else "#f85149"
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-left:4px solid {color};border-radius:6px;"
                    f"padding:10px 14px;margin:4px 0'>"
                    f"<b>{row['market']}</b> &nbsp; "
                    f"<span style='color:{color};font-weight:bold'>{row['avg_pnl']:+.2f}%</span> avg<br>"
                    f"<span style='color:#8b949e;font-size:12px'>"
                    f"{int(row['trades'])} trades · {row['win_rate']:.0f}% win rate"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("#### Recent closed trades")
            recent = dc.sort_values("exit_date", ascending=False).head(10)
            for _, r in recent.iterrows():
                p = r["pnl_pct"]
                color = "#3fb950" if p > 0 else "#f85149"
                icon  = "W" if p > 0 else "L"
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-left:3px solid {color};border-radius:4px;"
                    f"padding:5px 10px;margin:2px 0;font-size:12px'>"
                    f"<b style='color:{color}'>[{icon}]</b> "
                    f"<b>{r['asset']}</b> "
                    f"<span style='color:{color};font-weight:bold'>{p:+.2f}%</span>"
                    f"<span style='color:#8b949e'> · {r['exit_date']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer-box'>"
        "Past performance does not guarantee future results. "
        "These are paper trading results — no real money was used. "
        "This is not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT & DISCLAIMER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Stock Chart":
    # ══════════════════════════════════════════════════════════════════════════
    #  STOCK CHART
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 Stock Chart")
    st.markdown("Pick any stock from today's screener universe and see its price chart "
                "with the strategy indicators that drove the signal.")

    if df_universe.empty:
        st.info("No screener data loaded yet. Run the screener first.")
    else:
        # ── stock picker ──────────────────────────────────────────────────────
        # Collect unique assets with their strategy + market info
        asset_col    = next((c for c in df_universe.columns
                             if c.lower() in ("asset", "ticker", "symbol")), None)
        strat_col    = next((c for c in df_universe.columns
                             if "strategy" in c.lower()), None)
        verdict_col  = next((c for c in df_universe.columns
                             if "verdict" in c.lower()), None)
        market_col   = next((c for c in df_universe.columns
                             if "market" in c.lower()), None)
        ml_col       = next((c for c in df_universe.columns
                             if "ml" in c.lower() and "score" in c.lower()), None)
        tier_col     = next((c for c in df_universe.columns
                             if "tier" in c.lower()), None)
        size_col     = next((c for c in df_universe.columns
                             if "size" in c.lower() or "adj" in c.lower()), None)

        if not asset_col:
            st.error("Could not find an Asset/Ticker column in screener data.")
            st.stop()

        # Build a display list: BUY signals first, then rest
        def _verdict_rank(v):
            v = str(v).upper()
            if "TRADE" in v or "BUY" in v: return 0
            if "WATCH" in v or "HOLD" in v: return 1
            return 2

        df_pick = df_universe.copy()
        if verdict_col:
            df_pick["_rank"] = df_pick[verdict_col].apply(_verdict_rank)
            df_pick = df_pick.sort_values(["_rank", asset_col])
        else:
            df_pick = df_pick.sort_values(asset_col)

        all_assets = df_pick[asset_col].dropna().unique().tolist()

        # Filter by market in sidebar-like columns
        col_f1, col_f2, _ = st.columns([1, 1, 2])
        with col_f1:
            market_filter = st.selectbox(
                "Market", ["All"] + (
                    sorted(df_pick[market_col].dropna().unique().tolist())
                    if market_col else []
                ), key="chart_market"
            )
        with col_f2:
            verdict_filter = st.selectbox(
                "Show", ["All", "BUY signals only", "WATCH only"],
                key="chart_verdict"
            )

        filtered = df_pick.copy()
        if market_filter != "All" and market_col:
            filtered = filtered[filtered[market_col] == market_filter]
        if verdict_filter == "BUY signals only" and verdict_col:
            filtered = filtered[filtered[verdict_col].astype(str).str.upper()
                                .str.contains("TRADE|BUY")]
        elif verdict_filter == "WATCH only" and verdict_col:
            filtered = filtered[filtered[verdict_col].astype(str).str.upper()
                                .str.contains("WATCH|HOLD")]

        asset_options = filtered[asset_col].dropna().unique().tolist()
        if not asset_options:
            st.info("No stocks match those filters.")
            st.stop()

        selected_asset = st.selectbox(
            "Select a stock", asset_options, key="chart_asset"
        )

        # ── get that row's metadata ────────────────────────────────────────────
        row = df_universe[df_universe[asset_col] == selected_asset].iloc[0]
        strategy  = str(row[strat_col]).strip()  if strat_col  else "Unknown"
        verdict   = str(row[verdict_col]).strip() if verdict_col else "—"
        ml_score  = row[ml_col]                  if ml_col     else None
        tier      = row[tier_col]                 if tier_col   else None
        size_pct  = row[size_col]                 if size_col   else None
        market    = str(row[market_col]).strip()  if market_col else "US"

        # ── signal card ───────────────────────────────────────────────────────
        v_upper = verdict.upper()
        if "TRADE" in v_upper or "BUY" in v_upper:
            badge_clr, badge_lbl = "#3fb950", "🟢 BUY SIGNAL"
        elif "WATCH" in v_upper or "HOLD" in v_upper:
            badge_clr, badge_lbl = "#d29922", "🟡 WATCH"
        else:
            badge_clr, badge_lbl = "#8b949e", "⚪ PASS"

        strat_desc = STRATEGY_PLAIN.get(strategy, "Quantitative signal based on price history.")
        what_desc  = STRATEGY_WHAT_TO_LOOK_FOR.get(strategy, "")

        card_cols = st.columns([1, 1, 1, 1])
        with card_cols[0]:
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>Signal</div>"
                f"<div style='font-size:20px;font-weight:700;color:{badge_clr}'>{badge_lbl}</div>"
                f"</div>", unsafe_allow_html=True)
        with card_cols[1]:
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>Strategy</div>"
                f"<div style='font-size:15px;font-weight:600;color:#e6edf3'>{strategy}</div>"
                f"</div>", unsafe_allow_html=True)
        with card_cols[2]:
            ml_text = f"{float(ml_score):.0f}" if ml_score is not None else "—"
            ml_clr  = "#3fb950" if ml_score is not None and float(ml_score) >= 70 else "#d29922"
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>ML Score</div>"
                f"<div style='font-size:20px;font-weight:700;color:{ml_clr}'>{ml_text}</div>"
                f"</div>", unsafe_allow_html=True)
        with card_cols[3]:
            size_text = f"{float(size_pct):.1f}%" if size_pct is not None else "—"
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>Suggested Size</div>"
                f"<div style='font-size:20px;font-weight:700;color:#58a6ff'>{size_text}</div>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("")

        # strategy plain-English box
        st.markdown(
            f"<div style='background:#161b22;border-left:3px solid {badge_clr};"
            f"padding:12px 16px;border-radius:6px;margin-bottom:12px'>"
            f"<b style='color:#e6edf3'>{strategy}:</b> "
            f"<span style='color:#c9d1d9'>{strat_desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if what_desc:
            st.markdown(
                f"<div style='background:#0d1117;border:1px solid #30363d;"
                f"padding:12px 16px;border-radius:6px;margin-bottom:16px'>"
                f"<b style='color:#d29922'>What to look for on the chart:</b> "
                f"<span style='color:#c9d1d9'>{what_desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── fetch price data & draw chart ─────────────────────────────────────
        with st.spinner(f"Loading {selected_asset} price data…"):
            price_df = fetch_chart(selected_asset)

        if price_df.empty:
            st.warning(f"Could not load price data for **{selected_asset}**. "
                       "Try a different ticker or check your connection.")
        else:
            fig = build_strategy_chart(price_df, strategy, selected_asset, market)
            st.plotly_chart(fig, use_container_width=True)

            # quick stats row below the chart
            close = price_df["Close"].squeeze().dropna()
            if len(close) >= 20:
                s_cols = st.columns(5)
                last   = float(close.iloc[-1])
                hi52   = float(price_df["High"].squeeze().max()) if "High" in price_df.columns else last
                lo52   = float(price_df["Low"].squeeze().min())  if "Low"  in price_df.columns else last
                ma20   = float(close.rolling(20).mean().iloc[-1])
                rsi14  = float(_rsi(close).iloc[-1]) if not np.isnan(_rsi(close).iloc[-1]) else 0
                pct_hi = (last / hi52 - 1) * 100
                def _stat(label, val, colour="#e6edf3"):
                    return (f"<div style='background:#161b22;border-radius:8px;padding:10px;text-align:center'>"
                            f"<div style='font-size:10px;color:#8b949e;text-transform:uppercase'>{label}</div>"
                            f"<div style='font-size:16px;font-weight:700;color:{colour}'>{val}</div>"
                            f"</div>")
                rsi_clr = "#3fb950" if rsi14 < 30 else ("#f85149" if rsi14 > 70 else "#e6edf3")
                s_cols[0].markdown(_stat("Last Price", f"{last:.2f}"), unsafe_allow_html=True)
                s_cols[1].markdown(_stat("MA 20", f"{ma20:.2f}",
                    "#3fb950" if last > ma20 else "#f85149"), unsafe_allow_html=True)
                s_cols[2].markdown(_stat("RSI 14", f"{rsi14:.1f}", rsi_clr), unsafe_allow_html=True)
                s_cols[3].markdown(_stat("6M High", f"{hi52:.2f}"), unsafe_allow_html=True)
                s_cols[4].markdown(_stat("% From High",
                    f"{pct_hi:.1f}%", "#3fb950" if pct_hi > -5 else "#d29922"),
                    unsafe_allow_html=True)

        # disclaimer at the bottom
        st.markdown("---")
        st.caption("Chart data: yfinance (6 months daily). Indicators are illustrative. "
                   "Not financial advice — always do your own research.")


elif page == "About & Disclaimer":
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### About Goofy Screener")
        st.markdown("""
Goofy Screener is a free quantitative stock screener built for retail investors who want
a systematic, data-driven view of the market — without paying for Bloomberg or a quant fund.

**Who built this?**
A finance student at the University of Queensland, learning quantitative trading by building the tools
from scratch. Every strategy, every filter, every line of code is custom-built and documented.

**Why is it free?**
Because the best way to learn is to build something real, and the best way to validate it is
to be transparent about the results — including the losses.

---

### Methodology

**Universe:** 113 stocks across US (39), ASX (28), and JPX (46) markets.

**15 trading strategies**, each tested independently:
""")
        for name, desc in STRATEGY_PLAIN.items():
            st.markdown(f"- **{name}** — {desc}")

        st.markdown("""
---

**Walk-forward backtesting:** Strategies are trained on 2016-2021 data and tested on
2021-present data that was never seen during training. This avoids the most common
backtesting pitfall (overfitting to history).

**Strategy selection:** Each stock is paired with its strongest strategy based on:
out-of-sample Sharpe ratio (40%), consistency (20%), win rate (10%),
drawdown control (10%), and recent 6-month performance (20%).

**Regime filter:** Signals are only issued when the current market regime (Bull/Bear/Sideways)
matches the regimes where the strategy historically performs well.

**ML gate:** An XGBoost model trained on historical features (momentum, volatility, regime,
strategy score) gives a confidence score. Signals below the threshold are held back.

**Position sizing:** Kelly criterion, adjusted for volatility and correlation with
other open positions.

**Hold period:** 20 trading days (~4 calendar weeks). Early exits on stop-loss,
take-profit, or signal reversal.
""")

    with col_r:
        st.markdown("### Limitations")
        st.markdown("""
This screener is a **learning project**, not a professional trading system.
Be aware of its limitations:

- **Small sample size** — only ~100 closed trades across all runs. Not enough to be statistically definitive.
- **Free data** — uses yfinance (Yahoo Finance). Data can have gaps, errors, and delays.
- **No fundamental data** — purely technical/statistical signals. Ignores earnings, valuation, news.
- **Paper trading** — results are simulated, not real. Slippage, brokerage, and taxes are not modelled.
- **Single-stock focus** — no ETF or portfolio-level risk management.
- **Overfitting risk** — despite walk-forward testing, strategies may not generalise to future markets.
- **ASX/JPX data quality** — international data from yfinance is less reliable than US data.
""")

        st.markdown("---")
        st.markdown("### Legal Disclaimer")
        st.markdown(
            "<div class='disclaimer-box' style='color:#d29922;font-size:13px;line-height:1.6'>"
            "<b>IMPORTANT: This website is for educational and informational purposes only.</b><br><br>"
            "Nothing on this website constitutes financial advice, investment advice, trading advice, "
            "or any other sort of advice. You should not treat any of the website's content as such.<br><br>"
            "The screener signals, rankings, and track record shown on this site are generated by "
            "algorithmic models and are provided for informational purposes only. They do not "
            "represent recommendations to buy, sell, or hold any security.<br><br>"
            "Past performance of any strategy or signal does not guarantee future results. "
            "Investing involves risk, including the possible loss of principal.<br><br>"
            "Always conduct your own independent research and consider seeking advice from a "
            "licensed financial professional before making any investment decisions.<br><br>"
            "The author of this website is not a licensed financial advisor and makes no "
            "representations as to the accuracy, completeness, or suitability of the information "
            "contained herein."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Contact")
        st.markdown("""
Built by a UQ student learning quant trading.

- GitHub: [GoofyisDAWG](https://github.com/GoofyisDAWG)
- Questions or feedback: open an issue on GitHub
""")


