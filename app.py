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

STRATEGY_PLAIN_JA = {
    "MA Crossover":      "トレンドフォロー — 短期平均が長期平均を上抜けた時に買い",
    "RSI":               "平均回帰 — RSIが売られすぎ（30未満）の時に買い",
    "RSI Divergence":    "モメンタム — 価格は下落してもRSIが高水準を維持している時に買い",
    "MACD":              "トレンドモメンタム — 速い移動平均が遅い移動平均を上抜けた時に買い",
    "Bollinger Bands":   "平均回帰 — 価格がボリンジャーバンドの下限に触れた時に買い",
    "Mean Reversion":    "統計的 — 価格が平均から大きく下方乖離した時に買い",
    "Stochastic":        "モメンタム — ストキャスティクスが売られすぎを示した時に買い",
    "Donchian":          "ブレイクアウト — 価格が直近高値レンジを上抜けた時に買い",
    "ADX":               "トレンド強度 — トレンドが強く明確な時のみトレード",
    "Supertrend":        "トレンドフォロー — ボラティリティを使ってトレンド方向を定義",
    "Ichimoku":          "複合シグナル — 価格・時間・モメンタムを組み合わせた日本発のシステム",
    "Volume Breakout":   "出来高確認ブレイクアウト — 価格と出来高が同時に急増",
    "Relative Strength": "相対モメンタム — 市場指数を上回っているアウトパフォーム銘柄",
    "BB Squeeze":        "ボラティリティ拡大 — 低ボラ収縮後のブレイクアウトを狙う",
    "Gap Momentum":      "ギャップ + 出来高 — 高出来高を伴うギャップアップ銘柄を買い",
}

RUN_CONFIGS = {
    1:  "Base momentum",
    2:  "Vol filter",
    3:  "ML gate",
    4:  "Wide stop + no MACD",
    5:  "All ideas + day-10 cut",
    6:  "Tiered floor",
    7:  "All ideas + tiered floor",
    8:  "New strategies baseline",
    9:  "Wide stop + no MACD + tiered floor",
    10: "All ideas + tiered floor (new)",
    11: "Block MACD+BB, ML≥80, tiered floor",
    12: "Fund gate: tight quant + fundamentals",
    13: "Fund gate: medium quant + fundamentals",
    14: "Fund gate only (baseline quant)",
    15: "Fund gate ≥6/8 checks",
    16: "Fund gate + β≤1.0",
    17: "Block MACD+RSI-Div+RS, wide stop, tiered floor",
    18: "Block all 4 losers + ML≥80 + tiered floor",
    19: "Vol confirmation: R17 + vol ≥ 20d avg",
    20: "Winners only: block 4 losers, ML≥75, tiered floor",
    21: "Hold test: baseline R1 config, 10-day max hold",
    22: "RSI(2) + Proven Only: Wide Stop, ML≥70, No Underperformers",
    23: "Pure Proven Trio: RSI + MA-Cross + Mean Reversion Only, ML≥70",
    24: "Proven Trio + Fund Gate: 3 Winners + Fundamentals, ML≥70",
    25: "High Conviction ML: All Strategies, ML≥85, Wide Stop",
    26: "Volume Surge + Proven Winners: Block 4 Losers + Vol≥1.5×, ML≥75",
    27: "All-In Premium: Proven Trio + Fund Gate + Vol≥1.0× + ML≥80",
    28: "Phase 10A: Proven Trio + Tight Pullback 5% + RSI-65 Gate + 15d Cooldown",
    29: "Phase 10B: All Strategies ML≥70 + Breakout MA20 Confirmation",
    30: "Phase 10C: Full Combo — Proven Trio + All Phase 10 Filters",
    31: "Phase 10D: Premium + Phase 10 — R27 + Strict Entry Quality",
    32: "Phase 11A: Tight Trailing — Trigger +8%, Floor +3%",
    33: "Phase 11B: Aggressive TP Only — Lock In at +10%, No Trailing",
    34: "Phase 11C: Signal-Reversal Only — No TP, No Trailing",
    35: "Phase 11D: Best-of-Both — +10% TP + Tight Trailing 8%/3%",
    36: "Phase 11E: ATR Trailing — Floor = Peak − 2×ATR%, Trigger +5%",
    37: "⚡ Speculative Momentum — Quantum/Space/AI/Nuclear, RSI+MA, Wide Stop 20%",
    # Phase 12: Strategy Family Isolation
    38: "Family A: Trend Following Only — MA Crossover, ADX, Supertrend, Ichimoku",
    39: "Family B: Momentum/Breakout Only — MACD, Donchian, Vol Break, BB Squeeze, Gap Mom",
    40: "Family C: Mean Reversion Only — RSI, BB, Mean Rev, Stochastic, RSI(2)",
    41: "Family D: Cross-Family — MA Crossover + RSI + Mean Rev + Vol Breakout, ML≥70",
    42: "US Only: Proven Trio, US Market Filter, ML≥70",
    # Phase 12: Earnings Blackout & Quality Gates
    43: "Earnings Blackout: Baseline R1 + Skip 5d Before Earnings",
    44: "Earnings Blackout: Proven Trio + Blackout, ML≥70",
    45: "Earnings Blackout: All-In Premium R27 + Blackout",
    46: "Extreme Conviction: All Strategies, ML≥90, Wide Stop",
    47: "Full Stack: Proven Trio + Blackout + Volume 1.5×, ML≥70",
}

# Speculative / high-risk tickers — mirrored from paper_trader.py
SPECULATIVE_TICKERS = frozenset({
    "IONQ", "RGTI", "QBTS", "JOBY", "ACHR", "RKLB",
    "AI", "SOUN", "OKLO", "SMR", "LUNR",
})


# ── data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_screener_universe() -> tuple[pd.DataFrame, str, str]:
    """Load all stocks from the latest screener run (US + ASX + JPX sheets).

    Prefers screener_latest.json (written by sync_to_website.py) over xlsx files.
    Falls back to xlsx if JSON doesn't exist (local dev without sync).
    """
    # ── fast path: read pre-exported JSON ─────────────────────────────────────
    json_path = SCREENER_DIR / "screener_latest.json"
    date_path = SCREENER_DIR / "screener_latest_date.txt"
    if json_path.exists():
        try:
            df = pd.read_json(json_path, orient="records")
            df = df.dropna(how="all")
            run_date = date_path.read_text().strip() if date_path.exists() else ""
            if not df.empty:
                now = datetime.now()
                try:
                    rd = datetime.strptime(run_date, "%Y-%m-%d")
                    delta_h = (now - rd).total_seconds() / 3600
                    hours_ago = f"{delta_h:.0f}h ago" if delta_h < 48 else f"{int(delta_h/24)}d ago"
                except Exception:
                    hours_ago = ""
                return df, run_date, hours_ago
        except Exception:
            pass

    # ── fallback: read from xlsx (local dev) ──────────────────────────────────
    files = [f for f in SCREENER_DIR.glob("Goofy_Phase8*.xlsx")
             if not f.name.startswith("~")]
    if not files:
        return pd.DataFrame(), "", ""

    dated = []
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})_Run(\d+)", f.name)
        if m:
            dated.append((m.group(1), int(m.group(2)), f))
    if not dated:
        latest = sorted(files)[-1]
    else:
        most_recent_date = max(d for d, _, _ in dated)
        same_day = [(n, f) for d, n, f in dated if d == most_recent_date]
        same_day.sort(key=lambda x: x[0])
        latest = same_day[0][1]

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
    def fix_nan(s):
        s = re.sub(r'\bNaN\b', 'null', s)
        s = re.sub(r'\bInfinity\b', 'null', s)
        return s
    rows = []
    for run in range(1, 48):
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


def load_open_positions() -> pd.DataFrame:
    """Load all open trades with current unrealised P&L."""
    def fix_nan(s):
        s = re.sub(r'\bNaN\b', 'null', s)
        s = re.sub(r'\bInfinity\b', 'null', s)
        return s
    rows = []
    for run in range(1, 48):
        p = TRADES_DIR / f"run{run}_trades_log.json"
        if not p.exists():
            continue
        try:
            data = json.loads(fix_nan(p.read_text()))
        except Exception:
            continue
        for t in data.get("open", []):
            pnl = t.get("unrealised_pnl_pct")
            if pnl is None:
                continue
            rows.append({"run": run, "asset": t.get("asset", "?"), "pnl_pct": float(pnl)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400)
def _load_price_history() -> dict:
    """Load pre-computed OHLCV (written nightly by sync_to_website.py on private Mac)."""
    p = SCREENER_DIR / "price_history.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        data.pop("updated", None)
        return data
    except Exception:
        return {}


def load_price_data(ticker: str) -> pd.DataFrame:
    """Return pre-computed OHLCV DataFrame for ticker, or empty if not yet cached."""
    d = _load_price_history().get(ticker)
    if not d:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(
            {"Open": d["o"], "High": d["h"], "Low": d["l"], "Close": d["c"], "Volume": d["v"]},
            index=pd.to_datetime(d["d"]).tz_localize("UTC"),
        )
        df.index.name = "Date"
        return df
    except Exception:
        return pd.DataFrame()


def load_market_regime() -> dict:
    """Load pre-computed VIX/SPY market regime (written nightly by sync_to_website.py)."""
    p = SCREENER_DIR / "market_regime.json"
    if not p.exists():
        return {"status": "unknown"}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"status": "unknown"}


@st.cache_data(ttl=3600)
def _fetch_fund_chart(ticker: str) -> pd.DataFrame:
    """Up to 10-year daily price data for the Fundamental Rankings detail panel."""
    try:
        df = yf.Ticker(ticker).history(period="10y", interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError("empty")
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def _build_fund_chart(df: pd.DataFrame, ticker: str, period_label: str = "2Y") -> "go.Figure":
    from plotly.subplots import make_subplots
    close  = df["Close"].squeeze()
    open_  = df["Open"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    vol    = df["Volume"].squeeze() if "Volume" in df.columns else None

    n_rows = 2 if vol is not None else 1
    heights = [0.72, 0.28] if n_rows == 2 else [1.0]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        row_heights=heights, vertical_spacing=0.04)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=open_, high=high, low=low, close=close,
        increasing_line_color="#3fb950", decreasing_line_color="#f85149",
        name="Price", showlegend=True,
    ), row=1, col=1)

    # MA 20 + MA 50
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color="#58a6ff", width=1.5),
                             name="MA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma50, line=dict(color="#d29922", width=1.5),
                             name="MA 50"), row=1, col=1)

    # Volume bars
    if vol is not None:
        vol_clrs = ["#3fb950" if c >= o else "#f85149"
                    for c, o in zip(close, open_)]
        fig.add_trace(go.Bar(x=df.index, y=vol, marker_color=vol_clrs,
                             opacity=0.55, name="Volume"), row=2, col=1)

    last    = float(close.iloc[-1])
    prev    = float(close.iloc[-2]) if len(close) > 1 else last
    chg     = last - prev
    chgp    = chg / prev * 100 if prev else 0
    clr     = "#3fb950" if chg >= 0 else "#f85149"
    hi2y    = float(high.max())
    lo2y    = float(low.min())
    pct_hi  = (last / hi2y - 1) * 100
    pct_clr = "#3fb950" if pct_hi > -10 else "#d29922"

    fig.update_layout(
        height=400,
        autosize=True,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font_size=10),
        margin=dict(l=45, r=8, t=55, b=15),
        title=dict(
            text=(f"<b>{ticker}</b>  "
                  f"<span style='color:{clr}'>{last:.2f}  {chg:+.2f}  ({chgp:+.2f}%)</span>"
                  f"  <span style='color:#8b949e;font-size:12px'>"
                  f"{period_label} High: {hi2y:.2f}  ·  {period_label} Low: {lo2y:.2f}"
                  f"  ·  From High: <span style='color:{pct_clr}'>"
                  f"{pct_hi:.1f}%</span></span>"),
            font_size=14,
        ),
    )
    for r in range(1, n_rows + 1):
        fig.update_xaxes(gridcolor="#21262d", row=r, col=1)
        fig.update_yaxes(gridcolor="#21262d", row=r, col=1)
    if n_rows == 2:
        fig.update_yaxes(title_text="Vol", title_font_size=9, row=2, col=1)

    return fig


ETF_TICKERS = {"SPY","QQQ","GLD","TLT","IWM","STW.AX","IOZ.AX","VAS.AX","1321.T","1306.T"}

# ── asset class classification ─────────────────────────────────────────────────
ASSET_CLASS_MAP = {
    # Gold & precious metals
    "GLD":"Gold", "IAU":"Gold", "GOLD":"Gold", "GDX":"Gold",
    "GOLD.AX":"Gold", "GDX.AX":"Gold",
    "SLV":"Commodities", "SIVR":"Commodities",
    # Broad commodities
    "DJP":"Commodities", "GSG":"Commodities", "PDBC":"Commodities",
    # US bonds
    "TLT":"Bonds", "AGG":"Bonds", "BND":"Bonds", "IEF":"Bonds",
    "SHY":"Bonds", "LQD":"Bonds", "GOVT":"Bonds", "VCIT":"Bonds",
    "VCSH":"Bonds", "MUB":"Bonds", "HYG":"Bonds", "JNK":"Bonds",
    "SGOV":"Cash Equivalent", "SHV":"Cash Equivalent", "BIL":"Cash Equivalent",
    # ASX bonds
    "VAF.AX":"Bonds", "IAF.AX":"Bonds", "BOND.AX":"Bonds",
    "VGB.AX":"Bonds", "RGB.AX":"Bonds",
    # Broad market ETFs
    "SPY":"Broad Market ETF", "QQQ":"Broad Market ETF", "IWM":"Broad Market ETF",
    "VTI":"Broad Market ETF", "SCHB":"Broad Market ETF", "IVV":"Broad Market ETF",
    "STW.AX":"Broad Market ETF", "IOZ.AX":"Broad Market ETF",
    "VAS.AX":"Broad Market ETF", "A200.AX":"Broad Market ETF",
    "1321.T":"Broad Market ETF", "1306.T":"Broad Market ETF",
    # Real estate
    "VNQ":"Real Estate", "SCHH":"Real Estate",
}

ASSET_CLASS_ICONS = {
    "Gold":             "🥇",
    "Bonds":            "📜",
    "Cash Equivalent":  "💵",
    "Commodities":      "🛢️",
    "Broad Market ETF": "🌐",
    "Real Estate":      "🏠",
    "Stock":            "📈",
    "ETF":              "📦",
}

def classify_asset(ticker: str, info: dict) -> str:
    """Return asset class label for a ticker."""
    if ticker in ASSET_CLASS_MAP:
        return ASSET_CLASS_MAP[ticker]
    quote_type = info.get("quoteType", "")
    if quote_type == "ETF":
        name = (info.get("longName") or info.get("shortName") or "").lower()
        cat  = (info.get("category") or "").lower()
        if any(w in name + cat for w in ["bond","fixed income","treasury","credit","debt"]):
            return "Bonds"
        if any(w in name + cat for w in ["gold","silver","precious","metal","commodity","commodit"]):
            return "Gold" if "gold" in name + cat else "Commodities"
        if any(w in name + cat for w in ["real estate","reit","property"]):
            return "Real Estate"
        return "Broad Market ETF"
    return "Stock"


@st.cache_data(ttl=600)
def _load_fund_cache_raw() -> dict:
    """Return the raw fundamentals_cache.json as {ticker: dict}."""
    for p in [
        Path("screener_output/fundamentals_cache.json"),
        Path(__file__).parent / "screener_output" / "fundamentals_cache.json",
    ]:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


@st.cache_data(ttl=3600)
def fetch_stock_health(ticker: str) -> dict:
    """Fetch fundamental health metrics for the Portfolio Health Check page."""
    ticker = ticker.strip().upper()
    if ticker in ETF_TICKERS:
        asset_class = classify_asset(ticker, {})
        return {
            "ticker": ticker, "is_etf": True, "error": False,
            "name": ticker, "sector": "ETF / Index Fund",
            "asset_class": asset_class,
            "checks": [], "score": 6, "max": 6,
        }

    # fast_info is reliable on cloud; use it to verify ticker + get price data
    current_price = year_high = year_low = year_chg = None
    try:
        fi = yf.Ticker(ticker).fast_info
        current_price = fi.last_price
        year_high     = fi.year_high
        year_low      = fi.year_low
        _yc           = fi.year_change
        # guard against NaN (some tickers return nan for year_change)
        year_chg = _yc if (_yc is not None and _yc == _yc) else None
        if not current_price:
            return {"ticker": ticker, "error": True, "name": ticker}
    except Exception:
        return {"ticker": ticker, "error": True, "name": ticker}

    # ── prefer screener's fundamentals_cache.json over live yfinance .info ──
    # The cache is populated by the screener and is more reliable on cloud.
    info = {}
    cached = _load_fund_cache_raw().get(ticker, {})
    if cached and not cached.get("error") and not cached.get("is_etf"):
        # Map cache field names → yfinance .info key names used below
        info = {
            "trailingPE":         cached.get("pe") or cached.get("forward_pe"),
            "revenueGrowth":      cached.get("revenue_growth"),   # decimal
            "debtToEquity":       cached.get("debt_equity"),      # percentage pts
            "freeCashflow":       cached.get("free_cashflow"),    # absolute $
            "profitMargins":      cached.get("profit_margin"),    # decimal
            "recommendationMean": cached.get("analyst_rating"),   # 1–5
            "recommendationKey":  cached.get("analyst_label", ""),
            "returnOnEquity":     cached.get("return_on_equity"), # decimal
            "longName":           cached.get("company_name", ticker),
            "sector":             cached.get("sector", ""),
            "marketCap":          cached.get("market_cap"),
            "dividendYield":      cached.get("dividend_yield"),
        }
        info = {k: v for k, v in info.items() if v is not None}

    if not info:
        # Fall back to live yfinance .info (slower, less reliable on cloud)
        try:
            raw = yf.Ticker(ticker).info
            if isinstance(raw, dict) and len(raw) > 5:
                info = raw
        except Exception:
            pass

    checks = []

    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe is not None:
        try:
            pe_f = float(pe)
            ok = pe_f < 40
            checks.append({
                "label": "Valuation (P/E ratio)",
                "pass": ok,
                "detail": f"P/E = {pe_f:.1f} — {'reasonable' if ok else 'very high, may be overvalued'}",
                "plain": "Is the stock priced reasonably compared to its earnings?",
            })
        except Exception:
            pass

    rg = info.get("revenueGrowth")
    if rg is not None:
        try:
            rg_f = float(rg)
            ok = rg_f > -0.10
            checks.append({
                "label": "Revenue growth",
                "pass": ok,
                "detail": f"{rg_f*100:+.1f}% year-on-year — {'growing' if rg_f > 0 else 'declining'}",
                "plain": "Is the company's revenue growing over the past year?",
            })
        except Exception:
            pass

    dte = info.get("debtToEquity")
    if dte is not None:
        try:
            dte_f = float(dte)
            ok = dte_f < 300
            checks.append({
                "label": "Debt level",
                "pass": ok,
                "detail": f"Debt/Equity = {dte_f:.0f}% — {'manageable' if ok else 'high debt load'}",
                "plain": "Does the company have manageable levels of debt?",
            })
        except Exception:
            pass

    fcf = info.get("freeCashflow")
    if fcf is not None:
        try:
            fcf_f = float(fcf)
            ok = fcf_f > 0
            fcf_b = fcf_f / 1e9
            checks.append({
                "label": "Free cash flow",
                "pass": ok,
                "detail": f"${fcf_b:+.2f}B — {'generating cash' if ok else 'burning cash'}",
                "plain": "Is the company actually generating real cash from its business?",
            })
        except Exception:
            pass

    pm = info.get("profitMargins")
    if pm is not None:
        try:
            pm_f = float(pm)
            ok = pm_f > -0.05
            checks.append({
                "label": "Profitability",
                "pass": ok,
                "detail": f"Profit margin = {pm_f*100:.1f}% — {'profitable' if pm_f > 0 else 'loss-making'}",
                "plain": "Is the company profitable (making more than it spends)?",
            })
        except Exception:
            pass

    rec = info.get("recommendationMean")
    rec_key = info.get("recommendationKey", "")
    if rec is not None:
        try:
            rec_f = float(rec)
            ok = rec_f < 4.0
            checks.append({
                "label": "Analyst view",
                "pass": ok,
                "detail": f"Consensus = {rec_key.replace('_',' ').title()} ({rec_f:.1f}/5) — {'positive' if ok else 'analysts lean sell'}",
                "plain": "What do professional analysts think about this stock?",
            })
        except Exception:
            pass

    roe = info.get("returnOnEquity")
    if roe is not None:
        try:
            roe_f = float(roe)
            ok = roe_f > 0.08
            checks.append({
                "label": "Return on equity (ROE)",
                "pass": ok,
                "detail": f"ROE = {roe_f*100:.1f}% — {'strong' if roe_f > 0.15 else 'acceptable' if ok else 'below average'}",
                "plain": "Is the company generating good returns from shareholders' money?",
            })
        except Exception:
            pass

    passed = sum(1 for c in checks if c["pass"])
    total  = len(checks)

    if total < 2:
        overall = "limited_data"
    elif passed >= total * 0.67:
        overall = "healthy"
    elif passed >= total * 0.40:
        overall = "caution"
    else:
        overall = "concern"

    return {
        "ticker":        ticker,
        "error":         False,
        "is_etf":        False,
        "asset_class":   classify_asset(ticker, info),
        "name":          info.get("longName", ticker),
        "sector":        info.get("sector", ""),
        "industry":      info.get("industry", ""),
        "currency":      info.get("currency", ""),
        "market_cap":    info.get("marketCap"),
        "current_price": current_price,
        "year_high":     year_high,
        "year_low":      year_low,
        "year_chg":      year_chg,
        "dividend_yield":info.get("dividendYield"),
        "checks":        checks,
        "score":         passed,
        "max":           total,
        "overall":       overall,
    }


# ── translations ──────────────────────────────────────────────────────────────

_TR = {
    "en": {
        "fund_title":    "🌏 Fundamental Rankings",
        "fund_sub":      "All {n} stocks ranked by financial health — no jargon, just the numbers that matter.",
        "filter_market": "Filter by market",
        "all_markets":   "🌍 All",
        "sort_by":       "Sort by",
        "sort_score":    "Health Score",
        "sort_pe":       "P/E (lowest)",
        "sort_roe":      "ROE (highest)",
        "sort_div":      "Dividend Yield",
        "strong":   "Strong",  "watch": "Watch",  "weak": "Weak",
        "limited":  "Limited", "etf":   "ETF",
        "n_stocks": "stocks",
        "lbl_pe":   "P/E",   "lbl_roe": "ROE",    "lbl_div":     "Dividend",
        "lbl_debt": "Debt",  "lbl_rev": "Revenue", "lbl_analyst": "Analysts",
        "lbl_fcf":  "Cash Flow", "lbl_beta": "Beta", "lbl_cr": "Curr. Ratio",
        "lbl_pb":   "P/Book", "lbl_gm": "Gross Margin", "lbl_om": "Op. Margin",
        "pass_lbl": "✅ Strong", "watch_lbl": "⚠️ Watch",
        "fail_lbl": "❌ Weak",  "etf_lbl":   "📦 ETF",  "na_lbl": "❓ Limited",
        "no_cache": "No fundamental data yet — run the screener first.",
        "last_upd": "Data from",
        "tip_pe":   "Stock price ÷ earnings per share. Lower = cheaper. Under 20 is often good value.",
        "tip_roe":  "Profit as % of shareholders' money. Over 10% is healthy. Warren Buffett's favourite metric.",
        "tip_div":  "Annual dividend as % of stock price. Higher = more passive income.",
        "tip_rev":  "Year-on-year revenue change. Positive = growing business.",
        "tip_debt": "Total debt ÷ equity. Under 200% is generally safe.",
        "tip_fcf":  "Real cash generated by the business after expenses. Positive = self-funding.",
        "tip_anl":  "Average analyst rating. Ranges from Strong Buy (1) to Sell (5).",
        "tip_beta": "How much the stock moves vs the market. Under 1.0 = less volatile.",
        "tip_cr":   "Current assets ÷ current liabilities. Over 1.0 = can pay short-term bills.",
        "guide_title": "❓ How to use this page",
        "guide_body": (
            "**The simplest way to find financially healthy stocks — no charts needed.**\n\n"
            "All stocks are scored on **8 fundamental checks** and ranked highest to lowest. "
            "Use the filters to focus on your market, sort by the metric that matters most to you, "
            "and use this list to build a shortlist — then head to **Screener Rankings** to see which "
            "of these also have a technical signal today.\n\n"
            "**The 8 checks:** P/E ratio · Revenue growth · Debt level · Cash flow · "
            "Profitability · Analyst view · Return on equity (ROE) · Short-term safety"
        ),
        # ── home ──
        "home_sub":       "Free quantitative stock screener — United States · Australia · Japan",
        "home_expander":  "👋 New here? Start here — what is this and how do I use it?",
        "stat_screened":  "Stocks screened",   "stat_screened_s": "US + ASX + JPX",
        "stat_signals":   "Buy signals today", "stat_signals_s":  "across {n} markets",
        "stat_markets":   "Markets covered",   "stat_markets_s":  "US · ASX · JPX",
        "stat_strats":    "Strategies used",   "stat_strats_s":   "Walk-forward validated",
        "how_it_works":   "### How it works",
        "top_picks":      "### Today's top picks",
        "last_updated":   "Last updated:",
        "no_data_home":   "No screener data found. Run goofy_screener_phase8.py first.",
        # ── sidebar ──
        "exp_level":      "**Your experience level**",
        "tip_beginner":   "💡 <b>New here?</b> Start with <b>Fundamental Rankings</b> — see all stocks ranked by financial health, no jargon. Then use <b>Portfolio Health Check</b> to analyse stocks you already own.",
        "tip_inter":      "💡 <b>Tip:</b> Check <b>Screener Rankings</b> for today's signals, then use <b>Stock Chart</b> to see the strategy driving each signal.",
        "tip_advanced":   "💡 <b>Tip:</b> <b>Track Record</b> shows all 37 live paper trade runs. Compare win rates across runs to see which config is outperforming.",
        "footer_note":    "Data: yfinance · Not financial advice.",
        # ── screener rankings ──
        "sr_title":       "### 📊 Screener Rankings",
        "sr_expander":    "❓ How to read this page",
        "sr_mkt":         "Market",   "sr_tier": "Tier",
        "sr_show_all":    "Show all (incl. PASS)",
        "sr_buy":         "BUY signals ({n})",
        "sr_watch":       "Watch / Pass ({n})",
        "sr_hidden":      "+ {n} WATCH/PASS signals hidden. Toggle 'Show all' to see them.",
        "sr_no_signals":  "No signals match the current filters.",
        "sr_disclaimer":  "Signals are generated by algorithmic models and are NOT investment advice. Always conduct your own research.",
        # ── track record ──
        "tr_title":       "### 🏆 Track Record",
        "tr_sub":         "Real paper trading results — all trades shown including losses. No cherry-picking.",
        "tr_expander":    "❓ How to read this page",
        "tr_no_trades":   "No closed trades yet.",
        "tr_m1":  "Total trades",   "tr_m1s": "all markets",
        "tr_m2":  "Win rate",
        "tr_m3":  "Avg P&L",       "tr_m3s": "per trade",
        "tr_m4":  "Avg winner",    "tr_m4s": "when right",
        "tr_m5":  "Avg loser",     "tr_m5s": "when wrong",
        "tr_m6":  "Stop-losses hit",
        "tr_equity":   "#### Cumulative P&L over time",
        "tr_eq_sub":   "Each step = one closed trade, sorted by exit date.",
        "tr_strategy": "#### By strategy",
        "tr_exit":     "#### How trades ended",
        "tr_market":   "#### By market",
        "tr_recent":   "#### Recent closed trades",
        "tr_positive": "🟢 Positive",  "tr_negative": "🔴 Negative",
        "tr_col_strat":"Strategy", "tr_col_trades":"Trades",
        "tr_col_wr":   "Win rate %", "tr_col_pnl": "Avg P&L %", "tr_col_res": "Result",
        "tr_trades":   "trades",   "tr_avg": "avg",  "tr_wr": "win rate",
        "tr_disclaimer": "Past performance does not guarantee future results. These are paper trading results — no real money was used. This is not financial advice.",
        "tr_run_filter": "Filter by run",
        "tr_run_all":    "All runs combined",
        "tr_per_run":    "#### Per-run performance",
        "tr_per_run_sub":"Each row is one run config. Click a run to see what makes it different.",
        "tr_col_run":    "Run", "tr_col_cfg": "Config", "tr_col_pf": "Profit factor",
        "tr_snap":       "📸 If all positions closed right now",
        "tr_snap_sub":   "Simulation combining closed trade history with current unrealised P&L on open positions. Not a real return — just a snapshot of where we stand today.",
        "tr_col_closed": "Closed", "tr_col_open_n": "Open",
        "tr_col_combo_wr": "Win rate", "tr_col_combo_avg": "Avg P&L", "tr_col_total": "Total P&L",
        "tr_sim_header":  "💰 Investment Simulator",
        "tr_sim_label":   "If I had invested…",
        "tr_sim_becomes": "would have become",
        "tr_sim_disc":    "Sequential compounding — each signal reinvests the previous result. Trades overlap in reality so this overstates a bit, but the direction is right.",
        # ── stock chart ──
        "sc_title":    "## 📈 Stock Chart",
        "sc_expander": "❓ How to read this page",
        "sc_desc":     "Pick any stock from today's screener universe and see its price chart with the strategy indicators that drove the signal.",
        "sc_market":   "Market",
        # stock chart filters & selection
        "sc_select_stock":   "Select a stock",
        "sc_show":           "Show",
        "sc_show_all_opt":   "All",
        "sc_show_buy":       "BUY signals only",
        "sc_show_watch":     "WATCH only",
        # stock chart errors
        "sc_no_data_loaded": "No screener data loaded yet. Run the screener first.",
        "sc_no_asset_col":   "Could not find an Asset/Ticker column in screener data.",
        "sc_no_match":       "No stocks match those filters.",
        # signal card labels
        "sc_sig_signal":     "Signal",
        "sc_sig_strategy":   "Strategy",
        "sc_sig_ml":         "ML Score",
        "sc_sig_size":       "Suggested Size",
        # chart description + stats
        "sc_what_to_look":   "What to look for on the chart:",
        "sc_strat_unknown":  "Quantitative signal based on price history.",
        "sc_price_err":      "Could not load price data for **{ticker}**. Try a different ticker or check your connection.",
        "sc_chart_disc":     "Chart data: yfinance (up to 2 years daily). Indicators are illustrative. Not financial advice — always do your own research.",
        "sc_period_label":   "Chart period",
        "sc_stat_last":      "Last Price",
        "sc_stat_ma20":      "MA 20",
        "sc_stat_rsi14":     "RSI 14",
        "sc_stat_6mhigh":    "6M High",
        "sc_stat_pct_high":  "% From High",
        # screener rankings card labels
        "sr_card_strategy":  "Strategy",
        "sr_card_confidence":"Confidence",
        "sr_card_trend":     "Trend",
        "sr_card_size":      "Suggested size",
        "sr_spec_badge":     "⚡ High Risk",
        "sr_spec_tooltip":   "Speculative — pre-profitable, high volatility. Not traded by main runs.",
        # ── about ──
        "ab_title":    "### About Goofy Screener",
        "ab_method":   "### Methodology",
        "ab_limits":   "### Limitations",
        "ab_legal":    "### Legal Disclaimer",
        # ── portfolio health check ──
        "phc_title":         "## 🔍 Portfolio Health Check",
        "phc_expander":      "❓ How to use this page",
        "phc_guide": (
            "Enter the tickers of stocks you own or are interested in, separated by commas. "
            "We'll check each one against 6 fundamental health indicators and give you a plain-English summary.<br><br>"
            "<b>Examples:</b> &nbsp; US stocks: <code>AAPL, MSFT, JPM</code> &nbsp;·&nbsp; "
            "ASX stocks: <code>CBA.AX, BHP.AX, WBC.AX</code> &nbsp;·&nbsp; "
            "Japanese stocks: <code>7203.T, 6758.T</code>"
        ),
        "phc_placeholder":   "e.g. AAPL, CBA.AX, 7203.T",
        "phc_max_warning":   "Maximum 15 stocks at once to keep loading fast.",
        "phc_checking":      "Checking **{n} stock(s)**...",
        "phc_pick_label":    "Pick stocks from our universe",
        "phc_or_custom":     "Or add any ticker not in the list above (comma-separated)",
        "phc_error":         "could not load data. Check the ticker is correct (e.g. <code>CBA.AX</code> not <code>CBA</code> for ASX).",
        "phc_etf_msg":       "ETFs don't have individual company fundamentals — they track a basket of stocks. Generally a safer, diversified option for beginners.",
        "phc_healthy_label": "✅ Looks Healthy",
        "phc_healthy_sum":   "This company passes most of our fundamental checks.",
        "phc_caution_label": "⚠️  Worth Watching",
        "phc_caution_sum":   "Mixed results — some strengths, some concerns. Research further.",
        "phc_concern_label": "❌ Has Concerns",
        "phc_concern_sum":   "This company fails several fundamental checks. Proceed with caution.",
        "phc_limited_label": "❓ Limited Data",
        "phc_limited_sum":   "Not enough data to give a reliable score (common for some international stocks).",
        "phc_mcap":          "Market cap: ${n}B",
        "phc_checks_passed": "{score}/{maxs} checks passed",
        "phc_fundamentals":  "{score}/{maxs} fundamentals",
        "phc_check_pe":      "Valuation (P/E ratio)",
        "phc_check_rev":     "Revenue growth",
        "phc_check_debt":    "Debt level",
        "phc_check_fcf":     "Free cash flow",
        "phc_check_prof":    "Profitability",
        "phc_check_analyst": "Analyst view",
        "phc_plain_pe":      "Is the stock priced reasonably compared to its earnings?",
        "phc_plain_rev":     "Is the company's revenue growing over the past year?",
        "phc_plain_debt":    "Does the company have manageable levels of debt?",
        "phc_plain_fcf":     "Is the company actually generating real cash from its business?",
        "phc_plain_prof":    "Is the company profitable (making more than it spends)?",
        "phc_plain_analyst": "What do professional analysts think about this stock?",
        "phc_empty":         "Type your stock tickers above to get started",
        "phc_empty_sub":     "e.g. <code>CBA.AX, AAPL, BHP.AX</code>",
        "phc_snapshot":      "### 📊 Your portfolio snapshot",
        "phc_nudge_stocks":  (
            "Your entire portfolio appears to be in individual stocks. "
            "Consider whether you also hold bonds, gold, or cash as a buffer against market downturns. "
            "Most financial advisers suggest keeping at least some exposure to defensive assets."
        ),
        "phc_nudge_mostly":  (
            "You've entered mostly stocks. You can also run this health check on bond or gold ETFs "
            "(e.g. <code>TLT</code> for US bonds, <code>GLD</code> for gold, "
            "<code>VAF.AX</code> for Australian bonds)."
        ),
        "phc_nudge_diverse": "Good — your entered holdings span {n} asset class(es). Diversification across different types of assets helps smooth out volatility.",
        "phc_cant_check": (
            "<b>What this tool can't check (and what to do instead):</b><br><br>"
            "🏦 <b>Savings accounts &amp; term deposits</b> — These are bank products with no ticker. "
            "Factor them in manually when thinking about your total allocation. "
            "If your deposit is paying 4–5%, that's already a solid risk-free return to compare against.<br><br>"
            "🏠 <b>Property</b> — No price feed we can pull. If you own real estate, you likely already have "
            "significant assets outside the stock market.<br><br>"
            "🦺 <b>Superannuation</b> — Your super fund's website will show your current investment mix "
            "(e.g. balanced, growth, conservative). Check that it matches your risk tolerance and age."
        ),
        "phc_how_to_read": (
            "<b>How to read the health check:</b><br><br>"
            "🟢 <b>Healthy</b> — The company passes most of our 6 checks. This means it appears to be profitable, "
            "not over-indebted, and growing. It does NOT mean the stock will go up.<br><br>"
            "🟡 <b>Worth Watching</b> — Mixed results. Look into which checks it failed and why before deciding anything.<br><br>"
            "🔴 <b>Has Concerns</b> — The company shows signs of financial stress. Higher risk. Not necessarily a bad investment "
            "(turnaround stories exist), but you need to understand what you're getting into.<br><br>"
            "<b>The 7 checks:</b> Valuation (P/E ratio) · Revenue growth · Debt level · Free cash flow · Profitability · Analyst consensus · Return on equity (ROE)"
        ),
        "phc_disclaimer": (
            "This health check uses publicly available financial data from Yahoo Finance. "
            "Data may be delayed, incomplete, or inaccurate. This is not financial advice. "
            "Always conduct your own research before investing."
        ),
        # ── portfolio save/load ──
        "phc_save_header":       "💾 Save & load portfolios",
        "phc_save_placeholder":  "Portfolio name (e.g. 'My ASX picks')",
        "phc_save_btn":          "Save",
        "phc_saved_ok":          "Saved as **{name}**",
        "phc_load_header":       "Saved portfolios",
        "phc_load_none":         "No saved portfolios yet",
        "phc_load_btn":          "Load",
        "phc_delete_btn":        "Delete",
        "phc_share_tip":         "💡 Bookmark this URL to return to the same portfolio.",
        # ── home disclaimer ──
        "disclaimer":  "This website is for educational purposes only and does not constitute financial advice. Past performance does not guarantee future results. Always do your own research before investing.",
    },
    "ja": {
        "fund_title":    "🌏 ファンダメンタル ランキング",
        "fund_sub":      "{n}銘柄を財務健全性でランク付け — 専門用語なし、重要な数字だけ。",
        "filter_market": "市場でフィルター",
        "all_markets":   "🌍 全市場",
        "sort_by":       "並び替え",
        "sort_score":    "健全性スコア",
        "sort_pe":       "PER（低い順）",
        "sort_roe":      "ROE（高い順）",
        "sort_div":      "配当利回り",
        "strong":   "良好",   "watch": "注意",   "weak": "懸念",
        "limited":  "データ不足", "etf": "ETF",
        "n_stocks": "銘柄",
        "lbl_pe":   "PER",   "lbl_roe": "ROE",    "lbl_div":     "配当利回り",
        "lbl_debt": "負債比率", "lbl_rev": "売上成長", "lbl_analyst": "アナリスト",
        "lbl_fcf":  "キャッシュフロー", "lbl_beta": "ベータ", "lbl_cr": "流動比率",
        "lbl_pb":   "PBR", "lbl_gm": "売上総利益率", "lbl_om": "営業利益率",
        "pass_lbl": "✅ 良好", "watch_lbl": "⚠️ 注意",
        "fail_lbl": "❌ 懸念", "etf_lbl":   "📦 ETF",  "na_lbl": "❓ データ不足",
        "no_cache": "データなし — スクリーナーを実行してください。",
        "last_upd": "データ更新日",
        "tip_pe":   "株価 ÷ 1株当たり利益。低いほど割安。一般的に20以下が割安。",
        "tip_roe":  "株主資本に対する利益の割合。10%超が健全。バフェットが重視する指標。",
        "tip_div":  "株価に対する年間配当金の割合。高いほど配当収入が多い。",
        "tip_rev":  "前年比の売上成長率。プラスは成長中の事業を意味する。",
        "tip_debt": "総負債を自己資本で割った値。200%未満が一般的に安全。",
        "tip_fcf":  "費用差引後に事業が実際に生み出す現金。プラスは自己資金で運営できることを示す。",
        "tip_anl":  "アナリストの平均評価。1（強い買い）〜5（売り）の範囲。",
        "tip_beta": "市場全体と比較した株価の変動幅。1.0未満は市場より変動が少ない。",
        "tip_cr":   "流動資産 ÷ 流動負債。1.0超は短期債務を支払える状態。",
        "guide_title": "❓ このページの使い方",
        "guide_body": (
            "**チャート不要 — 財務的に健全な銘柄を見つける最もシンプルな方法。**\n\n"
            "全銘柄を**8項目のファンダメンタルチェック**で評価しランク付けしています。 "
            "フィルターで市場を絞り込み、重視する指標で並べ替え、候補リストを作成してください。"
            "その後、**スクリーナーランキング**で今日のシグナルを確認しましょう。\n\n"
            "**8つのチェック:** PER · 売上成長率 · 負債水準 · キャッシュフロー · "
            "収益性 · アナリスト評価 · ROE · 短期安全性（流動比率）"
        ),
        # ── home ──
        "home_sub":       "無料クオンツ株スクリーナー — 米国・オーストラリア・日本",
        "home_expander":  "👋 初めての方へ — これは何？どう使う？",
        "stat_screened":  "スクリーニング銘柄数", "stat_screened_s": "US + ASX + JPX",
        "stat_signals":   "本日の買いシグナル",  "stat_signals_s":  "{n}市場合計",
        "stat_markets":   "対象市場数",          "stat_markets_s":  "US · ASX · JPX",
        "stat_strats":    "使用戦略数",          "stat_strats_s":   "ウォークフォワード検証済み",
        "how_it_works":   "### 仕組み",
        "top_picks":      "### 本日のトップ候補",
        "last_updated":   "最終更新:",
        "no_data_home":   "スクリーナーデータなし。goofy_screener_phase8.py を実行してください。",
        # ── sidebar ──
        "exp_level":      "**あなたのレベル**",
        "tip_beginner":   "💡 <b>初めての方へ：</b>まず<b>ファンダメンタルランキング</b>をチェック — 財務健全性でランク付けされた全銘柄が一目でわかります。次に<b>ポートフォリオ健全性チェック</b>で保有銘柄を分析しましょう。",
        "tip_inter":      "💡 <b>ヒント：</b><b>スクリーナーランキング</b>で本日のシグナルを確認し、<b>株価チャート</b>で各シグナルの背景にある戦略を見てみましょう。",
        "tip_advanced":   "💡 <b>ヒント：</b><b>トラックレコード</b>では16本のライブペーパートレード実績を確認できます。各ランの勝率を比較して、どの設定が優れているか分析しましょう。",
        "footer_note":    "データ: yfinance · 投資アドバイスではありません。",
        # ── screener rankings ──
        "sr_title":       "### 📊 スクリーナー ランキング",
        "sr_expander":    "❓ このページの使い方",
        "sr_mkt":         "市場",  "sr_tier": "ティア",
        "sr_show_all":    "全て表示（PASSも含む）",
        "sr_buy":         "買いシグナル（{n}件）",
        "sr_watch":       "ウォッチ / パス（{n}件）",
        "sr_hidden":      "+ {n}件のWATCH/PASSシグナルを非表示中。「全て表示」で確認できます。",
        "sr_no_signals":  "現在のフィルターに一致するシグナルがありません。",
        "sr_disclaimer":  "シグナルはアルゴリズムモデルによって生成されており、投資アドバイスではありません。必ず自身でリサーチを行ってください。",
        # ── track record ──
        "tr_title":       "### 🏆 トラックレコード",
        "tr_sub":         "実際のペーパートレード結果 — 損失を含む全トレードを公開。",
        "tr_expander":    "❓ このページの使い方",
        "tr_no_trades":   "まだクローズドトレードはありません。",
        "tr_m1":  "総トレード数",   "tr_m1s": "全市場合計",
        "tr_m2":  "勝率",
        "tr_m3":  "平均損益",       "tr_m3s": "1トレードあたり",
        "tr_m4":  "平均利益",       "tr_m4s": "勝ちトレード平均",
        "tr_m5":  "平均損失",       "tr_m5s": "負けトレード平均",
        "tr_m6":  "損切り回数",
        "tr_equity":   "#### 累積損益の推移",
        "tr_eq_sub":   "各ステップ = 1つのクローズドトレード（決済日順）",
        "tr_strategy": "#### 戦略別",
        "tr_exit":     "#### 決済理由",
        "tr_market":   "#### 市場別",
        "tr_recent":   "#### 最近のクローズドトレード",
        "tr_positive": "🟢 プラス",  "tr_negative": "🔴 マイナス",
        "tr_col_strat":"戦略", "tr_col_trades":"取引数",
        "tr_col_wr":   "勝率%", "tr_col_pnl": "平均損益%", "tr_col_res": "結果",
        "tr_trades":   "件",   "tr_avg": "平均",  "tr_wr": "勝率",
        "tr_disclaimer": "過去の実績は将来の成果を保証するものではありません。これらはペーパートレード結果です — 実際の資金は使用していません。投資アドバイスではありません。",
        "tr_run_filter": "ランでフィルター",
        "tr_run_all":    "全ラン（合計）",
        "tr_per_run":    "#### ラン別パフォーマンス",
        "tr_per_run_sub":"各行は1つのランの設定です。",
        "tr_col_run":    "ラン", "tr_col_cfg": "設定", "tr_col_pf": "プロフィットファクター",
        "tr_snap":       "📸 今すぐ全決済した場合",
        "tr_snap_sub":   "クローズ済み取引実績と現在の含み益を合算したシミュレーションです。実際のリターンではなく、現時点でのスナップショットです。",
        "tr_col_closed": "決済済", "tr_col_open_n": "保有中",
        "tr_col_combo_wr": "勝率", "tr_col_combo_avg": "平均損益", "tr_col_total": "合計損益",
        "tr_sim_header":  "💰 投資シミュレーター",
        "tr_sim_label":   "もし投資していたら…",
        "tr_sim_becomes": "の最終価値",
        "tr_sim_disc":    "順次複利モデル — 各シグナルが前の結果を再投資。実際にはトレードが同時進行するため若干過大評価ですが、方向性は正確です。",
        # ── stock chart ──
        "sc_title":    "## 📈 株価チャート",
        "sc_expander": "❓ このページの使い方",
        "sc_desc":     "スクリーナーの任意の銘柄を選択し、シグナルを生成した戦略インジケーターを重ねた株価チャートを確認できます。",
        "sc_market":   "市場",
        # stock chart filters & selection
        "sc_select_stock":   "銘柄を選択",
        "sc_show":           "表示",
        "sc_show_all_opt":   "すべて",
        "sc_show_buy":       "買いシグナルのみ",
        "sc_show_watch":     "ウォッチのみ",
        # stock chart errors
        "sc_no_data_loaded": "スクリーナーデータがありません。先にスクリーナーを実行してください。",
        "sc_no_asset_col":   "スクリーナーデータにAsset/Ticker列が見つかりません。",
        "sc_no_match":       "そのフィルターに一致する銘柄がありません。",
        # signal card labels
        "sc_sig_signal":     "シグナル",
        "sc_sig_strategy":   "戦略",
        "sc_sig_ml":         "ML スコア",
        "sc_sig_size":       "推奨サイズ",
        # chart description + stats
        "sc_what_to_look":   "チャートで確認すべきポイント：",
        "sc_strat_unknown":  "価格履歴に基づくクオンツシグナル。",
        "sc_price_err":      "**{ticker}**の価格データを読み込めませんでした。別のティッカーを試すか、接続を確認してください。",
        "sc_chart_disc":     "チャートデータ: yfinance（最大2年日足）。インジケーターは参考表示です。投資アドバイスではありません。",
        "sc_period_label":   "チャート期間",
        "sc_stat_last":      "直近終値",
        "sc_stat_ma20":      "MA 20",
        "sc_stat_rsi14":     "RSI 14",
        "sc_stat_6mhigh":    "6ヶ月高値",
        "sc_stat_pct_high":  "高値からの乖離率",
        # screener rankings card labels
        "sr_card_strategy":  "戦略",
        "sr_card_confidence":"信頼度",
        "sr_card_trend":     "トレンド",
        "sr_card_size":      "推奨サイズ",
        "sr_spec_badge":     "⚡ 高リスク",
        "sr_spec_tooltip":   "投機的銘柄 — 無収益・高ボラティリティ。メインランでは取引されません。",
        # ── about ──
        "ab_title":    "### Goofy Screenerについて",
        "ab_method":   "### 方法論",
        "ab_limits":   "### 制限事項",
        "ab_legal":    "### 免責事項",
        # ── portfolio health check ──
        "phc_title":         "## 🔍 ポートフォリオ 健全性チェック",
        "phc_expander":      "❓ このページの使い方",
        "phc_guide": (
            "保有中または気になっている銘柄のティッカーをカンマ区切りで入力してください。"
            "6つのファンダメンタル指標をもとに各銘柄の財務健全性を評価します。<br><br>"
            "<b>例：</b> &nbsp; 米国株: <code>AAPL, MSFT, JPM</code> &nbsp;·&nbsp; "
            "ASX株: <code>CBA.AX, BHP.AX, WBC.AX</code> &nbsp;·&nbsp; "
            "日本株: <code>7203.T, 6758.T</code>"
        ),
        "phc_placeholder":   "例: AAPL, CBA.AX, 7203.T",
        "phc_max_warning":   "読み込みを高速に保つため、一度に最大15銘柄までです。",
        "phc_checking":      "**{n}銘柄**をチェック中...",
        "phc_pick_label":    "スクリーナー銘柄から選ぶ",
        "phc_or_custom":     "リストにない銘柄を追加（カンマ区切り）",
        "phc_error":         "データを読み込めませんでした。ティッカーが正しいか確認してください（例: ASX株は <code>CBA</code> ではなく <code>CBA.AX</code>）。",
        "phc_etf_msg":       "ETFには個別企業のファンダメンタルズはありません — 複数銘柄のバスケットを追跡する商品です。初心者には安全で分散された選択肢です。",
        "phc_healthy_label": "✅ 財務健全",
        "phc_healthy_sum":   "ほとんどのファンダメンタルチェックに合格しています。",
        "phc_caution_label": "⚠️ 要注意",
        "phc_caution_sum":   "結果は混在 — 強みと懸念点が共存しています。さらにリサーチしてください。",
        "phc_concern_label": "❌ 問題あり",
        "phc_concern_sum":   "複数のファンダメンタルチェックで不合格。慎重に判断してください。",
        "phc_limited_label": "❓ データ不足",
        "phc_limited_sum":   "信頼性の高いスコアを出すにはデータが不足しています（一部の海外株で一般的）。",
        "phc_mcap":          "時価総額: ${n}B",
        "phc_checks_passed": "{score}/{maxs}項目合格",
        "phc_fundamentals":  "{score}/{maxs}ファンダメンタル",
        "phc_check_pe":      "バリュエーション（PER）",
        "phc_check_rev":     "売上成長率",
        "phc_check_debt":    "負債水準",
        "phc_check_fcf":     "フリーキャッシュフロー",
        "phc_check_prof":    "収益性",
        "phc_check_analyst": "アナリスト評価",
        "phc_plain_pe":      "業績に対して株価は妥当な水準か？",
        "phc_plain_rev":     "過去1年で売上は成長しているか？",
        "phc_plain_debt":    "負債水準は管理可能か？",
        "phc_plain_fcf":     "事業から実際に現金を生み出しているか？",
        "phc_plain_prof":    "収益性はあるか（支出より収入が多いか）？",
        "phc_plain_analyst": "アナリストはこの銘柄をどう見ているか？",
        "phc_empty":         "上にティッカーを入力してスタート",
        "phc_empty_sub":     "例: <code>CBA.AX, AAPL, BHP.AX</code>",
        "phc_snapshot":      "### 📊 ポートフォリオのスナップショット",
        "phc_nudge_stocks":  (
            "入力銘柄はすべて個別株のようです。"
            "市場下落への備えとして、債券・金・現金なども検討してください。"
            "多くのファイナンシャルアドバイザーは守りの資産への一定のエクスポージャーを推奨しています。"
        ),
        "phc_nudge_mostly":  (
            "ほとんどが株式です。"
            "債券ETF（例: <code>TLT</code>：米国債、<code>GLD</code>：金、"
            "<code>VAF.AX</code>：オーストラリア債券）でもヘルスチェックを試してみてください。"
        ),
        "phc_nudge_diverse": "{n}種類のアセットクラスにまたがっています。✅ 異なる資産への分散はボラティリティの平滑化に役立ちます。",
        "phc_cant_check": (
            "<b>このツールで確認できないもの（別途確認方法）：</b><br><br>"
            "🏦 <b>普通預金・定期預金</b> — ティッカーのない銀行商品です。"
            "総資産配分を考える際に手動で組み込んでください。"
            "4〜5%の金利であれば、十分なリスクフリーリターンの基準として比較できます。<br><br>"
            "🏠 <b>不動産</b> — 価格データを取得できません。"
            "不動産を所有している場合、株式市場外に大きな資産があることになります。<br><br>"
            "🦺 <b>スーパーアニュエーション</b> — スーパーファンドのウェブサイトで現在の運用ミックス"
            "（バランス型・グロース型・コンサバティブ型など）を確認してください。"
            "リスク許容度と年齢に合っているか確認しましょう。"
        ),
        "phc_how_to_read": (
            "<b>ヘルスチェックの見方：</b><br><br>"
            "🟢 <b>健全</b> — 6項目のほとんどに合格。収益性があり、過剰債務なく、成長中です。株価が上がるとは限りません。<br><br>"
            "🟡 <b>要注意</b> — 結果は混在。何に不合格で、なぜかを調べてから判断してください。<br><br>"
            "🔴 <b>問題あり</b> — 財務的ストレスの兆候あり。リスクは高め。必ずしも悪い投資ではありませんが、"
            "何に投資しているかを理解することが重要です。<br><br>"
            "<b>6つのチェック：</b> バリュエーション（PER）· 売上成長率 · 負債水準 · フリーキャッシュフロー · 収益性 · アナリスト評価"
        ),
        "phc_disclaimer": (
            "このヘルスチェックはYahoo Financeから公開されている財務データを使用しています。"
            "データは遅延・不完全・不正確な場合があります。投資アドバイスではありません。"
            "投資前に必ずご自身でリサーチを行ってください。"
        ),
        # ── portfolio save/load ──
        "phc_save_header":       "💾 保存と読み込み",
        "phc_save_placeholder":  "ポートフォリオ名（例: 'ASXお気に入り'）",
        "phc_save_btn":          "保存",
        "phc_saved_ok":          "**{name}** として保存しました",
        "phc_load_header":       "保存済みポートフォリオ",
        "phc_load_none":         "保存済みポートフォリオはありません",
        "phc_load_btn":          "読み込む",
        "phc_delete_btn":        "削除",
        "phc_share_tip":         "💡 このURLをブックマークすると同じポートフォリオに戻れます。",
        # ── disclaimer ──
        "disclaimer":  "このウェブサイトは教育目的のみであり、投資アドバイスを構成するものではありません。過去の実績は将来の成果を保証するものではありません。投資前に必ずご自身でリサーチを行ってください。",
    },
}

def T(key: str, lang: str = "en", **kwargs) -> str:
    text = _TR.get(lang, _TR["en"]).get(key, _TR["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text


# ── fundamental rankings loader ────────────────────────────────────────────────

def _infer_market(ticker: str) -> str:
    if ticker.endswith(".AX"): return "ASX"
    if ticker.endswith(".T"):  return "JPX"
    return "US"

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return None

@st.cache_data(ttl=3600)
def load_fundamental_rankings() -> list[dict]:
    """Read the fundamentals cache and return all stocks ranked by health score."""
    cache_paths = [
        Path("screener_output/fundamentals_cache.json"),
        Path(__file__).parent / "screener_output" / "fundamentals_cache.json",
    ]
    cache = {}
    for p in cache_paths:
        if p.exists():
            try:
                cache = json.loads(p.read_text())
                break
            except Exception:
                pass
    if not cache:
        return []

    rows = []
    for ticker, r in cache.items():
        if not isinstance(r, dict):
            continue
        bd = r.get("breakdown", {})
        is_etf = r.get("is_etf", False)
        score  = r.get("score", 0)
        maxs   = r.get("max", 0)

        if is_etf:
            overall = "etf"
        elif maxs < 2:
            overall = "limited"
        elif score / maxs >= 0.67:
            overall = "strong"
        elif score / maxs >= 0.40:
            overall = "watch"
        else:
            overall = "weak"

        rows.append({
            "ticker":    ticker,
            "market":    _infer_market(ticker),
            "is_etf":    is_etf,
            "overall":   overall,
            "score":     score,
            "max":       maxs,
            "breakdown": bd,
            "name":      r.get("company_name", ticker),
            "sector":    r.get("sector", ""),
            "currency":  r.get("currency", ""),
            "market_cap":_safe_float(r.get("market_cap")),
            # scored metrics (raw values for display)
            "pe":        _safe_float(r.get("pe") or r.get("forward_pe")),
            "revenue_growth": _safe_float(r.get("revenue_growth")),
            "debt_equity":    _safe_float(r.get("debt_equity")),
            "free_cashflow":  _safe_float(r.get("free_cashflow")),
            "profit_margin":  _safe_float(r.get("profit_margin")),
            "analyst_rating": _safe_float(r.get("analyst_rating")),
            "analyst_label":  r.get("analyst_label", ""),
            "roe":            _safe_float(r.get("return_on_equity")),
            "current_ratio":  _safe_float(r.get("current_ratio")),
            # display-only
            "dividend_yield": _safe_float(r.get("dividend_yield")),
            "beta":           _safe_float(r.get("beta")),
            "price_to_book":  _safe_float(r.get("price_to_book")),
            "earnings_growth":_safe_float(r.get("earnings_growth")),
            "gross_margins":  _safe_float(r.get("gross_margins")),
            "fetched_at": r.get("fetched_at", ""),
        })

    rows.sort(key=lambda x: (-(x["score"] or 0), x["ticker"]))
    return rows


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
        autosize=True,
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font_size=10),
        margin=dict(l=45, r=8, t=60, b=30),
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

def _render_table(df: pd.DataFrame, verdict_col: str, simple: bool, lang: str = "en", fund_cache: dict = None):
    """Render a stock rankings table in simple or advanced mode."""
    _fc = fund_cache or {}
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

            _fdata    = _fc.get(asset, {})
            _cname    = _fdata.get("company_name", "")
            _sector   = _fdata.get("sector", "")

            badge = signal_badge(verdict, simple=True)
            is_spec    = asset in SPECULATIVE_TICKERS
            spec_badge = (f"<span title='{T('sr_spec_tooltip', lang)}' "
                          f"style='background:#2d1a00;color:#ff9500;border:1px solid #ff9500;"
                          f"padding:1px 6px;border-radius:4px;font-size:10px;font-weight:bold;"
                          f"margin-left:4px'>{T('sr_spec_badge', lang)}</span>"
                          if is_spec else "")
            ml_str = f"{float(ml):.0f}%" if pd.notna(ml) and ml not in ("—", None, "") else "—"
            try:
                ml_num = float(ml_str.replace("%",""))
                ml_color = "#3fb950" if ml_num >= 80 else ("#d29922" if ml_num >= 60 else "#8b949e")
            except Exception:
                ml_color = "#8b949e"
            size_str   = f"{float(size):.1f}%" if pd.notna(size) and size not in ("—", None, "") else "—"
            tier_color = TIER_LABELS.get(tier, ("", "#8b949e"))[1]
            trend_str  = trend_icon(trend)

            _name_html = (f"<span style='color:#c9d1d9;font-size:13px;font-weight:600'>{_cname}</span> "
                          if _cname else "")
            _sector_html = (f"<span style='color:#6e7681;font-size:11px'>{_sector}</span>"
                            if _sector else "")

            st.markdown(
                f"<div class='stock-row'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div>"
                f"<b style='font-size:16px'>{asset}</b> "
                f"<span style='color:#8b949e;font-size:12px'>{market}</span> &nbsp; "
                f"<span style='background:{tier_color}20;color:{tier_color};"
                f"padding:1px 7px;border-radius:4px;font-size:11px;font-weight:bold'>Tier {tier}</span>"
                f"{spec_badge}"
                f"</div>"
                f"<div style='font-size:18px;font-weight:bold'>{badge}</div>"
                f"</div>"
                f"<div style='margin-top:3px'>{_name_html}{_sector_html}</div>"
                f"<div style='margin-top:6px;font-size:13px;color:#8b949e'>"
                f"{T('sr_card_strategy', lang)}: <span style='color:#e6edf3'>{strat}</span> &nbsp;·&nbsp; "
                f"{T('sr_card_confidence', lang)}: <span style='color:{ml_color};font-weight:bold'>{ml_str}</span> &nbsp;·&nbsp; "
                f"{T('sr_card_trend', lang)}: {trend_str} &nbsp;·&nbsp; "
                f"{T('sr_card_size', lang)}: <b>{size_str}</b>"
                f"</div>"
                f"<div style='margin-top:4px;font-size:12px;color:#6e7681'>"
                f"{(STRATEGY_PLAIN_JA if lang == 'ja' else STRATEGY_PLAIN).get(strat, '')}"
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
            styled = styled.map(_c_verdict, subset=["Verdict"])
        if "Score" in adv.columns:
            styled = styled.map(_c_score, subset=["Score"])
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
.health-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0;
}
.health-check {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 14px;
}
.section-header { color: #e6edf3; font-size: 20px; font-weight: 700; margin: 24px 0 12px; }
.disclaimer-box {
    background: #3d2600; border: 1px solid #d29922; border-radius: 8px;
    padding: 12px 16px; font-size: 12px; color: #d29922; margin-top: 8px;
}
.guide-box {
    background: #0f2027; border: 1px solid #1f6feb; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 16px; font-size: 13px; color: #c9d1d9;
}
.level-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700; margin-left: 8px;
}
a { color: #58a6ff !important; }

/* ── Fundamental Rankings bright palette ──────────────── */
.fund-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 14px 18px; margin: 5px 0; transition: border-color 0.15s;
}
.fund-card:hover { border-color: #58a6ff; }
.fund-strong  { border-left: 4px solid #23d18b !important; }
.fund-watch   { border-left: 4px solid #ffca28 !important; }
.fund-weak    { border-left: 4px solid #ff4757 !important; }
.fund-etf     { border-left: 4px solid #a78bfa !important; }
.fund-limited { border-left: 4px solid #6b7280 !important; }
.bright-green  { color: #23d18b !important; font-weight: 700; }
.bright-amber  { color: #ffca28 !important; font-weight: 700; }
.bright-red    { color: #ff4757 !important; font-weight: 700; }
.bright-blue   { color: #4fc3f7 !important; font-weight: 700; }
.bright-purple { color: #a78bfa !important; font-weight: 700; }
.fund-summary-pill {
    display: inline-block; padding: 6px 16px; border-radius: 20px;
    font-size: 14px; font-weight: 700; margin: 4px 6px;
}
.market-badge {
    display: inline-block; font-size: 11px; font-weight: 700;
    padding: 2px 8px; border-radius: 4px; margin-left: 6px; vertical-align: middle;
}
.mbadge-us  { background: #1e3a5f; color: #4fc3f7; }
.mbadge-asx { background: #1a3a1a; color: #23d18b; }
.mbadge-jpx { background: #3a1a1a; color: #ff8c69; }
</style>
""", unsafe_allow_html=True)

# ── sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Goofy Screener")
    st.caption("Free quantitative stock screener · 無料クオンツ株スクリーナー\nUS · ASX · JPX")
    st.markdown("---")

    # language selector — top of sidebar so it affects everything below
    st.markdown("**Language / 言語**")
    lang_choice = st.radio(
        "lang",
        ["🇬🇧 English", "🇯🇵 日本語"],
        label_visibility="collapsed",
        horizontal=True,
    )
    lang = "ja" if "日本語" in lang_choice else "en"
    st.markdown("---")

    _nav_opts = (
        ["🏠 ホーム", "🔍 ポートフォリオ健全性チェック", "🌏 ファンダメンタルランキング",
         "📊 スクリーナーランキング", "📈 株価チャート", "🏆 トラックレコード",
         "ℹ️ 概要と免責事項"]
        if lang == "ja" else
        ["🏠 Home", "🔍 Portfolio Health Check", "🌏 Fundamental Rankings",
         "📊 Screener Rankings", "📈 Stock Chart", "🏆 Track Record",
         "ℹ️ About & Disclaimer"]
    )
    _nav_en = ["🏠 Home", "🔍 Portfolio Health Check", "🌏 Fundamental Rankings",
               "📊 Screener Rankings", "📈 Stock Chart", "🏆 Track Record",
               "ℹ️ About & Disclaimer"]
    _nav_sel = st.radio("nav", _nav_opts, label_visibility="collapsed")
    # always resolve to English key for page routing
    page = _nav_en[_nav_opts.index(_nav_sel)]
    st.markdown("---")

    # experience level selector
    _level_opts = (
        ["🟢 初心者", "🟡 中級者", "🔴 上級者"] if lang == "ja"
        else ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]
    )
    _level_en = ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"]
    _level_help = ("表示する詳細レベルを変更します。" if lang == "ja"
                   else "Changes how much detail is shown across the site.")
    st.markdown(T("exp_level", lang))
    _level_sel = st.radio(
        "level",
        _level_opts,
        label_visibility="collapsed",
        help=_level_help,
    )
    level = _level_en[_level_opts.index(_level_sel)]
    simple_mode = level == "🟢 Beginner"
    is_advanced  = level == "🔴 Advanced"

    st.markdown("---")
    if level == "🟢 Beginner":
        st.markdown(
            "<div style='background:#0f2a1a;border:1px solid #3fb950;border-radius:6px;"
            f"padding:10px 12px;font-size:12px;color:#c9d1d9'>{T('tip_beginner', lang)}"
            "</div>", unsafe_allow_html=True)
    elif level == "🟡 Intermediate":
        st.markdown(
            "<div style='background:#1a1a0f;border:1px solid #d29922;border-radius:6px;"
            f"padding:10px 12px;font-size:12px;color:#c9d1d9'>{T('tip_inter', lang)}"
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#1a0f0f;border:1px solid #f85149;border-radius:6px;"
            f"padding:10px 12px;font-size:12px;color:#c9d1d9'>{T('tip_advanced', lang)}"
            "</div>", unsafe_allow_html=True)

    st.markdown("---")
    # Market regime indicator — VIX + SPY vs 200d MA
    _regime = load_market_regime()
    _rs = _regime.get("status", "unknown")
    if _rs == "calm":
        _r_color, _r_icon = "#3fb950", "🟢"
        _r_txt = ("Market: Calm" if lang == "en" else "市場: 安定")
        _r_sub = (f"VIX {_regime['vix']} · SPY above 200d MA"
                  if lang == "en" else
                  f"VIX {_regime['vix']} · SPY 200日MA上回り")
    elif _rs == "risk_off":
        _r_color, _r_icon = "#f85149", "🔴"
        _r_txt = ("Market: Risk-Off" if lang == "en" else "市場: リスクオフ")
        _r_sub = (f"VIX {_regime['vix']} · {'SPY below 200d MA' if not _regime.get('above_ma') else 'elevated vol'}"
                  if lang == "en" else
                  f"VIX {_regime['vix']} · {'SPY 200日MA割れ' if not _regime.get('above_ma') else '高ボラティリティ'}")
    elif _rs == "caution":
        _r_color, _r_icon = "#d29922", "🟡"
        _r_txt = ("Market: Caution" if lang == "en" else "市場: 注意")
        _r_sub = (f"VIX {_regime['vix']} · watch closely"
                  if lang == "en" else
                  f"VIX {_regime['vix']} · 注意が必要")
    else:
        _r_color, _r_icon = "#8b949e", "⚪"
        _r_txt = ("Market: —" if lang == "en" else "市場: —")
        _r_sub = ("data unavailable" if lang == "en" else "データ取得不可")
    st.markdown(
        f"<div style='background:#0d1117;border:1px solid {_r_color};border-radius:6px;"
        f"padding:8px 10px;font-size:12px'>"
        f"<span style='color:{_r_color};font-weight:bold'>{_r_icon} {_r_txt}</span><br>"
        f"<span style='color:#8b949e'>{_r_sub}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(T("footer_note", lang))

# ── load data ─────────────────────────────────────────────────────────────────
df_universe, run_date, hours_ago = load_screener_universe()
df_history = load_trade_history()
df_open    = load_open_positions()


# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown(f"""
<div style='padding: 32px 0 16px'>
  <div style='font-size:36px;font-weight:800;color:#e6edf3'>Goofy Screener</div>
  <div style='font-size:18px;color:#8b949e;margin-top:6px'>
    {T("home_sub", lang)}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── new visitor guide ──────────────────────────────────────────────────────
    with st.expander(T("home_expander", lang), expanded=False):
        if lang == "ja":
            st.markdown("""
**Goofy Screenerは、米国・オーストラリア・日本市場の注目銘柄を見つけるための無料ツールです。**

毎日278銘柄に対して15種類の分析戦略を実行し、品質でランク付けします。
何を買うべきかを教えるものではありません。興味深いパターンを示している銘柄を特定し、さらに調査する価値があるものを示します。

---

### どのページに行けばいい？

| あなたの状況 | ページ |
|---|---|
| 保有銘柄の健全性を確認したい | 🔍 **ポートフォリオ健全性チェック** |
| 今日注目の銘柄を見たい | 📊 **スクリーナーランキング** |
| 特定の株のチャートを理解したい | 📈 **株価チャート** |
| システムのパフォーマンスを確認したい | 🏆 **トラックレコード** |

---

### サイドバーの3つのレベル

- **🟢 初心者** — 専門用語なし。シンプルな表示。
- **🟡 中級者** — 戦略の詳細、信頼スコア、ポジションサイズを追加表示。
- **🔴 上級者** — クオンツ分析を理解している方向けの完全な技術データ。

---

⚠️ **重要:** これは教育用ツールであり、投資アドバイスではありません。投資前に必ずご自身でリサーチを行ってください。
""")
        else:
            st.markdown("""
**Goofy Screener is a free tool that helps you find stocks worth looking at — across the US, Australian, and Japanese markets.**

It runs 15 different analysis strategies on 278 stocks every day and ranks them by quality.
It does NOT tell you what to buy. It tells you which stocks are showing interesting patterns and are worth further research.

---

### Which page should I go to?

| Your situation | Go to |
|---|---|
| I want to check if my existing stocks are healthy | 🔍 **Portfolio Health Check** |
| I want to see what stocks look interesting today | 📊 **Screener Rankings** |
| I want to understand a specific stock's chart | 📈 **Stock Chart** |
| I want to see how the system has performed | 🏆 **Track Record** |

---

### The three levels in the sidebar

- **🟢 Beginner** — plain English only. No jargon. Good if you're just starting out.
- **🟡 Intermediate** — adds strategy details, confidence scores, and position sizing.
- **🔴 Advanced** — full technical data for people who understand quantitative analysis.

---

⚠️ **Important:** This is an educational tool, not financial advice. Always do your own research before investing.
""")
    st.markdown("")

    # last updated badge
    if run_date:
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
            f"padding:8px 14px;display:inline-block;font-size:13px;color:#8b949e'>"
            f"{T('last_updated', lang)} <b style='color:#e6edf3'>{run_date}</b> ({hours_ago})"
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
            (c1, T("stat_screened", lang), str(total_stocks), T("stat_screened_s", lang)),
            (c2, T("stat_signals",  lang), str(n_signals),   T("stat_signals_s",  lang, n=n_markets)),
            (c3, T("stat_markets",  lang), "3",              T("stat_markets_s",  lang)),
            (c4, T("stat_strats",   lang), "15",             T("stat_strats_s",   lang)),
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
        st.markdown(T("how_it_works", lang))
        if lang == "ja":
            st.markdown("""
Goofy Screenerは毎日278銘柄に対して15種類のクオンツ取引戦略を実行します。
各銘柄について、モデルが学習で見たことのないデータでテストした、最も強い過去パフォーマンスを持つ戦略を選びます。

**パイプライン:**
1. **戦略選択** — 15戦略をバックテストしスコアリング。銘柄ごとに最良の戦略を選択。
2. **レジームフィルター** — 選択した戦略が歴史的に低パフォーマンスな市場環境（強気/弱気/横ばい）ではシグナルを抑制。
3. **MLゲート** — 過去のシグナルで学習したXGBoostモデルが信頼スコアを付与。低信頼スコアのシグナルは保留。
4. **ポジションサイジング** — ケリー基準＋ボラティリティ調整で推奨配分を算出。

シグナルは **BUY（買い）**、**WATCH（注目）**、**PASS（見送り）** のいずれかで表示されます。
""")
        else:
            st.markdown("""
The Goofy Screener runs 15 quantitative trading strategies across 278 stocks every day.
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
        st.markdown(T("top_picks", lang))
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
            st.info(T("no_data_home", lang))

    st.markdown("---")
    st.markdown(
        f"<div class='disclaimer-box'>{T('disclaimer', lang)}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Portfolio Health Check":
    # check label + plain-description lookup (built once per render)
    _phc_label_map = {
        "Valuation (P/E ratio)": T("phc_check_pe", lang),
        "Revenue growth":        T("phc_check_rev", lang),
        "Debt level":            T("phc_check_debt", lang),
        "Free cash flow":        T("phc_check_fcf", lang),
        "Profitability":         T("phc_check_prof", lang),
        "Analyst view":          T("phc_check_analyst", lang),
    }
    _phc_plain_map = {
        "Is the stock priced reasonably compared to its earnings?":       T("phc_plain_pe", lang),
        "Is the company's revenue growing over the past year?":           T("phc_plain_rev", lang),
        "Does the company have manageable levels of debt?":               T("phc_plain_debt", lang),
        "Is the company actually generating real cash from its business?":T("phc_plain_fcf", lang),
        "Is the company profitable (making more than it spends)?":        T("phc_plain_prof", lang),
        "What do professional analysts think about this stock?":          T("phc_plain_analyst", lang),
    }

    st.markdown(T("phc_title", lang))
    st.markdown(
        f"<div class='guide-box'>{T('phc_guide', lang)}</div>",
        unsafe_allow_html=True,
    )

    # ── build multiselect options from fundamentals cache ─────────────────────
    _fund_cache_path = SCREENER_DIR / "fundamentals_cache.json"
    _universe_tickers: list[str] = []
    if _fund_cache_path.exists():
        try:
            _fc = json.loads(_fund_cache_path.read_text())
            _us  = sorted(t for t in _fc if not t.endswith(".AX") and not t.endswith(".T"))
            _asx = sorted(t for t in _fc if t.endswith(".AX"))
            _jpx = sorted(t for t in _fc if t.endswith(".T"))
            def _opt_label(t, mkt):
                name = _fc.get(t, {}).get("company_name", "")
                return f"{t} — {name}  ·  {mkt}" if name else f"{t}  ·  {mkt}"
            _universe_tickers = (
                [_opt_label(t, "US")  for t in _us] +
                [_opt_label(t, "ASX") for t in _asx] +
                [_opt_label(t, "JPX") for t in _jpx]
            )
        except Exception:
            pass

    # ── seed custom input from URL query param on first visit ─────────────────
    if "phc_custom_input" not in st.session_state:
        _url_p = st.query_params.get("p", "")
        if _url_p:
            st.session_state["phc_custom_input"] = _url_p

    # ── initialise in-session saved portfolios store ───────────────────────────
    if "phc_saved_portfolios" not in st.session_state:
        st.session_state["phc_saved_portfolios"] = {}

    selected_from_list = st.multiselect(
        T("phc_pick_label", lang),
        options=_universe_tickers,
        default=[],
        placeholder="Search by ticker…",
        key="phc_multiselect",
    )
    custom_input = st.text_input(
        T("phc_or_custom", lang),
        placeholder=T("phc_placeholder", lang),
        key="phc_custom_input",
    )

    # combine: strip the "  ·  MARKET" suffix and optional "TICKER — Company" format
    picked = [opt.split("  ·  ")[0].split(" — ")[0].strip() for opt in selected_from_list]
    if custom_input.strip():
        picked += [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    raw_tickers = list(dict.fromkeys(picked))  # dedupe, preserve order

    ticker_input = ",".join(raw_tickers)  # keep downstream logic unchanged

    # ── sync tickers to URL so the portfolio is bookmarkable ──────────────────
    if raw_tickers:
        st.query_params["p"] = ",".join(raw_tickers)
    elif "p" in st.query_params:
        del st.query_params["p"]

    # ── save / load named portfolios (in-session) ──────────────────────────────
    _saved = st.session_state["phc_saved_portfolios"]
    with st.expander(T("phc_save_header", lang), expanded=False):
        # Save row
        _sc1, _sc2 = st.columns([4, 1])
        with _sc1:
            _save_name = st.text_input(
                "name", label_visibility="collapsed",
                placeholder=T("phc_save_placeholder", lang),
                key="phc_save_name",
            )
        with _sc2:
            if st.button(T("phc_save_btn", lang), key="phc_save_btn", disabled=not raw_tickers):
                if _save_name.strip():
                    _saved[_save_name.strip()] = ",".join(raw_tickers)
                    st.success(T("phc_saved_ok", lang, name=_save_name.strip()))

        # Load / delete saved portfolios
        if _saved:
            st.markdown(f"**{T('phc_load_header', lang)}**")
            for _pname, _ptickers in list(_saved.items()):
                _lc1, _lc2, _lc3 = st.columns([4, 1, 1])
                _lc1.markdown(f"**{_pname}** — `{_ptickers}`")
                if _lc2.button(T("phc_load_btn", lang), key=f"phc_load_{_pname}"):
                    st.session_state["phc_custom_input"] = _ptickers
                    st.rerun()
                if _lc3.button(T("phc_delete_btn", lang), key=f"phc_del_{_pname}"):
                    del _saved[_pname]
                    st.rerun()
        else:
            st.caption(T("phc_load_none", lang))

        if raw_tickers:
            st.caption(T("phc_share_tip", lang))

    if raw_tickers:
        if len(raw_tickers) > 15:
            st.warning(T("phc_max_warning", lang))
            raw_tickers = raw_tickers[:15]

        st.markdown(T("phc_checking", lang, n=len(raw_tickers)))
        results = []
        prog = st.progress(0)
        for i, t in enumerate(raw_tickers):
            results.append(fetch_stock_health(t))
            prog.progress((i + 1) / len(raw_tickers))
        prog.empty()

        # ── render each result ─────────────────────────────────────────────────
        for r in results:
            ticker = r["ticker"]

            if r.get("error"):
                st.markdown(
                    f"<div class='health-card'>"
                    f"<b style='color:#f85149'>{ticker}</b> — {T('phc_error', lang)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                continue

            if r.get("is_etf"):
                st.markdown(
                    f"<div class='health-card'>"
                    f"<b style='color:#58a6ff'>{ticker}</b> &nbsp; ETF / Index Fund<br>"
                    f"<span style='color:#8b949e;font-size:13px'>{T('phc_etf_msg', lang)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                continue

            overall = r.get("overall", "limited_data")
            score   = r["score"]
            maxs    = r["max"]
            name    = r["name"]
            sector  = r.get("sector", "")
            industry= r.get("industry", "")
            mcap    = r.get("market_cap")

            overall_cfg = {
                "healthy":      ("#3fb950", T("phc_healthy_label", lang), T("phc_healthy_sum", lang)),
                "caution":      ("#d29922", T("phc_caution_label", lang), T("phc_caution_sum", lang)),
                "concern":      ("#f85149", T("phc_concern_label", lang), T("phc_concern_sum", lang)),
                "limited_data": ("#8b949e", T("phc_limited_label", lang), T("phc_limited_sum", lang)),
            }
            clr, label, summary = overall_cfg.get(overall, overall_cfg["limited_data"])

            mcap_str = ""
            if mcap:
                try:
                    mcap_b = float(mcap) / 1e9
                    mcap_str = f" &nbsp;·&nbsp; {T('phc_mcap', lang, n=f'{mcap_b:.1f}')}"
                except Exception:
                    pass

            st.markdown(
                f"<div class='health-card' style='border-left: 4px solid {clr}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                f"<div>"
                f"<span style='font-size:18px;font-weight:700;color:#e6edf3'>{ticker}</span> &nbsp;"
                f"<span style='color:#8b949e;font-size:13px'>{name}</span><br>"
                f"<span style='font-size:12px;color:#8b949e'>{sector}{' — ' + industry if industry else ''}{mcap_str}</span>"
                f"</div>"
                f"<div style='text-align:right'>"
                f"<div style='font-size:16px;font-weight:700;color:{clr}'>{label}</div>"
                f"<div style='font-size:12px;color:#8b949e'>{T('phc_checks_passed', lang, score=score, maxs=maxs)}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # ── price snapshot row ──────────────────────────────────────────────
            curr_p = r.get("current_price")
            yh     = r.get("year_high")
            yl     = r.get("year_low")
            yc     = r.get("year_chg")
            div    = r.get("dividend_yield")
            if curr_p is not None:
                _yc_valid = yc is not None and yc == yc  # excludes NaN
                yc_str   = f"{yc*100:+.1f}%" if _yc_valid else "—"
                yc_color = "#3fb950" if _yc_valid and yc > 0 else ("#f85149" if _yc_valid and yc < 0 else "#8b949e")
                div_str  = f"{div:.1f}%" if div and div > 0 else "—"
                range_bar_html = ""
                if yh and yl and yh > yl:
                    pct = max(0, min(100, (curr_p - yl) / (yh - yl) * 100))
                    range_bar_html = (
                        f"<div style='margin:10px 0 4px'>"
                        f"<div style='font-size:11px;color:#8b949e;margin-bottom:4px'>52-week range</div>"
                        f"<div style='display:flex;align-items:center;gap:8px;font-size:11px;color:#8b949e'>"
                        f"<span>{yl:.2f}</span>"
                        f"<div style='flex:1;height:5px;background:#21262d;border-radius:3px;position:relative'>"
                        f"<div style='position:absolute;left:{pct:.0f}%;top:-3px;width:11px;height:11px;"
                        f"background:#58a6ff;border-radius:50%;transform:translateX(-50%)'></div></div>"
                        f"<span>{yh:.2f}</span></div></div>"
                    )
                st.markdown(
                    f"<div style='display:flex;gap:24px;margin:10px 0 4px;flex-wrap:wrap'>"
                    f"<div><div style='font-size:11px;color:#8b949e'>Current price</div>"
                    f"<div style='font-size:17px;font-weight:700;color:#e6edf3'>{curr_p:.2f}</div></div>"
                    f"<div><div style='font-size:11px;color:#8b949e'>1-year return</div>"
                    f"<div style='font-size:17px;font-weight:700;color:{yc_color}'>{yc_str}</div></div>"
                    f"<div><div style='font-size:11px;color:#8b949e'>Dividend yield</div>"
                    f"<div style='font-size:17px;font-weight:700;color:#e6edf3'>{div_str}</div></div>"
                    f"</div>" + range_bar_html,
                    unsafe_allow_html=True,
                )

            # ── Goofy screener view on this stock ──────────────────────────────
            if not df_universe.empty:
                asset_col = next((c for c in df_universe.columns if c.lower() in ("asset","ticker","symbol")), None)
                if asset_col:
                    match = df_universe[df_universe[asset_col] == ticker]
                    if not match.empty:
                        sc = match.iloc[0]
                        ml  = sc.get("ml_score", sc.get("ML Score", sc.get("score", "")))
                        sig = sc.get("signal",   sc.get("Signal", ""))
                        sig_color = "#3fb950" if str(sig).upper() == "BUY" else ("#d29922" if str(sig).upper() == "WATCH" else "#8b949e")
                        ml_str = f"{float(ml):.0f}/100" if ml != "" else "—"
                        st.markdown(
                            f"<div style='background:#1c2128;border:1px solid #3fb95050;"
                            f"border-radius:6px;padding:8px 12px;margin:8px 0;font-size:12px'>"
                            f"🤖 <b style='color:#3fb950'>In Goofy's screener</b> &nbsp;·&nbsp; "
                            f"ML confidence: <b style='color:#58a6ff'>{ml_str}</b> &nbsp;·&nbsp; "
                            f"Signal: <b style='color:{sig_color}'>{sig}</b>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── score bar ──────────────────────────────────────────────────────
            bar_filled = int((score / maxs * 10)) if maxs else 0
            bar_html = "".join(
                f"<span style='display:inline-block;width:24px;height:10px;border-radius:2px;margin:1px;"
                f"background:{'#3fb950' if i < bar_filled else '#21262d'}'></span>"
                for i in range(10)
            )
            st.markdown(
                f"<div style='padding:8px 0 4px'>{bar_html} &nbsp;"
                f"<span style='font-size:12px;color:#8b949e'>{T('phc_fundamentals', lang, score=score, maxs=maxs)}</span></div>"
                f"<div style='font-size:13px;color:#8b949e;margin-bottom:10px'>{summary}</div>",
                unsafe_allow_html=True,
            )

            # ── individual checks ──────────────────────────────────────────────
            for check in r["checks"]:
                icon  = "✅" if check["pass"] else "❌"
                color = "#3fb950" if check["pass"] else "#f85149"
                _check_label = _phc_label_map.get(check["label"], check["label"])
                detail_text  = (_phc_plain_map.get(check["plain"], check["plain"])
                                if simple_mode else check["detail"])
                st.markdown(
                    f"<div style='display:flex;gap:10px;align-items:flex-start;"
                    f"padding:5px 0;border-bottom:1px solid #21262d;font-size:13px'>"
                    f"<span>{icon}</span>"
                    f"<div><b style='color:{color}'>{_check_label}</b>"
                    f"<span style='color:#8b949e'> — {detail_text}</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

    else:
        # placeholder when nothing entered yet
        st.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;"
            "padding:40px;text-align:center;color:#8b949e;margin-top:16px'>"
            "<div style='font-size:40px'>🔍</div>"
            f"<div style='font-size:16px;margin-top:12px'>{T('phc_empty', lang)}</div>"
            f"<div style='font-size:13px;margin-top:8px'>{T('phc_empty_sub', lang)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── portfolio breakdown (only shown after analysis) ────────────────────────
    if ticker_input.strip() and results:
        valid = [r for r in results if not r.get("error")]
        if valid:
            by_class: dict[str, list[str]] = {}
            for r in valid:
                ac = r.get("asset_class", "Stock")
                by_class.setdefault(ac, []).append(r["ticker"])

            total_holdings = len(valid)
            stocks_count   = len(by_class.get("Stock", []))
            defensive_count= total_holdings - stocks_count

            st.markdown("---")
            st.markdown(T("phc_snapshot", lang))

            breakdown_rows = []
            for ac, tickers_in_class in sorted(by_class.items()):
                icon = ASSET_CLASS_ICONS.get(ac, "📦")
                pct  = len(tickers_in_class) / total_holdings * 100
                breakdown_rows.append(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:8px 12px;border-bottom:1px solid #21262d;font-size:14px'>"
                    f"<div>{icon} <b style='color:#e6edf3'>{ac}</b> &nbsp;"
                    f"<span style='color:#8b949e;font-size:12px'>{', '.join(tickers_in_class)}</span></div>"
                    f"<div style='color:#58a6ff;font-weight:700'>{pct:.0f}%</div>"
                    f"</div>"
                )

            st.markdown(
                "<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;"
                "overflow:hidden;margin-bottom:16px'>" +
                "".join(breakdown_rows) +
                "</div>",
                unsafe_allow_html=True,
            )

            # diversification nudge
            if stocks_count == total_holdings and total_holdings >= 2:
                nudge_color, nudge_icon, nudge_msg = "#d29922", "⚠️", T("phc_nudge_stocks", lang)
            elif defensive_count == 0:
                nudge_color, nudge_icon, nudge_msg = "#d29922", "💡", T("phc_nudge_mostly", lang)
            else:
                nudge_color, nudge_icon, nudge_msg = "#3fb950", "✅", T("phc_nudge_diverse", lang, n=len(by_class))

            st.markdown(
                f"<div style='background:#161b22;border:1px solid {nudge_color};"
                f"border-radius:8px;padding:14px 16px;font-size:13px;color:#c9d1d9;margin-bottom:8px'>"
                f"{nudge_icon} {nudge_msg}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── portfolio advisor ──────────────────────────────────────────────
            _adv_valid = [r for r in results if not r.get("error") and not r.get("is_etf")]
            if _adv_valid:
                _adv_healthy = [r for r in _adv_valid if r.get("overall") == "healthy"]
                _adv_caution = [r for r in _adv_valid if r.get("overall") == "caution"]
                _adv_concern = [r for r in _adv_valid if r.get("overall") == "concern"]

                # map screener signals for each ticker
                _adv_sigs: dict[str, dict] = {}
                if not df_universe.empty:
                    _a_col = next((c for c in df_universe.columns if c.lower() in ("asset","ticker","symbol")), None)
                    _s_col = next((c for c in df_universe.columns if c.lower() == "signal"), None)
                    _m_col = next((c for c in df_universe.columns if "ml" in c.lower() and "score" in c.lower()), None)
                    if _a_col and _s_col:
                        for _, _row in df_universe.iterrows():
                            _adv_sigs[_row[_a_col]] = {
                                "signal": str(_row.get(_s_col, "")).upper(),
                                "ml": _row.get(_m_col, "") if _m_col else "",
                            }

                # build action items
                _action_items = []

                # sector concentration
                _adv_sectors: dict[str, list] = {}
                for r in _adv_valid:
                    sec = r.get("sector", "")
                    if sec:
                        _adv_sectors.setdefault(sec, []).append(r["ticker"])
                if _adv_sectors:
                    _top_sec = max(_adv_sectors, key=lambda s: len(_adv_sectors[s]))
                    _top_n   = len(_adv_sectors[_top_sec])
                    if _top_n / len(_adv_valid) >= 0.5 and len(_adv_valid) >= 3:
                        _tickers_in_sec = ", ".join(_adv_sectors[_top_sec])
                        _action_items.append(("⚠️", _top_sec,
                            f"{_top_n}/{len(_adv_valid)} holdings — heavy concentration. Consider adding stocks from other sectors."
                            if lang == "en" else
                            f"{_top_n}/{len(_adv_valid)}銘柄が同セクター — 分散を検討してください。",
                            _tickers_in_sec))

                # concern stocks
                for r in _adv_concern:
                    _sig = _adv_sigs.get(r["ticker"], {}).get("signal", "")
                    if _sig == "BUY":
                        _action_items.append(("⚠️", r["ticker"],
                            "Screener says BUY but fundamentals look weak — do extra due diligence before acting."
                            if lang == "en" else
                            "スクリーナーはBUYですがファンダメンタルズが弱い — 追加調査を推奨。",
                            r.get("name", "")))
                    else:
                        _action_items.append(("🔴", r["ticker"],
                            f"Weak fundamentals ({r['score']}/{r['max']} checks passed). Review or reduce position."
                            if lang == "en" else
                            f"ファンダメンタルズが弱い（{r['score']}/{r['max']}通過）。見直しを検討。",
                            r.get("name", "")))

                # healthy stocks with no buy signal (hold-only)
                for r in _adv_healthy:
                    _sig = _adv_sigs.get(r["ticker"], {}).get("signal", "")
                    if _sig and _sig != "BUY":
                        _action_items.append(("💡", r["ticker"],
                            "Good fundamentals — no fresh BUY signal right now, but solid long-term hold."
                            if lang == "en" else
                            "ファンダメンタルズは良好 — 現在新規BUYシグナルなし。長期保有には適切。",
                            r.get("name", "")))

                # dividend income estimate
                _div_stocks = [(r["ticker"], r.get("name",""), r.get("dividend_yield",0))
                               for r in _adv_valid if r.get("dividend_yield") and r["dividend_yield"] > 0.01]

                # best picks: healthy or caution + BUY signal
                _best = []
                for r in _adv_healthy + _adv_caution:
                    _sig_info = _adv_sigs.get(r["ticker"], {})
                    if _sig_info.get("signal") == "BUY":
                        _ml = _sig_info.get("ml", "")
                        _ml_str = f" · ML {float(_ml):.0f}/100" if _ml != "" else ""
                        _best.append((r["ticker"], r.get("name",""), r.get("overall",""), _ml_str))

                _adv_title = "🧠 Portfolio Advisor" if lang == "en" else "🧠 ポートフォリオアドバイザー"
                st.markdown(f"### {_adv_title}")

                # summary line
                _summ_parts = []
                if _adv_healthy: _summ_parts.append(f"<span style='color:#3fb950;font-weight:700'>{len(_adv_healthy)} {'healthy' if lang == 'en' else '健全'}</span>")
                if _adv_caution: _summ_parts.append(f"<span style='color:#d29922;font-weight:700'>{len(_adv_caution)} {'caution' if lang == 'en' else '注意'}</span>")
                if _adv_concern: _summ_parts.append(f"<span style='color:#f85149;font-weight:700'>{len(_adv_concern)} {'concern' if lang == 'en' else '懸念'}</span>")
                st.markdown(
                    f"<div style='font-size:14px;color:#8b949e;margin-bottom:12px'>"
                    f"{'Your ' + str(len(_adv_valid)) + ' stocks:' if lang == 'en' else str(len(_adv_valid)) + '銘柄の内訳:'} {' · '.join(_summ_parts)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if _best:
                    st.markdown(f"**{'⭐ Best picks in your portfolio' if lang == 'en' else '⭐ おすすめ保有銘柄'}**")
                    for _t, _n, _health, _ml_str in _best:
                        _hclr = "#3fb950" if _health == "healthy" else "#d29922"
                        _hlbl = ("Healthy" if _health == "healthy" else "Caution") if lang == "en" else ("健全" if _health == "healthy" else "注意")
                        st.markdown(
                            f"<div style='background:#1c2128;border:1px solid #3fb95040;"
                            f"border-left:4px solid #3fb950;border-radius:6px;"
                            f"padding:10px 14px;margin:4px 0;font-size:13px'>"
                            f"🤖 <b style='color:#e6edf3'>{_t}</b> &nbsp;"
                            f"<span style='color:#8b949e'>{_n}</span> &nbsp;"
                            f"<span style='background:#3fb95020;color:#3fb950;padding:2px 8px;"
                            f"border-radius:4px;font-size:12px'>BUY</span> &nbsp;"
                            f"<span style='color:{_hclr};font-size:12px'>{_hlbl} fundamentals</span>"
                            f"<span style='color:#8b949e;font-size:12px'>{_ml_str}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                if _action_items:
                    st.markdown(f"**{'📋 Action items' if lang == 'en' else '📋 アクション項目'}**")
                    for _icon, _subject, _msg, _sub in _action_items:
                        _sub_html = (f"  <span style='color:#8b949e;font-size:12px'>({_sub})</span>"
                                     if _sub else "")
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid #30363d;"
                            f"border-radius:6px;padding:10px 14px;margin:4px 0;font-size:13px'>"
                            f"{_icon} <b style='color:#e6edf3'>{_subject}</b>{_sub_html}"
                            f"<br><span style='color:#8b949e'>{_msg}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                if _div_stocks:
                    st.markdown(f"**{'💸 Dividend payers in your portfolio' if lang == 'en' else '💸 配当銘柄'}**")
                    for _t, _n, _dy in sorted(_div_stocks, key=lambda x: -x[2]):
                        st.markdown(
                            f"<div style='background:#161b22;border:1px solid #30363d;"
                            f"border-radius:6px;padding:8px 14px;margin:3px 0;font-size:13px;"
                            f"display:flex;align-items:center'>"
                            f"💰 <b style='color:#e6edf3;margin-left:6px'>{_t}</b> &nbsp;"
                            f"<span style='color:#8b949e'>{_n}</span> &nbsp;"
                            f"<span style='color:#3fb950;font-weight:700'>{_dy:.1f}% yield</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                if not _best and not _action_items:
                    st.markdown(
                        f"<div style='color:#8b949e;font-size:13px;padding:8px 0'>"
                        f"{'✅ No urgent action items. Portfolio looks balanced.' if lang == 'en' else '✅ 緊急アクション不要。ポートフォリオはバランスが取れています。'}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

            # what the tool can't check
            st.markdown(
                f"<div class='guide-box'>{T('phc_cant_check', lang)}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        f"<div class='guide-box'>{T('phc_how_to_read', lang)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='disclaimer-box'>{T('phc_disclaimer', lang)}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌏 Fundamental Rankings":
    all_rows = load_fundamental_rankings()
    most_recent = max((r["fetched_at"] for r in all_rows if r.get("fetched_at")), default="")

    st.markdown(f"## {T('fund_title', lang)}")
    st.markdown(
        f"<div style='color:#c9d1d9;font-size:14px;margin-bottom:4px'>"
        f"{T('fund_sub', lang, n=len(all_rows))}</div>",
        unsafe_allow_html=True,
    )
    if most_recent:
        st.caption(f"{T('last_upd', lang)}: {most_recent[:10]}")

    with st.expander(T("guide_title", lang), expanded=False):
        st.markdown(T("guide_body", lang))

    if not all_rows:
        st.warning(T("no_cache", lang))
    else:
        # ── summary pills ──────────────────────────────────────────────────
        counts = {"strong": 0, "watch": 0, "weak": 0, "limited": 0, "etf": 0}
        for r in all_rows:
            counts[r["overall"]] = counts.get(r["overall"], 0) + 1

        pill_cfg = [
            ("strong",  "#23d18b", "#0f2a1a"),
            ("watch",   "#ffca28", "#2a1f00"),
            ("weak",    "#ff4757", "#2a0f0f"),
            ("limited", "#6b7280", "#1a1a1a"),
            ("etf",     "#a78bfa", "#1a0f2e"),
        ]
        pills_html = "<div style='margin:12px 0 16px'>"
        for key, fg, bg in pill_cfg:
            n = counts.get(key, 0)
            if n:
                label = T(key, lang)
                pills_html += (
                    f"<span style='background:{bg};color:{fg};border:1px solid {fg};"
                    f"display:inline-block;padding:5px 14px;border-radius:20px;"
                    f"font-size:13px;font-weight:700;margin:3px 4px'>"
                    f"{n} {label}</span>"
                )
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

        # ── filters & sort ─────────────────────────────────────────────────
        col_f, col_s, col_q = st.columns([2, 2, 3])
        with col_f:
            mkt_opts = [T("all_markets", lang), "🇺🇸 US", "🇦🇺 ASX", "🇯🇵 JPX"]
            mkt_sel  = st.selectbox(T("filter_market", lang), mkt_opts, label_visibility="visible")
        with col_s:
            sort_opts = [T("sort_score", lang), T("sort_pe", lang), T("sort_roe", lang), T("sort_div", lang)]
            sort_sel  = st.selectbox(T("sort_by", lang), sort_opts, label_visibility="visible")
        with col_q:
            health_filter = st.multiselect(
                "Show" if lang == "en" else "表示",
                options=["strong","watch","weak","limited","etf"],
                default=["strong","watch","weak","limited"],
                format_func=lambda x: T(x, lang),
            )

        # apply market filter
        mkt_map = {"🇺🇸 US": "US", "🇦🇺 ASX": "ASX", "🇯🇵 JPX": "JPX"}
        filtered = all_rows
        if mkt_sel in mkt_map:
            filtered = [r for r in filtered if r["market"] == mkt_map[mkt_sel]]
        if health_filter:
            filtered = [r for r in filtered if r["overall"] in health_filter]

        # apply sort
        if sort_sel == T("sort_pe", lang):
            filtered = sorted(filtered, key=lambda x: (x["pe"] is None, x["pe"] or 999))
        elif sort_sel == T("sort_roe", lang):
            filtered = sorted(filtered, key=lambda x: (x["roe"] is None, -(x["roe"] or 0)))
        elif sort_sel == T("sort_div", lang):
            filtered = sorted(filtered, key=lambda x: (x["dividend_yield"] is None, -(x["dividend_yield"] or 0)))
        # default: already sorted by score

        st.caption(f"{len(filtered)} {T('n_stocks', lang)}")
        st.markdown("")

        # ── render cards ───────────────────────────────────────────────────
        overall_cfg = {
            "strong":  ("#23d18b", "fund-strong",  T("pass_lbl",  lang)),
            "watch":   ("#ffca28", "fund-watch",   T("watch_lbl", lang)),
            "weak":    ("#ff4757", "fund-weak",    T("fail_lbl",  lang)),
            "etf":     ("#a78bfa", "fund-etf",     T("etf_lbl",   lang)),
            "limited": ("#6b7280", "fund-limited", T("na_lbl",    lang)),
        }
        mbadge = {"US": "mbadge-us", "ASX": "mbadge-asx", "JPX": "mbadge-jpx"}

        CHECK_ORDER = ["pe","revenue_growth","debt_equity","free_cashflow",
                       "profit_margin","analyst","roe","current_ratio"]
        CHECK_LABELS_EN = ["P/E","Rev","Debt","FCF","Profit","Analysts","ROE","Liq"]
        CHECK_LABELS_JA = ["PER","売上","負債","FCF","利益","評価","ROE","流動"]
        check_labels = CHECK_LABELS_JA if lang == "ja" else CHECK_LABELS_EN

        def fmt_val(val, mode):
            if val is None: return "—"
            if mode == "pct":   return f"{val*100:+.1f}%"
            if mode == "pct0":  return f"{val*100:.0f}%"
            if mode == "x1":    return f"{val:.1f}x"
            if mode == "x2":    return f"{val:.2f}"
            if mode == "b":     return ("+" if val > 0 else "") + f"${val/1e9:.1f}B"
            return f"{val:.1f}"

        def val_color(val, positive_is_good=True, neutral_zone=None):
            if val is None: return "#8b949e"
            if neutral_zone and abs(val) <= neutral_zone: return "#c9d1d9"
            good = val > 0 if positive_is_good else val < 0
            return "#23d18b" if good else "#ff4757"

        for rank, row in enumerate(filtered, 1):
            clr, css_class, health_lbl = overall_cfg.get(row["overall"], overall_cfg["limited"])
            mkt = row["market"]
            badge_cls = mbadge.get(mkt, "mbadge-us")
            mcap_str = ""
            mc = row.get("market_cap")
            if mc:
                mcap_str = f"${mc/1e9:.0f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"

            # score segment bar (ETFs have no individual fundamentals — skip bar)
            is_etf_row = row.get("is_etf", False)
            bd = row["breakdown"]
            if is_etf_row:
                segs = (
                    f"<span style='color:#8b949e;font-size:12px'>"
                    + ("ETFs track a basket of stocks — individual P/E, ROE etc. don't apply."
                       if lang == "en" else
                       "ETFは複数銘柄を追跡するため個別の財務指標（PER・ROE等）は適用されません。")
                    + "</span>"
                )
            else:
                segs = ""
                for ck in CHECK_ORDER:
                    v = bd.get(ck)
                    seg_clr = "#23d18b" if v == 1 else "#ff4757" if v == 0 else "#2d333b"
                    segs += (
                        f"<span style='display:inline-block;width:18px;height:8px;"
                        f"border-radius:2px;margin:1px;background:{seg_clr}'></span>"
                    )

            # stat pills
            stats = []
            pe_v    = row.get("pe")
            roe_v   = row.get("roe")
            rev_v   = row.get("revenue_growth")
            div_v   = row.get("dividend_yield")
            debt_v  = row.get("debt_equity")
            anl_v   = row.get("analyst_label", "")
            cr_v    = row.get("current_ratio")
            beta_v  = row.get("beta")
            pb_v    = row.get("price_to_book")

            def stat_pill(label, val_str, color):
                return (
                    f"<span style='background:#21262d;border-radius:6px;padding:3px 8px;"
                    f"font-size:12px;margin:2px;display:inline-block'>"
                    f"<span style='color:#8b949e'>{label}: </span>"
                    f"<span style='color:{color};font-weight:700'>{val_str}</span></span>"
                )

            if pe_v is not None:
                pc = "#23d18b" if pe_v < 20 else "#ffca28" if pe_v < 40 else "#ff4757"
                stats.append(stat_pill(T("lbl_pe",lang), f"{pe_v:.1f}", pc))
            if roe_v is not None:
                stats.append(stat_pill(T("lbl_roe",lang), f"{roe_v*100:.0f}%", val_color(roe_v)))
            if rev_v is not None:
                stats.append(stat_pill(T("lbl_rev",lang), f"{rev_v*100:+.0f}%", val_color(rev_v)))
            if div_v is not None and div_v > 0:
                # yfinance dividendYield is already in percentage points (2.94 = 2.94%)
                stats.append(stat_pill(T("lbl_div",lang), f"{div_v:.1f}%", "#4fc3f7"))
            if debt_v is not None:
                dc = "#23d18b" if debt_v < 100 else "#ffca28" if debt_v < 300 else "#ff4757"
                stats.append(stat_pill(T("lbl_debt",lang), f"{debt_v:.0f}%", dc))
            if cr_v is not None:
                stats.append(stat_pill(T("lbl_cr",lang), f"{cr_v:.1f}", val_color(cr_v - 1)))
            if beta_v is not None:
                bc = "#23d18b" if beta_v < 1 else "#ffca28" if beta_v < 1.5 else "#ff4757"
                stats.append(stat_pill(T("lbl_beta",lang), f"{beta_v:.2f}", bc))
            if anl_v:
                al_colors = {"strongbuy":"#23d18b","buy":"#3fb950","hold":"#ffca28",
                             "underperform":"#ff8c69","sell":"#ff4757"}
                ac = al_colors.get(anl_v.replace(" ","").lower(), "#8b949e")
                stats.append(stat_pill(T("lbl_analyst",lang), anl_v.replace("_"," ").title(), ac))

            stats_html = "".join(stats) if stats else ""

            sector_html = ("<span style='color:#6b7280;font-size:11px'>" + row.get("sector","") + "</span>") if row.get("sector") else ""
            mcap_html   = ("<span style='color:#6b7280;font-size:11px;margin-left:4px'>" + mcap_str + "</span>") if mcap_str else ""

            _ticker = row["ticker"]
            _exp_label = f"#{rank}  {_ticker}  ·  {row['name']}  —  {health_lbl}"
            with st.expander(_exp_label, expanded=False):
                # ── card summary ──────────────────────────────────────────────
                st.markdown(
                    f"<div class='fund-card {css_class}'>"
                    f"<div style='display:flex;align-items:flex-start;gap:14px'>"
                    f"<div style='font-size:17px;font-weight:800;color:#4fc3f7;"
                    f"min-width:30px;padding-top:2px'>#{rank}</div>"
                    f"<div style='flex:1;min-width:0'>"
                    f"<div style='display:flex;align-items:center;flex-wrap:wrap;gap:4px'>"
                    f"<span style='font-size:17px;font-weight:800;color:#ffffff'>{_ticker}</span>"
                    f"<span style='color:#c9d1d9;font-size:13px'>{row['name']}</span>"
                    f"<span class='market-badge {badge_cls}'>{mkt}</span>"
                    f"{sector_html}"
                    f"{mcap_html}"
                    f"</div>"
                    f"<div style='margin-top:7px;display:flex;align-items:center;gap:6px'>"
                    f"{segs}"
                    + (f"<span style='color:{clr};font-weight:700;font-size:13px'>{health_lbl}</span>"
                       if is_etf_row else
                       f"<span style='color:#ffffff;font-weight:700;font-size:13px'>"
                       f"{row['score']}/{row['max']}</span>"
                       f"<span style='color:{clr};font-weight:700;font-size:13px'>{health_lbl}</span>")
                    + f"</div>"
                    f"<div style='margin-top:7px'>{stats_html}</div>"
                    f"</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                # ── extra metrics ─────────────────────────────────────────────
                if not is_etf_row:
                    eg_v  = row.get("earnings_growth")
                    gm_v  = row.get("gross_margins")
                    _em_cols = [c for c in [
                        (T("lbl_pb",  lang), f"{pb_v:.2f}"             if pb_v  is not None else None, "#e6edf3"),
                        (T("lbl_gm",  lang), f"{gm_v*100:.1f}%"        if gm_v  is not None else None, val_color(gm_v)),
                        (("EPS Growth" if lang == "en" else "EPS成長率"),
                                             f"{eg_v*100:+.1f}%"        if eg_v  is not None else None, val_color(eg_v)),
                        (T("lbl_fcf", lang), fmt_val(row.get("free_cashflow"), "b")
                                             if row.get("free_cashflow") is not None else None, "#e6edf3"),
                    ] if c[1] is not None]

                    if _em_cols:
                        _ec = st.columns(len(_em_cols))
                        for i, (lbl, val, col) in enumerate(_em_cols):
                            _ec[i].markdown(
                                f"<div style='background:#161b22;border-radius:6px;padding:8px 10px;"
                                f"text-align:center'>"
                                f"<div style='font-size:10px;color:#8b949e;text-transform:uppercase'>{lbl}</div>"
                                f"<div style='font-size:15px;font-weight:700;color:{col}'>{val}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown("")

                # ── price chart ───────────────────────────────────────────────
                _chart_key = f"fchart_{_ticker}"
                _btn_lbl   = "📈 Show price chart" if lang == "en" else "📈 株価チャートを表示"
                _hide_lbl  = "▲ Hide chart" if lang == "en" else "▲ チャートを非表示"
                if st.session_state.get(_chart_key):
                    _fc_col1, _fc_col2 = st.columns([1, 3])
                    with _fc_col1:
                        if st.button(_hide_lbl, key=f"fhide_{_ticker}"):
                            st.session_state[_chart_key] = False
                            st.rerun()
                    with _fc_col2:
                        _fp_opts = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825}
                        _fp_labels = {"3M": "3 months" if lang == "en" else "3ヶ月",
                                      "6M": "6 months" if lang == "en" else "6ヶ月",
                                      "1Y": "1 year"   if lang == "en" else "1年",
                                      "2Y": "2 years"  if lang == "en" else "2年",
                                      "3Y": "3 years"  if lang == "en" else "3年",
                                      "5Y": "5 years"  if lang == "en" else "5年"}
                        _fp_sel = st.radio(
                            "Period" if lang == "en" else "期間",
                            list(_fp_opts.keys()),
                            index=2,
                            horizontal=True,
                            key=f"fp_{_ticker}",
                            format_func=lambda x: _fp_labels[x],
                        )
                    with st.spinner(f"Loading {_ticker}…"):
                        _pdf = _fetch_fund_chart(_ticker)
                    if _pdf.empty:
                        st.warning("Could not load chart data." if lang == "en"
                                   else "チャートデータを読み込めませんでした。")
                    else:
                        _tz = _pdf.index.tz
                        _cutoff = (pd.Timestamp.today(tz=_tz) - pd.Timedelta(days=_fp_opts[_fp_sel]))
                        _pdf_view = _pdf[_pdf.index >= _cutoff]
                        if _pdf_view.empty:
                            _pdf_view = _pdf
                        st.plotly_chart(_build_fund_chart(_pdf_view, _ticker, _fp_sel),
                                        use_container_width=True)
                else:
                    if st.button(_btn_lbl, key=f"fshow_{_ticker}"):
                        st.session_state[_chart_key] = True
                        st.rerun()

        # ── metric legend ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div class='guide-box'>"
            + ("<b>Score bar key</b>" if lang == "en" else "<b>スコアバーの見方</b>")
            + " &nbsp;—&nbsp; "
            + "<span style='display:inline-block;width:18px;height:8px;border-radius:2px;"
            "background:#23d18b;vertical-align:middle'></span> "
            + ("Pass &nbsp;" if lang == "en" else "合格 &nbsp;")
            + "<span style='display:inline-block;width:18px;height:8px;border-radius:2px;"
            "background:#ff4757;vertical-align:middle'></span> "
            + ("Fail &nbsp;" if lang == "en" else "不合格 &nbsp;")
            + "<span style='display:inline-block;width:18px;height:8px;border-radius:2px;"
            "background:#2d333b;vertical-align:middle'></span> "
            + ("No data" if lang == "en" else "データなし")
            + "<br><br>"
            + (
                "Each of the 8 segments = one check: "
                "<b>P/E · Revenue · Debt · Cash Flow · Profit · Analysts · ROE · Liquidity</b>. "
                "More green = more financially healthy. "
                "Dividend yield shown in <b style='color:#4fc3f7'>blue</b> — it's informational, not scored."
                if lang == "en" else
                "8つのセグメントはそれぞれ1つのチェック項目: "
                "<b>PER · 売上 · 負債 · CF · 利益 · アナリスト · ROE · 流動性</b>。 "
                "緑が多いほど財務健全性が高い。"
                "配当利回りは<b style='color:#4fc3f7'>青</b>で表示 — 参考情報（スコア対象外）。"
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='disclaimer-box'>"
            + (
                "Fundamental data sourced from Yahoo Finance. May be delayed or incomplete. "
                "Not financial advice. Always do your own research."
                if lang == "en" else
                "ファンダメンタルデータはYahoo Financeより取得。遅延または不完全な場合があります。"
                "投資助言ではありません。投資の際は必ずご自身でご確認ください。"
            )
            + "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SCREENER RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Screener Rankings":
    st.markdown(T("sr_title", lang))
    if run_date:
        if lang == "ja":
            mode_str = "シンプルモード — BUY/WATCH/PASS" if simple_mode else "上級モード — 完全なクオンツデータ"
            st.caption(f"データ更新日: {run_date} ({hours_ago})  |  {mode_str}")
        else:
            st.caption(f"Data from {run_date} ({hours_ago})  |  "
                       f"{'Simple mode — BUY/WATCH/PASS' if simple_mode else 'Advanced mode — full quant data'}")

    with st.expander(T("sr_expander", lang), expanded=False):
        if lang == "ja":
            st.markdown("""
**このページの内容:** 本日のシグナル強度でランク付けされたスクリーナー全銘柄。

| シグナル | 意味 |
|--------|------|
| 🟢 **BUY** | ポジティブなテクニカルパターンを検出し、MLモデルの信頼度も高い。追加調査の価値あり。 |
| 🟡 **WATCH** | 一部のポジティブシグナルはあるが、フルシグナルに必要な信頼度に達していない。継続注目。 |
| 🔴 **PASS** | 本日は実行可能なシグナルなし。良い企業でも今日はセットアップが出ていないだけ。 |

**重要な注意事項:**
- BUYシグナルは株価が上がることを**保証するものではありません**。過去に正のエッジがあったパターンを検出したことを意味します。
- シグナルは**価格パターンと統計**に基づいており、企業ニュース・決算・経済イベントは考慮していません。
- **信頼スコア（ML%）**はモデルの確信度を示します。高いほど良いですが、90%でも10%は外れます。
- **推奨サイズ%**は数理的な配分提案であり、多数のポジションへの分散を前提としています。

**ヒント:** フィルターで特定の市場に絞り込み、サイドバーの**初心者/上級者**切替で表示量を調整できます。
""")
        else:
            st.markdown("""
**What you're looking at:** Every stock in our universe, ranked by signal strength today.

| Signal | What it means |
|--------|--------------|
| 🟢 **BUY** | The system sees a positive technical pattern AND the ML model has high confidence. Worth researching further. |
| 🟡 **WATCH** | Some positive signals but not enough confidence to trigger a full signal. Keep an eye on it. |
| 🔴 **PASS** | No actionable signal today. The stock may be a great company but it's just not showing a setup right now. |

**Important things to understand:**
- A BUY signal does **not** mean the stock will go up. It means the system detected a pattern that has historically had a positive edge.
- Signals are based on **price patterns and statistics** — not company news, earnings, or economic events.
- The **confidence score** (ML %) shows how confident the model is. Higher is better, but even 90% confidence is wrong 10% of the time.
- **Suggested size %** is a mathematical recommendation for what portion of a portfolio to consider. It assumes diversification across many positions.

**Tip:** Use the filters to focus on specific markets, and toggle **Simple/Advanced** in the sidebar to control how much detail you see.
""")
    st.markdown("")

    if df_universe.empty:
        st.warning("No screener data found. Run the screener first.")
    else:
        verdict_col = "P7 Verdict" if "P7 Verdict" in df_universe.columns else "Today's Verdict"

        # ── filters ──
        f1, f2, f3 = st.columns(3)
        mkt_filter  = f1.multiselect(T("sr_mkt", lang), ["US", "ASX", "JPX"],
                                      default=["US", "ASX", "JPX"])
        tier_filter = f2.multiselect(T("sr_tier", lang), ["S", "A", "B"],
                                      default=["S", "A", "B"])
        show_all    = f3.toggle(T("sr_show_all", lang), value=False)

        df_filtered = df_universe[df_universe["Market"].isin(mkt_filter)].copy()
        if "Tier" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Tier"].isin(tier_filter)]
        if not show_all:
            df_filtered = df_filtered[
                df_filtered[verdict_col].str.contains("TRADE", na=False) |
                df_filtered[verdict_col].str.contains("HOLD", na=False)
            ]

        if df_filtered.empty:
            st.info(T("sr_no_signals", lang))
        else:
            # ── BUY signals first ──
            trade_mask = (df_filtered[verdict_col].str.contains("TRADE", na=False) &
                          ~df_filtered[verdict_col].str.contains("STAND", na=False))
            df_trade   = df_filtered[trade_mask].sort_values("ML Score", ascending=False)
            df_rest    = df_filtered[~trade_mask].sort_values("Score", ascending=False)

            _sr_fc = _load_fund_cache_raw()

            if not df_trade.empty:
                st.markdown(f"#### {T('sr_buy', lang, n=len(df_trade))}")
                _render_table(df_trade, verdict_col, simple_mode, lang, fund_cache=_sr_fc)

            if not df_rest.empty and show_all:
                st.markdown(f"#### {T('sr_watch', lang, n=len(df_rest))}")
                _render_table(df_rest, verdict_col, simple_mode, lang, fund_cache=_sr_fc)
            elif not df_rest.empty:
                st.caption(T("sr_hidden", lang, n=len(df_rest)))

    st.markdown(
        f"<div class='disclaimer-box' style='margin-top:24px'>{T('sr_disclaimer', lang)}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TRACK RECORD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Track Record":
    st.markdown(T("tr_title", lang))
    st.caption(T("tr_sub", lang))

    with st.expander(T("tr_expander", lang), expanded=False):
        if lang == "ja":
            st.markdown("""
**ペーパートレードとは？** 実際の市場価格を使って取引をシミュレーションしますが、実際のお金は使いません。
スクリーナーがBUYシグナルを生成するたびに「ペーパートレード」として記録し、実際に売買した場合どうなったかを追跡します。

**なぜ重要なのか？** システムの実際のパフォーマンスを最も正直に示す方法です。多くのスクリーナーはバックテスト結果のみを公開しますが、ここではシグナルが生成された日からのリアルタイム結果を公開しています。

**用語説明:**
- **勝率** — 利益が出たトレードの割合。50%なら半分が利益、半分が損失。
- **平均損益** — トレードあたりの平均損益（%）。
- **損切り** — 損失が拡大しすぎた場合に自動的にポジションをクローズ。
- **保有完了** — 20日間の保有期間を満了してクローズ。
- **シグナル反転** — スクリーナーが銘柄の見方を変えたため早期退出。

**27ランの説明:** 異なるルールセットで同時にスクリーナーを実行し、最良のアプローチを検証しています。
ラン1〜3はベースライン。ラン4〜11は様々な改善を検証。ラン12〜16はファンダメンタル分析を追加フィルターとして使用。
ラン17〜19は確認された負け戦略をブロック。ラン20は勝ち戦略のみ許可。ラン21は10日間の保有期間を検証します。
ラン22はMLスコア高閾値（75以上）のみ許可。ラン23〜27は実績ある3戦略（RSI・MA Crossover・Mean Reversion）を軸に、ファンダメンタルゲート・出来高サージ・高ML閾値を組み合わせた実験的構成です。
ラン28〜31はPhase 10（エントリータイミング改善）：プルバックフィルター・RSIゲート・クールダウンルール。ラン32〜36はPhase 11（動的エグジット）：利確タイミング・トレーリングストップ・ATRベースのフロアを検証。

**正直な注意:** ベースランでは約300件のクローズドトレードがあります。新しいランはまだデータ蓄積中です。統計的に十分なサンプルとは言えません。トラックレコードは早期指標として扱ってください。
""")
        else:
            st.markdown("""
**What is paper trading?** Paper trading means we simulate trades using real market prices, but no real money is involved.
Every time the screener generates a BUY signal, we record it as a "paper trade" — tracking what would have happened if someone actually bought and sold at those prices.

**Why does this matter?** It's the most honest way to show you how the system actually performs. Many screeners only show their backtested results (testing on historical data the system already "saw"). We show live, forward-looking results from the day each signal was generated.

**Key terms explained:**
- **Win rate** — the percentage of trades that made money. 50% means half made money, half lost.
- **Avg P&L** — the average profit or loss per trade, as a percentage.
- **Stop-loss** — we automatically close a trade if it falls too far, to limit losses.
- **Hold Complete** — the trade ran its full 20-day course and was then closed.
- **Signal Reversal** — the screener changed its view on the stock, so we exited early.

**The 47 runs explained:** We run the screener with different rule sets simultaneously to test which approach works best.
Runs 1–3 are our baseline. Runs 4–11 test different improvements. Runs 12–16 add fundamental analysis as an extra filter.
Runs 17–19 block confirmed losing strategies. Run 20 is "winners only". Run 21 tests a 10-day hold. Runs 22–27 focus on the proven trio (RSI, MA Crossover, Mean Reversion) with fundamental and volume gates.
Runs 28–31 are Phase 10 (entry quality): pullback filter, RSI gate, cooldown. Runs 32–36 are Phase 11 (dynamic exits): trailing stops, take-profit variants, ATR-based floors.
Run 37 is speculative-only: quantum computing, space, AI, and nuclear stocks with wide 20% stop.
Runs 38–42 isolate strategy families: R38 trend-following only, R39 momentum only, R40 mean reversion only. This tells us which *type* of strategy works best in current conditions.
Runs 43–47 test earnings blackout (skipping entry within 5 days of earnings) and extreme ML conviction (ML≥90).
The leaderboard above updates automatically — the top 3 are the runs producing the best real results right now.

**Honest caveat:** We currently have around 300 closed trades in the baseline runs, with newer runs still accumulating data. This is not yet a statistically large enough sample to draw firm conclusions. Treat the track record as an early indicator, not proof.
""")
    st.markdown("")

    # ── run leaderboard — always visible, shows which runs stand out ─────────
    if not df_history.empty:
        _lb_rows = []
        for _r in sorted(df_history["run"].unique()):
            _rdf = df_history[df_history["run"] == _r]
            _n   = len(_rdf)
            if _n < 3:
                continue   # skip runs with almost no data
            _w   = int(_rdf["win"].sum())
            _wr  = _w / _n * 100
            _avg = _rdf["pnl_pct"].mean()
            _win_avg  = _rdf[_rdf["win"]]["pnl_pct"].mean() if _w > 0 else 0
            _loss_avg = _rdf[~_rdf["win"]]["pnl_pct"].mean() if _n - _w > 0 else 0
            _pf  = abs(_win_avg / _loss_avg) if _loss_avg != 0 else float("inf")
            # Sortino
            if _n >= 5:
                _rets = _rdf["pnl_pct"].values / 100
                _hold = _rdf["days_held"].mean() if "days_held" in _rdf.columns else 20
                _af   = 252 / max(float(_hold), 1)
                _neg  = _rets[_rets < 0]
                _ds   = _neg.std() * np.sqrt(_af) if len(_neg) > 1 else np.nan
                _so   = (_rets.mean() * _af) / _ds if _ds and not np.isnan(_ds) else np.nan
            else:
                _so = np.nan
            _lb_rows.append({
                "run": _r,
                "label": RUN_CONFIGS.get(_r, f"Run {_r}"),
                "trades": _n,
                "win_rate": _wr,
                "avg_pnl": _avg,
                "profit_factor": _pf,
                "sortino": _so,
            })

        if _lb_rows:
            _lb_df = pd.DataFrame(_lb_rows).sort_values("avg_pnl", ascending=False).reset_index(drop=True)

            # podium — top 3
            _medals = ["🥇", "🥈", "🥉"]
            _top3   = _lb_df.head(3)
            _pod_cols = st.columns(3)
            for _i, (_col, (_, _row)) in enumerate(zip(_pod_cols, _top3.iterrows())):
                _c = "#3fb950" if _row["avg_pnl"] > 0 else "#f85149"
                _pf_s = f"{_row['profit_factor']:.2f}" if not np.isinf(_row["profit_factor"]) else "∞"
                _so_s = f"{_row['sortino']:.2f}" if not np.isnan(_row["sortino"]) else "—"
                _col.markdown(
                    f"<div style='background:#161b22;border:1px solid {_c};"
                    f"border-radius:10px;padding:14px 12px;text-align:center'>"
                    f"<div style='font-size:22px'>{_medals[_i]}</div>"
                    f"<div style='font-size:16px;font-weight:bold;color:{_c}'>"
                    f"R{int(_row['run'])}</div>"
                    f"<div style='font-size:11px;color:#8b949e;margin:4px 0 8px'>"
                    f"{str(_row['label'])[:40]}{'…' if len(str(_row['label'])) > 40 else ''}</div>"
                    f"<div style='font-size:18px;font-weight:800;color:{_c}'>"
                    f"{_row['avg_pnl']:+.2f}%</div>"
                    f"<div style='font-size:11px;color:#8b949e'>avg P&L · {_row['trades']} trades</div>"
                    f"<div style='font-size:11px;color:#8b949e;margin-top:4px'>"
                    f"WR {_row['win_rate']:.0f}% · PF {_pf_s} · Sortino {_so_s}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("")

            # full leaderboard table
            _lb_label = "All Runs — sorted by Avg P&L" if lang == "en" else "全ラン — 平均損益順"
            with st.expander(_lb_label, expanded=False):
                _display_rows = []
                for _, _row in _lb_df.iterrows():
                    _pf_s = f"{_row['profit_factor']:.2f}" if not np.isinf(_row["profit_factor"]) else "∞"
                    _so_s = f"{_row['sortino']:.2f}" if not np.isnan(_row["sortino"]) else "—"
                    _display_rows.append({
                        ("Run" if lang == "en" else "ラン"): f"R{int(_row['run'])}",
                        ("Config" if lang == "en" else "設定"): str(_row["label"])[:55],
                        ("Trades" if lang == "en" else "取引数"): int(_row["trades"]),
                        ("Win %" if lang == "en" else "勝率"): f"{_row['win_rate']:.1f}%",
                        ("Avg P&L" if lang == "en" else "平均損益"): f"{_row['avg_pnl']:+.2f}%",
                        ("Profit Factor" if lang == "en" else "PF"): _pf_s,
                        ("Sortino" if lang == "en" else "ソルティノ"): _so_s,
                    })
                _disp_df = pd.DataFrame(_display_rows)
                _avg_col = "Avg P&L" if lang == "en" else "平均損益"

                def _lb_color(v):
                    if not isinstance(v, str): return ""
                    if v.startswith("+"):  return "color:#3fb950;font-weight:bold"
                    if v.startswith("-"):  return "color:#f85149;font-weight:bold"
                    return ""

                st.dataframe(
                    _disp_df.style.map(_lb_color, subset=[_avg_col]),
                    use_container_width=True, hide_index=True,
                )
        st.markdown("---")

    if df_history.empty:
        st.info(T("tr_no_trades", lang))
    else:
        dc = df_history.copy()

        # ── run filter — only show runs that have actual data (open or closed trades) ─
        _runs_with_data = set(dc["run"].unique().tolist())
        _open_by_run    = (df_open.groupby("run").size().to_dict()
                           if not df_open.empty and "run" in df_open.columns else {})
        _runs_active    = sorted(r for r in RUN_CONFIGS.keys()
                                 if r in _runs_with_data or _open_by_run.get(r, 0) > 0)
        all_runs = _runs_active if _runs_active else sorted(RUN_CONFIGS.keys())

        def _run_label(r):
            cfg = RUN_CONFIGS.get(r, "")
            if r not in _runs_with_data:
                n_open = _open_by_run.get(r, 0)
                suffix = (f" — {n_open} open, no closed trades yet"
                          if lang == "en" else
                          f" — {n_open}件保有中・決済済みなし")
                return f"Run {r}{suffix}"
            return f"Run {r} — {cfg}"

        run_labels = [T("tr_run_all", lang)] + [_run_label(r) for r in all_runs]
        selected_label = st.selectbox(T("tr_run_filter", lang), run_labels, index=0)

        selected_run = None
        if selected_label != T("tr_run_all", lang):
            selected_run = int(selected_label.split()[1])
            dc = dc[dc["run"] == selected_run].copy()

        # handle runs with no closed trades yet
        if dc.empty and selected_run is not None:
            n_open = _open_by_run.get(selected_run, 0)
            _msg = (f"Run {selected_run} has **{n_open} open position(s)** but no closed trades yet — "
                    "check back once some trades exit."
                    if lang == "en" else
                    f"Run {selected_run}は**{n_open}件の保有ポジション**がありますが、まだ決済済みトレードはありません。")
            st.info(_msg)

        if dc.empty:
            n_total = n_wins = n_stops = 0
            win_rate = avg_pnl = avg_win = avg_loss = pf = 0.0
            wl_str = "0W / 0L" if lang == "en" else "0勝 / 0敗"
            stops_sub = "—"
            sortino_str = pf_str = "—"
        else:
            n_total  = len(dc)
            n_wins   = int(dc["win"].sum())
            win_rate = n_wins / n_total * 100
            avg_pnl  = dc["pnl_pct"].mean()
            avg_win  = dc[dc["win"]]["pnl_pct"].mean() if n_wins > 0 else 0
            avg_loss = dc[~dc["win"]]["pnl_pct"].mean() if n_total - n_wins > 0 else 0
            pf       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
            n_stops  = int((dc["exit_reason"] == "STOP_LOSS").sum())
            wl_str   = f"{n_wins}W / {n_total-n_wins}L" if lang == "en" else f"{n_wins}勝 / {n_total-n_wins}敗"
            stops_sub = f"{n_stops/n_total*100:.0f}% of trades" if lang == "en" else f"全体の{n_stops/n_total*100:.0f}%"
            pf_str   = f"{pf:.2f}" if not np.isinf(pf) else "∞"
            # Sortino ratio: annualised return / downside deviation (only negative returns)
            # Uses avg hold period to annualise; needs ≥5 closed trades to be meaningful.
            if n_total >= 5:
                _rets = dc["pnl_pct"].values / 100
                _avg_hold = dc["days_held"].mean() if "days_held" in dc.columns else 20
                _ann_factor = 252 / max(float(_avg_hold), 1)
                _ann_ret    = _rets.mean() * _ann_factor
                _neg_rets   = _rets[_rets < 0]
                _ds_std     = _neg_rets.std() * np.sqrt(_ann_factor) if len(_neg_rets) > 1 else np.nan
                _sortino    = _ann_ret / _ds_std if _ds_std and not np.isnan(_ds_std) else np.nan
                sortino_str = f"{_sortino:.2f}" if not np.isnan(_sortino) else "—"
            else:
                sortino_str = "—"

        # ── summary metrics row 1 ──
        cols = st.columns(6)
        for col, label, val, sub in [
            (cols[0], T("tr_m1", lang), str(n_total),        T("tr_m1s", lang)),
            (cols[1], T("tr_m2", lang), f"{win_rate:.1f}%",  wl_str),
            (cols[2], T("tr_m3", lang), f"{avg_pnl:+.2f}%",  T("tr_m3s", lang)),
            (cols[3], T("tr_m4", lang), f"{avg_win:+.2f}%",  T("tr_m4s", lang)),
            (cols[4], T("tr_m5", lang), f"{avg_loss:+.2f}%", T("tr_m5s", lang)),
            (cols[5], T("tr_m6", lang), str(n_stops),        stops_sub),
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

        # ── summary metrics row 2: risk-adjusted metrics ──
        _pf_label = "Profit Factor" if lang == "en" else "プロフィットファクター"
        _pf_sub   = "gross wins ÷ gross losses" if lang == "en" else "総利益 ÷ 総損失"
        _so_label = "Sortino Ratio" if lang == "en" else "ソルティノレシオ"
        _so_sub   = "ann. return ÷ downside vol" if lang == "en" else "年率リターン ÷ 下方ボラ"
        _so_note  = "vs Sharpe (penalises all vol)" if lang == "en" else "シャープ比は全変動を評価"
        _r2c1, _r2c2, _r2c3 = st.columns([1, 1, 4])
        _pf_color = "#3fb950" if pf_str not in ("—", "0.00") and pf != 0 and (pf == float("inf") or pf >= 1.0) else "#f85149"
        _so_color = "#3fb950" if sortino_str not in ("—",) and float(sortino_str) > 0 else "#f85149" if sortino_str != "—" else "#e6edf3"
        _r2c1.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{_pf_label}</div>"
            f"<div class='value' style='color:{_pf_color}'>{pf_str}</div>"
            f"<div class='sub'>{_pf_sub}</div>"
            f"</div>", unsafe_allow_html=True,
        )
        _r2c2.markdown(
            f"<div class='metric-card'>"
            f"<div class='label'>{_so_label}</div>"
            f"<div class='value' style='color:{_so_color}'>{sortino_str}</div>"
            f"<div class='sub'>{_so_note}</div>"
            f"</div>", unsafe_allow_html=True,
        )

        # ── investment simulator ─────────────────────────────────────────────────
        st.markdown(f"### {T('tr_sim_header', lang)}")
        _sim_c1, _sim_c2 = st.columns([1, 3])
        with _sim_c1:
            _sim_amt = st.number_input(
                T("tr_sim_label", lang),
                min_value=100, max_value=10_000_000,
                value=1000, step=100, key="tr_sim_amt",
            )
        if not dc.empty:
            _sim_sorted = dc.dropna(subset=["exit_date"]).copy()
            _sim_sorted["exit_date"] = pd.to_datetime(_sim_sorted["exit_date"], errors="coerce")
            _sim_sorted = _sim_sorted.dropna(subset=["exit_date"]).sort_values("exit_date")

            _sim_val = float(_sim_amt)
            _sim_curve = [_sim_val]
            for _p in _sim_sorted["pnl_pct"]:
                _sim_val *= (1 + _p / 100)
                _sim_curve.append(_sim_val)

            _sim_ret  = (_sim_val / _sim_amt - 1) * 100
            _sim_clr  = "#3fb950" if _sim_val >= _sim_amt else "#f85149"
            _sim_icon = "📈" if _sim_val >= _sim_amt else "📉"
            _sim_arrow = "▲" if _sim_val >= _sim_amt else "▼"

            _card_c, _chart_c = st.columns([1, 2])
            with _card_c:
                st.markdown(
                    f"<div style='background:#161b22;border:2px solid {_sim_clr};"
                    f"border-radius:12px;padding:22px 20px;text-align:center;margin-top:8px'>"
                    f"<div style='font-size:30px'>{_sim_icon}</div>"
                    f"<div style='font-size:13px;color:#8b949e;margin-top:6px'>"
                    f"${_sim_amt:,.0f} {T('tr_sim_becomes', lang)}</div>"
                    f"<div style='font-size:32px;font-weight:800;color:{_sim_clr};line-height:1.1'>"
                    f"${_sim_val:,.2f}</div>"
                    f"<div style='font-size:17px;color:{_sim_clr};font-weight:600;margin-top:2px'>"
                    f"{_sim_arrow} {_sim_ret:+.2f}%</div>"
                    f"<div style='font-size:11px;color:#8b949e;margin-top:6px'>"
                    f"{len(_sim_sorted)} {'closed trades' if lang == 'en' else '決済済み取引'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with _chart_c:
                _start_dt = _sim_sorted["exit_date"].iloc[0] - pd.Timedelta(days=20)
                _x_dates  = [_start_dt] + list(_sim_sorted["exit_date"])
                _fill_clr = "rgba(63,185,80,0.08)" if _sim_val >= _sim_amt else "rgba(248,81,73,0.08)"
                _fig_sim  = go.Figure()
                _fig_sim.add_trace(go.Scatter(
                    x=_x_dates, y=_sim_curve,
                    mode="lines", line=dict(color=_sim_clr, width=2),
                    fill="tozeroy", fillcolor=_fill_clr, showlegend=False,
                    hovertemplate="$%{y:,.2f}<extra></extra>",
                ))
                _fig_sim.add_hline(y=float(_sim_amt), line_dash="dash",
                                   line_color="#8b949e", opacity=0.5)
                _fig_sim.update_layout(
                    height=220, autosize=True,
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font_color="#e6edf3",
                    margin=dict(l=45, r=8, t=10, b=30),
                    xaxis=dict(gridcolor="#21262d"),
                    yaxis=dict(gridcolor="#21262d", tickprefix="$"),
                )
                st.plotly_chart(_fig_sim, use_container_width=True)
            st.caption(T("tr_sim_disc", lang))
        else:
            st.info("No closed trades yet — simulator will activate once trades close." if lang == "en"
                    else "まだ決済済み取引がありません。取引が決済されると有効になります。")

        st.markdown("---")

        col_l, col_r = st.columns([2, 1])

        with col_l:
            # ── equity curve ──
            st.markdown(T("tr_equity", lang))
            st.caption(T("tr_eq_sub", lang))
            dc_sorted = dc.dropna(subset=["exit_date"]).copy()
            dc_sorted["exit_date"] = pd.to_datetime(dc_sorted["exit_date"], errors="coerce")
            dc_sorted = dc_sorted.dropna(subset=["exit_date"]).sort_values("exit_date")
            dc_sorted["cumulative"] = dc_sorted["pnl_pct"].cumsum()
            dc_sorted["trade_num"]  = range(1, len(dc_sorted) + 1)

            dot_colors = ["#3fb950" if p > 0 else "#f85149" for p in dc_sorted["pnl_pct"]]
            hover_text = [f"{a} | {s} | {p:+.1f}% | {d}"
                          for a, s, p, d in zip(dc_sorted["asset"], dc_sorted["strategy"],
                                                dc_sorted["pnl_pct"],
                                                dc_sorted["exit_date"].dt.strftime("%b %d"))]
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=dc_sorted["exit_date"], y=dc_sorted["cumulative"],
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=6, color=dot_colors),
                text=hover_text, hoverinfo="text", showlegend=False,
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
            st.markdown(T("tr_strategy", lang))
            ss = (dc.groupby("strategy")
                  .agg(trades=("pnl_pct","count"),
                       wins=("win","sum"),
                       avg_pnl=("pnl_pct","mean"))
                  .reset_index()
                  .sort_values("avg_pnl", ascending=False))
            ss["win_rate"] = ss["wins"] / ss["trades"] * 100
            ss["Signal"] = ss["avg_pnl"].apply(
                lambda v: T("tr_positive", lang) if v > 0 else T("tr_negative", lang)
            )
            pnl_col = T("tr_col_pnl", lang)
            wr_col  = T("tr_col_wr",  lang)
            ss_display = ss[["strategy","trades","win_rate","avg_pnl","Signal"]].copy()
            ss_display.columns = [T("tr_col_strat",lang), T("tr_col_trades",lang),
                                  wr_col, pnl_col, T("tr_col_res",lang)]
            ss_display[wr_col]  = ss_display[wr_col].round(1)
            ss_display[pnl_col] = ss_display[pnl_col].round(2)

            def _color_pnl(v):
                if pd.isna(v): return ""
                return "color:#3fb950;font-weight:bold" if v > 0 else "color:#f85149;font-weight:bold"
            st.dataframe(
                ss_display.style
                    .map(_color_pnl, subset=[pnl_col])
                    .format({pnl_col: "{:+.2f}%", wr_col: "{:.1f}%"}),
                use_container_width=True, hide_index=True,
            )

        with col_r:
            # ── exit breakdown ──
            st.markdown(T("tr_exit", lang))
            exit_counts = dc["exit_reason"].value_counts()
            exit_colors = {
                "STOP_LOSS":       "#f85149",
                "HOLD_COMPLETE":   "#3fb950",
                "SIGNAL_REVERSAL": "#58a6ff",
                "TAKE_PROFIT":     "#39d353",
            }
            fig_pie = go.Figure(go.Pie(
                labels=[l.replace("_"," ").title() for l in exit_counts.index],
                values=exit_counts.values.tolist(),
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
            st.markdown(T("tr_market", lang))
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
                    f"{int(row['trades'])} {T('tr_trades',lang)} · {row['win_rate']:.0f}% {T('tr_wr',lang)}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(T("tr_recent", lang))
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

        # ── snapshot: if all closed now ─────────────────────────────────────────
        st.markdown("---")
        with st.expander(T("tr_snap", lang), expanded=False):
            st.caption(T("tr_snap_sub", lang))
            snap_rows = []
            all_runs_for_snap = sorted(df_history["run"].unique()) if not df_history.empty else []
            for r in all_runs_for_snap:
                rdf_c = df_history[df_history["run"] == r]
                n_c   = len(rdf_c)
                rdf_o = df_open[df_open["run"] == r] if not df_open.empty else pd.DataFrame()
                n_o   = len(rdf_o)
                all_pnls = list(rdf_c["pnl_pct"]) + (list(rdf_o["pnl_pct"]) if n_o > 0 else [])
                n_all    = len(all_pnls)
                if n_all == 0:
                    continue
                wr_all  = sum(1 for p in all_pnls if p > 0) / n_all * 100
                avg_all = sum(all_pnls) / n_all
                tot_all = sum(all_pnls)
                snap_rows.append({
                    T("tr_col_run",       lang): f"R{r}",
                    T("tr_col_cfg",       lang): RUN_CONFIGS.get(r, ""),
                    T("tr_col_closed",    lang): n_c,
                    T("tr_col_open_n",    lang): n_o,
                    T("tr_col_combo_wr",  lang): round(wr_all,  1),
                    T("tr_col_combo_avg", lang): round(avg_all, 2),
                    T("tr_col_total",     lang): round(tot_all, 1),
                })
            if snap_rows:
                snap_df    = pd.DataFrame(snap_rows)
                avg_col    = T("tr_col_combo_avg", lang)
                total_col  = T("tr_col_total",     lang)
                wr_col_s   = T("tr_col_combo_wr",  lang)

                def _snap_color(v):
                    if not isinstance(v, (int, float)): return ""
                    return "color:#3fb950;font-weight:bold" if v > 0 else "color:#f85149;font-weight:bold"

                st.dataframe(
                    snap_df.style
                        .map(_snap_color, subset=[avg_col, total_col])
                        .format({wr_col_s: "{:.1f}%", avg_col: "{:+.2f}%", total_col: "{:+.1f}%"}),
                    use_container_width=True, hide_index=True,
                )
                best = max(snap_rows, key=lambda x: x[total_col])
                st.caption(
                    f"Best run: **{best[T('tr_col_run', lang)]}** ({best[T('tr_col_cfg', lang)]}) "
                    f"at **{best[total_col]:+.1f}%** total · {best[wr_col_s]:.1f}% win rate"
                    if lang == "en" else
                    f"最良ラン: **{best[T('tr_col_run', lang)]}** ({best[T('tr_col_cfg', lang)]}) "
                    f"合計 **{best[total_col]:+.1f}%** · 勝率 {best[wr_col_s]:.1f}%"
                )

        # ── advanced: per-run comparison table ─────────────────────────────────
        if is_advanced and selected_label == T("tr_run_all", lang):
            st.markdown("---")
            st.markdown(T("tr_per_run", lang))
            st.caption(T("tr_per_run_sub", lang))
            run_rows = []
            full_dc = df_history.copy()
            for r in sorted(full_dc["run"].unique()):
                rdf = full_dc[full_dc["run"] == r]
                rt  = len(rdf)
                rw  = int(rdf["win"].sum())
                rp  = rdf["pnl_pct"].mean()
                rwin = rdf[rdf["win"]]["pnl_pct"].mean() if rw > 0 else 0
                rloss = rdf[~rdf["win"]]["pnl_pct"].mean() if rt - rw > 0 else 0
                rpf  = abs(rwin / rloss) if rloss != 0 else float("inf")
                run_rows.append({
                    T("tr_col_run", lang): f"R{r}",
                    T("tr_col_cfg", lang): RUN_CONFIGS.get(r, ""),
                    T("tr_col_trades", lang): rt,
                    T("tr_col_wr", lang): round(rw / rt * 100, 1) if rt > 0 else 0,
                    T("tr_col_pnl", lang): round(rp, 2),
                    T("tr_col_pf", lang): round(rpf, 2) if rpf != float("inf") else "∞",
                })
            run_df = pd.DataFrame(run_rows)
            pnl_c = T("tr_col_pnl", lang)
            wr_c  = T("tr_col_wr",  lang)

            def _rcolor(v):
                if pd.isna(v) or not isinstance(v, (int, float)): return ""
                return "color:#3fb950;font-weight:bold" if v > 0 else "color:#f85149;font-weight:bold"

            st.dataframe(
                run_df.style
                    .map(_rcolor, subset=[pnl_c])
                    .format({pnl_c: "{:+.2f}%", wr_c: "{:.1f}%"}),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")
    st.markdown(
        f"<div class='disclaimer-box'>{T('tr_disclaimer', lang)}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STOCK CHART
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Stock Chart":
    # ══════════════════════════════════════════════════════════════════════════
    #  STOCK CHART
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(T("sc_title", lang))

    with st.expander(T("sc_expander", lang), expanded=False):
        if lang == "ja":
            st.markdown("""
**このページの内容:** スクリーナーの任意の銘柄のチャート（3ヶ月〜2年）に、本日のシグナルを生成した具体的なインジケーターを重ねて表示します。

**色付きのラインとバンド**はスクリーナーが使用するテクニカル指標です。戦略ごとに異なるインジケーターを使用します。

**ローソク足**（緑/赤のバー）は各日の価格レンジを示します:
- バーの上端 = その日の最高値
- バーの下端 = その日の最安値
- 緑 = 終値が始値より高い
- 赤 = 終値が始値より低い

**黄色の点線**は本日の日付 = シグナルが生成された時点を示します。

**チャート下の統計:** RSI 30以下 = 売られすぎ（買いの機会の可能性）。RSI 70以上 = 買われすぎ（過熱の可能性）。
""")
        else:
            st.markdown("""
**What you're looking at:** A price chart (3 months to 2 years) for any stock in our universe, with the specific indicators
that drove today's signal overlaid on top.

**The coloured lines and bands** are the technical indicators the screener uses. Each strategy has different indicators —
the "What to look for" box below the signal card explains exactly what to focus on for each strategy.

**Candlesticks** (the green/red bars) show each day's price range:
- The top of the bar = the day's highest price
- The bottom of the bar = the day's lowest price
- Green = price closed higher than it opened
- Red = price closed lower than it opened

**The dashed yellow line** marks today's date — the point where the signal was generated.

**Quick stats below the chart:** RSI below 30 = oversold (potentially a buy opportunity). RSI above 70 = overbought (potentially overextended).
""")
    st.markdown("")
    st.markdown(T("sc_desc", lang))

    if df_universe.empty:
        st.info(T("sc_no_data_loaded", lang))
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
            st.error(T("sc_no_asset_col", lang))
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
                T("sc_market", lang),
                [T("sc_show_all_opt", lang)] + (
                    sorted(df_pick[market_col].dropna().unique().tolist())
                    if market_col else []
                ), key="chart_market"
            )
        with col_f2:
            verdict_filter = st.selectbox(
                T("sc_show", lang),
                [T("sc_show_all_opt", lang), T("sc_show_buy", lang), T("sc_show_watch", lang)],
                key="chart_verdict"
            )

        filtered = df_pick.copy()
        _all_opt = T("sc_show_all_opt", lang)
        _buy_opt = T("sc_show_buy", lang)
        _watch_opt = T("sc_show_watch", lang)
        if market_filter != _all_opt and market_col:
            filtered = filtered[filtered[market_col] == market_filter]
        if verdict_filter == _buy_opt and verdict_col:
            filtered = filtered[filtered[verdict_col].astype(str).str.upper()
                                .str.contains("TRADE|BUY")]
        elif verdict_filter == _watch_opt and verdict_col:
            filtered = filtered[filtered[verdict_col].astype(str).str.upper()
                                .str.contains("WATCH|HOLD")]

        asset_options = filtered[asset_col].dropna().unique().tolist()
        if not asset_options:
            st.info(T("sc_no_match", lang))
            st.stop()

        # build ticker → company name lookup from fundamentals cache (no API call)
        _fund_rows = load_fundamental_rankings()
        _name_map  = {r["ticker"]: r["name"] for r in _fund_rows if r.get("name") and r["name"] != r["ticker"]}

        selected_asset = st.selectbox(
            T("sc_select_stock", lang),
            asset_options,
            format_func=lambda t: f"{t}  —  {_name_map[t]}" if t in _name_map else t,
            key="chart_asset",
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

        _plain_dict = STRATEGY_PLAIN_JA if lang == "ja" else STRATEGY_PLAIN
        strat_desc = _plain_dict.get(strategy, T("sc_strat_unknown", lang))
        what_desc  = STRATEGY_WHAT_TO_LOOK_FOR.get(strategy, "")

        card_cols = st.columns([1, 1, 1, 1])
        with card_cols[0]:
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>{T('sc_sig_signal', lang)}</div>"
                f"<div style='font-size:20px;font-weight:700;color:{badge_clr}'>{badge_lbl}</div>"
                f"</div>", unsafe_allow_html=True)
        with card_cols[1]:
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>{T('sc_sig_strategy', lang)}</div>"
                f"<div style='font-size:15px;font-weight:600;color:#e6edf3'>{strategy}</div>"
                f"</div>", unsafe_allow_html=True)
        with card_cols[2]:
            ml_text = f"{float(ml_score):.0f}" if ml_score is not None else "—"
            ml_clr  = "#3fb950" if ml_score is not None and float(ml_score) >= 70 else "#d29922"
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>{T('sc_sig_ml', lang)}</div>"
                f"<div style='font-size:20px;font-weight:700;color:{ml_clr}'>{ml_text}</div>"
                f"</div>", unsafe_allow_html=True)
        with card_cols[3]:
            size_text = f"{float(size_pct):.1f}%" if size_pct is not None else "—"
            st.markdown(
                f"<div style='background:#161b22;border-radius:8px;padding:14px;text-align:center'>"
                f"<div style='font-size:11px;color:#8b949e;text-transform:uppercase'>{T('sc_sig_size', lang)}</div>"
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
                f"<b style='color:#d29922'>{T('sc_what_to_look', lang)}</b> "
                f"<span style='color:#c9d1d9'>{what_desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── fetch price data & draw chart ─────────────────────────────────────
        with st.spinner(f"Loading {selected_asset} chart…"):
            price_df = load_price_data(selected_asset)

        if price_df.empty:
            st.warning(T("sc_price_err", lang, ticker=selected_asset))
        else:
            # period selector
            _period_opts  = ["3M", "6M", "1Y", "2Y"]
            _period_labels = {"3M": "3 months" if lang == "en" else "3ヶ月",
                              "6M": "6 months" if lang == "en" else "6ヶ月",
                              "1Y": "1 year"   if lang == "en" else "1年",
                              "2Y": "2 years"  if lang == "en" else "2年"}
            _period_days  = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
            _p_col, _ = st.columns([2, 5])
            with _p_col:
                _sel_period = st.radio(
                    T("sc_period_label", lang),
                    _period_opts,
                    index=1,           # default: 6M
                    horizontal=True,
                    key="sc_period",
                    format_func=lambda x: _period_labels[x],
                )
            # slice to selected window
            _cutoff = pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=_period_days[_sel_period])
            price_df_view = price_df[price_df.index >= _cutoff]
            if price_df_view.empty:
                price_df_view = price_df  # fallback if not enough history

            fig = build_strategy_chart(price_df_view, strategy, selected_asset, market)
            st.plotly_chart(fig, use_container_width=True)

            # quick stats row below the chart (based on selected period)
            close = price_df_view["Close"].squeeze().dropna()
            if len(close) >= 20:
                s_cols = st.columns(5)
                last   = float(close.iloc[-1])
                _view_high = price_df_view["High"].squeeze() if "High" in price_df_view.columns else close
                _view_low  = price_df_view["Low"].squeeze()  if "Low"  in price_df_view.columns else close
                hi_period  = float(_view_high.max())
                ma20       = float(close.rolling(20).mean().iloc[-1])
                rsi14      = float(_rsi(close).iloc[-1]) if not np.isnan(_rsi(close).iloc[-1]) else 0
                pct_hi     = (last / hi_period - 1) * 100
                _hi_label  = (_sel_period + (" High" if lang == "en" else " 高値"))
                def _stat(label, val, colour="#e6edf3"):
                    return (f"<div style='background:#161b22;border-radius:8px;padding:10px;text-align:center'>"
                            f"<div style='font-size:10px;color:#8b949e;text-transform:uppercase'>{label}</div>"
                            f"<div style='font-size:16px;font-weight:700;color:{colour}'>{val}</div>"
                            f"</div>")
                rsi_clr = "#3fb950" if rsi14 < 30 else ("#f85149" if rsi14 > 70 else "#e6edf3")
                s_cols[0].markdown(_stat(T("sc_stat_last", lang), f"{last:.2f}"), unsafe_allow_html=True)
                s_cols[1].markdown(_stat(T("sc_stat_ma20", lang), f"{ma20:.2f}",
                    "#3fb950" if last > ma20 else "#f85149"), unsafe_allow_html=True)
                s_cols[2].markdown(_stat(T("sc_stat_rsi14", lang), f"{rsi14:.1f}", rsi_clr), unsafe_allow_html=True)
                s_cols[3].markdown(_stat(_hi_label, f"{hi_period:.2f}"), unsafe_allow_html=True)
                s_cols[4].markdown(_stat(T("sc_stat_pct_high", lang),
                    f"{pct_hi:.1f}%", "#3fb950" if pct_hi > -5 else "#d29922"),
                    unsafe_allow_html=True)

        # disclaimer at the bottom
        st.markdown("---")
        st.caption(T("sc_chart_disc", lang))


elif page == "ℹ️ About & Disclaimer":
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(T("ab_title", lang))
        if lang == "ja":
            st.markdown("""
Goofy Screenerは、ブルームバーグやクオンツファンドに費用をかけずに、体系的かつデータドリブンな市場分析を求める個人投資家向けの無料クオンツ株スクリーナーです。

**誰が作ったの？**
クオンツトレーディングをゼロから道具を作りながら学んでいるクイーンズランド大学（UQ）のファイナンス学生。すべての戦略・フィルター・コードがカスタムビルドされています。

**なぜ無料なの？**
本物を作ることが最良の学習方法であり、損失も含めて結果を透明に公開することが最良の検証方法だからです。

---

### 方法論

**対象銘柄:** 米国（155銘柄）・ASX（56銘柄）・JPX（67銘柄）の計278銘柄。

**15の取引戦略**、それぞれ独立してテスト：
""")
        else:
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

**Universe:** 278 stocks across US (155), ASX (56), and JPX (67) markets.

**15 trading strategies**, each tested independently:
""")
        _ab_plain = STRATEGY_PLAIN_JA if lang == "ja" else STRATEGY_PLAIN
        for name, desc in _ab_plain.items():
            st.markdown(f"- **{name}** — {desc}")

        if lang == "ja":
            st.markdown("""
---

**ウォークフォワードバックテスト:** 戦略は2016〜2021年データで学習し、学習に使用していない2021年〜現在のデータでテストしています。これにより最も一般的なバックテストの落とし穴（過学習）を回避します。

**戦略選択:** 各銘柄には最も優れた戦略を割り当て。評価基準はサンプル外シャープレシオ（40%）・一貫性（20%）・勝率（10%）・ドローダウン管理（10%）・直近6ヶ月のパフォーマンス（20%）。

**レジームフィルター:** 現在の市場レジーム（強気/弱気/横ばい）が各戦略の過去のパフォーマンスが良かったレジームと一致する場合のみシグナルを発行。

**MLゲート:** モメンタム・ボラティリティ・レジーム・戦略スコアなどの特徴量で学習したXGBoostモデルが信頼度スコアを付与。閾値未満のシグナルは保留。

**ポジションサイジング:** ボラティリティと他の保有ポジションとの相関を調整したケリー基準を採用。

**保有期間:** 20営業日（約4週間）。ストップロス・利確・シグナル反転で早期決済。
""")
        else:
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
        st.markdown(T("ab_limits", lang))
        if lang == "ja":
            st.markdown("""
このスクリーナーはプロの取引システムではなく、**学習プロジェクト**です。
以下の制限事項をご理解ください：

- **サンプル蓄積中** — 全ランで600件超のクローズドトレード。古いランほどデータが多く、新しいランは初期指標として扱ってください。
- **無料データ** — yfinance（Yahoo Finance）を使用。データにはギャップ、誤り、遅延が生じる場合があります。
- **ファンダメンタルデータなし** — 純粋にテクニカル/統計シグナル。業績・バリュエーション・ニュースは考慮しない。
- **ペーパートレード** — 結果はシミュレーション。スリッページ・手数料・税金は含まない。
- **個別株フォーカス** — ETFやポートフォリオレベルのリスク管理なし。
- **過学習リスク** — ウォークフォワードテストを実施しているが、戦略が将来市場に適用できない可能性がある。
- **ASX/JPXデータ品質** — yfinanceの国際データは米国データより信頼性が低い場合がある。
""")
        else:
            st.markdown("""
This screener is a **learning project**, not a professional trading system.
Be aware of its limitations:

- **Growing sample size** — 600+ closed trades across all runs, with newer runs still accumulating data. Early runs have the most data; treat newer runs as indicators only.
- **Free data** — uses yfinance (Yahoo Finance). Data can have gaps, errors, and delays.
- **No fundamental data** — purely technical/statistical signals. Ignores earnings, valuation, news.
- **Paper trading** — results are simulated, not real. Slippage, brokerage, and taxes are not modelled.
- **Single-stock focus** — no ETF or portfolio-level risk management.
- **Overfitting risk** — despite walk-forward testing, strategies may not generalise to future markets.
- **ASX/JPX data quality** — international data from yfinance is less reliable than US data.
""")

        st.markdown("---")
        st.markdown(T("ab_legal", lang))
        if lang == "ja":
            legal_title = "重要：このウェブサイトは教育・情報提供のみを目的としています。"
            legal_body = (
                "このウェブサイトのいかなる内容も、財務・投資・取引その他のアドバイスを構成するものではありません。"
                "本サイトのコンテンツをそのように解釈しないでください。<br><br>"
                "本サイトに表示されるスクリーナーシグナル・ランキング・トラックレコードはアルゴリズムモデルによって生成されており、"
                "情報提供のみを目的としています。いかなる有価証券の売買または保有を推奨するものではありません。<br><br>"
                "いかなる戦略またはシグナルの過去の実績も、将来の結果を保証するものではありません。"
                "投資にはリスクが伴い、元本を失う可能性があります。<br><br>"
                "投資判断を行う前に、必ずご自身で独立したリサーチを実施し、"
                "ライセンスを持つ金融専門家への相談を検討してください。<br><br>"
                "本ウェブサイトの著者はライセンスを持つ財務アドバイザーではなく、"
                "本サイトに含まれる情報の正確性・完全性・適合性について一切の表明を行いません。"
            )
        else:
            legal_title = "IMPORTANT: This website is for educational and informational purposes only."
            legal_body = (
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
            )
        st.markdown(
            "<div class='disclaimer-box' style='color:#d29922;font-size:13px;line-height:1.6'>"
            f"<b>{legal_title}</b><br><br>"
            f"{legal_body}"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        if lang == "ja":
            st.markdown("### お問い合わせ")
            st.markdown("""
クオンツトレーディングを学ぶUQの学生が制作しました。

- GitHub: [GoofyisDAWG](https://github.com/GoofyisDAWG)
- ご質問・フィードバック: GitHubでIssueを開いてください
""")
        else:
            st.markdown("### Contact")
            st.markdown("""
Built by a UQ student learning quant trading.

- GitHub: [GoofyisDAWG](https://github.com/GoofyisDAWG)
- Questions or feedback: open an issue on GitHub
""")


