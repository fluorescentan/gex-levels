"""
gex_engine.py
─────────────────────────────────────────────────────────────────────────────
Self-contained GEX computation from CBOE's free delayed options chain.

Data source:
    https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json
    - No API key required
    - 15-minute delayed (OI is always T-1 per OCC — same as all providers)
    - Full options chain: all strikes, all expirations

Formula (standard, matches InsiderFinance / Barchart / FlashAlpha naive model):
    GEX per strike = Gamma × OI × 100 × Spot² × 0.01
    Call GEX is positive (dealers short calls → long delta → buy on rally)
    Put GEX is negative (dealers long puts → short delta → sell on drop)
    Net GEX per strike = Call GEX + Put GEX

Key levels:
    Call Wall = strike with highest raw call-side GEX
    Put Wall  = strike with most negative raw put-side GEX
    HVL       = strike where cumulative net GEX (sorted ascending) crosses zero
"""

import re
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np

log = logging.getLogger("gex_engine")

# ── Constants ─────────────────────────────────────────────────────────────────
CBOE_URL   = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; GEXBot/1.0)"}
SYMBOL_RE  = re.compile(r'^(.+?)(\d{6})([PC])(\d+)$')
ET         = ZoneInfo("America/New_York")

# ── CBOE data fetch ───────────────────────────────────────────────────────────
def fetch_chain(ticker: str = "SPY", timeout: int = 15) -> dict:
    """
    Fetch raw options chain from CBOE.
    Returns dict with keys: current_price, options (list of dicts), timestamp
    Raises on network error or unexpected response shape.
    """
    url = CBOE_URL.format(ticker=ticker.upper())
    log.info(f"Fetching CBOE chain for {ticker} from {url}")

    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    body = resp.json()

    # Validate expected shape
    if "data" not in body:
        raise ValueError(f"Unexpected CBOE response — missing 'data' key. "
                         f"Keys found: {list(body.keys())}")

    data = body["data"]
    if "options" not in data or "current_price" not in data:
        raise ValueError(f"Unexpected CBOE data shape. Keys: {list(data.keys())}")

    log.info(f"Fetched {len(data['options'])} option contracts. "
             f"Spot: {data['current_price']}")

    return {
        "current_price": float(data["current_price"]),
        "options":       data["options"],
        "timestamp":     data.get("timestamp", ""),
    }


# ── Option symbol parser ──────────────────────────────────────────────────────
def parse_option_symbol(symbol: str) -> dict | None:
    """
    Parse OCC option symbol into components.
    Format: {ticker}{YYMMDD}{C|P}{strike * 1000}
    Example: SPY260516C00580000 → strike=580.0, call=True, exp=2026-05-16
    Returns None if symbol doesn't match.
    """
    m = SYMBOL_RE.match(symbol)
    if not m:
        return None

    try:
        exp_date = datetime.strptime("20" + m.group(2), "%Y%m%d").date()
        strike   = int(m.group(4)) / 1000.0
        is_call  = m.group(3) == "C"
        return {
            "strike":   strike,
            "is_call":  is_call,
            "exp_date": exp_date,
            "dte_days": (exp_date - datetime.now(ET).date()).days,
        }
    except (ValueError, IndexError):
        return None


# ── GEX computation ───────────────────────────────────────────────────────────
def compute_gex(ticker: str = "SPY") -> dict:
    """
    Full GEX computation pipeline.

    Returns:
        {
            "call_wall":     float,   # strike with highest call-side GEX
            "put_wall":      float,   # strike with most negative put-side GEX
            "hvl":           float,   # gamma flip — net GEX crosses zero
            "spot":          float,   # current underlying price
            "net_gex_total": float,   # total net GEX in $billions
            "regime":        str,     # "positive" or "negative"
            "as_of":         str,     # ISO timestamp
            "top_call_strikes": list, # top 5 call walls by GEX
            "top_put_strikes":  list, # top 5 put walls by GEX
            "error":         None | str,
        }
    """
    try:
        raw = fetch_chain(ticker)
    except Exception as e:
        log.error(f"Chain fetch failed: {e}")
        return _error_result(str(e))

    spot    = raw["current_price"]
    options = raw["options"]

    # ── Build DataFrame ───────────────────────────────────────────────────────
    rows = []
    skipped = 0

    for opt in options:
        symbol = opt.get("option", "")
        parsed = parse_option_symbol(symbol)
        if not parsed:
            skipped += 1
            continue

        # Skip expired contracts
        if parsed["dte_days"] < 0:
            skipped += 1
            continue

        gamma = opt.get("gamma", 0) or 0
        oi    = opt.get("open_interest", 0) or 0
        iv    = opt.get("iv", 0) or 0

        # Skip zero-gamma or zero-OI (no hedging obligation)
        if gamma <= 0 or oi <= 0:
            skipped += 1
            continue

        # GEX formula: Gamma × OI × 100 × Spot² × 0.01
        # = Gamma × OI × Spot² (the 100 contracts × 0.01 cancel to ×1)
        gex_raw = gamma * oi * 100 * (spot ** 2) * 0.01

        rows.append({
            "strike":   parsed["strike"],
            "is_call":  parsed["is_call"],
            "exp_date": parsed["exp_date"],
            "dte_days": parsed["dte_days"],
            "gamma":    gamma,
            "oi":       oi,
            "iv":       iv,
            # Sign convention: calls positive, puts negative
            "gex":      gex_raw if parsed["is_call"] else -gex_raw,
            "call_gex": gex_raw if parsed["is_call"] else 0.0,
            "put_gex":  -gex_raw if not parsed["is_call"] else 0.0,
        })

    log.info(f"Parsed {len(rows)} valid contracts, skipped {skipped}")

    if not rows:
        return _error_result("No valid option contracts parsed from chain")

    df = pd.DataFrame(rows)

    # ── Aggregate by strike ───────────────────────────────────────────────────
    by_strike = df.groupby("strike").agg(
        call_gex  = ("call_gex", "sum"),
        put_gex   = ("put_gex",  "sum"),
        net_gex   = ("gex",      "sum"),
        total_oi  = ("oi",       "sum"),
    ).reset_index().sort_values("strike")

    # ── Key levels ────────────────────────────────────────────────────────────
    # Call Wall: strike with largest call-side GEX
    call_wall_idx = by_strike["call_gex"].idxmax()
    call_wall     = float(by_strike.loc[call_wall_idx, "strike"])

    # Put Wall: strike with most negative put-side GEX
    put_wall_idx  = by_strike["put_gex"].idxmin()
    put_wall      = float(by_strike.loc[put_wall_idx, "strike"])

    # HVL (Gamma Flip): cumulative net GEX crosses zero
    # Sort by strike ascending, compute cumulative sum, find zero crossing
    by_strike_sorted = by_strike.sort_values("strike").reset_index(drop=True)
    cum_gex = by_strike_sorted["net_gex"].cumsum()

    # Find where cumulative GEX crosses zero — interpolate between brackets
    hvl = _find_zero_crossing(by_strike_sorted["strike"].values, cum_gex.values)

    # Regime
    net_gex_total = float(by_strike["net_gex"].sum())
    regime = "positive" if net_gex_total > 0 else "negative"

    # Top 5 call and put walls for context
    top_calls = (by_strike.nlargest(5, "call_gex")[["strike", "call_gex"]]
                 .assign(call_gex=lambda x: (x["call_gex"] / 1e9).round(3))
                 .to_dict("records"))
    top_puts  = (by_strike.nsmallest(5, "put_gex")[["strike", "put_gex"]]
                 .assign(put_gex=lambda x: (x["put_gex"] / 1e9).round(3))
                 .to_dict("records"))

    log.info(
        f"GEX computed → CW={call_wall} PW={put_wall} HVL={hvl:.2f} "
        f"NetGEX=${net_gex_total/1e9:.1f}B Regime={regime}"
    )

    return {
        "call_wall":        call_wall,
        "put_wall":         put_wall,
        "hvl":              round(hvl, 2),
        "spot":             spot,
        "net_gex_total":    round(net_gex_total / 1e9, 3),   # in $billions
        "regime":           regime,
        "as_of":            datetime.now(timezone.utc).isoformat(),
        "top_call_strikes": top_calls,
        "top_put_strikes":  top_puts,
        "contracts_used":   len(rows),
        "error":            None,
    }


# ── HVL zero-crossing interpolation ──────────────────────────────────────────
def _find_zero_crossing(strikes: np.ndarray, cum_gex: np.ndarray) -> float:
    """
    Find the strike where cumulative net GEX crosses zero.
    Uses linear interpolation between the bracketing strikes.
    Falls back to the strike closest to zero if no crossing exists.
    """
    for i in range(len(cum_gex) - 1):
        # Detect sign change
        if cum_gex[i] <= 0 <= cum_gex[i + 1] or cum_gex[i] >= 0 >= cum_gex[i + 1]:
            # Linear interpolation between strikes[i] and strikes[i+1]
            s0, s1 = float(strikes[i]), float(strikes[i + 1])
            g0, g1 = float(cum_gex[i]), float(cum_gex[i + 1])
            if g1 == g0:
                return (s0 + s1) / 2.0
            # Zero crossing: s0 + (s1-s0) * (-g0) / (g1-g0)
            return s0 + (s1 - s0) * (-g0) / (g1 - g0)

    # No crossing — return strike closest to zero cumulative GEX
    idx = np.argmin(np.abs(cum_gex))
    return float(strikes[idx])


# ── Error result template ─────────────────────────────────────────────────────
def _error_result(msg: str) -> dict:
    return {
        "call_wall":        None,
        "put_wall":         None,
        "hvl":              None,
        "spot":             None,
        "net_gex_total":    None,
        "regime":           None,
        "as_of":            datetime.now(timezone.utc).isoformat(),
        "top_call_strikes": [],
        "top_put_strikes":  [],
        "contracts_used":   0,
        "error":            msg,
    }
