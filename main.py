"""
main.py — GEX Levels UDF Server
────────────────────────────────────────────────────────────────────────────────
Serves live SPY GEX levels (Call Wall / Put Wall / HVL) computed directly
from CBOE's free delayed options chain. No API key. No subscription.

Data source: https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json
Cost: $0

Deployment: Render.com free tier (see render.yaml)

TradingView integration:
  - Speaks the TradingView UDF protocol
  - Pine Script reads via request.security("YOUR_RENDER_URL:GEX_LEVELS", ...)
  - GEX levels encoded in OHLC: open=call_wall, low=put_wall, close=hvl

Environment variables (set in Render dashboard):
  TICKER            Underlying to track (default: SPY)
  POLL_INTERVAL_MIN Poll interval in minutes (default: 30)
  PORT              Set automatically by Render
"""

import os
import time
import threading
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gex_engine import compute_gex

# ── Config ────────────────────────────────────────────────────────────────────
TICKER    = os.environ.get("TICKER", "SPY")
POLL_MIN  = int(os.environ.get("POLL_INTERVAL_MIN", "30"))
ET        = ZoneInfo("America/New_York")

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("server")

# ── Shared state ──────────────────────────────────────────────────────────────
_state: dict = {
    "call_wall":     None,
    "put_wall":      None,
    "hvl":           None,
    "spot":          None,
    "regime":        None,
    "net_gex_total": None,
    "as_of":         None,
    "last_poll":     None,
    "contracts_used":0,
    "error":         None,
    "top_call_strikes": [],
    "top_put_strikes":  [],
}
_lock = threading.Lock()


# ── Market hours check ────────────────────────────────────────────────────────
def _is_polling_window() -> bool:
    """Poll from 8:00 AM to 4:30 PM ET on weekdays."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (8, 0) <= (t.hour, t.minute) <= (16, 30)


# ── Background poller ─────────────────────────────────────────────────────────
def _poll_loop():
    """
    Background thread: compute GEX from CBOE chain on a schedule.
    Runs one immediate computation at startup, then every POLL_MIN minutes.
    """
    # First poll immediately so data is available right away
    _do_poll()

    while True:
        time.sleep(POLL_MIN * 60)
        if _is_polling_window():
            _do_poll()
        else:
            log.debug("Outside polling window — skipping")


def _do_poll():
    log.info(f"Starting GEX computation for {TICKER}...")
    result = compute_gex(TICKER)

    with _lock:
        _state.update(result)
        _state["last_poll"] = datetime.now(timezone.utc).isoformat()

    if result["error"]:
        log.error(f"GEX computation error: {result['error']}")
    else:
        log.info(
            f"Levels updated → "
            f"CW={result['call_wall']} "
            f"PW={result['put_wall']} "
            f"HVL={result['hvl']} "
            f"Regime={result['regime']} "
            f"NetGEX=${result['net_gex_total']}B "
            f"Contracts={result['contracts_used']}"
        )


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "GEX Levels — Free CBOE-Computed UDF Server",
    description = "SPY Call Wall / Put Wall / HVL from CBOE options chain. No API key.",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # Required for TradingView UDF
    allow_methods  = ["GET"],
    allow_headers  = ["*"],
)


@app.on_event("startup")
def _startup():
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    log.info(f"GEX server started | ticker={TICKER} | poll={POLL_MIN}min")


# ── UDF Protocol endpoints ────────────────────────────────────────────────────
@app.get("/config")
def udf_config():
    return {
        "supported_resolutions": ["1", "5", "15", "30", "60", "D"],
        "supports_group_request": False,
        "supports_marks":         False,
        "supports_search":        True,
        "supports_timescale_marks": False,
        "exchanges": [
            {"value": "GEX", "name": "GEX Levels (CBOE)", "desc": "Self-computed GEX"}
        ],
        "symbols_types": [{"name": "GEX", "value": "gex"}],
    }


@app.get("/symbols")
def udf_symbols(symbol: str = Query(...)):
    return {
        "name":         symbol,
        "description":  f"{TICKER} GEX Levels — Call Wall / Put Wall / HVL",
        "type":         "gex",
        "exchange":     "GEX",
        "timezone":     "America/New_York",
        "pricescale":   100,
        "minmov":       1,
        "session":      "0930-1600",
        "has_intraday": True,
        "supported_resolutions": ["1", "5", "15", "30", "60", "D"],
    }


@app.get("/search")
def udf_search(
    query: str = Query(""),
    type:  str = Query(""),
    limit: int = Query(10),
):
    return [{
        "symbol":      "GEX_LEVELS",
        "full_name":   "GEX:GEX_LEVELS",
        "description": f"{TICKER} GEX Levels — Call Wall / Put Wall / HVL",
        "exchange":    "GEX",
        "type":        "gex",
    }]


@app.get("/history")
def udf_history(
    symbol:     str = Query(...),
    resolution: str = Query(...),
    from_ts:    int = Query(..., alias="from"),
    to_ts:      int = Query(..., alias="to"),
    countback:  int = Query(None),
):
    """
    UDF OHLCV history endpoint.

    GEX levels encoded into OHLC so Pine Script can extract each independently:
        open  = call_wall   → Pine: open  → red line
        high  = call_wall   → (same, keeps OHLC valid — high >= close)
        low   = put_wall    → Pine: low   → blue line
        close = hvl         → Pine: close → yellow line
        volume = 0
    """
    with _lock:
        cw  = _state["call_wall"]
        pw  = _state["put_wall"]
        hvl = _state["hvl"]

    if cw is None or pw is None or hvl is None:
        return {"s": "no_data"}

    # Ensure OHLC validity: high >= open, close; low <= open, close
    # hvl sits between put_wall and call_wall by definition, so:
    # open=cw, high=cw, low=pw, close=hvl is always valid
    res_map = {"1": 60, "5": 300, "15": 900, "30": 1800,
               "60": 3600, "D": 86400}
    bar_sec = res_map.get(resolution, 1800)

    ts_list, o, h, l, c, v = [], [], [], [], [], []
    t = from_ts
    while t <= to_ts:
        ts_list.append(t)
        o.append(cw)
        h.append(cw)
        l.append(pw)
        c.append(hvl)
        v.append(0)
        t += bar_sec

    if not ts_list:
        return {"s": "no_data"}

    return {"s": "ok", "t": ts_list, "o": o, "h": h, "l": l, "c": c, "v": v}


# ── Non-UDF endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Render health check."""
    with _lock:
        ok = _state["call_wall"] is not None
    return {"status": "ok" if ok else "no_data", "last_poll": _state["last_poll"]}


@app.get("/levels")
def levels():
    """
    Human-readable JSON of current GEX levels.
    Use this to verify the server is computing correct values.
    Cross-reference against InsiderFinance and Barchart manually.
    """
    with _lock:
        return JSONResponse({
            "ticker":           TICKER,
            "call_wall":        _state["call_wall"],
            "put_wall":         _state["put_wall"],
            "hvl":              _state["hvl"],
            "spot":             _state["spot"],
            "regime":           _state["regime"],
            "net_gex_billions": _state["net_gex_total"],
            "contracts_used":   _state["contracts_used"],
            "top_call_strikes": _state["top_call_strikes"],
            "top_put_strikes":  _state["top_put_strikes"],
            "as_of":            _state["as_of"],
            "last_poll":        _state["last_poll"],
            "error":            _state["error"],
            "data_source":      "CBOE delayed options chain (free, no API key)",
            "methodology":      "GEX = Gamma × OI × 100 × Spot² × 0.01. "
                                "Naive dealer convention (short calls, long puts). "
                                "Full-chain aggregation across all expirations. "
                                "OI is T-1 (OCC settlement). 15-min price delay.",
        })
