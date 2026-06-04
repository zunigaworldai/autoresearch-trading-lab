from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",
    "entry_start": "09:45",
    "entry_end": "15:30",

    # Direction: "long", "short", "both"
    "direction": "both",

    # EMA trend structure
    "ema_fast_len": 9,
    "ema_mid_len": 21,
    "ema_slow_len": 50,
    "require_fast_over_mid": True,
    "require_mid_over_slow": True,

    # Pullback definition
    "pullback_ema": "mid",          # "fast" or "mid"
    "pullback_lookback": 5,
    "pullback_tolerance_atr": 0.15,

    # Confirmation mode: "close_reclaim", "break_prev_bar", "bull_bear_candle"
    "confirm_mode": "close_reclaim",

    # Trend quality filters
    "atr_len": 14,
    "slope_lookback": 5,
    "min_slope_atr": 0.02,
    "require_trend_slope": True,
    "require_vwap_alignment": True,

    # Volume / volatility filters
    "require_volume_confirm": True,
    "vol_len": 20,
    "vol_mult": 1.05,
    "min_atr_pct": 0.0010,
    "max_atr_pct": 0.0500,

    # Risk / target
    "stop_mode": "atr",       # "atr", "swing", "ema"
    "atr_stop_mult": 1.0,
    "swing_lookback": 8,
    "stop_buffer_atr": 0.10,
    "min_rr": 1.5,

    "one_trade_per_day": True,
}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    out = df.copy()

    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif "timestamp" in out.columns:
        out["datetime"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out["datetime"] = pd.to_datetime(out.index, errors="coerce")
    else:
        raise ValueError("DataFrame must contain datetime, timestamp, or DatetimeIndex")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    return out.sort_values("datetime").reset_index(drop=True)


def _time_value(value: str):
    return pd.to_datetime(value).time()


def _in_time_window(out: pd.DataFrame, start: str, end: str) -> pd.Series:
    t = pd.to_datetime(out["datetime"]).dt.time
    return (t >= _time_value(start)) & (t <= _time_value(end))


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return _rma(tr, length)


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = hlc3 * df["volume"]
    session_key = pd.to_datetime(df["datetime"]).dt.date
    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = df["volume"].groupby(session_key).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    EMA Pullback Trend Scalping.

    Hypothesis:
    In a confirmed intraday trend, a controlled pullback into a rising/falling EMA
    can offer continuation entries if price reclaims the pullback area with volatility,
    volume, and VWAP alignment.

    Compatible with engine.trade_simulator.run_trade_simulation:
    - entry_signal: 1 long, -1 short, 0 no entry
    - exit_stop
    - exit_tp
    - setup
    """

    params = {**DEFAULT_PARAMS, **(params or {})}
    out = _ensure_ohlcv(df)

    out["date"] = pd.to_datetime(out["datetime"]).dt.date
    out["in_session"] = _in_time_window(out, params["session_start"], params["session_end"])
    out["in_entry_window"] = _in_time_window(out, params["entry_start"], params["entry_end"])

    out["atr"] = _atr(out, int(params["atr_len"]))
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["vwap"] = _session_vwap(out)

    out["ema_fast"] = out["close"].ewm(span=int(params["ema_fast_len"]), adjust=False).mean()
    out["ema_mid"] = out["close"].ewm(span=int(params["ema_mid_len"]), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=int(params["ema_slow_len"]), adjust=False).mean()

    slope_lookback = int(params["slope_lookback"])
    out["ema_mid_slope"] = out["ema_mid"] - out["ema_mid"].shift(slope_lookback)

    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()

    pullback_ema_name = str(params["pullback_ema"]).lower()
    if pullback_ema_name == "fast":
        pullback_ema = out["ema_fast"]
    elif pullback_ema_name == "mid":
        pullback_ema = out["ema_mid"]
    else:
        raise ValueError("pullback_ema must be 'fast' or 'mid'")

    tol = out["atr"] * float(params["pullback_tolerance_atr"])
    lookback = int(params["pullback_lookback"])

    touched_long = (out["low"] <= pullback_ema + tol).rolling(lookback).max().fillna(0).astype(bool)
    touched_short = (out["high"] >= pullback_ema - tol).rolling(lookback).max().fillna(0).astype(bool)

    trend_long_ok = pd.Series(True, index=out.index)
    trend_short_ok = pd.Series(True, index=out.index)

    if bool(params.get("require_fast_over_mid", True)):
        trend_long_ok &= out["ema_fast"] > out["ema_mid"]
        trend_short_ok &= out["ema_fast"] < out["ema_mid"]

    if bool(params.get("require_mid_over_slow", True)):
        trend_long_ok &= out["ema_mid"] > out["ema_slow"]
        trend_short_ok &= out["ema_mid"] < out["ema_slow"]

    if bool(params.get("require_trend_slope", True)):
        min_slope = out["atr"] * float(params["min_slope_atr"])
        trend_long_ok &= out["ema_mid_slope"] > min_slope
        trend_short_ok &= out["ema_mid_slope"] < -min_slope

    if bool(params.get("require_vwap_alignment", True)):
        trend_long_ok &= out["close"] > out["vwap"]
        trend_short_ok &= out["close"] < out["vwap"]

    volume_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_volume_confirm", True)):
        volume_ok = out["volume"] >= out["vol_sma"] * float(params["vol_mult"])

    volatility_ok = (
        (out["atr_pct"] >= float(params["min_atr_pct"]))
        & (out["atr_pct"] <= float(params["max_atr_pct"]))
    )

    confirm_mode = str(params["confirm_mode"]).lower()
    if confirm_mode == "close_reclaim":
        long_confirm = (out["close"].shift(1) <= pullback_ema.shift(1)) & (out["close"] > pullback_ema)
        short_confirm = (out["close"].shift(1) >= pullback_ema.shift(1)) & (out["close"] < pullback_ema)
    elif confirm_mode == "break_prev_bar":
        long_confirm = (out["low"] <= pullback_ema + tol) & (out["close"] > out["high"].shift(1))
        short_confirm = (out["high"] >= pullback_ema - tol) & (out["close"] < out["low"].shift(1))
    elif confirm_mode == "bull_bear_candle":
        long_confirm = (out["low"] <= pullback_ema + tol) & (out["close"] > out["open"])
        short_confirm = (out["high"] >= pullback_ema - tol) & (out["close"] < out["open"])
    else:
        raise ValueError("confirm_mode must be 'close_reclaim', 'break_prev_bar', or 'bull_bear_candle'")

    base = (
        out["in_session"]
        & out["in_entry_window"]
        & out["atr"].notna()
        & out["vwap"].notna()
        & out["ema_fast"].notna()
        & out["ema_mid"].notna()
        & out["ema_slow"].notna()
        & volume_ok
        & volatility_ok
    )

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    out["long_signal_raw"] = allow_long & base & trend_long_ok & touched_long & long_confirm
    out["short_signal_raw"] = allow_short & base & trend_short_ok & touched_short & short_confirm

    if bool(params.get("one_trade_per_day", True)):
        raw_any = out["long_signal_raw"] | out["short_signal_raw"]
        already_had_signal = raw_any.groupby(out["date"]).cumsum().shift(fill_value=0) > 0
        out["long_signal"] = out["long_signal_raw"] & (~already_had_signal)
        out["short_signal"] = out["short_signal_raw"] & (~already_had_signal)
    else:
        out["long_signal"] = out["long_signal_raw"]
        out["short_signal"] = out["short_signal_raw"]

    stop_mode = str(params["stop_mode"]).lower()
    stop_buffer = out["atr"] * float(params["stop_buffer_atr"])

    if stop_mode == "atr":
        long_stop = out["close"] - out["atr"] * float(params["atr_stop_mult"])
        short_stop = out["close"] + out["atr"] * float(params["atr_stop_mult"])
    elif stop_mode == "swing":
        long_stop = out["low"].rolling(int(params["swing_lookback"])).min() - stop_buffer
        short_stop = out["high"].rolling(int(params["swing_lookback"])).max() + stop_buffer
    elif stop_mode == "ema":
        long_stop = pullback_ema - stop_buffer
        short_stop = pullback_ema + stop_buffer
    else:
        raise ValueError("stop_mode must be 'atr', 'swing', or 'ema'")

    long_risk = out["close"] - long_stop
    short_risk = short_stop - out["close"]
    rr = float(params["min_rr"])

    long_tp = out["close"] + long_risk * rr
    short_tp = out["close"] - short_risk * rr

    long_ok = out["long_signal"] & (long_risk > 0)
    short_ok = out["short_signal"] & (short_risk > 0)

    out["entry_signal"] = 0
    out.loc[long_ok, "entry_signal"] = 1
    out.loc[short_ok, "entry_signal"] = -1

    out["exit_stop"] = np.nan
    out["exit_tp"] = np.nan
    out["setup"] = ""

    out.loc[long_ok, "exit_stop"] = long_stop[long_ok]
    out.loc[long_ok, "exit_tp"] = long_tp[long_ok]
    out.loc[long_ok, "setup"] = "EMA_PULLBACK_TREND_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = "EMA_PULLBACK_TREND_SHORT"

    return out
