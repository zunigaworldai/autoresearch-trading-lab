from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",

    # Opening range
    "or_start": "09:30",
    "or_end": "09:45",

    # Entry window after opening range is complete
    "entry_start": "09:45",
    "entry_end": "15:30",

    # Direction: "long", "short", "both"
    "direction": "both",

    # Failed breakout logic
    "breakout_buffer_atr": 0.05,
    "reentry_buffer_atr": 0.02,
    "max_bars_after_breakout": 6,
    "require_close_back_inside_or": True,

    # OR quality filters
    "min_or_range_atr": 0.40,
    "max_or_range_atr": 3.50,

    # Regime filters
    "atr_len": 14,
    "ema_fast_len": 9,
    "ema_slow_len": 21,
    "require_ema_reversal_alignment": False,
    "require_vwap_reclaim": False,

    # Gap / volatility filters
    "use_gap_filter": True,
    "max_gap_pct": 0.0300,
    "max_prev_day_abs_return": 0.0700,
    "require_prev_day_filter": True,

    # Volume filter
    "require_volume_confirm": True,
    "vol_len": 20,
    "vol_mult": 1.05,
    "min_atr_pct": 0.0010,
    "max_atr_pct": 0.0600,

    # Risk / target
    "stop_mode": "breakout_extreme",  # "breakout_extreme", "atr", "opening_range"
    "atr_stop_mult": 1.0,
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


def _in_window_inclusive(out: pd.DataFrame, start: str, end: str) -> pd.Series:
    t = pd.to_datetime(out["datetime"]).dt.time
    return (t >= _time_value(start)) & (t <= _time_value(end))


def _in_window_left_closed(out: pd.DataFrame, start: str, end: str) -> pd.Series:
    t = pd.to_datetime(out["datetime"]).dt.time
    return (t >= _time_value(start)) & (t < _time_value(end))


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


def _map_daily_features(out: pd.DataFrame) -> pd.DataFrame:
    grouped = out.groupby("date", sort=False)
    daily = pd.DataFrame(
        {
            "day_open": grouped["open"].first(),
            "day_close": grouped["close"].last(),
        }
    )

    daily["prev_day_close"] = daily["day_close"].shift(1)
    daily["prev_prev_day_close"] = daily["day_close"].shift(2)
    daily["prev_day_return"] = daily["prev_day_close"] / daily["prev_prev_day_close"] - 1.0
    daily["gap_pct"] = daily["day_open"] / daily["prev_day_close"] - 1.0

    for col in daily.columns:
        out[col] = out["date"].map(daily[col])

    return out


def _map_opening_range(out: pd.DataFrame, or_mask: pd.Series) -> pd.DataFrame:
    or_df = out.loc[or_mask].copy()

    if or_df.empty:
        for col in ["or_open", "or_close", "or_high", "or_low", "or_range", "or_body", "or_direction"]:
            out[col] = np.nan
        return out

    grouped = or_df.groupby("date", sort=False)
    features = pd.DataFrame(
        {
            "or_open": grouped["open"].first(),
            "or_close": grouped["close"].last(),
            "or_high": grouped["high"].max(),
            "or_low": grouped["low"].min(),
        }
    )
    features["or_range"] = features["or_high"] - features["or_low"]
    features["or_body"] = (features["or_close"] - features["or_open"]).abs()
    features["or_direction"] = np.where(
        features["or_close"] > features["or_open"],
        1,
        np.where(features["or_close"] < features["or_open"], -1, 0),
    )

    for col in features.columns:
        out[col] = out["date"].map(features[col])

    return out


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Opening Failed Breakout / ORB Trap Reversal.

    Hypothesis:
    If price breaks the opening range high/low but fails and closes back inside
    the range within a limited number of bars, trapped breakout traders may fuel
    a short-term reversal.

    Compatible with engine.trade_simulator.run_trade_simulation:
    - entry_signal: 1 long, -1 short, 0 no entry
    - exit_stop
    - exit_tp
    - setup
    """

    params = {**DEFAULT_PARAMS, **(params or {})}
    out = _ensure_ohlcv(df)

    out["date"] = pd.to_datetime(out["datetime"]).dt.date
    out["in_session"] = _in_window_inclusive(out, params["session_start"], params["session_end"])
    out["in_entry_window"] = _in_window_inclusive(out, params["entry_start"], params["entry_end"])

    or_mask = out["in_session"] & _in_window_left_closed(out, params["or_start"], params["or_end"])
    out = _map_daily_features(out)
    out = _map_opening_range(out, or_mask)

    out["atr"] = _atr(out, int(params["atr_len"]))
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["vwap"] = _session_vwap(out)

    out["ema_fast"] = out["close"].ewm(span=int(params["ema_fast_len"]), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=int(params["ema_slow_len"]), adjust=False).mean()
    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()

    out["session_running_high"] = out.groupby("date")["high"].cummax()
    out["session_running_low"] = out.groupby("date")["low"].cummin()

    breakout_buffer = out["atr"] * float(params["breakout_buffer_atr"])
    reentry_buffer = out["atr"] * float(params["reentry_buffer_atr"])

    broke_or_high_now = out["high"] > out["or_high"] + breakout_buffer
    broke_or_low_now = out["low"] < out["or_low"] - breakout_buffer

    lookback = int(params["max_bars_after_breakout"])
    broke_or_high_recent = (
        broke_or_high_now.groupby(out["date"])
        .rolling(lookback, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(bool)
    )
    broke_or_low_recent = (
        broke_or_low_now.groupby(out["date"])
        .rolling(lookback, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(bool)
    )

    if bool(params.get("require_close_back_inside_or", True)):
        short_reentry = out["close"] < out["or_high"] - reentry_buffer
        long_reentry = out["close"] > out["or_low"] + reentry_buffer
    else:
        short_reentry = out["low"] < out["or_high"] - reentry_buffer
        long_reentry = out["high"] > out["or_low"] + reentry_buffer

    short_trigger = broke_or_high_recent & short_reentry
    long_trigger = broke_or_low_recent & long_reentry

    or_range_atr = out["or_range"] / out["atr"].replace(0, np.nan)
    or_quality_ok = (
        out["or_high"].notna()
        & out["or_low"].notna()
        & (or_range_atr >= float(params["min_or_range_atr"]))
        & (or_range_atr <= float(params["max_or_range_atr"]))
    )

    prev_day_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_prev_day_filter", True)):
        prev_day_ok = out["prev_day_return"].abs() <= float(params["max_prev_day_abs_return"])

    gap_ok = pd.Series(True, index=out.index)
    if bool(params.get("use_gap_filter", True)):
        gap_ok = out["gap_pct"].abs() <= float(params["max_gap_pct"])

    volume_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_volume_confirm", True)):
        volume_ok = out["volume"] >= out["vol_sma"] * float(params["vol_mult"])

    volatility_ok = (
        (out["atr_pct"] >= float(params["min_atr_pct"]))
        & (out["atr_pct"] <= float(params["max_atr_pct"]))
    )

    long_alignment = pd.Series(True, index=out.index)
    short_alignment = pd.Series(True, index=out.index)

    if bool(params.get("require_ema_reversal_alignment", False)):
        long_alignment &= out["close"] > out["ema_fast"]
        short_alignment &= out["close"] < out["ema_fast"]

    if bool(params.get("require_vwap_reclaim", False)):
        long_alignment &= out["close"] > out["vwap"]
        short_alignment &= out["close"] < out["vwap"]

    after_or = pd.to_datetime(out["datetime"]).dt.time >= _time_value(params["or_end"])

    base = (
        out["in_session"]
        & out["in_entry_window"]
        & after_or
        & out["atr"].notna()
        & out["prev_day_close"].notna()
        & or_quality_ok
        & prev_day_ok
        & gap_ok
        & volume_ok
        & volatility_ok
    )

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    out["long_signal_raw"] = allow_long & base & long_trigger & long_alignment
    out["short_signal_raw"] = allow_short & base & short_trigger & short_alignment

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

    if stop_mode == "breakout_extreme":
        long_stop = out["session_running_low"] - stop_buffer
        short_stop = out["session_running_high"] + stop_buffer
    elif stop_mode == "atr":
        long_stop = out["close"] - out["atr"] * float(params["atr_stop_mult"])
        short_stop = out["close"] + out["atr"] * float(params["atr_stop_mult"])
    elif stop_mode == "opening_range":
        long_stop = out["or_low"] - stop_buffer
        short_stop = out["or_high"] + stop_buffer
    else:
        raise ValueError("stop_mode must be 'breakout_extreme', 'atr', or 'opening_range'")

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
    out.loc[long_ok, "setup"] = "OPENING_FAILED_BREAKOUT_REVERSAL_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = "OPENING_FAILED_BREAKOUT_REVERSAL_SHORT"

    return out
