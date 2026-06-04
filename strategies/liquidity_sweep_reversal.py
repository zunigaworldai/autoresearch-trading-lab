from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",

    # Entry window
    "entry_start": "09:45",
    "entry_end": "15:30",

    # Direction: "long", "short", "both"
    "direction": "both",

    # Sweep source
    "sweep_level": "prev_day",  # "prev_day" only for v1

    # Sweep logic
    "sweep_buffer_atr": 0.05,
    "reclaim_buffer_atr": 0.02,
    "max_bars_after_sweep": 6,
    "require_close_reclaim": True,

    # Previous day / regime filters
    "require_prev_day_filter": True,
    "max_prev_day_abs_return": 0.0600,
    "min_prev_range_atr": 0.50,
    "max_prev_range_atr": 5.00,

    # Gap filters
    "use_gap_filter": True,
    "max_gap_pct": 0.0250,

    # Confirmation filters
    "atr_len": 14,
    "ema_fast_len": 9,
    "ema_slow_len": 21,
    "require_ema_reversal_alignment": False,
    "require_vwap_reclaim": False,

    # Volume / volatility filters
    "require_volume_confirm": True,
    "vol_len": 20,
    "vol_mult": 1.05,
    "min_atr_pct": 0.0010,
    "max_atr_pct": 0.0600,

    # Risk / target
    "stop_mode": "sweep_extreme",  # "sweep_extreme", "atr", "session_extreme"
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
            "day_high": grouped["high"].max(),
            "day_low": grouped["low"].min(),
            "day_close": grouped["close"].last(),
        }
    )

    daily["prev_day_high"] = daily["day_high"].shift(1)
    daily["prev_day_low"] = daily["day_low"].shift(1)
    daily["prev_day_close"] = daily["day_close"].shift(1)
    daily["prev_prev_day_close"] = daily["day_close"].shift(2)
    daily["prev_day_return"] = daily["prev_day_close"] / daily["prev_prev_day_close"] - 1.0
    daily["prev_day_range"] = daily["prev_day_high"] - daily["prev_day_low"]
    daily["gap_pct"] = daily["day_open"] / daily["prev_day_close"] - 1.0

    for col in daily.columns:
        out[col] = out["date"].map(daily[col])

    return out


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Previous-Day Range / Liquidity Sweep Reversal.

    Hypothesis:
    Price runs prior-day high/low liquidity, fails to hold outside the prior-day range,
    and reclaims the level. That failed auction can produce a short-term reversal.

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
    out = _map_daily_features(out)

    out["atr"] = _atr(out, int(params["atr_len"]))
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["vwap"] = _session_vwap(out)

    out["ema_fast"] = out["close"].ewm(span=int(params["ema_fast_len"]), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=int(params["ema_slow_len"]), adjust=False).mean()
    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()

    # Session running extremes up to current bar.
    out["session_running_high"] = out.groupby("date")["high"].cummax()
    out["session_running_low"] = out.groupby("date")["low"].cummin()

    sweep_buffer = out["atr"] * float(params["sweep_buffer_atr"])
    reclaim_buffer = out["atr"] * float(params["reclaim_buffer_atr"])

    swept_prev_high_now = out["high"] > out["prev_day_high"] + sweep_buffer
    swept_prev_low_now = out["low"] < out["prev_day_low"] - sweep_buffer

    swept_prev_high_recent = (
        swept_prev_high_now.groupby(out["date"])
        .rolling(int(params["max_bars_after_sweep"]), min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(bool)
    )
    swept_prev_low_recent = (
        swept_prev_low_now.groupby(out["date"])
        .rolling(int(params["max_bars_after_sweep"]), min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(bool)
    )

    # Reversal entries:
    # - Short after prior-day high sweep and close back below prior high.
    # - Long after prior-day low sweep and close back above prior low.
    if bool(params.get("require_close_reclaim", True)):
        short_reclaim = out["close"] < out["prev_day_high"] - reclaim_buffer
        long_reclaim = out["close"] > out["prev_day_low"] + reclaim_buffer
    else:
        short_reclaim = out["low"] < out["prev_day_high"] - reclaim_buffer
        long_reclaim = out["high"] > out["prev_day_low"] + reclaim_buffer

    short_trigger = swept_prev_high_recent & short_reclaim
    long_trigger = swept_prev_low_recent & long_reclaim

    # Regime filters
    prev_range_atr = out["prev_day_range"] / out["atr"].replace(0, np.nan)
    prev_range_ok = (
        (prev_range_atr >= float(params["min_prev_range_atr"]))
        & (prev_range_atr <= float(params["max_prev_range_atr"]))
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
        # For reversal, require the current bar to be moving back toward EMA structure.
        long_alignment &= out["close"] > out["ema_fast"]
        short_alignment &= out["close"] < out["ema_fast"]

    if bool(params.get("require_vwap_reclaim", False)):
        long_alignment &= out["close"] > out["vwap"]
        short_alignment &= out["close"] < out["vwap"]

    base = (
        out["in_session"]
        & out["in_entry_window"]
        & out["atr"].notna()
        & out["prev_day_high"].notna()
        & out["prev_day_low"].notna()
        & out["prev_day_close"].notna()
        & prev_range_ok
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

    if stop_mode == "sweep_extreme":
        long_stop = out["session_running_low"] - stop_buffer
        short_stop = out["session_running_high"] + stop_buffer
    elif stop_mode == "atr":
        long_stop = out["close"] - out["atr"] * float(params["atr_stop_mult"])
        short_stop = out["close"] + out["atr"] * float(params["atr_stop_mult"])
    elif stop_mode == "session_extreme":
        long_stop = out["day_low"] - stop_buffer
        short_stop = out["day_high"] + stop_buffer
    else:
        raise ValueError("stop_mode must be 'sweep_extreme', 'atr', or 'session_extreme'")

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
    out.loc[long_ok, "setup"] = "LIQUIDITY_SWEEP_REVERSAL_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = "LIQUIDITY_SWEEP_REVERSAL_SHORT"

    return out
