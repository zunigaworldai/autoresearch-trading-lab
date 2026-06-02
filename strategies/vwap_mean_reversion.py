from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",
    "entry_start": "09:30",
    "entry_end": "16:00",
    "direction": "both",
    "dev_mode": "atr",
    "dev_len": 20,
    "dev_mult": 1.5,
    "confirm_mode": "reclaim",
    "reclaim_buffer": 0.0,
    "atr_len": 20,
    "stop_mode": "atr",
    "atr_stop_mult": 1.0,
    "swing_lookback": 8,
    "stop_buffer_atr": 0.10,
    "target_mode": "vwap",
    "min_rr": 1.0,
    "one_trade_per_day": True,
    "require_volume_filter": False,
    "vol_len": 20,
    "vol_max_mult": 1.5,
    "avoid_strong_trend": False,
    "trend_len": 50,
    "trend_slope_atr_max": 0.25,
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


def _in_time_window(out: pd.DataFrame, start: str, end: str) -> pd.Series:
    t = pd.to_datetime(out["datetime"]).dt.time
    return (t >= _time_value(start)) & (t <= _time_value(end))


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    VWAP Mean Reversion strategy.

    Compatible with engine.trade_simulator.run_trade_simulation:
    entry_signal, exit_stop, exit_tp, setup.
    """

    params = {**DEFAULT_PARAMS, **(params or {})}
    out = _ensure_ohlcv(df)

    out["date"] = pd.to_datetime(out["datetime"]).dt.date
    out["in_session"] = _in_time_window(out, params["session_start"], params["session_end"])
    out["in_entry_window"] = _in_time_window(out, params["entry_start"], params["entry_end"])

    out["vwap"] = _session_vwap(out)
    out["atr"] = _atr(out, int(params["atr_len"]))

    dev_mode = str(params["dev_mode"]).lower()
    if dev_mode == "atr":
        out["dev"] = out["atr"]
    elif dev_mode == "stdev":
        out["dev"] = out["close"].rolling(int(params["dev_len"])).std()
    else:
        raise ValueError("dev_mode must be 'atr' or 'stdev'")

    dev_mult = float(params["dev_mult"])
    out["upper_band"] = out["vwap"] + out["dev"] * dev_mult
    out["lower_band"] = out["vwap"] - out["dev"] * dev_mult

    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()
    volume_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_volume_filter", False)):
        volume_ok = out["volume"] <= out["vol_sma"] * float(params["vol_max_mult"])

    out["trend_ma"] = out["close"].rolling(int(params["trend_len"])).mean()
    out["trend_slope"] = (out["trend_ma"] - out["trend_ma"].shift(10)).abs()
    trend_ok = pd.Series(True, index=out.index)
    if bool(params.get("avoid_strong_trend", False)):
        trend_ok = out["trend_slope"] <= out["atr"] * float(params["trend_slope_atr_max"])

    confirm_mode = str(params["confirm_mode"]).lower()
    reclaim_buffer = float(params["reclaim_buffer"])

    was_below = out["close"].shift(1) < out["lower_band"].shift(1)
    was_above = out["close"].shift(1) > out["upper_band"].shift(1)

    if confirm_mode == "reclaim":
        long_confirm = was_below & (out["close"] > out["lower_band"] + out["atr"] * reclaim_buffer)
        short_confirm = was_above & (out["close"] < out["upper_band"] - out["atr"] * reclaim_buffer)
    elif confirm_mode == "candle":
        long_confirm = (out["close"] < out["lower_band"]) & (out["close"] > out["open"])
        short_confirm = (out["close"] > out["upper_band"]) & (out["close"] < out["open"])
    else:
        raise ValueError("confirm_mode must be 'reclaim' or 'candle'")

    base = (
        out["in_session"]
        & out["in_entry_window"]
        & out["vwap"].notna()
        & out["dev"].notna()
        & (out["dev"] > 0)
        & volume_ok
        & trend_ok
    )

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    out["long_signal_raw"] = allow_long & base & long_confirm
    out["short_signal_raw"] = allow_short & base & short_confirm

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
    else:
        raise ValueError("stop_mode must be 'atr' or 'swing'")

    long_risk = out["close"] - long_stop
    short_risk = short_stop - out["close"]

    target_mode = str(params["target_mode"]).lower()
    min_rr = float(params["min_rr"])

    if target_mode == "vwap":
        long_tp = out["vwap"]
        short_tp = out["vwap"]
    elif target_mode == "rr":
        long_tp = out["close"] + long_risk * min_rr
        short_tp = out["close"] - short_risk * min_rr
    else:
        raise ValueError("target_mode must be 'vwap' or 'rr'")

    long_reward = long_tp - out["close"]
    short_reward = out["close"] - short_tp

    long_ok = out["long_signal"] & (long_risk > 0) & (long_reward > 0) & (long_reward / long_risk >= min_rr)
    short_ok = out["short_signal"] & (short_risk > 0) & (short_reward > 0) & (short_reward / short_risk >= min_rr)

    out["entry_signal"] = 0
    out.loc[long_ok, "entry_signal"] = 1
    out.loc[short_ok, "entry_signal"] = -1

    out["exit_stop"] = np.nan
    out["exit_tp"] = np.nan
    out["setup"] = ""

    out.loc[long_ok, "exit_stop"] = long_stop[long_ok]
    out.loc[long_ok, "exit_tp"] = long_tp[long_ok]
    out.loc[long_ok, "setup"] = "VWAP_MR_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = "VWAP_MR_SHORT"

    return out
