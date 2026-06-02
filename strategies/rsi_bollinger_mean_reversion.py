from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",
    "entry_start": "09:30",
    "entry_end": "16:00",

    # Direction: "long", "short", "both"
    "direction": "both",

    # Bollinger / RSI core
    "bb_len": 20,
    "bb_mult": 2.0,
    "rsi_len": 14,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,

    # Confirmation: "reclaim_band" or "reversal_candle"
    "confirm_mode": "reclaim_band",

    # Regime filters
    "avoid_strong_trend": True,
    "trend_len": 50,
    "trend_slope_atr_max": 0.20,
    "require_bandwidth_filter": True,
    "bandwidth_max": 0.030,
    "require_vwap_side_filter": False,

    # Risk / target
    "atr_len": 14,
    "stop_mode": "atr",      # "atr" or "swing"
    "atr_stop_mult": 1.0,
    "swing_lookback": 8,
    "stop_buffer_atr": 0.10,
    "target_mode": "midband",  # "midband", "vwap", or "rr"
    "min_rr": 1.0,

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


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = hlc3 * df["volume"]
    session_key = pd.to_datetime(df["datetime"]).dt.date
    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = df["volume"].groupby(session_key).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    RSI + Bollinger Bands Mean Reversion.

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

    bb_len = int(params["bb_len"])
    bb_mult = float(params["bb_mult"])
    out["bb_mid"] = out["close"].rolling(bb_len).mean()
    out["bb_std"] = out["close"].rolling(bb_len).std()
    out["bb_upper"] = out["bb_mid"] + out["bb_std"] * bb_mult
    out["bb_lower"] = out["bb_mid"] - out["bb_std"] * bb_mult
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, np.nan)

    out["rsi"] = _rsi(out["close"], int(params["rsi_len"]))
    out["atr"] = _atr(out, int(params["atr_len"]))
    out["vwap"] = _session_vwap(out)

    out["trend_ma"] = out["close"].rolling(int(params["trend_len"])).mean()
    out["trend_slope"] = (out["trend_ma"] - out["trend_ma"].shift(10)).abs()

    trend_ok = pd.Series(True, index=out.index)
    if bool(params.get("avoid_strong_trend", True)):
        trend_ok = out["trend_slope"] <= out["atr"] * float(params["trend_slope_atr_max"])

    bandwidth_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_bandwidth_filter", True)):
        bandwidth_ok = out["bb_width"] <= float(params["bandwidth_max"])

    vwap_long_ok = pd.Series(True, index=out.index)
    vwap_short_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_vwap_side_filter", False)):
        vwap_long_ok = out["close"] < out["vwap"]
        vwap_short_ok = out["close"] > out["vwap"]

    confirm_mode = str(params["confirm_mode"]).lower()
    if confirm_mode == "reclaim_band":
        long_confirm = (out["close"].shift(1) < out["bb_lower"].shift(1)) & (out["close"] > out["bb_lower"])
        short_confirm = (out["close"].shift(1) > out["bb_upper"].shift(1)) & (out["close"] < out["bb_upper"])
    elif confirm_mode == "reversal_candle":
        long_confirm = (out["low"] <= out["bb_lower"]) & (out["close"] > out["open"])
        short_confirm = (out["high"] >= out["bb_upper"]) & (out["close"] < out["open"])
    else:
        raise ValueError("confirm_mode must be 'reclaim_band' or 'reversal_candle'")

    long_rsi_ok = out["rsi"] <= float(params["rsi_oversold"])
    short_rsi_ok = out["rsi"] >= float(params["rsi_overbought"])

    base = (
        out["in_session"]
        & out["in_entry_window"]
        & out["bb_mid"].notna()
        & out["bb_std"].notna()
        & out["rsi"].notna()
        & out["atr"].notna()
        & trend_ok
        & bandwidth_ok
    )

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    out["long_signal_raw"] = allow_long & base & long_confirm & long_rsi_ok & vwap_long_ok
    out["short_signal_raw"] = allow_short & base & short_confirm & short_rsi_ok & vwap_short_ok

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
    if target_mode == "midband":
        long_tp = out["bb_mid"]
        short_tp = out["bb_mid"]
    elif target_mode == "vwap":
        long_tp = out["vwap"]
        short_tp = out["vwap"]
    elif target_mode == "rr":
        long_tp = out["close"] + long_risk * min_rr
        short_tp = out["close"] - short_risk * min_rr
    else:
        raise ValueError("target_mode must be 'midband', 'vwap', or 'rr'")

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
    out.loc[long_ok, "setup"] = "RSI_BB_MR_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = "RSI_BB_MR_SHORT"

    return out
