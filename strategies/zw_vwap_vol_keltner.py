from __future__ import annotations

import pandas as pd
import numpy as np


DEFAULT_PARAMS = {
    # Pine defaults
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",
    "min_rr": 2.0,

    "ema_len": 20,
    "atr_len": 20,
    "atr_mult": 2.0,

    "vol_len": 20,
    "vol_mult": 1.2,
    "pb_vol_max": 0.9,

    "enable_a": True,    # Setup A - Pullback VWAP
    "enable_b": False,   # Setup B - Rechazo VWAP
    "enable_c": False,   # Setup C - Breakout Keltner
}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    out = df.copy()

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _get_datetime_series(df: pd.DataFrame) -> pd.Series | None:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")

    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce")

    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)

    return None


def _in_session(df: pd.DataFrame, params: dict) -> pd.Series:
    if not params.get("use_rth", True):
        return pd.Series(True, index=df.index)

    dt = _get_datetime_series(df)

    if dt is None:
        # If no datetime exists, do not block signals.
        return pd.Series(True, index=df.index)

    start = params.get("session_start", "09:30")
    end = params.get("session_end", "16:00")

    t = dt.dt.time
    start_t = pd.to_datetime(start).time()
    end_t = pd.to_datetime(end).time()

    return pd.Series((t >= start_t) & (t <= end_t), index=df.index)


def _rma(series: pd.Series, length: int) -> pd.Series:
    # Pine ta.atr uses Wilder RMA.
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
    """
    Approximation of TradingView ta.vwap(hlc3).

    If datetime/timestamp/index exists, VWAP resets by date.
    If no datetime exists, VWAP is cumulative across the full dataset.
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = hlc3 * df["volume"]

    dt = _get_datetime_series(df)

    if dt is None:
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].cumsum()
        return cum_pv / cum_vol.replace(0, np.nan)

    session_key = dt.dt.date

    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = df["volume"].groupby(session_key).cumsum()

    return cum_pv / cum_vol.replace(0, np.nan)


def _bars_since(condition: pd.Series) -> pd.Series:
    result = []
    last_true_index = None

    for i, value in enumerate(condition.fillna(False).astype(bool).to_numpy()):
        if value:
            last_true_index = i
            result.append(0)
        elif last_true_index is None:
            result.append(np.nan)
        else:
            result.append(i - last_true_index)

    return pd.Series(result, index=condition.index)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Python version of:
    pine/ZW_VWAP_VOL_KELTNER_v1.pine

    Required input columns:
    - open
    - high
    - low
    - close
    - volume

    Optional:
    - datetime or timestamp column, or DatetimeIndex.

    Output columns:
    - entry_signal: 1 long, -1 short, 0 no entry
    - exit_stop
    - exit_tp
    - setup
    - vwap
    - basis
    - upper
    - lower
    - atr
    - no_trade
    """

    params = {**DEFAULT_PARAMS, **(params or {})}

    out = _ensure_ohlcv(df)

    ema_len = int(params["ema_len"])
    atr_len = int(params["atr_len"])
    atr_mult = float(params["atr_mult"])

    vol_len = int(params["vol_len"])
    vol_mult = float(params["vol_mult"])
    pb_vol_max = float(params["pb_vol_max"])
    min_rr = float(params["min_rr"])

    enable_a = bool(params["enable_a"])
    enable_b = bool(params["enable_b"])
    enable_c = bool(params["enable_c"])

    out["in_session"] = _in_session(out, params)

    # Indicators
    out["vwap"] = _session_vwap(out)
    out["basis"] = out["close"].ewm(span=ema_len, adjust=False).mean()
    out["atr"] = _atr(out, atr_len)

    out["upper"] = out["basis"] + atr_mult * out["atr"]
    out["lower"] = out["basis"] - atr_mult * out["atr"]

    out["vol_sma"] = out["volume"].rolling(vol_len).mean()
    out["vol_ok"] = out["volume"] > out["vol_sma"] * vol_mult

    out["width"] = out["upper"] - out["lower"]
    out["width_sma"] = out["width"].rolling(50).mean()

    out["squeeze"] = out["width_sma"].notna() & (out["width"] < out["width_sma"] * 0.70)
    out["vwap_flat"] = (
        out["vwap"].shift(10).notna()
        & ((out["vwap"] - out["vwap"].shift(10)).abs() < out["atr"] * 0.10)
    )

    out["no_trade"] = out["squeeze"] & out["vwap_flat"]

    out["long_bias"] = (out["close"] > out["vwap"]) & (out["basis"] > out["basis"].shift(1))
    out["short_bias"] = (out["close"] < out["vwap"]) & (out["basis"] < out["basis"].shift(1))

    # Setup A: Pullback VWAP Long
    out["touch_tol"] = out["atr"] * 0.15

    out["pb_condition"] = (
        out["long_bias"]
        & (out["low"] <= out["vwap"] + out["touch_tol"])
        & (out["volume"] < out["vol_sma"] * pb_vol_max)
    )

    out["bars_since_pb"] = _bars_since(out["pb_condition"])
    out["pb_recent"] = out["bars_since_pb"].notna() & (out["bars_since_pb"] <= 5)

    out["confirm_bull"] = (
        (out["close"] > out["open"])
        & (out["close"] > out["vwap"])
        & (out["volume"] > out["vol_sma"] * vol_mult)
    )

    out["a_long"] = (
        enable_a
        & out["in_session"]
        & (~out["no_trade"])
        & out["long_bias"]
        & out["pb_recent"]
        & out["confirm_bull"]
    )

    out["a_stop"] = out["low"].rolling(8).min() - out["atr"] * 0.10
    out["a_risk"] = out["close"] - out["a_stop"]
    out["a_tp2"] = out["close"] + out["a_risk"] * min_rr
    out["a_ok"] = out["a_long"] & (out["a_risk"] > 0)

    # Setup B: Rechazo VWAP Short
    out["b_rej"] = (
        enable_b
        & out["in_session"]
        & (~out["no_trade"])
        & out["short_bias"]
        & (out["high"].shift(1) >= (out["vwap"].shift(1) - out["touch_tol"]))
        & (out["close"].shift(1) < out["vwap"].shift(1))
        & (out["close"].shift(1) < out["open"].shift(1))
        & (out["volume"].shift(1) > out["vol_sma"].shift(1) * vol_mult)
    )

    out["b_short"] = out["b_rej"] & (out["close"] < out["low"].shift(1)) & out["vol_ok"]

    out["b_stop"] = pd.concat([out["vwap"], out["high"].shift(1)], axis=1).max(axis=1) + out["atr"] * 0.10
    out["b_risk"] = out["b_stop"] - out["close"]
    out["b_tp2"] = out["close"] - out["b_risk"] * min_rr
    out["b_ok"] = out["b_short"] & (out["b_risk"] > 0)

    # Setup C: Breakout Keltner + Volume
    out["c_long"] = (
        enable_c
        & out["in_session"]
        & (~out["no_trade"])
        & out["long_bias"]
        & out["squeeze"]
        & (out["close"] > out["upper"])
        & out["vol_ok"]
    )

    out["c_short"] = (
        enable_c
        & out["in_session"]
        & (~out["no_trade"])
        & out["short_bias"]
        & out["squeeze"]
        & (out["close"] < out["lower"])
        & out["vol_ok"]
    )

    out["c_stop_l"] = out["basis"] - out["atr"] * 0.10
    out["c_stop_s"] = out["basis"] + out["atr"] * 0.10

    out["c_risk_l"] = out["close"] - out["c_stop_l"]
    out["c_risk_s"] = out["c_stop_s"] - out["close"]

    out["c_tp2_l"] = out["close"] + out["c_risk_l"] * min_rr
    out["c_tp2_s"] = out["close"] - out["c_risk_s"] * min_rr

    out["c_long_ok"] = out["c_long"] & (out["c_risk_l"] > 0)
    out["c_short_ok"] = out["c_short"] & (out["c_risk_s"] > 0)

    # Order priority matches Pine order:
    # C long -> A long -> C short -> B short
    out["entry_signal"] = 0
    out["exit_stop"] = np.nan
    out["exit_tp"] = np.nan
    out["setup"] = ""

    c_long_mask = out["c_long_ok"]
    a_long_mask = (out["entry_signal"] == 0) & out["a_ok"]
    c_short_mask = (out["entry_signal"] == 0) & out["c_short_ok"]
    b_short_mask = (out["entry_signal"] == 0) & out["b_ok"]

    out.loc[c_long_mask, "entry_signal"] = 1
    out.loc[c_long_mask, "exit_stop"] = out.loc[c_long_mask, "c_stop_l"]
    out.loc[c_long_mask, "exit_tp"] = out.loc[c_long_mask, "c_tp2_l"]
    out.loc[c_long_mask, "setup"] = "L-C"

    out.loc[a_long_mask, "entry_signal"] = 1
    out.loc[a_long_mask, "exit_stop"] = out.loc[a_long_mask, "a_stop"]
    out.loc[a_long_mask, "exit_tp"] = out.loc[a_long_mask, "a_tp2"]
    out.loc[a_long_mask, "setup"] = "L-A"

    out.loc[c_short_mask, "entry_signal"] = -1
    out.loc[c_short_mask, "exit_stop"] = out.loc[c_short_mask, "c_stop_s"]
    out.loc[c_short_mask, "exit_tp"] = out.loc[c_short_mask, "c_tp2_s"]
    out.loc[c_short_mask, "setup"] = "S-C"

    out.loc[b_short_mask, "entry_signal"] = -1
    out.loc[b_short_mask, "exit_stop"] = out.loc[b_short_mask, "b_stop"]
    out.loc[b_short_mask, "exit_tp"] = out.loc[b_short_mask, "b_tp2"]
    out.loc[b_short_mask, "setup"] = "S-B"

    return out