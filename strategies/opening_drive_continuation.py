from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",

    # Opening drive definition
    "drive_start": "09:30",
    "drive_end": "09:45",

    # Entry window after drive is known
    "entry_start": "09:45",
    "entry_end": "11:30",

    # Direction: "long", "short", "both"
    "direction": "both",

    # Entry model: "breakout" or "two_bar_hold"
    "entry_mode": "breakout",
    "breakout_buffer_atr": 0.05,

    # Opening drive quality filters
    "min_drive_atr": 0.60,
    "max_drive_atr": 3.50,
    "min_drive_body_pct": 0.35,

    # Trend / alignment filters
    "atr_len": 14,
    "ema_fast_len": 9,
    "ema_slow_len": 21,
    "trend_len": 50,
    "require_ema_alignment": True,
    "require_vwap_alignment": True,

    # Volume filter
    "require_volume_confirm": True,
    "vol_len": 20,
    "vol_mult": 1.10,

    # Volatility filter
    "min_atr_pct": 0.0010,
    "max_atr_pct": 0.0500,

    # Risk / target
    "stop_mode": "atr",  # "atr", "drive_opposite", "swing"
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


def _map_opening_drive_features(out: pd.DataFrame, drive_mask: pd.Series) -> pd.DataFrame:
    drive = out.loc[drive_mask].copy()

    if drive.empty:
        for col in [
            "drive_open", "drive_close", "drive_high", "drive_low",
            "drive_volume", "drive_range", "drive_body", "drive_direction",
        ]:
            out[col] = np.nan
        return out

    grouped = drive.groupby("date", sort=False)

    features = pd.DataFrame({
        "drive_open": grouped["open"].first(),
        "drive_close": grouped["close"].last(),
        "drive_high": grouped["high"].max(),
        "drive_low": grouped["low"].min(),
        "drive_volume": grouped["volume"].sum(),
    })

    features["drive_range"] = features["drive_high"] - features["drive_low"]
    features["drive_body"] = (features["drive_close"] - features["drive_open"]).abs()
    features["drive_direction"] = np.where(
        features["drive_close"] > features["drive_open"], 1,
        np.where(features["drive_close"] < features["drive_open"], -1, 0),
    )

    for col in features.columns:
        out[col] = out["date"].map(features[col])

    return out


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Opening Drive Continuation.

    Hypothesis:
    If the first 15-30 minutes create a directional impulse with enough range/body
    and price later breaks/holds beyond that drive in the same direction, momentum may continue.

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

    drive_mask = out["in_session"] & _in_window_left_closed(out, params["drive_start"], params["drive_end"])
    out = _map_opening_drive_features(out, drive_mask)

    out["in_entry_window"] = (
        out["in_session"]
        & _in_window_inclusive(out, params["entry_start"], params["entry_end"])
        & (pd.to_datetime(out["datetime"]).dt.time >= _time_value(params["drive_end"]))
    )

    out["atr"] = _atr(out, int(params["atr_len"]))
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["vwap"] = _session_vwap(out)

    out["ema_fast"] = out["close"].ewm(span=int(params["ema_fast_len"]), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=int(params["ema_slow_len"]), adjust=False).mean()
    out["trend_ma"] = out["close"].rolling(int(params["trend_len"])).mean()

    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()

    drive_range_atr = out["drive_range"] / out["atr"].replace(0, np.nan)
    drive_body_pct = out["drive_body"] / out["drive_range"].replace(0, np.nan)

    drive_quality = (
        out["drive_high"].notna()
        & out["drive_low"].notna()
        & out["drive_direction"].notna()
        & (drive_range_atr >= float(params["min_drive_atr"]))
        & (drive_range_atr <= float(params["max_drive_atr"]))
        & (drive_body_pct >= float(params["min_drive_body_pct"]))
    )

    volume_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_volume_confirm", True)):
        volume_ok = out["volume"] >= out["vol_sma"] * float(params["vol_mult"])

    volatility_ok = (
        (out["atr_pct"] >= float(params["min_atr_pct"]))
        & (out["atr_pct"] <= float(params["max_atr_pct"]))
    )

    long_alignment = pd.Series(True, index=out.index)
    short_alignment = pd.Series(True, index=out.index)

    if bool(params.get("require_ema_alignment", True)):
        long_alignment &= out["ema_fast"] > out["ema_slow"]
        long_alignment &= out["close"] > out["trend_ma"]
        short_alignment &= out["ema_fast"] < out["ema_slow"]
        short_alignment &= out["close"] < out["trend_ma"]

    if bool(params.get("require_vwap_alignment", True)):
        long_alignment &= out["close"] > out["vwap"]
        short_alignment &= out["close"] < out["vwap"]

    buffer = out["atr"] * float(params["breakout_buffer_atr"])
    mode = str(params["entry_mode"]).lower()

    if mode == "breakout":
        long_trigger = (out["close"].shift(1) <= out["drive_high"].shift(1)) & (out["close"] > out["drive_high"] + buffer)
        short_trigger = (out["close"].shift(1) >= out["drive_low"].shift(1)) & (out["close"] < out["drive_low"] - buffer)
    elif mode == "two_bar_hold":
        long_trigger = (
            (out["close"].shift(1) > out["drive_high"].shift(1))
            & (out["close"] > out["drive_high"] + buffer)
        )
        short_trigger = (
            (out["close"].shift(1) < out["drive_low"].shift(1))
            & (out["close"] < out["drive_low"] - buffer)
        )
    else:
        raise ValueError("entry_mode must be 'breakout' or 'two_bar_hold'")

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    base = (
        out["in_entry_window"]
        & out["atr"].notna()
        & out["trend_ma"].notna()
        & drive_quality
        & volume_ok
        & volatility_ok
    )

    out["long_signal_raw"] = (
        allow_long
        & base
        & (out["drive_direction"] == 1)
        & long_trigger
        & long_alignment
    )

    out["short_signal_raw"] = (
        allow_short
        & base
        & (out["drive_direction"] == -1)
        & short_trigger
        & short_alignment
    )

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
    elif stop_mode == "drive_opposite":
        long_stop = out["drive_low"] - stop_buffer
        short_stop = out["drive_high"] + stop_buffer
    elif stop_mode == "swing":
        long_stop = out["low"].rolling(int(params["swing_lookback"])).min() - stop_buffer
        short_stop = out["high"].rolling(int(params["swing_lookback"])).max() + stop_buffer
    else:
        raise ValueError("stop_mode must be 'atr', 'drive_opposite', or 'swing'")

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
    out.loc[long_ok, "setup"] = "OPENING_DRIVE_CONT_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = "OPENING_DRIVE_CONT_SHORT"

    return out
