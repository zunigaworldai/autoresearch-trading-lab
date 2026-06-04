from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",

    # Opening impulse used for confirmation
    "drive_start": "09:30",
    "drive_end": "09:45",

    # Entry window after the opening impulse
    "entry_start": "09:45",
    "entry_end": "15:30",

    # Mode: "continuation" or "fade"
    "gap_mode": "continuation",

    # Direction: "long", "short", "both"
    "direction": "both",

    # Gap filters
    "min_gap_pct": 0.0030,
    "max_gap_pct": 0.0300,
    "max_prev_day_abs_return": 0.0600,
    "require_prev_day_filter": True,

    # Opening impulse confirmation
    "require_drive_direction": True,
    "min_drive_atr": 0.25,
    "max_drive_atr": 3.50,
    "min_drive_body_pct": 0.20,
    "breakout_buffer_atr": 0.05,

    # Trend / alignment filters
    "atr_len": 14,
    "ema_fast_len": 9,
    "ema_slow_len": 21,
    "require_ema_alignment": False,
    "require_vwap_alignment": True,

    # Volume / volatility filters
    "require_volume_confirm": True,
    "vol_len": 20,
    "vol_mult": 1.05,
    "min_atr_pct": 0.0010,
    "max_atr_pct": 0.0600,

    # Risk / target
    "stop_mode": "atr",  # "atr", "drive", "session_open"
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


def _map_session_features(out: pd.DataFrame, drive_mask: pd.Series) -> pd.DataFrame:
    grouped = out.groupby("date", sort=False)
    sessions = pd.DataFrame(
        {
            "session_open": grouped["open"].first(),
            "session_high": grouped["high"].max(),
            "session_low": grouped["low"].min(),
            "session_close": grouped["close"].last(),
        }
    )

    sessions["prev_close"] = sessions["session_close"].shift(1)
    sessions["prev_prev_close"] = sessions["session_close"].shift(2)
    sessions["prev_day_return"] = sessions["prev_close"] / sessions["prev_prev_close"] - 1.0
    sessions["gap_pct"] = sessions["session_open"] / sessions["prev_close"] - 1.0

    drive = out.loc[drive_mask].copy()
    if not drive.empty:
        drive_grouped = drive.groupby("date", sort=False)
        drive_features = pd.DataFrame(
            {
                "drive_open": drive_grouped["open"].first(),
                "drive_close": drive_grouped["close"].last(),
                "drive_high": drive_grouped["high"].max(),
                "drive_low": drive_grouped["low"].min(),
                "drive_volume": drive_grouped["volume"].sum(),
            }
        )
        drive_features["drive_range"] = drive_features["drive_high"] - drive_features["drive_low"]
        drive_features["drive_body"] = (drive_features["drive_close"] - drive_features["drive_open"]).abs()
        drive_features["drive_direction"] = np.where(
            drive_features["drive_close"] > drive_features["drive_open"],
            1,
            np.where(drive_features["drive_close"] < drive_features["drive_open"], -1, 0),
        )
        sessions = sessions.join(drive_features, how="left")
    else:
        for col in [
            "drive_open",
            "drive_close",
            "drive_high",
            "drive_low",
            "drive_volume",
            "drive_range",
            "drive_body",
            "drive_direction",
        ]:
            sessions[col] = np.nan

    for col in sessions.columns:
        out[col] = out["date"].map(sessions[col])

    return out


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Gap Continuation / Fade.

    Hypothesis:
    Overnight/RTH gap creates directional imbalance. After the opening impulse,
    price either continues in gap direction or fades back through the session open.
    Trades are filtered by gap size, ATR regime, volume, VWAP, and initial drive.

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
    out = _map_session_features(out, drive_mask)

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
    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()

    drive_range_atr = out["drive_range"] / out["atr"].replace(0, np.nan)
    drive_body_pct = out["drive_body"] / out["drive_range"].replace(0, np.nan)

    drive_quality = (
        out["drive_high"].notna()
        & out["drive_low"].notna()
        & out["gap_pct"].notna()
        & (drive_range_atr >= float(params["min_drive_atr"]))
        & (drive_range_atr <= float(params["max_drive_atr"]))
        & (drive_body_pct >= float(params["min_drive_body_pct"]))
    )

    gap_abs = out["gap_pct"].abs()
    gap_size_ok = (
        (gap_abs >= float(params["min_gap_pct"]))
        & (gap_abs <= float(params["max_gap_pct"]))
    )

    prev_day_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_prev_day_filter", True)):
        prev_day_ok = out["prev_day_return"].abs() <= float(params["max_prev_day_abs_return"])

    volume_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_volume_confirm", True)):
        volume_ok = out["volume"] >= out["vol_sma"] * float(params["vol_mult"])

    volatility_ok = (
        (out["atr_pct"] >= float(params["min_atr_pct"]))
        & (out["atr_pct"] <= float(params["max_atr_pct"]))
    )

    trend_long_ok = pd.Series(True, index=out.index)
    trend_short_ok = pd.Series(True, index=out.index)

    if bool(params.get("require_ema_alignment", False)):
        trend_long_ok &= out["ema_fast"] > out["ema_slow"]
        trend_short_ok &= out["ema_fast"] < out["ema_slow"]

    if bool(params.get("require_vwap_alignment", True)):
        trend_long_ok &= out["close"] > out["vwap"]
        trend_short_ok &= out["close"] < out["vwap"]

    if bool(params.get("require_drive_direction", True)):
        drive_long_ok = out["drive_direction"] == 1
        drive_short_ok = out["drive_direction"] == -1
    else:
        drive_long_ok = pd.Series(True, index=out.index)
        drive_short_ok = pd.Series(True, index=out.index)

    buffer = out["atr"] * float(params["breakout_buffer_atr"])
    gap_mode = str(params["gap_mode"]).lower()

    if gap_mode == "continuation":
        long_gap_ok = out["gap_pct"] > 0
        short_gap_ok = out["gap_pct"] < 0
        long_trigger = (out["close"].shift(1) <= out["drive_high"].shift(1)) & (out["close"] > out["drive_high"] + buffer)
        short_trigger = (out["close"].shift(1) >= out["drive_low"].shift(1)) & (out["close"] < out["drive_low"] - buffer)
    elif gap_mode == "fade":
        long_gap_ok = out["gap_pct"] < 0
        short_gap_ok = out["gap_pct"] > 0
        long_trigger = (out["close"].shift(1) <= out["session_open"].shift(1)) & (out["close"] > out["session_open"] + buffer)
        short_trigger = (out["close"].shift(1) >= out["session_open"].shift(1)) & (out["close"] < out["session_open"] - buffer)
    else:
        raise ValueError("gap_mode must be 'continuation' or 'fade'")

    base = (
        out["in_entry_window"]
        & out["atr"].notna()
        & out["vwap"].notna()
        & drive_quality
        & gap_size_ok
        & prev_day_ok
        & volume_ok
        & volatility_ok
    )

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    out["long_signal_raw"] = (
        allow_long
        & base
        & long_gap_ok
        & long_trigger
        & trend_long_ok
        & drive_long_ok
    )

    out["short_signal_raw"] = (
        allow_short
        & base
        & short_gap_ok
        & short_trigger
        & trend_short_ok
        & drive_short_ok
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
    elif stop_mode == "drive":
        long_stop = out["drive_low"] - stop_buffer
        short_stop = out["drive_high"] + stop_buffer
    elif stop_mode == "session_open":
        long_stop = out["session_open"] - stop_buffer
        short_stop = out["session_open"] + stop_buffer
    else:
        raise ValueError("stop_mode must be 'atr', 'drive', or 'session_open'")

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

    long_setup = "GAP_CONT_LONG" if gap_mode == "continuation" else "GAP_FADE_LONG"
    short_setup = "GAP_CONT_SHORT" if gap_mode == "continuation" else "GAP_FADE_SHORT"

    out.loc[long_ok, "exit_stop"] = long_stop[long_ok]
    out.loc[long_ok, "exit_tp"] = long_tp[long_ok]
    out.loc[long_ok, "setup"] = long_setup

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = short_setup

    return out
