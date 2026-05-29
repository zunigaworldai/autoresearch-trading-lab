from __future__ import annotations

import pandas as pd


VALID_EXIT_POLICIES = {"fixed", "break_even_1r"}


def _prepare_signals(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()

    if "datetime" in out.columns:
        out["_dt"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif "timestamp" in out.columns:
        out["_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out["_dt"] = pd.to_datetime(out.index, errors="coerce")
    else:
        raise ValueError("Signals must contain datetime, timestamp, or DatetimeIndex")

    if out["_dt"].isna().any():
        raise ValueError("Signals contain invalid datetime values")

    out["_date"] = out["_dt"].dt.date
    out["_next_date"] = out["_date"].shift(-1)
    out["_is_eod_bar"] = out["_next_date"].isna() | (out["_next_date"] != out["_date"])

    return out


def _reset_position() -> dict:
    return {
        "position": 0,
        "entry_price": None,
        "entry_time": None,
        "initial_stop_price": None,
        "stop_price": None,
        "tp_price": None,
        "setup": None,
        "be_triggered": False,
    }


def _close_trade(
    trades: list[dict],
    trade_returns: list[float],
    position_state: dict,
    exit_time,
    exit_price: float,
    exit_reason: str,
    cost_per_side: float,
) -> float:
    position = int(position_state["position"])
    entry_price = float(position_state["entry_price"])

    if position == 1:
        gross_return = exit_price / entry_price - 1.0
        side = "LONG"
    elif position == -1:
        gross_return = entry_price / exit_price - 1.0
        side = "SHORT"
    else:
        raise ValueError("Cannot close trade without open position")

    net_return = gross_return - (cost_per_side * 2)

    trade_returns.append(net_return)

    trades.append(
        {
            "entry_time": str(position_state["entry_time"]),
            "exit_time": str(exit_time),
            "side": side,
            "setup": position_state["setup"],
            "entry": entry_price,
            "exit": exit_price,
            "stop": position_state.get("initial_stop_price", position_state["stop_price"]),
            "final_stop": position_state["stop_price"],
            "tp": position_state["tp_price"],
            "exit_reason": exit_reason,
            "be_triggered": bool(position_state.get("be_triggered", False)),
            "gross_return": gross_return,
            "net_return": net_return,
        }
    )

    return net_return


def _stop_exit_reason(position_state: dict) -> str:
    entry_price = float(position_state["entry_price"])
    stop_price = float(position_state["stop_price"])

    if bool(position_state.get("be_triggered", False)) and abs(stop_price - entry_price) < 1e-9:
        return "BE"

    return "STOP"


def _check_exit(
    row: pd.Series,
    position_state: dict,
) -> tuple[bool, float | None, str | None]:
    position = int(position_state["position"])
    stop_price = float(position_state["stop_price"])
    tp_price = float(position_state["tp_price"])

    if position == 1:
        stop_hit = float(row["low"]) <= stop_price
        tp_hit = float(row["high"]) >= tp_price

        if stop_hit or tp_hit:
            exit_reason = _stop_exit_reason(position_state) if stop_hit else "TP"
            exit_price = stop_price if stop_hit else tp_price
            return True, exit_price, exit_reason

    elif position == -1:
        stop_hit = float(row["high"]) >= stop_price
        tp_hit = float(row["low"]) <= tp_price

        if stop_hit or tp_hit:
            exit_reason = _stop_exit_reason(position_state) if stop_hit else "TP"
            exit_price = stop_price if stop_hit else tp_price
            return True, exit_price, exit_reason

    return False, None, None


def _apply_break_even_1r(row: pd.Series, position_state: dict) -> None:
    if bool(position_state.get("be_triggered", False)):
        return

    position = int(position_state["position"])
    entry_price = float(position_state["entry_price"])
    initial_stop = float(position_state["initial_stop_price"])

    if position == 1:
        initial_risk = entry_price - initial_stop

        if initial_risk <= 0:
            return

        trigger_price = entry_price + initial_risk

        if float(row["high"]) >= trigger_price:
            position_state["stop_price"] = max(float(position_state["stop_price"]), entry_price)
            position_state["be_triggered"] = True

    elif position == -1:
        initial_risk = initial_stop - entry_price

        if initial_risk <= 0:
            return

        trigger_price = entry_price - initial_risk

        if float(row["low"]) <= trigger_price:
            position_state["stop_price"] = min(float(position_state["stop_price"]), entry_price)
            position_state["be_triggered"] = True


def _apply_exit_policy(row: pd.Series, position_state: dict, exit_policy: str) -> None:
    if exit_policy == "fixed":
        return

    if exit_policy == "break_even_1r":
        _apply_break_even_1r(row, position_state)
        return

    raise ValueError(f"Unsupported exit_policy: {exit_policy}")


def run_trade_simulation(
    signals: pd.DataFrame,
    cost_per_side: float = 0.0005,
    force_eod_exit: bool = True,
    exit_policy: str = "fixed",
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Conservative single-position backtest.

    Execution model:
    - Entry at signal bar close.
    - Stop/TP checked from the next bars using high/low.
    - If stop and TP are touched in the same bar, stop is assumed first.
    - Cost is charged on both entry and exit.
    - If force_eod_exit=True, open positions are closed at the final bar of each trading day.
    - exit_policy="fixed" preserves original behavior.
    - exit_policy="break_even_1r" moves stop to entry after trade reaches +1R.
      Conservative ordering: BE activation applies after checking the current bar exit,
      so same-bar +1R/BE reversal is not credited.
    """

    if exit_policy not in VALID_EXIT_POLICIES:
        raise ValueError(f"Invalid exit_policy={exit_policy}. Valid: {sorted(VALID_EXIT_POLICIES)}")

    signals = _prepare_signals(signals)

    equity_values = []
    trade_returns = []
    trades = []

    cash_equity = 1.0
    state = _reset_position()

    for _, row in signals.iterrows():
        current_equity = cash_equity

        if int(state["position"]) != 0:
            close_price = float(row["close"])

            if int(state["position"]) == 1:
                unrealized = close_price / float(state["entry_price"]) - 1.0
            else:
                unrealized = float(state["entry_price"]) / close_price - 1.0

            current_equity = cash_equity * (1 + unrealized)

            exit_hit, exit_price, exit_reason = _check_exit(row, state)

            if exit_hit:
                net_return = _close_trade(
                    trades=trades,
                    trade_returns=trade_returns,
                    position_state=state,
                    exit_time=row["_dt"],
                    exit_price=float(exit_price),
                    exit_reason=str(exit_reason),
                    cost_per_side=cost_per_side,
                )

                cash_equity *= 1 + net_return
                current_equity = cash_equity
                state = _reset_position()

            else:
                _apply_exit_policy(row, state, exit_policy)

                if force_eod_exit and bool(row["_is_eod_bar"]):
                    net_return = _close_trade(
                        trades=trades,
                        trade_returns=trade_returns,
                        position_state=state,
                        exit_time=row["_dt"],
                        exit_price=close_price,
                        exit_reason="EOD",
                        cost_per_side=cost_per_side,
                    )

                    cash_equity *= 1 + net_return
                    current_equity = cash_equity
                    state = _reset_position()

        allow_new_entry = not (force_eod_exit and bool(row["_is_eod_bar"]))

        if allow_new_entry and int(state["position"]) == 0 and int(row["entry_signal"]) != 0:
            if not pd.isna(row["exit_stop"]) and not pd.isna(row["exit_tp"]):
                state = {
                    "position": int(row["entry_signal"]),
                    "entry_price": float(row["close"]),
                    "entry_time": row["_dt"],
                    "initial_stop_price": float(row["exit_stop"]),
                    "stop_price": float(row["exit_stop"]),
                    "tp_price": float(row["exit_tp"]),
                    "setup": row.get("setup", ""),
                    "be_triggered": False,
                }

        equity_values.append(current_equity)

    equity = pd.Series(equity_values, index=signals.index, name="equity")
    returns = equity.pct_change().fillna(0.0)
    trade_returns_series = pd.Series(trade_returns, name="trade_returns")
    trades_df = pd.DataFrame(trades)

    return equity, trade_returns_series, trades_df