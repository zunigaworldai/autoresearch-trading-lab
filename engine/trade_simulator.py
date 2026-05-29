from __future__ import annotations

import pandas as pd


VALID_EXIT_POLICIES = {"fixed", "break_even_1r", "partial_50_at_1r_be"}


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
        "partial_taken": False,
        "partial_fraction": 0.0,
        "remaining_fraction": 1.0,
        "partial_exit_price": None,
        "partial_exit_time": None,
        "partial_exit_reason": "",
        "partial_gross_return": 0.0,
    }


def _side_label(position: int) -> str:
    if position == 1:
        return "LONG"
    if position == -1:
        return "SHORT"
    raise ValueError("Invalid position")


def _gross_return_for_exit(position: int, entry_price: float, exit_price: float) -> float:
    if position == 1:
        return exit_price / entry_price - 1.0
    if position == -1:
        return entry_price / exit_price - 1.0
    raise ValueError("Cannot compute return without open position")


def _total_gross_return(position_state: dict, final_exit_price: float) -> float:
    position = int(position_state["position"])
    entry_price = float(position_state["entry_price"])
    final_gross = _gross_return_for_exit(position, entry_price, final_exit_price)

    partial_fraction = float(position_state.get("partial_fraction", 0.0))
    remaining_fraction = float(position_state.get("remaining_fraction", 1.0))
    partial_gross_return = float(position_state.get("partial_gross_return", 0.0))

    return (partial_fraction * partial_gross_return) + (remaining_fraction * final_gross)


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

    gross_return = _total_gross_return(position_state, exit_price)
    net_return = gross_return - (cost_per_side * 2)

    trade_returns.append(net_return)

    trades.append(
        {
            "entry_time": str(position_state["entry_time"]),
            "exit_time": str(exit_time),
            "side": _side_label(position),
            "setup": position_state["setup"],
            "entry": entry_price,
            "exit": exit_price,
            "stop": position_state.get("initial_stop_price", position_state["stop_price"]),
            "final_stop": position_state["stop_price"],
            "tp": position_state["tp_price"],
            "exit_reason": exit_reason,
            "be_triggered": bool(position_state.get("be_triggered", False)),
            "partial_taken": bool(position_state.get("partial_taken", False)),
            "partial_fraction": float(position_state.get("partial_fraction", 0.0)),
            "remaining_fraction": float(position_state.get("remaining_fraction", 1.0)),
            "partial_exit_price": position_state.get("partial_exit_price"),
            "partial_exit_time": str(position_state["partial_exit_time"]) if position_state.get("partial_exit_time") is not None else "",
            "partial_exit_reason": position_state.get("partial_exit_reason", ""),
            "partial_gross_return": float(position_state.get("partial_gross_return", 0.0)),
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


def _check_exit(row: pd.Series, position_state: dict) -> tuple[bool, float | None, str | None]:
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


def _initial_risk(position_state: dict) -> float:
    position = int(position_state["position"])
    entry_price = float(position_state["entry_price"])
    initial_stop = float(position_state["initial_stop_price"])

    if position == 1:
        return entry_price - initial_stop
    if position == -1:
        return initial_stop - entry_price
    return 0.0


def _one_r_trigger_price(position_state: dict) -> float | None:
    position = int(position_state["position"])
    entry_price = float(position_state["entry_price"])
    risk = _initial_risk(position_state)

    if risk <= 0:
        return None
    if position == 1:
        return entry_price + risk
    if position == -1:
        return entry_price - risk
    return None


def _reached_price(row: pd.Series, position_state: dict, trigger_price: float) -> bool:
    position = int(position_state["position"])

    if position == 1:
        return float(row["high"]) >= trigger_price
    if position == -1:
        return float(row["low"]) <= trigger_price
    return False


def _move_stop_to_break_even(position_state: dict) -> None:
    position = int(position_state["position"])
    entry_price = float(position_state["entry_price"])

    if position == 1:
        position_state["stop_price"] = max(float(position_state["stop_price"]), entry_price)
    elif position == -1:
        position_state["stop_price"] = min(float(position_state["stop_price"]), entry_price)

    position_state["be_triggered"] = True


def _apply_break_even_1r(row: pd.Series, position_state: dict) -> None:
    if bool(position_state.get("be_triggered", False)):
        return

    trigger_price = _one_r_trigger_price(position_state)
    if trigger_price is None:
        return

    if _reached_price(row, position_state, trigger_price):
        _move_stop_to_break_even(position_state)


def _apply_partial_50_at_1r_be(row: pd.Series, position_state: dict) -> None:
    if bool(position_state.get("partial_taken", False)):
        return

    trigger_price = _one_r_trigger_price(position_state)
    if trigger_price is None:
        return

    if not _reached_price(row, position_state, trigger_price):
        return

    position = int(position_state["position"])
    entry_price = float(position_state["entry_price"])

    position_state["partial_taken"] = True
    position_state["partial_fraction"] = 0.5
    position_state["remaining_fraction"] = 0.5
    position_state["partial_exit_price"] = float(trigger_price)
    position_state["partial_exit_time"] = row["_dt"]
    position_state["partial_exit_reason"] = "PARTIAL_1R"
    position_state["partial_gross_return"] = _gross_return_for_exit(position, entry_price, float(trigger_price))

    _move_stop_to_break_even(position_state)


def _apply_exit_policy(row: pd.Series, position_state: dict, exit_policy: str) -> None:
    if exit_policy == "fixed":
        return
    if exit_policy == "break_even_1r":
        _apply_break_even_1r(row, position_state)
        return
    if exit_policy == "partial_50_at_1r_be":
        _apply_partial_50_at_1r_be(row, position_state)
        return

    raise ValueError(f"Unsupported exit_policy: {exit_policy}")


def run_trade_simulation(
    signals: pd.DataFrame,
    cost_per_side: float = 0.0005,
    force_eod_exit: bool = True,
    exit_policy: str = "fixed",
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
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
            position = int(state["position"])
            entry_price = float(state["entry_price"])

            unrealized_full = _gross_return_for_exit(position, entry_price, close_price)

            partial_fraction = float(state.get("partial_fraction", 0.0))
            remaining_fraction = float(state.get("remaining_fraction", 1.0))
            partial_gross_return = float(state.get("partial_gross_return", 0.0))

            unrealized_total = (partial_fraction * partial_gross_return) + (remaining_fraction * unrealized_full)
            current_equity = cash_equity * (1 + unrealized_total)

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
                    "partial_taken": False,
                    "partial_fraction": 0.0,
                    "remaining_fraction": 1.0,
                    "partial_exit_price": None,
                    "partial_exit_time": None,
                    "partial_exit_reason": "",
                    "partial_gross_return": 0.0,
                }

        equity_values.append(current_equity)

    equity = pd.Series(equity_values, index=signals.index, name="equity")
    trade_returns_series = pd.Series(trade_returns, name="trade_returns")
    trades_df = pd.DataFrame(trades)

    return equity, trade_returns_series, trades_df