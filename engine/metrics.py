import math
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0 or returns.empty:
        return 0.0
    return float((returns.mean() / returns.std()) * math.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0 or downside.empty:
        return 0.0
    return float((returns.mean() / downside.std()) * math.sqrt(periods_per_year))


def profit_factor(trade_returns: pd.Series) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def expectancy(trade_returns: pd.Series) -> float:
    if trade_returns.empty:
        return 0.0
    return float(trade_returns.mean())


def score_report(equity: pd.Series, returns: pd.Series, trade_returns: pd.Series) -> dict:
    mdd = abs(max_drawdown(equity))
    annual_cagr = cagr(equity)
    calmar = annual_cagr / mdd if mdd > 0 else 0.0

    pf = profit_factor(trade_returns)
    if pf == float("inf"):
        pf = 10.0

    report = {
        "cagr": annual_cagr,
        "max_drawdown": mdd,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "calmar": calmar,
        "profit_factor": pf,
        "expectancy": expectancy(trade_returns),
        "trades": int(len(trade_returns)),
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
    }

    report["global_score"] = (
        report["cagr"] * 2.0
        + report["sharpe"] * 0.5
        + report["sortino"] * 0.35
        + report["calmar"] * 0.75
        + min(report["profit_factor"], 3.0) * 0.25
        + report["expectancy"] * 100
        - report["max_drawdown"] * 3.0
    )

    return report
