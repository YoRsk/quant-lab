"""绩效指标：所有函数输入日频净值/收益序列，输出口径注明（252 交易日年化）。"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def perf_summary(equity: pd.Series, turnover: pd.Series | None = None,
                 cost: pd.Series | None = None) -> dict:
    """核心绩效汇总。equity 为总资产日序列。"""
    ret = equity.pct_change().dropna()
    n_years = len(ret) / TRADING_DAYS
    if len(ret) < 2 or n_years <= 0:
        return {"note": "样本不足"}

    total = equity.iloc[-1] / equity.iloc[0] - 1
    ann = (1 + total) ** (1 / n_years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann / vol if vol > 0 else np.nan          # 简化：无风险利率取 0
    dd = (equity / equity.cummax() - 1)
    mdd = dd.min()
    out = {
        "起止": f"{equity.index[0].date()} ~ {equity.index[-1].date()}",
        "累计收益": f"{total:.1%}",
        "年化收益": f"{ann:.2%}",
        "年化波动": f"{vol:.2%}",
        "夏普比率": round(sharpe, 2),
        "最大回撤": f"{mdd:.1%}",
        "Calmar": round(ann / abs(mdd), 2) if mdd < 0 else np.nan,
    }
    if turnover is not None and len(turnover) > 0:
        rebs_per_year = len(turnover) / n_years
        out["年均单边换手"] = f"{turnover.mean() * rebs_per_year:.0%}"
    if cost is not None and len(cost) > 0:
        out["累计交易成本"] = f"${cost.sum():,.0f}"
        out["成本占初始资金"] = f"{cost.sum() / equity.iloc[0]:.2%}"
    return out
