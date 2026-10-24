"""组合权重：Top-N 选股 + 等权。

学习期从等权起步：不引入优化器，避免把"权重优化"这个复杂问题
和"因子有没有效"混在一起。后续可扩展风险平价/最小方差。
"""

import pandas as pd


def top_n_equal_weight(score: pd.Series, top_n: int = 30) -> pd.Series:
    """取总分最高的 top_n 只，等权。返回权重 Series（和为 1）。"""
    s = score.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    n = min(top_n, len(s))
    top = s.nlargest(n)
    return pd.Series(1.0 / n, index=top.index)


def build_weight_schedule(combined_score: pd.DataFrame,
                          rebal_dates: pd.DatetimeIndex,
                          signal_lag_days: int,
                          close_index: pd.DatetimeIndex,
                          top_n: int = 30) -> dict[pd.Timestamp, pd.Series]:
    """为每个调仓日生成目标权重。

    无未来函数关键：调仓日 T 的信号取自「T 日前第 signal_lag_days 个交易日」
    的因子值（signal_lag_days≥1），即只用 ≤T-1 的数据。
    """
    schedule = {}
    pos = {d: i for i, d in enumerate(close_index)}
    for dt in rebal_dates:
        if dt not in pos:
            continue
        sig_idx = pos[dt] - signal_lag_days
        if sig_idx < 0:
            continue
        sig_date = close_index[sig_idx]
        if sig_date not in combined_score.index:
            continue
        w = top_n_equal_weight(combined_score.loc[sig_date], top_n)
        if len(w) > 0:
            schedule[dt] = w
    return schedule
