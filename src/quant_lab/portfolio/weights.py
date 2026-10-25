"""组合权重：Top-N 选股 + 等权。

学习期从等权起步：不引入优化器，避免把"权重优化"这个复杂问题
和"因子有没有效"混在一起。后续可扩展风险平价/最小方差。
"""

import pandas as pd

from .bands import top_n_with_band


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
                          top_n: int = 30, buffer: int = 0) -> dict[pd.Timestamp, pd.Series]:
    """为每个调仓日生成目标权重。

    无未来函数关键：调仓日 T 的信号取自「T 日前第 signal_lag_days 个交易日」
    的因子值（signal_lag_days≥1），即只用 ≤T-1 的数据。

    buffer > 0 时启用换手缓冲带：已持有股票跌出 Top(N+buffer) 才卖出，
    新股票进入 Top(N-buffer) 才买入，用于削减无效交易与手续费。
    """
    from .bands import top_n_with_band
    pos = {d: i for i, d in enumerate(close_index)}
    prev = None
    schedule = {}
    for dt in rebal_dates:
        if dt not in pos:
            continue
        sig_idx = pos[dt] - signal_lag_days
        if sig_idx < 0:
            continue
        sig_date = close_index[sig_idx]
        if sig_date not in combined_score.index:
            continue
        w = top_n_with_band(combined_score.loc[sig_date], prev, top_n, buffer)
        if len(w) > 0:
            schedule[dt] = w
            prev = w
    return schedule
