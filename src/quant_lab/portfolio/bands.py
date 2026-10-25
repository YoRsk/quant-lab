"""调仓缓冲带（No-Trade Band）：只在"真的有必要"时才换仓。

问题起点：本项目第一版策略年换手 1037%（一年把组合换 10 遍），
16 年交了 $90,200 手续费，把大部分超额收益磨光。

原理：多因子打分每天都在微动，但排名第 28 名变成第 32 名这种抖动
通常只是噪声，不值得为此付一次佣金+滑点。于是给排名加一圈"缓冲区"：
  - 已持有的股票，只要没跌出 Top(N + buffer)，就继续持有（哪怕它已不在 Top N）；
  - 新股票要挤进 Top(N - buffer) 才够资格买入。

buffer 越大越省成本，但组合会偏离理想目标；buffer=0 就是原来的每周全量换仓。
这对相邻两次调仓之间的抖动尤其有效——Asness 反复强调的一点：
因子收益是毛估的，交易成本是确定的。
"""

import pandas as pd


def top_n_with_band(score: pd.Series, prev_weights: pd.Series | None,
                    top_n: int = 30, buffer: int = 5) -> pd.Series:
    """带缓冲带的 Top-N 等权选股。

    score        : 当日合成打分
    prev_weights : 上一期权重（第一次调仓时为 None）
    top_n        : 目标持仓数
    buffer       : 排名缓冲区宽度（单位：名次）
    """
    s = score.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    n = min(top_n, len(s))
    ranked = s.rank(ascending=False, method="first")   # 1 = 分数最高

    if prev_weights is None or len(prev_weights) == 0:
        selected = ranked[ranked <= n].index.tolist()
    else:
        # 存量持仓：只要还在 N+buffer 名以内就保留
        keep = [t for t in prev_weights.index
                if t in ranked.index and ranked[t] <= n + buffer]
        # 新人：必须挤进 N-buffer 名才有资格
        fresh = [t for t in ranked[ranked <= max(n - buffer, 1)].index
                 if t not in keep]
        # 按分数从高到低补齐到 n 只
        need = n - len(keep)
        selected = keep + fresh[:max(need, 0)]
        # 若 keep 已超额（极端情况），按分数保序截断
        if len(selected) > n:
            ordered = ranked[selected].sort_values().index.tolist()
            selected = ordered[:n]

    if len(selected) == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(selected), index=selected)
