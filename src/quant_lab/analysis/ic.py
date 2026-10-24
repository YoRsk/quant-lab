"""因子有效性检验：Rank IC / ICIR / 分层回测 / 换手与衰减。

口径约定：
- IC = 因子值与未来 horizon 日收益的横截面 Spearman 秩相关（Rank IC）。
  用秩相关而非线性相关，对异常值稳健，是业界主流口径。
- ICIR = mean(IC) / std(IC)，衡量因子稳定性。
- 分层：每个调仓日按因子值把股票分 n_groups 层，看各层未来收益是否单调。
- 所有计算严格只用 ≤T 日的因子值与 >T 日的未来收益配对，无未来函数。
"""

import numpy as np
import pandas as pd


def rank_ic_series(factor: pd.DataFrame, close: pd.DataFrame,
                   horizon: int = 21) -> pd.Series:
    """逐日 Rank IC 序列：T 日因子值 vs T→T+horizon 收益的横截面秩相关。"""
    fwd = close.shift(-horizon) / close - 1.0   # 未来收益（仅用于评价，不用于交易）
    common_dates = factor.index.intersection(fwd.index)
    ics = {}
    for dt in common_dates:
        f = factor.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < 20:          # 截面样本太少则跳过
            continue
        ics[dt] = f[mask].corr(r[mask], method="spearman")
    return pd.Series(ics).dropna()


def ic_summary(ic: pd.Series) -> dict:
    """IC 汇总：均值、标准差、ICIR、IC>0 占比、t 统计量。"""
    n = len(ic)
    if n < 5:
        return {"n": n, "note": "样本不足"}
    mean, std = ic.mean(), ic.std()
    return {
        "n": n,
        "IC_mean": round(mean, 4),
        "IC_std": round(std, 4),
        "ICIR": round(mean / std, 3) if std > 0 else np.nan,
        "IC_positive_ratio": round((ic > 0).mean(), 3),
        "t_stat": round(mean / (std / np.sqrt(n)), 2) if std > 0 else np.nan,
    }


def quantile_returns(factor: pd.DataFrame, close: pd.DataFrame,
                     horizon: int = 21, n_groups: int = 5,
                     freq: int = 21) -> pd.DataFrame:
    """分层回测：每 freq 日按因子分 n_groups 层，统计各层未来 horizon 日均收益。

    返回各层平均年化收益（简单年化），用于肉眼检查单调性。
    """
    fwd = close.shift(-horizon) / close - 1.0
    dates = factor.index[::freq]
    layer_rets = {g: [] for g in range(n_groups)}
    for dt in dates:
        if dt not in fwd.index:
            continue
        f = factor.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < n_groups * 10:
            continue
        ranked = f[mask].rank(pct=True)
        groups = np.ceil(ranked * n_groups).clip(1, n_groups).astype(int) - 1
        for g in range(n_groups):
            sel = groups == g
            if sel.sum() > 0:
                layer_rets[g].append(r[mask][sel].mean())
    ann = {}
    for g, rets in layer_rets.items():
        if rets:
            ann[f"Q{g+1}"] = round(np.mean(rets) * (252 / horizon), 4)
    out = pd.DataFrame(ann, index=["年化收益(分层均值)"]).T
    out.index.name = "分层(因子值升序)"
    return out


def factor_turnover(factor: pd.DataFrame, top_n: int = 30,
                    freq: int = 5) -> float:
    """Top-N 组合的平均换手率：每 freq 日取因子前 top_n，看成员变动比例。"""
    dates = factor.index[::freq]
    prev = None
    turns = []
    for dt in dates:
        f = factor.loc[dt].dropna()
        if len(f) < top_n:
            continue
        cur = set(f.nlargest(top_n).index)
        if prev is not None:
            turns.append(1 - len(cur & prev) / top_n)
        prev = cur
    return round(float(np.mean(turns)), 3) if turns else np.nan


def ic_decay(factor: pd.DataFrame, close: pd.DataFrame,
             horizons: tuple = (5, 10, 21, 42, 63)) -> pd.DataFrame:
    """IC 衰减：不同预测周期下的 IC 均值，看因子信息能管多久。"""
    rows = {}
    for h in horizons:
        s = ic_summary(rank_ic_series(factor, close, horizon=h))
        rows[f"{h}日"] = s.get("IC_mean", np.nan)
    return pd.DataFrame(rows, index=["IC_mean"]).T
