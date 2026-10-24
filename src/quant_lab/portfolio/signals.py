"""信号合成：因子值 → 横截面 z-score → 加权合成总分。

z-score（标准化）的含义：把每只股票某天在某因子上的取值，
减去当天全部股票的均值、除以标准差，得到"比平均水平高/低几个标准差"。
作用：不同因子量纲不同（动量是收益率、低波是负波动率），
标准化后才能放在同一杆秤上加权。
"""

import pandas as pd


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """逐日横截面标准化。保留 NaN（数据不足的标的不参与）。"""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, pd.NA)
    return df.sub(mean, axis=0).div(std, axis=0)


def combine_factors(factors: dict[str, pd.DataFrame],
                    weights: dict[str, float] | None = None) -> pd.DataFrame:
    """多因子 z-score 加权合成。weights=None 时等权。"""
    zs = {name: zscore(f) for name, f in factors.items()}
    if weights is None:
        weights = {name: 1.0 / len(zs) for name in zs}
    total_w = sum(weights.values())
    combined = sum(zs[name] * (w / total_w) for name, w in weights.items())
    return combined
