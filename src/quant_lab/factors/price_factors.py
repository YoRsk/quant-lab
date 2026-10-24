"""纯价格因子库。

第一阶段只用价格/成交量可构造的因子，保证数据完全免费且可复现。

【数据缺口登记 · 基本面因子】
价值（PE/PB/EV-EBITDA）、质量（ROE/毛利率）因子需要历史财报时序，
yfinance 只提供当前快照，免费源无可靠历史 → 本阶段留桩不实现。
严谨版本需 Sharadar / Polygon / Compustat 等付费数据。

所有因子函数约定：
  输入  close: DataFrame(index=日期, columns=标的) 复权收盘价
  输出  DataFrame(index=日期, columns=标的) 因子原始值（越大越好方向已统一）
  因子在 T 日的值只用 ≤ T 日的数据计算；由调用方保证在 T+1 日才用于交易。
"""

import numpy as np
import pandas as pd


def momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """动量因子：过去 lookback 日收益，跳过最近 skip 日（规避短期反转）。

    经典定义（Jegadeesh & Titman 1993）：MOM = P[t-skip] / P[t-lookback] - 1
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def low_volatility(close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """低波动因子：过去 window 日日收益波动率的相反数（波动越低分越高）。"""
    ret = close.pct_change()
    return -ret.rolling(window).std()


def reversal(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """短期反转因子：过去 window 日收益的相反数（近期跌得多的分高）。"""
    return -(close / close.shift(window) - 1.0)


def trend(close: pd.DataFrame, ma_window: int = 200) -> pd.DataFrame:
    """趋势因子：现价相对 ma_window 日均线的偏离（站上均线为正）。"""
    ma = close.rolling(ma_window).mean()
    return close / ma - 1.0


# 留桩：基本面因子（需要付费数据源，详见模块 docstring）
def value_factor(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "价值因子需要历史财报数据（免费源不可得），本阶段留桩。"
        "接 Sharadar/Polygon 后在此实现。"
    )


def quality_factor(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "质量因子需要历史财报数据（免费源不可得），本阶段留桩。"
    )


FACTOR_REGISTRY = {
    "momentum": momentum,
    "low_vol": low_volatility,
    "reversal": reversal,
    "trend": trend,
}
