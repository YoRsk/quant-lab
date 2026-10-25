"""中性化工具：把原始因子打分里的"风格敞口"剥掉。

为什么要做这件事（对应 Asness 的低风险因子思想）：
你以为在买"低波动股票"，但如果没剔除市场 Beta，实际上是在押"低 Beta 股票"。
2010–2025 这轮美股长牛里高 Beta 股票涨得最好，于是因子方向看起来是反的
（本项目第一版回测：low_vol 的 IC 达 -8.73 个 t 值，方向明显背离）。

Beta 中性化 = 在每个交易日横截面上，把因子打分对个股 Beta 做一元回归，
只保留"回归残差"部分（也就是剔除掉 Beta 能解释的那部分）后再标准化。
剩下的才是"同样是某个 Beta 水平下，哪只股票更便宜/更优质/更有动量"。

注意：这是降波动的工具，不是白送收益——行业轮动 Alpha 会被一并削掉。
"""

import pandas as pd


def rolling_beta(stock_returns: pd.DataFrame, market_returns: pd.Series,
                 window: int = 252, min_periods: int = 60) -> pd.DataFrame:
    """用滚动窗口估计每只股票对市场指数的 Beta。

    Beta 的含义：市场涨跌 1% 时这只股票平均涨跌百分之多少。
    Beta=1.3 说明它是"放大器"，市场涨 1% 它涨 1.3%；Beta=0.6 则是"减震器"。
    """
    mkt = market_returns.reindex(stock_returns.index)
    var_m = mkt.rolling(window, min_periods=min_periods).var()
    cov = (stock_returns.multiply(mkt, axis=0)
           .rolling(window, min_periods=min_periods).mean()
           - stock_returns.rolling(window, min_periods=min_periods).mean()
           .multiply(mkt.rolling(window, min_periods=min_periods).mean(), axis=0))
    return cov.divide(var_m, axis=0)


def cross_sectional_neutralize(score: pd.DataFrame, exposure: pd.DataFrame,
                               method: str = "ols") -> pd.DataFrame:
    """逐日横截面回归取残差，再标准化为 z-score。

    score    : 因子原始打分（日期 × 标的）
    exposure : 要剔除的敞口（通常是 Beta；也可以传市值/行业哑变量）

    返回：中性化后的打分（同样 shape）
    """
    x = exposure.sub(exposure.mean(axis=1), axis=0)   # 每日去均值
    y = score.sub(score.mean(axis=1), axis=0)
    denom = (x * x).sum(axis=1, min_count=1)
    b = (x * y).sum(axis=1, min_count=1).divide(denom, axis=0).replace(
        [float("inf"), -float("inf")], pd.NA)
    resid = y - x.mul(b, axis=0)
    # 再标准化，使每日横向可比
    sd = resid.std(axis=1).replace(0, pd.NA)
    return resid.div(sd, axis=0)
