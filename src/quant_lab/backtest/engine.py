"""向量化回测引擎（周频/月频调仓，收盘价成交）。

无未来函数契约（关键，学习期必须搞懂的一条）：
  - 调仓日 T 的目标权重，由调用方用「截至 T-1 日」的数据计算；
  - 引擎在 T 日收盘价撮合成交（对应实盘中的 MOC 收盘单）；
  - 新持仓从 T→T+1 的隔夜收益开始享受/承担；
  - 引擎内部不访问任何 >T 日的数据决定 T 日的成交。
tests/test_engine.py 里有专门的未来函数点检：篡改未来数据不得改变历史净值。
"""

import numpy as np
import pandas as pd

from .broker import CostModel, PaperBroker


def rebalance_calendar(dates: pd.DatetimeIndex, freq: str = "weekly") -> pd.DatetimeIndex:
    """生成调仓日：每周最后一个交易日（通常周五）或每月最后一个交易日。"""
    s = pd.Series(dates, index=dates)
    if freq == "weekly":
        grp = s.groupby([s.index.isocalendar().year, s.index.isocalendar().week])
    elif freq == "monthly":
        grp = s.groupby([s.index.year, s.index.month])
    else:
        raise ValueError(f"未知调仓频率: {freq}")
    return pd.DatetimeIndex(grp.max().values)


class BacktestEngine:
    """逐日推进的组合回测：持仓随价格漂移，调仓日按目标权重再平衡。"""

    def __init__(self, initial_cash: float, cost: CostModel,
                 max_participation: float = 0.10):
        self.initial_cash = initial_cash
        self.cost = cost
        self.max_participation = max_participation

    def run(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        target_weights: dict[pd.Timestamp, pd.Series],
    ) -> dict:
        """target_weights: {调仓日: 目标权重 Series}。

        返回 {"nav": 日净值序列, "equity": 日总资产, "turnover": 各调仓日单边换手,
              "cost": 各调仓日总成本, "trade_log": DataFrame, "broker": PaperBroker}
        """
        dates = close.index
        broker = PaperBroker(self.cost, self.max_participation)
        holdings = pd.Series(dtype=float)
        cash = self.initial_cash

        equity_curve = []
        turnover_rec, cost_rec = {}, {}
        rebal_dates = set(target_weights.keys())

        prev_prices = None
        for dt in dates:
            px = close.loc[dt]
            if dt in rebal_dates and prev_prices is not None:
                tw = target_weights[dt]
                tw = tw[tw > 0]
                tw = tw / tw.sum() if tw.sum() > 1.0 else tw  # 防御：权重和≤1
                # 漂移后权重（调仓前）
                old_val = float((holdings * px.reindex(holdings.index).fillna(0)).sum())
                old_eq = cash + old_val
                old_w = (holdings * px.reindex(holdings.index).fillna(0) / old_eq) if old_eq > 0 else holdings * 0
                all_t = old_w.index.union(tw.index)
                turnover = 0.5 * float((tw.reindex(all_t).fillna(0)
                                        - old_w.reindex(all_t).fillna(0)).abs().sum())
                holdings, cash, cost = broker.rebalance(
                    dt, holdings, cash, tw, px, volume.loc[dt]
                )
                turnover_rec[dt] = turnover
                cost_rec[dt] = cost
            equity = cash + float((holdings * px.reindex(holdings.index).fillna(0)).sum())
            equity_curve.append((dt, equity))
            prev_prices = px

        eq = pd.Series(dict(equity_curve)).sort_index()
        return {
            "equity": eq,
            "nav": eq / self.initial_cash,
            "turnover": pd.Series(turnover_rec),
            "cost": pd.Series(cost_rec),
            "trade_log": broker.log.to_frame(),
            "broker": broker,
        }
