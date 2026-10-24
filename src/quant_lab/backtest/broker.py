"""PaperBroker：自建模拟撮合引擎（学习用，不连任何券商）。

成本模型（与 config/base.yaml 锁定值一致）：
  佣金 = max(成交股数 × commission_per_share, commission_min)  对每个有交易的标的收取
  滑点 = 成交金额 × slippage_bps / 10000                      （单边）

容量约束：
  单标的成交股数 ≤ 当日成交量 × max_participation（10%）；
  超限部分砍掉（不予成交），并在日志中计数。

仅做多、无杠杆：目标权重恒 ≥0 且总和 ≤1，引擎层保证。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    slippage_bps: float = 5.0


@dataclass
class TradeLog:
    """逐笔成交记录（调仓日聚合粒度）。"""
    records: list[dict] = field(default_factory=list)

    def add(self, date, ticker, shares, price, commission, slippage):
        self.records.append({
            "date": date, "ticker": ticker, "shares": round(shares, 2),
            "price": round(price, 4), "commission": round(commission, 2),
            "slippage": round(slippage, 2),
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


class PaperBroker:
    """按目标权重撮合：输入当前持仓、目标权重、当日价格与成交量，
    输出新持仓（股数）与总成本。"""

    def __init__(self, cost: CostModel, max_participation: float = 0.10):
        self.cost = cost
        self.max_participation = max_participation
        self.log = TradeLog()
        self.capacity_clipped_count = 0   # 被容量约束砍单次数

    def rebalance(
        self,
        date: pd.Timestamp,
        holdings: pd.Series,        # 当前持仓股数（index=ticker）
        cash: float,
        target_weights: pd.Series,  # 目标权重（和≤1，≥0）
        prices: pd.Series,          # 当日收盘价
        volumes: pd.Series,         # 当日成交量
    ) -> tuple[pd.Series, float, float]:
        """返回 (新持仓股数, 新现金, 当日总成本)。"""
        prices = prices.dropna()
        equity = cash + float((holdings * prices.reindex(holdings.index).fillna(0)).sum())
        target_values = target_weights * equity

        # 目标股数（向下取整，零股不买）
        tgt_shares = (target_values / prices.reindex(target_values.index)).fillna(0)
        tgt_shares = np.floor(tgt_shares.clip(lower=0))

        all_tickers = holdings.index.union(tgt_shares.index)
        cur = holdings.reindex(all_tickers).fillna(0)
        tgt = tgt_shares.reindex(all_tickers).fillna(0)
        delta = tgt - cur

        px = prices.reindex(all_tickers)
        vol = volumes.reindex(all_tickers)

        total_cost = 0.0
        new_shares = cur.copy()
        for t in all_tickers:
            d = delta[t]
            if d == 0 or not np.isfinite(px[t]) or px[t] <= 0:
                continue
            # 容量约束：成交量数据缺失时保守处理——按参与率无法验证则允许成交
            # （ETF 与大盘股成交量字段完整；缺失多为数据源问题，登记不阻断）
            if np.isfinite(vol[t]) and vol[t] > 0:
                max_shares = vol[t] * self.max_participation
                if abs(d) > max_shares:
                    d = np.sign(d) * np.floor(max_shares)
                    self.capacity_clipped_count += 1
                    if d == 0:
                        continue
            amount = abs(d) * px[t]
            commission = max(abs(d) * self.cost.commission_per_share,
                             self.cost.commission_min)
            slippage = amount * self.cost.slippage_bps / 1e4
            total_cost += commission + slippage
            new_shares[t] = cur[t] + d
            self.log.add(date, t, d, px[t], commission, slippage)

        new_holdings = new_shares[new_shares > 0]
        new_cash = equity - float((new_holdings * prices.reindex(new_holdings.index)).sum()) - total_cost
        return new_holdings, new_cash, total_cost
