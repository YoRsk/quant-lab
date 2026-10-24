"""PaperBroker 成本计算与容量约束单元测试。"""

import pandas as pd
import pytest

from quant_lab.backtest.broker import CostModel, PaperBroker


@pytest.fixture
def broker():
    return PaperBroker(CostModel(commission_per_share=0.005,
                                 commission_min=1.0, slippage_bps=5.0),
                       max_participation=0.10)


def test_commission_min_applies(broker):
    """小额交易触发最低佣金 $1。"""
    holdings = pd.Series(dtype=float)
    target = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 100.0})
    volumes = pd.Series({"AAA": 1_000_000.0})
    h, cash, cost = broker.rebalance(pd.Timestamp("2024-01-05"),
                                     holdings, 10_000.0, target, prices, volumes)
    # 买 100 股：佣金 = max(100*0.005, 1) = $1；滑点 = 10000*5/10000 = $5
    assert h["AAA"] == 100
    assert cost == pytest.approx(1.0 + 5.0, abs=1e-6)
    assert cash == pytest.approx(10_000 - 100 * 100 - cost, abs=1e-6)


def test_commission_per_share_above_min(broker):
    """大额交易按每股 $0.005 计。"""
    holdings = pd.Series(dtype=float)
    target = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 10.0})
    volumes = pd.Series({"AAA": 10_000_000.0})
    h, cash, cost = broker.rebalance(pd.Timestamp("2024-01-05"),
                                     holdings, 100_000.0, target, prices, volumes)
    # 买 10000 股：佣金 = 10000*0.005 = $50；滑点 = 100000*5/10000 = $50
    assert h["AAA"] == 10_000
    assert cost == pytest.approx(50.0 + 50.0, abs=1e-6)


def test_capacity_clip(broker):
    """成交股数不得超过当日成交量 × 参与率。"""
    holdings = pd.Series(dtype=float)
    target = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 100.0})
    volumes = pd.Series({"AAA": 1_000.0})   # 最多买 1000*10% = 100 股
    h, cash, cost = broker.rebalance(pd.Timestamp("2024-01-05"),
                                     holdings, 100_000.0, target, prices, volumes)
    assert h["AAA"] == 100
    assert broker.capacity_clipped_count == 1


def test_no_negative_positions(broker):
    """仅做多：持仓恒非负。"""
    holdings = pd.Series({"AAA": 50.0})
    target = pd.Series(dtype=float)  # 清仓信号
    prices = pd.Series({"AAA": 100.0})
    volumes = pd.Series({"AAA": 1_000_000.0})
    h, cash, cost = broker.rebalance(pd.Timestamp("2024-01-05"),
                                     holdings, 0.0, target, prices, volumes)
    assert (h >= 0).all()
    assert len(h) == 0  # 清仓后无持仓
