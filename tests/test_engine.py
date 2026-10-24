"""回测引擎无未来函数点检 + 基本正确性。"""

import numpy as np
import pandas as pd
import pytest

from quant_lab.backtest.broker import CostModel
from quant_lab.backtest.engine import BacktestEngine, rebalance_calendar
from quant_lab.portfolio.weights import build_weight_schedule


def _toy_data(n_days=300, n_stocks=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rets = rng.normal(0.0005, 0.01, size=(n_days, n_stocks))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)),
                         index=dates, columns=[f"S{i}" for i in range(n_stocks)])
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    return close, volume


def test_no_lookahead_in_weights():
    """篡改未来因子值，不得改变历史调仓日的权重（无未来函数硬证据）。"""
    close, _ = _toy_data()
    score = close.pct_change(20)   # 玩具信号
    rebal = rebalance_calendar(close.index, "weekly")
    sch1 = build_weight_schedule(score, rebal, 1, close.index, top_n=3)

    score_tampered = score.copy()
    score_tampered.iloc[-30:] = 999.0   # 篡改最后30天
    sch2 = build_weight_schedule(score_tampered, rebal, 1, close.index, top_n=3)

    cutoff = close.index[-40]
    for dt, w in sch1.items():
        if dt <= cutoff:
            pd.testing.assert_series_equal(w, sch2[dt])


def test_engine_costs_reduce_nav():
    """有成本的净值必须低于无成本的净值（成本确实被扣除）。"""
    close, volume = _toy_data()
    score = close.pct_change(20)
    rebal = rebalance_calendar(close.index, "weekly")
    schedule = build_weight_schedule(score, rebal, 1, close.index, top_n=3)

    free = BacktestEngine(100_000, CostModel(0, 0, 0)).run(close, volume, schedule)
    costly = BacktestEngine(100_000, CostModel(0.005, 1.0, 5.0)).run(close, volume, schedule)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]
    assert costly["cost"].sum() > 0


def test_nav_starts_at_one():
    close, volume = _toy_data()
    score = close.pct_change(20)
    rebal = rebalance_calendar(close.index, "weekly")
    schedule = build_weight_schedule(score, rebal, 1, close.index, top_n=3)
    res = BacktestEngine(100_000, CostModel()).run(close, volume, schedule)
    assert res["nav"].iloc[0] == pytest.approx(1.0)
