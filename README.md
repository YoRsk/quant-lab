# quant-lab

从零搭建的量化策略学习系统：美股多因子选股全链路（因子研究 → 回测 → 过拟合检验），
数据用免费的 yfinance，资金用自建 PaperBroker 虚拟撮合，**仅供学习研究，不连实盘**。

## 五要素基线（config/base.yaml）

| 要素 | 锁定值 |
|---|---|
| 标的池 | 标普500成分抽样 150 只 + SPY/QQQ/IWM/TLT/GLD 基准 ETF |
| 频率 | 周频（每周最后交易日收盘调仓），可切月频 |
| 资金 | $100,000 虚拟美元；单标的参与率 ≤10%；仅做多、无杠杆 |
| 区间 | 2010–2025；样本内 2010–2019 / 样本外 2020–2025 |
| 成本 | 佣金 $0.005/股（最低$1）+ 单边滑点 5bp（IBKR 真实档） |

## 快速开始

```bash
pip install -r requirements.txt

# 1. 跑主策略（首次运行会下载数据并缓存到 data_cache/，约几分钟）
PYTHONPATH=src python strategies/multifactor.py

# 2. 单因子有效性检验（IC/分层/衰减/换手）
PYTHONPATH=src python scripts/run_factor_test.py

# 3. 过拟合三件套（参数敏感性/成本敏感性/蒙特卡洛+Deflated Sharpe）
PYTHONPATH=src python scripts/run_sensitivity.py

# 4. 单元测试
PYTHONPATH=src pytest tests/ -v
```

产出在 `outputs/`：`nav_curve.png`（净值曲线）、`report.md`（绩效与因子报告）、
`param_heatmap.png` / `mc_sharpe_dist.png` / `sensitivity_report.md`（过拟合检验）。

## 目录结构

```
config/base.yaml          # 五要素基线（改参数先改这里）
src/quant_lab/
  data/                   # 标的池（静态快照）+ yfinance 加载器（带缓存与质检）
  factors/                # 动量/低波/反转/趋势（纯价格因子）
  portfolio/              # z-score 信号合成 + Top-N 等权
  backtest/               # PaperBroker 撮合 + 引擎 + 绩效指标
  analysis/               # IC/分层检验 + 过拟合三件套
strategies/multifactor.py # 示例策略：全链路串起来
scripts/                  # 因子检验、敏感性分析入口
tests/                    # 单测（含无未来函数点检）
```

## 已知数据缺口（诚实登记）

1. **生存者偏差**：股票池是"当前"标普500成分快照，回测 2010 年历史时遗漏了
   当年被踢出指数的弱势股 → 收益数字系统性偏乐观，应视为上界。
   严谨版需 CRSP/Compustat/Sharadar 历史成分数据。
2. **基本面因子留桩**：价值/质量因子需要历史财报时序，免费源不可得，
   `factors/price_factors.py` 中留 `NotImplementedError` 桩位。
3. **yfinance 稳定性**：雅虎接口偶发限流/字段变更；缓存机制保证下载一次后可离线复现。

## 无未来函数契约

调仓日 T 的权重只用 ≤ T-1 日数据计算（`signal_lag_days=1`），T 日收盘价成交。
`tests/test_engine.py::test_no_lookahead_in_weights` 是该契约的自动化证据：
篡改未来数据不得改变历史权重。

## 合规声明

投资有风险。本项目仅供研究与教育，不构成投资建议；不荐股、不给买卖点；
回测结果不等于实盘表现，历史业绩不代表未来。
