"""示例策略：多因子选股全链路（动量+低波+反转+趋势 → Top-N 等权）。

用法：
    PYTHONPATH=src python strategies/multifactor.py [--config config/base.yaml]

产出：outputs/ 下的净值图、指标表、因子检验报告。
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_lab.analysis.ic import (ic_decay, ic_summary, quantile_returns,
                                   rank_ic_series, factor_turnover)
from quant_lab.backtest.broker import CostModel
from quant_lab.backtest.engine import BacktestEngine, rebalance_calendar
from quant_lab.backtest.metrics import perf_summary
from quant_lab.config import load_config
from quant_lab.data.universe import BENCHMARK_ETFS, get_stock_pool
from quant_lab.data.yf_loader import load_ohlcv
from quant_lab.factors.neutralize import (cross_sectional_neutralize,
                                         rolling_beta)
from quant_lab.factors.price_factors import FACTOR_REGISTRY
from quant_lab.portfolio.signals import combine_factors, zscore
from quant_lab.portfolio.weights import build_weight_schedule

OUT = Path(__file__).resolve().parents[1] / "outputs"


def run(cfg: dict, out_dir: Path = OUT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bt, cap, cost_cfg = cfg["backtest"], cfg["capital"], cfg["backtest"]["cost"]

    # ---- 数据 ----
    stocks = get_stock_pool(cfg["universe"]["sample_size"], cfg["universe"]["sample_seed"])
    tickers = stocks + BENCHMARK_ETFS
    data = load_ohlcv(tickers, bt["start"], bt["end"])
    close, volume = data["close"], data["volume"]
    stocks = [s for s in stocks if s in close.columns]  # 剔除下载失败的
    stock_close = close[stocks].dropna(how="all")

    # ---- 因子（可选：先做 Beta 中性化，对应 Asness 的低风险因子思想）----
    factor_names = cfg["strategy"]["factors"]
    raw_factors = {n: FACTOR_REGISTRY[n](stock_close) for n in factor_names}
    if cfg["strategy"].get("beta_neutralize", False):
        beta = rolling_beta(stock_close.pct_change(), close["SPY"].pct_change())
        factors = {n: cross_sectional_neutralize(zscore(f), beta)
                   for n, f in raw_factors.items()}
    else:
        factors = raw_factors
    combined = combine_factors(factors)

    # ---- 因子有效性检验（输出到报告）----
    factor_report = {}
    for n, f in factors.items():
        ic = rank_ic_series(f, stock_close, horizon=21)
        factor_report[n] = {
            "IC汇总": ic_summary(ic),
            "衰减": ic_decay(f, stock_close),
            "分层": quantile_returns(f, stock_close),
            "Top30换手(周)": factor_turnover(f, top_n=cfg["strategy"]["top_n"]),
        }

    # ---- 回测 ----
    rebal_dates = rebalance_calendar(close.index, cfg["trading"]["rebalance"])
    schedule = build_weight_schedule(
        combined, rebal_dates, signal_lag_days=1,
        close_index=close.index, top_n=cfg["strategy"]["top_n"],
        buffer=int(cfg["strategy"].get("no_trade_band", 0)),
    )
    engine = BacktestEngine(
        initial_cash=cap["initial_cash"],
        cost=CostModel(**cost_cfg),
        max_participation=cap["max_participation"],
    )
    result = engine.run(close, volume, schedule)

    # ---- 基准对照（SPY 买入持有）----
    spy = close["SPY"].dropna()
    spy_nav = spy / spy.iloc[0]

    # ---- 样本内/外分段指标 ----
    is_end = pd.Timestamp(bt["in_sample_end"])
    eq = result["equity"]
    summary_all = perf_summary(eq, result["turnover"], result["cost"])
    summary_is = perf_summary(eq[eq.index <= is_end])
    summary_oos = perf_summary(eq[eq.index > is_end])
    summary_spy = perf_summary(spy)

    # ---- 图：净值曲线（含样本内外分界）----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(result["nav"].index, result["nav"].values, label="多因子策略", lw=1.4)
    ax.plot(spy_nav.index, spy_nav.values, label="SPY 买入持有", lw=1.0, alpha=0.8)
    ax.axvline(is_end, color="gray", ls="--", lw=0.8)
    ax.text(is_end, ax.get_ylim()[1] * 0.95, " 样本内 | 样本外", fontsize=9, color="gray")
    ax.set_title(f"quant-lab 多因子策略净值 | {bt['start']}~{bt['end']} | "
                 f"成本: 佣金${cost_cfg['commission_per_share']}/股+滑点{cost_cfg['slippage_bps']}bp")
    ax.legend()
    ax.set_ylabel("净值（期初=1）")
    fig.tight_layout()
    fig.savefig(out_dir / "nav_curve.png", dpi=120)
    plt.close(fig)

    # ---- 报告（markdown）----
    lines = ["# 多因子策略回测报告\n",
             f"- 区间: {bt['start']} ~ {bt['end']}（样本内至 {bt['in_sample_end']}）",
             f"- 股票池: {len(stocks)} 只（标普500抽样，**存在生存者偏差，收益偏乐观**）",
             f"- 调仓: {cfg['trading']['rebalance']}，持仓 {cfg['strategy']['top_n']} 只等权",
             f"- 成本: 佣金 ${cost_cfg['commission_per_share']}/股(最低${cost_cfg['commission_min']}) "
             f"+ 滑点 {cost_cfg['slippage_bps']}bp 单边",
             f"- 容量砍单次数: {result['broker'].capacity_clipped_count}\n",
             "\n## 绩效指标\n",
             "| 指标 | 全区间 | 样本内 | 样本外 | SPY基准 |", "|---|---|---|---|---|"]
    keys = ["年化收益", "年化波动", "夏普比率", "最大回撤", "Calmar"]
    for k in keys:
        lines.append(f"| {k} | {summary_all.get(k, '-')} | {summary_is.get(k, '-')} "
                     f"| {summary_oos.get(k, '-')} | {summary_spy.get(k, '-')} |")
    lines += ["", f"- 年均单边换手: {summary_all.get('年均单边换手', '-')}",
              f"- 累计成本: {summary_all.get('累计交易成本', '-')} "
              f"（占初始资金 {summary_all.get('成本占初始资金', '-')}）",
              "\n## 因子有效性（样本全区间，21 日前瞻 Rank IC）\n",
              "| 因子 | IC均值 | ICIR | IC>0占比 | t值 | Top30周换手 |", "|---|---|---|---|---|---|"]
    for n, rep in factor_report.items():
        s = rep["IC汇总"]
        lines.append(f"| {n} | {s.get('IC_mean', '-')} | {s.get('ICIR', '-')} "
                     f"| {s.get('IC_positive_ratio', '-')} | {s.get('t_stat', '-')} "
                     f"| {rep['Top30换手(周)']} |")
    lines.append("\n> 合规声明：本报告仅供学习研究，不构成投资建议；"
                 "回测≠实盘，历史业绩不代表未来。")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    return {"result": result, "summary_all": summary_all,
            "summary_is": summary_is, "summary_oos": summary_oos,
            "factor_report": factor_report, "close": close, "volume": volume,
            "stock_close": stock_close}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = run(cfg)
    print("=== 全区间 ===")
    for k, v in out["summary_all"].items():
        print(f"{k}: {v}")
    print("\n报告与图表已输出到 outputs/")
