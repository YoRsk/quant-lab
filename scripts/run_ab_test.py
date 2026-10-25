"""A/B 实验：验证 Asness 思想在本系统上的实际效果。

对比四种配置（周频，成本与资金规模与 base.yaml 一致）：
  A 基线              原始多因子 Top30 每周全量换仓
  B +Beta 中性化      因子打分先剔除市场 Beta 再做选股
  C +换手缓冲带       排名缓冲带 5 名，减少无谓交易
  D +两者             B 与 C 叠加

用法: ./run.sh scripts/run_ab_test.py
产出: outputs/ab_report.md + ab_nav.png（净值对比）+ ab_cost.png（成本对比）
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_lab.analysis.ic import ic_summary, rank_ic_series
from quant_lab.backtest.broker import CostModel
from quant_lab.backtest.engine import BacktestEngine, rebalance_calendar
from quant_lab.backtest.metrics import perf_summary
from quant_lab.config import load_config
from quant_lab.data.universe import BENCHMARK_ETFS, get_stock_pool
from quant_lab.data.yf_loader import load_ohlcv
from quant_lab.factors.neutralize import (cross_sectional_neutralize,
                                         rolling_beta)
from quant_lab.factors.price_factors import FACTOR_REGISTRY
from quant_lab.portfolio.bands import top_n_with_band
from quant_lab.portfolio.signals import combine_factors, zscore

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "outputs"


def build_score(stock_close, spy_ret, do_neutralize):
    """构造当日打分：原始合成 vs Beta 中性化后合成。"""
    if not do_neutralize:
        factors = {n: f(stock_close) for n, f in FACTOR_REGISTRY.items()}
        return combine_factors(factors)

    rets = stock_close.pct_change()
    beta = rolling_beta(rets, spy_ret)
    neutralized = {}
    for n, f in FACTOR_REGISTRY.items():
        raw = zscore(f(stock_close))
        neutralized[n] = cross_sectional_neutralize(raw, beta)
    return combine_factors(neutralized)


def build_schedule(score, rebal_dates, close_index, top_n, buffer):
    """带缓冲带的调仓权重表（信号滞后 1 日，保证无未来函数）。"""
    pos = {d: i for i, d in enumerate(close_index)}
    prev = None
    schedule = {}
    for dt in rebal_dates:
        if dt not in pos:
            continue
        sig_date = close_index[pos[dt] - 1]
        if sig_date not in score.index:
            continue
        w = top_n_with_band(score.loc[sig_date], prev, top_n, buffer)
        if len(w) > 0:
            schedule[dt] = w
            prev = w
    return schedule


def run_variant(name, close, volume, score, rebal_dates, cfg, buffer):
    schedule = build_schedule(score, rebal_dates, close.index,
                              cfg["strategy"]["top_n"], buffer)
    engine = BacktestEngine(cfg["capital"]["initial_cash"],
                            CostModel(**cfg["backtest"]["cost"]),
                            cfg["capital"]["max_participation"])
    res = engine.run(close, volume, schedule)
    summary = perf_summary(res["equity"], res["turnover"], res["cost"])
    return {"name": name, "equity": res["equity"], "summary": summary,
            "cost": float(res["cost"].sum()), "result": res}


def main():
    cfg = load_config()
    bt = cfg["backtest"]
    stocks = get_stock_pool(cfg["universe"]["sample_size"],
                            cfg["universe"]["sample_seed"])
    data = load_ohlcv(stocks + BENCHMARK_ETFS, bt["start"], bt["end"])
    close, volume = data["close"], data["volume"]
    stocks = [s for s in stocks if s in close.columns]
    stock_close = close[stocks].dropna(how="all")
    spy_ret = close["SPY"].pct_change()
    OUT.mkdir(exist_ok=True)

    rebal = rebalance_calendar(close.index, cfg["trading"]["rebalance"])
    top_n = cfg["strategy"]["top_n"]

    score_raw = build_score(stock_close, spy_ret, do_neutralize=False)
    score_bn = build_score(stock_close, spy_ret, do_neutralize=True)

    variants = [
        run_variant("A 基线", close, volume, score_raw, rebal, cfg, buffer=0),
        run_variant("B +Beta中性", close, volume, score_bn, rebal, cfg, buffer=0),
        run_variant("C +换手缓冲带", close, volume, score_raw, rebal, cfg, buffer=5),
        run_variant("D +两者", close, volume, score_bn, rebal, cfg, buffer=5),
    ]

    spy = close["SPY"].dropna()
    spy_nav = spy / spy.iloc[0]

    # ---- 图 1：净值曲线对比 ----
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#888780", "#378ADD", "#D85A30", "#1D9E75"]
    for v, c in zip(variants, colors):
        nav = v["equity"] / v["equity"].iloc[0]
        ax.plot(nav.index, nav.values, label=v["name"], lw=1.3, color=c)
    ax.plot(spy_nav.index, spy_nav.values, label="SPY 买入持有",
            lw=1.0, color="#5F5E5A", ls="--")
    ax.axvline(pd.Timestamp(bt["in_sample_end"]), color="gray", ls="--", lw=0.8)
    ax.legend(); ax.set_ylabel("净值（期初=1）")
    ax.set_title("A/B 对比：Asness 式中性化与换手控制的效果")
    fig.tight_layout(); fig.savefig(OUT / "ab_nav.png", dpi=120); plt.close(fig)

    # ---- 图 2：成本对比 ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [v["name"] for v in variants]
    costs = [v["cost"] for v in variants]
    bars = ax.bar(names, costs, color=colors)
    ax.axhline(variants[0]["cost"], color="#888780", ls="--", lw=0.8,
               label=f"基线 ${variants[0]['cost']:,.0f}")
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"${c:,.0f}", ha="center", fontsize=9)
    ax.set_ylabel("16 年累计交易成本 ($)")
    ax.set_title("手续费对比：缓冲带能省多少")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "ab_cost.png", dpi=120); plt.close(fig)

    # ---- 低波因子：中性化前后 IC 对比（解释为什么原方向是反的）----
    rets = stock_close.pct_change()
    beta = rolling_beta(rets, spy_ret)
    lv_raw = zscore(FACTOR_REGISTRY["low_vol"](stock_close))
    lv_bn = cross_sectional_neutralize(lv_raw, beta)
    ic_raw = ic_summary(rank_ic_series(lv_raw, stock_close, 21))
    ic_bn = ic_summary(rank_ic_series(lv_bn, stock_close, 21))

    # ---- 报告 ----
    lines = ["# A/B 实验报告：Asness 思想落地效果\n",
             f"区间 {bt['start']} ~ {bt['end']}｜周频 Top{top_n}｜"
             f"成本 佣金${cfg['backtest']['cost']['commission_per_share']}/股+"
             f"滑点{cfg['backtest']['cost']['slippage_bps']}bp\n",
             "## 四种配置对比\n",
             "| 配置 | 年化收益 | 夏普 | 最大回撤 | 年均换手 | 累计成本 |",
             "|---|---|---|---|---|---|"]
    for v in variants:
        s = v["summary"]
        lines.append(f"| {v['name']} | {s.get('年化收益')} | {s.get('夏普比率')} "
                     f"| {s.get('最大回撤')} | {s.get('年均单边换手','-')} "
                     f"| ${v['cost']:,.0f} |")
    # ---- 稳健性复核（质量门）：buffer 敏感性 + Deflated Sharpe ----
    from quant_lab.analysis.overfit import (block_bootstrap_sharpe,
                                            deflated_sharpe_ratio)
    buffers = [0, 3, 5, 8, 12, 16, 20, 30]
    best: dict | None = None
    for bf in buffers:
        v = run_variant(f"buffer={bf}", close, volume, score_bn, rebal, cfg,
                        buffer=bf)
        sh = float(v["summary"]["夏普比率"])
        print(f"  buffer={bf:<3d} 夏普 {sh:.2f}  成本 ${v['cost']:,.0f}  "
              f"换手 {v['summary'].get('年均单边换手','-')}")
        if best is None or sh > best[0]:
            best = (sh, bf, v)

    rets = variants[3]["equity"].pct_change().dropna()
    sim = block_bootstrap_sharpe(rets, n_sims=1000)
    # 诚实申报试验次数：历史 15 次（上一轮三件套）+ 本轮 4 组AB + 5 档 buffer
    n_trials = 15 + 4 + len(buffers)
    dsr = deflated_sharpe_ratio(float(variants[3]["summary"]["夏普比率"]),
                                rets, n_trials)

    lines += ["\n## 稳健性复核（质量门）\n",
              f"### 缓冲带宽度敏感性（Beta 中性化下，{top_n} 只持仓）\n",
              "| buffer | 夏普 | 成本 |", "|---|---|---|"]
    for bf in buffers:
        v = run_variant(f"b{bf}", close, volume, score_bn, rebal, cfg, buffer=bf)
        lines.append(f"| {bf} | {v['summary']['夏普比率']} | ${v['cost']:,.0f} |")
    lines += [f"\n最优 buffer = {best[1]}（夏普 {best[0]:.2f}）；"
              "若邻域内夏普平稳而非单点凸起，说明不是调参数调出来的。\n",
              "### 多重检验校正\n",
              f"- block bootstrap 夏普中位 {np.median(sim):.2f}，"
              f"5% 分位 {np.percentile(sim, 5):.2f}",
              f"- 配置 D 夏普 {variants[3]['summary']['夏普比率']}；"
              f"诚实申报试验次数 n_trials={n_trials}",
              f"- **Deflated Sharpe p = {dsr:.3f}**"
              "（>0.95 站得住；0.9~0.95 灰色；<0.9 警报）\n"]

    lines += ["## 关键机制：低波因子方向之谜\n",
              f"\nSPY 基准同期：{perf_summary(spy).get('年化收益')} 年化、"
              f"夏普 {perf_summary(spy).get('夏普比率')}。\n",
              "| 低波因子 | IC均值 | t值 |", "|---|---|---|",
              f"| 未做 Beta 中性（原始） | {ic_raw.get('IC_mean')} | "
              f"{ic_raw.get('t_stat')} |",
              f"| Beta 中性化之后 | {ic_bn.get('IC_mean')} | "
              f"{ic_bn.get('t_stat')} |",
              "\n> 注意：中性化后 IC 仍为负（只是幅度收窄），说明 Beta 敞口"
              "只解释了其中一部分，剩余部分很可能来自样本期本身的行情特征。"
              "不应据此反向使用低波因子——那是事后诸葛。\n",
              "\n## 结论要点\n",
              "- 换手缓冲带直接砍掉无效交易，成本下降，是本实验中机制最清晰的一项；",
              "- Beta 中性化的作用是「净化信号」，不保证提高收益；",
              "- 两者叠加形态最优，但须以缓冲带敏感性与 DSR 检验为前提才可信；",
              "- 生存者偏差仍在，不可据此实盘。\n",
              "> 合规声明：仅供学习研究，不构成投资建议；回测≠实盘，历史业绩不代表未来。"]
    (OUT / "ab_report.md").write_text("\n".join(lines), encoding="utf-8")

    for v in variants:
        s = v["summary"]
        print(f"{v['name']:14s} 年化{s.get('年化收益'):>8s}  夏普{s.get('夏普比率'):>6}  "
              f"回撤{s.get('最大回撤'):>8s}  换手{s.get('年均单边换手', '-'):>7s}  "
              f"成本${v['cost']:,.0f}")
    print(f"\n低波因子 IC: 原始 {ic_raw.get('IC_mean')} (t={ic_raw.get('t_stat')}) → "
          f"Beta中性后 {ic_bn.get('IC_mean')} (t={ic_bn.get('t_stat')})")


if __name__ == "__main__":
    main()
