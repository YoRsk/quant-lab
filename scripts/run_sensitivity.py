"""过拟合三件套运行脚本：参数敏感性 × 成本敏感性 × 蒙特卡洛。

用法： PYTHONPATH=src python scripts/run_sensitivity.py
产出： outputs/sensitivity_report.md + 参数热力图
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_lab.analysis.overfit import (block_bootstrap_sharpe,
                                        deflated_sharpe_ratio,
                                        sensitivity_table)
from quant_lab.backtest.broker import CostModel
from quant_lab.backtest.engine import BacktestEngine, rebalance_calendar
from quant_lab.backtest.metrics import perf_summary
from quant_lab.config import load_config
from quant_lab.data.universe import BENCHMARK_ETFS, get_stock_pool
from quant_lab.data.yf_loader import load_ohlcv
from quant_lab.factors.price_factors import FACTOR_REGISTRY, momentum
from quant_lab.portfolio.signals import combine_factors
from quant_lab.portfolio.weights import build_weight_schedule

OUT = Path(__file__).resolve().parents[1] / "outputs"


def _run_one(close, volume, stock_close, cfg, mom_lookback, top_n,
             slippage_bps) -> dict:
    """单组参数的一次完整回测，返回核心指标。"""
    factors = {
        "momentum": momentum(stock_close, lookback=mom_lookback),
        "low_vol": FACTOR_REGISTRY["low_vol"](stock_close),
        "reversal": FACTOR_REGISTRY["reversal"](stock_close),
        "trend": FACTOR_REGISTRY["trend"](stock_close),
    }
    combined = combine_factors(factors)
    rebal = rebalance_calendar(close.index, cfg["trading"]["rebalance"])
    schedule = build_weight_schedule(combined, rebal, 1, close.index, top_n)
    cost_cfg = dict(cfg["backtest"]["cost"])
    cost_cfg["slippage_bps"] = slippage_bps
    engine = BacktestEngine(cfg["capital"]["initial_cash"], CostModel(**cost_cfg),
                            cfg["capital"]["max_participation"])
    res = engine.run(close, volume, schedule)
    summary = perf_summary(res["equity"])
    ann = float(summary["年化收益"].strip("%")) / 100
    return {"mom_lookback": mom_lookback, "top_n": top_n,
            "slippage_bps": slippage_bps, "sharpe": summary["夏普比率"],
            "ann_return": ann, "equity": res["equity"]}


def main():
    cfg = load_config()
    bt = cfg["backtest"]
    stocks = get_stock_pool(cfg["universe"]["sample_size"],
                            cfg["universe"]["sample_seed"])
    data = load_ohlcv(stocks + BENCHMARK_ETFS, bt["start"], bt["end"])
    close, volume = data["close"], data["volume"]
    stocks = [s for s in stocks if s in close.columns]  # 剔除下载失败的
    stock_close = close[stocks].dropna(how="all")
    OUT.mkdir(exist_ok=True)

    # ---- 1. 参数敏感性：动量窗口 × 持仓数（基准成本 5bp）----
    grid = []
    for lb in (126, 189, 252, 315):
        for n in (20, 30, 50):
            r = _run_one(close, volume, stock_close, cfg, lb, n, 5.0)
            grid.append(r)
            print(f"mom={lb} top_n={n} -> Sharpe {r['sharpe']}")
    heat = sensitivity_table(grid, "mom_lookback", "top_n")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(heat.values.astype(float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(heat.columns)), [f"{c}只" for c in heat.columns])
    ax.set_yticks(range(len(heat.index)), [f"{i}日" for i in heat.index])
    ax.set_xlabel("持仓数")
    ax.set_ylabel("动量窗口")
    ax.set_title("参数敏感性：夏普比率热力图（基准成本5bp）")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=9)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "param_heatmap.png", dpi=120)
    plt.close(fig)

    # ---- 2. 成本敏感性：滑点 2/5/10bp（基准参数 mom=252, top_n=30）----
    cost_rows = []
    for bp in (2.0, 5.0, 10.0):
        r = _run_one(close, volume, stock_close, cfg, 252, 30, bp)
        cost_rows.append(r)
        print(f"滑点{bp}bp -> Sharpe {r['sharpe']}, 年化 {r['ann_return']:.2%}")

    # ---- 3. 蒙特卡洛 + Deflated Sharpe（基准参数，5bp）----
    base_eq = cost_rows[1]["equity"]
    rets = base_eq.pct_change().dropna()
    sim_sharpes = block_bootstrap_sharpe(rets, n_sims=1000)
    base_sharpe = float(perf_summary(base_eq)["夏普比率"])
    n_trials = 12 + 3   # 本次参数网格12组 + 成本3档，诚实申报
    dsr = deflated_sharpe_ratio(base_sharpe, rets, n_trials)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sim_sharpes, bins=40, color="steelblue", alpha=0.8)
    ax.axvline(base_sharpe, color="red", ls="--", label=f"实际夏普 {base_sharpe:.2f}")
    ax.set_title("蒙特卡洛 block bootstrap：夏普分布（1000 次重组历史）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mc_sharpe_dist.png", dpi=120)
    plt.close(fig)

    # ---- 汇总报告 ----
    lines = ["# 过拟合检验报告\n",
             "## 1. 参数敏感性（夏普比率；行=动量窗口，列=持仓数）\n",
             heat.to_markdown(),
             "\n判读：邻域内夏普若剧烈波动（相邻格差>0.5）→ 过拟合警报。\n",
             "## 2. 成本敏感性（mom=252, top_n=30）\n",
             "| 滑点(单边) | 夏普 | 年化收益 |", "|---|---|---|"]
    for r in cost_rows:
        lines.append(f"| {r['slippage_bps']}bp | {r['sharpe']} | {r['ann_return']:.2%} |")
    lines += ["\n判读：滑点翻 4 倍（2→10bp）夏普仍>0 才算对成本不敏感。\n",
              "## 3. 蒙特卡洛 + Deflated Sharpe\n",
              f"- block bootstrap 1000 次：夏普中位数 "
              f"{np.median(sim_sharpes):.2f}，"
              f"5%分位 {np.percentile(sim_sharpes, 5):.2f}，"
              f"95%分位 {np.percentile(sim_sharpes, 95):.2f}",
              f"- 实际夏普 {base_sharpe:.2f}；诚实申报试验次数 n_trials={n_trials}",
              f"- **Deflated Sharpe p 值 = {dsr:.3f}**"
              f"（>0.95 站得住；0.9~0.95 灰色；<0.9 过拟合警报）",
              "\n> 合规声明：仅供学习研究，不构成投资建议；回测≠实盘。"]
    (OUT / "sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n过拟合三件套完成，见 outputs/sensitivity_report.md")


if __name__ == "__main__":
    main()
