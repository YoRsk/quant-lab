"""单因子有效性检验脚本：IC/IR、分层、衰减、换手。

用法： PYTHONPATH=src python scripts/run_factor_test.py [--factor momentum]
产出： outputs/factor_report.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_lab.analysis.ic import (factor_turnover, ic_decay, ic_summary,
                                   quantile_returns, rank_ic_series)
from quant_lab.config import load_config
from quant_lab.data.universe import get_stock_pool
from quant_lab.data.yf_loader import load_ohlcv
from quant_lab.factors.price_factors import FACTOR_REGISTRY

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", default=None, help="只检验单个因子")
    args = ap.parse_args()

    cfg = load_config()
    bt = cfg["backtest"]
    stocks = get_stock_pool(cfg["universe"]["sample_size"],
                            cfg["universe"]["sample_seed"])
    close = load_ohlcv(stocks, bt["start"], bt["end"])["close"]
    close = close.dropna(how="all")

    names = [args.factor] if args.factor else list(FACTOR_REGISTRY)
    OUT.mkdir(exist_ok=True)
    lines = ["# 因子有效性检验报告\n",
             f"区间: {bt['start']} ~ {bt['end']} | 股票池 {len(stocks)} 只"
             f"（标普500抽样，存在生存者偏差）\n",
             "> 口径：Rank IC = 因子值与未来21日收益的横截面 Spearman 秩相关；"
             "ICIR = IC均值/IC标准差；|t|>2 视为统计显著。\n"]
    for n in names:
        f = FACTOR_REGISTRY[n](close)
        ic = rank_ic_series(f, close, horizon=21)
        s = ic_summary(ic)
        lines += [f"\n## {n}\n",
                  f"- IC均值 {s.get('IC_mean')} | ICIR {s.get('ICIR')} "
                  f"| IC>0占比 {s.get('IC_positive_ratio')} | t值 {s.get('t_stat')}",
                  f"- Top30 周换手: {factor_turnover(f)}\n",
                  "### IC 衰减\n", ic_decay(f, close).to_markdown(),
                  "\n### 分层年化收益（Q1=因子值最低层，Q5=最高层）\n",
                  quantile_returns(f, close).to_markdown(),
                  "\n判读：Q1→Q5 应大致单调递增；Q5-Q1 价差是因子多空收益的代理。\n"]
    lines.append("\n> 合规声明：仅供学习研究，不构成投资建议。")
    (OUT / "factor_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("因子检验完成，见 outputs/factor_report.md")


if __name__ == "__main__":
    main()
