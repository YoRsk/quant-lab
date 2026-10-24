"""过拟合检验三件套（样本外对比由策略脚本直接分段实现）：

1. 参数敏感性：扰动关键参数（动量窗口 × 持仓数），看绩效是否"悬崖式"恶化。
   健康策略在参数邻域内应表现平稳；只有单一参数组合好看 = 过拟合警报。
2. 成本敏感性：滑点 2/5/10bp 三档，看策略在成本加压下还能不能活。
3. 蒙特卡洛 block bootstrap + Deflated Sharpe：
   - block bootstrap：把日收益按"块"抽样重组（保留局部相关性），
     模拟"如果历史换一种顺序发生"，夏普比率的分布有多宽；
   - Deflated Sharpe（Bailey & López de Prado 2014）：校正"试了很多参数
     才挑出一个好看的"这种多重检验偏差，回答"这个夏普是真的还是挑出来的"。
"""

import numpy as np
import pandas as pd


def block_bootstrap_sharpe(returns: pd.Series, n_sims: int = 1000,
                           block: int = 21, seed: int = 42) -> np.ndarray:
    """对日收益序列做块自助法，返回模拟夏普比率数组（年化）。"""
    r = returns.dropna().values
    n = len(r)
    if n < block * 3:
        return np.array([])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_sims, n_blocks))
    sharpes = np.empty(n_sims)
    for i in range(n_sims):
        idx = (starts[i][:, None] + np.arange(block)).ravel()[:n]
        sim = r[idx]
        mu, sd = sim.mean(), sim.std()
        sharpes[i] = mu / sd * np.sqrt(252) if sd > 0 else 0.0
    return sharpes


def deflated_sharpe_ratio(observed_sharpe_ann: float, returns: pd.Series,
                          n_trials: int) -> float:
    """Deflated Sharpe（Bailey & López de Prado, 2014 原始公式）。

    回答的问题："试了 n_trials 个参数/因子组合后挑出的这个夏普，
    是真实能力还是多重检验的运气？"

    实现口径：
    - 统一在【日频夏普】尺度计算（observed_sharpe_ann 为年化值，内部换算）；
    - 试验 N 次时"运气最好那次"的期望夏普 e_max 必须乘以单次试验
      夏普估计的标准误 σ_SR（漏乘会把阈值放大约两个数量级——已修正）；
    - n_trials 必须诚实申报（研究过程中实际尝试过的组合总数）。

    返回值 ∈ (0,1)：>0.95 统计上站得住；0.9~0.95 灰色地带；<0.9 过拟合警报。
    """
    from scipy import stats

    r = returns.dropna()
    T = len(r)
    if T < 30 or n_trials < 1:
        return np.nan
    sr = observed_sharpe_ann / np.sqrt(252)          # 年化 → 日频尺度
    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0                 # pandas 超额峰度 +3 还原
    var_sr = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (T - 1)
    if var_sr <= 0:
        return np.nan
    sigma_sr = np.sqrt(var_sr)
    euler = 0.5772156649                             # 欧拉常数 γ
    n = max(n_trials, 1)
    # E[max SR] = σ_SR · [(1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))]
    e_max = sigma_sr * ((1 - euler) * stats.norm.ppf(1 - 1 / n)
                        + euler * stats.norm.ppf(1 - 1 / (n * np.e)))
    z = (sr - e_max) / sigma_sr
    return float(stats.norm.cdf(z))


def sensitivity_table(records: list[dict], row_key: str, col_key: str,
                      value_key: str = "sharpe") -> pd.DataFrame:
    """把参数网格结果透视成表：行=参数A，列=参数B，值=夏普。"""
    df = pd.DataFrame(records)
    return df.pivot(index=row_key, columns=col_key, values=value_key)
