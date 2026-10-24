"""标的池定义。

【数据缺口登记 · 生存者偏差】
本模块内置的是「当前时点」标普 500 主要成分的静态快照（2025 年整理）。
用今天的成分股回测 2010 年的历史，天然遗漏了当年被踢出指数的弱势股，
回测结果会系统性偏乐观（即"生存者偏差"）。学习期可接受，但：
  - 报告中所有收益数字须视为"偏乐观的上界"；
  - 严谨研究需换用含历史成分变更的付费数据（CRSP/Compustat/Sharadar）。
"""

import random

# 基准与对照 ETF
BENCHMARK_ETFS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

# 标普 500 主要成分静态快照（约 180 只，覆盖 11 个行业，按市值与流动性筛选）
# 快照时点：2025 年；仅作学习用途。
SP500_SNAPSHOT = [
    # 信息技术
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO",
    "ACN", "IBM", "INTC", "TXN", "QCOM", "NOW", "INTU", "AMAT", "MU",
    "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MSI", "HPQ", "DELL", "ANET",
    # 通信服务
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "EA",
    # 非必需消费
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
    "CMG", "ORLY", "MAR", "GM", "F", "YUM", "DHI", "LEN",
    # 必需消费
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL", "MDLZ", "KHC",
    "STZ", "KMB", "GIS", "SYY", "HSY",
    # 金融
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SCHW",
    "C", "BLK", "SPGI", "CB", "PGR", "MMC", "AON", "USB", "PNC", "TFC",
    "COF", "AIG", "MET", "PRU", "ALL", "TRV",
    # 医疗保健
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
    "AMGN", "BMY", "GILD", "ISRG", "SYK", "REGN", "VRTX", "MDT", "CI",
    "ELV", "HUM", "ZTS", "BSX", "BDX",
    # 工业
    "CAT", "HON", "UNP", "UPS", "BA", "RTX", "LMT", "DE", "GE", "MMM",
    "GD", "NOC", "EMR", "ETN", "ITW", "CSX", "NSC", "FDX", "WM", "RSG",
    # 能源
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI",
    "WMB", "HAL", "DVN", "FANG", "BKR",
    # 公用事业
    "NEE", "SO", "DUK", "CEG", "SRE", "AEP", "D", "EXC", "XEL", "PCG",
    # 房地产
    "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "AVB",
    # 材料
    "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "ECL", "DD", "DOW", "VMC",
]


def get_stock_pool(sample_size: int = 150, seed: int = 42) -> list[str]:
    """从静态快照中抽取固定种子的子样本。

    sample_size >= len(SP500_SNAPSHOT) 时返回全集。
    固定种子保证任何机器上复现同一股票池。
    """
    pool = sorted(SP500_SNAPSHOT)
    if sample_size >= len(pool):
        return pool
    rng = random.Random(seed)
    return sorted(rng.sample(pool, sample_size))


def get_universe(sample_size: int = 150, seed: int = 42,
                 include_etfs: bool = True) -> list[str]:
    """完整标的池 = 股票子样本 + 基准 ETF。"""
    pool = get_stock_pool(sample_size, seed)
    if include_etfs:
        pool = pool + [t for t in BENCHMARK_ETFS if t not in pool]
    return pool
