"""yfinance 数据加载器：批量下载 + parquet 本地缓存 + 数据质量自检。

设计原则：
- 首次运行联网下载并缓存；之后离线复现同一结果（缓存键 = 标的+区间+字段的哈希）。
- auto_adjust=True：价格已做复权处理（拆股与分红均调整），直接用于收益计算。
- 数据质量自检不过就抛错，不带着脏数据跑回测。
"""

import hashlib
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache"

# yfinance 默认把时区缓存写进包安装目录（可能只读）→ 改到项目缓存目录
CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR / "tz_cache"))


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    raw = ",".join(sorted(tickers)) + f"|{start}|{end}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _quality_check(df: pd.DataFrame, name: str) -> None:
    """数据质量自检。

    - 空数据、负价格：硬错误，直接抛（脏数据不回测）。
    - 高缺失率：仅警告不抛——DELL(2018重新上市)/CEG(2022分拆)/KHC(2015合并)
      这类"上市晚"是正常情况，因子层会自然跳过 NaN。
    """
    if df.empty:
        raise ValueError(f"{name}: 数据为空，下载失败或区间无效")
    if (df < 0).any().any():
        bad = df.columns[(df < 0).any()].tolist()
        raise ValueError(f"{name}: 存在负价格，标的={bad[:5]}")
    miss = df.isna().mean()
    late_ipo = miss[miss > 0.20]
    if not late_ipo.empty:
        warnings.warn(
            f"{name}: 以下标的缺失率>20%（多为上市较晚，因子计算会自动跳过）: "
            f"{late_ipo.index.tolist()[:10]}"
        )


def load_ohlcv(tickers: list[str], start: str, end: str,
               cache_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """下载/读取缓存，返回 {"close": DataFrame, "volume": DataFrame}。

    close: 复权收盘价，index=交易日期，columns= ticker。
    volume: 成交量（股数），用于容量参与率约束。
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(tickers, start, end)
    fp = cache_dir / f"ohlcv_{key}.parquet"

    if fp.exists():
        store = pd.read_parquet(fp)
    else:
        raw = yf.download(
            tickers=" ".join(tickers),
            start=start, end=end,
            auto_adjust=True,       # 复权价
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise ValueError("yfinance 下载为空：检查网络或标的代码")
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
        # 单标的时 yfinance 返回单层列名，统一成 MultiIndex 结构
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
            volume = volume.to_frame(tickers[0])
        store = pd.concat({"close": close, "volume": volume}, axis=1)
        store.to_parquet(fp)

    close = store["close"].dropna(axis=1, how="all")
    # 下载完全失败的标的（全 NaN 已被剔除）登记警告
    missing_cols = [t for t in tickers if t not in close.columns]
    if missing_cols:
        warnings.warn(f"以下标的下载失败已从池中剔除: {missing_cols}")
    volume = store["volume"].reindex(columns=close.columns)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]

    _quality_check(close, "close")
    return {"close": close, "volume": volume}


def coverage_report(close: pd.DataFrame) -> pd.DataFrame:
    """各标的数据覆盖情况：首日/末日/缺失率。用于报告附录与排障。"""
    return pd.DataFrame({
        "first_date": close.apply(lambda s: s.first_valid_index()),
        "last_date": close.apply(lambda s: s.last_valid_index()),
        "missing_ratio": close.isna().mean(),
        "n_obs": close.notna().sum(),
    })
