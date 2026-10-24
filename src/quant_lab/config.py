"""加载五要素基线配置（config/base.yaml）。"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "base.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """读取 YAML 配置为 dict。默认读 config/base.yaml。"""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
