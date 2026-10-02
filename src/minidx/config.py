"""設定の読み込み。閾値・モデル名をコードに直書きしないための単一の入口。"""
from __future__ import annotations
import functools
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]


@functools.cache
def load(path: Path | None = None) -> dict[str, Any]:
    p = path or ROOT / "config.yaml"
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
