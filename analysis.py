"""高层分析入口的兼容模块。

示例：
    from analysis import analyze
    result = analyze("data", "2026-09-05T08:00")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from relabel.ledger import Ledger
from relabel.loader import Dataset, load_dataset
from relabel.planning import build_store_plans, evaluate_count_slots
from relabel.service import ScenarioService

__all__ = [
    "analyze",
    "load_dataset",
    "ScenarioService",
    "Ledger",
    "old_ranges",
]


def analyze(data_dir: str | Path = "data", as_of: str = "2026-09-05T08:00") -> dict[str, Any]:
    """生成完整计划报告和同批货轨迹。"""
    service = ScenarioService(load_dataset(data_dir))
    report = service.planning_report(as_of)
    report["trace"] = service.batch_trace(report["sku"], as_of=as_of)
    report["retrospective"] = service.retrospective_review(as_of)
    return report


def old_ranges(
    data: Dataset | str | Path = "data",
    at: str = "2026-09-05T08:00",
    as_of: str | None = "2026-09-05T08:00",
    sku: str | None = None,
) -> dict[str, dict[str, Any]]:
    """返回各门店在指定物理时点的旧版可行范围。"""
    dataset = load_dataset(data) if not isinstance(data, Dataset) else data
    sku = sku or dataset.skus()[0]
    ledger = Ledger(dataset, as_of=as_of)
    return {store: ledger.old_range(store, sku, at).as_dict() for store in dataset.stores(sku)}
