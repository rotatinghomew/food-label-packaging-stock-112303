"""读取并校验换签场景的 CSV 交接资料。"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .timeutil import parse_ts

REQUIRED_TABLES = {
    "opening.csv",
    "movements.csv",
    "transfer.csv",
    "counts.csv",
    "slots.csv",
    "historical.csv",
    "deadlines.csv",
}

INT_FIELDS = {
    "opening.csv": {"total", "min_old", "max_old"},
    "movements.csv": {"qty", "min_old", "max_old"},
    "transfer.csv": {"qty", "min_old", "max_old"},
    "counts.csv": {"old", "new"},
    "slots.csv": {"max_units"},
    "historical.csv": {"observed", "actual_old_end"},
}

TIME_FIELDS = {
    "opening.csv": {"at"},
    "movements.csv": {"at", "known"},
    "transfer.csv": {"picked", "arrived", "known"},
    "counts.csv": {"at", "known"},
    "slots.csv": {"opens", "closes"},
    "deadlines.csv": {"deadline", "confirm_before"},
}


class DataValidationError(ValueError):
    """交接资料不满足基本约束。"""


@dataclass
class Dataset:
    root: Path
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    headers: dict[str, list[str]] = field(default_factory=dict)

    def rows(self, name: str) -> list[dict[str, Any]]:
        key = name if name.endswith(".csv") else f"{name}.csv"
        return self.tables[key]

    # 常用快捷访问。
    @property
    def opening(self) -> list[dict[str, Any]]:
        return self.rows("opening")

    @property
    def movements(self) -> list[dict[str, Any]]:
        return self.rows("movements")

    @property
    def transfers(self) -> list[dict[str, Any]]:
        return self.rows("transfer")

    @property
    def counts(self) -> list[dict[str, Any]]:
        return self.rows("counts")

    @property
    def slots(self) -> list[dict[str, Any]]:
        return self.rows("slots")

    @property
    def historical(self) -> list[dict[str, Any]]:
        return self.rows("historical")

    @property
    def deadlines(self) -> list[dict[str, Any]]:
        return self.rows("deadlines")

    def deadline_for(self, store: str) -> dict[str, Any] | None:
        return next((r for r in self.deadlines if r["store"] == store), None)

    def stores(self, sku: str | None = None) -> list[str]:
        values = set()
        for row in self.opening:
            if sku is None or row["sku"] == sku:
                values.add(row["store"])
        return sorted(values)

    def skus(self) -> list[str]:
        return sorted({r["sku"] for r in self.opening})


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames is None:
            raise DataValidationError(f"{path.name} 缺少表头")
        header = list(reader.fieldnames)
        if len(header) != len(set(header)):
            raise DataValidationError(f"{path.name} 有重复列名")
        rows: list[dict[str, str]] = []
        for raw in reader:
            if None in raw:
                raise DataValidationError(f"{path.name} 存在超出表头宽度的行")
            rows.append(dict(raw))
        return header, rows


def load_dataset(directory: str | Path, validate: bool = True) -> Dataset:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(root)
    dataset = Dataset(root=root)
    for path in sorted(root.glob("*.csv")):
        header, raw_rows = _read_csv(path)
        rows: list[dict[str, Any]] = []
        ints = INT_FIELDS.get(path.name, set())
        times = TIME_FIELDS.get(path.name, set())
        for raw in raw_rows:
            if any(raw[k] is None for k in header):
                raise DataValidationError(f"{path.name} 有宽度不一致的记录")
            row: dict[str, Any] = dict(raw)
            for key in ints:
                if key in row and row[key] != "":
                    try:
                        row[key] = int(row[key])
                    except ValueError as exc:
                        raise DataValidationError(
                            f"{path.name} 的 {key} 不是整数: {row[key]!r}"
                        ) from exc
            for key in times:
                if key in row:
                    row[key] = parse_ts(row[key])
            if path.name == "slots.csv" and "confirmed" in row:
                row["confirmed"] = str(row["confirmed"]).strip() in {"1", "true", "True", "yes"}
            if path.name == "deadlines.csv" and "max_residual_probability" in row:
                row["max_residual_probability"] = float(row["max_residual_probability"])
            rows.append(row)
        dataset.headers[path.name] = header
        dataset.tables[path.name] = rows

    missing = REQUIRED_TABLES - set(dataset.tables)
    if missing:
        raise DataValidationError(f"缺少资料文件: {sorted(missing)}")
    if validate:
        validate_dataset(dataset)
    return dataset


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def validate_dataset(data: Dataset) -> None:
    """检查非负、版本上下界和时间顺序等基本交接约束。"""
    for row in data.opening:
        _require(row["total"] >= 0, f"期初总量为负: {row}")
        _require(0 <= row["min_old"] <= row["max_old"] <= row["total"], f"期初旧版界无效: {row}")

    for row in data.movements:
        _require(row["qty"] >= 0, f"收发存数量为负: {row['event']}")
        _require(
            0 <= row["min_old"] <= row["max_old"] <= row["qty"],
            f"收发存旧版界无效: {row['event']}",
        )
        _require(row["at"] is not None, f"{row['event']} 缺少 at")

    for row in data.transfers:
        _require(row["qty"] >= 0, f"调拨数量为负: {row['shipment']}")
        _require(
            0 <= row["min_old"] <= row["max_old"] <= row["qty"],
            f"调拨旧版界无效: {row['shipment']}",
        )
        _require(row["picked"] is not None, f"{row['shipment']} 缺少 picked")
        if row["arrived"] is not None:
            _require(row["arrived"] >= row["picked"], f"{row['shipment']} 送达早于取走")
        if row["known"] is not None and row["arrived"] is not None:
            # known 是交接确认时间；允许晚于送达，不应早于取走。
            _require(row["known"] >= row["picked"], f"{row['shipment']} 确认早于取走")

    ids = [r["shipment"] for r in data.transfers]
    _require(len(ids) == len(set(ids)), "transfer shipment 重复")

    for row in data.counts:
        _require(row["old"] >= 0 and row["new"] >= 0, f"盘点分量为负: {row['count']}")
        _require(row["at"] is not None and row["known"] is not None, f"{row['count']} 时间不完整")

    for row in data.slots:
        _require(row["closes"] > row["opens"], f"时段无效: {row['slot']}")
        _require(row["max_units"] > 0, f"时段容量无效: {row['slot']}")

    for row in data.historical:
        _require(row["observed"] >= 0, "historical observed 为负")
        _require(row["actual_old_end"] >= 0, "historical actual_old_end 为负")
