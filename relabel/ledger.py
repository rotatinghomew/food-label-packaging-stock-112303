"""用线性可行流恢复跨店、跨调拨一致的旧版数量范围。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .loader import Dataset
from .timeutil import parse_ts

SELLABLE_ADD = {"收货", "可再上架退货", "退货", "退货上架"}
SALE_KINDS = {"收银", "销售"}
REMOVE_KINDS = {"撤签", "撤架", "撤下"}
PENDING_RETURN = "待验退"


class LedgerInfeasible(RuntimeError):
    """所有非负流转假设均无法同时满足实盘和单据。"""


def is_pending_return(kind: str) -> bool:
    return PENDING_RETURN in kind or ("待验" in kind and "退" in kind)


def movement_effect(kind: str) -> str:
    """把近义中文单据类型归一为 add/sale/remove/pending。"""
    if is_pending_return(kind):
        return "pending"
    if "撤" in kind:
        return "remove"
    if any(token in kind for token in ("收银", "销售", "售出")):
        return "sale"
    if any(token in kind for token in ("收货", "入库", "退货上架", "可再上架")) or (
        "退" in kind and "待验" not in kind
    ):
        return "add"
    raise DataShapeError(f"未知 movement.kind: {kind}")


def _visible(row: dict[str, Any], as_of: datetime | None) -> bool:
    if as_of is None:
        return True
    known = row.get("known") or row.get("at")
    if known is None:
        return True
    return known <= as_of


@dataclass(frozen=True)
class StockRange:
    store: str
    sku: str
    at: datetime
    as_of: datetime | None
    total: int
    old_min: int
    old_max: int
    inspection_units: int = 0

    @property
    def exact(self) -> bool:
        return self.old_min == self.old_max

    @property
    def confirmed_old(self) -> int:
        """所有可行流转中都必然存在的旧版件数。"""
        return self.old_min

    @property
    def possible_extra_old(self) -> int:
        """可能是旧版、但尚不能确认的额外件数。"""
        return max(0, self.old_max - self.old_min)

    @property
    def new_min(self) -> int:
        return max(0, self.total - self.old_max)

    @property
    def new_max(self) -> int:
        return max(0, self.total - self.old_min)

    def as_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "sku": self.sku,
            "at": self.at.isoformat(),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "sellable_total": self.total,
            "old_min": self.old_min,
            "old_max": self.old_max,
            "confirmed_old": self.confirmed_old,
            "possible_extra_old": self.possible_extra_old,
            "new_min": self.new_min,
            "new_max": self.new_max,
            "inspection_units": self.inspection_units,
            "exact": self.exact,
        }


class Ledger:
    """按同一套变量求解某 SKU 的全局可行旧版流。"""

    def __init__(self, data: Dataset, as_of: datetime | str | None = None):
        self.data = data
        self.as_of = parse_ts(as_of) if isinstance(as_of, str) else as_of
        self._vars: dict[str, int] = {}
        self._bounds: list[tuple[float, float]] = []
        self._build_variables()

    # ---------- 变量与基础数量 ----------
    def _add_var(self, name: str, lower: float, upper: float) -> int:
        if name in self._vars:
            return self._vars[name]
        idx = len(self._bounds)
        self._vars[name] = idx
        self._bounds.append((lower, upper))
        return idx

    def _opening_rows(self, sku: str) -> list[dict[str, Any]]:
        return [r for r in self.data.opening if r["sku"] == sku]

    def _movement_rows(self, sku: str) -> list[dict[str, Any]]:
        return [
            r
            for r in self.data.movements
            if r["sku"] == sku and _visible(r, self.as_of)
        ]

    def _transfer_rows(self, sku: str) -> list[dict[str, Any]]:
        # transfer.known 是交接确认状态；空值仍代表一张已到、但未确认的交接单。
        return [r for r in self.data.transfers if r["sku"] == sku]

    def _count_rows(self, sku: str) -> list[dict[str, Any]]:
        # 交接资料的 counts.csv 没有 sku 列；当前场景按门店逐批盘点同一 SKU。
        # 若后续资料增加可选 sku 列，则按 SKU 过滤。
        return [
            r
            for r in self.data.counts
            if r.get("sku", sku) == sku and _visible(r, self.as_of)
        ]

    def _build_variables(self) -> None:
        for row in self.data.opening:
            self._add_var(
                f"open:{row['store']}:{row['sku']}", row["min_old"], row["max_old"]
            )
        for row in self.data.movements:
            if not _visible(row, self.as_of):
                continue
            if is_pending_return(row["kind"]):
                # 待验退货单独记录版本界，但不进入可售旧版流。
                self._add_var(
                    f"pending:{row['event']}", row["min_old"], row["max_old"]
                )
            else:
                self._add_var(
                    f"move:{row['event']}", row["min_old"], row["max_old"]
                )
        for row in self.data.transfers:
            self._add_var(
                f"ship:{row['shipment']}", row["min_old"], row["max_old"]
            )

    def _var_index(self, name: str) -> int:
        if name not in self._vars:
            # 没有期初行的门店视为期初为 0，而不是 KeyError。
            if name.startswith("open:"):
                return self._add_var(name, 0, 0)
            raise KeyError(name)
        return self._vars[name]

    def _expr_vector(self, expr: dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self._bounds), dtype=float)
        for name, coef in expr.items():
            vec[self._var_index(name)] += coef
        return vec

    # ---------- 确定性总件数 ----------
    def inspection_qty(self, store: str, sku: str, at: datetime | str) -> int:
        at_ts = parse_ts(at) if isinstance(at, str) else at
        assert at_ts is not None
        total = 0
        for row in self._movement_rows(sku):
            if row["store"] != store or row["at"] > at_ts:
                continue
            if is_pending_return(row["kind"]):
                total += row["qty"]
        return total

    def total_qty(self, store: str, sku: str, at: datetime | str) -> int:
        """门店实际可售总量；待验退货和在途均不计入。"""
        at_ts = parse_ts(at) if isinstance(at, str) else at
        assert at_ts is not None
        at = at_ts
        total = 0
        for row in self._opening_rows(sku):
            if row["store"] == store and row["at"] <= at:
                total += row["total"]

        for row in self._movement_rows(sku):
            if row["store"] != store or row["at"] > at:
                continue
            kind = row["kind"]
            effect = movement_effect(kind)
            if effect == "pending":
                continue
            if effect == "add":
                total += row["qty"]
            elif effect in {"sale", "remove"}:
                total -= row["qty"]
            else:
                raise DataShapeError(f"未知 movement.kind: {kind}")

        for ship in self._transfer_rows(sku):
            if ship["from_store"] == store and ship["picked"] <= at:
                total -= ship["qty"]
            if (
                ship["to_store"] == store
                and ship["arrived"] is not None
                and ship["arrived"] <= at
            ):
                total += ship["qty"]

        if total < 0:
            raise LedgerInfeasible(f"{store} {sku} 在 {at} 可售总量为负: {total}")
        return total

    # ---------- 旧版累计表达式 ----------
    def old_expression(self, store: str, sku: str, at: datetime) -> dict[str, float]:
        expr: dict[str, float] = {}

        def add(name: str, coef: float) -> None:
            expr[name] = expr.get(name, 0.0) + coef

        for row in self._opening_rows(sku):
            if row["store"] == store and row["at"] <= at:
                add(f"open:{store}:{sku}", 1.0)

        for row in self._movement_rows(sku):
            if row["store"] != store or row["at"] > at:
                continue
            kind = row["kind"]
            effect = movement_effect(kind)
            if effect == "pending":
                continue
            sign = 1.0 if effect == "add" else -1.0
            add(f"move:{row['event']}", sign)

        for ship in self._transfer_rows(sku):
            var = f"ship:{ship['shipment']}"
            if ship["from_store"] == store and ship["picked"] <= at:
                add(var, -1.0)
            if (
                ship["to_store"] == store
                and ship["arrived"] is not None
                and ship["arrived"] <= at
            ):
                add(var, 1.0)
        return expr

    def _constraints_until(
        self, sku: str, at: datetime
    ) -> tuple[list[np.ndarray], list[float], list[np.ndarray], list[float]]:
        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []
        a_eq: list[np.ndarray] = []
        b_eq: list[float] = []

        # 实盘给出精确旧/新分量。只有在 as_of 前已知结果的盘点才能约束决策。
        for count in self._count_rows(sku):
            if count["at"] > at:
                continue
            expr = self.old_expression(count["store"], sku, count["at"])
            a_eq.append(self._expr_vector(expr))
            b_eq.append(float(count["old"]))
            book_total = self.total_qty(count["store"], sku, count["at"])
            counted_total = count["old"] + count["new"]
            if book_total != counted_total:
                raise LedgerInfeasible(
                    f"{count['count']} 实盘总量 {counted_total} 与可售账面总量 "
                    f"{book_total} 不一致；待验库位不应计入可售盘点"
                )

        # 在每个业务时点后保证旧版、新版均非负。
        timestamps: set[datetime] = {at}
        for row in self._opening_rows(sku):
            if row["at"] <= at:
                timestamps.add(row["at"])
        for row in self._movement_rows(sku):
            if row["at"] <= at:
                timestamps.add(row["at"])
        for ship in self._transfer_rows(sku):
            if ship["picked"] <= at:
                timestamps.add(ship["picked"])
            if ship["arrived"] is not None and ship["arrived"] <= at:
                timestamps.add(ship["arrived"])
        for ts in sorted(timestamps):
            for store in self.data.stores(sku):
                if not any(
                    True
                    for r in self._opening_rows(sku)
                    if r["store"] == store and r["at"] <= ts
                ) and not any(
                    True for r in self._movement_rows(sku) if r["store"] == store and r["at"] <= ts
                ):
                    continue
                expr = self.old_expression(store, sku, ts)
                vec = self._expr_vector(expr)
                total = self.total_qty(store, sku, ts)
                # old >= 0  <=> -old <= 0
                a_ub.append(-vec)
                b_ub.append(0.0)
                # new = total-old >= 0 <=> old <= total
                a_ub.append(vec)
                b_ub.append(float(total))
        return a_ub, b_ub, a_eq, b_eq

    def _optimize_expression(
        self,
        expr: dict[str, float],
        sku: str,
        at: datetime,
        minimize: bool,
    ) -> float:
        a_ub, b_ub, a_eq, b_eq = self._constraints_until(sku, at)
        c = self._expr_vector(expr)
        if not minimize:
            c = -c
        result = linprog(
            c,
            A_ub=np.array(a_ub) if a_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(a_eq) if a_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=self._bounds,
            method="highs",
        )
        if not result.success:
            raise LedgerInfeasible(f"旧版可行流求解失败: {result.message}")
        value = float(result.fun)
        if abs(value - round(value)) > 1e-5:
            value = self._optimize_integer_expression(c, a_ub, b_ub, a_eq, b_eq)
        return value if minimize else -value

    def _optimize_integer_expression(
        self,
        objective: np.ndarray,
        a_ub: list[np.ndarray],
        b_ub: list[float],
        a_eq: list[np.ndarray],
        b_eq: list[float],
    ) -> float:
        constraints = []
        if a_ub:
            constraints.append(
                LinearConstraint(np.array(a_ub), -np.inf, np.array(b_ub, dtype=float))
            )
        if a_eq:
            eq = np.array(a_eq, dtype=float)
            constraints.append(LinearConstraint(eq, np.array(b_eq, dtype=float), np.array(b_eq, dtype=float)))
        lower = np.array([lo for lo, _ in self._bounds], dtype=float)
        upper = np.array([hi for _, hi in self._bounds], dtype=float)
        result = milp(
            c=objective,
            integrality=np.ones(len(self._bounds)),
            bounds=Bounds(lower, upper),
            constraints=constraints,
        )
        if not result.success:
            raise LedgerInfeasible(f"整数旧版可行流求解失败: {result.message}")
        return float(result.fun)

    @staticmethod
    def _integer_endpoint(value: float) -> int:
        rounded = int(round(value))
        if abs(value - rounded) > 1e-5:
            # 数量模型的系数矩阵是整数流网络，正常端点应为整数。
            raise LedgerInfeasible(f"得到非整数数量端点: {value}")
        return rounded

    def sku_totals(self, sku: str, at: datetime | str) -> dict[str, int]:
        ts = parse_ts(at) if isinstance(at, str) else at
        assert ts is not None
        sellable_by_store = {
            store: self.total_qty(store, sku, ts) for store in self.data.stores(sku)
        }
        in_transit = 0
        for ship in self._transfer_rows(sku):
            if ship["picked"] <= ts and (ship["arrived"] is None or ship["arrived"] > ts):
                in_transit += int(ship["qty"])
        inspection = sum(
            self.inspection_qty(store, sku, ts) for store in self.data.stores(sku)
        )
        sellable = sum(sellable_by_store.values())
        return {
            "sellable_by_store": sellable_by_store,
            "sellable_total": sellable,
            "in_transit_total": in_transit,
            "inspection_total": inspection,
            "network_total": sellable + in_transit,
        }

    def old_range(self, store: str, sku: str, at: datetime | str) -> StockRange:
        ts = parse_ts(at) if isinstance(at, str) else at
        assert ts is not None
        expr = self.old_expression(store, sku, ts)
        low = self._integer_endpoint(self._optimize_expression(expr, sku, ts, True))
        high = self._integer_endpoint(self._optimize_expression(expr, sku, ts, False))
        total = self.total_qty(store, sku, ts)
        inspection = self.inspection_qty(store, sku, ts)
        low = max(0, min(low, total))
        high = max(0, min(high, total))
        if low > high:
            raise LedgerInfeasible(f"{store} 旧版下界 {low} 大于上界 {high}")
        return StockRange(store, sku, ts, self.as_of, total, low, high, inspection)

    def global_old_range(
        self,
        sku: str,
        at: datetime | str,
        include_in_transit: bool = False,
        include_inspection: bool = False,
    ) -> tuple[int, int]:
        """所有门店可售旧版合计的联合范围（共享同一套调拨变量）。

        include_in_transit=True 时，把已取走但尚未送达的调拨旧版加回网络总量；
        这不会重复计算，因为原店表达式已经扣减、收货店表达式尚未增加。
        """
        ts = parse_ts(at) if isinstance(at, str) else at
        assert ts is not None
        expr: dict[str, float] = {}
        for store in self.data.stores(sku):
            for name, coef in self.old_expression(store, sku, ts).items():
                expr[name] = expr.get(name, 0.0) + coef
        if include_in_transit:
            for ship in self._transfer_rows(sku):
                if ship["picked"] <= ts and (ship["arrived"] is None or ship["arrived"] > ts):
                    name = f"ship:{ship['shipment']}"
                    expr[name] = expr.get(name, 0.0) + 1.0
        if include_inspection:
            for row in self._movement_rows(sku):
                if row["at"] <= ts and is_pending_return(row["kind"]):
                    name = f"pending:{row['event']}"
                    expr[name] = expr.get(name, 0.0) + 1.0
        low = self._integer_endpoint(self._optimize_expression(expr, sku, ts, True))
        high = self._integer_endpoint(self._optimize_expression(expr, sku, ts, False))
        return low, high

    def shipment_old_range(self, shipment_id: str, at: datetime | str | None = None) -> tuple[int, int]:
        row = next(r for r in self.data.transfers if r["shipment"] == shipment_id)
        sku = row["sku"]
        expr = {f"ship:{shipment_id}": 1.0}
        target = parse_ts(at) if isinstance(at, str) else at
        if target is None:
            timestamps = [r["at"] for r in self.data.movements if r["sku"] == sku]
            timestamps.extend(s["picked"] for s in self.data.transfers if s["sku"] == sku)
            timestamps.extend(s["arrived"] for s in self.data.transfers if s["sku"] == sku and s["arrived"])
            target = max(timestamps)
        low = self._integer_endpoint(self._optimize_expression(expr, sku, target, True))
        high = self._integer_endpoint(self._optimize_expression(expr, sku, target, False))
        return low, high

    def feasible(self, sku: str, at: datetime | str) -> bool:
        try:
            ts = parse_ts(at) if isinstance(at, str) else at
            assert ts is not None
            self._optimize_expression({f"open:{self.data.stores(sku)[0]}:{sku}": 1.0}, sku, ts, True)
            return True
        except LedgerInfeasible:
            return False


class DataShapeError(ValueError):
    pass


def shipment_state(ship: dict[str, Any], at: datetime | str, as_of: datetime | str | None = None) -> str:
    """返回调拨在指定物理时间/认知时间的状态。"""
    ts = parse_ts(at) if isinstance(at, str) else at
    view = parse_ts(as_of) if isinstance(as_of, str) else as_of
    assert ts is not None
    picked = ship["picked"]
    arrived = ship.get("arrived")
    confirmed = ship.get("known")
    if ts < picked:
        return "未取走"
    if arrived is None or ts < arrived:
        return "在途" if not (confirmed and view and confirmed <= view and arrived) else "在途"
    # 已送达。
    if confirmed is None or (view is not None and confirmed > view):
        return "已送达待交接确认"
    return "已交接确认"
