"""可交互的场景服务：交接单、实盘和撤签确认分别改变不同状态。"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any

from .forecast import DemandModel
from .ledger import Ledger, shipment_state
from .loader import Dataset
from .planning import (
    build_store_plans,
    evaluate_count_slots,
    optimize_count_slots,
    plan_vehicle_capacity,
)
from .timeutil import parse_ts


class ScenarioService:
    def __init__(self, data: Dataset):
        # 服务内更新不污染调用方已经加载的 Dataset。
        self.data = copy.deepcopy(data)
        self.demand = DemandModel(self.data)

    def ledger(self, as_of: datetime | str | None = None) -> Ledger:
        return Ledger(self.data, as_of=as_of)

    def stock(self, store: str, sku: str, at: datetime | str, as_of: datetime | str | None = None):
        return self.ledger(as_of).old_range(store, sku, at)

    def add_transfer(
        self,
        shipment: str,
        from_store: str,
        to_store: str,
        sku: str,
        qty: int,
        min_old: int,
        max_old: int,
        picked_at: datetime | str,
        arrived_at: datetime | str | None = None,
        known_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        row = {
            "shipment": shipment,
            "from_store": from_store,
            "to_store": to_store,
            "sku": sku,
            "qty": int(qty),
            "min_old": int(min_old),
            "max_old": int(max_old),
            "picked": parse_ts(picked_at) if isinstance(picked_at, str) else picked_at,
            "arrived": parse_ts(arrived_at) if isinstance(arrived_at, str) else arrived_at,
            "known": parse_ts(known_at) if isinstance(known_at, str) else known_at,
        }
        self.data.transfers.append(row)
        return row

    def record_handoff_confirmation(
        self,
        shipment: str,
        confirmed_at: datetime | str,
        arrived_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        """新到交接确认只改变交接/责任状态；没有 arrived 时不会变成可售库存。"""
        row = next(r for r in self.data.transfers if r["shipment"] == shipment)
        when = parse_ts(confirmed_at) if isinstance(confirmed_at, str) else confirmed_at
        row["known"] = when
        if arrived_at is not None:
            row["arrived"] = parse_ts(arrived_at) if isinstance(arrived_at, str) else arrived_at
        return row

    def record_arrival(self, shipment: str, arrived_at: datetime | str) -> dict[str, Any]:
        """实际送达使货物进入收货店可售量；known 仍可空，表示交接尚未确认。"""
        row = next(r for r in self.data.transfers if r["shipment"] == shipment)
        row["arrived"] = parse_ts(arrived_at) if isinstance(arrived_at, str) else arrived_at
        return row

    def add_count(
        self,
        count_id: str,
        store: str,
        at: datetime | str,
        known_at: datetime | str,
        old: int,
        new: int,
        sku: str | None = None,
    ) -> dict[str, Any]:
        """实际盘点在 known_at 后改变旧/新分量的确定性，而不是改变实物总量。"""
        row: dict[str, Any] = {
            "count": count_id,
            "store": store,
            "at": parse_ts(at) if isinstance(at, str) else at,
            "known": parse_ts(known_at) if isinstance(known_at, str) else known_at,
            "old": int(old),
            "new": int(new),
        }
        if sku is not None:
            row["sku"] = sku
        self.data.counts.append(row)
        return row

    def add_movement(
        self,
        event_id: str,
        store: str,
        sku: str,
        at: datetime | str,
        known_at: datetime | str,
        kind: str,
        qty: int,
        min_old: int,
        max_old: int,
    ) -> dict[str, Any]:
        row = {
            "event": event_id,
            "store": store,
            "sku": sku,
            "at": parse_ts(at) if isinstance(at, str) else at,
            "known": parse_ts(known_at) if isinstance(known_at, str) else known_at,
            "kind": kind,
            "qty": int(qty),
            "min_old": int(min_old),
            "max_old": int(max_old),
        }
        self.data.movements.append(row)
        return row

    def confirm_removal(
        self,
        event_id: str,
        store: str,
        sku: str,
        at: datetime | str,
        qty: int,
        known_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        """撤签确认把确切旧版件数移入“已撤下”状态，并从可售量扣除。"""
        when = parse_ts(at) if isinstance(at, str) else at
        return self.add_movement(
            event_id,
            store,
            sku,
            when,
            known_at or when,
            "撤签",
            qty,
            qty,
            qty,
        )

    def planning_report(self, as_of: datetime | str, sku: str | None = None) -> dict[str, Any]:
        sku = sku or self.data.skus()[0]
        plans = build_store_plans(self.data, as_of, sku, self.demand)
        candidates = evaluate_count_slots(self.data, as_of, sku, self.demand)
        ledger = self.ledger(as_of)
        hq_summary = {
            "sku": sku,
            **ledger.sku_totals(sku, parse_ts(as_of)),
            "global_sellable_old_range": list(ledger.global_old_range(sku, as_of)),
            "global_network_old_range": list(
                ledger.global_old_range(sku, as_of, include_in_transit=True)
            ),
        }
        count_by_info = max(candidates, key=lambda c: c["stock_ambiguity_units"], default=None)
        shared_batch = next(
            (c for c in candidates if c["resolves_shipment_provenance"]), None
        )
        insights = {
            "largest_stock_ambiguity_slot": count_by_info["slot"] if count_by_info else None,
            "largest_stock_ambiguity_store": count_by_info["store"] if count_by_info else None,
            "shared_batch_source_slot": shared_batch["slot"] if shared_batch else None,
            "decision_rule": (
                "先按 deadline/confirm_before 排序；同时展示存量宽度、"
                "调拨批次复用和班组重叠，不用预测代替零旧版实盘"
            ),
        }
        return {
            "as_of": parse_ts(as_of).isoformat() if isinstance(as_of, str) else as_of.isoformat(),  # type: ignore[union-attr]
            "sku": sku,
            "hq_summary": hq_summary,
            "insights": insights,
            "stores": plans,
            "count_slots": candidates,
            "parallel_count_plan": optimize_count_slots(candidates, single_crew=False),
            "single_crew_count_plan": optimize_count_slots(candidates, single_crew=True),
            "vehicles": plan_vehicle_capacity(self.data, plans, as_of),
            "demand_backtest": self.demand.backtest_trials(),
            "block_backtest": self.demand.block_backtest(),
            "old_end_calibration": self.demand.old_end_calibration(),
        }

    def retrospective_review(
        self,
        decision_as_of: datetime | str = "2026-09-05T08:00",
        known_as_of: datetime | str | None = None,
    ) -> dict[str, Any]:
        """用后到单据按事件时间回算，但不改变决策时点的可用状态。"""
        decision = parse_ts(decision_as_of) if isinstance(decision_as_of, str) else decision_as_of
        assert decision is not None
        review_ledger = self.ledger(known_as_of)
        late_events = []
        for row in sorted(self.data.movements, key=lambda r: r["at"]):
            if row.get("known") is None or row["known"] <= decision:
                continue
            at_range = review_ledger.old_range(row["store"], row["sku"], row["at"])
            late_events.append(
                {
                    "event": row["event"],
                    "store": row["store"],
                    "kind": row["kind"],
                    "event_at": row["at"].isoformat(),
                    "known_at": row["known"].isoformat(),
                    "qty": row["qty"],
                    "applied_in_event_time_order": True,
                    "range_after_event": [at_range.old_min, at_range.old_max],
                    "total_after_event": at_range.total,
                    "effect_on_current_sellable": "none_to_removed_units__chronological_balance_only",
                }
            )

        store_reviews = []
        for deadline in self.data.deadlines:
            store = deadline["store"]
            sku = self.data.skus()[0]
            forecast = self.demand.calibrated_block_forecast(store, decision, deadline["deadline"])
            window_sale_rows = [
                r
                for r in self.data.movements
                if r["store"] == store
                and r["sku"] == sku
                and any(token in r["kind"] for token in ("收银", "销售", "售出"))
                and decision < r["at"] <= deadline["deadline"]
            ]
            window_sales = sum(r["qty"] for r in window_sale_rows)
            removed = sum(
                r["qty"]
                for r in self.data.movements
                if r["store"] == store
                and r["sku"] == sku
                and "撤" in r["kind"]
                and r["at"] <= deadline["deadline"]
            )
            terminal_count = any(
                c["store"] == store and decision < c["at"] <= deadline["deadline"]
                for c in self.data.counts
            )
            outcome_complete = bool(removed or terminal_count)
            end_range = review_ledger.old_range(store, sku, deadline["deadline"])
            store_reviews.append(
                {
                    "store": store,
                    "deadline": deadline["deadline"].isoformat(),
                    "forecast_demand_median_at_decision": forecast.quantiles[0.5],
                    "observed_window_sales_event_time": window_sales,
                    "window_sale_record_count": len(window_sale_rows),
                    "outcome_complete_in_data": outcome_complete,
                    "demand_forecast_error_median_minus_observed": (
                        forecast.quantiles[0.5] - window_sales
                        if window_sale_rows and outcome_complete
                        else None
                    ),
                    "actual_removed_by_deadline": removed,
                    "old_range_after_chronological_events_and_removal": [
                        end_range.old_min,
                        end_range.old_max,
                    ],
                    "note": "后到销售用于回算误差；已撤下数量不会被加回当前可售量",
                }
            )
        return {"late_events": late_events, "stores": store_reviews}

    def batch_trace(self, sku: str | None = None, as_of: datetime | str = "2026-09-05T08:00") -> dict[str, Any]:
        sku = sku or self.data.skus()[0]
        ledger = Ledger(self.data)
        decision_ledger = self.ledger(as_of)
        entries: list[dict[str, Any]] = []

        for row in sorted(self.data.opening, key=lambda r: r["at"]):
            if row["sku"] != sku:
                continue
            rng = ledger.old_range(row["store"], sku, row["at"])
            entries.append(
                {
                    "time": row["at"].isoformat(),
                    "kind": "期初",
                    "ref": f"opening:{row['store']}",
                    "store": row["store"],
                    "qty": row["total"],
                    "old_bounds_recorded": [row["min_old"], row["max_old"]],
                    "old_range_after": [rng.old_min, rng.old_max],
                    "sellable_total_after": rng.total,
                }
            )

        for row in sorted(self.data.movements, key=lambda r: r["at"]):
            if row["sku"] != sku:
                continue
            rng = ledger.old_range(row["store"], sku, row["at"])
            entries.append(
                {
                    "time": row["at"].isoformat(),
                    "known_at": row["known"].isoformat() if row.get("known") else None,
                    "kind": row["kind"],
                    "ref": row["event"],
                    "store": row["store"],
                    "qty": row["qty"],
                    "old_bounds_recorded": [row["min_old"], row["max_old"]],
                    "old_range_after": [rng.old_min, rng.old_max],
                    "sellable_total_after": rng.total,
                    "inspection_units": rng.inspection_units,
                }
            )

        for ship in self.data.transfers:
            for label, ts, state in (
                ("调拨取走", ship["picked"], "在途/已取走"),
                ("调拨送达", ship["arrived"], "已送达"),
                ("交接确认", ship["known"], "已确认"),
            ):
                if ts is None:
                    continue
                old_range = ledger.shipment_old_range(ship["shipment"], ts)
                entries.append(
                    {
                        "time": ts.isoformat(),
                        "kind": label,
                        "ref": ship["shipment"],
                        "from_store": ship["from_store"],
                        "to_store": ship["to_store"],
                        "qty": ship["qty"],
                        "shipment_old_range": list(old_range),
                        "state": shipment_state(ship, ts, ts),
                    }
                )

        for count in sorted(self.data.counts, key=lambda r: r["at"]):
            rng = ledger.old_range(count["store"], sku, count["at"])
            entries.append(
                {
                    "time": count["at"].isoformat(),
                    "known_at": count["known"].isoformat(),
                    "kind": "实盘",
                    "ref": count["count"],
                    "store": count["store"],
                    "counted_old_new": [count["old"], count["new"]],
                    "old_range_after": [rng.old_min, rng.old_max],
                    "sellable_total_after": rng.total,
                }
            )

        entries.sort(key=lambda e: e["time"])
        removals = [
            {
                "event": r["event"],
                "store": r["store"],
                "at": r["at"].isoformat(),
                "known_at": r["known"].isoformat() if r.get("known") else None,
                "qty": r["qty"],
                "old_qty": r["min_old"],
            }
            for r in self.data.movements
            if r["sku"] == sku and "撤" in r["kind"]
        ]

        count_effects = []
        for count in sorted(self.data.counts, key=lambda r: r["at"]):
            before_ts = count["at"] - timedelta(seconds=1)
            before = ledger.old_range(count["store"], sku, before_ts)
            after = ledger.old_range(count["store"], sku, count["at"])
            count_effects.append(
                {
                    "count": count["count"],
                    "store": count["store"],
                    "at": count["at"].isoformat(),
                    "known_at": count["known"].isoformat(),
                    "before_range": [before.old_min, before.old_max],
                    "after_range": [after.old_min, after.old_max],
                    "ambiguity_reduced_by": max(0, (before.old_max - before.old_min) - (after.old_max - after.old_min)),
                    "counted_old_new": [count["old"], count["new"]],
                }
            )

        snapshot_times = [
            ("HQ决策时点", parse_ts(as_of)),
        ]
        for deadline in self.data.deadlines:
            snapshot_times.append((f"{deadline['store']}期限", deadline["deadline"]))
        snapshots = []
        for label, ts in snapshot_times:
            assert ts is not None
            view_ledger = decision_ledger if label == "HQ决策时点" else ledger
            snapshots.append(
                {
                    "label": label,
                    "physical_at": ts.isoformat(),
                    "knowledge_as_of": as_of if label == "HQ决策时点" else None,
                    "stores": [
                        view_ledger.old_range(store, sku, ts).as_dict()
                        for store in self.data.stores(sku)
                    ],
                }
            )

        return {
            "sku": sku,
            "entries": entries,
            "count_effects": count_effects,
            "snapshots": snapshots,
            "actual_removals": removals,
        }
