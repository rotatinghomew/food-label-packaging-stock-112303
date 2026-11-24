"""盘点信息价值、班组/时段冲突与撤架车容量安排。"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from itertools import combinations
from typing import Any

import numpy as np

from .forecast import (
    DemandModel,
    residual_envelope,
    safe_reserve_for_probability,
)
from .ledger import Ledger, is_pending_return, movement_effect, shipment_state
from .timeutil import parse_ts


def _hours_to(value: datetime, deadline: datetime) -> float:
    return max(0.0, (deadline - value).total_seconds() / 3600.0)


def exemption_status(
    data: DatasetLike,  # noqa: F821
    ledger: Ledger,
    store: str,
    sku: str,
) -> dict[str, Any]:
    """只有当前期实盘 old=0 且之后无旧版入库，才能证明可免撤。

    预测自然卖完、LP 下界为 0 或推算为 0 都不能替代实盘。
    """
    visible_counts = []
    for row in data.counts:
        if row["store"] != store:
            continue
        if ledger.as_of is not None and row["known"] is not None and row["known"] > ledger.as_of:
            continue
        visible_counts.append(row)
    counts = sorted(visible_counts, key=lambda r: r["at"])
    zero_count = None
    for count in reversed(counts):
        if count["old"] == 0:
            zero_count = count
            break
    if zero_count is None:
        return {
            "can_certify_zero": False,
            "basis_count": None,
            "reason": "没有 old=0 的已知实盘；预测或区间下界不能作为免撤依据",
        }

    for row in data.movements:
        if row["store"] != store or row["at"] <= zero_count["at"]:
            continue
        if ledger.as_of is not None and row.get("known") and row["known"] > ledger.as_of:
            continue
        if movement_effect(row["kind"]) == "add" and row["max_old"] > 0:
            return {
                "can_certify_zero": False,
                "basis_count": zero_count["count"],
                "reason": f"零旧版实盘 {zero_count['count']} 后仍有 max_old>0 的入库/退回",
            }
    for ship in data.transfers:
        if ship["to_store"] != store or ship["arrived"] is None:
            continue
        if ship["arrived"] > zero_count["at"] and ship["max_old"] > 0:
            return {
                "can_certify_zero": False,
                "basis_count": zero_count["count"],
                "reason": f"零旧版实盘 {zero_count['count']} 后仍有 max_old>0 的调拨到达",
            }
    return {
        "can_certify_zero": True,
        "basis_count": zero_count["count"],
        "reason": "零旧版实盘后只有销售/撤下或已证明为新版的入库",
    }


def build_store_plans(
    data: Any,
    as_of: datetime | str,
    sku: str | None = None,
    demand_model: DemandModel | None = None,
) -> list[dict[str, Any]]:
    asof = parse_ts(as_of) if isinstance(as_of, str) else as_of
    assert asof is not None
    sku = sku or data.skus()[0]
    ledger = Ledger(data, as_of=asof)
    demand = demand_model or DemandModel(data)
    plans: list[dict[str, Any]] = []

    for deadline_row in sorted(data.deadlines, key=lambda r: r["deadline"]):
        store = deadline_row["store"]
        stock = ledger.old_range(store, sku, asof)
        alpha = float(deadline_row["max_residual_probability"])
        forecast = demand.calibrated_block_forecast(store, asof, deadline_row["deadline"])
        demand_low = forecast.quantiles.get(alpha, float(np.quantile(forecast.samples, alpha)))  # type: ignore[arg-type]
        envelope = residual_envelope(
            stock.total,
            stock.old_min,
            stock.old_max,
            demand_low,
            forecast.quantiles[0.5],
            forecast.quantiles[0.9],
        )
        samples = forecast.samples
        assert samples is not None
        reserve = safe_reserve_for_probability(stock.total, stock.old_max, samples, alpha)
        unconfirmed_outgoing = []
        for ship in data.transfers:
            if ship["from_store"] != store or ship["sku"] != sku or ship["picked"] > asof:
                continue
            confirmed = ship["known"] is not None and ship["known"] <= asof
            if not confirmed:
                unconfirmed_outgoing.append(
                    {
                        "shipment": ship["shipment"],
                        "qty": ship["qty"],
                        "old_range": [ship["min_old"], ship["max_old"]],
                        "arrived": ship["arrived"].isoformat() if ship["arrived"] else None,
                    }
                )
        appointment_guard = {
            "sellable_old_min": stock.old_min,
            "unconfirmed_outgoing": unconfirmed_outgoing,
            "cannot_cancel_as_zero": bool(stock.old_min > 0 or unconfirmed_outgoing),
            "note": "在途/交接未确认批次不计入可售总量，但不能据此把原店撤签预约按零取消",
        }
        worst_residual = np.minimum(stock.old_max, np.maximum(0, stock.total - samples))
        natural = {
            "mean_rate_hours_to_sell_old_max": stock.old_max / forecast.mean_units_per_hour,
            "mean_rate_suggests_natural_sellout": (
                stock.old_max / forecast.mean_units_per_hour <= forecast.open_hours
            ),
            "prob_total_sellout_guarantees_no_old": float(np.mean(samples >= stock.total)),
            "prob_worst_case_residual": float(np.mean(worst_residual > 0)),
            "prob_possible_old_sellout_at_old_min": float(np.mean(samples >= stock.old_min)),
            "risk_threshold": alpha,
            "risk_too_high_without_certified_count": bool(np.mean(worst_residual > 0) > alpha),
        }
        demand_rows = demand.training_rows(store)
        plan = {
            "store": store,
            "sku": sku,
            "as_of": asof.isoformat(),
            "deadline": deadline_row["deadline"].isoformat(),
            "confirm_before": deadline_row["confirm_before"].isoformat(),
            "crew": deadline_row["crew"],
            "hours_to_deadline_open": forecast.open_hours,
            "stock": stock.as_dict(),
            "demand": {
                "model": "complete_trial_block_rate_scaled_to_open_hours",
                "historical_days": len(demand_rows),
                "early_unavailable_days": sum(
                    1 for r in demand_rows if r["exposure_hours"] < r["clock_hours"]
                ),
                "mean_units_per_hour": forecast.mean_units_per_hour,
                "mean_demand": forecast.mean_demand,
                "quantiles": {str(k): v for k, v in forecast.quantiles.items()},
                "basis": "historical.csv 的 observed/exposure_hours；短日不补零",
            },
            "residual_envelope": envelope,
            "risk": natural,
            "risk_adjusted_removal_reserve": reserve,
            "appointment_guard": appointment_guard,
            "exemption": exemption_status(data, ledger, store, sku),
        }
        plans.append(plan)
    return plans


def evaluate_count_slots(
    data: Any,
    as_of: datetime | str,
    sku: str | None = None,
    demand_model: DemandModel | None = None,
) -> list[dict[str, Any]]:
    asof = parse_ts(as_of) if isinstance(as_of, str) else as_of
    assert asof is not None
    sku = sku or data.skus()[0]
    ledger = Ledger(data, as_of=asof)
    demand = demand_model or DemandModel(data)
    deadline_by_store = {r["store"]: r for r in data.deadlines}
    result: list[dict[str, Any]] = []

    for slot in data.slots:
        if slot["kind"] != "盘点" or slot["opens"] < asof:
            continue
        store = slot["store"]
        if store not in deadline_by_store:
            continue
        stock = ledger.old_range(store, sku, slot["opens"])
        deadline = deadline_by_store[store]
        ready = slot["closes"] <= deadline["confirm_before"]
        resolves: list[str] = []
        downstream_reuse: list[dict[str, Any]] = []
        visible_counts = [
            c for c in data.counts
            if c["store"] == store and c["known"] is not None and c["known"] <= asof
        ]
        last_count = max(visible_counts, key=lambda c: c["at"]) if visible_counts else None
        for ship in data.transfers:
            if ship["from_store"] != store or ship["picked"] > slot["opens"]:
                continue
            # 只有“上一次已知实盘之后取出的唯一一批货”，新实盘才能通过差额反推其旧版分量。
            if last_count is None or ship["picked"] <= last_count["at"]:
                continue
            intervening_movements = [
                m for m in data.movements
                if m["store"] == store
                and last_count["at"] < m["at"] <= slot["opens"]
                and m["qty"] != 0
                and not is_pending_return(m["kind"])
            ]
            other_shipments = [
                x for x in data.transfers
                if x["from_store"] == store
                and x["shipment"] != ship["shipment"]
                and last_count["at"] < x["picked"] <= slot["opens"]
            ]
            if intervening_movements or other_shipments:
                continue
            resolves.append(ship["shipment"])
            downstream_reuse.append(
                {
                    "shipment": ship["shipment"],
                    "downstream_store": ship["to_store"],
                    "same_batch_count_can_be_reused": True,
                    "required_before_reuse": [
                        "source_actual_count_result_known",
                        "arrival_record",
                        "handoff_confirmation",
                    ],
                    "note": "来源店早盘厘清后，下游不必再为同一批货单独占盘点时段；但 C 自有库存仍需盘点",
                }
            )

        alpha = float(deadline["max_residual_probability"])
        after_count_forecast = demand.calibrated_block_forecast(store, slot["closes"], deadline["deadline"])
        samples = after_count_forecast.samples
        assert samples is not None
        reserve_if_low_old = safe_reserve_for_probability(
            stock.total, stock.old_min, samples, alpha
        )
        reserve_if_high_old = safe_reserve_for_probability(
            stock.total, stock.old_max, samples, alpha
        )
        item = {
            "slot": slot["slot"],
            "store": store,
            "opens": slot["opens"].isoformat(),
            "closes": slot["closes"].isoformat(),
            "max_units": slot["max_units"],
            "confirmed": bool(slot["confirmed"]),
            "crew": deadline["crew"],
            "deadline": deadline["deadline"].isoformat(),
            "confirm_before": deadline["confirm_before"].isoformat(),
            "ready_before_confirm_deadline": ready,
            "capacity_ok": stock.total <= slot["max_units"],
            "total_to_count": stock.total,
            "old_range_at_slot": [stock.old_min, stock.old_max],
            "new_pack_range_at_slot": [stock.new_min, stock.new_max],
            "conditional_removal_reserve_range": [
                min(reserve_if_low_old, reserve_if_high_old),
                max(reserve_if_low_old, reserve_if_high_old),
            ],
            "forecast_hours_after_count": after_count_forecast.open_hours,
            "stock_ambiguity_units": stock.possible_extra_old if not stock.exact else 0,
            "exact_old_at_slot": stock.exact,
            "resolves_shipment_provenance": resolves,
            "downstream_batch_reuse": downstream_reuse,
            "information_note": (
                "实盘将把当前旧/新总量范围收缩为一个确定分量，并把需求不确定性单独留下"
            ),
        }
        result.append(item)
    return result


def slots_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    start_a = parse_ts(a["opens"])
    end_a = parse_ts(a["closes"])
    start_b = parse_ts(b["opens"])
    end_b = parse_ts(b["closes"])
    assert start_a and end_a and start_b and end_b
    return start_a < end_b and start_b < end_a


def optimize_count_slots(
    candidates: list[dict[str, Any]],
    single_crew: bool = True,
) -> dict[str, Any]:
    """枚举极少盘点时段的可行组合。"""
    fixed = [c for c in candidates if c["confirmed"]]
    optional = [c for c in candidates if not c["confirmed"]]

    def feasible(selected: list[dict[str, Any]]) -> bool:
        for item in selected:
            if not item["capacity_ok"] or not item["ready_before_confirm_deadline"]:
                return False
        if single_crew:
            for i, a in enumerate(selected):
                for b in selected[i + 1 :]:
                    if a.get("crew") == b.get("crew") and slots_overlap(a, b):
                        return False
        return True

    def score(selected: list[dict[str, Any]]) -> tuple[float, ...]:
        # 期限越早越优先；同一期限再看能消除的存量宽度和是否厘清调拨批次。
        urgency = 0.0
        info = 0.0
        shipment_bonus = 0.0
        for item in selected:
            deadline = parse_ts(item["deadline"])
            assert deadline
            urgency += 100000.0 / max(1.0, (deadline - datetime(2026, 9, 1, 0, 0)).total_seconds() / 3600.0)
            info += float(item["stock_ambiguity_units"])
            shipment_bonus += 5.0 * len(item["resolves_shipment_provenance"])
        return (urgency, info + shipment_bonus)

    best: tuple[float, ...] | None = None
    best_selection: list[dict[str, Any]] = []
    # 已确认预约必须保留；若其本身不可行则明确报告，不静默丢弃。
    for size in range(len(optional) + 1):
        for combo in combinations(optional, size):
            selection = fixed + list(combo)
            if not feasible(selection):
                continue
            value = score(selection)
            if best is None or value > best:
                best = value
                best_selection = selection

    alternatives: list[dict[str, Any]] = []
    for size in range(len(optional) + 1):
        for combo in combinations(optional, size):
            selection = fixed + list(combo)
            if feasible(selection):
                alternatives.append(
                    {
                        "slots": [x["slot"] for x in selection],
                        "score": list(score(selection)),
                        "optional_slots": [x["slot"] for x in combo],
                    }
                )
    alternatives.sort(key=lambda x: x["score"], reverse=True)
    return {
        "single_crew": single_crew,
        "selected_slots": [x["slot"] for x in best_selection],
        "selected": best_selection,
        "feasible_alternatives": alternatives[:10],
        "locked_confirmed_slots": [x["slot"] for x in fixed],
    }


def plan_vehicle_capacity(
    data: Any,
    store_plans: list[dict[str, Any]],
    as_of: datetime | str,
) -> dict[str, Any]:
    asof = parse_ts(as_of) if isinstance(as_of, str) else as_of
    assert asof is not None
    removal_slots = [
        r for r in data.slots if r["kind"] == "撤架车" and r["closes"] >= asof
    ]
    priority = {p["store"]: i for i, p in enumerate(sorted(store_plans, key=lambda p: p["deadline"]))}
    allocations: list[dict[str, Any]] = []
    warnings: list[str] = []

    for vehicle in removal_slots:
        capacity = int(vehicle["max_units"])
        remaining = capacity
        vehicle_allocations = []
        # 资料没有 region→stores 映射；东组车窗口在示例中作为所有三店的共享资源。
        for plan in sorted(store_plans, key=lambda p: priority[p["store"]]):
            reserve = int(plan["risk_adjusted_removal_reserve"])
            allocated = min(reserve, remaining)
            remaining -= allocated
            confirm_before = parse_ts(plan["confirm_before"])
            vehicle_opens = vehicle["opens"]
            assert confirm_before is not None
            can_use_latest_confirmation = confirm_before <= vehicle_opens
            item = {
                "vehicle_slot": vehicle["slot"],
                "region_or_store": vehicle["store"],
                "store": plan["store"],
                "requested_reserve": reserve,
                "current_old_max": int(plan["stock"]["old_max"]),
                "confirmation_deadline": plan["confirm_before"],
                "latest_confirmation_before_vehicle_open": can_use_latest_confirmation,
                "conservative_reserve_without_late_confirmation": int(plan["stock"]["old_max"]),
                "allocated": allocated,
                "unmet": max(0, reserve - allocated),
                "deadline": plan["deadline"],
            }
            if not can_use_latest_confirmation:
                warnings.append(
                    f"{vehicle['slot']} 早于 {plan['store']} 的最晚确认期限 "
                    f"{plan['confirm_before']}；若要装该车，必须在发车前提前取得盘点/补充确认，"
                    "否则应按当前旧版上限做保守预留或另排车辆"
                )
            vehicle_allocations.append(item)
            if item["unmet"]:
                warnings.append(
                    f"{vehicle['slot']} 容量不足：{plan['store']} 仍有 {item['unmet']} 件风险预留缺口"
                )
        max_old_total = sum(item["current_old_max"] for item in vehicle_allocations)
        allocations.append(
            {
                "slot": vehicle["slot"],
                "region_or_store": vehicle["store"],
                "window": [vehicle["opens"].isoformat(), vehicle["closes"].isoformat()],
                "capacity": capacity,
                "confirmed": bool(vehicle["confirmed"]),
                "remaining_capacity": remaining,
                "undiscounted_current_old_max_total": max_old_total,
                "undiscounted_capacity_shortfall": max(0, max_old_total - capacity),
                "allocations": vehicle_allocations,
            }
        )
        if max_old_total > capacity:
            warnings.append(
                f"{vehicle['slot']} 若完全不确认自然销售，当前旧版上限合计 {max_old_total} 件，"
                f"超过容量 {capacity} 件；必须利用盘点后的确认窗口更新预留，不能直接按最大上限派车"
            )

    constraints = [
        f"{s['slot']} 已确认（{s['opens'].isoformat()}–{s['closes'].isoformat()}，容量 {s['max_units']}），替代安排不能假定其不存在"
        for s in data.slots
        if s["confirmed"]
    ]
    return {"vehicles": allocations, "warnings": warnings, "constraints": constraints}


def transfer_pipeline(data: Any, at: datetime | str) -> list[dict[str, Any]]:
    ts = parse_ts(at) if isinstance(at, str) else at
    assert ts is not None
    ledger = Ledger(data)
    result = []
    for ship in data.transfers:
        result.append(
            {
                "shipment": ship["shipment"],
                "from": ship["from_store"],
                "to": ship["to_store"],
                "qty": ship["qty"],
                "old_range": list(ledger.shipment_old_range(ship["shipment"], ts)),
                "state": shipment_state(ship, ts, ts),
                "handoff_confirmed": bool(ship["known"]),
                "sellable_at_from": ship["picked"] > ts,
                "sellable_at_to": ship["arrived"] is not None and ship["arrived"] <= ts,
            }
        )
    return result
