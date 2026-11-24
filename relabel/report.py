"""把分析结果渲染成可交接的 JSON 和 Markdown。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .service import ScenarioService


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"不可序列化: {type(value)!r}")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def _fmt_range(pair: list[int] | tuple[int, ...]) -> str:
    return f"[{pair[0]}, {pair[1]}]"


def render_markdown(
    report: dict[str, Any],
    trace: dict[str, Any],
    retrospective: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# 换签期限前旧包装余量与盘点安排\n")
    lines.append(f"- 计划认知时点 `as_of={report['as_of']}`")
    lines.append(f"- 商品码：`{report['sku']}`")
    hq = report.get("hq_summary")
    if hq:
        lines.append(
            f"- 总部商品码口径：门店可售 {hq['sellable_total']} 件，在途 {hq['in_transit_total']} 件，"
            f"待验 {hq['inspection_total']} 件，网络合计 {hq['network_total']} 件；"
            f"可售旧版联合区间 {_fmt_range(hq['global_sellable_old_range'])}，"
            f"含在途旧版 {_fmt_range(hq['global_network_old_range'])}。"
        )
    lines.append("- 数量单位：件；旧版区间由跨店同一套可行调拨流求解。\n")

    lines.append("## 1. 门店余量、需求风险与撤下预留\n")
    lines.append(
        "| 门店 | 期限 | 可售总量 | 可确认旧签下限 | 可能旧签上限 | 仅可能多出 | "
        "有效营业h | 需求中位数 | 旧版残余包络 | 最坏口径残余概率 | 风险预留 | 可实盘证明免撤 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for p in report["stores"]:
        s = p["stock"]
        env = p["residual_envelope"]
        lines.append(
            "| {store} | {deadline} | {total} | {lo} | {hi} | {extra} | {hours:.1f} | "
            "{demand:.1f} | {env} | {prob:.1%} | {reserve} | {exempt} |".format(
                store=p["store"],
                deadline=p["deadline"].replace("T", " "),
                total=s["sellable_total"],
                lo=s["confirmed_old"],
                hi=s["old_max"],
                extra=s["possible_extra_old"],
                hours=p["hours_to_deadline_open"],
                demand=p["demand"]["quantiles"]["0.5"],
                env=_fmt_range((env["combined_min"], env["combined_max"])),
                prob=p["risk"]["prob_worst_case_residual"],
                reserve=p["risk_adjusted_removal_reserve"],
                exempt="是" if p["exemption"]["can_certify_zero"] else "否",
            )
        )
    lines.append("")
    lines.append(
        "> “可确认旧签下限”是所有可行流转中都必然存在的旧版数量；“仅可能多出”"
        "对应当前包装分量无法分清的部分。自然卖完预测不能代替 old=0 的本期实盘。\n"
    )

    lines.append("### 不确定性分解\n")
    for p in report["stores"]:
        env = p["residual_envelope"]
        lines.append(
            f"- **{p['store']}**：按历史平均速度卖掉旧版上限约需 "
            f"{p['risk']['mean_rate_hours_to_sell_old_max']:.1f}h，"
            f"期限前有 {p['hours_to_deadline_open']:.1f}h；"
            f"当前存量宽度 {env['current_stock_width']}；"
            f"中位需求下仅存量不清造成 {_fmt_range(env['stock_uncertainty_at_median_demand'])}；"
            f"中间存量下仅需求不确定造成 {_fmt_range(env['demand_uncertainty_at_mid_stock'])}。"
        )
    lines.append("")
    for p in report["stores"]:
        guard = p.get("appointment_guard", {})
        for ship in guard.get("unconfirmed_outgoing", []):
            lines.append(
                f"- `{p['store']}` 已调出但交接未确认：{ship['shipment']} "
                f"{ship['qty']} 件、旧版界 {_fmt_range(ship['old_range'])}；"
                "它不计入当前可售总量，但不能把原店撤签预约按零取消。"
            )
    lines.append("")

    lines.append("## 2. 盘点时段与信息作用\n")
    insights = report.get("insights")
    if insights:
        lines.append(
            f"- 存量不清宽度最大的是 `{insights['largest_stock_ambiguity_slot']}`"
            f"（{insights['largest_stock_ambiguity_store']}）；"
            f"能厘清同批调拨来源的是 `{insights['shared_batch_source_slot']}`。"
        )
    lines.append("| 时段 | 门店 | 时间 | 容量 | 待盘总量 | 当前旧版区间 | 可消除宽度 | 已确认 | 厘清调拨 |")
    lines.append("|---|---|---|---:|---:|---|---:|---|---|")
    for c in report["count_slots"]:
        lines.append(
            f"| {c['slot']} | {c['store']} | {c['opens'][5:16].replace('T', ' ')}–{c['closes'][11:16]} | "
            f"{c['max_units']} | {c['total_to_count']} | {_fmt_range(c['old_range_at_slot'])} | "
            f"{c['stock_ambiguity_units']} | {'是' if c['confirmed'] else '否'} | "
            f"{', '.join(c['resolves_shipment_provenance']) or '—'} |"
        )
    lines.append("")
    for c in report["count_slots"]:
        lines.append(
            f"- `{c['slot']}`：实盘会把旧/新分量从旧版 {_fmt_range(c['old_range_at_slot'])}、"
            f"新版 {_fmt_range(c['new_pack_range_at_slot'])} 收缩为确定值；"
            f"按风险阈值，盘点后需预留撤下量落在 "
            f"{_fmt_range(c['conditional_removal_reserve_range'])}，具体点由实盘给出。"
        )
        for reuse in c.get("downstream_batch_reuse", []):
            lines.append(
                f"  - 来源实盘结果可知且 {reuse['shipment']} 送达/交接确认后，"
                f"{reuse['downstream_store']} 可复用该批次结论，不再为同一批货占用盘点时段。"
            )
    par = report["parallel_count_plan"]
    crew = report["single_crew_count_plan"]
    lines.append(
        f"- 若各店可并行自查，可行选择为 **{', '.join(par['selected_slots'])}**；"
        f"已锁定预约：{', '.join(par['locked_confirmed_slots']) or '无'}。"
    )
    lines.append(
        f"- 若 `deadlines.crew=K1` 表示同一盘点班组，s2 与 s3 时间重叠，"
        f"单班组方案为 **{', '.join(crew['selected_slots'])}**。"
        "期限更早的 B 优先，并可由 B 实盘反推 x2 旧版分量；C 仍需另找时段，不能因预测而免撤。"
    )
    lines.append("")

    lines.append("## 3. 撤架车与确认约束\n")
    for v in report["vehicles"]["vehicles"]:
        lines.append(
            f"- **{v['slot']} / {v['region_or_store']}**："
            f"{v['window'][0][5:16].replace('T', ' ')}–{v['window'][1][11:16]}，"
            f"容量 {v['capacity']}，已确认={v['confirmed']}，风险预留后剩余 {v['remaining_capacity']}，"
            f"不扣减自然销售的当前旧版上限合计 {v['undiscounted_current_old_max_total']}。"
        )
        for a in v["allocations"]:
            lines.append(
                f"  - {a['store']}：风险预留 {a['requested_reserve']}，"
                f"可分配 {a['allocated']}，缺口 {a['unmet']}"
            )
    for warning in report["vehicles"]["warnings"]:
        lines.append(f"- ⚠️ {warning}")
    for constraint in report["vehicles"]["constraints"]:
        lines.append(f"- 现有约束：{constraint}")
    lines.append("")

    bt = report["demand_backtest"]["leave_one_trial_out"]
    roll = report["demand_backtest"]["rolling"]
    lines.append("## 4. 多期完整换签回测\n")
    lines.append(
        f"- 留一期交叉验证：n={bt['n']}，MAE={bt['mae']:.1f}，RMSE={bt['rmse']:.1f}，"
        f"中位绝对百分比误差={bt['median_abs_pct_error']:.1%}，"
        f"90%区间覆盖={bt['coverage_90']:.0%}，80%区间覆盖={bt['coverage_80']:.0%}。"
    )
    lines.append(
        f"- 时间顺序滚动验证：n={roll['n']}，MAE={roll['mae']:.1f}，"
        f"偏差={roll['bias']:.1f}，90%区间覆盖={roll['coverage_90']:.0%}。"
    )
    block = report.get("block_backtest")
    if block:
        lines.append(
            f"- 完整换签块级留一期回测：n={block['n']}，MAE={block['mae']:.1f}，"
            f"原始90%区间覆盖={block['coverage_90']:.0%}；当前需求区间已用其相对误差校准。"
        )
    for store, cal in report.get("old_end_calibration", {}).items():
        lines.append(
            f"- `actual_old_end` 期末校准 {store}：LOO MAE={cal['leave_one_out_mae']:.1f}，"
            f"80%经验区间覆盖={cal['coverage_80_empirical']:.0%}，"
            f"期中位数={cal['quantiles']['0.5']:.1f}。"
        )
    lines.append("- 短营业日只按 `opens→available_until` 计算暴露时长，不把后半天补为零需求。\n")

    lines.append("## 5. 同批货轨迹、盘点作用与实际撤下\n")
    for effect in trace.get("count_effects", []):
        lines.append(
            f"- 实盘 {effect['count']}（{effect['store']}，结果可知 {effect['known_at'][5:16].replace('T', ' ')}）："
            f"旧版区间 {_fmt_range(effect['before_range'])} → {_fmt_range(effect['after_range'])}，"
            f"消除宽度约 {effect['ambiguity_reduced_by']} 件。"
        )
    lines.append("")
    lines.append("| 时间 | 环节 | 单号 | 位置 | 数量 | 之后可售总量 | 旧版区间 |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for e in trace["entries"]:
        location = e.get("store") or f"{e.get('from_store', '')}→{e.get('to_store', '')}"
        qty = e.get("qty", "")
        total = e.get("sellable_total_after", "")
        rng = e.get("old_range_after") or e.get("shipment_old_range") or e.get("old_bounds_recorded") or ""
        rng_text = _fmt_range(rng) if isinstance(rng, list) and rng else ""
        lines.append(
            f"| {e['time'][5:16].replace('T', ' ')} | {e['kind']} | {e['ref']} | "
            f"{location} | {qty} | {total} | {rng_text} |"
        )
    lines.append("")
    if trace["actual_removals"]:
        lines.append("实际撤下记录：")
        for r in trace["actual_removals"]:
            lines.append(
                f"- {r['event']} {r['store']} at {r['at']}：撤下 {r['old_qty']} 件旧签"
                f"（HQ 可知时间 {r['known_at']}）。晚到销售汇总只能回算该时点以前的预测误差，"
                "不能把已撤下旧签加回当前可售量。"
            )
    if retrospective:
        lines.append("")
        lines.append("### 晚到单据回算\n")
        for event in retrospective.get("late_events", []):
            lines.append(
                f"- {event['event']}（{event['store']} {event['kind']}，{event['qty']} 件）"
                f"业务时间 {event['event_at'][5:16].replace('T', ' ')}、"
                f"HQ 可知 {event['known_at'][5:16].replace('T', ' ')}；"
                f"按业务时间补算后区间 {_fmt_range(event['range_after_event'])}，"
                "但不回改决策时点状态，也不恢复已撤下旧签。"
            )
        for review in retrospective.get("stores", []):
            error = review["demand_forecast_error_median_minus_observed"]
            error_text = "资料不足以完整回算" if error is None else f"中位误差 {error:.1f}"
            lines.append(
                f"- {review['store']} 期限回算：决策时需求中位 "
                f"{review['forecast_demand_median_at_decision']:.1f}，"
                f"窗口内后验收银 {review['observed_window_sales_event_time']}，"
                f"{error_text}，"
                f"实际撤下 {review['actual_removed_by_deadline']}，"
                f"撤后旧版范围 {_fmt_range(review['old_range_after_chronological_events_and_removal'])}。"
            )
        lines.append("")
    return "\n".join(lines)


def generate_outputs(
    service: ScenarioService,
    out_dir: str | Path,
    as_of: str = "2026-09-05T08:00",
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = service.planning_report(as_of)
    trace = service.batch_trace(report["sku"], as_of=as_of)
    retrospective = service.retrospective_review(as_of)
    payload = {"report": report, "trace": trace, "retrospective": retrospective}
    json_path = out / "analysis.json"
    md_path = out / "report.md"
    write_json(payload, json_path)
    md_path.write_text(render_markdown(report, trace, retrospective), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
