"""基于历史可用时长的需求速度、回测误差与残旧版风险区间。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Sequence
import zlib

import numpy as np

from .loader import Dataset
from .timeutil import daily_open_intervals, exposure_hours, parse_hhmm, parse_ts

FULL_DAY_HOURS = 10.0


@dataclass(frozen=True)
class DemandForecast:
    store: str
    start: datetime
    end: datetime
    open_hours: float
    mean_units_per_hour: float
    mean_demand: float
    quantiles: dict[float, float]
    samples: np.ndarray | None = None

    def q(self, p: float) -> float:
        if p in self.quantiles:
            return self.quantiles[p]
        raise KeyError(p)


class DemandModel:
    """经验自助法需求模型。

    历史短日按 ``observed * FULL_DAY_HOURS / exposure_hours`` 归一化为完整
    营业日需求，避免把提前无货/不可售之后的时段当成零需求训练样本。
    """

    def __init__(self, data: Dataset, seed: int = 20260905, n_samples: int = 40000):
        self.data = data
        self.seed = seed
        self.n_samples = n_samples
        self._rows = self._prepare_rows()

    def _prepare_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in self.data.historical:
            item = dict(raw)
            item["exposure_hours"] = exposure_hours(raw["opens"], raw["available_until"])
            item["clock_hours"] = exposure_hours(raw["opens"], raw["closes"])
            item["full_day_equivalent"] = (
                raw["observed"] * FULL_DAY_HOURS / item["exposure_hours"]
            )
            item["trial_index"] = int(str(raw["trial"]).lstrip("vV"))
            rows.append(item)
        return rows

    def training_rows(
        self,
        store: str,
        exclude_trial: str | None = None,
        before_trial_index: int | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for row in self._rows:
            if row["store"] != store:
                continue
            if exclude_trial is not None and row["trial"] == exclude_trial:
                continue
            if before_trial_index is not None and row["trial_index"] >= before_trial_index:
                continue
            result.append(row)
        return result

    def rate(self, store: str) -> float:
        rows = self.training_rows(store)
        return float(sum(r["observed"] for r in rows) / sum(r["exposure_hours"] for r in rows))

    def _simulate_hours(
        self,
        intervals_hours: Sequence[float],
        rows: Sequence[dict[str, Any]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not rows:
            raise ValueError("没有可用于预测的历史记录")
        full_values = np.array([r["full_day_equivalent"] for r in rows], dtype=float)
        total = np.zeros(self.n_samples, dtype=float)
        for hours in intervals_hours:
            draws = rng.choice(full_values, size=self.n_samples, replace=True)
            total += draws * hours / FULL_DAY_HOURS
        # 需求件数为非负整数；保留连续分位数用于稳健比较，同时给出整数样本。
        return total

    def forecast(
        self,
        store: str,
        start: datetime | str,
        end: datetime | str,
        opens: time | str = time(8, 0),
        closes: time | str = time(18, 0),
        quantiles: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95),
        samples: bool = False,
    ) -> DemandForecast:
        st = parse_ts(start) if isinstance(start, str) else start
        en = parse_ts(end) if isinstance(end, str) else end
        assert st is not None and en is not None
        open_t = parse_hhmm(opens if isinstance(opens, str) else opens.strftime("%H:%M"), time(8, 0))
        close_t = parse_hhmm(closes if isinstance(closes, str) else closes.strftime("%H:%M"), time(18, 0))
        intervals = daily_open_intervals(st, en, open_t, close_t)
        hours = [interval.hours for interval in intervals]
        total_hours = sum(hours)
        rows = self.training_rows(store)
        seed_text = f"{store}|{st.isoformat()}|{en.isoformat()}".encode()
        seed_offset = zlib.crc32(seed_text) % 100000
        rng = np.random.default_rng(self.seed + seed_offset)
        sim = self._simulate_hours(hours, rows, rng)
        qvals = {p: float(np.quantile(sim, p)) for p in quantiles}
        rate = sum(r["observed"] for r in rows) / sum(r["exposure_hours"] for r in rows)
        return DemandForecast(
            store=store,
            start=st,
            end=en,
            open_hours=total_hours,
            mean_units_per_hour=float(rate),
            mean_demand=float(sim.mean()),
            quantiles=qvals,
            samples=sim if samples else None,
        )

    def block_rates(self, store: str) -> np.ndarray:
        """每个完整 trial/store 的“实销件数 / 有效暴露小时”。"""
        totals: dict[str, float] = {}
        hours: dict[str, float] = {}
        for row in self.training_rows(store):
            trial = row["trial"]
            totals[trial] = totals.get(trial, 0.0) + float(row["observed"])
            hours[trial] = hours.get(trial, 0.0) + float(row["exposure_hours"])
        return np.array([totals[t] / hours[t] for t in totals], dtype=float)

    def block_forecast(
        self,
        store: str,
        start: datetime | str,
        end: datetime | str,
        opens: time | str = time(8, 0),
        closes: time | str = time(18, 0),
        quantiles: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95),
    ) -> DemandForecast:
        """按完整换签期的块级速度外推到目标有效营业小时数。"""
        st = parse_ts(start) if isinstance(start, str) else start
        en = parse_ts(end) if isinstance(end, str) else end
        assert st is not None and en is not None
        open_t = parse_hhmm(opens if isinstance(opens, str) else opens.strftime("%H:%M"), time(8, 0))
        close_t = parse_hhmm(closes if isinstance(closes, str) else closes.strftime("%H:%M"), time(18, 0))
        total_hours = sum(i.hours for i in daily_open_intervals(st, en, open_t, close_t))
        rates = self.block_rates(store)
        sim = np.repeat(rates * total_hours, max(1, self.n_samples // max(1, len(rates))))
        qvals = {p: float(np.quantile(rates * total_hours, p)) for p in quantiles}
        all_rows = self.training_rows(store)
        rate = sum(r["observed"] for r in all_rows) / sum(r["exposure_hours"] for r in all_rows)
        return DemandForecast(store, st, en, total_hours, rate, float((rates * total_hours).mean()), qvals, sim)

    def block_backtest(self) -> dict[str, Any]:
        """用完整 trial 块级速度做留一期回测，并保存相对误差分布。"""
        items = []
        error_quantiles: dict[str, dict[str, float]] = {}
        for store in sorted({r["store"] for r in self._rows}):
            rates = self.block_rates(store)
            totals = rates * 36.0  # 合成资料中每个 trial/store 的有效暴露均为 36h
            relative_errors = []
            for i, actual in enumerate(totals):
                train = np.delete(totals, i)
                lo, p50, hi90, hi95 = np.quantile(train, [0.05, 0.5, 0.90, 0.95])
                pred = float(np.median(train))
                rel = (float(actual) - pred) / pred
                relative_errors.append(rel)
                items.append(
                    {
                        "store": store,
                        "trial": f"v{i + 1}",
                        "actual": float(actual),
                        "pred_p05": float(lo),
                        "pred_p50": float(p50),
                        "pred_p90": float(hi90),
                        "pred_p95": float(hi95),
                        "relative_error": float(rel),
                        "abs_error": abs(float(actual) - pred),
                        "covered_90": bool(lo <= actual <= hi95),
                    }
                )
            arr = np.array(relative_errors)
            error_quantiles[store] = {
                str(p): float(np.quantile(arr, p))
                for p in (0.05, 0.10, 0.15, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95)
            }
        abs_errors = np.array([x["abs_error"] for x in items])
        return {
            "n": len(items),
            "mae": float(abs_errors.mean()),
            "coverage_90": float(np.mean([x["covered_90"] for x in items])),
            "error_quantiles": error_quantiles,
            "items": items,
        }

    def calibrated_block_forecast(
        self,
        store: str,
        start: datetime | str,
        end: datetime | str,
        quantiles: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95),
    ) -> DemandForecast:
        """用多期回测相对误差校准块级需求区间。"""
        base = self.block_forecast(store, start, end, quantiles=quantiles)
        backtest = self.block_backtest()
        rel = np.array(
            [x["relative_error"] for x in backtest["items"] if x["store"] == store],
            dtype=float,
        )
        center = base.quantiles[0.5]
        rng = np.random.default_rng(self.seed + zlib.crc32(store.encode()) % 10000)
        error_draws = rng.choice(rel, size=self.n_samples, replace=True)
        samples = np.maximum(0.0, center * (1.0 + error_draws))
        qvals = {p: float(np.quantile(samples, p)) for p in quantiles}
        return DemandForecast(
            store=store,
            start=base.start,
            end=base.end,
            open_hours=base.open_hours,
            mean_units_per_hour=base.mean_units_per_hour,
            mean_demand=float(samples.mean()),
            quantiles=qvals,
            samples=samples,
        )

    def backtest_trials(self, min_train_trials: int = 4) -> dict[str, Any]:
        """逐期完整换签回测。

        rolling 使用更早 trial 训练；leave_one_trial_out 给出全部 16 期的
        交叉验证结果。预测目标是该 trial 各日 ``observed`` 之和。
        """
        trials = sorted({r["trial"] for r in self._rows}, key=lambda x: int(x[1:]))
        stores = sorted({r["store"] for r in self._rows})
        rolling: list[dict[str, Any]] = []
        loo: list[dict[str, Any]] = []
        rng = np.random.default_rng(self.seed + 777)

        for store in stores:
            for trial in trials:
                actual_rows = [r for r in self._rows if r["store"] == store and r["trial"] == trial]
                actual = float(sum(r["observed"] for r in actual_rows))
                hours = [r["exposure_hours"] for r in actual_rows]
                idx = int(trial[1:])

                for mode, train in (
                    ("loo", self.training_rows(store, exclude_trial=trial)),
                    ("rolling", self.training_rows(store, before_trial_index=idx)),
                ):
                    if mode == "rolling" and len({r["trial"] for r in train}) < min_train_trials:
                        continue
                    sim = self._simulate_hours(hours, train, rng)
                    q05, q10, q50, q90, q95 = np.quantile(sim, [0.05, 0.10, 0.5, 0.90, 0.95])
                    item = {
                        "store": store,
                        "trial": trial,
                        "actual": actual,
                        "pred_p05": float(q05),
                        "pred_p10": float(q10),
                        "pred_p50": float(q50),
                        "pred_p90": float(q90),
                        "pred_p95": float(q95),
                        "error": float(q50 - actual),
                        "abs_error": float(abs(q50 - actual)),
                        "pct_error": float((q50 - actual) / actual),
                        "covered_90": bool(q05 <= actual <= q95),
                        "covered_80": bool(q10 <= actual <= q90),
                    }
                    (loo if mode == "loo" else rolling).append(item)

        def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
            if not items:
                return {"n": 0}
            errors = np.array([x["error"] for x in items])
            abs_errors = np.array([x["abs_error"] for x in items])
            pct = np.array([abs(x["pct_error"]) for x in items])
            return {
                "n": len(items),
                "mae": float(abs_errors.mean()),
                "rmse": float(np.sqrt((errors**2).mean())),
                "median_abs_pct_error": float(np.median(pct)),
                "bias": float(errors.mean()),
                "coverage_90": float(np.mean([x["covered_90"] for x in items])),
                "coverage_80": float(np.mean([x["covered_80"] for x in items])),
                "items": items,
            }

        return {"rolling": summarize(rolling), "leave_one_trial_out": summarize(loo)}

    def old_end_calibration(self) -> dict[str, Any]:
        """使用 historical.actual_old_end 校准完整换签期末旧版残余。

        资料没有提供历史各日期初存量，因此这里只做经验端点误差校准；
        当前批次的存量因果范围仍由 Ledger 的实盘可行流负责。
        """
        result: dict[str, Any] = {}
        stores = sorted({r["store"] for r in self._rows})
        for store in stores:
            endpoints = [
                r
                for trial in {x["trial"] for x in self._rows if x["store"] == store}
                for r in [
                    sorted(
                        [x for x in self._rows if x["store"] == store and x["trial"] == trial],
                        key=lambda x: x["day"],
                    )[-1]
                ]
            ]
            values = np.array([r["actual_old_end"] for r in endpoints], dtype=float)
            errors = []
            covered = []
            for i, actual_row in enumerate(endpoints):
                train = np.delete(values, i)
                lo, med, hi = np.quantile(train, [0.1, 0.5, 0.9])
                actual = actual_row["actual_old_end"]
                errors.append(abs(med - actual))
                covered.append(bool(lo <= actual <= hi))
            result[store] = {
                "n_endpoints": len(values),
                "mean_actual_old_end": float(values.mean()),
                "quantiles": {
                    str(p): float(np.quantile(values, p))
                    for p in (0.1, 0.25, 0.5, 0.75, 0.9)
                },
                "leave_one_out_mae": float(np.mean(errors)),
                "coverage_80_empirical": float(np.mean(covered)),
            }
        return result


def residual_envelope(
    total: int,
    old_min: int,
    old_max: int,
    demand_low: float,
    demand_median: float,
    demand_high: float,
) -> dict[str, Any]:
    """分解当前存量不确定性和未来需求不确定性。

    给定总存量 T、旧版范围 [L,U] 和需求分位 [Dlo,Dmed,Dhi]：
    所有可行旧版残余的外包络为
    ``[max(0,L-Dhi), min(U,max(0,T-Dlo))]``。
    """
    total = max(0, int(total))
    old_min = max(0, int(old_min))
    old_max = max(old_min, int(old_max))

    total_range = (
        max(0, old_min - np.ceil(demand_high)),
        min(old_max, max(0, total - np.floor(demand_low))),
    )
    stock_only = (
        max(0, old_min - demand_median),
        min(old_max, max(0, total - demand_median)),
    )
    mid_old = (old_min + old_max) / 2.0
    demand_only = (
        max(0, mid_old - demand_high),
        min(mid_old, max(0, total - demand_low)),
    )
    return {
        "combined_min": int(np.floor(total_range[0])),
        "combined_max": int(np.ceil(total_range[1])),
        "stock_uncertainty_at_median_demand": [
            int(np.floor(stock_only[0])),
            int(np.ceil(stock_only[1])),
        ],
        "demand_uncertainty_at_mid_stock": [
            int(np.floor(demand_only[0])),
            int(np.ceil(demand_only[1])),
        ],
        "current_stock_width": old_max - old_min,
    }


def residual_risk_samples(
    total: int,
    old_max: int,
    demand_samples: np.ndarray,
) -> dict[str, float | int]:
    """按最不利于撤架的旧版上限计算残余风险和安全预留量。"""
    residual = np.minimum(old_max, np.maximum(0, total - demand_samples))
    alpha_quantiles = {
        "p50_residual": float(np.quantile(residual, 0.5)),
        "p85_residual": float(np.quantile(residual, 0.85)),
        "p90_residual": float(np.quantile(residual, 0.90)),
        "p95_residual": float(np.quantile(residual, 0.95)),
        "mean_residual": float(residual.mean()),
        "prob_any_residual": float(np.mean(residual > 0)),
    }
    return alpha_quantiles


def safe_reserve_for_probability(
    total: int,
    old_max: int,
    demand_samples: np.ndarray,
    max_residual_probability: float,
) -> int:
    """求最小预留 R，使最坏口径下 P(旧版残余 > R) ≤ alpha。"""
    residual = np.minimum(old_max, np.maximum(0, total - demand_samples))
    alpha = min(max(max_residual_probability, 0.001), 0.5)
    target_coverage = 1.0 - alpha
    reserve = int(np.ceil(np.quantile(residual, target_coverage, method="higher")))
    # 经验分位数在离散结点上可能仍有略高于 alpha 的越界率；向上搜索到满足条件。
    upper = int(residual.max())
    for candidate in range(reserve, upper + 1):
        if float(np.mean(residual > candidate)) <= max_residual_probability + 1e-12:
            return candidate
    return upper
