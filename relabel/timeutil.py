"""时间窗口与有效销售时长工具。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, Iterator, Sequence

ISO_FMT = "%Y-%m-%dT%H:%M"


def parse_ts(value: str | datetime | None) -> datetime | None:
    """解析资料中的 ISO 时间；空值返回 None。"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    # CSV 中秒通常为 00；fromisoformat 能兼容日期与秒。
    return datetime.fromisoformat(text)


def parse_hhmm(value: str | time | None, default: time) -> time:
    if value is None or value == "":
        return default
    if isinstance(value, time):
        return value
    return time.fromisoformat(str(value))


def hours_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 3600.0)


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime

    @property
    def hours(self) -> float:
        return hours_between(self.start, self.end)


def daily_open_intervals(
    start: datetime,
    end: datetime,
    opens: time = time(8, 0),
    closes: time = time(18, 0),
) -> list[Interval]:
    """返回 [start, end] 与每日营业区间的交集。

    历史资料的营业区间为 08:00-18:00。调用方可以传入门店自己的营业时间。
    """
    if end < start:
        return []
    result: list[Interval] = []
    day = start.date()
    last_day = end.date()
    while day <= last_day:
        day_start = datetime.combine(day, opens)
        day_end = datetime.combine(day, closes)
        lo = max(start, day_start)
        hi = min(end, day_end)
        if hi > lo:
            result.append(Interval(lo, hi))
        day += timedelta(days=1)
    return result


def total_open_hours(
    start: datetime,
    end: datetime,
    opens: time = time(8, 0),
    closes: time = time(18, 0),
) -> float:
    return sum(item.hours for item in daily_open_intervals(start, end, opens, closes))


def exposure_hours(opens: str, available_until: str) -> float:
    """历史记录中实际可产生收银的暴露时长。

    available_until 早于 closes 时，只按 opens→available_until 计暴露，
    绝不把 unavailable 的后半日补成零需求样本。
    """
    o = parse_hhmm(opens, time(8, 0))
    a = parse_hhmm(available_until, time(18, 0))
    d0 = datetime(2000, 1, 1, o.hour, o.minute)
    d1 = datetime(2000, 1, 1, a.hour, a.minute)
    return hours_between(d0, d1)
