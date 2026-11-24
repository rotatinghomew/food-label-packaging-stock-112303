"""命令行入口：生成 JSON 与 Markdown 分析结果。"""
from __future__ import annotations

import argparse
from pathlib import Path

from .loader import load_dataset
from .report import generate_outputs
from .service import ScenarioService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="换签前旧包装余量与盘点安排分析")
    parser.add_argument("data", type=Path, help="CSV 资料目录，例如 data")
    parser.add_argument("--as-of", default="2026-09-05T08:00", help="HQ 认知时点")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(args.data)
    service = ScenarioService(dataset)
    paths = generate_outputs(service, args.out, args.as_of)
    print(f"已写入 {paths['markdown']}")
    print(f"已写入 {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
