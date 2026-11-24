#!/usr/bin/env python3
"""仓库根目录便捷入口：python3 solve.py [data] [--as-of ...]。"""
from __future__ import annotations

import sys

from relabel.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        args = [args[0], *args[1:]]
    raise SystemExit(main(args or ["data"]))
