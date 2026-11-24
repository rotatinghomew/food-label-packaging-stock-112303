"""Compatibility entry point. Prefer ``python3 solve.py`` or ``python3 -m relabel``."""
from analysis import *  # noqa: F401,F403
from analysis import analyze  # explicit export

if __name__ == "__main__":
    from relabel.cli import main

    raise SystemExit(main(["data"]))
