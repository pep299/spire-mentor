"""`python -m ppo.combat_ctx <command>`: see the package docstring."""

from __future__ import annotations

import argparse
import sys

from . import data, gen, metrics, play, power, referee, train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.combat_ctx", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gen.add_parser(sub)
    power.add_parser(sub)
    data.add_parser(sub)
    train.add_parser(sub)
    train.add_crosscheck_parser(sub)
    train.add_predict_parser(sub)
    metrics.add_parser(sub)
    referee.add_parser(sub)
    play.add_parser(sub)
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
