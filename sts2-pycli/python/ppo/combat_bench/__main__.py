"""`python -m ppo.combat_bench {build,run,report} ...`."""

from __future__ import annotations

import sys

USAGE = "usage: python -m ppo.combat_bench {build,run,report} [args...]"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("build", "run", "report"):
        print(USAGE, file=sys.stderr)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "build":
        from ppo.combat_bench import build

        return build.main(rest)
    if cmd == "run":
        from ppo.combat_bench import run

        return run.main(rest)
    from ppo.combat_bench import report

    return report.main(rest)


if __name__ == "__main__":
    sys.exit(main())
