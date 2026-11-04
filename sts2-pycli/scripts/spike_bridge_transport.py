#!/usr/bin/env python3
"""GH #709 transport spike: can the browser-only workbench reach the advisor?

Starts two fake advisor control planes on loopback (realgame_control
with the test suite's FakeClient answering the rewind verbs) — one with
https://your-workbench.example.com allow-listed (the CORS + Private Network
Access opt-in), one with no allow-list (the server before the opt-in) — then
runs web/scripts/bridge-transport-spike.mjs, which loads the REAL deployed
page in Playwright's Chromium and Firefox and fetches both planes from it.
Prints the per-browser results as JSON. Findings and the setup they imply:
docs/browser-workbench.md, "Live-game bridge".

Usage (from sts2-pycli/, needs web/node_modules and the Playwright
browsers)::

    .venv/bin/python scripts/spike_bridge_transport.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from realgame_control import ControlPlane  # noqa: E402
from realgame_livetree import LiveRunTree  # noqa: E402
from test_realgame_control import FakeClient  # noqa: E402  # ty: ignore[unresolved-import]


def plane(allow: list[str]) -> ControlPlane:
    control = ControlPlane(LiveRunTree())
    control.start(0, allow_origins=allow)
    control.set_client(FakeClient(control))
    return control


def main() -> None:
    opt_in = plane(["https://your-workbench.example.com"])
    plain = plane([])
    ports = {"opt-in": opt_in.port, "no-headers": plain.port}
    print("planes", ports, file=sys.stderr)
    env = dict(os.environ, PORTS=json.dumps(ports))
    try:
        r = subprocess.run(
            ["node", "scripts/bridge-transport-spike.mjs"],
            cwd=str(ROOT / "web"),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    finally:
        opt_in.stop()
        plain.stop()
    print(r.stdout)
    print(r.stderr[-2000:], file=sys.stderr)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
