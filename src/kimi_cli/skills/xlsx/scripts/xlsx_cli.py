#!/usr/bin/env python3
"""Run the bundled Xlsx CLI on supported systems."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def main() -> int:
    system = platform.system()
    machine = platform.machine().casefold()
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        print(
            "The bundled Xlsx validation/PivotTable CLI currently supports only "
            "Linux x86_64. On macOS or Windows, use Python/openpyxl for workbook "
            "creation and formula checks; do not attempt to execute the Linux binary.",
            file=sys.stderr,
        )
        return 2

    executable = Path(__file__).with_name("Xlsx-linux-x86_64")
    if not executable.is_file():
        print(f"Bundled Xlsx CLI is missing: {executable}", file=sys.stderr)
        return 2
    os.execv(executable, [str(executable), *sys.argv[1:]])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
