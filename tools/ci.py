"""Local CI: format check, lint and tests. Run with `uv run python tools/ci.py`.

Exits with a non-zero code if any step fails. GitHub automations are not
available for this project, so this is the CI.
"""

import subprocess
import sys

STEPS = [
    ("format", ["ruff", "format", "--check", "."]),
    ("lint", ["ruff", "check", "."]),
    ("tests", ["pytest", "-q"]),
]


def main() -> int:
    failed = []
    for name, args in STEPS:
        print(f"== {name}: {' '.join(args)}", flush=True)
        result = subprocess.run([sys.executable, "-m", *args], check=False)
        if result.returncode != 0:
            failed.append(name)
    print("== CI " + (f"FAILED: {', '.join(failed)}" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
