"""One-command reproduction of the bundled numerical results and figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(module: str) -> None:
    command = [sys.executable, "-m", module]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only verify data and numerical tables.",
    )
    args = parser.parse_args()
    run("scripts.verify_paper_results")
    if not args.skip_figures:
        run("scripts.paper_figures.make_all_paper_figures")
    print("Paper snapshot reproduction completed successfully.")


if __name__ == "__main__":
    main()
