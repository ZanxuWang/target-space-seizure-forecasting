"""Regenerate all scripted final-paper figures.

Figures 1 and 2 are maintained as editable PowerPoint artwork and are not
generated here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    script = HERE / script_name
    print(f"[paper figures] {script.name}")
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    run("figure3_delta_auc_statistics.py")
    run("figure4_5_roc_performance.py")
    run("figure6_7_generation_quality.py")
    run("figure8_target_space_diagnostic.py")


if __name__ == "__main__":
    main()
