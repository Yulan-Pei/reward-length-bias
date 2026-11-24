"""One-command Nectar download, embedding, training, and report pipeline."""
from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, "prepare_nectar.py"], check=True)
    subprocess.run([sys.executable, "nectar_listwise_experiment.py", "--local-files-only"], check=True)
    subprocess.run([sys.executable, "odin_pl_sweep.py"], check=True)


if __name__ == "__main__":
    main()
