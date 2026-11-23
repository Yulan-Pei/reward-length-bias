"""Prepare HH-RLHF and run the frozen MiniLM reward-model experiment."""
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
environment = dict(os.environ, USE_TF="0")


def run(script: str, *arguments: str):
    subprocess.run(
        [sys.executable, str(ROOT / script), *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    run("prepare_hh_rlhf.py")
    run("transformer_experiment.py")
