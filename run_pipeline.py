"""Run data preparation, training, evaluation, and report generation."""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, *arguments: str):
    subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, check=True)


if __name__ == "__main__":
    run("prepare_hh_rlhf.py")
    run("real_data_experiment.py", "--quiet")
    print("complete: reports/hh_rlhf/report.md")

