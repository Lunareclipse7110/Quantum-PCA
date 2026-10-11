"""
main.py
========
Team 4 — Quantum Principal Component Analysis
One-command entry point: runs the full experiment matrix (both datasets,
all 8 configurations, repeated trials), then generates every required plot,
the report tables and the HTML report.

    python main.py               # full run (several hours, see README)
    python main.py --quick       # fast smoke test with tiny settings
    python main.py --repeats 5   # 5 seeds per configuration (assignment suggests 5)

Everything printed is also saved to results/run_log.txt, and the exact
library versions + settings used are saved to results/environment.txt and
results/requirements_used.txt, so the report can quote them accurately.
"""
import argparse
import os
import platform
import sys
from importlib import metadata

from src.generate_report import main as generate_report
from src.make_tables import make_tables
from src.noise_models import NOISE_LEVELS
from src.run_experiments import RESULTS_DIR, run_all
from src.visualize import make_all_plots


class Tee:
    """Send everything printed to both the console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def write_environment(args):
    pkgs = ["qiskit", "qiskit-aer", "scikit-learn", "numpy", "scipy", "pandas", "matplotlib"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = "not installed"

    lines = [
        f"Python: {platform.python_version()}",
        f"OS: {platform.platform()}",
        f"Processor: {platform.processor()}",
    ]
    lines += [f"{p}: {v}" for p, v in versions.items()]
    lines += [
        "",
        "Train/test split: stratified 80:20 (random_state=42)",
        "Seeds: seed = 1000*rep + crc32(f'{dataset}-{exp_id}') % 1000 ; simulator seeded with the same value",
        f"Repeats: {args.repeats}   quick mode: {args.quick}",
        "Shots: 1024   Optimizer: COBYLA (maxiter=150 for full runs)",
        "Noise levels (p1 = 1q depolarizing, p2 = 2q depolarizing, p_ro = readout flip):",
    ]
    lines += [f"  {k}: {v}" for k, v in NOISE_LEVELS.items()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "requirements_used.txt"), "w", encoding="utf-8") as f:
        for p, v in versions.items():
            if v != "not installed":
                f.write(f"{p}=={v}\n")
    print("\n".join(lines))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Tiny settings, just to check everything runs.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = open(os.path.join(RESULTS_DIR, "run_log.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    write_environment(args)

    if args.quick:
        run_all(repeats=1, n_train_samples=4, n_eval_samples=6, maxiter=8)
    else:
        run_all(repeats=args.repeats)

    make_all_plots()
    make_tables()
    generate_report()
    print("\nDone. See results/results.csv, results/results_aggregated.csv, results/tables/*.md,")
    print("results/figures/*.png, results/run_log.txt, results/environment.txt")
    print("and results/Team4_Quantum_PCA_Report.html \u2014 open that one in a browser.")
    sys.stdout = sys.__stdout__
    log_file.close()
