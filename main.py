"""
main.py
========
Team 4 — Quantum Principal Component Analysis
One-command entry point: runs the full experiment matrix (both datasets,
all 8 configurations, repeated trials), then generates every required plot.

    python main.py               # full run (can take a while, see README)
    python main.py --quick       # fast smoke test with tiny settings
    python main.py --repeats 1   # one trial per configuration instead of 3
"""
import argparse

from src.run_experiments import run_all
from src.visualize import make_all_plots
from src.generate_report import main as generate_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Tiny settings, just to check everything runs.")
    args = parser.parse_args()

    if args.quick:
        run_all(repeats=1, n_train_samples=4, n_eval_samples=6, maxiter=8)
    else:
        run_all(repeats=args.repeats)

    make_all_plots()
    generate_report()
    print("\nDone. See results/results.csv, results/results_aggregated.csv, results/figures/*.png")
    print("and results/Team4_Quantum_PCA_Report.html \u2014 open that one in a browser.")
