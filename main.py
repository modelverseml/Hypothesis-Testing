"""Run everything: the worked examples and the power-curves plot.

    python main.py                # run all examples, then save the power plot
    python main.py --no-plot      # examples only, skip the plot
    python main.py --show         # also open the plot in a window
"""

import argparse
import os
import sys

# Make both the package (repo root) and the example helpers importable.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
for path in (REPO_ROOT, EXAMPLES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import power_curves  # noqa: E402  (lives in examples/, added to path above)
import worked_examples  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run all hypothesis-testing demos.")
    parser.add_argument("--no-plot", action="store_true", help="skip the power-curves plot")
    parser.add_argument("--show", action="store_true", help="open the plot in a window")
    args = parser.parse_args()

    # 1. Print every worked example.
    worked_examples.main()

    # 2. Save (and optionally show) the power curves.
    if not args.no_plot:
        fig = power_curves.build_figure()
        fig.savefig(power_curves.OUTPUT_PATH, dpi=150)
        print(f"Saved power curves to {power_curves.OUTPUT_PATH}")
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
