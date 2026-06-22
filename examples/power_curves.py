"""Plot power curves for the two-sample t-test.

Shows how power grows with the per-group sample size for a few effect sizes,
with the usual 0.80 target marked. Replaces the plotting cell from the old
notebook.

    python examples/power_curves.py            # saves images/power_curves.png
    python examples/power_curves.py --show     # also opens a window
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from _helpers import add_repo_root_to_path

add_repo_root_to_path()

from hypothesis_tests import PowerAnalysis  # noqa: E402  (after sys.path tweak)

# Output figure path, relative to the repo root.
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "images",
    "power_curves.png",
)


def build_figure():
    """Build and return the power-curves figure."""
    power = PowerAnalysis(alpha=0.05)
    sample_sizes = np.arange(5, 150, 5)
    effect_sizes = [0.2, 0.5, 0.8]  # small / medium / large (Cohen's d)

    fig, ax = plt.subplots(figsize=(8, 5))
    for d in effect_sizes:
        powers = [power.power_two_sample(d, n1=n) for n in sample_sizes]
        ax.plot(sample_sizes, powers, label=f"d = {d}")

    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="Target power = 0.8")
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Power")
    ax.set_title("Power curves for the two-sample t-test")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot two-sample t-test power curves.")
    parser.add_argument(
        "--show", action="store_true", help="open a window as well as saving the figure"
    )
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved power curves to {OUTPUT_PATH}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
