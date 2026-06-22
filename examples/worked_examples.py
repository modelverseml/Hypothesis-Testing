"""Worked examples for every test in the hypothesis_tests package.

These reproduce the examples from README.md and replace the old Jupyter
notebook so they can be run and reviewed as plain Python.

    python examples/worked_examples.py
"""

import numpy as np

from _helpers import add_repo_root_to_path, print_result, print_section

add_repo_root_to_path()

from hypothesis_tests import (  # noqa: E402  (import after the sys.path tweak)
    AnovaFTest,
    ChiSquareTest,
    EffectSize,
    PowerAnalysis,
    TTest,
    ZTest,
)

# Fixed seed so the random samples, and the printed numbers, are reproducible.
RNG_SEED = 42


def z_test_examples():
    """Section 1: one-sample z-test (sigma known)."""
    print_section("1. Z-Test (one-sample)")
    # Claim: Amazon July-2025 sales = $10B. Sample mean = $9B, sigma = 2, n = 40.
    z_test = ZTest(alpha=0.05, tail="two")
    result = z_test.one_sample(sample_mean=9, pop_mean=10, pop_std=2, n=40)
    print_result(result)
    # Sample mean 9 falls below the LCV (~9.38), so we reject H0.


def t_test_examples(rng):
    """Section 2: one-sample, two-sample and paired t-tests."""
    t_test = TTest(alpha=0.05, tail="two")

    print_section("2a. T-Test (one-sample)")
    # Claim: average delivery time = 30 min. Sample of 25 deliveries.
    delivery_times = rng.normal(loc=32, scale=4, size=25)
    print_result(t_test.one_sample(delivery_times, pop_mean=30))

    print_section("2b. T-Test (two-sample independent)")
    # Model A vs model B accuracy over 30 evaluation runs each.
    model_a = rng.normal(loc=0.82, scale=0.03, size=30)
    model_b = rng.normal(loc=0.85, scale=0.03, size=30)
    print_result(t_test.two_sample_independent(model_a, model_b, equal_var=True))

    print_section("2c. T-Test (paired)")
    # Same 20 students measured before and after a training program.
    before = rng.normal(loc=70, scale=5, size=20)
    after = before + rng.normal(loc=3, scale=2, size=20)
    print_result(t_test.paired(before, after))


def chi_square_examples():
    """Section 3: chi-square goodness-of-fit and independence."""
    chi = ChiSquareTest(alpha=0.05)

    print_section("3a. Chi-Square (goodness-of-fit)")
    # 100 coin flips -> 60 heads, 40 tails. Is the coin fair?
    print_result(chi.goodness_of_fit(observed=[60, 40], expected=[50, 50]))

    print_section("3b. Chi-Square (independence)")
    # Is device type independent of the purchase decision?
    #            Purchased   Did-not-purchase
    #   Mobile       120            80
    #   Desktop       60           140
    contingency = np.array([[120, 80], [60, 140]])
    print_result(chi.independence(contingency))


def anova_examples(rng):
    """Section 4: one-way ANOVA and the F-test for variances."""
    anova = AnovaFTest(alpha=0.05)

    print_section("4a. One-way ANOVA")
    # Three training algorithms: do their mean accuracies differ?
    alg_1 = rng.normal(0.80, 0.02, 25)
    alg_2 = rng.normal(0.83, 0.02, 25)
    alg_3 = rng.normal(0.81, 0.02, 25)
    print_result(anova.one_way(alg_1, alg_2, alg_3), width=15)

    print_section("4b. F-Test for equality of variances")
    sample1 = rng.normal(50, 5, 30)
    sample2 = rng.normal(50, 8, 30)
    print_result(anova.variance_ratio(sample1, sample2), width=15)


def effect_size_and_power_examples(rng):
    """Section 5: effect sizes and power analysis."""
    print_section("5. Effect Size & Power Analysis")

    # Same kind of two-sample data as section 2b, plus the section 3b table.
    model_a = rng.normal(loc=0.82, scale=0.03, size=30)
    model_b = rng.normal(loc=0.85, scale=0.03, size=30)
    contingency = np.array([[120, 80], [60, 140]])

    es = EffectSize()
    d = es.cohens_d_two_sample(model_a, model_b)
    print(f"Cohen's d (model A vs B): {d:.3f} ({es.interpret_cohens_d(d)})")
    print(f"Cramer's V (device vs purchase): {es.cramers_v(contingency):.3f}")

    power = PowerAnalysis(alpha=0.05)
    print(f"\nPower at n=30, d=0.5: {power.power_two_sample(0.5, n1=30):.3f}")
    print(
        "Sample size for d=0.5 at 80% power: "
        f"{power.sample_size_two_sample(0.5):.1f} per group"
    )
    print(
        "Minimum detectable effect at n=50, 80% power: "
        f"{power.minimum_detectable_effect(50):.3f}"
    )


def main():
    rng = np.random.default_rng(RNG_SEED)
    z_test_examples()
    t_test_examples(rng)
    chi_square_examples()
    anova_examples(rng)
    effect_size_and_power_examples(rng)
    print()


if __name__ == "__main__":
    main()
