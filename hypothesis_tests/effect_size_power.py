"""Effect sizes and statistical power.

A p-value says whether an effect is unlikely under H0, not how big it is. With
a large sample even tiny differences turn "significant", so report an effect
size alongside every test. Power analysis links four quantities (alpha, effect
size, sample size, power); fix any three and the fourth follows, which is what
sample-size planning relies on.

    EffectSize     - Cohen's d / h, Cramer's V, and a size label.
    PowerAnalysis  - power, required sample size and minimum detectable effect
                     for t-tests (uses statsmodels).
"""

import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower


class EffectSize:
    """Standardised effect-size measures for the tests in this package."""

    def cohens_d_one_sample(self, sample, pop_mean):
        """Cohen's d for a one-sample test: (x_bar - mu0) / s."""
        sample = np.asarray(sample, dtype=float)
        return (sample.mean() - pop_mean) / sample.std(ddof=1)

    def cohens_d_two_sample(self, sample1, sample2):
        """Cohen's d for two independent samples, using the pooled std dev."""
        s1 = np.asarray(sample1, dtype=float)
        s2 = np.asarray(sample2, dtype=float)
        n1, n2 = len(s1), len(s2)
        pooled_std = np.sqrt(
            ((n1 - 1) * s1.var(ddof=1) + (n2 - 1) * s2.var(ddof=1)) / (n1 + n2 - 2)
        )
        return (s1.mean() - s2.mean()) / pooled_std

    def cohens_h(self, p1, p2):
        """Cohen's h for the difference between two proportions."""
        return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

    def cramers_v(self, contingency_table):
        """Cramer's V: effect size for a chi-square test of independence."""
        table = np.asarray(contingency_table, dtype=float)
        row_totals = table.sum(axis=1, keepdims=True)
        col_totals = table.sum(axis=0, keepdims=True)
        n = table.sum()
        expected = row_totals @ col_totals / n
        chi2 = np.sum((table - expected) ** 2 / expected)
        k = min(table.shape) - 1
        return np.sqrt(chi2 / (n * k))

    def interpret_cohens_d(self, d):
        """Label a Cohen's d magnitude using Cohen's conventional cut-offs."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"


class PowerAnalysis:
    """Power, sample-size and minimum-detectable-effect helpers for t-tests."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def power_one_sample(self, effect_size, n):
        """Power of a one-sample t-test at the given effect size and n."""
        return TTestPower().power(effect_size=effect_size, nobs=n, alpha=self.alpha)

    def power_two_sample(self, effect_size, n1, ratio=1.0):
        """Power of a two-sample t-test (ratio = n2 / n1, 1.0 for equal groups)."""
        return TTestIndPower().power(
            effect_size=effect_size, nobs1=n1, alpha=self.alpha, ratio=ratio
        )

    def sample_size_one_sample(self, effect_size, power=0.8):
        """Sample size a one-sample t-test needs to reach the target power."""
        return TTestPower().solve_power(
            effect_size=effect_size, power=power, alpha=self.alpha
        )

    def sample_size_two_sample(self, effect_size, power=0.8, ratio=1.0):
        """Per-group sample size a two-sample t-test needs to reach the power."""
        return TTestIndPower().solve_power(
            effect_size=effect_size, power=power, alpha=self.alpha, ratio=ratio
        )

    def minimum_detectable_effect(self, n, power=0.8):
        """Smallest effect a one-sample t-test can detect at fixed n and power."""
        return TTestPower().solve_power(nobs=n, power=power, alpha=self.alpha)
