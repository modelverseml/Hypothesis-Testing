"""Student's t-tests for means when the population standard deviation is unknown.

The t-test uses the sample standard deviation in place of sigma and the
t-distribution, whose heavier tails account for that extra uncertainty.
Covers the one-sample, two-sample (pooled or Welch) and paired cases.
"""

from typing import Dict

import numpy as np
from scipy import stats

VALID_TAILS = ("two", "left", "right")


class TTest:
    """One-sample, two-sample and paired t-tests."""

    def __init__(self, alpha=0.05, tail="two"):
        if tail not in VALID_TAILS:
            raise ValueError(f"tail must be one of {VALID_TAILS}, got {tail!r}")
        self.alpha = alpha
        self.tail = tail

    def _p_value(self, t_stat, df):
        """p-value for a t statistic given df and the configured tail."""
        if self.tail == "two":
            return 2 * (1 - stats.t.cdf(abs(t_stat), df))
        if self.tail == "left":
            return stats.t.cdf(t_stat, df)
        return 1 - stats.t.cdf(t_stat, df)  # right

    def _critical(self, df):
        """Critical t value for the configured tail and alpha."""
        if self.tail == "two":
            return stats.t.ppf(1 - self.alpha / 2, df)
        return stats.t.ppf(1 - self.alpha, df)

    def one_sample(self, sample, pop_mean) -> Dict[str, float]:
        """Test a sample mean against a hypothesised population mean (mu0)."""
        sample = np.asarray(sample, dtype=float)
        n = len(sample)
        x_bar = sample.mean()
        s = sample.std(ddof=1)        # sample std dev, divides by n - 1
        se = s / np.sqrt(n)
        t_stat = (x_bar - pop_mean) / se
        df = n - 1

        p_value = self._p_value(t_stat, df)
        return {
            "t_statistic": t_stat,
            "df": df,
            "p_value": p_value,
            "t_critical": self._critical(df),
            "standard_error": se,
            "reject_null": bool(p_value < self.alpha),
        }

    def two_sample_independent(self, sample1, sample2, equal_var=True) -> Dict[str, float]:
        """Compare the means of two independent samples.

        With equal_var=True this is Student's pooled t-test; with
        equal_var=False it is Welch's test (no pooling, Satterthwaite df).
        """
        s1 = np.asarray(sample1, dtype=float)
        s2 = np.asarray(sample2, dtype=float)
        n1, n2 = len(s1), len(s2)
        m1, m2 = s1.mean(), s2.mean()
        v1, v2 = s1.var(ddof=1), s2.var(ddof=1)

        if equal_var:
            # Pool the two variances and use the combined df.
            pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            # Welch: keep variances separate, approximate the df.
            se = np.sqrt(v1 / n1 + v2 / n2)
            df = (v1 / n1 + v2 / n2) ** 2 / (
                (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            )

        t_stat = (m1 - m2) / se
        p_value = self._p_value(t_stat, df)

        return {
            "t_statistic": t_stat,
            "df": df,
            "p_value": p_value,
            "t_critical": self._critical(df),
            "standard_error": se,
            "reject_null": bool(p_value < self.alpha),
        }

    def paired(self, before, after) -> Dict[str, float]:
        """Paired t-test on two measurements of the same units.

        Runs a one-sample t-test on the within-pair differences against 0.
        Pairing cancels out between-subject variation, so it is usually more
        powerful than the independent test when the data are truly paired.
        """
        before = np.asarray(before, dtype=float)
        after = np.asarray(after, dtype=float)
        diffs = after - before
        return self.one_sample(diffs, pop_mean=0)
