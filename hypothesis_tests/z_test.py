"""Z-tests for means when the population standard deviation is known.

Use a z-test when sigma is known (or the sample is large enough that the
sample standard deviation is a safe substitute). If sigma is unknown and the
sample is small, use the t-test in t_test.py instead.
"""

from typing import Dict, Optional

import numpy as np
from scipy import stats

# Allowed alternative hypotheses:
#   "two"   -> mu != mu0
#   "left"  -> mu <  mu0
#   "right" -> mu >  mu0
VALID_TAILS = ("two", "left", "right")


class ZTest:
    """One- and two-sample z-tests using the standard normal distribution."""

    def __init__(self, alpha=0.05, tail="two"):
        if tail not in VALID_TAILS:
            raise ValueError(f"tail must be one of {VALID_TAILS}, got {tail!r}")
        self.alpha = alpha
        self.tail = tail

    def one_sample(self, sample_mean, pop_mean, pop_std, n) -> Dict[str, Optional[float]]:
        """Test a sample mean against a hypothesised population mean.

        Args:
            sample_mean: observed sample mean (x-bar).
            pop_mean: hypothesised population mean under H0 (mu0).
            pop_std: known population standard deviation (sigma).
            n: sample size.

        Returns a dict with the z-score, p-value, lower/upper critical values
        (the untested side is None), the standard error and the decision.
        """
        se = pop_std / np.sqrt(n)            # standard error of the mean
        z = (sample_mean - pop_mean) / se

        if self.tail == "two":
            # Reject when the sample mean falls outside [lcv, ucv].
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
            lcv = pop_mean - z_crit * se
            ucv = pop_mean + z_crit * se
            reject = (sample_mean < lcv) or (sample_mean > ucv)
        elif self.tail == "left":
            # Only a low sample mean counts against H0.
            p_value = stats.norm.cdf(z)
            z_crit = stats.norm.ppf(self.alpha)   # negative
            lcv = pop_mean + z_crit * se
            ucv = None
            reject = sample_mean < lcv
        else:  # right
            # Only a high sample mean counts against H0.
            p_value = 1 - stats.norm.cdf(z)
            z_crit = stats.norm.ppf(1 - self.alpha)
            lcv = None
            ucv = pop_mean + z_crit * se
            reject = sample_mean > ucv

        return {
            "z_score": z,
            "p_value": p_value,
            "lcv": lcv,
            "ucv": ucv,
            "standard_error": se,
            "reject_null": bool(reject),
        }

    def two_sample(self, mean1, mean2, std1, std2, n1, n2) -> Dict[str, float]:
        """Test whether two independent population means differ.

        Takes summary statistics for each group. The decision uses the p-value,
        since the two-sample statistic is centred at zero under H0.
        """
        # Standard error of the difference of two independent means.
        se = np.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
        z = (mean1 - mean2) / se

        if self.tail == "two":
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        elif self.tail == "left":
            p_value = stats.norm.cdf(z)
        else:  # right
            p_value = 1 - stats.norm.cdf(z)

        return {
            "z_score": z,
            "p_value": p_value,
            "standard_error": se,
            "reject_null": bool(p_value < self.alpha),
        }
