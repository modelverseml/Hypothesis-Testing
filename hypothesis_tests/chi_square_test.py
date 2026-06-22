"""Chi-square tests for categorical data.

Both tests use the same statistic, chi2 = sum((O - E)^2 / E), and reject when
the gap between observed and expected counts is large:

    goodness_of_fit  - does one variable follow an expected distribution?
    independence     - are two categorical variables associated?
"""

from typing import Dict

import numpy as np
from scipy import stats


class ChiSquareTest:
    """Chi-square goodness-of-fit and independence tests."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def goodness_of_fit(self, observed, expected) -> Dict[str, float]:
        """Test whether observed counts match an expected distribution.

        observed and expected must have the same length and total.
        """
        observed = np.asarray(observed, dtype=float)
        expected = np.asarray(expected, dtype=float)

        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1                       # categories - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)
        chi2_crit = stats.chi2.ppf(1 - self.alpha, df)

        return {
            "chi2_statistic": chi2,
            "df": df,
            "p_value": p_value,
            "chi2_critical": chi2_crit,
            "reject_null": bool(chi2 > chi2_crit),
        }

    def independence(self, contingency_table) -> Dict[str, object]:
        """Test whether two categorical variables are independent.

        contingency_table is a 2-D array of observed counts. The returned dict
        also includes the matrix of expected frequencies.
        """
        table = np.asarray(contingency_table, dtype=float)
        row_totals = table.sum(axis=1, keepdims=True)
        col_totals = table.sum(axis=0, keepdims=True)
        grand_total = table.sum()

        # Expected count per cell under independence:
        #   E_ij = row_total_i * col_total_j / grand_total
        expected = row_totals @ col_totals / grand_total
        chi2 = np.sum((table - expected) ** 2 / expected)
        df = (table.shape[0] - 1) * (table.shape[1] - 1)
        p_value = 1 - stats.chi2.cdf(chi2, df)
        chi2_crit = stats.chi2.ppf(1 - self.alpha, df)

        return {
            "chi2_statistic": chi2,
            "df": df,
            "p_value": p_value,
            "chi2_critical": chi2_crit,
            "expected_frequencies": expected,
            "reject_null": bool(chi2 > chi2_crit),
        }
