import numpy as np
from scipy import stats


class ChiSquareTest:

    def __init__(self, alpha=0.05):

        self.alpha = alpha

    def goodness_of_fit(self, observed, expected):

        observed = np.asarray(observed, dtype=float)
        expected = np.asarray(expected, dtype=float)

        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)
        chi2_crit = stats.chi2.ppf(1 - self.alpha, df)

        return {
            'chi2_statistic': chi2,
            'df': df,
            'p_value': p_value,
            'chi2_critical': chi2_crit,
            'reject_null': chi2 > chi2_crit,
        }

    def independence(self, contingency_table):

        table = np.asarray(contingency_table, dtype=float)
        row_totals = table.sum(axis=1, keepdims=True)
        col_totals = table.sum(axis=0, keepdims=True)
        grand_total = table.sum()

        expected = row_totals @ col_totals / grand_total
        chi2 = np.sum((table - expected) ** 2 / expected)
        df = (table.shape[0] - 1) * (table.shape[1] - 1)
        p_value = 1 - stats.chi2.cdf(chi2, df)
        chi2_crit = stats.chi2.ppf(1 - self.alpha, df)

        return {
            'chi2_statistic': chi2,
            'df': df,
            'p_value': p_value,
            'chi2_critical': chi2_crit,
            'expected_frequencies': expected,
            'reject_null': chi2 > chi2_crit,
        }
