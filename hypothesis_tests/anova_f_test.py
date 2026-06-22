"""ANOVA and F-tests built on the F-distribution.

    one_way        - compare the means of three or more groups. Running many
                     pairwise t-tests would inflate the false-positive rate;
                     ANOVA tests them together by comparing the variance
                     between groups with the variance within groups.
    variance_ratio - F-test for equality of two variances, handy as a check
                     before choosing between Student's and Welch's t-test.
"""

from typing import Dict

import numpy as np
from scipy import stats


class AnovaFTest:
    """One-way ANOVA and the F-test for equality of variances."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def one_way(self, *groups) -> Dict[str, float]:
        """One-way ANOVA across two or more groups.

        Tests H0: all group means are equal vs H1: at least one differs.
        Pass each group as a separate sequence.
        """
        groups = [np.asarray(g, dtype=float) for g in groups]
        k = len(groups)                              # number of groups
        n_total = sum(len(g) for g in groups)        # total observations
        grand_mean = np.concatenate(groups).mean()

        # Between: how far each group mean sits from the grand mean (weighted
        # by group size). Within: spread of points around their own group mean.
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

        df_between = k - 1
        df_within = n_total - k

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        f_stat = ms_between / ms_within
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)
        f_crit = stats.f.ppf(1 - self.alpha, df_between, df_within)

        return {
            "f_statistic": f_stat,
            "df_between": df_between,
            "df_within": df_within,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ms_between": ms_between,
            "ms_within": ms_within,
            "p_value": p_value,
            "f_critical": f_crit,
            "reject_null": bool(f_stat > f_crit),
        }

    def variance_ratio(self, sample1, sample2) -> Dict[str, float]:
        """Two-tailed F-test for the equality of two population variances."""
        s1 = np.asarray(sample1, dtype=float)
        s2 = np.asarray(sample2, dtype=float)

        var1 = s1.var(ddof=1)
        var2 = s2.var(ddof=1)

        # Put the larger variance on top so F >= 1, and order the df to match.
        if var1 >= var2:
            f_stat = var1 / var2
            df1, df2 = len(s1) - 1, len(s2) - 1
        else:
            f_stat = var2 / var1
            df1, df2 = len(s2) - 1, len(s1) - 1

        # Two-tailed, so split alpha across both tails.
        p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
        f_crit = stats.f.ppf(1 - self.alpha / 2, df1, df2)

        return {
            "f_statistic": f_stat,
            "df1": df1,
            "df2": df2,
            "p_value": p_value,
            "f_critical": f_crit,
            "reject_null": bool(f_stat > f_crit),
        }
