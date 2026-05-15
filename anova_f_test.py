import numpy as np
from scipy import stats


class AnovaFTest:

    def __init__(self, alpha=0.05):

        self.alpha = alpha

    def one_way(self, *groups):

        groups = [np.asarray(g, dtype=float) for g in groups]
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        grand_mean = np.concatenate(groups).mean()

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
            'f_statistic': f_stat,
            'df_between': df_between,
            'df_within': df_within,
            'ss_between': ss_between,
            'ss_within': ss_within,
            'ms_between': ms_between,
            'ms_within': ms_within,
            'p_value': p_value,
            'f_critical': f_crit,
            'reject_null': f_stat > f_crit,
        }

    def variance_ratio(self, sample1, sample2):

        s1 = np.asarray(sample1, dtype=float)
        s2 = np.asarray(sample2, dtype=float)

        var1 = s1.var(ddof=1)
        var2 = s2.var(ddof=1)

        if var1 >= var2:
            f_stat = var1 / var2
            df1, df2 = len(s1) - 1, len(s2) - 1
        else:
            f_stat = var2 / var1
            df1, df2 = len(s2) - 1, len(s1) - 1

        p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
        f_crit = stats.f.ppf(1 - self.alpha / 2, df1, df2)

        return {
            'f_statistic': f_stat,
            'df1': df1,
            'df2': df2,
            'p_value': p_value,
            'f_critical': f_crit,
            'reject_null': f_stat > f_crit,
        }
