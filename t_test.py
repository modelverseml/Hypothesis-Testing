import numpy as np
from scipy import stats


class TTest:

    def __init__(self, alpha=0.05, tail='two'):

        self.alpha = alpha
        self.tail = tail

    def _p_value(self, t_stat, df):

        if self.tail == 'two':
            return 2 * (1 - stats.t.cdf(abs(t_stat), df))
        elif self.tail == 'left':
            return stats.t.cdf(t_stat, df)
        else:
            return 1 - stats.t.cdf(t_stat, df)

    def _critical(self, df):

        if self.tail == 'two':
            return stats.t.ppf(1 - self.alpha / 2, df)
        return stats.t.ppf(1 - self.alpha, df)

    def one_sample(self, sample, pop_mean):

        sample = np.asarray(sample)
        n = len(sample)
        x_bar = sample.mean()
        s = sample.std(ddof=1)
        se = s / np.sqrt(n)
        t_stat = (x_bar - pop_mean) / se
        df = n - 1

        return {
            't_statistic': t_stat,
            'df': df,
            'p_value': self._p_value(t_stat, df),
            't_critical': self._critical(df),
            'standard_error': se,
            'reject_null': self._p_value(t_stat, df) < self.alpha,
        }

    def two_sample_independent(self, sample1, sample2, equal_var=True):

        s1 = np.asarray(sample1)
        s2 = np.asarray(sample2)
        n1, n2 = len(s1), len(s2)
        m1, m2 = s1.mean(), s2.mean()
        v1, v2 = s1.var(ddof=1), s2.var(ddof=1)

        if equal_var:
            pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            se = np.sqrt(v1 / n1 + v2 / n2)
            df = (v1 / n1 + v2 / n2) ** 2 / (
                (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            )

        t_stat = (m1 - m2) / se
        p_value = self._p_value(t_stat, df)

        return {
            't_statistic': t_stat,
            'df': df,
            'p_value': p_value,
            't_critical': self._critical(df),
            'standard_error': se,
            'reject_null': p_value < self.alpha,
        }

    def paired(self, before, after):

        before = np.asarray(before)
        after = np.asarray(after)
        diffs = after - before
        return self.one_sample(diffs, pop_mean=0)
