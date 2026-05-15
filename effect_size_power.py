import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower, TTestIndPower


class EffectSize:

    def cohens_d_one_sample(self, sample, pop_mean):

        sample = np.asarray(sample, dtype=float)
        return (sample.mean() - pop_mean) / sample.std(ddof=1)

    def cohens_d_two_sample(self, sample1, sample2):

        s1 = np.asarray(sample1, dtype=float)
        s2 = np.asarray(sample2, dtype=float)
        n1, n2 = len(s1), len(s2)
        pooled_std = np.sqrt(
            ((n1 - 1) * s1.var(ddof=1) + (n2 - 1) * s2.var(ddof=1)) / (n1 + n2 - 2)
        )
        return (s1.mean() - s2.mean()) / pooled_std

    def cohens_h(self, p1, p2):

        return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

    def cramers_v(self, contingency_table):

        table = np.asarray(contingency_table, dtype=float)
        row_totals = table.sum(axis=1, keepdims=True)
        col_totals = table.sum(axis=0, keepdims=True)
        n = table.sum()
        expected = row_totals @ col_totals / n
        chi2 = np.sum((table - expected) ** 2 / expected)
        k = min(table.shape) - 1
        return np.sqrt(chi2 / (n * k))

    def interpret_cohens_d(self, d):

        d = abs(d)
        if d < 0.2:
            return 'negligible'
        if d < 0.5:
            return 'small'
        if d < 0.8:
            return 'medium'
        return 'large'


class PowerAnalysis:

    def __init__(self, alpha=0.05):

        self.alpha = alpha

    def power_one_sample(self, effect_size, n):

        analysis = TTestPower()
        return analysis.power(effect_size=effect_size, nobs=n, alpha=self.alpha)

    def power_two_sample(self, effect_size, n1, ratio=1.0):

        analysis = TTestIndPower()
        return analysis.power(
            effect_size=effect_size, nobs1=n1, alpha=self.alpha, ratio=ratio
        )

    def sample_size_one_sample(self, effect_size, power=0.8):

        analysis = TTestPower()
        return analysis.solve_power(
            effect_size=effect_size, power=power, alpha=self.alpha
        )

    def sample_size_two_sample(self, effect_size, power=0.8, ratio=1.0):

        analysis = TTestIndPower()
        return analysis.solve_power(
            effect_size=effect_size, power=power, alpha=self.alpha, ratio=ratio
        )

    def minimum_detectable_effect(self, n, power=0.8):

        analysis = TTestPower()
        return analysis.solve_power(nobs=n, power=power, alpha=self.alpha)
