import numpy as np
from scipy import stats


class ZTest:

    def __init__(self, alpha=0.05, tail='two'):

        self.alpha = alpha
        self.tail = tail

    def one_sample(self, sample_mean, pop_mean, pop_std, n):

        se = pop_std / np.sqrt(n)
        z = (sample_mean - pop_mean) / se

        if self.tail == 'two':
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
            lcv = pop_mean - z_crit * se
            ucv = pop_mean + z_crit * se
            reject = (sample_mean < lcv) or (sample_mean > ucv)
        elif self.tail == 'left':
            p_value = stats.norm.cdf(z)
            z_crit = stats.norm.ppf(self.alpha)
            lcv = pop_mean + z_crit * se
            ucv = None
            reject = sample_mean < lcv
        else:
            p_value = 1 - stats.norm.cdf(z)
            z_crit = stats.norm.ppf(1 - self.alpha)
            lcv = None
            ucv = pop_mean + z_crit * se
            reject = sample_mean > ucv

        return {
            'z_score': z,
            'p_value': p_value,
            'lcv': lcv,
            'ucv': ucv,
            'standard_error': se,
            'reject_null': reject,
        }

    def two_sample(self, mean1, mean2, std1, std2, n1, n2):

        se = np.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
        z = (mean1 - mean2) / se

        if self.tail == 'two':
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        elif self.tail == 'left':
            p_value = stats.norm.cdf(z)
        else:
            p_value = 1 - stats.norm.cdf(z)

        return {
            'z_score': z,
            'p_value': p_value,
            'standard_error': se,
            'reject_null': p_value < self.alpha,
        }
