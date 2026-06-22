"""Hypothesis-testing toolkit.

Classical hypothesis tests implemented from their definitions on top of NumPy
and SciPy. Every test returns a plain dict (statistic, degrees of freedom,
p-value, critical value, and the reject/fail-to-reject decision), so results
are easy to print or drop into a DataFrame.

See README.md for the statistics behind each test and examples/ for runnable
worked examples.
"""

from .anova_f_test import AnovaFTest
from .chi_square_test import ChiSquareTest
from .effect_size_power import EffectSize, PowerAnalysis
from .t_test import TTest
from .z_test import ZTest

__all__ = [
    "ZTest",
    "TTest",
    "ChiSquareTest",
    "AnovaFTest",
    "EffectSize",
    "PowerAnalysis",
]

__version__ = "1.0.0"
