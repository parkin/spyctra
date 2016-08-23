import numpy as np
import unittest

from scipy.optimize import curve_fit, leastsq

from spyctra import multifit

def linef( x, *p):
    return p[0]*np.power(x, 2) + p[1]

class TestErrorEstimate(unittest.TestCase):

    def test_linear(self):
        """
        Test function linear in parameters.
        """
        prng = np.random.RandomState(123459)
        N = 1000

        data_spread = 10.0
        uncertainty_each = 100.

        p0 = [1.5, 0.5]

        # Generate random data
        datax = np.linspace(0., 10, N)
        datay0 = linef(datax, *p0)
        datay = datay0 + prng.normal(0., data_spread, len(datay0))

        # The y errors of each datapoint
        datayerrors = np.ones_like(datay0)*uncertainty_each


        pfit, perr = multifit(linef, datax, datay, datayerrors, p0, _random_generator=prng)

        # Assert only 10 % difference in
        a_should_be = 1.51737
        aerr_should_be = 0.10368
        b_should_be = 0.32585
        berr_should_be = 4.65996
        self.assertLess((a_should_be-pfit[0])/a_should_be, 0.1)
        self.assertLess((aerr_should_be-perr[0])/aerr_should_be, 0.1)
        self.assertLess((b_should_be-pfit[1])/b_should_be, 0.15)
        self.assertLess((berr_should_be-perr[1])/berr_should_be, 0.1)
