import unittest
from spyctra import arPLS
import numpy as np
from scipy.stats import norm


class TestArPLS(unittest.TestCase):

    def test_same_return_shape(self):
        """
        """
        y = np.ones(100)
        y += np.random.random(100)

        z = arPLS(y)

        self.assertEqual(y.shape, z.shape)

    def test_constant_baseline_no_spikes(self):
        offset = 99.
        y = np.zeros(1000) + offset
        y += np.random.random(1000)*0.5 - 0.25

        z = arPLS(y)

        d = y-z
        self.assertAlmostEqual(np.mean(d), 0, places=1)

        # test that it has been smoothed
        self.assertLess(np.std(z,ddof=1), np.std(y,ddof=1))

    def test_negativeconstant_baseline_no_spikes(self):
        offset = -99.
        y = np.zeros(1000) + offset
        y += np.random.random(1000)*0.5 - 0.25

        z = arPLS(y)

        d = y-z
        self.assertAlmostEqual(np.mean(d), 0, places=1)

        # test that it has been smoothed
        self.assertLess(np.std(z,ddof=1), np.std(y,ddof=1))

    def test_constant_baseline_no_spikes_no_noise(self):
        offset = 99.
        y = np.zeros(1000) + offset
        y += np.random.random(1000)*0.5 - 0.25

        z = arPLS(y)

        d = y-z
        self.assertAlmostEqual(np.mean(d), 0, places=1)

        # test that it has been smoothed
        self.assertLess(np.std(z,ddof=1), np.std(y,ddof=1))

    def test_zero_baseline_no_spikes(self):
        offset = 0.
        y = np.zeros(1000) + offset
        y += np.random.random(1000)*0.5 - 0.25

        z = arPLS(y)

        d = y-z
        self.assertAlmostEqual(np.mean(d), 0, places=1)

        # test that it has been smoothed
        self.assertLess(np.std(z,ddof=1), np.std(y,ddof=1))

    def test_zero_baseline_no_spikes_no_noise(self):
        offset = 0.
        y = np.zeros(1000) + offset

        z = arPLS(y)

        d = y-z
        self.assertAlmostEqual(np.mean(d), 0, places=1)


    def test_linear_baseline_no_spikes(self):
        x=np.arange(0,1000,1)
        slope = 2.
        offset = 10.

        y = offset + slope * x
        y += np.random.random(1000)*0.5 - 0.25

        z = arPLS(y)
        d = y - z

        fit = np.polyfit(x, z, 2)

        self.assertAlmostEqual(fit[0], 0, places=1) # x**2
        self.assertAlmostEqual(fit[1], slope, places=1) # x
        self.assertAlmostEqual(fit[2]-offset, 0., places=1) # const

        f = np.poly1d(fit)
        y_sub = y - f(x)
        z_sub = z - f(x)

        # test that it is smoother
        self.assertAlmostEqual(np.mean(z_sub), 0., places=3)
        self.assertLess(np.std(z_sub,ddof=1), np.std(y_sub,ddof=1))


    def test_linear_baseline_with_spikes(self):
        x=np.arange(0,1000,1)
        slope = 2.
        offset = 10.

        g1=norm(loc = 100, scale = 1.0) # generate three gaussian as a signal
        g2=norm(loc = 300, scale = 3.0)
        g3=norm(loc = 750, scale = 5.0)

        y = offset + slope*x + 200.*g1.pdf(x) + 300.*g2.pdf(x) + 500.*g3.pdf(x)
        y += np.random.random(1000)*0.5 - 0.25

        z = arPLS(y)
        d = y - z

        fit = np.polyfit(x, z, 2)

        self.assertAlmostEqual(fit[0], 0, places=1) # x**2
        self.assertAlmostEqual(fit[1], slope, places=1) # x
        self.assertAlmostEqual(fit[2], offset, places=1) # const

    def test_linear_baseline_with_spikes_data_in_list(self):
        x=np.arange(0,1000,1)
        slope = 2.
        offset = 10.

        g1=norm(loc = 100, scale = 1.0) # generate three gaussian as a signal
        g2=norm(loc = 300, scale = 3.0)
        g3=norm(loc = 750, scale = 5.0)

        y = offset + slope*x + 200.*g1.pdf(x) + 300.*g2.pdf(x) + 500.*g3.pdf(x)
        y += np.random.random(1000)*0.5 - 0.25

        y = y.tolist()

        z = arPLS(y)
        d = y - z

        fit = np.polyfit(x, z, 2)

        self.assertAlmostEqual(fit[0], 0, places=1) # x**2
        self.assertAlmostEqual(fit[1], slope, places=1) # x
        self.assertAlmostEqual(fit[2], offset, places=1) # const

    def test_linear_baseline_with_spikes_with_first_zeros(self):
        """
        Tests that you can fit a linear baseline, when the second 1000 points are
        0 and the first 1000 points have a slope.
        """
        N = 1010
        x=np.arange(0,N,1)
        slope = 2.
        offset = 10.

        g1=norm(loc = 100, scale = 1.0) # generate three gaussian as a signal
        g2=norm(loc = 300, scale = 3.0)
        g3=norm(loc = 750, scale = 5.0)

        y = np.zeros(N)

        y[:1000] = offset + slope*x[:1000] + 100.*g1.pdf(x[:1000]) + 300.*g2.pdf(x[:1000]) + 500.*g3.pdf(x[:1000])
        y[:1000] += np.random.random(1000)*0.5 - 0.25

        z = arPLS(y, lambda_=(10.)**2)
        d = y - z

        fit = np.polyfit(x[:500], z[:500], 1)

        self.assertAlmostEqual(fit[0], slope, places=1) # x
        self.assertAlmostEqual(fit[1], offset, places=1) # const
