import unittest
import numpy as np

from spyctra import lorentz, gaussian


class TestLorentz(unittest.TestCase):
    def test_single_float_values(self):
        x = 10.
        xo = 9.
        amplitude = 20.
        gamma = 6.

        params = [
            gamma,
            xo,
            amplitude
        ]

        result = lorentz(params, x)
        self.assertAlmostEqual(result, 19.459459459459)

        x = -40.
        xo = 10.
        amplitude = -89.
        gamma = 60.

        params = [
            gamma,
            xo,
            amplitude
        ]

        result = lorentz(params, x)
        self.assertAlmostEqual(result, -3204./61.)

    def test_single_integer_values(self):
        """
        Make sure passing in integer values works.
        """
        x = -12
        xo = -13
        gamma = 9
        amplitude = 12

        params = [gamma, xo, amplitude]

        result = lorentz(params, x)
        self.assertAlmostEqual(result, 486./41.)

    def test_array_shape(self):
        shape = (10,)
        x = np.ones(shape)
        xo = 12
        gamma = 3
        amplitude = 33

        params = [gamma, xo, amplitude]

        result = lorentz(params, x)

        self.assertEqual(result.shape, shape)

    def test_float_array_values(self):
        ## Test with array of all the same value
        shape = (10,)
        x = np.ones(shape)
        xo = 12
        gamma = 3
        amplitude = 33

        params = [gamma, xo, amplitude]

        result = lorentz(params, x)
        result_should_be = np.ones(shape) * 297./130.

        np.testing.assert_array_almost_equal(result, result_should_be)

        x = np.array([1., 2., 3.])

        result = lorentz(params, x)
        self.assertAlmostEqual(result[0], 297./130.)
        self.assertAlmostEqual(result[1], 297./109.)
        self.assertAlmostEqual(result[2], 33./10.)

    def test_multiple_lorentz(self):
        """
        Tests that passing in 2D params will return sum of lorentzians.
        """
        x = np.array([0., 1., 2., 3.])

        gamma1 = 3
        xo1 = 12
        amplitude1 = 33

        gamma2 = 3
        xo2 = 15
        amplitude2 = 24

        params = [
            [gamma1, xo1, amplitude1],
            [gamma2, xo2, amplitude2]
        ]

        result = lorentz(params, x)

        self.assertEqual(x.shape, result.shape)

        result_should_be = np.array([633./221., 17793./5330., 38205./9701., 801./170.])

        np.testing.assert_array_almost_equal(result, result_should_be)

    def test_multiple_lorentz_with_1d_params(self):
        """
        Tests adding multiple lorentzians where the parameter array is flat.
        """
        x = np.array([0., 1., 2., 3.])

        gamma1 = 3
        xo1 = 12
        amplitude1 = 33

        gamma2 = 3
        xo2 = 15
        amplitude2 = 24

        params = [gamma1, xo1, amplitude1,gamma2, xo2, amplitude2]

        result = lorentz(params, x)

        self.assertEqual(x.shape, result.shape)

        result_should_be = np.array([633./221., 17793./5330., 38205./9701., 801./170.])

        np.testing.assert_array_almost_equal(result, result_should_be)


class TestGaussian(unittest.TestCase):

    def test_single_float_values(self):
        x = 4.
        sigma = 5.
        xo = 12.
        amplitude = 15.

        params = [sigma, xo, amplitude]

        result = gaussian(params, x)
        self.assertAlmostEqual(result, 4.1705595067979)

    def test_single_integer_values(self):
        x = 4
        sigma = 5
        xo = 12
        amplitude = 15

        params = [sigma, xo, amplitude]

        result = gaussian(params, x)
        self.assertAlmostEqual(result, 4.1705595067979)

    def test_array_shape(self):
        x = np.array([1., 2., 3.])
        sigma = 5
        xo = 12
        amplitude = 15

        params = [sigma, xo, amplitude]

        result = gaussian(params, x)
        self.assertEqual(result.shape, (3,))

    def test_float_array_values(self):
        x = np.array([1., 2., 4.])
        sigma = 5
        xo = 12
        amplitude = 15

        params = [sigma, xo, amplitude]

        result = gaussian(params, x)
        result_should_be = np.array([1.33382426189,2.03002925,4.1705595067979])

        np.testing.assert_array_almost_equal(result, result_should_be)

    def test_multiple_gaussian(self):
        """
        Tests that passing in 2D params will return sum of gaussians.
        """
        x = 4.

        sigma1 = 5.
        xo1 = 12.
        amplitude1 = 15.

        sigma2 = 4.
        xo2 = 10.
        amplitude2 = 9.

        params = [
            [sigma1, xo1, amplitude1],
            [sigma2, xo2, amplitude2]
        ]

        result = gaussian(params, x)

        self.assertAlmostEqual(result, 7.092431713)

    def test_multiple_gaussian_with_1d_params(self):
        x = 4.

        sigma1 = 5.
        xo1 = 12.
        amplitude1 = 15.

        sigma2 = 4.
        xo2 = 10.
        amplitude2 = 9.

        params = [sigma1, xo1, amplitude1, sigma2, xo2, amplitude2]

        result = gaussian(params, x)
