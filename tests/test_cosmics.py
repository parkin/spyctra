import unittest
import numpy as np
from spyctra import remove_cosmics


class TestRemoveCosmics(unittest.TestCase):

    def test_no_cosmics(self):
        """
        Tests that the input array is unchanged if there are no cosmics
        """
        prng = np.random.RandomState(84287)
        y = prng.normal(size=1000)

        y2 = remove_cosmics(y)

        np.testing.assert_array_equal(y, y2)


    def test_onepoint_cosmic(self):
        """
        Tests the removal of one cosmic of one point width.
        """
        prng = np.random.RandomState(18859)
        y = prng.normal(size=1000)

        y[500] = 3000.

        wher = np.where(y < 1500.)

        y2 = remove_cosmics(y)

        # assert that the other points are unchanged
        np.testing.assert_array_equal(y[wher], y2[wher])

        # assert that the one point has been changed
        self.assertLessEqual(y2[500], 5.)

    def test_twopoint_cosmics(self):
        """
        Tests the removal of one cosmic of two point width.
        """
        prng = np.random.RandomState(159)
        y = prng.normal(size=1000)

        y[500] = 4000.
        y[501] = 12000.

        wher = np.where(y < 1500.)

        y2 = remove_cosmics(y)

        # assert that the other points are unchanged
        np.testing.assert_array_equal(y[wher], y2[wher])

        # assert that the one point has been changed
        self.assertLessEqual(y2[500], 5.)
        self.assertLessEqual(y2[501], 5.)

    def test_two_cosmics(self):
        """
        Tests the removal of two comics.
        """
        prng = np.random.RandomState(1259)
        y = prng.normal(size=1000)

        y[500] = 4000.
        y[601] = 12000.

        wher = np.where(y < 1500.)

        y2 = remove_cosmics(y)

        # assert that the other points are unchanged
        np.testing.assert_array_equal(y[wher], y2[wher])

        # assert that the one point has been changed
        self.assertLessEqual(y2[500], 5.)
        self.assertLessEqual(y2[601], 5.)

    def test_cosmic_at_beginning_ignored(self):
        """
        Test that spikes at the beginning are ignored, as they will throw an error.
        """
        prng = np.random.RandomState(123459)
        y = prng.normal(size=1000)

        y[0] = -14000.

        y2 = remove_cosmics(y)

        # assert that all the points are unchanged
        np.testing.assert_array_equal(y, y2)

    def test_cosmic_at_2nd_point_ignored(self):
        prng = np.random.RandomState(12345)
        y = prng.normal(size=1000)

        y[1] = 14000.

        y2 = remove_cosmics(y)

        # assert that all the points are unchanged
        np.testing.assert_array_equal(y, y2)

    def test_cosmic_at_3rd_point_ignored(self):
        prng = np.random.RandomState(333)
        y = prng.normal(size=1000)

        y[2] = 14000.

        y2 = remove_cosmics(y)

        # assert that all the points are unchanged
        np.testing.assert_array_equal(y, y2)

    def test_cosmic_at_end_ignored(self):
        prng = np.random.RandomState(4582398)
        y = prng.normal(size=1000)

        y[999] = -14000.

        y2 = remove_cosmics(y)

        # assert that all the points are unchanged
        np.testing.assert_array_equal(y, y2)

    def test_cosmic_2nd_from_end_ignored(self):
        prng = np.random.RandomState(9)
        y = prng.normal(size=1000)

        y[998] = 14000.

        y2 = remove_cosmics(y)

        # assert that all the points are unchanged
        np.testing.assert_array_equal(y, y2)

    def test_cosmic_3rd_from_end_ignored(self):
        prng = np.random.RandomState(1)
        y = prng.normal(size=1000)

        y[997] = 14000.

        y2 = remove_cosmics(y)

        # assert that all the points are unchanged
        np.testing.assert_array_equal(y, y2)
