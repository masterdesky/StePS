import math
import unittest

import numpy as np

import stepsic.stereographic as mod     #   <-- change to the real filename if different


def allclose(a, b, rtol=1e-11, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


class TestSphericalHelpers(unittest.TestCase):
    def test_centroid_formula(self):
        for r0, r1 in [(1.0, 2.0), (0.1, 10.0), (3.3, 3.9)]:
            expected = (
                0.25 * (r1 - r0) * (r0**2 + 2*r0*r1 + 3*r1**2)
                / (r0**2 + r0*r1 + r1**2)
                + r0
            )
            got = mod.SphericalBinner.centroid(r0, r1)
            allclose(got, expected)

    def test_invert_round_trip(self):
        for y in (0.0, 0.42, 1.0, np.pi):
            x = mod.SphericalBinner.invert_x_minus_sin_x(y)
            self.assertTrue(np.isfinite(x))
            allclose(x - math.sin(x), y)

    def test_invert_out_of_range(self):
        with self.assertRaises(ValueError):
            mod.SphericalBinner.invert_x_minus_sin_x(-1e-3)
        with self.assertRaises(ValueError):
            mod.SphericalBinner.invert_x_minus_sin_x(np.pi + 1e-6)


class TestSphericalLinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bl = mod.SphericalLinear(d_s=1.5, n_bins=8, last_cell_size=1)

    def test_limits(self):
        i = 3
        expected = self.bl.d_s * math.tan(i * self.bl.d_omega)
        allclose(self.bl.r_limit(i), expected)

    def test_centroid_inside_shell(self):
        for i in range(self.bl.n_bins):
            r0 = self.bl.r_limit(i)
            r1 = self.bl.r_limit(i + 1)
            r_c = self.bl.r_centroid(i)
            self.assertGreater(r_c, r0)
            self.assertLess(r_c, r1)


class TestCylindricalHelpers(unittest.TestCase):
    def test_centroid_formula(self):
        for r0, r1 in [(1.0, 2.0), (0.3, 8.7)]:
            expected = (2.0 / 3.0) * (r1**3 - r0**3) / (r1**2 - r0**2)
            got = mod.CylindricalBinner.centroid(r0, r1)
            allclose(got, expected)


class TestCylindricalLinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cl = mod.CylindricalLinear(R_max=4.0, n_bins=8)

    def test_uniform_width(self):
        for i in range(self.cl.n_bins):
            r0, r1 = self.cl.r_limit(i), self.cl.r_limit(i + 1)
            allclose(r1 - r0, self.cl.dr)

    def test_centroid_inside(self):
        for i in range(self.cl.n_bins):
            r0, r1 = self.cl.r_limit(i), self.cl.r_limit(i + 1)
            r_c = self.cl.r_centroid(i)
            self.assertGreater(r_c, r0)
            self.assertLess(r_c, r1)


class TestCylindricalConstantVolume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cv = mod.CylindricalConstantVolume(R_max=10.0, n_bins=5)

    def test_boundaries_monotone_and_cover_domain(self):
        limits = [self.cv.r_limit(i) for i in range(self.cv.n_bins + 1)]
        self.assertAlmostEqual(limits[-1], self.cv.R_max, places=13)
        self.assertTrue(all(a < b for a, b in zip(limits, limits[1:])))


class TestMassNsampleLUT(unittest.TestCase):
    def test_shape_and_types(self):
        n_grid = 4
        mass_list = np.array([1e12, 2e12, 5e11, 6e12])
        M_box = 1e15

        lut = mod.create_mass_nsample_lut(n_grid, mass_list, M_box)

        # shape
        self.assertEqual(lut.shape, (n_grid, 2))

        # masses subset & dtypes
        self.assertTrue(set(lut[:, 0]).issubset(set(mass_list)))  # TODO: check
        self.assertTrue(np.issubdtype(lut.dtype, np.uint32))

        # sample counts positive
        self.assertTrue(np.all(lut[:, 1] > 0))

if __name__ == "__main__":
    unittest.main(verbosity=2)
