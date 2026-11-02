import unittest
import numpy as np
from interfextract import (hard_bin_density, gaussian_density, average_density,
                          surface_from_density, neighbour_density, neighbour_surface, power_spectrum)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.box = np.array([[0., 10.], [0., 10.], [0., 10.]])

    def test_hard_mass_and_periodic_translation(self):
        p = np.array([[0, 0, 0], [10, 10, 10], [2.3, 4.5, 6.7]])
        a = hard_bin_density(p, self.box, (5, 5, 5))
        b = hard_bin_density(p + [10, -20, 30], self.box, (5, 5, 5))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.sum() * 8, len(p))
        self.assertEqual(a[0, 0, 0], 2 / 8)

    def test_gaussian_direct_periodic_oracle(self):
        point = np.array([[.1, 1.2, 9.9]])
        rho = gaussian_density(point, self.box, (20, 20, 20), width=.4)
        weights = []
        for x in point[0]:
            delta = ((np.arange(20) + .5) * .5 - x + 5) % 10 - 5
            w = np.where(abs(delta) <= 1.6, np.exp(-.5 * (delta / .4) ** 2), 0.)
            weights.append(w / w.sum())
        oracle = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :] / .5 ** 3
        np.testing.assert_allclose(rho, oracle, rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(rho.sum() * .5 ** 3, 1.)
        shifted = gaussian_density(point + [10, -10, 20], self.box, (20, 20, 20), width=.4)
        np.testing.assert_allclose(rho, shifted, rtol=1e-12, atol=1e-14)

    def test_gaussian_rejects_unresolved_support(self):
        with self.assertRaises(ValueError):
            gaussian_density([[0, 0, 0]], self.box, (2, 2, 2), width=.01)

    def test_time_average_density_mass_and_weights(self):
        p = np.array([[1., 1., 1.], [3., 3., 3.]])
        frames = [p, p + 1, p + 2]
        result = average_density(iter(frames), self.box, (10, 10, 10), method="hard", weights=[.5, 1, .5])
        expected = sum(w * hard_bin_density(f, self.box, (10, 10, 10)) for f, w in zip(frames, [.5, 1, .5])) / 2
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.sum(), 2)
        with self.assertRaises(ValueError):
            average_density(frames, self.box, (10, 10, 10), method="hard", weights=[1.])
        with self.assertRaises(ValueError):
            average_density(frames, self.box, (10, 10, 10), method="hard", weights=[1, -1, 1])

    def known_surface(self):
        n = 16
        box = np.array([[0., n], [0., n], [-4., 40.]])
        h = 12 + .2 * np.cos(2 * np.pi * np.arange(n) / n)[:, None] + np.zeros((n, n))
        z = -4 + np.arange(44) + .5
        rho = np.clip(.5 + (h[:, :, None] - z) / 2, 0, 1)
        return rho, h, box

    def test_actual_linear_roots_and_detached_vapour(self):
        rho, expected, box = self.known_surface()
        rho[2:4, 3:5, 37:40] = 1.
        for branch in ["nearest", "continuous"]:
            h, info = surface_from_density(rho, box, level=.5, reference_height=12., branch=branch)
            self.assertTrue(info["valid"])
            self.assertLess(info["root_residual"], 1e-12)
            np.testing.assert_allclose(h, expected, rtol=0, atol=1e-10)
            self.assertLess(h.max(), 20.)

    def test_missing_columns_preserved_and_FFT_rejected(self):
        rho, _, box = self.known_surface()
        rho[2, 3, :] = 0
        h, info = surface_from_density(rho, box, level=.5, reference_height=12., branch="nearest")
        self.assertFalse(info["valid"])
        self.assertEqual(info["raw_missing"], 1)
        self.assertTrue(np.isnan(h[2, 3]))
        with self.assertRaises(ValueError):
            power_spectrum(h, box)

    def test_no_crossing_returns_missing(self):
        h, info = surface_from_density(np.zeros((3, 3, 3)), self.box, level=.1, reference_height=5.)
        self.assertFalse(info["valid"])
        self.assertTrue(np.isnan(h).all())

    def test_continuous_branch_also_preserves_hole(self):
        rho, _, box = self.known_surface()
        rho[2, 3, :] = 0
        h, info = surface_from_density(rho, box, level=.5, reference_height=12., branch="continuous")
        self.assertFalse(info["valid"])
        self.assertTrue(np.isnan(h[2, 3]))

    def test_neighbours_periodic_self_exclusion_and_cutoff(self):
        p = np.array([[.1, 1., 1.], [9.9, 1., 1.], [2., 1., 1.]])
        _, counts = neighbour_density(p, self.box, radius=.3)
        np.testing.assert_array_equal(counts, [1, 1, 0])
        _, counts = neighbour_density([[0, 0, 0], [1, 0, 0]], self.box, radius=1.)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_neighbours_independent_minimum_image_oracle(self):
        p = np.random.default_rng(12).uniform(0, 10, (40, 3))
        _, counts = neighbour_density(p, self.box, radius=2.)
        for i in range(len(p)):
            delta = abs(p - p[i]);delta = np.minimum(delta, 10 - delta)
            oracle = np.count_nonzero(np.sum(delta * delta, axis=1) <= 4) - 1
            self.assertEqual(counts[i], oracle)

    def test_neighbour_highest_actual_bead_and_empty_bins(self):
        p = np.array([[1., 1., 4.], [1., 1., 5.]])
        h, info = neighbour_surface(p, self.box, (2, 2), radius=2., density_threshold=0.)
        self.assertEqual(h[0, 0], 5.)
        self.assertEqual(info["missing"], 3)
        self.assertFalse(info["valid"])

    def test_per_mode_fft_normalization(self):
        h = 2 + .3 * np.cos(2 * np.pi * np.arange(16) / 16)[:, None] + np.zeros((16, 12))
        p, _, _ = power_spectrum(h, self.box)
        self.assertAlmostEqual(p[1, 0], .3 ** 2 / 4)
        self.assertAlmostEqual(p[-1, 0], .3 ** 2 / 4)
        self.assertAlmostEqual(p.sum(), h.var())

    def test_memory_guard(self):
        with self.assertRaises(MemoryError):
            hard_bin_density([[1, 1, 1]], self.box, (1000, 1000, 1000))


if __name__ == "__main__":
    unittest.main()
