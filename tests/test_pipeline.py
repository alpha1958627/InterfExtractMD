import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from interfextract import (gaussian_filter_density, harmonic_fill, power_spectrum,
                          radial_spectrum)
from interfextract.io import read_lammps_dump


class PipelineTests(unittest.TestCase):
    def test_radial_mean_keeps_qminus2_and_omits_empty_bands(self):
        qx = 2 * np.pi * np.fft.fftfreq(32, d=.5)
        qy = 2 * np.pi * np.fft.fftfreq(16, d=.5)
        q = np.hypot(qx[:, None], qy[None, :])
        p = np.zeros_like(q)
        p[q > 0] = 3.2 / q[q > 0]**2
        r = radial_spectrum(p, qx, qy, np.geomspace(.01, 6., 50))
        np.testing.assert_allclose(r["power_mean"], 3.2 / r["q_effective"]**2, rtol=1e-13)
        self.assertTrue((r["mode_count"] > 0).all())
        self.assertLess(len(r["mode_count"]), 49)

    def test_harmonic_preserves_observations_and_solves_periodic_equation(self):
        x, y = np.indices((12, 10))
        h = np.cos(x) + .2 * np.sin(y)
        h[:2, :3] = np.nan
        filled = harmonic_fill(h)
        mask = np.isnan(h)
        np.testing.assert_array_equal(filled[~mask], h[~mask])
        laplacian = sum(np.roll(filled, d, axis=a) for a in (0, 1) for d in (-1, 1)) - 4 * filled
        np.testing.assert_allclose(laplacian[mask], 0., atol=1e-12)
        with self.assertRaises(ValueError):
            harmonic_fill(np.full((4, 4), np.nan))

    def test_binned_gaussian_mass_and_exact_discrete_mode_response(self):
        n, width, length = 32, .4, 16.
        x = np.arange(n)
        rho = np.broadcast_to((1 + .1*np.cos(2*np.pi*3*x/n))[:, None, None], (n, n, n)).copy()
        box = np.array([[0., length]] * 3)
        filtered = gaussian_filter_density(rho, box, width=width)
        sigma = width / (length/n)
        k = np.arange(-int(4*sigma+.5), int(4*sigma+.5)+1)
        weights = np.exp(-.5*(k/sigma)**2)
        weights /= weights.sum()
        response = np.sum(weights * np.cos(2*np.pi*3*k/n))
        np.testing.assert_allclose(filtered[:, 0, 0], 1+.1*response*np.cos(2*np.pi*3*x/n), atol=1e-14)
        self.assertAlmostEqual(filtered.sum(), rho.sum(), places=9)

    def dump(self, path):
        text = ""
        for step, z in ((0, .5), (10, .6)):
            text += (f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n"
                     "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n-2 8\n"
                     f"ITEM: ATOMS id type xs ys zs\n2 2 .1 .2 .1\n1 1 .3 .4 {z}\n")
        with gzip.open(path, "wt") as f:
            f.write(text)

    def test_dump_filters_wall_and_converts_scaled_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "traj.dump.gz"
            self.dump(p)
            frames = list(read_lammps_dump(p, types=[1]))
            self.assertEqual([f[0] for f in frames], [0, 10])
            np.testing.assert_allclose(frames[0][1], [[3., 4., 3.]])
            np.testing.assert_allclose(frames[1][1], [[3., 4., 4.]])

    def test_dump_rejects_truncated_and_nonperiodic_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.dump"
            p.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp ff\n")
            with self.assertRaises(ValueError):
                list(read_lammps_dump(p, types=[1]))

    def test_cli_retains_missing_mask_and_only_fills_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "traj.dump.gz"
            self.dump(p)
            common = [sys.executable, "-m", "interfextract", str(p), "--types", "1",
                      "--methods", "hard", "--shape", "4", "4", "8", "--level", ".001",
                      "--reference-height", "4", "--branch", "nearest"]
            for setting in ("none", "harmonic"):
                out = Path(d) / setting
                run = subprocess.run(common + ["--fill", setting, "--output", str(out)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                summary = json.loads((out / "summary.json").read_text())["methods"]["hard"]
                self.assertEqual(summary["fft_written"], setting == "harmonic")
                with np.load(out / "hard.npz") as a:
                    self.assertTrue(a["missing_mask"].any())
                    self.assertTrue(np.isnan(a["height_observed"]).any())
                    if setting == "harmonic":
                        power, _, _ = power_spectrum(a["height_fft"], a["box"])
                        np.testing.assert_array_equal(power, a["power"])


if __name__ == "__main__":
    unittest.main()
