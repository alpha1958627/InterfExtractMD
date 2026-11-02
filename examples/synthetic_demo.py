"""Small synthetic particle film: algorithm demonstration, not MD evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from interfextract import hard_bin_density, gaussian_density, surface_from_density, neighbour_surface, power_spectrum

rng = np.random.default_rng(42)
box = np.array([[0., 12.], [0., 12.], [0., 20.]])
x, y, layer = np.meshgrid((np.arange(24) + .5) * .5,
                         (np.arange(24) + .5) * .5,
                         (np.arange(7) + .5) / 7, indexing="ij")
top = 9 + .3 * np.cos(2 * np.pi * x / 12) + .2 * np.sin(2 * np.pi * y / 12)
positions = np.column_stack([x.ravel(), y.ravel(), (2 + (top - 2) * layer).ravel()])
positions += rng.uniform(-.025, .025, positions.shape)
print("Synthetic demonstration only; not a molecular-dynamics trajectory.")
for name, function in [("Hard bins", hard_bin_density), ("Gaussian", gaussian_density)]:
    kwargs = {"width": .4} if name == "Gaussian" else {}
    rho = function(positions, box, (24, 24, 40), **kwargs)
    h, info = surface_from_density(rho, box, level=.4, reference_height=9., branch="continuous")
    print(name, "missing columns:", info["missing"], "valid:", info["valid"])
    if info["valid"]:
        power, _, _ = power_spectrum(h, box)
        print("  height std:", float(h.std()), "power sum:", float(power.sum()))
h, info = neighbour_surface(positions, box, (24, 24))
print("Neighbour density", "missing columns:", info["missing"], "valid:", info["valid"])
if info["valid"]:
    power, _, _ = power_spectrum(h, box)
    print("  height std:", float(h.std()), "power sum:", float(power.sum()))
