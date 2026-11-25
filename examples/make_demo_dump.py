"""Write a three-frame synthetic particle film for the command-line example."""
import gzip
from pathlib import Path
import sys

import numpy as np

target = Path(sys.argv[1] if len(sys.argv) > 1 else "output/demo.dump.gz")
target.parent.mkdir(parents=True, exist_ok=True)
x, y, layer = np.meshgrid((np.arange(24) + .5) * .5,
                          (np.arange(24) + .5) * .5,
                          (np.arange(7) + .5) / 7, indexing="ij")
rng = np.random.default_rng(42)
jitter = rng.uniform(-.025, .025, (x.size, 3))
with gzip.open(target, "wt") as stream:
    for step in (0, 10, 20):
        top = 9 + .3*np.cos(2*np.pi*x/12 + step*.005) + .2*np.sin(2*np.pi*y/12)
        positions = np.column_stack([x.ravel(), y.ravel(), (2 + (top-2)*layer).ravel()]) + jitter
        stream.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(positions)}\n")
        stream.write("ITEM: BOX BOUNDS pp pp pp\n0 12\n0 12\n0 20\nITEM: ATOMS id type x y z\n")
        for i, (px, py, pz) in enumerate(positions, 1):
            stream.write(f"{i} 1 {px:.17g} {py:.17g} {pz:.17g}\n")
print(f"Synthetic demonstration, three frames written to {target}")
