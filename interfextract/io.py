"""Streaming LAMMPS custom dump reader for fixed, fully periodic orthorhombic boxes."""
import gzip
from pathlib import Path

import numpy as np


def read_lammps_dump(path, *, types):
    """Yield (step, positions, box), selecting only explicitly specified atom types.

    Accepts Cartesian x/y/z or xu/yu/zu and scaled xs/ys/zs columns. Coordinates
    are wrapped by the density routines. Triclinic and nonperiodic boxes raise.
    Atom order may change between frames. Molecular topology is not required.
    """
    selected = tuple(types)
    if not selected or any(int(t) != t or t < 1 for t in selected):
        raise ValueError("types must contain positive integer fluid atom types")
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        while True:
            first = stream.readline()
            if not first:
                return
            if first.strip() != "ITEM: TIMESTEP":
                raise ValueError("expected ITEM: TIMESTEP")
            step = int(stream.readline())
            if stream.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("expected ITEM: NUMBER OF ATOMS")
            n = int(stream.readline())
            if n < 0 or stream.readline().strip() != "ITEM: BOX BOUNDS pp pp pp":
                raise ValueError("only fully periodic orthorhombic dumps are supported")
            box = np.array([list(map(float, stream.readline().split())) for _ in range(3)])
            if box.shape != (3, 2) or not np.isfinite(box).all() or np.any(box[:, 1] <= box[:, 0]):
                raise ValueError("invalid orthorhombic box")
            header = stream.readline().split()
            if header[:2] != ["ITEM:", "ATOMS"] or "type" not in header[2:]:
                raise ValueError("custom dump must include type and three coordinate columns")
            columns = header[2:]
            coordinates = next((names for names in (("x", "y", "z"), ("xu", "yu", "zu"),
                                                    ("xs", "ys", "zs"))
                                if all(name in columns for name in names)), None)
            if coordinates is None:
                raise ValueError("expected x y z, xu yu zu, or xs ys zs")
            indices = [columns.index(name) for name in coordinates]
            atom_type = columns.index("type")
            positions = []
            for _ in range(n):
                row = stream.readline().split()
                if len(row) != len(columns):
                    raise ValueError("truncated or malformed atom row")
                if int(row[atom_type]) in selected:
                    positions.append([float(row[k]) for k in indices])
            positions = np.array(positions, dtype=float).reshape(-1, 3)
            if not len(positions) or not np.isfinite(positions).all():
                raise ValueError("frame contains no selected fluid atoms or nonfinite coordinates")
            if coordinates[0] == "xs":
                positions = box[:, 0] + positions * (box[:, 1] - box[:, 0])
            yield step, positions, box
