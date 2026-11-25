"""Run a selected trajectory window through interface extraction and 2-D spectra."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import (hard_bin_density, gaussian_density, gaussian_filter_density,
               neighbour_surface, surface_from_density, power_spectrum,
               harmonic_fill, radial_spectrum)
from .io import read_lammps_dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="LAMMPS custom dump or dump.gz")
    parser.add_argument("--types", type=int, nargs="+", required=True, help="fluid types, excluding the wall")
    parser.add_argument("--methods", nargs="+", choices=("hard", "gaussian", "neighbor"),
                        default=["hard", "gaussian", "neighbor"])
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--level", type=float, required=True, help="absolute density threshold")
    parser.add_argument("--reference-height", type=float, required=True)
    parser.add_argument("--branch", choices=("continuous", "nearest"), default="continuous")
    parser.add_argument("--gaussian-mode", choices=("direct", "binned"), default="binned")
    parser.add_argument("--width", type=float, default=1 / np.sqrt(20))
    parser.add_argument("--radius", type=float, default=3.5)
    parser.add_argument("--neighbor-threshold", type=float, default=.415)
    parser.add_argument("--start", type=int, help="first included step")
    parser.add_argument("--stop", type=int, help="last included step")
    parser.add_argument("--weights", choices=("trapezoid", "equal"), default="trapezoid")
    parser.add_argument("--fill", choices=("none", "harmonic"), default="none")
    parser.add_argument("--max-cells", type=int, default=20_000_000)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if min(args.shape) < 2 or args.bins < 1:
        parser.error("shape dimensions must be at least 2 and bins must be positive")
    if args.start is not None and args.stop is not None and args.start > args.stop:
        parser.error("start must not exceed stop")
    if args.output.exists():
        parser.error("output directory already exists; choose a new path")
    methods = list(dict.fromkeys(args.methods))
    fields, previous, total_weight, frames = {}, None, 0., 0
    first_step, last_step, box = None, None, None

    for step, positions, frame_box in read_lammps_dump(args.dump, types=args.types):
        if args.start is not None and step < args.start:
            continue
        if args.stop is not None and step > args.stop:
            break
        if box is None:
            box = frame_box
            first_step = step
        if not np.array_equal(box, frame_box) or (last_step is not None and step <= last_step):
            raise ValueError("selected frames must have one fixed box and strictly increasing steps")
        current = {}
        if "hard" in methods or ("gaussian" in methods and args.gaussian_mode == "binned"):
            current["hard"] = hard_bin_density(positions, box, args.shape, max_cells=args.max_cells)
        if "gaussian" in methods and args.gaussian_mode == "direct":
            current["gaussian"] = gaussian_density(positions, box, args.shape, width=args.width,
                                                   max_cells=args.max_cells)
        if "neighbor" in methods:
            current["neighbor"] = neighbour_surface(positions, box, args.shape[:2], radius=args.radius,
                                                     density_threshold=args.neighbor_threshold)[0]
        if args.weights == "equal":
            weight = 1.
            contribution = current
        elif previous is None:
            weight = 0.
            contribution = current
        else:
            weight = float(step - last_step)
            contribution = {name: (previous[name] + field) / 2 for name, field in current.items()}
        for name, field in contribution.items():
            if name not in fields:
                fields[name] = np.zeros_like(field)
            if weight > 0:
                fields[name] += weight * field
        total_weight += weight
        previous = current
        last_step = step
        frames += 1
    if frames == 0:
        raise ValueError("no frames in the requested step window")
    if frames == 1:
        fields, total_weight = previous, 1.
    fields = {name: field / total_weight for name, field in fields.items()}
    if "gaussian" in methods and args.gaussian_mode == "binned":
        fields["gaussian"] = gaussian_filter_density(fields["hard"], box, width=args.width)
    heights, checks = {}, {}
    for method in methods:
        if method == "neighbor":
            heights[method] = fields[method]
            checks[method] = {"valid": bool(np.isfinite(fields[method]).all()),
                              "missing": int(np.isnan(fields[method]).sum())}
        else:
            heights[method], checks[method] = surface_from_density(
                fields[method], box, level=args.level, reference_height=args.reference_height,
                branch=args.branch)
    args.output.mkdir(parents=True)
    metadata = {"input": str(args.dump), "selected_types": args.types, "box": box.tolist(),
                "frames": frames, "steps": [first_step, last_step], "weight_sum": total_weight,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "methods": {}}
    for method, observed in heights.items():
        missing = ~np.isfinite(observed)
        arrays = {"height_observed": observed, "missing_mask": missing, "box": box}
        info = checks[method]
        reconstructed = args.fill == "harmonic" and missing.any() and not missing.all()
        can_fft = bool(info["valid"] or reconstructed)
        if can_fft:
            h = harmonic_fill(observed) if reconstructed else observed
            power, qx, qy = power_spectrum(h, box)
            qmin = min(np.min(abs(qx[qx != 0])), np.min(abs(qy[qy != 0])))
            # Complete angular coverage stops at the smaller axis Nyquist.
            lengths = box[:, 1] - box[:, 0]
            qmax = min(np.pi * h.shape[0] / lengths[0], np.pi * h.shape[1] / lengths[1])
            radial = radial_spectrum(power, qx, qy, np.geomspace(qmin, qmax, args.bins + 1))
            with (args.output / (method + "_radial.csv")).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(radial)
                writer.writerows(zip(*radial.values()))
            arrays.update(height_fft=h, power=power, qx=qx, qy=qy)
        path = args.output / (method + ".npz")
        np.savez_compressed(path, **arrays)
        metadata["methods"][method] = {"extraction": info, "fft_written": can_fft,
                                        "harmonic_reconstruction": bool(reconstructed),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({name: {"missing": c["extraction"]["missing"], "fft_written": c["fft_written"]}
                      for name, c in metadata["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
