"""Find the absolute horizontal detector center with TomoPy.

The input projections must already be flat-field corrected and converted to
line integrals. TomoPy returns the detector-center coordinate in pixels;
this script converts it to TIGRE's ``offDetector[0]`` physical offset.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init_px", type=float, default=None)
    parser.add_argument("--tol", type=float, default=0.25)
    parser.add_argument("--algorithm", default="scipy")
    parser.add_argument("--slice_step", type=int, default=8)
    parser.add_argument("--slice_margin", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main(args):
    try:
        import tomopy
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TomoPy is not installed in the active Python environment. "
            "Install it first, for example: conda install -c conda-forge tomopy"
        ) from exc

    with (args.data / "meta_data.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    scanner = metadata["scanner"]
    projections = np.load(args.data / "proj_all.npy").astype(np.float32)
    angles = np.asarray(metadata["source"]["angles_all"], dtype=np.float32)
    if projections.ndim != 3 or len(projections) != len(angles):
        raise ValueError(
            f"Expected projections (N,H,W) matching angles, got "
            f"{projections.shape} and {angles.shape}"
        )

    endpoint_name = metadata.get("source", {}).get("endpoint_pair")
    if endpoint_name:
        endpoint_path = args.data / endpoint_name
        if endpoint_path.is_file():
            endpoint_pair = np.load(endpoint_path).astype(np.float32)
            if endpoint_pair.shape[0] == 2 and endpoint_pair.shape[1:] == projections.shape[1:]:
                projections = np.concatenate([projections, endpoint_pair[1:2]], axis=0)
                angles = np.concatenate([angles, np.asarray([np.pi], dtype=np.float32)])

    n_slices = projections.shape[1]
    margin = max(0, int(args.slice_margin))
    indices = np.arange(margin, n_slices - margin, max(1, int(args.slice_step)))
    if indices.size == 0:
        raise ValueError("No detector rows remain after applying slice_margin")

    init_px = projections.shape[2] / 2.0 if args.init_px is None else args.init_px
    centers = []
    rows = []
    for index in indices:
        kwargs = {
            "ind": int(index),
            "init": float(init_px),
            "tol": float(args.tol),
            "mask": True,
            "ratio": 0.5,
            "sinogram_order": False,
            "verbose": False,
        }
        try:
            center = tomopy.find_center(
                projections,
                angles,
                algorithm=args.algorithm,
                **kwargs,
            )
        except TypeError:
            center = tomopy.find_center(projections, angles, **kwargs)
        center = float(np.asarray(center).reshape(-1)[0])
        centers.append(center)
        rows.append({"slice": int(index), "center_px": center})

    center_px = float(np.median(centers))
    detector_width = float(projections.shape[2])
    offset_px = detector_width / 2.0 - center_px
    detector_pixel_u = float(scanner["sDetector"][1]) / float(scanner["nDetector"][1])
    offset = offset_px * detector_pixel_u

    output = args.output or (args.data / "fbp_center_scan_tomopy.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice", "center_px"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"TomoPy median center: {center_px:.6f} pixels")
    print(f"Absolute detector offset: {offset_px:+.6f} pixels")
    print(f"Set scanner.offDetector[0] to approximately {offset:.8g}")
    print(f"Center spread: p10={np.percentile(centers, 10):.6f}, "
          f"p90={np.percentile(centers, 90):.6f} pixels")
    print(f"Saved per-slice results to {output}")


if __name__ == "__main__":
    main(parse_args())
