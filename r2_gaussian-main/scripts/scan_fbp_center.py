"""Search the horizontal detector offset that gives the best FBP consistency.

The scanner metadata uses ``offDetector=[u, v]`` while TIGRE stores it as
``[v, u]`` internally. The search range below is expressed in detector pixels
along the horizontal u direction.
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tigre
import tigre.algorithms as algs

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--u_min_px", type=float, default=-10.0)
    parser.add_argument("--u_max_px", type=float, default=10.0)
    parser.add_argument("--u_step_px", type=float, default=0.5)
    parser.add_argument("--v_offset_px", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main(args):
    with (args.data / "meta_data.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    scanner = metadata["scanner"]
    projections = np.load(args.data / "proj_all.npy").astype(np.float32)
    angles = np.asarray(metadata["source"]["angles_all"], dtype=np.float32)
    if len(projections) != len(angles):
        raise ValueError("proj_all.npy and source.angles_all have different lengths")

    base_offset = np.asarray(scanner.get("offDetector", [0.0, 0.0]), dtype=np.float32)
    detector_pixel_u = float(scanner["sDetector"][1]) / float(scanner["nDetector"][1])
    detector_pixel_v = float(scanner["sDetector"][0]) / float(scanner["nDetector"][0])
    input_tigre = projections[:, ::-1, :]
    candidates = np.arange(
        args.u_min_px, args.u_max_px + args.u_step_px * 0.5, args.u_step_px
    )
    results = []
    best = None

    for u_px in candidates:
        scanner_candidate = copy.deepcopy(scanner)
        offset = base_offset.copy()
        offset[0] = base_offset[0] + u_px * detector_pixel_u
        offset[1] = base_offset[1] + args.v_offset_px * detector_pixel_v
        scanner_candidate["offDetector"] = offset.tolist()
        geo = get_geometry_tigre(scanner_candidate)
        volume = algs.fbp(input_tigre, geo, angles)
        rendered = tigre.Ax(volume, geo, angles)
        residual = rendered - input_tigre
        scale = max(float(np.sqrt(np.mean(input_tigre ** 2))), 1e-8)
        rmse = float(np.sqrt(np.mean(residual ** 2)) / scale)
        result = {"u_offset_px": float(u_px), "u_offset": float(offset[0]), "rmse": rmse}
        results.append(result)
        if best is None or rmse < best["rmse"]:
            best = result
        print(f"u_offset_px={u_px:+.2f}, normalized_rmse={rmse:.6g}")

    output = args.output or (args.data / "fbp_center_scan.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["u_offset_px", "u_offset", "rmse"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Best horizontal detector offset: {best['u_offset_px']:+.3f} pixels")
    print(f"Set scanner.offDetector[0] to approximately {best['u_offset']:.8g}")
    print(f"Saved scan results to {output}")


if __name__ == "__main__":
    main(parse_args())
