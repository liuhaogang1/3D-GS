"""Prepare the RefCorr full-angle projections for R2-Gaussian.

The source data is parallel-beam TIFF data. This script creates an R2-Gaussian
meta_data.json dataset with an evenly spaced sparse training split and a
non-overlapping test split. The SART volume is stored as the pseudo ground
truth volume used by R2-Gaussian for initialization/evaluation.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom


ANGLE_RE = re.compile(r"_(\d+(?:\.\d+)?)_Degree_(\d+)of(\d+)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--sart_volume", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--pixel_subsample", type=int, default=4)
    parser.add_argument("--volume_scale", type=float, default=1.0)
    parser.add_argument("--sVoxel", nargs=3, type=float, default=[75.0, 75.0, 75.0])
    return parser.parse_args()


def read_entries(input_dir):
    entries = []
    for path in sorted(input_dir.glob("*.tif")):
        match = ANGLE_RE.search(path.name)
        if match is None:
            raise ValueError(f"Cannot parse angle from {path.name}")
        entries.append((float(match.group(1)), int(match.group(2)), path))
    entries.sort(key=lambda item: item[1])
    # 180 degrees duplicates the 0 degree view for a 180-degree scan.
    entries = [item for item in entries if not np.isclose(item[0], 180.0)]
    if len(entries) != 360:
        raise ValueError(f"Expected 360 unique views after removing 180 degrees, got {len(entries)}")
    return entries


def process_projection(projection, pixel_subsample):
    projection = np.asarray(projection, dtype=np.float32)
    if not np.isfinite(projection).all():
        raise ValueError("Projection contains NaN or Inf")
    if pixel_subsample != 1:
        height = projection.shape[0] // pixel_subsample
        width = projection.shape[1] // pixel_subsample
        projection = zoom(projection, (height / projection.shape[0], width / projection.shape[1]), order=1)
        # Match the repository's real-data preprocessing: crop the wider axis.
        if projection.shape[0] > projection.shape[1]:
            offset = (projection.shape[0] - projection.shape[1]) // 2
            projection = projection[offset:-offset]
        elif projection.shape[1] > projection.shape[0]:
            offset = (projection.shape[1] - projection.shape[0]) // 2
            projection = projection[:, offset:-offset]
    return projection.astype(np.float32)


def write_projection(path, projection, scene_scale):
    np.save(path, (projection * scene_scale).astype(np.float32))


def main():
    args = parse_args()
    entries = read_entries(args.input_dir)
    train_indices = np.linspace(0, len(entries) - 1, args.n_train).round().astype(int)
    train_indices = np.unique(train_indices)
    if len(train_indices) != args.n_train:
        raise ValueError("n_train must be smaller than the number of unique views")
    train_set = set(train_indices.tolist())
    test_indices = [index for index in range(len(entries)) if index not in train_set]

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_dir = output / "proj_train"
    test_dir = output / "proj_test"
    all_dir = output / "proj_all"
    for path in (train_dir, test_dir, all_dir):
        path.mkdir(exist_ok=True)

    # R2-Gaussian works in a normalized scene. A robust SART percentile avoids
    # one hot voxel determining the entire density range.
    volume = np.load(args.sart_volume).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D SART volume, got {volume.shape}")
    robust_max = float(np.percentile(volume, 99.5))
    if robust_max <= 0:
        raise ValueError("SART volume has no positive density")
    volume = np.clip(volume / robust_max, 0.0, 1.0)
    if args.volume_scale != 1.0:
        volume = np.clip(volume * args.volume_scale, 0.0, 1.0)
    np.save(output / "vol_sart.npy", volume)
    np.save(output / "vol_gt.npy", volume)

    first = process_projection(tifffile.imread(entries[0][2]), args.pixel_subsample)
    detector_shape = list(first.shape)
    # The metadata uses R2-Gaussian's normalized scene coordinates.
    scanner = {
        "mode": "parallel",
        "DSD": 7.0,
        "DSO": 5.0,
        "nDetector": detector_shape,
        "sDetector": [2.0, 2.0 * detector_shape[1] / detector_shape[0]],
        "nVoxel": list(volume.shape),
        "sVoxel": [2.0, 2.0, 2.0],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "accuracy": 0.5,
        "totalAngle": 180.0,
        "startAngle": 0.0,
        "noise": False,
        "filter": None,
    }
    projection_scale = 1.0

    def split_record(index, directory, prefix):
        angle_deg, _, source = entries[index]
        projection = process_projection(tifffile.imread(source), args.pixel_subsample)
        if projection.shape != tuple(detector_shape):
            raise ValueError(f"Inconsistent projection shape in {source.name}")
        target = directory / f"{index:04d}.npy"
        write_projection(target, projection, projection_scale)
        return {"file_path": f"{prefix}/{target.name}", "angle": float(np.deg2rad(angle_deg))}

    proj_train = [split_record(index, train_dir, "proj_train") for index in train_indices]
    proj_test = [split_record(index, test_dir, "proj_test") for index in test_indices]
    for index in range(len(entries)):
        split_record(index, all_dir, "proj_all")

    meta_data = {
        "scanner": scanner,
        "vol": "vol_sart.npy",
        "radius": 1.0,
        "bbox": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "proj_train": proj_train,
        "proj_test": proj_test,
        "source": {
            "input_dir": str(args.input_dir.resolve()),
            "sart_volume": str(args.sart_volume.resolve()),
            "unique_views": len(entries),
            "removed_duplicate_endpoint": "180 degrees",
            "pixel_subsample": args.pixel_subsample,
            "volume_percentile_scale": robust_max,
            "volume_scale": args.volume_scale,
        },
    }
    with open(output / "meta_data.json", "w", encoding="utf-8") as handle:
        json.dump(meta_data, handle, indent=2)
    print(f"Saved {len(proj_train)} training and {len(proj_test)} test views to {output}")
    print(f"Projection shape: {detector_shape}; volume shape: {volume.shape}")


if __name__ == "__main__":
    main()
