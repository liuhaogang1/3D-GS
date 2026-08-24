"""Create an R2-Gaussian dataset from parallel-beam TIFF projections.

This script is intentionally independent of Scene/initialize_pcd.py.  It reads
TIFF files, performs a parallel-beam FBP reconstruction with TIGRE, writes the
pseudo ground-truth volume, and optionally creates the train/test projection
splits and metadata consumed later by train.py.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom
import tigre
import tigre.algorithms as algs


def parse_config(path):
    """Read the angle information from a FIPS-style config.txt if present."""
    values = {}
    if path is None or not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def build_angles(args, n_views):
    config = parse_config(args.config)
    start = float(config.get("AngleFirst", args.angle_start))
    interval = float(config.get("AngleInterval", args.angle_interval))
    configured_count = int(config.get("NumberImages", n_views))
    if configured_count != n_views:
        raise ValueError(
            f"config declares {configured_count} views, but found {n_views} TIFF files"
        )
    angles_deg = start + np.arange(n_views, dtype=np.float32) * interval
    # A 180-degree endpoint duplicates the zero-degree view for parallel beam.
    if (
        not args.keep_duplicate_endpoint
        and len(angles_deg) > 1
        and np.isclose(angles_deg[-1] - angles_deg[0], 180.0, atol=1e-4)
    ):
        angles_deg = angles_deg[:-1]
    return np.deg2rad(angles_deg).astype(np.float32)


def process_projection(image, pixel_subsample):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("A TIFF projection contains no finite pixels")
    # Keep the script usable with raw files containing Inf/NaN while avoiding a
    # single saturated pixel dominating the reconstruction.
    cap = np.percentile(image[finite], 99.9)
    image = np.nan_to_num(image, nan=0.0, posinf=cap, neginf=0.0)
    if pixel_subsample != 1:
        h, w = image.shape
        new_h = max(1, h // pixel_subsample)
        new_w = max(1, w // pixel_subsample)
        image = zoom(image, (new_h / h, new_w / w), order=1)
        if image.shape[0] > image.shape[1]:
            offset = (image.shape[0] - image.shape[1]) // 2
            image = image[offset:-offset, :]
        elif image.shape[1] > image.shape[0]:
            offset = (image.shape[1] - image.shape[0]) // 2
            image = image[:, offset:-offset]
    return image.astype(np.float32)


def build_parallel_geometry(args, detector_shape):
    height, width = detector_shape
    geo = tigre.geometry(
        mode="parallel", nVoxel=np.asarray(args.nVoxel[::-1], dtype=np.int32)
    )
    geo.DSD = float(args.DSD)
    geo.DSO = float(args.DSO)
    geo.nDetector = np.asarray([height, width], dtype=np.int32)
    geo.sDetector = np.asarray(
        [height * args.pixel_size, width * args.pixel_size], dtype=np.float32
    )
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.nVoxel = np.asarray(args.nVoxel[::-1], dtype=np.int32)
    geo.sVoxel = np.asarray(args.sVoxel[::-1], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.asarray(args.offOrigin[::-1], dtype=np.float32)
    geo.offDetector = np.asarray(
        [args.offDetector[1], args.offDetector[0]], dtype=np.float32
    )
    geo.accuracy = float(args.accuracy)
    geo.filter = None
    return geo


def save_split(output, name, indices, projections, angles):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        filename = f"{index:04d}.npy"
        np.save(directory / filename, projections[index])
        records.append(
            {
                "file_path": f"{name}/{filename}",
                "angle": float(angles[index]),
            }
        )
    return records


def main(args):
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    if args.config:
        args.config = Path(args.config)
    else:
        config_candidates = sorted(args.input_dir.glob("*.txt"))
        args.config = config_candidates[0] if len(config_candidates) == 1 else args.input_dir / "config.txt"

    paths = sorted(args.input_dir.glob("*.tif"))
    paths += sorted(args.input_dir.glob("*.tiff"))
    if not paths:
        raise ValueError(f"No TIFF files found in {args.input_dir}")

    projections = np.stack(
        [process_projection(tifffile.imread(path), args.pixel_subsample) for path in paths],
        axis=0,
    )
    projections *= np.float32(args.projection_scale)
    if args.log_projection:
        projections = np.clip(projections, args.log_eps, None)
        projections = -np.log(projections)
    if not np.isfinite(projections).all():
        raise ValueError("Processed projections contain NaN or Inf")

    angles = build_angles(args, len(projections))
    if len(angles) != len(projections):
        projections = projections[: len(angles)]
    if len(angles) < 2:
        raise ValueError("At least two projection angles are required")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "proj_all.npy", projections)

    geo = build_parallel_geometry(args, projections.shape[1:])
    # TIGRE expects the detector row order opposite to the repository TIFF order.
    volume_tigre = algs.fbp(projections[:, ::-1, :], geo, angles)
    volume = np.transpose(volume_tigre, (2, 1, 0)).astype(np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.clip(volume, 0.0, None)
    np.save(output / "vol_fbp.npy", volume)

    rng = random.Random(args.seed)
    train_indices = np.linspace(0, len(angles) - 1, args.n_train).round().astype(int)
    train_indices = np.unique(train_indices)
    if len(train_indices) != args.n_train:
        raise ValueError("n_train must be smaller than the number of views")
    remaining = sorted(set(range(len(angles))) - set(train_indices.tolist()))
    if args.n_test > len(remaining):
        raise ValueError("n_test is larger than the remaining views")
    test_indices = sorted(rng.sample(remaining, args.n_test))

    train_records = save_split(output, "proj_train", train_indices, projections, angles)
    test_records = save_split(output, "proj_test", test_indices, projections, angles)

    scanner = {
        "mode": "parallel",
        "DSD": float(args.DSD),
        "DSO": float(args.DSO),
        "nDetector": [int(x) for x in projections.shape[1:]],
        "sDetector": [
            float(projections.shape[1] * args.pixel_size),
            float(projections.shape[2] * args.pixel_size),
        ],
        "nVoxel": [int(x) for x in args.nVoxel],
        "sVoxel": [float(x) for x in args.sVoxel],
        "offOrigin": [float(x) for x in args.offOrigin],
        "offDetector": [float(x) for x in args.offDetector],
        "accuracy": float(args.accuracy),
        "totalAngle": float(np.rad2deg(angles[-1] - angles[0])),
        "startAngle": float(np.rad2deg(angles[0])),
        "noise": False,
        "filter": None,
    }
    metadata = {
        "scanner": scanner,
        "vol": "vol_fbp.npy",
        "radius": 1.0,
        "bbox": [
            (np.asarray(args.offOrigin) - np.asarray(args.sVoxel) / 2).tolist(),
            (np.asarray(args.offOrigin) + np.asarray(args.sVoxel) / 2).tolist(),
        ],
        "proj_train": train_records,
        "proj_test": test_records,
        "source": {
            "input_dir": str(args.input_dir.resolve()),
            "algorithm": "TIGRE parallel-beam FBP",
            "n_views": len(angles),
            "removed_duplicate_endpoint": not args.keep_duplicate_endpoint,
        },
    }
    with (output / "meta_data.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved FBP volume: {output / 'vol_fbp.npy'}")
    print(f"Saved {len(train_records)} training and {len(test_records)} test views")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--angle_start", type=float, default=0.0)
    parser.add_argument("--angle_interval", type=float, default=0.5)
    parser.add_argument("--keep_duplicate_endpoint", action="store_true")
    parser.add_argument("--pixel_subsample", type=int, default=1)
    parser.add_argument("--projection_scale", type=float, default=1.0)
    parser.add_argument("--log_projection", action="store_true")
    parser.add_argument("--log_eps", type=float, default=1e-6)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--nVoxel", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--sVoxel", nargs=3, type=float, default=[2.0, 2.0, 2.0])
    parser.add_argument("--offOrigin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--offDetector", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--DSD", type=float, default=7.0)
    parser.add_argument("--DSO", type=float, default=5.0)
    parser.add_argument("--pixel_size", type=float, default=1.0)
    parser.add_argument("--accuracy", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
