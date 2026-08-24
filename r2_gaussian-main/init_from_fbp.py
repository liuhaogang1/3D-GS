"""Create an R2-Gaussian initialization point cloud from an FBP volume."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output point cloud. Defaults to <volume_dir>/init_<volume_dir_name>.npy.",
    )
    parser.add_argument("--n_points", type=int, default=50000)
    parser.add_argument("--density_thresh", type=float, default=0.05)
    parser.add_argument("--density_rescale", type=float, default=0.15)
    parser.add_argument(
        "--normalize_percentile",
        type=float,
        default=99.5,
        help="Normalize positive volume values by this percentile; 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    volume = np.load(args.volume).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}")
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.clip(volume, 0.0, None)

    if args.normalize_percentile > 0:
        positive = volume[volume > 0]
        if positive.size == 0:
            raise ValueError("Volume has no positive voxel")
        scale = float(np.percentile(positive, args.normalize_percentile))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("Invalid volume normalization scale")
        volume = np.clip(volume / scale, 0.0, 1.0)

    mask = volume > args.density_thresh
    indices = np.argwhere(mask)
    if len(indices) < args.n_points:
        raise ValueError(
            f"Only {len(indices)} voxels exceed density_thresh={args.density_thresh}; "
            "lower the threshold or reduce n_points."
        )

    rng = np.random.default_rng(args.seed)
    selected = indices[rng.choice(len(indices), args.n_points, replace=False)]
    shape = np.asarray(volume.shape, dtype=np.float32)
    # Volume coordinates are already in the R2-Gaussian [-1, 1]^3 convention.
    positions = (selected.astype(np.float32) + 0.5) / shape * 2.0 - 1.0
    densities = volume[tuple(selected.T)] * args.density_rescale
    point_cloud = np.concatenate([positions, densities[:, None]], axis=1).astype(np.float32)

    canonical_output = args.volume.parent / f"init_{args.volume.parent.name}.npy"
    output = args.output if args.output is not None else canonical_output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, point_cloud)
    print(f"Saved {point_cloud.shape[0]} points to {output}")

    # initialize_gaussian() looks for init_<dataset_name>.npy when --ply_path
    # is omitted. Keep that convention even when a custom output name is used.
    if output.resolve() != canonical_output.resolve():
        np.save(canonical_output, point_cloud)
        print(f"Saved canonical initialization to {canonical_output}")
    print(f"density range: {densities.min():.6g} ~ {densities.max():.6g}")


if __name__ == "__main__":
    main()
