"""Create an R2-Gaussian initialization point cloud from a SART volume."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_points", type=int, default=50000)
    parser.add_argument("--density_thresh", type=float, default=0.05)
    parser.add_argument("--density_rescale", type=float, default=0.15)
    args = parser.parse_args()

    volume = np.load(args.volume).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}")
    mask = np.isfinite(volume) & (volume > args.density_thresh)
    indices = np.argwhere(mask)
    if len(indices) < args.n_points:
        raise ValueError(
            f"Only {len(indices)} voxels exceed density_thresh={args.density_thresh}; "
            "lower the threshold or reduce n_points."
        )

    rng = np.random.default_rng(0)
    selected = indices[rng.choice(len(indices), args.n_points, replace=False)]
    shape = np.asarray(volume.shape, dtype=np.float32)
    # R2-Gaussian normalized scene coordinates: volume spans [-1, 1]^3.
    positions = (selected.astype(np.float32) + 0.5) / shape * 2.0 - 1.0
    densities = volume[tuple(selected.T)] * args.density_rescale
    point_cloud = np.concatenate([positions, densities[:, None]], axis=1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, point_cloud)
    print(f"Saved {point_cloud.shape[0]} points to {args.output}")
    print(f"density range: {densities.min():.6g} ~ {densities.max():.6g}")


if __name__ == "__main__":
    main()
