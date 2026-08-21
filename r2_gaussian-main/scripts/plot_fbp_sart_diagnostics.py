"""Create separate robust X/Y/Z slice grids for FBP and SART experiments."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FBP_GT = Path("output/refcorr_parallel_75_v128/test/iter_30000/vol_gt.npy")
DEFAULT_FBP_PRED = Path("output/refcorr_parallel_75_v128/test/iter_30000/vol_pred.npy")
DEFAULT_SART_GT = Path("output/refcorr_sparse_50_sart128/test/iter_30000/vol_gt.npy")
DEFAULT_SART_PRED = Path("output/refcorr_sparse_50_sart128/test/iter_30000/vol_pred.npy")
DEFAULT_OUTPUT_DIR = Path("output/diagnostic_slices")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot robustly normalized X/Y/Z slices for FBP and SART experiments."
    )
    parser.add_argument("--fbp_gt", type=Path, default=DEFAULT_FBP_GT)
    parser.add_argument("--fbp_pred", type=Path, default=DEFAULT_FBP_PRED)
    parser.add_argument("--sart_gt", type=Path, default=DEFAULT_SART_GT)
    parser.add_argument("--sart_pred", type=Path, default=DEFAULT_SART_PRED)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--positions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Relative slice positions in [0, 1]. Default: 0.25 0.5 0.75.",
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=[1.0, 99.7],
        metavar=("LOW", "HIGH"),
        help="Per-volume display percentiles. Default: 1.0 99.7.",
    )
    return parser.parse_args()


def load_volume(path):
    if not path.is_file():
        raise FileNotFoundError(f"Volume not found: {path}")
    volume = np.load(path).astype(np.float32, copy=False)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape} from {path}")
    if not np.isfinite(volume).all():
        raise ValueError(f"Volume contains NaN or Inf: {path}")
    return volume


def display_limits(volume, low_percentile, high_percentile):
    lo, hi = np.percentile(volume, [low_percentile, high_percentile])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def slice_for_axis(volume, axis, index):
    image = np.take(volume, index, axis=axis)
    # Image coordinates: origin at upper-left; transpose keeps X/Y/Z views comparable.
    return image.T


def save_slice_grid(output_path, title, volume, axes, axis_names, indices, low, high):
    vmin, vmax = display_limits(volume, low, high)
    fig, panels = plt.subplots(3, 3, figsize=(10, 10), facecolor="black")
    fig.suptitle(title, color="white", fontsize=16, y=0.995)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.03, wspace=0.03, hspace=0.14)

    for row_index, (axis, axis_name) in enumerate(zip(axes, axis_names)):
        for col_index, index in enumerate(indices[row_index]):
            panel = panels[row_index, col_index]
            panel.imshow(
                slice_for_axis(volume, axis, index),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            panel.set_facecolor("black")
            panel.set_xticks([])
            panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_visible(False)
            panel.set_title(
                f"{axis_name} {index} | p{low:g}-p{high:g}",
                color="white",
                fontsize=10,
                loc="left",
            )

    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic slices: {output_path}")


def main():
    args = parse_args()
    if not args.positions or any(position < 0 or position > 1 for position in args.positions):
        raise ValueError("--positions values must lie between 0 and 1.")
    low, high = args.percentiles
    if not 0 <= low < high <= 100:
        raise ValueError("--percentiles must satisfy 0 <= LOW < HIGH <= 100.")

    volumes = [
        ("fbp_pseudo_gt", "FBP pseudo ground truth (full-view reconstruction)", load_volume(args.fbp_gt)),
        ("fbp_r2_gaussian_pred", "FBP-initialized R2-Gaussian prediction (75 views)", load_volume(args.fbp_pred)),
        ("sart_pseudo_gt", "SART pseudo ground truth (50-view reconstruction)", load_volume(args.sart_gt)),
        ("sart_r2_gaussian_pred", "SART-initialized R2-Gaussian prediction (50 views)", load_volume(args.sart_pred)),
    ]
    shapes = {volume.shape for _, _, volume in volumes}
    if len(shapes) != 1:
        raise ValueError(f"All volumes must have the same shape; got {sorted(shapes)}")

    axes = (0, 1, 2)
    axis_names = ("X", "Y", "Z")
    volume_shape = volumes[0][2].shape
    indices_by_axis = [
        [round(position * (volume_shape[axis] - 1)) for position in args.positions]
        for axis in axes
    ]
    if len(args.positions) != 3:
        raise ValueError("This 3x3 layout needs exactly three --positions values.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, title, volume in volumes:
        save_slice_grid(
            args.output_dir / f"{filename}.png",
            title,
            volume,
            axes,
            axis_names,
            indices_by_axis,
            low,
            high,
        )


if __name__ == "__main__":
    main()
