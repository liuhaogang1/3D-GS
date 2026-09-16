"""Voxelize an R2-Gaussian initialization point cloud into a volume.

The input point cloud is the four-column ``.npy`` file produced by
``init_from_fbp.py``: ``x, y, z, density``.  Scanner geometry is read from
the dataset ``meta_data.json`` so the output uses the same normalized world
coordinates as training and testing.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def read_scanner(meta_data: Path):
    """Read and normalize scanner geometry in the same way as the dataset loader."""
    with meta_data.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    scanner = metadata["scanner"]
    raw_s_voxel = np.asarray(scanner["sVoxel"], dtype=np.float32)
    scene_scale = 2.0 / float(raw_s_voxel.max())

    n_voxel = np.asarray(scanner["nVoxel"], dtype=np.int32)
    s_voxel = raw_s_voxel * scene_scale
    center = np.asarray(scanner["offOrigin"], dtype=np.float32) * scene_scale
    return n_voxel, s_voxel, center


def save_slice_preview(volume: np.ndarray, output_dir: Path) -> None:
    """Save center axial, coronal, and sagittal PNG previews."""
    import matplotlib.pyplot as plt

    center = [size // 2 for size in volume.shape]
    views = {
        "slice_x.png": volume[center[0], :, :],
        "slice_y.png": volume[:, center[1], :],
        "slice_z.png": volume[:, :, center[2]],
    }
    value_min = float(volume.min())
    value_max = float(volume.max())
    if value_max <= value_min:
        value_max = value_min + 1.0

    for filename, image in views.items():
        figure, axis = plt.subplots(figsize=(7, 6))
        image_handle = axis.imshow(
            image.T,
            cmap="gray",
            origin="lower",
            vmin=value_min,
            vmax=value_max,
        )
        axis.set_title(filename.removesuffix(".png"))
        axis.set_axis_off()
        figure.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an R2-Gaussian initialization .npy point cloud to a volume."
    )
    parser.add_argument(
        "--point_cloud",
        type=Path,
        required=True,
        help="Four-column initialization .npy: x, y, z, density.",
    )
    parser.add_argument(
        "--source_path",
        type=Path,
        default=None,
        help="Dataset directory containing meta_data.json. Defaults to the point-cloud directory.",
    )
    parser.add_argument(
        "--meta_data",
        type=Path,
        default=None,
        help="Explicit meta_data.json path; overrides --source_path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <point_cloud_dir>/init_volume.",
    )
    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.0005,
        help="Minimum Gaussian scale as a fraction of the normalized volume size.",
    )
    parser.add_argument(
        "--scale_max",
        type=float,
        default=0.5,
        help="Maximum Gaussian scale as a fraction of the normalized volume size.",
    )
    parser.add_argument(
        "--scaling_modifier",
        type=float,
        default=1.0,
        help="Scale multiplier passed to the voxelizer.",
    )
    parser.add_argument(
        "--save_slices",
        action="store_true",
        help="Also save center slice_x/slice_y/slice_z PNG previews.",
    )
    args = parser.parse_args()

    point_cloud_path = args.point_cloud.resolve()
    if args.meta_data is not None:
        meta_data_path = args.meta_data.resolve()
    else:
        source_path = (args.source_path or point_cloud_path.parent).resolve()
        meta_data_path = source_path / "meta_data.json"
    if not meta_data_path.exists():
        raise FileNotFoundError(
            f"Cannot find scanner metadata: {meta_data_path}. "
            "Use --source_path or --meta_data."
        )

    point_cloud = np.load(point_cloud_path).astype(np.float32)
    if point_cloud.ndim != 2 or point_cloud.shape[1] < 4:
        raise ValueError(
            f"Expected an Nx4 initialization point cloud, got {point_cloud.shape}"
        )
    point_cloud = point_cloud[:, :4]
    if not np.isfinite(point_cloud).all():
        raise ValueError("Point cloud contains NaN or infinite values.")
    if point_cloud.shape[0] == 0:
        raise ValueError("Point cloud is empty.")
    if np.any(point_cloud[:, 3] < 0):
        raise ValueError("Initialization densities must be non-negative.")
    if args.scale_min <= 0 or args.scale_max <= args.scale_min:
        raise ValueError("Require 0 < scale_min < scale_max.")
    if args.scaling_modifier <= 0:
        raise ValueError("scaling_modifier must be positive.")

    n_voxel, s_voxel, center = read_scanner(meta_data_path)
    output_dir = (args.output_dir or point_cloud_path.parent / "init_volume").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from r2_gaussian.gaussian import GaussianModel, query
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyTorch and the project's CUDA extensions are required for voxelization. "
            "Activate the R2-Gaussian environment before running this script."
        ) from error

    # Training multiplies scale_min/scale_max by max(sVoxel) after normalization.
    volume_to_world = float(s_voxel.max())
    scale_bound = np.asarray(
        [args.scale_min, args.scale_max], dtype=np.float32
    ) * volume_to_world

    gaussians = GaussianModel(scale_bound)
    gaussians.create_from_pcd(
        point_cloud[:, :3], point_cloud[:, 3:4], spatial_lr_scale=1.0
    )
    pipeline = SimpleNamespace(compute_cov3D_python=False, debug=False)

    with torch.no_grad():
        volume = query(
            gaussians,
            center,
            n_voxel,
            s_voxel,
            pipeline,
            scaling_modifier=args.scaling_modifier,
        )["vol"]
        volume = volume.detach().cpu().numpy().astype(np.float32)

    np.save(output_dir / "vol_pred.npy", volume)

    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "SimpleITK is required to write vol_pred.nii.gz. "
            "Install it with: pip install SimpleITK"
        ) from error

    # SimpleITK expects z,y,x array order; preserve physical voxel spacing.
    spacing = tuple(float(size) / float(count) for size, count in zip(s_voxel, n_voxel))
    nii_image = sitk.GetImageFromArray(volume.transpose(2, 1, 0))
    nii_image.SetSpacing(spacing)
    sitk.WriteImage(nii_image, str(output_dir / "vol_pred.nii.gz"))

    if args.save_slices:
        save_slice_preview(volume, output_dir)

    print(f"Loaded {point_cloud.shape[0]:,} initialization points")
    print(f"Volume shape: {tuple(volume.shape)}")
    print(f"Volume range: {volume.min():.6g} ~ {volume.max():.6g}")
    print(f"Saved NumPy volume: {output_dir / 'vol_pred.npy'}")
    print(f"Saved NIfTI volume: {output_dir / 'vol_pred.nii.gz'}")
    if args.save_slices:
        print(f"Saved center slice previews: {output_dir}")


if __name__ == "__main__":
    main()
