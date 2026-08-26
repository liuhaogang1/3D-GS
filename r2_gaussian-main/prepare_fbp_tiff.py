"""Prepare parallel-beam TIFF projections and reconstruct an FBP volume.

The reconstruction input must be line integrals. For transmission/intensity
TIFFs this script therefore applies ``-log(I / I0)`` before TIGRE FBP.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, zoom
import tigre
import tigre.algorithms as algs


def save_fbp_preview(output, volume, s_voxel, preview_percentiles=(1.0, 99.5)):
    """Save immediately viewable orthogonal slices and a correctly scaled NIfTI."""
    import SimpleITK as sitk
    from PIL import Image

    volume = np.asarray(volume, dtype=np.float32)
    positive = volume[np.isfinite(volume) & (volume > 0)]
    if positive.size == 0:
        raise ValueError("FBP volume contains no positive voxels for preview")
    low, high = np.percentile(positive, preview_percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(positive.min())
        high = float(positive.max())
    display = np.clip((volume - low) / (high - low), 0.0, 1.0)
    center = tuple(int(size // 2) for size in volume.shape)

    # The array convention is (x, y, z); save all three central planes.
    planes = {
        "x": display[center[0], :, :],
        "y": display[:, center[1], :],
        "z": display[:, :, center[2]],
    }
    for axis, plane in planes.items():
        Image.fromarray(np.round(plane * 255.0).astype(np.uint8)).save(
            output / f"vol_fbp_preview_{axis}.png"
        )

    # Keep the full dynamic range for Slicer; PNGs are only for quick viewing.
    preview_image = sitk.GetImageFromArray(volume.transpose(2, 0, 1))
    spacing = tuple(float(size) / float(count) for size, count in zip(s_voxel, volume.shape))
    preview_image.SetSpacing(spacing)
    sitk.WriteImage(preview_image, str(output / "vol_fbp_preview.nii.gz"))
    return float(low), float(high)


def parse_config(path):
    values = {}
    if path is not None:
        path = Path(path)
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
    if (
        not args.keep_duplicate_endpoint
        and len(angles_deg) > 1
        and np.isclose(angles_deg[-1] - angles_deg[0], 180.0, atol=1e-4)
    ):
        angles_deg = angles_deg[:-1]
    return np.deg2rad(angles_deg).astype(np.float32)


def fill_nearest(image, bad):
    if not bad.any():
        return image
    good = ~bad
    if not good.any():
        raise ValueError("A projection contains no valid pixels")
    _, indices = distance_transform_edt(bad, return_distances=True, return_indices=True)
    image[bad] = image[tuple(indices[:, bad])]
    return image


def process_projection(image, args):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("A TIFF projection contains no finite pixels")

    finite_values = image[finite]
    cap = float(np.percentile(finite_values, args.clip_percentile))
    image = np.nan_to_num(image, nan=0.0, posinf=cap, neginf=0.0)

    if args.input_type in {"transmission", "intensity"}:
        bad = image <= 0
        if args.zero_policy == "nearest":
            image = fill_nearest(image, bad)
        elif args.zero_policy == "clip":
            positive = image[image > 0]
            if positive.size == 0:
                raise ValueError("A projection contains no positive intensity")
            image = np.maximum(image, np.percentile(positive, 0.1))
        image = np.maximum(image, args.log_eps)
        if args.i0 > 0:
            i0 = args.i0
        else:
            positive = image[image > 0]
            i0 = float(np.percentile(positive, args.i0_percentile))
        if i0 <= 0 or not np.isfinite(i0):
            raise ValueError(f"Invalid I0 estimate: {i0}")
        image = -np.log(image / i0)
        image = np.maximum(image, 0.0)
    else:
        image = np.maximum(image, 0.0)

    if args.shift_v != 0:
        shifted = np.zeros_like(image)
        if args.shift_v > 0:
            shifted[:-args.shift_v] = image[args.shift_v:]
        else:
            shift = -args.shift_v
            shifted[shift:] = image[:-shift]
        image = shifted

    if args.pixel_subsample != 1:
        height, width = image.shape
        new_height = max(1, height // args.pixel_subsample)
        new_width = max(1, width // args.pixel_subsample)
        image = zoom(
            image,
            (new_height / height, new_width / width),
            order=args.resize_order,
        )
        if image.shape[0] > image.shape[1]:
            offset = (image.shape[0] - image.shape[1]) // 2
            image = image[offset : image.shape[0] - offset]
        elif image.shape[1] > image.shape[0]:
            offset = (image.shape[1] - image.shape[0]) // 2
            image = image[:, offset : image.shape[1] - offset]

    image *= np.float32(args.projection_scale)
    if not np.isfinite(image).all():
        raise ValueError("Processed projection contains NaN or Inf")
    return image.astype(np.float32)


def build_parallel_geometry(args, detector_shape):
    height, width = detector_shape
    effective_pixel_size = args.pixel_size * args.pixel_subsample
    geo = tigre.geometry(
        mode="parallel", nVoxel=np.asarray(args.nVoxel[::-1], dtype=np.int32)
    )
    geo.DSD = float(args.DSD)
    geo.DSO = float(args.DSO)
    geo.nDetector = np.asarray([height, width], dtype=np.int32)
    geo.sDetector = np.asarray(
        [height * effective_pixel_size, width * effective_pixel_size],
        dtype=np.float32,
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
    geo.filter = args.filter
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
    args.config = Path(args.config) if args.config else None
    if args.config is None:
        config_candidates = sorted(args.input_dir.glob("*.txt"))
        args.config = config_candidates[0] if len(config_candidates) == 1 else None

    paths = sorted(args.input_dir.glob("*.tif"))
    paths += sorted(args.input_dir.glob("*.tiff"))
    if not paths:
        raise ValueError(f"No TIFF files found in {args.input_dir}")

    projections_all = np.stack(
        [process_projection(tifffile.imread(path), args) for path in paths], axis=0
    )
    angles = build_angles(args, len(projections_all))
    config_values = parse_config(args.config)
    angle_interval = float(config_values.get("AngleInterval", args.angle_interval))
    removed_endpoint = (
        not args.keep_duplicate_endpoint
        and len(paths) == len(angles) + 1
        and np.isclose(angle_interval * len(angles), 180.0, atol=1e-3)
    )
    endpoint_pair = projections_all[[0, -1]].copy() if removed_endpoint else None
    projections = projections_all[: len(angles)]
    if len(angles) < 2:
        raise ValueError("At least two projection angles are required")
    total_angle = angle_interval * len(angles) if removed_endpoint else float(
        np.rad2deg(angles[-1] - angles[0])
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "proj_all.npy", projections)
    if endpoint_pair is not None:
        np.save(output / "proj_endpoint_pair.npy", endpoint_pair)

    geo = build_parallel_geometry(args, projections.shape[1:])
    volume_tigre = algs.fbp(projections[:, ::-1, :], geo, angles)
    volume_raw = np.transpose(volume_tigre, (2, 1, 0)).astype(np.float32)
    volume_raw = np.nan_to_num(volume_raw, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.clip(volume_raw, 0.0, None)
    np.save(output / "vol_fbp_raw.npy", volume_raw)
    np.save(output / "vol_fbp.npy", volume)
    preview_low, preview_high = save_fbp_preview(
        output,
        volume,
        args.sVoxel,
    )

    rng = random.Random(args.seed)
    train_indices = np.linspace(0, len(angles) - 1, args.n_train).round().astype(int)
    train_indices = np.unique(train_indices)
    if len(train_indices) != args.n_train:
        raise ValueError("n_train must be smaller than the number of unique views")
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
            float(projections.shape[1] * args.pixel_size * args.pixel_subsample),
            float(projections.shape[2] * args.pixel_size * args.pixel_subsample),
        ],
        "nVoxel": [int(x) for x in args.nVoxel],
        "sVoxel": [float(x) for x in args.sVoxel],
        "offOrigin": [float(x) for x in args.offOrigin],
        "offDetector": [float(x) for x in args.offDetector],
        "accuracy": float(args.accuracy),
        "totalAngle": float(total_angle),
        "startAngle": float(np.rad2deg(angles[0])),
        "noise": False,
        "filter": args.filter,
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
            "angles_all": [float(angle) for angle in angles],
            "removed_duplicate_endpoint": bool(removed_endpoint),
            "input_type": args.input_type,
            "i0": float(args.i0),
            "i0_percentile": float(args.i0_percentile),
            "zero_policy": args.zero_policy,
            "shift_v": int(args.shift_v),
            "pixel_subsample": int(args.pixel_subsample),
            "input_pixel_size": float(args.pixel_size),
            "effective_pixel_size": float(args.pixel_size * args.pixel_subsample),
            "projection_scale": float(args.projection_scale),
            "endpoint_pair": "proj_endpoint_pair.npy" if endpoint_pair is not None else None,
        },
    }
    with (output / "meta_data.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved FBP volume: {output / 'vol_fbp.npy'}")
    print(
        f"Saved FBP previews: {output / 'vol_fbp_preview_z.png'} and "
        f"{output / 'vol_fbp_preview.nii.gz'} "
        f"(window {preview_low:.6g} ~ {preview_high:.6g})"
    )
    print(
        f"Saved {len(angles)} full views, {len(train_records)} training views, "
        f"and {len(test_records)} test views"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--angle_start", type=float, default=0.0)
    parser.add_argument("--angle_interval", type=float, default=0.5)
    parser.add_argument("--keep_duplicate_endpoint", action="store_true")
    parser.add_argument(
        "--input_type",
        choices=["transmission", "intensity", "line_integral"],
        default="transmission",
        help="transmission/intensity applies -log(I/I0); line_integral skips it",
    )
    parser.add_argument(
        "--i0", type=float, default=1.0,
        help="Incident intensity; use 0 to estimate per view",
    )
    parser.add_argument("--i0_percentile", type=float, default=99.5)
    parser.add_argument(
        "--zero_policy", choices=["nearest", "clip", "keep"], default="nearest"
    )
    parser.add_argument("--log_eps", type=float, default=1e-6)
    parser.add_argument("--clip_percentile", type=float, default=99.9)
    parser.add_argument(
        "--shift_v", type=int, default=5,
        help="Detector-row shift; 5 matches the old real-data pipeline",
    )
    parser.add_argument("--pixel_subsample", type=int, default=1)
    parser.add_argument("--resize_order", type=int, choices=[0, 1, 3], default=1)
    parser.add_argument("--projection_scale", type=float, default=1.0)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--nVoxel", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--sVoxel", nargs=3, type=float, default=[2.0, 2.0, 2.0])
    parser.add_argument("--offOrigin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument(
        "--offDetector", nargs=2, type=float, default=[0.0, 0.0],
        help="[u, v] detector offsets in physical units",
    )
    parser.add_argument("--DSD", type=float, default=7.0)
    parser.add_argument("--DSO", type=float, default=5.0)
    parser.add_argument("--pixel_size", type=float, default=1.0)
    parser.add_argument("--accuracy", type=float, default=0.5)
    parser.add_argument(
        "--filter",
        choices=["ram_lak", "shepp_logan", "cosine", "hamming", "hann"],
        default="hann",
    )
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
