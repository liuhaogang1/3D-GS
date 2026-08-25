"""Convert a 3D NumPy array stored in ``.npy`` format to a multi-page TIFF.

Examples
--------
保存浮点体数据，默认输入轴顺序为 Z, Y, X::

    python npytotiff.py volume.npy -o volume.tiff

将当前轴顺序为 X, Y, Z 的数据转换为 TIFF 的 Z, Y, X 顺序，并保存为
16-bit 图像::

    python npytotiff.py volume.npy -o volume_uint16.tiff \
        --input-axes XYZ --dtype uint16
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


AXIS_PERMUTATIONS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a 3D .npy array to a multi-page TIFF stack."
    )
    parser.add_argument("input", type=Path, help="输入的 3D .npy 文件")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出 TIFF 文件；省略时使用输入文件同名的 .tiff 文件",
    )
    parser.add_argument(
        "--input-axes",
        default="ZYX",
        type=str.upper,
        choices=AXIS_PERMUTATIONS,
        help="输入数组的轴顺序，输出统一为 ZYX；默认：ZYX",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "uint8", "uint16"),
        default="float32",
        help="输出数据类型；默认 float32，uint8/uint16 会进行范围映射",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help="映射到整数 TIFF 时的输入下限；默认使用数据最小有限值",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="映射到整数 TIFF 时的输入上限；默认使用数据最大有限值",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "deflate"),
        default="none",
        help="TIFF 压缩方式；默认 none",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        metavar=("SZ", "SY", "SX"),
        default=None,
        help="可选体素尺寸，按 Z Y X 顺序写入 TIFF 元数据",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的输出文件",
    )
    return parser.parse_args()


def to_zyx(array, input_axes):
    """Reorder an array whose axes are named by ``input_axes`` to ZYX."""
    if array.ndim != 3:
        raise ValueError(f"只支持 3D 数组，当前数组维度为 {array.ndim}: {array.shape}")
    order = tuple(input_axes.index(axis) for axis in "ZYX")
    return np.transpose(array, order)


def convert_dtype(array, dtype, min_value=None, max_value=None):
    if dtype == "float32":
        return np.asarray(array, dtype=np.float32)

    output_dtype = np.dtype(dtype)
    output_max = np.iinfo(output_dtype).max
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("数组中没有有限数值，无法转换为整数 TIFF")

    data_min = float(np.min(array[finite])) if min_value is None else min_value
    data_max = float(np.max(array[finite])) if max_value is None else max_value
    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
        raise ValueError(
            f"无效的映射范围 [{data_min}, {data_max}]，要求 max-value > min-value"
        )

    data = np.nan_to_num(array, nan=data_min, posinf=data_max, neginf=data_min)
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min) * output_max
    return np.rint(data).astype(output_dtype)


def main():
    args = parse_args()
    if args.input.suffix.lower() != ".npy":
        raise ValueError(f"输入文件必须是 .npy: {args.input}")
    if not args.input.is_file():
        raise FileNotFoundError(f"找不到输入文件: {args.input}")

    output = args.output or args.input.with_suffix(".tiff")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，请添加 --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    array = np.load(args.input, allow_pickle=False)
    array = to_zyx(array, args.input_axes)
    array = convert_dtype(
        array,
        args.dtype,
        min_value=args.min_value,
        max_value=args.max_value,
    )

    metadata = {
        "axes": "ZYX",
        "source": str(args.input.resolve()),
        "input_axes": args.input_axes,
    }
    if args.voxel_size is not None:
        metadata["voxel_size"] = [float(value) for value in args.voxel_size]

    compression = None if args.compression == "none" else args.compression
    tifffile.imwrite(
        output,
        array,
        compression=compression,
        metadata=metadata,
    )
    print(f"已转换: {args.input} -> {output}")
    print(f"形状: {tuple(array.shape)}，dtype: {array.dtype}，轴顺序: ZYX")


if __name__ == "__main__":
    main()
