# 2026.8.21 修改
#########################################################################################

1.位置：data_generator\real_dataset\generate_data.py 
    第25行新增：geometry_type = "ConeBeam"

    第30行修改：
        原：
            for config_line in f.readlines():
                if "NumberImages" in config_line:

        修：
            for config_line in f.readlines():
                if "GeometryType" in config_line:
                    geometry_type = config_line.split("=")[-1].strip()
                elif "NumberImages" in config_line:
-----------------------------------------------------------------------------------------
    作用：config.txt中记录GeometryType=ParallelBeam，通过config识别平行束锥束

-----------------------------------------------------------------------------------------
2.位置：data_generator\real_dataset\generate_data.py
    第94行修改：
        原：
            proj = scipy.io.loadmat(proj_mat_path)["img"] / proj_rescale * object_scale
            proj = proj.astype(np.float32)
            proj[proj < 0] = 0

        修：
            proj = scipy.io.loadmat(proj_mat_path)["img"].astype(np.float32)

            # Remove invalid values generated during raw-data preprocessing.
            valid = np.isfinite(proj)
            if not valid.any():
                raise ValueError(f"No finite pixel exists in {proj_mat_path}")

            # Inf usually comes from zero transmission / saturated detector pixels.
            # Clip positive Inf to the 99.9 percentile of valid pixels.
            cap = np.percentile(proj[valid], 99.9)
            proj = np.nan_to_num(
                proj,
                nan=0.0,
                posinf=cap,
                neginf=0.0,
            )

            proj = proj / proj_rescale * object_scale
            proj[proj < 0] = 0
-----------------------------------------------------------------------------------------
    作用：原tiff数据经原论文流程经过matlab中转换后存在大量Inf，清理投影中的Inf

        #怀疑是这步导致第一次复现结果不好
-----------------------------------------------------------------------------------------
3.位置：data_generator\real_dataset\generate_data.py
    第149行scanner_cfg中修改：
        原：
            "mode": "cone",

        修：
            "mode": "parallel" if geometry_type == "ParallelBeam" else "cone",
-----------------------------------------------------------------------------------------
    作用：在生成最终的meta_data.json时识别锥束和平行束

-----------------------------------------------------------------------------------------
4.位置：data_generator\real_dataset\generate_data.py
    第179行修改：
        原：
            ct_gt = algs.fdk(projs[:, ::-1, :], geo, angles[::skip])

        修：
            if scanner_cfg["mode"] == "parallel":
                ct_gt = algs.fbp(projs[:, ::-1, :], geo, angles[::skip])
            else:
                ct_gt = algs.fdk(projs[:, ::-1, :], geo, angles[::skip])
-----------------------------------------------------------------------------------------
    作用：
        平行束调用fbp；锥束调用fdk。
    
-----------------------------------------------------------------------------------------
5.位置：data_generator\real_dataset\generate_data.py
    第190行修改：
        原：
            "ct": "vol_gt.npy",

        修：
            "vol": "vol_gt.npy",
-----------------------------------------------------------------------------------------
作用：
    dataset_readers.py中读取meta_data["vol"]，若不修改导致KeyError: 'vol'

-----------------------------------------------------------------------------------------
6.位置：r2_gaussian\utils\ct_utils.py
    第17行recon_volume函数修改：
        原：
            if recon_method == "fdk":
                vol = algs.fdk(projs[:, ::-1, :], geo, angles)
            elif recon_method == "cgls":

        修：
            if recon_method == "fdk":
                # TIGRE uses FBP for parallel-beam geometry.
                if geo.mode == "parallel":
                    vol = algs.fbp(projs[:, ::-1, :], geo, angles)
                else:
                    vol = algs.fdk(projs[:, ::-1, :], geo, angles)
            elif recon_method == "cgls":
-----------------------------------------------------------------------------------------
作用：
    初始化点云时同步切换fbp和fdk
    命令中保持--recon_method fdk但实际代码会对平行束和锥束采用不同算法

-----------------------------------------------------------------------------------------
7.新增prepare_refcorr_r2.py和init_from_sart.py两文件

prepare_refcorr_r2.py作用：
    直接读取原始 TIFF；
    删除 180° 重复端点；
    从 361 张变成 360 个唯一角度；
    4 倍降采样；
    中心裁剪为 176 x 177；
    生成 50 张训练投影和 310 张测试投影；
    读取 SART 体；
    按 99.5% 分位数归一化 SART 体；
    保存 vol_sart.npy 和 vol_gt.npy；
    生成 R2-Gaussian 所需的 meta_data.json

init_from_sart.py作用：
    读取 128³ SART 重建体；
    找出密度大于 0.05 的体素；
    随机选择 50000 个体素；
    将体素坐标映射到 [-1, 1]^3；
    将体素密度乘以 0.15；
    生成init_refcorr_sart128.npy作为初始点云
-----------------------------------------------------------------------------------------

#########################################################################################
# 2026.8.24 修改

新增prepare_fbp_tiff.py init_from_fbp.py文件

prepare_fbp_tiff.py 作用：
    361 张 TIFF
    → 删除 180° 重复端点
    → 360 个角度
    → 降采样/裁剪
    → 并行束 FBP
    → vol_fbp.npy

python prepare_fbp_tiff.py \
  --input_dir data_generator/real_dataset/FIPS_raw/refcorr \
  --output_dir data/real_dataset/parallel_fbp_refcorr \
  --pixel_subsample 4 \
  --projection_scale 0.125 \
  --n_train 50 \
  --n_test 100 \
  --nVoxel 128 128 128 \
  --sVoxel 2 2 2 \
  --DSD 7 \
  --DSO 5 \
  --pixel_size 0.02

python init_from_fbp.py \
  --volume data/real_dataset/parallel_fbp_refcorr/vol_fbp.npy \
  --output data/real_dataset/parallel_fbp_refcorr/init_fbp.npy \
  --n_points 50000 \
  --density_thresh 0.05 \
  --density_rescale 0.15

#########################################################################################
# 2026.8.24 FBP流程进一步修改

一、修改 prepare_fbp_tiff.py
-----------------------------

1. TIFF投影默认按照强度图处理，执行：

    projection = -log(I / I0)

    默认参数：
        --input_type transmission
        --i0 1.0

    如果输入已经是线积分数据，则使用：

        --input_type line_integral

2. 增加无效像素处理：

    - 清理 NaN 和 Inf；
    - 对零值像素使用最近有效像素填充；
    - 对投影值进行有限性检查；
    - 防止无效值在 Ram-Lak 或其他滤波器中被放大。

3. 恢复旧 real-data 流程中的 5 像素 detector 位移：

    --shift_v 5

    正值表示沿 detector 第0维方向向前移动。可以通过 --shift_v 0 关闭。

4. 增加 FBP 滤波器参数：

    --filter {ram_lak,shepp_logan,cosine,hamming,hann}

    默认使用 hann，降低噪声和坏点对 FBP 的影响。

5. FBP 使用全部唯一视角，不再使用训练集中的 50/75 张稀疏投影。

6. 删除平行束 180° 重复端点：

    361 张 TIFF -> 360 张唯一视角
    角度范围：0°、0.5°、...、179.5°
    总覆盖角度：180°

7. 增加输出文件：

    proj_all.npy       全部线积分投影
    vol_fbp_raw.npy    未进行非负裁剪的 FBP 体
    vol_fbp.npy        非负裁剪后的 FBP 体
    meta_data.json     包含角度和预处理参数

二、修改 initialize_pcd.py
----------------------------

1. 默认重建方法从 fdk 改为 fbp。

2. 新增 --recon_split 参数，默认值为 all。

3. 当数据目录中存在 proj_all.npy 和 source.angles_all 时：

    初始化点云使用全部 360 张投影；
    不再使用只有 50/75 张投影的训练集进行 FBP 初始化。

4. 如果找不到 proj_all.npy，则自动回退到训练视角，并输出提示信息。

三、修改 r2_gaussian/utils/ct_utils.py
---------------------------------------

1. 平行束 geometry.mode == parallel 时调用 algs.fbp。

2. 锥束 geometry.mode == cone 时调用 algs.fdk。

3. recon_volume 支持 fbp 和 fdk 两种方法名。

四、修改 scripts/run_traditional_methods.py
--------------------------------------------

1. 传统算法列表中的平行束重建名称由 fdk 改为 fbp。

2. 如果存在 proj_all.npy，则传统重建使用全部唯一视角。

五、修改 data_generator/real_dataset/generate_data.py
------------------------------------------------------

1. 平行束数据自动删除 180° 重复端点。

2. 生成 proj_all.npy，供 FBP 初始化和传统算法使用。

3. 平行束使用 FBP，锥束使用 FDK。

4. 增加 FBP filter 参数，默认使用 hann。

5. 在 meta_data.json 的 source 中保存：

    n_views
    angles_all
    removed_duplicate_endpoint
    algorithm

六、新增 scripts/scan_fbp_center.py
-----------------------------------

作用：搜索 detector 水平方向 u 的旋转中心偏移。

搜索范围默认：

    -10 ~ +10 detector pixels
    步长：0.5 pixel

粗搜索命令：

    python scripts/scan_fbp_center.py \
      --data data/real_dataset/parallel_fbp_refcorr_v2 \
      --u_min_px -10 \
      --u_max_px 10 \
      --u_step_px 1

精搜索命令：

    python scripts/scan_fbp_center.py \
      --data data/real_dataset/parallel_fbp_refcorr_v2 \
      --u_min_px -2 \
      --u_max_px 2 \
      --u_step_px 0.25

输出结果中的：

    Best horizontal detector offset

即推荐的水平 detector 偏移像素值。对应配置中的：

    scanner.offDetector[0]

该值是 detector 的 u 方向，也就是投影图像的列方向。

七、实际运行验证
------------------

使用项目环境：

    D:\CondaData\envs\r2_gaussian\python.exe

实际运行命令：

    python prepare_fbp_tiff.py \
      --input_dir data_generator/real_dataset/FIPS_raw/refcorr \
      --output_dir data/real_dataset/parallel_fbp_refcorr_v2 \
      --pixel_subsample 4 \
      --projection_scale 0.125 \
      --n_train 50 \
      --n_test 100 \
      --nVoxel 128 128 128 \
      --sVoxel 2 2 2 \
      --DSD 7 \
      --DSO 5 \
      --pixel_size 0.02 \
      --shift_v 5 \
      --filter hann \
      --input_type transmission \
      --i0 1.0 \
      --zero_policy nearest

验证结果：

    TIFF数量：361
    唯一投影数量：360
    投影尺寸：176 x 177
    FBP体尺寸：128 x 128 x 128
    FBP体数据：全部为有限值
    训练投影：50
    测试投影：100
    总角度：180°

初始化命令执行成功：

    python init_from_fbp.py \
      --volume data/real_dataset/parallel_fbp_refcorr_v2/vol_fbp.npy \
      --output data/real_dataset/parallel_fbp_refcorr_v2/init_parallel_fbp_refcorr.npy \
      --n_points 50000 \
      --density_thresh 0.05 \
      --density_rescale 0.15

生成结果：

    50000个 Gaussian 初始化点
    点云形状：(50000, 4)

八、Python语法检查
--------------------

    python -m py_compile `
      prepare_fbp_tiff.py `
      init_from_fbp.py `
      initialize_pcd.py `
      data_generator/real_dataset/generate_data.py `
      r2_gaussian/utils/ct_utils.py `
      scripts/run_traditional_methods.py `
      scripts/scan_fbp_center.py

检查结果：通过。

九、Git提交命令
----------------

    Set-Location D:\TEM\3D-GS\r2_gaussian-main
    git status
    git diff --check

    git add -- `
      prepare_fbp_tiff.py `
      initialize_pcd.py `
      data_generator\real_dataset\generate_data.py `
      r2_gaussian\utils\ct_utils.py `
      scripts\run_traditional_methods.py `
      scripts\scan_fbp_center.py

    git diff --cached --stat
    git commit -m "Improve parallel-beam FBP preprocessing and initialization"
    git status

    # 如需推送到远程仓库：
    git push origin HEAD

说明：data/real_dataset/parallel_fbp_refcorr_v2 位于 .gitignore 中，不会被提交。

#########################################################################################
# 2026.8.24 代码变更前后对比记录

说明：本节以提交 2b27025 作为“原代码”基线，以提交 8983f8d 作为“修改后代码”。
只列出与 FBP 效果、投影角度、初始化和旋转中心直接相关的代码片段。

一、prepare_fbp_tiff.py
-----------------------

修改位置：process_projection()
修改目的：原代码只是清理 NaN/Inf，默认没有把 TIFF 强度转换为 FBP 所需的线积分。

原代码：

```python
def process_projection(image, pixel_subsample):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("A TIFF projection contains no finite pixels")
    cap = np.percentile(image[finite], 99.9)
    image = np.nan_to_num(image, nan=0.0, posinf=cap, neginf=0.0)
    if pixel_subsample != 1:
        h, w = image.shape
        new_h = max(1, h // pixel_subsample)
        new_w = max(1, w // pixel_subsample)
        image = zoom(image, (new_h / h, new_w / w), order=1)
    return image.astype(np.float32)
```

修改后代码：

```python
def fill_nearest(image, bad):
    if not bad.any():
        return image
    good = ~bad
    if not good.any():
        raise ValueError("A projection contains no valid pixels")
    _, indices = distance_transform_edt(
        bad, return_distances=True, return_indices=True
    )
    image[bad] = image[tuple(indices[:, bad])]
    return image


def process_projection(image, args):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("A TIFF projection contains no finite pixels")

    cap = float(np.percentile(image[finite], args.clip_percentile))
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
        i0 = args.i0 if args.i0 > 0 else np.percentile(
            image[image > 0], args.i0_percentile
        )
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
```

修改结果：

    TIFF强度 -> 无效值处理 -> -log(I/I0) -> 5像素位移
              -> 降采样/裁剪 -> 投影缩放 -> FBP输入

注意：如果输入已经是线积分，不能再次执行 -log，应使用：

```powershell
--input_type line_integral
```

二、prepare_fbp_tiff.py 的 FBP 几何和输出
------------------------------------------

修改位置：build_parallel_geometry() 和 main()

原代码：

```python
geo.accuracy = float(args.accuracy)
geo.filter = None

projections *= np.float32(args.projection_scale)
if args.log_projection:
    projections = np.clip(projections, args.log_eps, None)
    projections = -np.log(projections)

volume_tigre = algs.fbp(projections[:, ::-1, :], geo, angles)
volume = np.transpose(volume_tigre, (2, 1, 0)).astype(np.float32)
volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
volume = np.clip(volume, 0.0, None)
np.save(output / "vol_fbp.npy", volume)
```

修改后代码：

```python
geo.accuracy = float(args.accuracy)
geo.filter = args.filter

volume_tigre = algs.fbp(projections[:, ::-1, :], geo, angles)
volume_raw = np.transpose(volume_tigre, (2, 1, 0)).astype(np.float32)
volume_raw = np.nan_to_num(
    volume_raw, nan=0.0, posinf=0.0, neginf=0.0
)
volume = np.clip(volume_raw, 0.0, None)
np.save(output / "vol_fbp_raw.npy", volume_raw)
np.save(output / "vol_fbp.npy", volume)
```

修改原因：

1. 原代码默认 `geo.filter = None`，TIGRE实际使用 Ram-Lak；对噪声和坏点比较敏感。
2. 修改后默认使用 `hann`，同时保留 `vol_fbp_raw.npy`，便于区分滤波伪影和非负裁剪影响。
3. `-log(I/I0)` 移入单帧投影预处理，避免先缩放强度再错误地对投影执行对数变换。

滤波器参数定义：

```python
parser.add_argument(
    "--filter",
    choices=["ram_lak", "shepp_logan", "cosine", "hamming", "hann"],
    default="hann",
)
```

三、prepare_fbp_tiff.py 的角度端点和元数据
--------------------------------------------

原代码虽然删除了最后一张重复投影，但元数据只使用：

```python
"totalAngle": float(np.rad2deg(angles[-1] - angles[0])),
```

这会把 `0°...179.5°` 的离散数组写成约 `179.5°`，容易误解实际扫描覆盖范围。

修改后代码：

```python
config_values = parse_config(args.config)
angle_interval = float(
    config_values.get("AngleInterval", args.angle_interval)
)
removed_endpoint = bool(
    not args.keep_duplicate_endpoint
    and len(paths) == len(angles) + 1
    and np.isclose(angle_interval * len(angles), 180.0, atol=1e-3)
)
total_angle = (
    angle_interval * len(angles)
    if removed_endpoint
    else float(np.rad2deg(angles[-1] - angles[0]))
)
```

并在 `meta_data.json` 中保存：

```python
"totalAngle": float(total_angle),
"source": {
    "n_views": len(angles),
    "angles_all": [float(angle) for angle in angles],
    "removed_duplicate_endpoint": bool(removed_endpoint),
    "input_type": args.input_type,
    "shift_v": int(args.shift_v),
    "pixel_subsample": int(args.pixel_subsample),
    "projection_scale": float(args.projection_scale),
},
```

对 refcorr 数据的结果：

    原始 TIFF：361张
    删除180°重复端点后：360张
    角度：0°到179.5°，间隔0.5°
    扫描覆盖：180°平行束完整数据

四、initialize_pcd.py
---------------------

修改位置：InitParams 和 main()
修改目的：原代码从 Scene 中读取训练相机，因此 FBP 初始化可能只用50/75张稀疏视角。

原代码：

```python
class InitParams(ParamGroup):
    def __init__(self, parser):
        self.recon_method = "fdk"
        self.n_points = 50000
        self.density_thresh = 0.05
        self.density_rescale = 0.15

...

train_cameras = scene.getTrainCameras()
projs_train = np.concatenate(
    [t2a(cam.original_image) for cam in train_cameras], axis=0
)
angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
```

修改后代码：

```python
class InitParams(ParamGroup):
    def __init__(self, parser):
        self.recon_method = "fbp"
        self.recon_split = "all"
        self.n_points = 50000
        self.density_thresh = 0.05
        self.density_rescale = 0.15

...

train_cameras = scene.getTrainCameras()
if init_args.recon_split == "all":
    full_projection_path = Path(data_path) / "proj_all.npy"
    metadata_path = Path(data_path) / "meta_data.json"
    if full_projection_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        angles_all = metadata.get("source", {}).get("angles_all")
        if angles_all is None:
            raise ValueError(
                "meta_data.json has no source.angles_all for --recon_split all"
            )
        projs_train = np.load(full_projection_path).astype(np.float32)
        angles_train = np.asarray(angles_all, dtype=np.float32)
        if len(projs_train) != len(angles_train):
            raise ValueError(
                "proj_all.npy and source.angles_all have different lengths"
            )
        projs_train *= scene.scene_scale
        print(f"FBP initialization uses all {len(angles_train)} views")
    else:
        print("proj_all.npy not found; falling back to training views")
        projs_train = np.concatenate(
            [t2a(cam.original_image) for cam in train_cameras], axis=0
        )
        angles_train = np.stack(
            [t2a(cam.angle) for cam in train_cameras], axis=0
        )
```

修改结果：

    默认：使用 proj_all.npy + source.angles_all
    备用：没有全角度文件时，回退到训练视角
    可选：--recon_split train 强制使用训练视角

五、r2_gaussian/utils/ct_utils.py
---------------------------------

修改位置：recon_volume() 和 run_ct_recon_algs()

原代码：

```python
if recon_method == "fdk":
    # TIGRE uses FBP for parallel-beam geometry.
    if geo.mode == "parallel":
        vol = algs.fbp(projs[:, ::-1, :], geo, angles)
    else:
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
```

以及：

```python
if method == "fdk":
    ct_pred = algs.fdk(projs[:, ::-1, :], geo, angles)
```

修改后代码：

```python
if recon_method in {"fdk", "fbp"}:
    # TIGRE uses FBP for parallel-beam geometry and FDK for cone-beam.
    if geo.mode == "parallel":
        vol = algs.fbp(projs[:, ::-1, :], geo, angles)
    else:
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
```

以及：

```python
if method in {"fdk", "fbp"}:
    if geo.mode == "parallel":
        ct_pred = algs.fbp(projs[:, ::-1, :], geo, angles)
    else:
        ct_pred = algs.fdk(projs[:, ::-1, :], geo, angles)
```

修改原因：

    平行束不能调用 FDK；平行束必须调用 FBP。
    锥束仍然使用 FDK。
    现在算法名称和 geometry.mode 同时参与判断，避免把平行束结果误保存为 FDK。

六、data_generator/real_dataset/generate_data.py
------------------------------------------------

修改位置：角度生成、MAT读取、scanner_cfg和元数据。

原代码角度生成：

```python
angles = np.concatenate(
    [np.arange(angle_start, angle_last, angle_interval), [angle_last]]
)
angles = angles / 180.0 * np.pi
```

修改后代码：

```python
angles = np.concatenate(
    [np.arange(angle_start, angle_last, angle_interval), [angle_last]]
)
if (
    geometry_type == "ParallelBeam"
    and len(angles) > 1
    and np.isclose(angles[-1] - angles[0], 180.0, atol=1e-4)
):
    # For parallel beam, the 180-degree endpoint duplicates the 0-degree view.
    angles = angles[:-1]
    n_proj = len(angles)
angles = angles / 180.0 * np.pi
```

原代码只保存训练/测试投影：

```python
proj = np.load(osp.join(output_path, projection_train_list[0]["file_path"]))
nDetector = [proj.shape[0], proj.shape[1]]
```

修改后增加全部投影：

```python
proj = np.load(osp.join(output_path, projection_train_list[0]["file_path"]))
all_proj_paths = sorted(glob.glob(osp.join(all_save_path, "*.npy")))
all_projs = np.stack([np.load(path) for path in all_proj_paths], axis=0)
np.save(osp.join(output_path, "proj_all.npy"), all_projs)
nDetector = [proj.shape[0], proj.shape[1]]
```

原代码scanner滤波器：

```python
"filter": None,
```

修改后：

```python
"filter": args.filter,
```

原代码元数据结束于训练和测试列表：

```python
meta_data = {
    "scanner": scanner_cfg,
    "vol": "vol_gt.npy",
    "radius": 1.0,
    "bbox": bbox,
    "proj_train": projection_train_list,
    "proj_test": projection_test_list,
}
```

修改后增加来源信息：

```python
meta_data = {
    "scanner": scanner_cfg,
    "vol": "vol_gt.npy",
    "radius": 1.0,
    "bbox": bbox,
    "proj_train": projection_train_list,
    "proj_test": projection_test_list,
    "source": {
        "algorithm": (
            "TIGRE parallel-beam FBP"
            if scanner_cfg["mode"] == "parallel"
            else "TIGRE FDK"
        ),
        "n_views": len(angles),
        "angles_all": [float(angle) for angle in angles],
        "removed_duplicate_endpoint": geometry_type == "ParallelBeam",
    },
}
```

七、scripts/run_traditional_methods.py
---------------------------------------

修改位置：传统算法输入投影和算法列表。

原代码：

```python
projs_train = np.concatenate(
    [t2a(c.original_image) for c in scene.getTrainCameras()],
    axis=0,
)
train_angles = np.stack([c.angle for c in scene.getTrainCameras()], axis=0)
methods = ["fdk", "sart", "asd_pocs"]
```

修改后：

```python
projs_train = np.concatenate(
    [t2a(c.original_image) for c in scene.getTrainCameras()], axis=0
)
train_angles = np.stack([c.angle for c in scene.getTrainCameras()], axis=0)

full_projection_path = osp.join(dataset.source_path, "proj_all.npy")
metadata_path = osp.join(dataset.source_path, "meta_data.json")
if osp.exists(full_projection_path) and osp.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    angles_all = metadata.get("source", {}).get("angles_all")
    if angles_all is not None:
        projs_train = np.load(full_projection_path).astype(np.float32)
        projs_train *= scene.scene_scale
        train_angles = np.asarray(angles_all, dtype=np.float32)
        print(f"Traditional reconstruction uses all {len(train_angles)} views")

methods = ["fbp", "sart", "asd_pocs"]
```

修改原因：传统重建也应使用全部360个唯一视角；平行束结果目录使用 `fbp`，避免实际调用 FBP 却命名为 `fdk`。

八、新增 scripts/scan_fbp_center.py
-----------------------------------

原代码：不存在旋转中心自动搜索脚本。

修改后核心代码：

```python
base_offset = np.asarray(
    scanner.get("offDetector", [0.0, 0.0]), dtype=np.float32
)
detector_pixel_u = (
    float(scanner["sDetector"][1]) / float(scanner["nDetector"][1])
)
input_tigre = projections[:, ::-1, :]

for u_px in candidates:
    scanner_candidate = copy.deepcopy(scanner)
    offset = base_offset.copy()
    offset[0] = base_offset[0] + u_px * detector_pixel_u
    scanner_candidate["offDetector"] = offset.tolist()
    geo = get_geometry_tigre(scanner_candidate)
    volume = algs.fbp(input_tigre, geo, angles)
    rendered = tigre.Ax(volume, geo, angles)
    residual = rendered - input_tigre
    rmse = float(
        np.sqrt(np.mean(residual ** 2))
        / max(float(np.sqrt(np.mean(input_tigre ** 2))), 1e-8)
    )
```

脚本搜索 `offDetector[0]`，即投影列方向的 u 偏移。默认搜索范围为 -10 到 +10 像素，默认步长为0.5像素；最终以归一化重投影 RMSE 最小的候选值作为旋转中心建议值。

九、最终代码验证
-----------------

使用项目环境执行：

```powershell
& D:\CondaData\envs\r2_gaussian\python.exe -m py_compile `
  prepare_fbp_tiff.py `
  init_from_fbp.py `
  initialize_pcd.py `
  data_generator\real_dataset\generate_data.py `
  r2_gaussian\utils\ct_utils.py `
  scripts\run_traditional_methods.py `
  scripts\scan_fbp_center.py
```

结果：语法检查通过。

实际 FBP 结果：

    输入 TIFF：361张
    唯一角度：360张
    投影尺寸：176 x 177
    FBP体：128 x 128 x 128，全部为有限值
    训练投影：50张
    测试投影：100张
    FBP滤波器：hann
    detector位移：5像素

实际初始化结果：

    初始化点数：50000
    输出形状：(50000, 4)
    四列：x、y、z、density
