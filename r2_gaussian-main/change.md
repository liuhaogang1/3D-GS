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
