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
    生成 R2-Gaussian 所需的 meta_data.json。

init_from_sart.py作用：
    读取 128³ SART 重建体；
    找出密度大于 0.05 的体素；
    随机选择 50000 个体素；
    将体素坐标映射到 [-1, 1]^3；
    将体素密度乘以 0.15；
    生成init_refcorr_sart128.npy作为初始点云
-----------------------------------------------------------------------------------------