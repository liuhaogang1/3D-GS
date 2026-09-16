# R2-Gaussian FBP 项目配置清单

## 1. 项目范围

当前项目流程为：

1. 旋转中心估计或扫描
2. TIFF 投影预处理与 TIGRE 平行束 FBP 重建
3. 从 FBP 体数据生成 Gaussian 初始化点云
4. R2-Gaussian 训练
5. R2-Gaussian 测试与评估

项目不包含原项目中的 `data`、`output`、数据生成器、传统重建脚本、实验批处理脚本、可视化脚本和 Git 文件。数据和训练输出建议放在项目目录之外。

## 2. 运行环境

建议使用 Python 3.9。训练和测试需要 NVIDIA GPU、CUDA、PyTorch 以及本项目中的两个 CUDA 扩展。

基础版本参考：

- Python 3.9
- PyTorch 1.12.1 + CUDA 11.6，或已经验证过的 PyTorch/CUDA 组合
- torchvision 0.13.1（与 PyTorch 1.12.1 配套）
- NumPy 1.24.x
- SciPy 1.10.x
- TIGRE 2.3
- Open3D 0.18.0
- SimpleITK 2.4.0
- scikit-image 0.21.0
- tifffile
- matplotlib
- tqdm
- PyYAML
- tensorboard / tensorboardX
- plyfile

旋转中心的 TomoPy 方法额外需要：

- TomoPy
- numexpr

## 3. 安装顺序

在项目根目录 `D:\TEM\3D-GS\r2_gaussian_fbp` 执行：

```powershell
conda create -n r2_gaussian_fbp python=3.9 -y
conda activate r2_gaussian_fbp

# 按目标 CUDA 版本安装 PyTorch 和 torchvision
# 下面是原项目 CUDA 11.6 的参考版本
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.6 -c pytorch

pip install numpy scipy tifffile matplotlib tqdm pyyaml plyfile
pip install tensorboard tensorboardX SimpleITK open3d==0.18.0 scikit-image
pip install TIGRE==2.3
pip install tomopy numexpr

pip install -e r2_gaussian/submodules/simple-knn
pip install -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
```

如果 `TIGRE==2.3` 无法直接安装，应使用 TIGRE v2.3 源码目录执行 `pip install <TIGRE目录>/Python --no-build-isolation`。

Windows 编译 CUDA 扩展前可设置：

```powershell
$env:DISTUTILS_USE_SDK = "1"
```

## 4. 迁移后的代码

### 流程入口

- `scripts/scan_fbp_center.py`：使用 0/180 度投影端点扫描 detector-u 旋转中心
- `scripts/scan_fbp_center_tomopy.py`：使用 TomoPy 估计旋转中心
- `prepare_fbp_tiff.py`：TIFF 预处理、中心偏移、FBP、数据集生成
- `init_from_fbp.py`：从 `vol_fbp.npy` 采样初始化点云
- `train.py`：Gaussian 训练
- `test.py`：训练结果测试

### 运行依赖

- `scripts/fbp_preprocess.py`：中心估计和 FBP 共用的角度、TIFF、无效值处理
- `r2_gaussian/arguments/`
- `r2_gaussian/dataset/`
- `r2_gaussian/gaussian/`
- `r2_gaussian/utils/`
- `r2_gaussian/submodules/simple-knn/`
- `r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/`

其中 `r2_gaussian/utils/ct_utils.py` 没有迁移，因为它只服务旧的传统重建和旧版 `initialize_pcd.py`。

## 5. 数据目录要求

### TIFF 原始投影目录

目录中应包含投影 TIFF 文件。若使用配置文本，配置中至少应能提供：

- `AngleFirst`
- `AngleInterval`
- `NumberImages`

也可以通过 `--angles_file` 提供每个 TIFF 对应的角度。`flat*.tif` 和 `flat*.tiff` 会被排除。

### FBP 输出目录

`prepare_fbp_tiff.py` 会生成：

- `meta_data.json`
- `proj_all.npy`
- `proj_endpoint_pair.npy`（检测到 180 度重复端点时）
- `proj_train/`
- `proj_test/`
- `vol_fbp_raw.npy`
- `vol_fbp_adjusted.npy`
- `vol_fbp.npy`
- FBP 预览图和 NIfTI 文件

`meta_data.json` 是训练和测试识别数据集的关键文件；`vol_fbp.npy` 是 FBP 初始化的输入。

## 6. 推荐运行流程

### 6.1 估计旋转中心

TomoPy 方法：

```powershell
python scripts/scan_fbp_center_tomopy.py `
  --input_dir D:\path\to\tiff_projections `
  --config D:\path\to\acquisition.txt `
  --output D:\path\to\center.json
```

端点配对扫描方法：

```powershell
python scripts/scan_fbp_center.py `
  --data D:\path\to\fbp_dataset `
  --output D:\path\to\center_scan.json
```

第二种方法要求 `--data` 中已有 `meta_data.json` 和 `proj_endpoint_pair.npy`，因此通常在一次 FBP 数据准备之后用于复核中心。

### 6.2 FBP 重建并生成训练数据

```powershell
python prepare_fbp_tiff.py `
  --input_dir D:\path\to\tiff_projections `
  --output_dir D:\path\to\fbp_dataset `
  --config D:\path\to\acquisition.txt `
  --center_json D:\path\to\center.json `
  --input_type transmission `
  --n_train 50 `
  --n_test 100 `
  --nVoxel 128 128 128 `
  --sVoxel 2 2 2
```

如果输入已经是 line integral，使用 `--input_type line_integral`。如果输入是原始强度或透射率，使用 `transmission` 或 `intensity`，并检查 `--i0` 或 `--i0_percentile`。

### 6.3 生成 FBP 初始化点云

```powershell
python init_from_fbp.py `
  --volume D:\path\to\fbp_dataset\vol_fbp.npy `
  --n_points 50000 `
  --density_thresh 0.05 `
  --density_rescale 0.15
```

脚本会在数据目录中生成 `init_<数据目录名>.npy`，这是 `initialize_gaussian()` 的默认查找命名。

### 6.4 训练

```powershell
python train.py `
  -s D:\path\to\fbp_dataset `
  -m D:\path\to\training_output
```

常用参数：`--iterations`、`--test_iterations`、`--save_iterations`、`--checkpoint_iterations`、`--start_checkpoint`、`--config`。

### 6.5 测试

```powershell
python test.py `
  -m D:\path\to\training_output `
  -s D:\path\to\fbp_dataset
```

可用 `--iteration` 指定模型迭代次数，也可使用 `--skip_render_train`、`--skip_render_test`、`--skip_recon` 跳过对应评估。

## 7. 关键一致性检查

- `nVoxel`、`sVoxel`、`nDetector`、`sDetector` 和投影实际尺寸必须一致。
- `offDetector[0]` 使用 detector-u 方向的物理单位，中心 JSON 可直接传给 `--center_json`。
- FBP 输出体数据会归一化到 `[0, 1]`，初始化阈值应根据实际体数据重新检查。
- `n_train + n_test` 必须小于等于去除重复 180 度端点后的投影数量。
- 训练和测试默认使用 CUDA；当前核心代码不提供 CPU 训练路径。
- 每次 FBP 独立运行应使用新的空 `--output_dir`。
