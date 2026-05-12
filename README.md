# DynamicPoint

本仓库当前围绕 `data/gs360` 中的 360 视频到 COLMAP/3DGS 数据准备流程展开。下面先说明样例数据目录 `data/gs360/20260210_182109` 中各文件夹的作用，再说明 `scripts/` 中每个脚本的用途与参数。

## 参考文档

- `UT-X5 GS / Reference Alignment Pipeline`:
  - [docs/utx5_gs_ref_alignment_pipeline.md](/mnt/d/develop/master_thesis/DynamicPoint/docs/utx5_gs_ref_alignment_pipeline.md)

## `data/gs360/20260210_182109` 目录说明

这个目录是一次完整运行产生的工作目录，包含原始帧、投影视图、COLMAP 稀疏重建结果，以及为 Gaussian Splatting 准备的兼容布局。

### 顶层文件与目录

| 路径 | 作用 |
| --- | --- |
| `database.db` | COLMAP 特征提取与匹配数据库。`feature_extractor` 和 matcher 的结果都保存在这里。 |
| `frames/` | 从 360 视频抽出的等距柱状投影原始帧。当前样例里有 100 张 `frame_*.jpg`。 |
| `images/` | 从 `frames/` 投影得到的 pinhole 视图，用于后续 SfM。当前按 `(yaw, pitch)` 拆成多个子目录。 |
| `input` | 指向 `images/` 的符号链接，供 `submodules/gaussian-splatting/convert.py` 按 3DGS 期望目录结构读取图像。 |
| `sparse/` | COLMAP `mapper` 输出的稀疏重建模型。 |
| `distorted/` | 3DGS 兼容目录，内部用符号链接复用 `database.db` 和 `sparse/`。 |
| `stereo/` | COLMAP 稠密重建工作区。当前已有配置文件，但尚未生成深度图、法线图和一致性图文件。 |
| `ply/` | 导出的点云文件目录。当前 `points3D.ply` 是由稀疏模型转换得到。 |
| `run-colmap-geometric.sh` | 针对当前工作目录执行 COLMAP dense stereo 的几何一致性版本。 |
| `run-colmap-photometric.sh` | 针对当前工作目录执行 COLMAP dense stereo 的纯光度版本。 |

### `images/` 子目录

| 路径 | 作用 |
| --- | --- |
| `images/yaw_000_pitch_000/` | 朝向 `yaw=0, pitch=0` 的 pinhole 视图，当前有 100 张。 |
| `images/yaw_090_pitch_000/` | 朝向 `yaw=90, pitch=0` 的 pinhole 视图，当前有 100 张。 |
| `images/yaw_180_pitch_000/` | 预留的 `yaw=180` 视图目录。当前为空，因为流水线复制到 run 目录后会删除 `*yaw_180_*` 文件，避免参与 SfM。 |
| `images/yaw_270_pitch_000/` | 朝向 `yaw=270, pitch=0` 的 pinhole 视图，当前有 100 张。 |

### `sparse/` 子目录

| 路径 | 作用 |
| --- | --- |
| `sparse/0/` | COLMAP 主稀疏模型目录，包含 `cameras.bin`、`images.bin`、`points3D.bin` 和 `project.ini`。 |
| `sparse/0/1/` | 额外的重建子模型，通常表示 `mapper` 输出的另一个连通分量。 |
| `sparse/0/2/` | 同上，另一份子模型。 |
| `sparse/0/3/` | 同上，另一份子模型。 |

当前流程里 `scripts/export_sparse_ply.sh` 默认只导出 `sparse/0/`，也就是主模型。

### `distorted/` 与 `stereo/`

| 路径 | 作用 |
| --- | --- |
| `distorted/database.db` | 指向顶层 `database.db` 的符号链接。 |
| `distorted/sparse` | 指向顶层 `sparse/` 的符号链接。 |
| `stereo/patch-match.cfg` | COLMAP PatchMatch Stereo 配置。 |
| `stereo/fusion.cfg` | COLMAP Stereo Fusion 配置。 |
| `stereo/depth_maps/` | 稠密深度图输出目录。当前目录已创建，但没有文件。 |
| `stereo/normal_maps/` | 稠密法线图输出目录。当前目录已创建，但没有文件。 |
| `stereo/consistency_graphs/` | 几何一致性图输出目录。当前目录已创建，但没有文件。 |

## `scripts/` 脚本说明

### `scripts/equirect_to_pinhole.py`

作用：把 360 等距柱状投影图像批量转换成多个 pinhole 视角图像。

调用示例：

```bash
python3 scripts/equirect_to_pinhole.py \
  --input_dir data/gs360/frames \
  --output_dir data/gs360/images \
  --yaw 0,90,270 \
  --pitch 0 \
  --fov 90 \
  --size 800x800 \
  --ext jpg
```

参数：

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--input_dir` | 是 | 无 | 输入目录，放等距柱状投影帧。 |
| `--output_dir` | 是 | 无 | 输出目录，生成 pinhole 图像。 |
| `--yaw` | 否 | `0,90,180,270` | 逗号分隔的水平角列表，单位为度。 |
| `--pitch` | 否 | `0` | 逗号分隔的俯仰角列表，单位为度。 |
| `--fov` | 否 | `90.0` | 水平视场角，单位为度。 |
| `--size` | 否 | `800x800` | 输出分辨率，格式为 `WxH`。 |
| `--ext` | 否 | `jpg` | 输出格式，脚本注释约定为 `jpg` 或 `png`。 |

补充：

- 输出文件名格式为 `frame_000000_yaw_090_pitch_000.jpg`。
- 脚本按输入文件排序后重新编号为 `frame_%06d`，不会保留原始文件名。

### `scripts/split_images_by_view.sh`

作用：把同一目录下的 `frame_*_yaw_*_pitch_*.*` 图片，按 `yaw/pitch` 拆分进子目录。

调用示例：

```bash
bash scripts/split_images_by_view.sh data/gs360/images
```

参数：

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `images_dir` | 否 | `data/gs360/images` | 要整理的图片目录。脚本只接受这一个位置参数。 |

补充：

- 文件名必须匹配 `yaw_XXX_pitch_YYY` 模式，否则会被跳过并输出告警。
- 脚本使用 `mv`，会直接移动文件，而不是复制。

### `scripts/gs360_pipeline.sh`

作用：串起 gs360 数据准备主流程，包括抽帧、equirect 转 pinhole、COLMAP 稀疏重建，以及转换为 3DGS 兼容目录结构。

调用示例：

```bash
bash scripts/gs360_pipeline.sh \
  --video /path/to/video.mp4 \
  --workdir data/gs360 \
  --fps 2 \
  --yaw 0,90,270 \
  --pitch 0 \
  --fov 90 \
  --size 800x800 \
  --matcher sequential \
  --pre
```

参数：

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--video` | 条件必需 | 空 | 输入视频路径。只有传 `--pre` 时必须提供。 |
| `--workdir` | 否 | `data/gs360` | 工作目录，运行结果会放在其中的时间戳子目录下。 |
| `--fps` | 否 | `2` | 预处理阶段抽帧帧率。当前脚本里 `ffmpeg` 命令仍是注释，需要手动启用。 |
| `--yaw` | 否 | 实际代码为 `0,90,270` | pinhole 视角的 yaw 列表。 |
| `--pitch` | 否 | `0` | pinhole 视角的 pitch 列表。 |
| `--fov` | 否 | `90` | pinhole 相机视场角。 |
| `--size` | 否 | `800x800` | pinhole 输出分辨率。 |
| `--matcher` | 否 | `sequential` | COLMAP 匹配器，支持 `sequential` 或 `exhaustive`。 |
| `--pre` | 否 | 关闭 | 是否执行预处理阶段，即抽帧和 equirect 转 pinhole。 |
| `-h`, `--help` | 否 | 无 | 输出帮助。 |

流程细节：

1. 生成 `TIMESTAMP` 目录，例如 `data/gs360/20260210_182109`。
2. 如果传了 `--pre`，先从视频抽帧，再调用 `equirect_to_pinhole.py` 生成多视角图像。
3. 把基础 `frames/` 和 `images/` 复制到当前 run 目录。
4. 删除 `*yaw_180_*` 图片，避免 180° 反向视图参与 SfM。
5. 运行 `colmap feature_extractor`、matcher、`mapper`。
6. 创建 `input`、`distorted/database.db`、`distorted/sparse` 这些 3DGS 兼容链接。
7. 调用 `submodules/gaussian-splatting/convert.py -s <RUN_DIR> --skip_matching`。

注意：

- 帮助文本里写的 `--yaw` 默认值是 `0,90,180,270`，但脚本实际变量默认值是 `0,90,270`。这里以脚本实现为准。
- 脚本会对当前 run 目录中的 `frames/`、`images/` 执行 `rm -rf` 后重建；它不会删基础目录 `data/gs360/frames` 和 `data/gs360/images`。
- 依赖外部可执行程序：`python3`、`colmap`，以及启用抽帧时需要 `ffmpeg`。

### `scripts/export_sparse_ply.sh`

作用：把 COLMAP 稀疏模型导出为 `PLY` 点云文件。

调用示例：

```bash
bash scripts/export_sparse_ply.sh data/gs360 20260210_182109
```

参数：

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `<workdir>` | 是 | 无 | 工作目录，例如 `data/gs360`。 |
| `<timestamp>` | 是 | 无 | 某次运行的时间戳目录名，例如 `20260210_182109`。 |

输入输出：

- 输入模型目录：`<workdir>/<timestamp>/sparse/0`
- 输出文件路径：`<workdir>/<timestamp>/ply/points3D.ply`

依赖：

- 需要系统里可调用 `colmap model_converter`。

## 样例目录与脚本的对应关系

- `frames/` 通常来自 `gs360_pipeline.sh --pre` 中的视频抽帧步骤。
- `images/` 来自 `equirect_to_pinhole.py`，也可能再经 `split_images_by_view.sh` 整理。
- `database.db` 与 `sparse/` 来自 `gs360_pipeline.sh` 中的 COLMAP SfM 步骤。
- `distorted/` 与 `input` 来自 `gs360_pipeline.sh` 的 3DGS 布局准备步骤。
- `ply/points3D.ply` 来自 `export_sparse_ply.sh`。
- `stereo/` 与 `run-colmap-*.sh` 配合使用，对当前工作目录继续做 COLMAP dense reconstruction。

## 大型 PLY 三分类管线

仓库新增了一条面向大体量点云的离线规则分类流程，目标是把单个 `PLY` 点云分成：

- `0 = Ground`
- `1 = Shrubs`
- `2 = Trees`

实现入口：

```bash
python3 pipeline.py \
  --in /path/to/input.ply \
  --out /path/to/labeled.ply \
  --config config.yaml
```

实现位置：

- `pipeline.py`：总入口
- `config.yaml`：默认参数
- `src/pointcloud_classify/io_ply.py`：读取/回写 PLY，保留原字段，仅新增 `label`
- `src/pointcloud_classify/downsample.py`：体素下采样
- `src/pointcloud_classify/dem.py`：DEM 地面高度场
- `src/pointcloud_classify/height_norm.py`：离地高度
- `src/pointcloud_classify/features_pca.py`：下采样点上的 kNN-PCA verticality
- `src/pointcloud_classify/rules.py`：Ground/Shrubs/Trees 规则分类
- `src/pointcloud_classify/postprocess.py`：邻域多数投票平滑
- `src/pointcloud_classify/propagate.py`：体素哈希回传原始点标签
- `src/pointcloud_classify/inspect_ply.py`：打印 PLY 元信息

默认参数面向自然场景点云，遵循“下采样判别 + 原始点回传”的策略，避免在全量点上做逐点 PCA。

### Logan Park 测试

测试输入：

- `loganPark-test/loganPark-test/supersplat scenes/0-interested-large02.ply`

运行命令：

```bash
python3 pipeline.py \
  --in "loganPark-test/loganPark-test/supersplat scenes/0-interested-large02.ply" \
  --out "outputs/0-interested-large02/labeled.ply" \
  --config config.yaml \
  --log "logs/run_loganPark_large02.json"
```

本次测试结果：

- 输入点数：`1,828,491`
- 下采样后点数：`31,188`
- 输出文件：`outputs/0-interested-large02/labeled.ply`
- 日志文件：`logs/run_loganPark_large02.json`
- 总耗时：约 `26s`

输出验证：

- 保留了原始 vertex 字段
- 新增 `label:uint8`
- 输出标签只包含 `{0,1,2}`
