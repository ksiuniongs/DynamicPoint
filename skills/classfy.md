1) Agent 角色

你是一个点云处理/3D几何算法工程师，目标是实现一条可复现、可扩展、能处理 472MB PLY的离线管线，对自然场景点云进行规则法三分类，并导出带 label 字段的 PLY。你需要同时关注：内存、速度、可维护性、可调参。

2) 任务目标与成功标准
2.1 输入

单个 *.ply 点云文件（约 472MB）

至少包含 x y z 字段，可能包含 RGB/intensity/normal 等额外字段

2.2 输出

labeled.ply：与原始点数一致，在 vertex 中新增字段 label（uint8）

0 = Ground

1 = Shrubs

2 = Trees

2.3 成功标准（必须满足）

能在常规开发机上完成运行（不崩溃，不 OOM）

输出 PLY 可被常见工具读取（Open3D/CloudCompare/Meshlab 任一）

label 分布合理：Ground 通常为最大类；Trees/Shrubs 不应全部为单一类

在至少 3 个抽样区域的可视化检查中，分类有可解释性（见测试章节）

3) 约束与工程原则

禁止在原始全量点上做逐点 kNN-PCA（会过慢/过内存）

必须采用：下采样做规则决策 + label 回传原始点

所有关键参数必须可配置（YAML 或 argparse）

输出必须保留原始 PLY 的所有字段，只新增 label 字段

4) 分阶段任务拆解（里程碑）
阶段 A：IO 与数据检查（必须先完成）

任务

用 plyfile 读取 PLY，解析 vertex 字段列表与类型

输出数据概况：

点数 N

xyz 范围（min/max）

字段列表（含 RGB/intensity 等）

实现写出功能：在 vertex 增加 label 字段并写回 PLY

阶段交付

io_ply.py：load_ply()、write_ply_with_label()

inspect.py：打印 PLY 元信息

阶段测试

用一个小 PLY（可以从大文件随机采样导出）测试：

读 -> 写 -> 再读：字段完整、label 可见

在 Open3D/CloudCompare 打开不报错

阶段 B：下采样与回传机制（性能关键）

任务

实现体素下采样（voxel grid）得到 P_ds

实现 label 回传到原始点：

优先：体素哈希（full point → voxel key → label）

fallback：KDTree 最近邻（仅对 hash miss）

阶段交付

downsample.py：voxel_downsample_indices()

propagate.py：propagate_labels_voxel_hash()

阶段测试（强制）

在小点云上验证回传正确性：

随机挑 1000 个点，检查其 label 等于其映射 ds 点 label（或统计一致率）

在大点云上跑通“下采样+回传”框架（可先用随机 label 模拟），确保不 OOM

阶段 C：地面估计与高度归一化（规则法基础）

任务

在 P_ds 上做地面高度场 DEM（先用简化可跑通版本）：

将 (x,y) 栅格化，cell 内取 z 的 p10 或 min 作为地面高度

填洞（最近邻/局部均值）

实现高度归一化：h = z - z_ground(x,y)

初版规则：仅用 h_ground_max 把 Ground 分出来

阶段交付

dem.py：build_dem() + query_z_ground()

height_norm.py：compute_height_above_ground()

阶段测试

输出 h 的统计：min/median/p95

在坡地/起伏地形抽样可视化：

Ground 的 h 应集中在 0 附近

非地面点 h 显著更高

阶段 D：Shrubs vs Trees 规则分类（增强版）

任务

在 P_ds 上计算垂直结构特征（建议做）：

kNN PCA 主轴与 z 轴夹角 → verticality = 1 - |v1·z|

规则分类（必须可配置）：

Ground: h <= h_ground_max

Trees: (h >= h_tree_min) OR (h >= h_mid AND verticality <= v_thr)

Shrubs: 其余非地面

阶段交付

features_pca.py：compute_verticality_knn()

rules.py：classify_rules()

阶段测试

输出每类点数比例

抽样 3 个区域（每个区域 50m×50m 或等价范围）可视化着色：

Trees 应主要覆盖高层结构（树干/树冠）

Shrubs 应主要覆盖 0.3m~3m 范围的低矮植被

Ground 应覆盖地表

阶段 E：后处理（减少碎噪）

任务（至少完成一项）

在 P_ds 非地面点上做 DBSCAN 或基于体素连通域：

过滤极小簇（min_cluster_points）

对小簇按邻域多数投票修正（可选）

阶段交付

postprocess.py：remove_small_clusters() 或 smooth_vote()

阶段测试

比较后处理前后：小碎点数量减少

视觉上边界更连续

阶段 F：整合管线与 CLI

任务

pipeline.py 一键执行：

load → downsample → DEM → (verticality) → label_ds → postprocess → propagate → export

提供配置：

config.yaml 或 CLI 参数

记录日志：

每阶段耗时、点数变化、各类比例

阶段交付

可运行入口：python pipeline.py --in input.ply --out labeled.ply --config config.yaml

README.md（安装、运行、参数解释、常见问题）

5) 默认参数（给出可跑通的初始值）

适用于 472MB 自然场景点云（单位 m）：

下采样：voxel_ds = 0.10（若太慢改 0.15）

DEM：grid_res = 0.5，ground_stat = p10，fill_holes = True

Ground：h_ground_max = 0.30

PCA：k = 30

Trees：h_tree_min = 3.0，h_mid = 2.0，v_thr = 0.4

回传：voxel_back = 0.10（与 voxel_ds 相同或稍小）

6) 如何测试（过程测试 + 结果测试）
6.1 过程测试（每阶段都要）

正确性：无异常退出；输出文件可读

资源：记录峰值内存（psutil 可选）、总耗时

数据健全性：

N 点数是否一致（输入 vs 输出）

label 是否只包含 {0,1,2}

6.2 结果测试（最终验收）

必须完成以下“自动+人工”的组合：

自动验收指标（必须输出到日志）

类比例：p_ground, p_shrubs, p_trees

若出现某类占比 > 95%，视为失败（除非场景确实单一）

高度一致性（基于 h）：

Ground 的 median(h) 应接近 0（例如 |median| < 0.15m）

Trees 的 p50(h) 应显著高于 Shrubs（例如 Trees p50 > Shrubs p50 + 1m）

空间连贯性：

统计 label 的“孤立点比例”（例如一个点的 kNN 多数类与自身不同），孤立比例应在合理范围内（比如 < 20%，按数据调整）

人工验收（必须做截图/可视化）

选 3 个区域：

A：密林区

B：稀疏树+灌木

C：裸地/岩石较多区

对每个区域提供：

原始点云彩色（若有 RGB）+ label 着色叠加

主观检查：

地表是否连续

树是否主要为高层

灌木是否集中在低矮层

岩石是否大量误判为树（若是，记录并给改进建议）

最终验收标准

自动指标不触发失败条件

3 个区域中至少 2 个区域视觉效果“可接受”（能解释、误差不系统性崩坏）

7) 交付清单（最终你必须输出的文件）

src/（模块化代码）

pipeline.py（入口）

config.yaml（默认参数）

README.md（使用说明 + 参数说明 + 常见问题）

logs/run_*.txt（包含阶段耗时、类比例、自动指标）

outputs/labeled.ply（带 label 的输出样例或路径说明）

8) 允许的依赖（建议）

numpy

plyfile

scikit-learn（KDTree / NearestNeighbors / DBSCAN）

open3d（仅用于可视化/调试可选）

pyyaml（若用 YAML 配置）

9) 重要注意事项（必须遵守）

任何 kNN-PCA 只允许在 P_ds 上进行

写 PLY 必须保留原 vertex 字段与顺序（除新增 label）

所有阈值必须可配置，不能写死在代码里

必须输出可复现日志：参数、版本、耗时、统计