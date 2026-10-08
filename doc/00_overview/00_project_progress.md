# SwellEx 项目进展

更新日期：2026-10-08

项目已统一为 Python 数据处理和定位网络，依赖由 uv 管理。代码在本地修改，
数据生成、数值检查和模型运行在 SSH 主机 haiqin 进行。

## 当前结构

```text
scripts/data_processing/          RBD/ELM/SCM 生成、配置、数值验证
scripts/localization_networks/    公共加载/训练/预测、三套网络、SCM-GRNN
pyproject.toml / uv.lock           依赖与冻结版本
Origindata/                       原始输入（不入 Git）
outputs/                          数据集与结果（不入 Git）
```

原 MATLAB 实现保留在 `data_processing/matlab_reference/`，供公式与迁移核对。
实际执行使用 NumPy/SciPy/h5py/PyTorch，不依赖 MATLAB 运行时。数值 MAT
输入由 Python 读取；CTD 直接读取原始 `i9605.prn` 深度/声速列。

## 已构建能力

- RBD：保留变声速传播延迟积分、平面波回退、Bartlett 波束与多径旁瓣剔除。
  HDF5 保存实际 Green 特征；旧分段入口误存 FFT 的数据需要重新生成。
- ELM：严格上三角阵元对的频域 least-squares ratio；SCM：含对角线的
  归一化空间协方差。三套 loader 与 model registry 保持独立。
- 每套网络保留 complex CNN、real CNN、ResNet18、ResNet50；输出为归一化
  距离 `[batch,1]`，支持 normalized/km 损失。GRNN 独立 build/predict。
- 共享采样、标签插值、periodic/Range_nearby split 与 full/mel/deep/shallow/
  adapt 选频；adapt 只在有效训练窗口拟合。
- 工作流固定源码快照与 uv 锁环境，记录输入哈希、运行配置、设备、检查结果。
  远端任务支持脱离 SSH、失败报告和结果回收；数据缓存需内容校验后复用。

## 验证状态

本地路径、Python 语法和文档链接检查通过。haiqin 已完成冻结 uv 环境的
全链路 smoke 验证，实际使用 RTX A5000 / CUDA：

- `20261008T102048Z-f0357789`：21 个阶段全部 PASS，重新生成三类数据集。
- `20261008T102335Z-31cc07b9`：后台运行与结果回收通过；20 个阶段 PASS，
  数据生成阶段因复用上一次产物标记 SKIP，缓存内容哈希及数据契约重新验证通过。
- 27 项 Python 快速检查通过，其中包含 29 个工作流单元测试、12 个模型的
  GPU forward/backward，以及加载器损坏数据拒绝检查；8 项数据处理数值检查通过。
- 三类数据共生成 6 个 HDF5 文件，各有 48 个训练窗口和 12 个测试窗口、
  22 个频率。12 项数据契约检查通过，包含跨方法标签、时间、划分与频率对齐。
- 三套 complex CNN 的 2-epoch 训练和预测、SCM-GRNN build/predict、
  断点恢复均通过。报告、CSV 和图表已回收至本地
  `outputs/remote_validation/<run_id>/`。

验证在独立源码快照中完成；远端已有源码仓库保持不变。

默认 smoke 使用 S5 0–59 s 起始窗、1 s 非重叠快拍、4:1 periodic split，
每套 complex CNN 训练 2 epochs，GRNN 做 3-fold sigma 选择，并验证断点恢复。
这是功能检查规模；正式精度对比需扩大数据范围、重复随机种子并定义独立
验收指标。Ns=4 等不同窗几何的研究还需单独设计跨方法对齐。

使用入口与配置说明见
[远端工作流](../20_python_workflows/30_remote_validation.md)。数据字段见
[HDF5 契约](../10_datasets/00_hdf5_datasets_and_splits.md)。
