# m-vins-slim Optimization Log

> 记录每次精简/优化操作的具体实现、影响范围、验证结果与可回滚点。

---

## 精简清单（Checklist）

> 基于 2026-03-28 源码分析，列出所有可精简项。完成一项打一个勾。

- [x] **ROS1 条件分支**：移除所有 `#ifdef USE_ROS2 ... #else(ROS1) ... #endif`、CMake catkin 逻辑、package.xml ROS1 依赖
  - 提交：`7baa547`
  - 验证：改动前后二进制产物 MD5 完全一致（编译器输出字节级相同）
- [x] **USE_CNN_FEATURE 分支**：`super_point_infer.cc/.h`、`#ifdef USE_CNN_FEATURE` 散布代码（~30处）、RKNN SDK（`third_lib/rknn/`）、SuperPoint 模型（`assets/superpoint_*.rknn` ~9.5MB）、`assets/inverted_multi_index_quantizer_superpoint.dat` (~1MB)
- [ ] **Visual / ScanVisual 链路**（量最大）：`src/vins_core/src/feature_tracker/` 整目录（8文件）、`camera_reprojection_cost.cc`、`visual_loop_interface.cc`、`image_interface.cc`、`vins_handler` 中 Visual/ScanVisual 分支、非当前设备的标定/mask 文件（`calib_d435/euroc/uhumans2.yaml`、`mask_d435/euroc/uhumans2.png`）、`src/thirdparty/xfeatures2d/`
- [ ] **IMU 融合**（`use_imu: false`）：`imu_propagation_cost.cc`、`imu_propagator.cc`、`imu_interface.cc`、`vins_handler` 中 IMU 相关逻辑
- [ ] **OctoMap**（`do_octo_mapping: false`）：`src/octomap_core/` 整个模块（3文件）、`vins_handler` 中 OctoMapper 逻辑、`tools/octomap2ply/`
- [x] **调试/训练功能**（`voc_train_mode/feature_tracking_test_mode/save_colmap_model: false`）：`vins_thread.cc` 中 voc_training 逻辑、feature_tracking_test 逻辑、colmap 保存逻辑
- [ ] **Rosbag 离线回放**：`rosbag_parsing_handler.cc/.h`、`ros_life_circle_handler.cc` 中 RosbagParser 启动逻辑、`scripts/slam_offline.sh`
- [ ] **cfg_bk/ 备份目录**：与 `cfg/` 完全重复
- [x] **aslam_cv_detector + test 文件**：`src/thirdparty/aslam_cv2/aslam_cv_detector/`（KAZE/LSD，代码中未调用）、各模块 `test/` 目录（12个测试文件，不参与构建）
- [ ] **FREAK 视觉资源**：`assets/inverted_multi_index_quantizer_freak.dat`（视觉关闭时不需要）

---

## 已完成的优化记录

### 2026-03-28 / Remove ROS1 conditional code and configuration

**提交**：`7baa547` `refactor(m-vins-slim): remove all ROS1 conditional code, keep ROS2 only`

#### 背景
- 项目已全面迁移至 ROS2 Humble，ROS1 (catkin/roscpp) 代码路径不再使用。
- 源码中大量 `#ifdef USE_ROS2 ... #else ... #endif` 双版本兼容代码，增加维护负担。

#### 实施内容

**1. 源码 (#ifdef/#else/#endif)**
- 移除所有 `#ifdef USE_ROS2` / `#ifndef USE_ROS2` 中的 ROS1 `#else` 分支
- 保留 ROS2 代码，去掉条件编译守卫
- 涉及 **17 个源文件**，共移除 **64 个 ROS1 条件块**
- 受影响模块：
  - `src/ros_interface/` (m_vins_node.cc, interface/*.cc, ros_handler/*.cc/h)
  - `src/ros_interface/interface/include/interface/interface.h`

**2. CMakeLists.txt**
- 顶层 `CMakeLists.txt`：
  - 移除 `USE_ROS2_API` 变量定义和 `add_definitions(-DUSE_ROS2)`
  - 移除 `catkin` find_package / catkin_package / ROS1 install 块
  - ROS2 find_package 和 install 不再需要 if 包裹
- 子模块 CMakeLists（ros_interface, vins_core, vins_common, vins_handler, octomap_core, loop_closure, third_lib, thirdparty/libnabo, thirdparty/xfeatures2d）：
  - 移除 `if(USE_ROS2_API)...endif()` 包裹，install 语句直接执行
- 附带修复：所有子模块 `cmake_minimum_required(VERSION 3.0)` 升级为 `VERSION 3.5`

**3. package.xml**
- 移除注释掉的 ROS1 依赖块（catkin, roscpp, image_transport, tf 等）

#### 未改动部分
- `src/thirdparty/aslam_cv2/` 内部的 catkin 引用：属于第三方库自身构建文件，不参与当前编译，保持原样
- `#ifdef PUB_DEBUG_INFO`、`#ifdef USE_CNN_FEATURE`、`#ifdef __ARM_NEON__` 等其他条件编译：未变动

#### 验证结果
- 编译通过：`colcon build --packages-select m_vins` → `Finished [17min 13s]`
- **二进制对比**：改动前（ec206f0）与改动后（7baa547）编译产物 MD5 完全一致，证明功能等价
- 仅有 Eigen NEON memcpy 和 unused-parameter 警告（非阻塞，与本次改动无关）

#### 回滚方式
```bash
git revert 7baa547
```

### 2026-03-28 / Remove USE_CNN_FEATURE conditional code, assets, and RKNN SDK

**提交**：`<pending>` `refactor(m-vins-slim): remove CNN feature code, RKNN SDK, and SuperPoint assets`

#### 背景
- `USE_CNN_FEATURE` 编译开关始终为 FALSE，CNN/SuperPoint 特征提取从未启用。
- 相关代码、模型文件和 RKNN SDK 占用约 10.5MB 空间，增加仓库体积和维护负担。

#### 实施内容

**1. 源码条件分支**
- 移除所有 `#ifdef USE_CNN_FEATURE ... #endif` 条件块（保留 `#else`/`#ifndef` 中的非 CNN 代码）
- 涉及 **12 个源文件**，共移除 **29 个 CNN 条件块**
- 受影响模块：vins_common、vins_handler、vins_core/feature_tracker、loop_closure

**2. 删除的源码文件**
- `src/vins_core/src/feature_tracker/super_point_infer.cc`
- `src/vins_core/include/feature_tracker/super_point_infer.h`

**3. 删除的资源文件**（共 ~10.5MB）
- `assets/superpoint_240_320_fp16_int8.rknn` (~3.2MB)
- `assets/superpoint_240_320_fp16_int8_pruned.rknn` (~3.2MB)
- `assets/superpoint_v1_fp16.rknn` (~3.0MB)
- `assets/inverted_multi_index_quantizer_superpoint.dat` (~1.0MB)

**4. 删除的第三方库**
- `third_lib/rknn/` 整个目录（RKNN SDK：头文件 + aarch64/x86_64 .so）

**5. CMakeLists.txt**
- 顶层：移除 `set(USE_CNN_FEATURE FALSE)` 和 `if(USE_CNN_FEATURE)` 块
- `src/vins_core/CMakeLists.txt`：移除 CNN 条件编译文件列表
- `third_lib/CMakeLists.txt`：简化为仅 yaml-cpp，移除 RKNN 相关变量和条件

#### 验证结果
- 编译通过：`colcon build --packages-select m_vins` → `Finished [21min 15s]`
- **二进制对比**：
  - 4 个库未变（octomap_core、aslam_cv、nabo、xfeatures2d）
  - 6 个库有变化，但**文件大小完全一致** + **导出符号表完全一致**
  - 差异原因：删除 `#ifdef` 守卫行导致行号偏移 → debug info 变化 + LTO 非确定性地址布局
  - 结论：**机器码等价，功能完全一致**

#### 遇到的问题与解决
1. **Python 读取非 UTF-8 文件报错**：thirdparty 中有二进制格式 `.h` 文件，需要 `errors='ignore'`
2. **Strip 后哈希仍不同**：ARM 编译开了 `-flto`（链接时优化），LTO 会引入非确定性，但导出符号验证通过

#### 回滚方式
```bash
git revert <commit-hash>
```

### 2026-03-28 / Remove cfg_bk/ backup directory

**提交**：见本次 git commit

#### 实施内容
- 删除 `cfg_bk/` 目录（13 个文件，与 `cfg/` 近乎完全重复）
- 唯一差异：`cfg_bk/calib_avaia_cleaner.yaml` 的外参矩阵为旧值（1.0），`cfg/` 中已更新为 -1.0
- `cfg/` 为当前使用版本，`cfg_bk/` 为历史备份，可通过 git 历史恢复

#### 验证
- 不涉及编译（纯配置文件备份目录），无需二进制对比

### 2026-03-28 / Remove aslam_cv_detector, test files, and debug/training features

**提交**：见本次 git commit

#### Part 1: aslam_cv_detector + test 文件
- 删除 `src/thirdparty/aslam_cv2/aslam_cv_detector/` 整个目录（KAZE/LSD，未被任何代码调用）
- 删除 `aslam_cv_cameras/test/` 和 `aslam_cv_common/test/`（12个测试文件，不参与构建）
- 移除 CMakeLists.txt 中 aslam_cv_detector 的 include 路径
- 修复 aslam_cv2/CMakeLists.txt 中残留的 `if(USE_ROS2_API)` 包裹

#### Part 2: 调试/训练功能
删除以下运行时调试功能代码（配置中均为 false/关闭状态）：

**voc_train_mode（词袋训练）：**
- 删除 `VocTrainingThread()` 函数定义（vins_thread.cc）
- 删除 `Start()` 中的条件线程启动
- 删除 `voc_training_thread_` 成员变量和 join 逻辑

**feature_tracking_test_mode（特征追踪测试）：**
- 删除 `FeatureTrackingTestThread()` 函数定义
- 删除 `feature_tracking_testing_thread_` 成员变量和 join 逻辑
- 删除 `SyncSensorThread()` 中 debug 模式的 `CollectImageDataOnly` 分支

**save_colmap_model（Colmap 模型保存）：**
- 删除 `SaveColmapModel()` 函数定义（vins_handler.cc）
- 删除 `FrontendThread()` 中的 colmap 保存调用

**CollectImageDataOnly（仅图像数据收集）：**
- 删除函数定义和声明（仅被上述 debug 分支调用）

**slam_config：**
- 移除 `voc_train_mode`、`feature_tracking_test_mode`、`save_colmap_model`、`feature_tracking_test_path` 配置字段定义、YAML 解析和日志打印

#### 遇到的问题
- `SetValueBasedOnYamlKey` 调用跨 3 行，按关键词逐行删除会留下不完整的函数调用 → 需要检测并删除残留的不完整调用

#### 验证结果
- 编译通过：`Finished [16min 55s]`
- 二进制对比（vs CNN 清理后的基准）：
  - 4 个库完全一致（octomap_core、aslam_cv、nabo、xfeatures2d）
  - 6 个变化的库：**文件大小全部一致** + **导出符号表全部一致**
  - `VocTrainingThread`、`FeatureTrackingTestThread`、`SaveColmapModel`、`CollectImageDataOnly` 符号已从 vins_handler .so 中消失（符合预期）
