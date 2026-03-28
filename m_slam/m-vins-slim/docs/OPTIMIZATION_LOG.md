# m-vins-slim Optimization Log

> 记录每次精简/优化操作的具体实现、影响范围、验证结果与可回滚点。

---

## 精简清单（Checklist）

> 基于 2026-03-28 源码分析，列出所有可精简项。完成一项打一个勾。

- [x] **ROS1 条件分支**：移除所有 `#ifdef USE_ROS2 ... #else(ROS1) ... #endif`、CMake catkin 逻辑、package.xml ROS1 依赖
  - 提交：`7baa547`
  - 验证：改动前后二进制产物 MD5 完全一致（编译器输出字节级相同）
- [ ] **USE_CNN_FEATURE 分支**：`super_point_infer.cc/.h`、`#ifdef USE_CNN_FEATURE` 散布代码（~30处）、RKNN SDK（`third_lib/rknn/`）、SuperPoint 模型（`assets/superpoint_*.rknn` ~9.5MB）、`assets/inverted_multi_index_quantizer_superpoint.dat` (~1MB)
- [ ] **Visual / ScanVisual 链路**（量最大）：`src/vins_core/src/feature_tracker/` 整目录（8文件）、`camera_reprojection_cost.cc`、`visual_loop_interface.cc`、`image_interface.cc`、`vins_handler` 中 Visual/ScanVisual 分支、非当前设备的标定/mask 文件（`calib_d435/euroc/uhumans2.yaml`、`mask_d435/euroc/uhumans2.png`）、`src/thirdparty/xfeatures2d/`
- [ ] **IMU 融合**（`use_imu: false`）：`imu_propagation_cost.cc`、`imu_propagator.cc`、`imu_interface.cc`、`vins_handler` 中 IMU 相关逻辑
- [ ] **OctoMap**（`do_octo_mapping: false`）：`src/octomap_core/` 整个模块（3文件）、`vins_handler` 中 OctoMapper 逻辑、`tools/octomap2ply/`
- [ ] **调试/训练功能**（`voc_train_mode/feature_tracking_test_mode/save_colmap_model: false`）：`vins_thread.cc` 中 voc_training 逻辑、feature_tracking_test 逻辑、colmap 保存逻辑
- [ ] **Rosbag 离线回放**：`rosbag_parsing_handler.cc/.h`、`ros_life_circle_handler.cc` 中 RosbagParser 启动逻辑、`scripts/slam_offline.sh`
- [ ] **cfg_bk/ 备份目录**：与 `cfg/` 完全重复
- [ ] **aslam_cv_detector + test 文件**：`src/thirdparty/aslam_cv2/aslam_cv_detector/`（KAZE/LSD，代码中未调用）、各模块 `test/` 目录（12个测试文件，不参与构建）
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
