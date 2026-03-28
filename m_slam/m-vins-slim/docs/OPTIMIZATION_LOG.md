# m-vins-slim Optimization Log

> 记录每次精简/优化操作的具体实现、影响范围、验证结果与可回滚点。

---

## 2026-03-28 / Remove ROS1 conditional code and configuration

### 背景
- 项目已全面迁移至 ROS2 Humble，ROS1 (catkin/roscpp) 代码路径不再使用。
- 源码中大量 #ifdef USE_ROS2 ... #else ... #endif 双版本兼容代码，增加维护负担。

### 实施内容

#### 1. 源码 (#ifdef/#else/#endif)
- 移除所有 #ifdef USE_ROS2 / #ifndef USE_ROS2 中的 ROS1 #else 分支
- 保留 ROS2 代码，去掉条件编译守卫
- 涉及 **17 个源文件**，共移除 **64 个 ROS1 条件块**
- 受影响模块：
  - src/ros_interface/ (m_vins_node.cc, interface/*.cc, ros_handler/*.cc/h)
  - src/ros_interface/interface/include/interface/interface.h

#### 2. CMakeLists.txt
- 顶层 CMakeLists.txt：
  - 移除 USE_ROS2_API 变量定义和 dd_definitions(-DUSE_ROS2) 
  - 移除 catkin find_package / catkin_package / ROS1 install 块
  - ROS2 find_package 和 install 不再需要 if 包裹
- 子模块 CMakeLists（ros_interface, vins_core, vins_common, vins_handler, octomap_core, loop_closure, third_lib, thirdparty/libnabo, thirdparty/xfeatures2d）：
  - 移除 if(USE_ROS2_API)...endif() 包裹，install 语句直接执行
- 附带修复：所有子模块 cmake_minimum_required(VERSION 3.0) 升级为 VERSION 3.5

#### 3. package.xml
- 移除注释掉的 ROS1 依赖块（catkin, roscpp, image_transport, tf 等）

### 未改动部分
- src/thirdparty/aslam_cv2/ 内部的 catkin 引用：属于第三方库自身构建文件，不参与当前编译，保持原样
- #ifdef PUB_DEBUG_INFO、#ifdef USE_CNN_FEATURE、#ifdef __ARM_NEON__ 等其他条件编译：未变动

### 验证结果
`
$ colcon build --symlink-install --packages-select m_vins
Finished <<< m_vins [17min 13s]
`
- 仅有 Eigen NEON memcpy 和 unused-parameter 警告（非阻塞，与本次改动无关）

### 回滚方式
`ash
git revert <commit-hash>
`

