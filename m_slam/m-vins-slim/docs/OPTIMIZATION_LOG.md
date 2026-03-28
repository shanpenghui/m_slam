# m-vins-slim Optimization Log

> 记录每次“精简/优化”操作的具体实现、影响范围、验证结果与可回滚点。

## 2026-03-28 / refactor(m-vins-slim): remove visual pipeline and keep compilable non-visual node

### 背景与目标
- 目标：仅在 `m-vins-slim` 内部精简，删除视觉相关源码路径，同时保证包仍可编译。
- 约束：不删除 `m-vins-slim` 文件夹本身。

### 实施内容（具体改动）
1. **入口最小化**
   - 新增：`src/m_vins_node.cc`
   - 功能：仅保留 ROS2 `rclcpp` 最小节点，启动并打印日志。

2. **构建系统最小化**
   - 重写：`CMakeLists.txt`
   - 仅保留：
     - `find_package(ament_cmake REQUIRED)`
     - `find_package(rclcpp REQUIRED)`
     - `add_executable(m_vins_node src/m_vins_node.cc)`
     - `ament_target_dependencies(m_vins_node rclcpp)`
     - 安装 `cfg/` 到 `share/${PROJECT_NAME}`

3. **包依赖最小化**
   - 重写：`package.xml`
   - 仅保留 `ament_cmake` + `rclcpp` 依赖。

4. **旧实现保留为可回滚备份（未纳入当前编译）**
   - 原 `src/` 整体迁移为：`src_visual_old/`
   - 包含模块：
     - `ros_interface`
     - `vins_common`
     - `thirdparty`
     - `vins_handler`
     - `vins_core`
     - `octomap_core`
     - `loop_closure`

### 当前状态判定
- 当前编译参与代码：`src/m_vins_node.cc`
- 当前不参与编译（历史备份）：`src_visual_old/**`

### 验证
- 命令：
  ```bash
  source /opt/ros/humble/setup.bash
  cd /home/dev/new_m_slam_ws
  colcon build --symlink-install --packages-select m_vins
  ```
- 结果：`m_vins` 编译完成（Finished）。

### 待处理冗余（下一轮可选）
1. `src_visual_old/`：历史备份代码（约 1.9MB，174 文件），可在确认不再回滚后删除。
2. `third_lib/`：当前最小构建未链接使用，可评估移除。
3. `scripts/`：当前最小构建不依赖，可按运行需求保留或下线。
4. `cfg/`：仅保留运行必需配置，其余可按场景继续裁剪。

### 回滚方式
- 若要恢复旧逻辑：将 `src_visual_old/` 迁回 `src/`，并恢复旧版 `CMakeLists.txt` / `package.xml`。

