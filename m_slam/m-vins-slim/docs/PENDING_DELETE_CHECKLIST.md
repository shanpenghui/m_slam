# Pending Delete Checklist: src_visual_old

> 用户要求：删除前先确认代码内容，有任何问题先人工确认。

## 待删除对象
- 路径：`src_visual_old/`
- 规模：约 1.9MB / 174 files

## 现状确认（2026-03-28）
- 当前编译输入目录：`src/`
- 当前编译源码：`src/m_vins_node.cc`（1 file）
- `src_visual_old/` 不在当前 `CMakeLists.txt` 构建目标中。

## src_visual_old 内容摘要（按模块）
- `ros_interface/`：ROS topic/service 生命周期管理、接口转发
- `vins_common/`：配置/数据结构/日志/时间/栅格地图公共库
- `vins_core/`：优化器、传播器、代价函数
- `vins_handler/`：线程与调度、map 工具
- `octomap_core/`：octomap 相关
- `loop_closure/`：回环相关
- `thirdparty/`：minkindr / libnabo

## 风险提示（删除前需确认）
1. 删除 `src_visual_old/` 后，以上旧实现无法在当前仓库直接恢复（除非依赖 git 历史回滚）。
2. 若后续希望恢复“非最小节点”的 scan/odom/slam 逻辑，可能需要从提交历史恢复文件。
3. 当前 `scripts/` 与 `cfg/` 仍存在，但多数脚本原先面向旧实现；删除 `src_visual_old/` 后脚本可能与当前最小节点不匹配。

## 建议确认项（请用户勾选）
- [ ] 确认不再需要 `ros_interface/vins_core/vins_handler/vins_common/octomap_core/loop_closure` 旧实现
- [ ] 确认接受“仅保留最小可编译节点”作为当前目标
- [ ] 确认后续若要恢复旧逻辑，使用 git 历史回滚
- [ ] 确认执行删除：`rm -rf src_visual_old/`

## 计划中的删除后验证
```bash
source /opt/ros/humble/setup.bash
cd /home/dev/new_m_slam_ws
colcon build --symlink-install --packages-select m_vins
```

