# m-vins-slim

基于原始 `m-vins` 工程，按当前仓库里实际启用的运行方式做了裁剪，只保留：

- ROS2
- 在线运行
- `scan + odom` 主流程
- `mapping / reloc / idle` 三种在线状态切换
- 2D 栅格建图、定位、scan loop closure、mask 数据库

已移除或停用的部分：

- ROS1 / catkin 路径
- 离线 rosbag 回放链路
- 图像 / 深度 / IMU 订阅入口
- CNN / RKNN / SuperPoint 资源
- 当前未使用的多套标定与 mask
- 备份配置、评测工具、未参与当前构建的第三方源码

当前工程默认按 `cfg/launch_params.yaml` + `cfg/config_{idle,mapping,reloc}.yaml` 工作，
并以 `cfg/calib_avaia_cleaner.yaml` 为当前标定。

## Optimization Record

- 详细优化记录见：`docs/OPTIMIZATION_LOG.md`
- 约定：后续每次精简/优化都在该文档追加“目标、改动明细、验证结果、可回滚点”。
