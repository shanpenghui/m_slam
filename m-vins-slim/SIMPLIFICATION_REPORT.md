# Simplification report

本次裁剪的目标不是保留所有历史能力，而是保留当前仓库中已经固定下来的在线 ROS2 scan 模式。

## 保留的主链路

`m_vins_node -> ros_interface -> vins_handler -> scan loop / 2D occupancy map`

## 主要删除项

1. 只为 ROS1 / catkin 准备的分支
2. 离线 rosbag 回放入口与脚本
3. 图像 / 深度 / IMU 在线订阅入口
4. RKNN / SuperPoint 资源与相关第三方二进制
5. 备份配置目录 `cfg_bk`
6. 评测工具目录 `tools`
7. 当前未使用的多套标定与 mask
8. 未参与当前构建的第三方源码（例如 `aslam_cv_detector`、未编译的 xfeatures2d 源文件等）

## 仍然保留但未深拆的部分

为了尽量降低改动风险，`vins_handler / vins_core / loop_closure` 里与视觉相关的深层实现仍保留在源码中，
但当前简化工程的启动链路不会再主动构造图像订阅、离线回放和 CNN 资源依赖。


## 体量变化

- 原始工程：约 21M，约 589 个文件/目录项
- 裁剪后工程：约 6.1M，约 455 个文件/目录项

说明：这里的统计包含源码、配置、脚本与第三方子目录，不代表最终安装产物大小。
