# 项目名称

本项目是美的集团AIIC机器人团队设计研发的基于多传感器融合的SLAM（simultaneous localization and mapping）系统。
实现在室内环境下的环境感知建图、定位功能。

### 前提条件

安装和使用项目环境：

- Ubuntu18.04以上
- ROS melodic或ROS2 foxy以上
- git

### 安装

1. 克隆项目仓库

```bash
cd ~/row_ws/src
git clone https://git.midea.com/navigation/m-vins.git
```

2. 编译项目
```bash
catkin_make
```
