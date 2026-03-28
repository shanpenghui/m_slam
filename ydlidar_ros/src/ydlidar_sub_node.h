#ifndef YDLIDAR_SUB_NODE_H
#define YDLIDAR_SUB_NODE_H

#include <mutex>
#include <thread>
#include <chrono>

#include "ydlidar_node_config.h"
#include "CYdLidar.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/int8.hpp"
#include "mirobot_msgs/srv/ydlidar_ros_action.hpp"

namespace ydlidar_node {

typedef sensor_msgs::msg::LaserScan LaserScanMsg;
typedef mirobot_msgs::srv::YdlidarRosAction::Request::SharedPtr YdlidarRosActionRequestPtr;
typedef mirobot_msgs::srv::YdlidarRosAction::Response::SharedPtr YdlidarRosActionResponsePtr;

class YdlidarSubNode {
public:
  explicit YdlidarSubNode(std::string name, 
                          const YdlidarNodeCfg &node_cfg,
                          std::shared_ptr<rclcpp::Node> nh);
  ~YdlidarSubNode();

  void Start();
  void Stop();
private:
  bool sleep_mode_;
  bool sleep_state_;
  int sleep_delay_;
  std::chrono::time_point<std::chrono::steady_clock> activate_time_;
  int dev_num_;
  bool cascade_;
  std::string name_;

  std::vector<CYdLidar *> lidar_;
  std::vector<std::mutex *> lidar_mutex_;
  std::vector<std::string> model_;
  std::vector<LidarVersion> version_;
  std::vector<float> max_range_;
  std::vector<float> freq_;
  std::vector<int> sample_rate_;
  std::vector<int> fail_cnt_;
  std::thread monitor_;
  bool monitor_flg_;
  int monitor_prd_;
  std::thread processor_;
  bool processor_flg_;
  int processor_prd_;

  void Monitor();
  void Processor();

  std::vector<rclcpp::Publisher<LaserScanMsg>::SharedPtr> scan_pub_;
  std::vector<std::string> frame_id_;
  rclcpp::Service<mirobot_msgs::srv::YdlidarRosAction>::SharedPtr action_srv_;
  bool ActionCallBack(YdlidarRosActionRequestPtr req_ptr,
                      YdlidarRosActionResponsePtr res_ptr);
};

}

#endif
