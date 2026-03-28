#include "ydlidar_node.h"

#include "glog/logging.h"

namespace ydlidar_node {

YdlidarNode::YdlidarNode(const YdlidarConfig &ydlidar_cfg,
                        std::shared_ptr<rclcpp::Node> nh) {
  int node_num = std::min(ydlidar_cfg.ydlidar_node_list.size(), 
                          ydlidar_cfg.node_cfg.size());
  for (int i = 0; i < node_num; ++i) {
    LOG(INFO) << "create sub node: " << ydlidar_cfg.ydlidar_node_list[i];
    sub_node_.push_back(new YdlidarSubNode(ydlidar_cfg.ydlidar_node_list[i],
                                          ydlidar_cfg.node_cfg[i],
                                          nh));
  }
}

YdlidarNode::~YdlidarNode() {
  for (auto sub_node: sub_node_) {
    delete sub_node;
  }
}

void YdlidarNode::Start() {
  for (auto sub_node: sub_node_) {
    sub_node->Start();
  }
}

void YdlidarNode::Stop() {
  for (auto sub_node: sub_node_) {
    sub_node->Stop();
  }
}

}
