#ifndef YDLIDAR_NODE_H
#define YDLIDAR_NODE_H

#include "ydlidar_node_config.h"
#include "ydlidar_sub_node.h"

namespace ydlidar_node {

class YdlidarNode {
public:
  explicit YdlidarNode(const YdlidarConfig &ydlidar_cfg,
                      std::shared_ptr<rclcpp::Node> nh);
  ~YdlidarNode();

  YdlidarNode(const YdlidarNode &) = delete;
  YdlidarNode &operator=(const YdlidarNode &) = delete;

  void Start();
  void Stop();
private:
  std::vector<YdlidarSubNode *> sub_node_;
};

}

#endif
