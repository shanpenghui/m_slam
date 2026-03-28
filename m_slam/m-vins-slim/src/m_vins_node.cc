#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("m_vins_node");
  RCLCPP_INFO(node->get_logger(), "m_vins (non-visual slim) node started");
  rclcpp::spin_some(node);
  rclcpp::shutdown();
  return 0;
}
