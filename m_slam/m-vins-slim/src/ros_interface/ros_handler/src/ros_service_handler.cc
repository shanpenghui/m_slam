#include "ros_handler/ros_service_handler.h"

namespace mvins {

void InitializeServiceOptions(
    NodeHandle* node_handler_ptr,
    ServiceOptions* service_options_ptr) {
    NodeHandle& node_handler = *CHECK_NOTNULL(node_handler_ptr);
    ServiceOptions& service_options = *CHECK_NOTNULL(service_options_ptr);
#ifdef USE_ROS2
    service_options.slam_interactive_callback_group_ = 
        node_handler.create_callback_group(rclcpp::CallbackGroupType::Reentrant);
#endif
}

void InitializeServices(NodeHandle* node_handler_ptr,
    Interface* interface_ptr,
    ServiceOptions* service_options_ptr,
    Services* services_ptr) {
    NodeHandle& node_handler = *CHECK_NOTNULL(node_handler_ptr);
    Interface& interface = *CHECK_NOTNULL(interface_ptr);
    ServiceOptions& service_options = *CHECK_NOTNULL(service_options_ptr);
    Services& services = *CHECK_NOTNULL(services_ptr);
#ifdef USE_ROS2
    services.slam_interactive_service_ = node_handler.create_service<mirobot_msgs::srv::SlamService>(
        "slam_service",
        std::bind(&Interface::SlamServiceCall,
                  &interface,
                  std::placeholders::_1,
                  std::placeholders::_2),
                  rclcpp::ServicesQoS().get_rmw_qos_profile(),
                  service_options.slam_interactive_callback_group_);
#else
    services.slam_interactive_service_ = std::make_shared<ros::ServiceServer>(node_handler.advertiseService(
        "slam_service",
        &Interface::SlamServiceCall,
        &interface));
#endif
    VLOG(0) << "SLAM services have been initialize successfull";
}

void ShutdownServices(Services* services_ptr) {
    Services& services = *CHECK_NOTNULL(services_ptr);
#ifdef USE_ROS2
    services.slam_interactive_service_.reset();
#else
    services.slam_interactive_service_->shutdown();
#endif
    VLOG(0) << "SLAM services have been shutdown";
}
}