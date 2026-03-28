#ifndef MISLAM_ROS_SERVICE_HANDLER_H_
#define MISLAM_ROS_SERVICE_HANDLER_H_

#include "interface/interface.h"
#include "ros_handler/configurate_interface.h"

namespace mvins {

struct ServiceOptions {
    rclcpp::CallbackGroup::SharedPtr slam_interactive_callback_group_;
};

struct Services {
    rclcpp::Service<mirobot_msgs::srv::SlamService>::SharedPtr slam_interactive_service_;
};

void InitializeServiceOptions(
    NodeHandle* node_handler_ptr,
    ServiceOptions* service_options_ptr);

void InitializeServices(
        NodeHandle* node_handler_ptr,
        Interface* interface_ptr,
        ServiceOptions* service_options_ptr,
        Services* services_ptr);

void ShutdownServices(Services* services_ptr);
}

#endif