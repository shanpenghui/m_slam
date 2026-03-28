#ifndef MISLAM_ROS_LIFE_CIRCLE_HANDLER_H_
#define MISLAM_ROS_LIFE_CIRCLE_HANDLER_H_

#include "ros_handler/ros_subscriber_handler.h"
#include "ros_handler/ros_publisher_handler.h"
#include "ros_handler/ros_service_handler.h"

namespace mvins {
void StartRosLifeCircle(const int sleep_time_ms,
    const common::SlamConfigPtr& config,
    const mvins::TopicMap& topic_names,
    const std::shared_ptr<NodeHandle>& node_handler_ptr,
    Interface* mislam_interface_ptr,
    TopicSubscriberOptions* subscriber_options_ptr,
    TopicSubscribers* topic_subscribers_ptr,
    TopicPublishers* topic_publishers_ptr,
    ServiceOptions* service_options_ptr,
    Services* services_ptr,
    bool* received_end_signal);
}

#endif