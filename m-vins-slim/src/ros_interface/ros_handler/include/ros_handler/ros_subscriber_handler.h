#ifndef MISLAM_ROS_SUBSCRIBER_HANDLER_PLAYER_H
#define MISLAM_ROS_SUBSCRIBER_HANDLER_PLAYER_H

#include "interface/interface.h"
#include "ros_handler/configurate_interface.h"

namespace mvins {

enum class ShutDownType {
    ALL,
    EXCEPT_ODOM
};

struct TopicSubscriberOptions {
#ifdef USE_ROS2
    rclcpp::CallbackGroup::SharedPtr callback_group_scan_;
    rclcpp::SubscriptionOptions sub_options_scan_;
    rclcpp::CallbackGroup::SharedPtr callback_group_odom_;
    rclcpp::SubscriptionOptions sub_options_odom_;
    rclcpp::CallbackGroup::SharedPtr callback_group_gt_;
    rclcpp::SubscriptionOptions sub_options_gt_;
    rclcpp::CallbackGroup::SharedPtr callback_group_reset_pose_;
    rclcpp::SubscriptionOptions sub_options_reset_pose_;
#endif
};

struct TopicSubscribers {
#ifdef USE_ROS2
    rclcpp::Subscription<LaserScanMsg>::SharedPtr scan_sub_;
    rclcpp::Subscription<OdometryMsg>::SharedPtr odom_sub_;
    rclcpp::Subscription<OdometryMsg>::SharedPtr gt_sub_;
    rclcpp::Subscription<PoseWithCovarianceStampedMsg>::SharedPtr reset_pose_sub_;
#else
    std::shared_ptr<ros::Subscriber> scan_sub_;
    std::shared_ptr<ros::Subscriber> odom_sub_;
    std::shared_ptr<ros::Subscriber> gt_sub_;
    std::shared_ptr<ros::Subscriber> reset_pose_sub_;
#endif
};

void InitializeSubscriberOptions(
        NodeHandle* node_handler_ptr,
        TopicSubscriberOptions* subscriber_options_ptr);

void InitializeSubscribers(
        const TopicMap& ros_topics,
        NodeHandle* node_handler_ptr,
        Interface* interface_ptr,
        TopicSubscriberOptions* subscriber_options_ptr,
        TopicSubscribers* topic_subscriber_ptr);

void ShutdownSubscribers(TopicSubscribers* topic_subscriber_ptr, ShutDownType type);
}  // namespace mvins

#endif  //MISLAM_ROS_SUBSCRIBER_HANDLER_PLAYER_H
