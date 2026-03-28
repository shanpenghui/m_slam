#ifndef MISLAM_ROS_SUBSCRIBER_HANDLER_PLAYER_H
#define MISLAM_ROS_SUBSCRIBER_HANDLER_PLAYER_H

#include "interface/interface.h"
#include "ros_handler/configurate_interface.h"

namespace mvins {

enum class ShutDownType {
    ALL,
    EXCEPT_ODOM
};

// Topic subscribers and options for online mode.
struct TopicSubscriberOptions {
    rclcpp::CallbackGroup::SharedPtr callback_group_scan_;
    rclcpp::SubscriptionOptions sub_options_scan_;
    rclcpp::CallbackGroup::SharedPtr callback_group_img_;
    rclcpp::SubscriptionOptions sub_options_img_;
    rclcpp::CallbackGroup::SharedPtr callback_group_depth_;
    rclcpp::SubscriptionOptions sub_options_depth_;
    rclcpp::CallbackGroup::SharedPtr callback_group_imu_;
    rclcpp::SubscriptionOptions sub_options_imu_;
    rclcpp::CallbackGroup::SharedPtr callback_group_odom_;
    rclcpp::SubscriptionOptions sub_options_odom_;
    rclcpp::CallbackGroup::SharedPtr callback_group_gt_;
    rclcpp::SubscriptionOptions sub_options_gt_;
    rclcpp::CallbackGroup::SharedPtr callback_group_reset_pose_;
    rclcpp::SubscriptionOptions sub_options_reset_pose_;
};

struct TopicSubscribers {
    rclcpp::Subscription<LaserScanMsg>::SharedPtr scan_sub_;
    rclcpp::Subscription<PointCloudMsg>::SharedPtr scan_pc2_sub_;
    rclcpp::Subscription<ImageMsg>::SharedPtr img_sub_;
    rclcpp::Subscription<ImageMsg>::SharedPtr depth_sub_;
    rclcpp::Subscription<ImuMsg>::SharedPtr imu_sub_;
    rclcpp::Subscription<OdometryMsg>::SharedPtr odom_sub_;
    rclcpp::Subscription<OdometryMsg>::SharedPtr gt_sub_;
    rclcpp::Subscription<PoseWithCovarianceStampedMsg>::SharedPtr reset_pose_sub_;
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
}  // namespace mislam_application

#endif  //MISLAM_ROS_SUBSCRIBER_HANDLER_PLAYER_H
