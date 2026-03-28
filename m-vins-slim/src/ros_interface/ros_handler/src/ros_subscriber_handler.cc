#include "ros_handler/ros_subscriber_handler.h"
#ifdef USE_ROS2
#include <rclcpp/rclcpp.hpp>
#else
#include <ros/ros.h>
#endif
#include <glog/logging.h>

namespace mvins {
void InitializeSubscriberOptions(
        NodeHandle* node_handler_ptr,
        TopicSubscriberOptions* subscriber_options_ptr) {
    TopicSubscriberOptions& subscriber_options = *CHECK_NOTNULL(subscriber_options_ptr);
#ifdef USE_ROS2
    subscriber_options.callback_group_odom_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_odom_.callback_group = subscriber_options.callback_group_odom_;

    subscriber_options.callback_group_scan_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_scan_.callback_group = subscriber_options.callback_group_scan_;

    subscriber_options.callback_group_gt_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_gt_.callback_group = subscriber_options.callback_group_gt_;

    subscriber_options.callback_group_reset_pose_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_reset_pose_.callback_group = subscriber_options.callback_group_reset_pose_;
#endif
}

void InitializeSubscribers(
        const mvins::TopicMap& ros_topics,
        NodeHandle* node_handler_ptr,
        mvins::Interface* interface_ptr,
        TopicSubscriberOptions* subscriber_options_ptr,
        TopicSubscribers* topic_subscriber_ptr) {
    NodeHandle& node_handler = *CHECK_NOTNULL(node_handler_ptr);
    mvins::Interface& interface = *CHECK_NOTNULL(interface_ptr);
    TopicSubscriberOptions& subscriber_options = *CHECK_NOTNULL(subscriber_options_ptr);
    TopicSubscribers& topic_subscriber = *CHECK_NOTNULL(topic_subscriber_ptr);
#ifdef USE_ROS2
    topic_subscriber.odom_sub_ = node_handler.create_subscription<OdometryMsg>(
        ros_topics.at(OdomTopicName), rclcpp::SensorDataQoS(),
        std::bind(&Interface::FillOdomMsg,
                  &interface,
                  std::placeholders::_1),
        subscriber_options.sub_options_odom_);

    topic_subscriber.scan_sub_ = node_handler.create_subscription<LaserScanMsg>(
        ros_topics.at(ScanTopicName), rclcpp::SensorDataQoS(),
        std::bind(&Interface::FillScanMsg,
                  &interface,
                  std::placeholders::_1),
        subscriber_options.sub_options_scan_);

    topic_subscriber.gt_sub_ = node_handler.create_subscription<OdometryMsg>(
        ros_topics.at(GroundTruthTopicName), rclcpp::SensorDataQoS(),
        std::bind(&Interface::FillGroundTruthMsg,
                  &interface,
                  std::placeholders::_1),
        subscriber_options.sub_options_gt_);

    topic_subscriber.reset_pose_sub_ = node_handler.create_subscription<PoseWithCovarianceStampedMsg>(
        ros_topics.at(ResetPoseTopicName), rclcpp::SystemDefaultsQoS(),
        std::bind(&Interface::FillResetPoseMsg,
                  &interface,
                  std::placeholders::_1),
                  subscriber_options.sub_options_reset_pose_);
#else
    topic_subscriber.odom_sub_ = std::make_shared<ros::Subscriber>(node_handler.subscribe(
        ros_topics.at(OdomTopicName), 100,
        &Interface::FillOdomMsg,
        &interface));

    topic_subscriber.scan_sub_ = std::make_shared<ros::Subscriber>(node_handler.subscribe(
        ros_topics.at(ScanTopicName), 10,
        &Interface::FillScanMsg,
        &interface));

    topic_subscriber.gt_sub_ = std::make_shared<ros::Subscriber>(node_handler.subscribe(
        ros_topics.at(GroundTruthTopicName), 100,
        &Interface::FillGroundTruthMsg,
        &interface));

    topic_subscriber.reset_pose_sub_ = std::make_shared<ros::Subscriber>(node_handler.subscribe(
        ros_topics.at(ResetPoseTopicName), 2,
        &Interface::FillResetPoseMsg,
        &interface));
#endif
    VLOG(0) << "SLAM subscribers have been initialize successfully";
}

void ShutdownSubscribers(
        TopicSubscribers* topic_subscriber_ptr, ShutDownType type) {
    TopicSubscribers& topic_subscriber = *CHECK_NOTNULL(topic_subscriber_ptr);
#ifdef USE_ROS2
    topic_subscriber.scan_sub_.reset();
    topic_subscriber.reset_pose_sub_.reset();
    if (type == ShutDownType::ALL) {
        topic_subscriber.odom_sub_.reset();
    }
#else
    topic_subscriber.scan_sub_->shutdown();
    topic_subscriber.reset_pose_sub_->shutdown();
    if (type == ShutDownType::ALL) {
        topic_subscriber.odom_sub_->shutdown();
    }
#endif
    VLOG(0) << "SLAM subscribers have been shutdown";
}

}  // namespace mvins
