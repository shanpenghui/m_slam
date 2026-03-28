#include "ros_handler/ros_subscriber_handler.h"
#include <rclcpp/rclcpp.hpp>
#include <glog/logging.h>

namespace mvins {
void InitializeSubscriberOptions(
        NodeHandle* node_handler_ptr,
        TopicSubscriberOptions* subscriber_options_ptr) {
    TopicSubscriberOptions& subscriber_options = *CHECK_NOTNULL(subscriber_options_ptr);
    subscriber_options.callback_group_img_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_img_.callback_group = subscriber_options.callback_group_img_;

    subscriber_options.callback_group_depth_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_depth_.callback_group = subscriber_options.callback_group_depth_;

    subscriber_options.callback_group_odom_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_odom_.callback_group = subscriber_options.callback_group_odom_;

    subscriber_options.callback_group_imu_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_imu_.callback_group = subscriber_options.callback_group_imu_;

    subscriber_options.callback_group_scan_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_scan_.callback_group = subscriber_options.callback_group_scan_;

    subscriber_options.callback_group_gt_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_gt_.callback_group = subscriber_options.callback_group_gt_;

    subscriber_options.callback_group_reset_pose_ =
        node_handler_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    subscriber_options.sub_options_reset_pose_.callback_group = subscriber_options.callback_group_reset_pose_;
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
    // Set image subscriber.
    topic_subscriber.img_sub_ = node_handler.create_subscription<ImageMsg>(
        ros_topics.at(CameraTopicName), rclcpp::SensorDataQoS(),
        std::bind(&Interface::FillImageMsg,
                  &interface,
                  std::placeholders::_1),
        subscriber_options.sub_options_img_);

    // Set depth subscriber.
    topic_subscriber.depth_sub_ = node_handler.create_subscription<ImageMsg>(
        ros_topics.at(CameraDepthTopicName), rclcpp::SensorDataQoS(),
        std::bind(&Interface::FillDepthMsg,
                  &interface,
                  std::placeholders::_1),
        subscriber_options.sub_options_depth_);

    // Set odom subscriber.
    topic_subscriber.odom_sub_ = node_handler.create_subscription<OdometryMsg>(
        ros_topics.at(OdomTopicName), rclcpp::SensorDataQoS(),
        std::bind(&Interface::FillOdomMsg,
                  &interface,
                  std::placeholders::_1),
        subscriber_options.sub_options_odom_);

    // Set imu subscriber.
    topic_subscriber.imu_sub_ = node_handler.create_subscription<ImuMsg>(
            ros_topics.at(ImuTopicName), rclcpp::SensorDataQoS(),
            std::bind(&Interface::FillImuMsg,
                      &interface,
                      std::placeholders::_1),
            subscriber_options.sub_options_imu_);

    if (interface_ptr->UseScanPointCloud()) {
        // Set scan_pc2 subscriber.
        topic_subscriber.scan_pc2_sub_ = node_handler.create_subscription<PointCloudMsg>(
            ros_topics.at(ScanPointCloud2TopicName), 10,
            std::bind(&Interface::FillScanPc2Msg,
                      &interface,
                      std::placeholders::_1),
            subscriber_options.sub_options_scan_);
    } else {
        // Set scan subscriber.
        topic_subscriber.scan_sub_ = node_handler.create_subscription<LaserScanMsg>(
            ros_topics.at(ScanTopicName), rclcpp::SensorDataQoS(),
            std::bind(&Interface::FillScanMsg,
                      &interface,
                      std::placeholders::_1),
            subscriber_options.sub_options_scan_);
    }
    
    // Set ground truth subscriber.
    topic_subscriber.gt_sub_ = node_handler.create_subscription<OdometryMsg>(
        ros_topics.at(GroundTruthTopicName), rclcpp::SensorDataQoS(),
        std::bind(&Interface::FillGroundTruthMsg,
                  &interface,
                  std::placeholders::_1),
        subscriber_options.sub_options_gt_);

    // Set reset pose subscriber.
    topic_subscriber.reset_pose_sub_ = node_handler.create_subscription<PoseWithCovarianceStampedMsg>(
        ros_topics.at(ResetPoseTopicName), rclcpp::SystemDefaultsQoS(),
        std::bind(&Interface::FillResetPoseMsg,
                  &interface,
                  std::placeholders::_1),
                  subscriber_options.sub_options_reset_pose_);
    VLOG(0) << "SLAM subscribers have been initialize successfully";
}

void ShutdownSubscribers(
        TopicSubscribers* topic_subscriber_ptr, ShutDownType type) {
    TopicSubscribers& topic_subscriber = *CHECK_NOTNULL(topic_subscriber_ptr);
    topic_subscriber.img_sub_.reset();
    topic_subscriber.depth_sub_.reset();
    topic_subscriber.imu_sub_.reset();
    if (topic_subscriber.scan_sub_ != nullptr) {
        topic_subscriber.scan_sub_.reset();
    } else {
        CHECK_NOTNULL(topic_subscriber.scan_pc2_sub_);
        topic_subscriber.scan_pc2_sub_.reset();
    }
    topic_subscriber.reset_pose_sub_.reset();
    if (type == ShutDownType::ALL) {
        topic_subscriber.odom_sub_.reset();
    }
    VLOG(0) << "SLAM subscribers have been shutdown";
}

}  // namespace app
