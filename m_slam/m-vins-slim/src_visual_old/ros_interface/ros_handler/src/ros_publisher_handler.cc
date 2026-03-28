#include "ros_handler/ros_publisher_handler.h"

namespace mvins {

void InitializePublishers(const mvins::TopicMap& ros_topics,
    NodeHandle* node_handler_ptr,
    TopicPublishers* topic_publisher_ptr) {
    NodeHandle& node_handler = *CHECK_NOTNULL(node_handler_ptr);
    mvins::TopicPublishers& topic_publishers = *CHECK_NOTNULL(topic_publisher_ptr);

    // Publish message flows.
#ifdef USE_ROS2
    topic_publishers.feature_tracking_pub_ = node_handler.create_publisher<ImageMsg>(
            ros_topics.at(FeatureTrackingTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.feature_depth_pub_ = node_handler.create_publisher<ImageMsg>(
            ros_topics.at(FeatureDepthTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.obs_pub_ = node_handler.create_publisher<MarkerMsg>(
            ros_topics.at(ObsTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.reloc_obs_pub_ = node_handler.create_publisher<MarkerMsg>(
            ros_topics.at(RelocObsTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.edge_pub_ = node_handler.create_publisher<MarkerMsg>(
            ros_topics.at(EdgeTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.scan_cloud_pub_ = node_handler.create_publisher<PointCloudMsg>(
            ros_topics.at(ScanCloudTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.live_cloud_pub_ = node_handler.create_publisher<PointCloudMsg>(
            ros_topics.at(LiveCloudTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.map_cloud_pub_ = node_handler.create_publisher<PointCloudMsg>(
            ros_topics.at(MapCloudTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.path_pub_ = node_handler.create_publisher<PathMsg>(
            ros_topics.at(PathTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.gt_path_pub_ = node_handler.create_publisher<PathMsg>(
            ros_topics.at(GroundTruthPathTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.pose_cam_pub_ = node_handler.create_publisher<PoseStampedMsg>(
            ros_topics.at(PoseCameraTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.pose_loop_pub_ = node_handler.create_publisher<PoseStampedMsg>(
            ros_topics.at(PoseLoopTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.pose_pub_ = node_handler.create_publisher<PoseStampedMsg>(
            ros_topics.at(PoseTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.pose_local_pub_ = node_handler.create_publisher<OdometryMsg>(
            ros_topics.at(PoseLocalTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.map_pub_ = node_handler.create_publisher<OccupancyGridMsg>(
            ros_topics.at(MapTopicName), rclcpp::SystemDefaultsQoS());
    topic_publishers.tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster(node_handler));
#else
    image_transport::ImageTransport img_trans(node_handler);
    topic_publishers.feature_tracking_pub_ = std::make_shared<image_transport::Publisher>(img_trans.advertise(
            ros_topics.at(FeatureTrackingTopicName), 10));
    topic_publishers.feature_depth_pub_ = std::make_shared<image_transport::Publisher>(img_trans.advertise(
            ros_topics.at(FeatureDepthTopicName), 10));
    topic_publishers.obs_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<MarkerMsg>(
            ros_topics.at(ObsTopicName), 10));
    topic_publishers.reloc_obs_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<MarkerMsg>(
            ros_topics.at(RelocObsTopicName), 10));
    topic_publishers.edge_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<MarkerMsg>(
            ros_topics.at(EdgeTopicName), 10));
    topic_publishers.scan_cloud_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PointCloudMsg>(
            ros_topics.at(ScanCloudTopicName), 10));
    topic_publishers.live_cloud_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PointCloudMsg>(
            ros_topics.at(LiveCloudTopicName), 10));
    topic_publishers.map_cloud_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PointCloudMsg>(
            ros_topics.at(MapCloudTopicName), 10));
    topic_publishers.path_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PathMsg>(
            ros_topics.at(PathTopicName), 10));
    topic_publishers.gt_path_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PathMsg>(
            ros_topics.at(GroundTruthPathTopicName), 10));
    topic_publishers.pose_cam_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PoseStampedMsg>(
            ros_topics.at(PoseCameraTopicName), 10));
    topic_publishers.pose_loop_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PoseStampedMsg>(
            ros_topics.at(PoseLoopTopicName), 10));
    topic_publishers.pose_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<PoseStampedMsg>(
            ros_topics.at(PoseTopicName), 10));
    topic_publishers.pose_local_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<OdometryMsg>(
            ros_topics.at(PoseLocalTopicName), 10));
    topic_publishers.map_pub_ = std::make_shared<ros::Publisher>(node_handler.advertise<OccupancyGridMsg>(
            ros_topics.at(MapTopicName), 1));
    topic_publishers.tf_broadcaster_.reset(new tf::TransformBroadcaster());
#endif
    VLOG(0) << "SLAM publishers have been initialize successfull";
}

void ShutdownPublishers(TopicPublishers* topic_publisher_ptr) {
    TopicPublishers& topic_publisher = *CHECK_NOTNULL(topic_publisher_ptr);
#ifdef USE_ROS2
    topic_publisher.feature_tracking_pub_.reset();
    topic_publisher.feature_depth_pub_.reset();
    topic_publisher.obs_pub_.reset();
    topic_publisher.reloc_obs_pub_.reset();
    topic_publisher.edge_pub_.reset();
    topic_publisher.scan_cloud_pub_.reset();
    topic_publisher.live_cloud_pub_.reset();
    topic_publisher.map_cloud_pub_.reset();
    topic_publisher.path_pub_.reset();
    topic_publisher.pose_cam_pub_.reset();
    topic_publisher.pose_loop_pub_.reset();
    topic_publisher.pose_pub_.reset();
    topic_publisher.map_pub_.reset();
    topic_publisher.tf_broadcaster_.reset();
#else
    topic_publisher.feature_tracking_pub_->shutdown();
    topic_publisher.feature_depth_pub_->shutdown();
    topic_publisher.obs_pub_->shutdown();
    topic_publisher.reloc_obs_pub_->shutdown();
    topic_publisher.edge_pub_->shutdown();
    topic_publisher.scan_cloud_pub_->shutdown();
    topic_publisher.live_cloud_pub_->shutdown();
    topic_publisher.map_cloud_pub_->shutdown();
    topic_publisher.path_pub_->shutdown();
    topic_publisher.pose_cam_pub_->shutdown();
    topic_publisher.pose_loop_pub_->shutdown();
    topic_publisher.pose_pub_->shutdown();
    topic_publisher.map_pub_->shutdown();
    topic_publisher.tf_broadcaster_.reset();
#endif
    VLOG(0) << "SLAM publishers have been shutdown";
}
}