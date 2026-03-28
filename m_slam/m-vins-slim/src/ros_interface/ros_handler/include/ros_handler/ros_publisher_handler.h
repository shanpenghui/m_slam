#ifndef MVINS_ROS_PUBLISHER_HANDER_
#define MVINS_ROS_PUBLISHER_HANDER_

#include "interface/interface.h"
#include "ros_handler/configurate_interface.h"


namespace mvins {

struct TopicPublishers {
    rclcpp::Publisher<ImageMsg>::SharedPtr feature_tracking_pub_;
    rclcpp::Publisher<ImageMsg>::SharedPtr feature_depth_pub_;
    rclcpp::Publisher<MarkerMsg>::SharedPtr obs_pub_;
    rclcpp::Publisher<MarkerMsg>::SharedPtr reloc_obs_pub_;
    rclcpp::Publisher<MarkerMsg>::SharedPtr edge_pub_;
    rclcpp::Publisher<PointCloudMsg>::SharedPtr scan_cloud_pub_;
    rclcpp::Publisher<PointCloudMsg>::SharedPtr live_cloud_pub_;
    rclcpp::Publisher<PointCloudMsg>::SharedPtr map_cloud_pub_;
    rclcpp::Publisher<PathMsg>::SharedPtr path_pub_;
    rclcpp::Publisher<PathMsg>::SharedPtr gt_path_pub_;
    rclcpp::Publisher<PoseStampedMsg>::SharedPtr pose_cam_pub_;
    rclcpp::Publisher<PoseStampedMsg>::SharedPtr pose_loop_pub_;
    rclcpp::Publisher<PoseStampedMsg>::SharedPtr pose_pub_;
    rclcpp::Publisher<OdometryMsg>::SharedPtr pose_local_pub_;
    rclcpp::Publisher<OccupancyGridMsg>::SharedPtr map_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

void InitializePublishers(const TopicMap& ros_topics,
    NodeHandle* node_handler_ptr,
    TopicPublishers* topic_publisher_ptr);

void ShutdownPublishers(TopicPublishers* topic_publisher_ptr);
}   // namespace app

#endif
