#include "ros_handler/configurate_interface.h"

namespace mvins {

TopicMap ConfigurateRosTopics() {
    TopicMap ros_topics_map;
    ros_topics_map[CameraTopicName] = "/camera/rgb/image_raw";
    ros_topics_map[CameraDepthTopicName] = "/camera/depth/image_raw";
    ros_topics_map[OdomTopicName] = "/odom";
    ros_topics_map[ImuTopicName] = "/imu";
    ros_topics_map[ScanTopicName] = "/scan";
    ros_topics_map[ScanPointCloud2TopicName] = "/scan_pc2";
    ros_topics_map[GroundTruthTopicName] = "/odom_gt";
    ros_topics_map[ResetPoseTopicName] = "/initialpose";
    ros_topics_map[FeatureTrackingTopicName] = "/feature_tracking";
    ros_topics_map[FeatureDepthTopicName] = "/feature_depth";
    ros_topics_map[ObsTopicName] = "/obs_in_opt";
    ros_topics_map[RelocObsTopicName] = "/reloc_obs_in_opt";
    ros_topics_map[EdgeTopicName] = "/edge_in_opt";
    ros_topics_map[ScanCloudTopicName] = "/scan_cloud";
    ros_topics_map[LiveCloudTopicName] = "/live_cloud";
    ros_topics_map[MapCloudTopicName] = "/map_cloud";
    ros_topics_map[PathTopicName] = "/trajectory";
    ros_topics_map[GroundTruthPathTopicName] = "/gt_trajectory";
    ros_topics_map[PoseCameraTopicName] = "/pose_cam";
    ros_topics_map[PoseLoopTopicName] = "/pose_loop";
    ros_topics_map[PoseTopicName] = "/pose";
    ros_topics_map[PoseLocalTopicName] = "/pose_local";
    ros_topics_map[MapTopicName] = "/map";

    return ros_topics_map;
}

}
