#ifndef MISLAM_CONFIGURATE_TOPICS_H
#define MISLAM_CONFIGURATE_TOPICS_H

#include <string>
#include <utility>
#include <unordered_map>

const char CameraTopicName[] = "camera";
const char CameraDepthTopicName[] = "depth";
const char OdomTopicName[] = "odom";
const char ImuTopicName[] = "imu";
const char ScanTopicName[] = "scan";
const char ScanPointCloud2TopicName[] = "scan_pc2";
const char GroundTruthTopicName[] = "ground_truth";
const char ResetPoseTopicName[] = "reset_pose";
const char FeatureTrackingTopicName[] = "feature_tracking";
const char FeatureDepthTopicName[] = "feature_depth";
const char ObsTopicName[] = "obs_in_opt";
const char RelocObsTopicName[] = "reloc_obs_in_opt";
const char EdgeTopicName[] = "edge_in_opt";
const char ScanCloudTopicName[] = "scan_cloud";
const char LiveCloudTopicName[] = "live_cloud";
const char MapCloudTopicName[] = "map_cloud";
const char PathTopicName[] = "trajectory";
const char GroundTruthPathTopicName[] = "gt_trajectory";
const char PoseCameraTopicName[] = "pose_cam";
const char PoseLoopTopicName[] = "pose_loop";
const char PoseTopicName[] = "pose";
const char PoseLocalTopicName[] = "pose_local";
const char MapTopicName[] = "map";

namespace mvins {
typedef std::unordered_map<std::string, std::string> TopicMap;

TopicMap ConfigurateRosTopics();
}

#endif //MISLAM_CONFIGURATE_TOPICS_H
