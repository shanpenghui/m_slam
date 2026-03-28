#ifndef MVINS_ROS_INTERFACE_
#define MVINS_ROS_INTERFACE_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int8.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <mirobot_msgs/srv/slam_service.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "cfg_common/slam_config.h"
#include "vins_handler/vins_handler.h"
#include "interface/interface_manager.h"

typedef Eigen::Matrix<double, Eigen::Dynamic, 2> ScanMask;

constexpr char kConfigIdleFileName[] = "config_idle.yaml";
constexpr char kConfigMappingFileName[] = "config_mapping.yaml";
constexpr char kConfigRelocFileName[] = "config_reloc.yaml";

constexpr char kOccupancyMapFileName[] = "map.pgm";
constexpr char kOccupancyMapYamlFileName[] = "map.yaml";

typedef mirobot_msgs::srv::SlamService::Request ServiceRequest;
typedef ServiceRequest::SharedPtr ServiceRequestPtr;
typedef mirobot_msgs::srv::SlamService::Response ServiceResponse;
typedef ServiceResponse::SharedPtr ServiceResponsePtr;
typedef sensor_msgs::msg::LaserScan LaserScanMsg;
typedef LaserScanMsg::SharedPtr LaserScanMsgPtr;
typedef sensor_msgs::msg::Image ImageMsg;
typedef ImageMsg::SharedPtr ImageMsgPtr;
typedef std_msgs::msg::Int8 Int8Msg;
typedef Int8Msg::SharedPtr Int8MsgPtr;
typedef nav_msgs::msg::Odometry OdometryMsg;
typedef OdometryMsg::SharedPtr OdometryMsgPtr;
typedef sensor_msgs::msg::Imu ImuMsg;
typedef ImuMsg::SharedPtr ImuMsgPtr;
typedef sensor_msgs::msg::PointCloud2 PointCloudMsg;
typedef PointCloudMsg::SharedPtr PointCloudMsgPtr;
typedef sensor_msgs::msg::PointField PointFieldMsg;
typedef visualization_msgs::msg::Marker MarkerMsg;
typedef MarkerMsg::SharedPtr MarkerMsgPtr;
typedef nav_msgs::msg::Path PathMsg;
typedef PathMsg::SharedPtr PathMsgPtr;
typedef geometry_msgs::msg::PoseStamped PoseStampedMsg;
typedef PoseStampedMsg::SharedPtr PoseStampedMsgPtr;
typedef geometry_msgs::msg::PoseWithCovarianceStamped PoseWithCovarianceStampedMsg;
typedef PoseWithCovarianceStampedMsg::SharedPtr PoseWithCovarianceStampedMsgPtr;
typedef nav_msgs::msg::OccupancyGrid OccupancyGridMsg;
typedef OccupancyGridMsg::SharedPtr OccupancyGridMsgPtr;
typedef std_msgs::msg::UInt64 UInt64Msg;
typedef UInt64Msg::SharedPtr UInt64MsgPtr;
typedef geometry_msgs::msg::PointStamped PointStampedMsg;
typedef PointStampedMsg::SharedPtr PointStampedMsgPtr;
typedef geometry_msgs::msg::Point32 Point32Msg;
typedef nav_msgs::msg::MapMetaData MapMetaDataMsg;
typedef geometry_msgs::msg::Point PointMsg;
typedef geometry_msgs::msg::Quaternion QuaternionMsg;
typedef geometry_msgs::msg::Vector3 Vector3Msg;
typedef geometry_msgs::msg::TransformStamped TransformStampedMsg;
typedef std_msgs::msg::Header HeaderMsg;
typedef rclcpp::Node NodeHandle;
typedef rclcpp::Time RosTime;

namespace mvins {

enum class SLAM_STATUS {
    RUNNING,
    IDLE,
    STOP,
    OFFLINE
};

enum class SLAM_MODE {
    MAPPING,
    RELOC,
    IDLE
};

class Interface {
public:
    //! Interface constructor.
    Interface(const std::string& slam_yamls_path,
              const common::SlamConfigPtr& config);

    //! Interface destructor.
    ~Interface();

    //! Add end-of-dataset signal to handler.
    void AddEndSignal();

    //! Add camera data from ROS messages.
    void FillImageMsg(
        const ImageMsgPtr msg);
    void FillImageMsgImpl(
        const ImageMsgPtr msg);
    void FillDepthMsg(
        const ImageMsgPtr msg);
    void FillDepthMsgImpl(
        const ImageMsgPtr msg);
    //! Scan related.
    //! Add scan message to handler.
    void FillScanMsg(
        const LaserScanMsgPtr msg);

    //! Add scan pointcloud2 message to handler.
    void FillScanPc2Msg(
        const PointCloudMsgPtr msg);

    //! IMU related.
    //! Add IMU message to handler.
    void FillImuMsg(
        const ImuMsgPtr imu);

    //! Odom related.
    //! Add odom message to handler.
    void FillOdomMsg(
        const OdometryMsgPtr odom);

    void FillGroundTruthMsg(
        const OdometryMsgPtr gt);

    //! Add reset pose message to handler.
    void FillResetPoseMsg(
        const PoseWithCovarianceStampedMsgPtr reset_pose);
        
    bool SlamServiceCall(ServiceRequestPtr req,
                         ServiceResponsePtr res);

    //! Status checker.
    bool IsNewPose() const;
    bool IsDataFinished() const;
    bool IsAllFinished() const;

    //! Getter
    bool GetNewMap(cv::Mat* map_ptr,
                   Eigen::Vector2d* origin_ptr,
                   common::EigenVector3dVec* map_cloud_ptr);
    bool GetNewLoop(common::LoopResult* loop_result_ptr);
    void GetNewKeyFrame(common::KeyFrame* keyframe_ptr);
    void GetTGtoM(aslam::Transformation* T_GtoM_ptr);
    void GetTOtoC(aslam::Transformation* T_OtoC_ptr, size_t id);
    void GetTOtoS(aslam::Transformation* T_OtoS_ptr);
    void GetGtStatus(std::deque<common::OdomData>* gt_status_ptr);
    void GetLiveScan(common::EigenVector3dVec* live_scan_ptr);
    void GetLiveCloud(common::EigenVector4dVec* live_cloud_ptr);
    void GetRelocLandmarks(common::EigenVector3dVec* reloc_landmarks_ptr);
    void GetVizEdges(common::EdgeVec* viz_edge_ptr);
    bool GetLastOdom(common::OdomData& odom);
    SLAM_STATUS GetSlamStatus() const;

    //! Get the last detected visual feature image.
    bool GetShowImage(common::CvMatConstPtrVec* imgs_ptr) const;

    bool HasPoseGraphComplete() const;
    bool HasMappingComplete() const;

    void ResetConfigAndVinsHandlerPtr(const SLAM_MODE& mode);

    void GetPGPoses(common::EigenMatrix4dVec* poses_ptr) const;

    void SetAllFinish();

    void ResetPose(ServiceResponse* res_ref_ptr);
    void SlamStart(ServiceResponse* res_ref_ptr);
    void SlamIdle();
    void LoadMap();
    void SaveMap();
    void RemoveMap();
    void SetMovedPose();
    void ShutDown();
    void GetDockerPose(ServiceResponse* res_ref_ptr);
    void OperateMask(const ServiceRequest& req_ref, ServiceResponse* res_ref_ptr);     
    bool CheckMapExistence();
    bool UseScanPointCloud();
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:

    mutable std::mutex vins_handler_ptr_mutex_;
    std::mutex slam_status_mutex_;
    std::mutex interface_manager_ptr_mutex_;

    std::unique_ptr<vins_handler::VinsHandler> vins_handler_ptr_;
    std::unique_ptr<InterfaceManager> interface_manager_ptr_;

    SLAM_STATUS slam_status_;
    common::SlamConfigPtr config_;
    std::string slam_yamls_path_;
};
}

#endif
