#ifndef MVINS_ROS_INTERFACE_MANAGER_
#define MVINS_ROS_INTERFACE_MANAGER_

#include <mutex>

#include <octomap/ColorOcTree.h>

#include "aslam/common/pose-types.h"
#include "cfg_common/slam_config.h"
#include "data_common/state_structures.h"
#include "file_common/file_system_tools.h"
#include "occ_common/live_submaps.h"
#include "sqlite_common/sqlite_util.h"
#include "summary_map/summary_map.h"

using Line = Eigen::Vector4f;
using Polygon = Eigen::Matrix2Xf;
namespace mvins {

class InterfaceManager
{
public:
    enum class MaskOperation {
        ADD,
        MODIFY,
        DELETE,
        ERROR
    };

    InterfaceManager(const common::SlamConfigPtr& config);
    InterfaceManager() = delete;
    ~InterfaceManager();

    bool IsMovedPosePtrNull();
    void ResetMovedPosePtr(const aslam::Transformation& pose);
    void ResetMovedPosePtr();
    aslam::Transformation* GetMovedPosePtr();

    bool IsDockerPosePtrNull();
    void ResetDockerPosePtr(const aslam::Transformation& pose);
    void ResetDockerPosePtr();
    aslam::Transformation* GetDockerPosePtr();

    bool IsMaskDatabaseOpen();
    MaskOperation ParamToMaskOperation(const float& param);
    bool SaveMaskOperationInMaskDatabase(const std::vector<float>& params);

    void AddOdom(const common::OdomData& odom_meas);

    bool GetLastOdom(common::OdomData& odom);
    void SetLastOdomPtr();

    common::KeyFrame* GetLastKeyFrameBeforeIdlePtr();
    void SetLastKeyFrameBeforeIdlePtr(const common::KeyFrame& last_keyframe);
    void SetLastKeyFrameBeforeIdlePtr();

    aslam::Transformation* GetLastTGtoMBeforeIdlePtr();
    void SetLastTGtoMBeforeIdlePtr(const aslam::Transformation& T_GtoM);
    void SetLastTGtoMBeforeIdlePtr();

    void MapThresholding(cv::Mat* map_ptr, Eigen::Vector2d* origin_ptr);

    void CreateMaskTable();
    void TryLoadMap(const std::string& map_path);

    std::shared_ptr<common::LiveSubmaps> scan_maps_ptr_;
    std::shared_ptr<loop_closure::SummaryMap> summary_map_ptr_;
    std::shared_ptr<octomap::ColorOcTree> octree_map_ptr_;

private:

    void DrawMaskOnMap(cv::Mat* map_ptr, Eigen::Vector2d* origin_ptr);
    cv::Point2i ChangePointWorldToMap(const cv::Point2f& wp, const Eigen::Vector2d& origin, int map_rows, int map_cols);
    cv::Point2f ChangePointMapToWorld(const cv::Point2i& mp, const Eigen::Vector2d& origin, int map_rows, int map_cols);

    void LoadMaskTable();
    void LoadMaskLine();
    void LoadMaskPolygon();

    bool IsMaskLineExist(const size_t line_index);
    void AddMaskLine(const size_t line_index, const Line& line);
    void ModifyMaskLine(const size_t line_index, const Line& line);
    void DeleteMaskLine(const size_t line_index);
    
    bool IsMaskPolygonExist(const size_t polygon_index);
    void AddMaskPolygon(const size_t polygon_index, const int num_of_points, const Polygon& polygon);
    void ModifyMaskPolygon(const size_t polygon_index, const int num_of_points, const Polygon& polygon);
    void DeleteMaskPolygon(const size_t polygon_index);

    common::SlamConfigPtr config_;

    std::map<size_t, Line> mask_lines_;
    std::map<size_t, Polygon> mask_polygons_;
    sqlite3* mask_database_;

    std::unique_ptr<aslam::Transformation> moved_pose_ptr_;
    std::mutex move_pose_mutex_;

    std::unique_ptr<aslam::Transformation> docker_pose_ptr_;
    std::mutex docker_pose_mutex_;

    // last keyframe from vins handler
    std::unique_ptr<common::KeyFrame> last_keyframe_before_idle_ptr_;
    std::mutex last_key_frame_mutex_;

    // last T_GtoM from vins handler
    std::unique_ptr<aslam::Transformation> last_T_GtoM_before_idle_ptr_;
    std::mutex lase_T_GtoM_before_idle_mutex_;

    // last odom refreshed in IDLE mode
    std::unique_ptr<common::OdomData> last_odom_ptr_;
    std::mutex odom_mutex_;
    
    
};
}
#endif