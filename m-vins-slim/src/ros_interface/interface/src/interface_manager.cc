#include "interface/interface_manager.h"

#include "data_common/constants.h"
#include "summary_map/map_loader.h"

namespace mvins {

InterfaceManager::InterfaceManager(const common::SlamConfigPtr& config)
    : config_(config),
    mask_database_(nullptr), 
    moved_pose_ptr_(nullptr), 
    docker_pose_ptr_(nullptr),
    last_keyframe_before_idle_ptr_(nullptr),
    last_T_GtoM_before_idle_ptr_(nullptr),
    last_odom_ptr_(nullptr) {
    TryLoadMap(config_->map_path);
}

InterfaceManager::~InterfaceManager() {
    if (!config_->online && config_->mapping) {
        CreateMaskTable();
    }
    if (IsMaskDatabaseOpen()) {
        sqlite3_close(mask_database_);
    }
}

void InterfaceManager::ResetMovedPosePtr(const aslam::Transformation& pose) {
    std::unique_lock<std::mutex> lock(move_pose_mutex_);
    moved_pose_ptr_.reset(new aslam::Transformation(pose));
}

void InterfaceManager::ResetMovedPosePtr() {
    std::unique_lock<std::mutex> lock(move_pose_mutex_);
    moved_pose_ptr_.reset(nullptr);
}

bool InterfaceManager::IsMovedPosePtrNull() {
    std::unique_lock<std::mutex> lock(move_pose_mutex_);
    return moved_pose_ptr_ == nullptr;
}

aslam::Transformation* InterfaceManager::GetMovedPosePtr() {
    std::unique_lock<std::mutex> lock(move_pose_mutex_);
    return moved_pose_ptr_.get();
}

void InterfaceManager::ResetDockerPosePtr(const aslam::Transformation& pose) {
    std::unique_lock<std::mutex> lock(docker_pose_mutex_);
    docker_pose_ptr_.reset(new aslam::Transformation(pose));
}

void InterfaceManager::ResetDockerPosePtr() {
    std::unique_lock<std::mutex> lock(docker_pose_mutex_);
    docker_pose_ptr_.reset(nullptr);
}

bool InterfaceManager::IsDockerPosePtrNull() {
    std::unique_lock<std::mutex> lock(docker_pose_mutex_);
    return docker_pose_ptr_ == nullptr;
}

aslam::Transformation* InterfaceManager::GetDockerPosePtr() {
    std::unique_lock<std::mutex> lock(docker_pose_mutex_);
    return docker_pose_ptr_.get();
}

void InterfaceManager::AddOdom(const common::OdomData& odom_meas) {
    std::unique_lock<std::mutex> lock(odom_mutex_);
    last_odom_ptr_.reset(new common::OdomData(odom_meas));
}


bool InterfaceManager::GetLastOdom(common::OdomData& odom) {
    std::unique_lock<std::mutex> lock(odom_mutex_);
    if (last_odom_ptr_ == nullptr) {
        return false;
    }
    odom = *(last_odom_ptr_.get());
    return  true;
}

void InterfaceManager::SetLastOdomPtr() {
    std::unique_lock<std::mutex> lock(odom_mutex_);
    last_odom_ptr_.reset(nullptr);
}

common::KeyFrame* InterfaceManager::GetLastKeyFrameBeforeIdlePtr() {
    std::unique_lock<std::mutex> lock(last_key_frame_mutex_);
    return last_keyframe_before_idle_ptr_.get();
}

void InterfaceManager::SetLastKeyFrameBeforeIdlePtr(const common::KeyFrame& last_keyframe) {
    std::unique_lock<std::mutex> lock(last_key_frame_mutex_);
    last_keyframe_before_idle_ptr_.reset(new common::KeyFrame(last_keyframe));
}

void InterfaceManager::SetLastKeyFrameBeforeIdlePtr() {
    std::unique_lock<std::mutex> lock(last_key_frame_mutex_);
    last_keyframe_before_idle_ptr_.reset(nullptr);
}

aslam::Transformation* InterfaceManager::GetLastTGtoMBeforeIdlePtr() {
    std::unique_lock<std::mutex> lock(lase_T_GtoM_before_idle_mutex_);
    return last_T_GtoM_before_idle_ptr_.get();
}

void InterfaceManager::SetLastTGtoMBeforeIdlePtr(const aslam::Transformation& T_GtoM) {
    std::unique_lock<std::mutex> lock(lase_T_GtoM_before_idle_mutex_);
    last_T_GtoM_before_idle_ptr_.reset(new aslam::Transformation(T_GtoM));
}

void InterfaceManager::SetLastTGtoMBeforeIdlePtr() {
    std::unique_lock<std::mutex> lock(lase_T_GtoM_before_idle_mutex_);
    last_T_GtoM_before_idle_ptr_.reset(nullptr);
}

void InterfaceManager::MapThresholding(cv::Mat* map_ptr, Eigen::Vector2d* origin_ptr) {
    cv::Mat& map = *CHECK_NOTNULL(map_ptr);
    const double occ_threshold = 0.65;
    const double free_threshold = 0.196;
    const int th_h = 255 * occ_threshold;
    const int th_l = 255 * free_threshold;
    for (int i = 0; i < map.rows; ++i) {
        for (int j = 0; j < map.cols; ++j) {
            const int ip = 255 - map.at<uchar>(i,j);
            if (ip > th_h) {
                map.at<uchar>(i,j) = 32; // Occupied.
            } else if (ip < th_l) {
                map.at<uchar>(i,j) = 0; // Free
            } else {
                map.at<uchar>(i,j) = 16; // Unknown.
            }
        }
    }

    if (!mask_lines_.empty() || !mask_polygons_.empty()) {
        DrawMaskOnMap(map_ptr, origin_ptr);
    }
}

void InterfaceManager::DrawMaskOnMap(cv::Mat* map_ptr, Eigen::Vector2d* origin_ptr) {
    const Eigen::Vector2d& origin = *CHECK_NOTNULL(origin_ptr);
    cv::Mat& map = *CHECK_NOTNULL(map_ptr);
    // Draw mask map
    cv::Mat mask_map(map.rows, map.cols, CV_8UC1, cv::Scalar(255));
    if (!mask_lines_.empty()) {
        for (const auto& line : mask_lines_) {
            cv::Point2f world_point_start(line.second(0), line.second(1));
            cv::Point2f world_point_end(line.second(2), line.second(3));
            cv::Point2i map_point_start = ChangePointWorldToMap(world_point_start, origin, map.rows, map.cols);
            cv::Point2i map_point_end = ChangePointWorldToMap(world_point_end, origin, map.rows, map.cols);
            cv::line(mask_map, map_point_start, map_point_end, cv::Scalar(0));
        }
    }

    if (!mask_polygons_.empty()) {
        std::vector<std::vector<cv::Point2i>> contours;
        for (const auto& polygon : mask_polygons_) {
            std::vector<cv::Point2i> contour;
            for (int i = 0; i < polygon.second.cols(); ++i) {
                cv::Point2f polygon_point_world;
                polygon_point_world.x = polygon.second(0,i);
                polygon_point_world.y = polygon.second(1,i);
                cv::Point2i polygon_point_map;
                polygon_point_map = ChangePointWorldToMap(polygon_point_world, origin, map.rows, map.cols);
                contour.push_back(polygon_point_map);
            }
            contours.push_back(contour);
        }
        cv::polylines(mask_map, contours, true, cv::Scalar(0));
        cv::fillPoly(mask_map, contours, cv::Scalar(0));
    }
    
    // Draw mask on map
    for (int i = 0; i < mask_map.rows; i++) {
        for (int j = 0; j < mask_map.cols; j++) {
            // Occupancied
            if (map.at<uchar>(i,j) == 32) {
                if (mask_map.at<uchar>(i,j) == 0) {
                    map.at<uchar>(i,j) = static_cast<uchar>(96);
                }
            }
            // Unknow
            else if (map.at<uchar>(i,j) == 16) {
                if (mask_map.at<uchar>(i,j) == 0) {
                    map.at<uchar>(i,j) = static_cast<uchar>(80);
                }
            }
            // Passable
            else if (map.at<uchar>(i, j) == 0) {
                if (mask_map.at<uchar>(i, j) == 0) {
                    map.at<uchar>(i, j) = static_cast<uchar>(64);
                }
            }
        }
    }
}

void InterfaceManager::TryLoadMap(const std::string& map_path) {
    // Load occupancy grid map.
    const std::string complete_map_yaml_name =
        common::ConcatenateFilePathFrom(
            common::getRealPath(config_->map_path), common::kMapYamlFileName);
    const std::string complete_map_name =
        common::ConcatenateFilePathFrom(
            common::getRealPath(config_->map_path), common::kMapFileName);
    const std::string complete_mask_file_name = 
        common::ConcatenateFilePathFrom(config_->map_path, common::kMaskFileName);
    if (common::fileExists(complete_map_name) &&
            common::fileExists(complete_map_yaml_name) &&
            common::fileExists(complete_mask_file_name)) {
        LOG(INFO) << "Occupancy map exist, try load map.";
        YAML::Node node = YAML::LoadFile(complete_map_yaml_name.c_str());
        LOG(INFO) << "Loaded the map YAML file "
                    << complete_map_yaml_name;
        double resolution;
        SetValueBasedOnYamlKey(node,
                                "resolution",
                                &resolution);

        Eigen::Matrix<double, 1, 3> origin;
        SetValueBasedOnYamlKey(node,
                                "origin",
                                &origin);

        Eigen::Matrix<double, 1, 3> docker_pose_v;
        SetValueBasedOnYamlKey(node,
                                "docker_pose",
                                &docker_pose_v);
        aslam::Transformation docker_pose_T(Eigen::Quaterniond::Identity(), docker_pose_v);
        docker_pose_ptr_.reset(new aslam::Transformation(docker_pose_T));
        
        Eigen::Vector3d p_origin(origin(0), origin(1), 0.);
        Eigen::Quaterniond q_origin = common::EulerToQuat(Eigen::Vector3d(0., 0., origin(2)));
        const aslam::Transformation T_origin(p_origin, q_origin);

        scan_maps_ptr_.reset(new common::LiveSubmaps(-1, config_->resolution));
        cv::Mat raw_map = cv::imread(complete_map_name, 0);
        cv::transpose(raw_map, raw_map);
        scan_maps_ptr_->LoadMap(raw_map, T_origin);

        const int open_failed = sqlite3_open(complete_mask_file_name.c_str(), &mask_database_);
        if (open_failed) {
            LOG(FATAL) << "ERROR: Cannot open mask database: "
                       << sqlite3_errmsg(mask_database_);
        }
        LoadMaskTable();
        LOG(INFO) << "Occupancy map loaded.";
    }

    // Load octomap.
    const std::string complete_octo_map_name =
    common::ConcatenateFilePathFrom(
        common::getRealPath(map_path), common::kOctoMapFileName);
    if (common::fileExists(complete_octo_map_name)) {
        octomap::AbstractOcTree* tree = 
            octomap::AbstractOcTree::read(complete_octo_map_name);
        octree_map_ptr_.reset(dynamic_cast<octomap::ColorOcTree*>(tree));
        CHECK_EQ(octree_map_ptr_->getResolution(), config_->resolution);
        LOG(INFO) << "Octo map loaded.";
    }

    // Load visual summary map.
    const std::string complete_visual_map_name =
        common::ConcatenateFilePathFrom(
            common::getRealPath(config_->map_path), common::kVisualMapFileName);
    if (common::fileExists(complete_visual_map_name)) {
        LOG(INFO) << "Visual map existed, load map.";
        summary_map_ptr_.reset(new loop_closure::SummaryMap());
        loop_closure::MapLoader map_loader(complete_visual_map_name);
        map_loader.LoadMap(summary_map_ptr_.get());
        LOG(INFO) << "Visual map loaded.";
    }
}

cv::Point2i InterfaceManager::ChangePointWorldToMap(const cv::Point2f& wp, const Eigen::Vector2d& origin, int map_rows, int map_cols) {
    double x = origin(0);
    double y = origin(1);
    double resolution = config_->resolution;
    
    cv::Point2i map_point;
    // transpose and flip
    map_point.x = map_cols - (wp.y - y) / resolution;
    map_point.y = map_rows - (wp.x - x) / resolution;
    return map_point;
}

cv::Point2f InterfaceManager::ChangePointMapToWorld(const cv::Point2i& mp, const Eigen::Vector2d& origin, int map_rows, int map_cols) {
    double x = origin(0);
    double y = origin(1);
    double resolution = config_->resolution;
    
    cv::Point2f world_point;
    // transpose and flip
    world_point.x = (map_rows - mp.y) * resolution + x;
    world_point.y = (map_cols - mp.x) * resolution + y;
    return world_point;
}

}