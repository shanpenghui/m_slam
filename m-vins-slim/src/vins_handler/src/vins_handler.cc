#include "vins_handler/vins_handler.h"

#include <syscall.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

#include "feature_tracker/depth_estimator.h"
#include "feature_tracker/gyro_tracker.h"
#include "feature_tracker/reprojection_checker.h"
#include "flann_common/nano_flann.hpp"
#include "flann_common/pointcloud.h"
#include "time_common/time.h"
#include "time_common/time_table.h"
#include "math_common/math.h"
#include "occ_common/fast_correlative_scan_matcher.h"
#include "occ_common/realtime_correlative_scan_matcher.h"
#include "occ_common/voxel_filter.h"
#include "parallel_common/parallel_process.h"
#include "vins_handler/map_tool.h"

namespace vins_handler {

constexpr bool kSyncOnceTime = false;

constexpr int kMaxLoopCandidateSize = 3;

constexpr int kMaxBadFusionTime = 6;

constexpr double kMaxTimeDiffBetweenScanAndImage = 0.1; // second
constexpr double kMaxTimeDiffBetweenImageAndDepth = 0.05; // second
constexpr double kMaxWaittingTime = 100.0;  // milliseconds

constexpr double kMinTransMotion = 0.001; // meter
constexpr double kMinRotMotion = 0.01; // degree

constexpr bool kShowTrackingSubmap = false;
constexpr bool kShowMatchingResult = false;
constexpr bool kShowScanLoopResult = false;

void SortObservations(common::ObservationDeq& observations) {
    std::sort(observations.begin(), observations.end(),
              [](const common::Observation& a, const common::Observation& b) {
                    return a.keyframe_id < b.keyframe_id;});
}

VinsHandler::VinsHandler(const common::SlamConfigPtr& slam_config,
                         const std::shared_ptr<common::LiveSubmaps>& scan_maps_ptr,
                         const std::shared_ptr<loop_closure::SummaryMap>& summary_map_ptr,
                         const std::shared_ptr<octomap::ColorOcTree>& octree_map_ptr)
    : config_(slam_config),
      reloc_(false),
      call_reloc_init_(false),
      call_posegraph_(false) {
    main_sensor_type_ = common::StringToKeyframeType(config_->main_sensor_type);
    CHECK(main_sensor_type_ != common::KeyFrameType::InValid);
    tuning_mode_ = common::StringToTuningMode(config_->tuning_mode);
    LoadCalibParams(slam_config->calib_yaml_config);
    cameras_ = aslam::NCamera::loadFromYaml(slam_config->calib_yaml_config);
    T_GtoM_ = aslam::Transformation();
    visual_map_loaded_ = false;
    have_new_pose_ = false;
    have_new_loop_ = false;
    offline_posegraph_completed_ = false;
    mapping_completed_ = false;
    sys_inited_ = false;
    data_finish_ = false;
    odom_wait_time_ns_ = 0u;
    imu_wait_time_ns_ = 0u;
    continuous_sleep_time_ms_ = 0;
    track_id_provider_ = 0;
    keyframe_id_provider_ = 0;
    new_online_loop_counter_ = 0;
    docker_pose_func_ptr_ = nullptr;

    if (config_->mapping) {
        has_docker_pose_ = false;
    } else {
        has_docker_pose_ = true;
    }
    vins_core::ImuPropagator::NoiseManager noise_manager(config_);
    imu_propagator_ptr_.reset(
                new vins_core::ImuPropagator(noise_manager));

    odom_propagator_ptr_.reset(new vins_core::OdomPropagator);

    motion_checker_ptr_.reset(new vins_core::MotionChecker(config_,
                                                           kMinTransMotion,
                                                           kMinRotMotion));

    if (main_sensor_type_ != common::KeyFrameType::Scan) {
        feature_tracker_ptr_.reset(
                    new vins_core::GyroTracker(cameras_, config_));

        vins_core::DepthEstimatorOptions depth_estimator_options;
        depth_estimator_options.min_dist = common::kMinRange;
        depth_estimator_options.max_dist = common::kMaxRangeVisual;
        depth_estimator_options.outlier_rejection_threshold =
            config_->visual_sigma_pixel * config_->outlier_rejection_scale;
        depth_estimator_ptr_.reset(
                    new vins_core::DepthEstimator(depth_estimator_options));

        visual_loop_interface_ptr_.reset(
                    new loop_closure::VisualLoopInterface(cameras_, config_));
    }

    hybrid_optimizer_ptr_.reset(
                new vins_core::HybridOptimizer(config_));

    scan_loop_interface_ptr_.reset(
                new loop_closure::ScanLoopInterface());

    slam_state_monitor_ptr_.reset(
                new vins_handler::SlamStateMonitor(kMaxBadFusionTime));

    if (config_->mapping && tuning_mode_ == common::TuningMode::Off) {
        live_local_submaps_ptr_.reset(new common::LiveSubmaps(
                                      -1, config_->resolution));
    } else {
        live_local_submaps_ptr_.reset(new common::LiveSubmaps(
            config_->range_size_per_submap, config_->resolution));
        
        octomap_interface_ptr_.reset(
            new octomap::OctomapInterface(config_->resolution));
    }

    if (config_->mapping && config_->do_octo_mapping) {
        full_octo_mapper_ptr_.reset(new octomap::OctoMapper(config_->resolution,
                    5,
                    cameras_,
                    0)); // Full octo mapping not do outlier removing.
    } else if (config_->do_octo_mapping) {
        LOG(FATAL) << "Octo mapping is only support for mapping & offline tuning mode.";
    }

    if (!config_->mapping) {
        if (scan_maps_ptr != nullptr) {
            live_global_submaps_ptr_ = scan_maps_ptr;
            reloc_init_matcher_ptr_.reset(new common::FastCorrelativeScanMatcher(
                *(live_global_submaps_ptr_->submaps().front()->grid()), 7));
            reloc_ = true;
        }
        if (summary_map_ptr != nullptr && visual_loop_interface_ptr_ != nullptr) {
            visual_loop_interface_ptr_->LoadSummaryMap(summary_map_ptr);
            map_cloud_ = visual_loop_interface_ptr_->GetMapClouds();
            reloc_ = true;
        }
        if (octree_map_ptr != nullptr) {
            octomap_interface_ptr_->SetOcTree(octree_map_ptr);
            // Do nothing.
        }
    }

    first_reloc_init_done_ = reloc_ ? false : true;

    loop_time_interval_s_ = 1.0 / config_->loop_frequency;
    last_loop_time_s_ = 0.0;
}

void VinsHandler::ReleaseJoinableThreads() {
    if (sync_data_thread_ != nullptr && sync_data_thread_->joinable()) {
        sync_data_thread_->join();
    }
    if (frontend_thread_ != nullptr && frontend_thread_->joinable()) {
        frontend_thread_->join();
    }
    if (backend_thread_ != nullptr && backend_thread_->joinable()) {
        backend_thread_->join();
    }
    if (loop_thread_ != nullptr && loop_thread_->joinable()) {
        loop_thread_->join();
    }
    if (posegraph_thread_ != nullptr && posegraph_thread_->joinable()) {
        posegraph_thread_->join();
    }
    if (reloc_init_thread_ != nullptr && reloc_init_thread_->joinable()) {
        reloc_init_thread_->join();
    }
    if (voc_training_thread_ != nullptr && voc_training_thread_->joinable()) {
        voc_training_thread_->join();
    }
    if (feature_tracking_testing_thread_ != nullptr && feature_tracking_testing_thread_->joinable()) {
        feature_tracking_testing_thread_->join();
    }
}

void VinsHandler::LoadCalibParams(const std::string& path) {
    constexpr char kYamlFieldNameScanToOdomerty[] = "T_B_S";

    YAML::Node calib_node;
    try {
        calib_node = YAML::LoadFile(path.c_str());
        cameras_ = aslam::NCamera::deserializeFromYaml(calib_node);
        LOG(INFO) << "Loaded the Calib YAML file " << path;
    } catch (const std::exception& ex) {
        LOG(FATAL) << "Failed to open and parse the calib YAML file "
                   << path << " with the error: " << ex.what();
    }

    Eigen::Matrix4d T_StoO_mat;
    if (!(calib_node[static_cast<std::string>(kYamlFieldNameScanToOdomerty)]
          && YAML::safeGet(calib_node,
                           static_cast<std::string>(kYamlFieldNameScanToOdomerty),
                           &T_StoO_mat))) {
        LOG(FATAL) << "Unable to find the " << kYamlFieldNameScanToOdomerty;
    }

    T_StoO_ = aslam::Transformation(T_StoO_mat);
}

bool VinsHandler::SaveOccupancyMap() {
    if (!(live_local_submaps_ptr_->submaps().front()->insertion_finished())) {
        std::unique_lock<std::mutex> lock(matching_grid_mutex_);
        live_local_submaps_ptr_->submaps().front()->Finish();
    }

    loop_closure::ScanLoopInterface occ_map_collector;

    cv::Mat raw_map;
    common::MapLimits raw_map_limits;
    Eigen::Vector2d raw_origin;
    occ_map_collector.CollectMap(live_local_submaps_ptr_->submaps(),
                                 common::KeyFrames(),
                                 &raw_map,
                                 &raw_map_limits,
                                 &raw_origin);

    cv::Mat map_show = CreateShowMapMat(raw_map, 
                                        config_->do_obstacle_removal,
                                        config_->contour_length);

    if (map_show.rows < 5 && map_show.cols < 5) {
        return false;
    }

    std::vector<cv::Vec4i> lines;
    std::vector<int> inlier_indices;
    MapPoseRectangulate(map_show, &key_frames_ba_, &lines, &inlier_indices);
    
    cv::Mat lines_show;
    cvtColor(map_show, lines_show, cv::COLOR_GRAY2BGR);
    for (size_t i = 0u; i < lines.size(); ++i) {
        const cv::Vec4i& line = lines[i];
        cv::Point p1(line[0], line[1]);
        cv::Point p2(line[2], line[3]);
        cv::Scalar color;
        if (inlier_indices[i] == 1) {
            color = cv::Scalar(255, 0, 0);
        } else if (inlier_indices[i] == 2) {
            color = cv::Scalar(0, 255, 0);
        } else {
            color = cv::Scalar(0, 0, 255);
        }
        cv::line(lines_show, p1, p2, color, 2);
    }
    cv::imwrite(common::ConcatenateFilePathFrom(config_->map_path, "lines.jpg"),
                lines_show);

    ReCastrayAllRangeData();

    cv::Mat map_32f;
    common::MapLimits map_limits;
    Eigen::Vector2d origin;
    occ_map_collector.CollectMap(live_local_submaps_ptr_->submaps(),
                                 common::KeyFrames(),
                                 &map_32f,
                                 &map_limits,
                                 &origin);

    cv::Mat map_8u = cv::Mat(map_32f.rows, map_32f.cols, CV_8UC1);
    for (int i = 0; i < map_32f.rows; ++i) {
        for (int j = 0; j < map_32f.cols; ++j) {
            const float occ = map_32f.at<float>(i, j);
            if (occ < 0.) {
                map_8u.at<uchar>(i, j) = 128;
            } else {
                map_8u.at<uchar>(i, j) =
                    255 - common::ProbabilityToLogOddsInteger(occ);
            }
        }
    }

    cv::Mat keyframes_show;
    map_8u.copyTo(keyframes_show);
    cv::cvtColor(keyframes_show, keyframes_show, cv::COLOR_GRAY2BGR);
    for (size_t i = 0u; i < key_frames_ba_.size(); ++i) {
        const Eigen::Vector2d cell_homo(key_frames_ba_[i].state.T_OtoG.getPosition()(0),
                                        key_frames_ba_[i].state.T_OtoG.getPosition()(1));
        const Eigen::Array2i cell_index = map_limits.GetCellIndex(cell_homo);
        const cv::Scalar color = (i == 0u) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        const int circle_size = (i == 0u) ? 4 : 2;
        cv::circle(keyframes_show, cv::Point2i(cell_index(0), cell_index(1)),
                   circle_size, color, cv::FILLED);
    }
    cv::imwrite(common::ConcatenateFilePathFrom(config_->map_path, "keyframes.jpg"),
               keyframes_show);

    FILE* yaml = fopen(common::ConcatenateFilePathFrom(config_->map_path,
                                                       common::kMapYamlFileName).c_str(),
                       "w");
    fprintf(yaml, "image: %s\nresolution: %f\norigin:\n  cols: 3\n  rows: 1\n  data: [%f, %f, %f]\n",
            "map.pgm",
            config_->resolution,
            origin(0),
            origin(1),
            0.);

    // docker pose call back
    if (docker_pose_func_ptr_ != nullptr && !key_frames_ba_.empty() && !has_docker_pose_) {
        const aslam::Transformation& docker_pose = key_frames_ba_[0].state.T_OtoG;
        docker_pose_func_ptr_(docker_pose);
        has_docker_pose_ = true;
        
        const double docker_yaw = common::QuatToEuler(docker_pose.getEigenQuaternion())(2);
        LOG(INFO) << "DockerPoseCallBack: get docker pose: " << docker_pose.getPosition().x() << ", "
                    << docker_pose.getPosition().x() << ", " << docker_yaw;
        
        fprintf(yaml, "docker_pose:\n  cols: 3\n  rows: 1\n  data: [%f, %f, %f]\n",
            docker_pose.getPosition().x(),
            docker_pose.getPosition().x(),
            docker_yaw);
    }
    fclose(yaml);

    cv::imwrite(common::ConcatenateFilePathFrom(config_->map_path,
                                                common::kMapFileName),
                map_8u);

    map_show = CreateShowMapMat(map_32f, 
                                config_->do_obstacle_removal,
                                config_->contour_length);

    cv::imwrite(common::ConcatenateFilePathFrom(config_->map_path,
                                                "map_show.jpg"),
            map_show);

    return true;

}

bool VinsHandler::TryInitImuState(const common::OdomDatas& odom_datas,
                                  const common::ImuDatas& imu_datas,
                                  common::State* state_ptr) {
    common::State& state = *CHECK_NOTNULL(state_ptr);
    bool success = false;
    if (odom_datas.empty() &&
        imu_propagator_ptr_->StaticInitialize(imu_datas,
                                              false,
                                              &state)) {
        hybrid_optimizer_ptr_->SetImuInitSuccess();
        success = true;
    } else {
        if (static_cast<int>(imu_datas_for_init_.size()) < config_->window_size) {
            imu_datas_for_init_.push_back(imu_datas);
            odom_datas_for_init_.push_back(odom_datas);
        } else if (imu_propagator_ptr_->DynamicInitialize(odom_datas_for_init_,
                                                          imu_datas_for_init_,
                                                          &state)) {
            hybrid_optimizer_ptr_->SetImuInitSuccess();
            success = true;
        }
    }
    return success;           
}

void VinsHandler::TryInitImuState(common::KeyFrame* keyframe_ptr) {
    common::KeyFrame& keyframe = *CHECK_NOTNULL(keyframe_ptr);
    TryInitImuState(keyframe.sensor_meas.odom_datas,
                    keyframe.sensor_meas.imu_datas,
                    &keyframe.state);
}

void VinsHandler::ProcessDepthCloud(const common::ImageData& img_data,
                                    octomap::OctoMapper* octo_mapper_ptr,
                                    std::deque<common::PointCloudWithTimeStamp>* depth_cloud_buffer_ptr) {
    CHECK_NOTNULL(octo_mapper_ptr);
    octomap::PointCloudXYZRGB point_cloud_filted;
    octo_mapper_ptr->GetPointCloud(img_data, &point_cloud_filted);
    if (point_cloud_filted.xyz.size() == 0u) {
        return;
    }

    // NOTE(chien): We use local point cloud base on robot body frame.
    const aslam::Transformation T_CtoO = cameras_->get_T_BtoC(0).inverse();
    const octomath::Vector3 p_CtoO(T_CtoO.getPosition()(0),
                                   T_CtoO.getPosition()(1),
                                   T_CtoO.getPosition()(2));
    const octomath::Quaternion q_CtoO(T_CtoO.getEigenQuaternion().w(),
                                      T_CtoO.getEigenQuaternion().x(),
                                      T_CtoO.getEigenQuaternion().y(),
                                      T_CtoO.getEigenQuaternion().z());
    const octomath::Pose6D T_CtoO_octo(p_CtoO, q_CtoO);

    point_cloud_filted.xyz.transform(T_CtoO_octo);

    depth_cloud_buffer_ptr->emplace_back(img_data.timestamp_ns,
                                         point_cloud_filted);
}

void VinsHandler::OctoMapping(
        const common::OctoMappingInput& pose_and_depth,
        const bool do_castray,
        octomap::OctoMapper* octo_mapper_ptr,
        common::OctomapKeySetPairs* octomap_keysets_ptr) {
    const aslam::Transformation& T_OtoG = pose_and_depth.first;
    const Eigen::Vector3d& p_OinG_plane = T_OtoG.getPosition();
    const Eigen::Quaterniond& q_OtoG_plane = T_OtoG.getEigenQuaternion();

    octomap::Pointcloud point_cloud = pose_and_depth.second.xyz;
    const Eigen::Vector3f p_OinG = p_OinG_plane.template cast<float>();
    const octomath::Vector3 p_OinG_octo(p_OinG(0), p_OinG(1), p_OinG(2));
    const Eigen::Quaternionf q_OtoG = q_OtoG_plane.template cast<float>();
    const octomath::Quaternion q_OtoG_octo(q_OtoG.w(), q_OtoG.x(), q_OtoG.y(), q_OtoG.z());
    const octomath::Pose6D T_OtoG_octo(p_OinG_octo, q_OtoG_octo);

    // Note(chien): All point cloud must be transform to IMU(odom) frame first.
    point_cloud.transform(T_OtoG_octo);

    std::function<void(const octomap::Pointcloud&, const octomath::Vector3&, const size_t, const size_t,
                       octomap::OctoMapper*, octomap::KeySet*, octomap::KeySet*)>
            CastRayInParallel = [&](const octomap::Pointcloud& point_cloud,
                                   const octomath::Vector3& origin,
                                   const size_t start_idx,
                                   const size_t end_idx,
                                   octomap::OctoMapper* octo_mapper_ptr,
                                   octomap::KeySet* free_cells_ptr,
                                   octomap::KeySet* occupied_cells_ptr) {
        octomap::KeySet& free_cells = *CHECK_NOTNULL(free_cells_ptr);
        octomap::KeySet& occupied_cells = *CHECK_NOTNULL(occupied_cells_ptr);
        for (size_t i = start_idx; i < end_idx; ++i) {
            octomap::OcTreeKey key =
                    octo_mapper_ptr->GetOcTree()->coordToKey(point_cloud[i]);
            if (occupied_cells.find(key) == occupied_cells.end()) {
                if (do_castray) {
                    octo_mapper_ptr->CastRay(origin, point_cloud[i],
                                              &free_cells, &occupied_cells);
                } else {
                    occupied_cells.insert(key);
                }
            }
        }
    };

    constexpr int kNumThreads = 40;
    const size_t block_size = point_cloud.size() / kNumThreads;
    std::vector<std::thread> threads(kNumThreads);
    std::vector<octomap::KeySet> free_cells_tmp(kNumThreads);
    std::vector<octomap::KeySet> occupied_cells_tmp(kNumThreads);
    for (size_t i = 0u; i < kNumThreads; ++i) {
        const size_t start_idx = i * block_size;
        const size_t end_idx = (i + 1u) * block_size < point_cloud.size() ? (i + 1u) * block_size : point_cloud.size();
        threads[i] = std::thread(CastRayInParallel, point_cloud, T_OtoG_octo.trans(), start_idx, end_idx,
                                 octo_mapper_ptr, &free_cells_tmp[i], &occupied_cells_tmp[i]);
    }
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    octomap::KeySet free_cells, occupied_cells;
    for (size_t i = 0u; i < kNumThreads; ++i) {
        free_cells.insert(free_cells_tmp[i].begin(), free_cells_tmp[i].end());
        occupied_cells.insert(occupied_cells_tmp[i].begin(), occupied_cells_tmp[i].end());
    }

    octo_mapper_ptr->UpdateOccupancy(&free_cells, &occupied_cells);
    octo_mapper_ptr->UpdateColor(point_cloud, pose_and_depth.second.rgb);

    if (do_castray) {
        octo_mapper_ptr->UpdateInnerOccupancy();
    }
    if (octomap_keysets_ptr != nullptr) {
        octomap_keysets_ptr->emplace_back(free_cells, occupied_cells);
    }
}

void VinsHandler::OctoMappingMult() {
    std::vector<common::OctoMappingInput> full_octo_mapping_inputs;
    const float total_pose_size = static_cast<float>(key_frames_ba_.size());

    for (size_t i = 0u; i < key_frames_ba_.size(); ++i) {
        std::deque<common::PointCloudWithTimeStamp> point_cloud_buffer_tmp;
        ProcessDepthCloud(key_frames_ba_[i].sensor_meas.img_data,
                           full_octo_mapper_ptr_.get(),
                           &point_cloud_buffer_tmp);

        while (!point_cloud_buffer_tmp.empty()) {
            full_octo_mapping_inputs.emplace_back(key_frames_ba_[i].state.T_OtoG,
                                                   point_cloud_buffer_tmp.front().second);
            point_cloud_buffer_tmp.pop_front();
        }

        const float complete_rate = (static_cast<float>(i) / total_pose_size) * 100.f;
        LOG(INFO) << "Full point cloud processing " << complete_rate << "% completed.";
    }

    // Clear keyframe buffer for release memory.
    key_frames_ba_.clear();
    common::KeyFrames().swap(key_frames_ba_);

    for (size_t i = 0u; i < full_octo_mapping_inputs.size(); ++i) {
        OctoMapping(full_octo_mapping_inputs[i], true, full_octo_mapper_ptr_.get(), nullptr);

        const float complete_rate = (static_cast<float>(i) / total_pose_size) * 100.f;
        LOG(INFO) << "Full octo mapping " << complete_rate << "% completed.";
    }

    full_octo_mapper_ptr_->Prune();
}

void VinsHandler::ReCastrayAllRangeData() {
    CHECK_NOTNULL(live_local_submaps_ptr_);
    live_local_submaps_ptr_.reset(new common::LiveSubmaps(
                                      -1, config_->resolution));

    for (const auto& key_frame : key_frames_ba_) {
        common::PointCloud pc_global;
        pc_global.points.reserve(key_frame.points.points.size());
        for (const auto& pt_local : key_frame.points.points) {
            pc_global.points.push_back(key_frame.state.T_OtoG.transform(pt_local));
        }
        pc_global.miss_points.reserve(key_frame.points.miss_points.size());
        for (const auto& pt_local : key_frame.points.miss_points) {
            pc_global.miss_points.push_back(key_frame.state.T_OtoG.transform(pt_local));
        }
        live_local_submaps_ptr_->InsertRangeData(key_frame, pc_global, T_StoO_, 1/*max_submap_size*/);
    }

    live_local_submaps_ptr_->submaps().front()->Finish();
}

bool VinsHandler::EnoughMotionCheck(const common::State& prev_state,
                                      const common::State& curr_state) {

    constexpr double kDeltaPThreshold = 0.05;
    constexpr double kDeltaQThreshold = 0.01;
    const Eigen::Quaterniond delta_q =
            prev_state.T_OtoG.getEigenQuaternion().conjugate() *
            curr_state.T_OtoG.getEigenQuaternion();
    const Eigen::Vector3d delta_p =
            prev_state.T_OtoG.getEigenQuaternion().conjugate() *
            (curr_state.T_OtoG.getPosition() - prev_state.T_OtoG.getPosition());

    const double sign_q = delta_q.w() > 0. ? 1. : -1.;
    const Eigen::Vector3d delta_q_vec = sign_q * 2.0 * delta_q.vec();

    const double delta_p_norm = delta_p.norm();
    const double delta_q_norm = delta_q_vec.norm();

    if (delta_p_norm < kDeltaPThreshold && delta_q_norm < kDeltaQThreshold) {
        return false;
    } else {
        return true;
    }
}

template <typename DataType>
void VinsHandler::ConcatSensorData(const std::vector<DataType>& meas,
                                    std::vector<DataType>* key_meas) {
    CHECK_NOTNULL(key_meas);
    if (meas.empty()) {
        return;
    }

    if (key_meas->empty()) {
        *key_meas = meas;
    } else {
        // The last data of meas_keyframe is interpolated.
        key_meas->pop_back();

        // The first data of meas is interpolated.
        key_meas->insert(key_meas->end(),
                          meas.begin() + 1,
                          meas.end());
    }
}

template <typename SensorType>
void VinsHandler::GetFrontData(std::deque<SensorType>* sensor_buffer_ptr,
                               std::mutex* mutex_ptr,
                               SensorType* get_data_ptr) {
    std::deque<SensorType>& sensor_buffer = *CHECK_NOTNULL(sensor_buffer_ptr);
    std::mutex& mutex = *CHECK_NOTNULL(mutex_ptr);
    SensorType& get_data = *CHECK_NOTNULL(get_data_ptr);

    mutex.lock();
    get_data = sensor_buffer.front();
    sensor_buffer.pop_front();
    mutex.unlock();
}

template <typename SensorType>
bool VinsHandler::FindSyncCameraData(const uint64_t time_query_ns,
                                     const double time_diff_threshold_s,
                                     std::deque<SensorType>* sensor_buffer_ptr,
                                     SensorType* find_data_ptr) {
    std::deque<SensorType>& sensor_buffer = *CHECK_NOTNULL(sensor_buffer_ptr);
    SensorType& find_data = *CHECK_NOTNULL(find_data_ptr);

    bool find_sync = false;
    bool early_break = false;
    if (sensor_buffer.empty()) {
        return false;
    }

    common::TicToc timer;
    while (!early_break) {
        if ((sensor_buffer.front().timestamp_ns > time_query_ns) &&
                (std::abs(common::NanoSecondsToSeconds(static_cast<int64_t>(time_query_ns)) -
                          common::NanoSecondsToSeconds(static_cast<int64_t>(sensor_buffer.front().timestamp_ns))) >=
                 time_diff_threshold_s)) {
            find_sync = false;
            break;
        }

        for (size_t i = 0u; i < sensor_buffer.size(); ++i) {
            if (std::abs(common::NanoSecondsToSeconds(static_cast<int64_t>(time_query_ns)) -
                         common::NanoSecondsToSeconds(static_cast<int64_t>(sensor_buffer.at(i).timestamp_ns))) <
                    time_diff_threshold_s) {
                find_data = sensor_buffer[i];
                if (i != 0u) {
                    sensor_buffer.erase(sensor_buffer.begin(), sensor_buffer.begin() + i - 1u);
                }            
                early_break = true;
                find_sync = true;
                break;
            } else if ((sensor_buffer[i].timestamp_ns > time_query_ns) &&
                       (std::abs(common::NanoSecondsToSeconds(static_cast<int64_t>(time_query_ns)) -
                                 common::NanoSecondsToSeconds(static_cast<int64_t>(sensor_buffer.at(i).timestamp_ns))) >=
                        time_diff_threshold_s)) {
                early_break = true;
                find_sync = false;
                break;
            }
        }
        
        if (kSyncOnceTime) {
            early_break = true;
        } else if (timer.toc() > kMaxWaittingTime) {
            early_break = true;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    return find_sync;
}

void VinsHandler::SyncSensorData() {
    const bool buffer_empty = main_sensor_type_ == common::KeyFrameType::Visual ?
        img_buffer_.empty() : scan_buffer_.empty();
    if (buffer_empty) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continuous_sleep_time_ms_ += 10;
        if (continuous_sleep_time_ms_ > 1000) {
            LOG(WARNING) << "No scan or image measurements in for more than 1 second.";
            continuous_sleep_time_ms_ = 0;
        }
        return;
    }

    continuous_sleep_time_ms_ = 0;
    
    common::SensorDataConstPtr scan_data = nullptr;
    common::ImageData img_data;
    uint64_t main_sensor_timestamp_ns = std::numeric_limits<uint64_t>::max();

    switch (main_sensor_type_) {
        case common::KeyFrameType::ScanAndVisual: {
            if (!scan_buffer_.empty()) {
                const uint64_t front_t_scan_ns = scan_buffer_.front()->timestamp_ns;
                GetFrontData<common::SensorDataConstPtr>(&scan_buffer_,
                                                       &scan_mutex_,
                                                       &scan_data);

                img_mutex_.lock();
                img_buffer_.insert(img_buffer_.end(), img_l2_buffer_.begin(),
                                   img_l2_buffer_.end());
                img_l2_buffer_.clear();
                img_mutex_.unlock();
                const bool find_img = FindSyncCameraData<common::ImageData>(
                                                front_t_scan_ns,
                                                kMaxTimeDiffBetweenScanAndImage,
                                                &img_buffer_,
                                                &img_data);
                if (find_img) {
                    depth_mutex_.lock();
                    depth_buffer_.insert(depth_buffer_.end(), depth_l2_buffer_.begin(),
                                         depth_l2_buffer_.end());
                    depth_l2_buffer_.clear();
                    depth_mutex_.unlock();
                    common::DepthData depth_data_assosiated;
                    const bool find_depth = FindSyncCameraData<common::DepthData>(
                                                    img_data.timestamp_ns,
                                                    kMaxTimeDiffBetweenImageAndDepth,
                                                    &depth_buffer_,
                                                    &depth_data_assosiated);
                    if (find_depth) {
                        img_data.depth = depth_data_assosiated.depth;
                    }
                }
                main_sensor_timestamp_ns = scan_data->timestamp_ns;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            break;
        }
        case common::KeyFrameType::Scan: {
            if (!scan_buffer_.empty()) {
                GetFrontData<common::SensorDataConstPtr>(&scan_buffer_,
                                                       &scan_mutex_,
                                                       &scan_data);
                main_sensor_timestamp_ns = scan_data->timestamp_ns;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            break;
        }
        case common::KeyFrameType::Visual: {
            if (!img_buffer_.empty()) {
                GetFrontData<common::ImageData>(&img_buffer_,
                                                &img_mutex_,
                                                &img_data);    

                depth_mutex_.lock();
                depth_buffer_.insert(depth_buffer_.end(), depth_l2_buffer_.begin(),
                                     depth_l2_buffer_.end());
                depth_l2_buffer_.clear();
                depth_mutex_.unlock();
                common::DepthData depth_data_assosiated;
                const bool find_depth = FindSyncCameraData<common::DepthData>(
                                                img_data.timestamp_ns,
                                                kMaxTimeDiffBetweenImageAndDepth,
                                                &depth_buffer_,
                                                &depth_data_assosiated);
                if (find_depth) {
                    img_data.depth = depth_data_assosiated.depth;
                }

                main_sensor_timestamp_ns = img_data.timestamp_ns;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            break;
        }
        default: {
            LOG(FATAL) << "Unknow support main sensor type.";
        }
    }
    if (!sys_inited_) {
        if ((!gt_status_.empty() && InitializeStateByGt(main_sensor_timestamp_ns,
                                                        &last_state_)) ||
            gt_status_.empty()) {
            sys_inited_ = true;
        }
        if (sys_inited_) {
            init_success_time_ns_ = main_sensor_timestamp_ns;
            odom_propagator_ptr_->RemovePropagateData(init_success_time_ns_,
                                                      &odom_buffer_);
            imu_propagator_ptr_->RemovePropagateData(init_success_time_ns_,
                                                     &imu_buffer_);
        }
    } else {
        common::OdomDatas odom_datas_temp;
        odom_mutex_.lock();
        bool get_odom_success = odom_propagator_ptr_->GetPropagateData(
                    main_sensor_timestamp_ns,
                    &odom_buffer_,
                    &odom_datas_temp);
        odom_mutex_.unlock();
#if 1
        if (!get_odom_success) {
            if ((config_->online && !data_finish_))
            odom_wait_time_ns_ = main_sensor_timestamp_ns;
            std::unique_lock<std::mutex> lock(odom_mutex_);
            odom_waiter_.wait_for(lock, std::chrono::seconds(1));
            get_odom_success = odom_propagator_ptr_->GetPropagateData(
                        main_sensor_timestamp_ns,
                        &odom_buffer_,
                        &odom_datas_temp);
        }
#endif
        ConcatSensorData(odom_datas_temp, &odom_datas_assosiated_);

        common::ImuDatas imu_datas_temp;
        imu_mutex_.lock();
        bool get_imu_success = imu_propagator_ptr_->GetPropagateData(
                    main_sensor_timestamp_ns,
                    &imu_buffer_,
                    &imu_datas_temp);
        imu_mutex_.unlock();
#if 1
        if (!get_imu_success) {
            imu_wait_time_ns_ = main_sensor_timestamp_ns;
            std::unique_lock<std::mutex> lock(imu_mutex_);
            imu_waiter_.wait_for(lock, std::chrono::seconds(1));
            get_imu_success = imu_propagator_ptr_->GetPropagateData(
                        main_sensor_timestamp_ns,
                        &imu_buffer_,
                        &imu_datas_temp);
        }
#endif
        ConcatSensorData(imu_datas_temp, &imu_datas_assosiated_);

        if (!get_odom_success || (config_->use_imu && !get_imu_success)) {
            return;
        }

        if (!motion_checker_ptr_->CheckIsMotionEnough(odom_datas_assosiated_)) {
            VLOG(5) << "Odom motion not enough, skip this mesurement!";
            return;
        }
#if 0
        if (config_->use_imu &&
            !motion_checker_ptr_->CheckIsMotionEnough(imu_datas_assosiated_)) {
            VLOG(5) << "IMU motion not enough, skip this mesurement!";
            return;
        }
#endif
        std::unique_lock<std::mutex> lock(hybrid_buffer_mutex_);
        hybrid_buffer_.emplace_back(main_sensor_timestamp_ns,
                                    imu_datas_assosiated_,
                                    odom_datas_assosiated_,
                                    img_data,
                                    scan_data);
        CHECK_EQ(odom_datas_assosiated_.back().timestamp_ns, main_sensor_timestamp_ns);
        if (config_->use_imu) {
            CHECK_EQ(imu_datas_assosiated_.back().timestamp_ns, main_sensor_timestamp_ns);
        }
        imu_datas_assosiated_.clear();
        odom_datas_assosiated_.clear();
    }
}

void VinsHandler::CollectImageDataOnly() {
    if (img_buffer_.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continuous_sleep_time_ms_ += 10;
        if (continuous_sleep_time_ms_ > 1000) {
            LOG(WARNING) << "No image measurements come in for more than 1 second.";
            continuous_sleep_time_ms_ = 0;
        }
        return;
    }

    while (!img_buffer_.empty()) {
        img_mutex_.lock();
        common::ImageData img_data = img_buffer_.front();
        img_buffer_.pop_front();
        img_mutex_.unlock();
        hybrid_buffer_.emplace_back(img_data.timestamp_ns,
                                    common::ImuDatas(),
                                    common::OdomDatas(),
                                    img_data,
                                    nullptr);
    }
}

bool VinsHandler::InitializeStateByGt(const uint64_t timestamp_ns,
                                      common::State* state_ptr) {
    common::State& state = *CHECK_NOTNULL(state_ptr);
    
    common::OdomData tmp;
    if (common::GetGtStateByTimeNs(timestamp_ns, gt_status_, &tmp)) {
        state.T_OtoG = aslam::Transformation(tmp.q, tmp.p);
#if 0
        LOG(INFO) << "State initialize success! state: ";
        state.Print(0, 0.0, "INIT");
#else
        LOG(INFO) << "State initialize success!";
#endif
        return true;
    }
    return false;
}

double VinsHandler::TrackFeature(const common::SyncedHybridSensorData& hybrid_data,
                                 common::KeyFrame* key_frame_k_ptr,
                                 common::KeyFrame* key_frame_kp1_ptr,
                                 std::vector<common::FrameToFrameMatchesWithScore>* matches_vec_ptr) {
    common::KeyFrame& key_frame_kp1 =
            *CHECK_NOTNULL(key_frame_kp1_ptr);
    std::vector<common::FrameToFrameMatchesWithScore>& matches_vec =
            *CHECK_NOTNULL(matches_vec_ptr);
    common::VisualFrameDataPtrVec visual_frame_datas;
    for (size_t cam_idx = 0u; cam_idx < hybrid_data.img_data.images.size(); ++cam_idx) {
        common::VisualFrameData visual_frame_data_kp1;
        TIME_TIC(FEATURE_DETECTION_EXTRACTION);
#ifdef USE_CNN_FEATURE
        feature_tracker_ptr_->InferFeature(hybrid_data.img_data.timestamp_ns,
                                           hybrid_data.img_data.images[cam_idx],
                                           &visual_frame_data_kp1,
                                           &track_id_provider_);
#else
        feature_tracker_ptr_->DetectAndExtractFeature(hybrid_data.img_data.timestamp_ns,
                                                      hybrid_data.img_data.images[cam_idx],
                                                      &visual_frame_data_kp1,
                                                      &track_id_provider_);
#endif
        TIME_TOC(FEATURE_DETECTION_EXTRACTION);

        TIME_TIC(FEATURE_MATCHING);
        if (key_frame_k_ptr != nullptr) {
            common::KeyFrame& key_frame_k = *CHECK_NOTNULL(key_frame_k_ptr);
            const Eigen::Quaterniond q_CtoO = cameras_->get_T_BtoC(cam_idx).getEigenQuaternion().conjugate();
            const Eigen::Quaterniond q_kp1 = key_frame_kp1.state.T_OtoG.getEigenQuaternion() * q_CtoO;
            const Eigen::Quaterniond q_k = key_frame_k.state.T_OtoG.getEigenQuaternion() * q_CtoO;
            const Eigen::Quaterniond q_kp1_k = q_kp1.conjugate() * q_k;
            common::FrameToFrameMatchesWithScore matches;
            feature_tracker_ptr_->TrackFeature(static_cast<int>(cam_idx),
                                              q_kp1_k,
                                              *(key_frame_k.visual_datas.at(cam_idx)),
                                              &visual_frame_data_kp1,
                                              &matches,
                                              &track_id_provider_);
            // Initialize track lengths, bearings, and look up table.
            visual_frame_data_kp1.InitializeTrackLengths(
                        visual_frame_data_kp1.track_ids.rows());

            feature_tracker_ptr_->RansacFiltering(
                        cameras_->getCamera(cam_idx),
                        q_kp1_k.toRotationMatrix(),
                        key_frame_k.visual_datas.at(cam_idx)->key_points,
                        visual_frame_data_kp1.key_points,
                        &matches);

            feature_tracker_ptr_->LengthFiltering(
                        static_cast<int>(cam_idx),
                        *(key_frame_k.visual_datas.at(cam_idx)),
                        visual_frame_data_kp1,
                        &matches);

            // Update track id and track length.
            if (!matches.empty()) {
                std::unordered_map<int, int> matched_counter;
                for (const auto& match : matches) {
                    const int kp1_idx = match.GetKeypointIndexAppleFrame();
                    if (matched_counter.find(kp1_idx) == matched_counter.end()) {
                        matched_counter[kp1_idx] = 1;
                    } else {
                        matched_counter[kp1_idx] += 1;
                    }
                }
                for (const auto iter : matched_counter) {
                    CHECK_GE(iter.second, 1);
                }

                const common::VisualFrameDataPtr& visual_frame_data_k =
                        key_frame_k.visual_datas.at(cam_idx);
                for (const auto& match : matches) {
                    const int k_idx = match.GetKeypointIndexBananaFrame();
                    CHECK_GE(k_idx, 0);
                    const int kp1_idx = match.GetKeypointIndexAppleFrame();
                    CHECK_GE(kp1_idx, 0);
                    visual_frame_data_kp1.track_ids(kp1_idx) =
                            visual_frame_data_k->track_ids(k_idx);
                    visual_frame_data_kp1.track_lengths(kp1_idx) =
                            visual_frame_data_k->track_lengths(k_idx) + 1;
                }
            }

            const double match_rate = static_cast<double>(matches.size()) /
                static_cast<double>(visual_frame_data_kp1.key_points.cols()) * 100.0;
            VLOG(1) << "Get " << matches.size()
                    << " matches from Cam: " << cam_idx
                    << " in total " << visual_frame_data_kp1.key_points.cols()
                    << " keypoints, matched rate: " << match_rate << "%.";
            const double matched_rate = static_cast<double>(matches.size()) /
                    static_cast<double>(visual_frame_data_kp1.key_points.cols());
            if (matched_rate < 0.1) {
                LOG(WARNING) << "Feature tracking interrupt!!! total feature: "
                               << visual_frame_data_kp1.key_points.cols()
                               << " in current frame.";
            }

            matches_vec.push_back(matches);
        } else {
            // Initialize track lengths, bearings, and look up table.
            visual_frame_data_kp1.InitializeTrackLengths(
                        visual_frame_data_kp1.track_ids.rows());
        }
        TIME_TOC(FEATURE_MATCHING);

        // Get default depth measurements.
        const Eigen::Matrix6Xd& keypoints = visual_frame_data_kp1.key_points;
        visual_frame_data_kp1.depths.resize(keypoints.cols());
        for (int i = 0; i < keypoints.cols(); ++i) {
            double depth = common::kInValidDepth;
            visual_frame_data_kp1.depths(i) = depth;
        }

        for (int i = 0; i < keypoints.cols(); ++i) {
            const double x = keypoints(X, i);
            const double y = keypoints(Y, i);
            double depth = common::kInValidDepth;
            if (hybrid_data.img_data.depth != nullptr) {
                depth = static_cast<double>(
                    hybrid_data.img_data.depth->at<unsigned short>(
                        std::ceil(y), std::ceil(x))) / 1000.0;
                if (depth < common::kMinRange || depth > common::kMaxRangeVisual) {
                    depth = common::kInValidDepth;
                }
            }
            visual_frame_data_kp1.depths(i) = depth;
        }
        CHECK_EQ(visual_frame_data_kp1.track_ids.rows(),
                   visual_frame_data_kp1.key_points.cols());
        CHECK_EQ(visual_frame_data_kp1.track_ids.rows(),
                   visual_frame_data_kp1.track_lengths.rows());
#ifndef USE_CNN_FEATURE
        CHECK_EQ(visual_frame_data_kp1.track_ids.rows(),
                 visual_frame_data_kp1.descriptors.cols());
        if (config_->mapping || reloc_) {
            // Project Uint8 descriptors into Float32.
            visual_loop_interface_ptr_->ProjectDescriptors(visual_frame_data_kp1.descriptors,
                                                           &visual_frame_data_kp1.projected_descriptors);
            CHECK_EQ(visual_frame_data_kp1.descriptors.cols(),
                     visual_frame_data_kp1.projected_descriptors.cols());
        } else {
            visual_frame_data_kp1.projected_descriptors.resize(0,
                                                               visual_frame_data_kp1.descriptors.cols());
        }
#else
        CHECK_EQ(visual_frame_data_kp1.track_ids.rows(),
                 visual_frame_data_kp1.projected_descriptors.cols());
#endif
        common::VisualFrameDataPtr visual_frame_data_kp1_ptr =
                std::make_shared<common::VisualFrameData>(visual_frame_data_kp1);
        const int vertex_id = key_frame_kp1.keyframe_id;
        visual_frame_data_kp1_ptr->SetFrameIdAndIdx(vertex_id, static_cast<int>(cam_idx));
        visual_frame_data_kp1_ptr->SetLUT();
        visual_frame_datas.push_back(visual_frame_data_kp1_ptr);
    }

    key_frame_kp1.SetVisualFrameDatas(visual_frame_datas);

    double avg_pixel_diff = 0.0;
    if (key_frame_k_ptr) {
        avg_pixel_diff = CheckTrackingDiff(*key_frame_k_ptr, key_frame_kp1);
    }

    return avg_pixel_diff;
}

void VinsHandler::CreateObservations(const common::KeyFrame& key_frame,
                                     common::ObservationDeq* observations_ptr) {
    common::ObservationDeq& observations = *CHECK_NOTNULL(observations_ptr);
    // Update LUT.
    for (size_t i = 0u; i < key_frames_.size(); ++i) {
        for (size_t cam_idx = 0u; cam_idx < key_frames_[i].visual_datas.size(); ++cam_idx) {
            common::VisualFrameDataPtr& this_visual_frame_data =
                    key_frames_[i].visual_datas.at(cam_idx);
            if (this_visual_frame_data->map_track_id_to_idx.empty()) {
                this_visual_frame_data->SetLUT();
            }
        }
    }
    for (size_t cam_idx = 0; cam_idx < key_frame.visual_datas.size(); ++cam_idx) {
        CHECK_EQ(key_frame.visual_datas.at(cam_idx)->track_ids.rows(),
                 key_frame.visual_datas.at(cam_idx)->track_lengths.rows());
        for (int idx = 0; idx < key_frame.visual_datas.at(cam_idx)->track_ids.rows(); ++idx) {
            const int track_id = key_frame.visual_datas.at(cam_idx)->track_ids(idx);
            CHECK_GE(track_id, 0);
            const common::VisualFrameDataPtr& visual_frame_data =
                  key_frame.visual_datas.at(cam_idx);
            const auto iter = visual_frame_data->map_track_id_to_idx.find(track_id);
            CHECK(iter != visual_frame_data->map_track_id_to_idx.end());
            const int tracked_idx = iter->second;
            const Eigen::Vector2d key_point = visual_frame_data->key_points.col(tracked_idx).head<2>();
            if (config_->use_depth && visual_frame_data->depths.rows() > 0) {
                CHECK_EQ(visual_frame_data->depths.rows(), visual_frame_data->key_points.cols());
                const double depth = visual_frame_data->depths(tracked_idx);
                common::Observation obs(key_frame.keyframe_id,
                        cam_idx, //NOTE: only support the same camera tracking in now.
                        track_id,
                        depth,
                        key_point,
                        visual_frame_data->projected_descriptors.col(tracked_idx));
                observations.push_back(obs);
            } else if (!config_->use_depth) {
                common::Observation obs(key_frame.keyframe_id,
                                        cam_idx, //NOTE: only support the same camera tracking in now.
                                        track_id,
                                        key_point,
                                        visual_frame_data->projected_descriptors.col(tracked_idx));
                observations.push_back(obs);
            }
        }
    }
}

void VinsHandler::FeatureTriangulation(common::KeyFrame* key_frame_ptr) {
    common::KeyFrame& key_frame = *CHECK_NOTNULL(key_frame_ptr);
    // LUT.
    std::unordered_map<int, int> track_id_to_idx;
    for (int i = 0; i < static_cast<int>(features_.size()); ++i) {
        track_id_to_idx[features_[i]->track_id] = i;
    }

    std::unordered_map<int, int> keyframe_id_to_idx;
    for (size_t i = 0u; i < key_frames_.size(); ++i) {
        keyframe_id_to_idx[key_frames_[i].keyframe_id] = i;
    }

    common::ObservationDeq observation_sets;
    CreateObservations(key_frame, &observation_sets);

    for (const auto& observation : observation_sets) {
        const int target_track_id = observation.track_id;
        if (track_id_to_idx.find(target_track_id) != track_id_to_idx.end()) {
            common::FeaturePointPtr& feature =
                    features_[track_id_to_idx.at(target_track_id)];
            // Compute feature velocity in pixel.
            const auto& prev_observation = feature->observations.back();
            const int prev_keyframe_id = prev_observation.keyframe_id;
            const int curr_keyframe_id = observation.keyframe_id;
            const uint64_t prev_time_ns = key_frames_[keyframe_id_to_idx.at(prev_keyframe_id)].state.timestamp_ns;
            const uint64_t curr_time_ns = key_frames_[keyframe_id_to_idx.at(curr_keyframe_id)].state.timestamp_ns;
            CHECK_GE(curr_time_ns, prev_time_ns);
            const double dt_s = common::NanoSecondsToSeconds(curr_time_ns - prev_time_ns);
            const Eigen::Vector2d keypoint_diff = observation.key_point - prev_observation.key_point;
            const Eigen::Vector2d velocity = keypoint_diff / dt_s;

            feature->observations.push_back(observation);
            feature->observations.back().SetVelocity(velocity);
            if (config_->mapping) {
                const auto& itor = track_id_to_idx_ba_.find(target_track_id);
                CHECK(itor != track_id_to_idx_ba_.end());
                common::FeaturePointPtr& feature_ba = features_ba_[itor->second];
                feature_ba->observations.push_back(observation);
            }
        } else {
            common::FeaturePoint feature(target_track_id);
            feature.observations.push_back(observation);
            feature.observations.back().SetVelocity(Eigen::Vector2d::Zero());
            features_.push_back(std::make_shared<common::FeaturePoint>(feature));
            if (config_->mapping) {
                const auto& itor = track_id_to_idx_ba_.find(target_track_id);
                if (itor != track_id_to_idx_ba_.end()) {
                    common::FeaturePointPtr& feature_ba = features_ba_[itor->second];
                    feature_ba->observations.push_back(observation);
                } else {
                    features_ba_.push_back(std::make_shared<common::FeaturePoint>(feature));
                    const size_t current_map_size = track_id_to_idx_ba_.size();
                        track_id_to_idx_ba_[feature.track_id] = current_map_size;
                    CHECK_EQ(features_ba_.size(), track_id_to_idx_ba_.size());
                }
            }
        }
    }

    int successful_counter = 0;
    for (common::FeaturePointPtr& feature : features_) {
        if (feature->observations.empty()) {
            continue;
        }
        if (feature->anchor_frame_idx == -1) {
            bool success = false;
            if (config_->use_depth) {
                success = depth_estimator_ptr_->EstimateDepth(cameras_, key_frames_,
                                                              feature->observations,
                                                              feature.get());
            } else {
                success = depth_estimator_ptr_->Triangulation(cameras_, key_frames_,
                                                              feature->observations,
                                                              feature.get());
            }
            if (success) {
                successful_counter++;
                if (config_->mapping) {
                    common::FeaturePointPtr& feature_ba =
                        features_ba_[track_id_to_idx_ba_.at(feature->track_id)];
                    feature_ba->anchor_frame_idx = feature->anchor_frame_idx;
                    feature_ba->inv_depth = feature->inv_depth;
                }
            }
        }
    }
    VLOG(1) << "Triangulation success " << successful_counter
            << " feature in keyframe " << key_frame.keyframe_id;
}

void VinsHandler::FeatureReTriangulation() {
    if (features_ba_.empty()) {
        return;
    }

    VLOG(0) << "Start feature retriangulation. ";

    int successful_counter = 0;
    std::mutex counter_mutex;

    size_t thread_num = 8u;
    size_t step = features_ba_.size() / thread_num;
    std::vector<std::thread> threads;
    for (size_t i = 0u; i < thread_num; ++i) {
        size_t start_idx = i * step;
        size_t end_idx = start_idx + step;
        if (i == thread_num - 1u) {
            end_idx = features_ba_.size() - 1u;
        }
        threads.push_back(std::thread(&VinsHandler::TriangulateSubVec,
                                      this,
                                      start_idx,
                                      end_idx,
                                      &successful_counter,
                                      &counter_mutex));
    }

    for (size_t i = 0u; i < thread_num; ++i) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }

    VLOG(0) << "Re-tirangulation success " << successful_counter
            << " feature before batch optimization.";
}

void VinsHandler::TriangulateSubVec(const size_t start_idx,
                                    const size_t end_idx,
                                    int* successful_counter_ptr,
                                    std::mutex* mutex_ptr) {
    int& successful_counter = *CHECK_NOTNULL(successful_counter_ptr);
    for (size_t i = start_idx; i < end_idx; ++i) {
        if (config_->use_depth) {
            const bool ret = depth_estimator_ptr_->EstimateDepth(cameras_,
                                                key_frames_ba_,
                                                features_ba_[i]->observations,
                                                features_ba_[i].get());
            if (ret) {
                mutex_ptr->lock();
                successful_counter++;
                mutex_ptr->unlock();
            }
        } else {
            const bool ret = depth_estimator_ptr_->Triangulation(cameras_,
                                                        key_frames_ba_,
                                                        features_ba_[i]->observations,
                                                        features_ba_[i].get());
            if (ret) {
                mutex_ptr->lock();
                successful_counter++;
                mutex_ptr->unlock();
            }
        }
    }
}

double VinsHandler::CheckTrackingDiff(const common::KeyFrame& key_frame_k,
                                     const common::KeyFrame& key_frame_kp1) {
    const common::VisualFrameDataPtrVec& last_visual_frame_datas =
            key_frame_kp1.visual_datas;
    const common::VisualFrameDataPtrVec& second_last_visual_frame_datas =
            key_frame_k.visual_datas;
    int adjacent_tracked_counter = 0;
    double pixel_diff_summed = 0.0;
    for (size_t cam_idx = 0u; cam_idx < last_visual_frame_datas.size(); ++cam_idx) {
        const common::VisualFrameDataPtr& curr_visual_frame_data =
                last_visual_frame_datas.at(cam_idx);
        const common::VisualFrameDataPtr& pre_visual_frame_data =
                second_last_visual_frame_datas.at(cam_idx);
        for (int i = 0; i < curr_visual_frame_data->key_points.cols(); ++i) {
            cv::Point2i pt_c(curr_visual_frame_data->key_points(X, i),
                           curr_visual_frame_data->key_points(Y, i));
            const int track_id = curr_visual_frame_data->track_ids(i);
            const int track_length = curr_visual_frame_data->track_lengths(i);
            if (track_length > 1) {
                const auto iter = pre_visual_frame_data->map_track_id_to_idx.find(track_id);
                CHECK(iter != pre_visual_frame_data->map_track_id_to_idx.end());
                const int tracked_idx = iter->second;
                cv::Point2i pt_p(pre_visual_frame_data->key_points(X, tracked_idx),
                               pre_visual_frame_data->key_points(Y, tracked_idx));
                cv::Point2i pixel_diff = pt_c - pt_p;
                pixel_diff_summed += std::sqrt(pixel_diff.x * pixel_diff.x + pixel_diff.y * pixel_diff.y);
                adjacent_tracked_counter++;
            }
        }
    }
    double avg_diff = (adjacent_tracked_counter == 0) ?
                0. : pixel_diff_summed / static_cast<double>(adjacent_tracked_counter);
    return avg_diff;
}

void VinsHandler::OutlierRejectionByReprojectionError(
        const common::KeyFrames& key_frames,
        const std::unordered_map<int, size_t>& keyframe_id_to_idx,
        const common::FeaturePointPtrVec& feature_points) {
    for (const common::FeaturePointPtr& feature_point : feature_points) {
        if (feature_point->using_in_optimization) {
            Eigen::VectorXd dists;
            vins_core::GetVisualReprojectionError(cameras_,
                                                  key_frames,
                                                  keyframe_id_to_idx,
                                                  *feature_point,
                                                  &dists);

            if (dists.maxCoeff() > config_->outlier_rejection_scale * config_->visual_sigma_pixel) {
                feature_point->anchor_frame_idx = -1;
                feature_point->using_in_optimization = false;
            }
        }
    }
}

void VinsHandler::SelectRelocFeatures(
        const common::KeyFrames& key_frames,
        const aslam::Transformation& T_GtoM,
        const bool do_outlier_rejection,
        common::LoopResults* loop_results_ptr) {
    common::LoopResults& loop_results = *CHECK_NOTNULL(loop_results_ptr);
    last_reloc_landmarks_.clear();

    // LUT.
    std::unordered_map<int, size_t> keyframe_id_to_idx;
    for (size_t i = 0; i < key_frames.size(); ++i) {
        keyframe_id_to_idx[key_frames[i].keyframe_id] = i;
    }

    for (auto& loop_result : loop_results) {
        const auto iter = keyframe_id_to_idx.find(loop_result.keyframe_id_query);
        if (iter == keyframe_id_to_idx.end()) {
            continue;
        }
        if (do_outlier_rejection) {
            Eigen::VectorXd dists;
            vins_core::GetMapVisualReprojectionError(cameras_,
                                                    loop_result,
                                                    key_frames[iter->second].state.T_OtoG,
                                                    T_GtoM,
                                                    &dists);

            const Eigen::Matrix3Xd& p_LinMs = loop_result.positions;
            for (int j = 0; j < dists.rows(); ++j) {
                if (dists(j) > config_->outlier_rejection_scale * config_->visual_sigma_pixel) {
                    loop_result.pnp_inliers[j].second = false;
                } else {
                    const Eigen::Vector3d p_LinM = p_LinMs.col(loop_result.pnp_inliers[j].first);
                    loop_result.pnp_inliers[j].second = true;
                    last_reloc_landmarks_.push_back(p_LinM);
                }
            }
        } else {
            const Eigen::Matrix3Xd& p_LinMs = loop_result.positions;
            for (size_t j = 0u; j < loop_result.pnp_inliers.size(); ++j) {
                const Eigen::Vector3d p_LinM = p_LinMs.col(loop_result.pnp_inliers[j].first);
                loop_result.pnp_inliers[j].second = true;
                last_reloc_landmarks_.push_back(p_LinM);
            }
        }
    }
}

void VinsHandler::ComputeVisualReprojectionErrorAndShow(
    const double* reloc_avg_error_pixel_ptr = nullptr) {
    constexpr int kCamIdxShow = 0;

    if (key_frames_.back().visual_datas.empty()) {
        return;
    }

    const int r1 = 255, g1 = 255, b1 = 0;
    const int r2 = 255, g2 = 255, b2 = 255;
    common::CvMatConstPtrVec last_viz_imgs_tmp;

    const int frame_id_show = key_frames_.back().keyframe_id;

    cv::Mat tracking_img;
    if (key_frames_.back().visual_datas.at(kCamIdxShow)->image_ptr->type() == CV_8UC1) {
        cv::cvtColor(*(key_frames_.back().visual_datas.at(kCamIdxShow)->image_ptr),
                     tracking_img, cv::COLOR_GRAY2RGB);
    } else {
        key_frames_.back().visual_datas.at(kCamIdxShow)->image_ptr->copyTo(tracking_img);
    }

    // We only use mono camera for visualization.
    // First. show detection.
    const common::VisualFrameDataPtrVec& last_visual_frame_datas =
            key_frames_.back().visual_datas;
    const common::VisualFrameDataPtr& last_visual_frame_data =
            last_visual_frame_datas.at(kCamIdxShow);
    for (int i = 0; i < last_visual_frame_data->key_points.cols(); ++i) {
        cv::Point2i pt(last_visual_frame_data->key_points(X, i),
                     last_visual_frame_data->key_points(Y, i));
        cv::circle(tracking_img, pt, 3, cv::Scalar(255, 0, 0), cv::FILLED);
    }

    // LUT.
    std::unordered_map<int, size_t> keyframe_id_to_idx;
    for (size_t i = 0; i < key_frames_.size(); ++i) {
        keyframe_id_to_idx[key_frames_[i].keyframe_id] = i;
    }

    // Second. show tracking history and highlight feature that in optimization.
    int total_track_length = 0;
    int tracked_counter = 0;
    int opt_counter = 0;
    double total_error = 0.0;
    for (const common::FeaturePointPtr& feature : features_) {
        if (feature->observations.empty() ||
                feature->observations.back().keyframe_id != frame_id_show) {
            continue;
        } else {
            tracked_counter++;
        }
        const common::ObservationDeq& observations = feature->observations;
        total_track_length += static_cast<int>(observations.size());
        const int start_idx = static_cast<int>(observations.size() - 1u);
        cv::Point2d pt_n;
        for (int i = start_idx; i >= 0; --i) {
            const int frame_id = observations[i].keyframe_id;
            const int cam_idx = observations[i].camera_idx;
            CHECK_EQ(cam_idx, kCamIdxShow);
            const Eigen::Vector2d& keypoint = observations[i].key_point;
            cv::Point2d pt_c(keypoint(0), keypoint(1));
            if (frame_id == frame_id_show) {
                if (observations[i].used_counter == 2u) {
                    // Get visual reprojection error in last optimization.
                    Eigen::VectorXd dists;
                    vins_core::GetVisualReprojectionError(cameras_,
                                                          key_frames_,
                                                          keyframe_id_to_idx,
                                                          *feature,
                                                          &dists);
                    total_error += dists.mean();
                    // Highlight feature in last optimization.
                    cv::Point2d pt_l_top = cv::Point2d(pt_c.x-6,pt_c.y-6);
                    cv::Point2d pt_l_bot = cv::Point2d(pt_c.x+6,pt_c.y+6);
                    cv::rectangle(tracking_img, pt_l_top, pt_l_bot, cv::Scalar(0,255,0), 2);
                    opt_counter++;
                }
                pt_n = pt_c;
                continue;
            } else {
                int color_r = r2-(int)(r1/start_idx*i);
                int color_g = g2-(int)(g1/start_idx*i);
                int color_b = b2-(int)(b1/start_idx*i);
                cv::line(tracking_img, pt_c, pt_n, cv::Scalar(color_r,color_g,color_b));
                cv::circle(tracking_img, pt_c, 1, cv::Scalar(color_r, color_g, color_b), cv::FILLED);
                pt_n = pt_c;
            }
        }
    }

    const aslam::Transformation& T_OtoG = key_frames_.back().state.T_OtoG;
    const aslam::Transformation T_CtoG = T_OtoG * cameras_->get_T_BtoC(kCamIdxShow).inverse();
    const aslam::Transformation T_CtoM = T_GtoM_ * T_CtoG;
    const aslam::Transformation T_MtoC = T_CtoM.inverse();

    const bool reloc_init = hybrid_optimizer_ptr_->HasRelocInited();
    cv::Scalar loop_circle_color = reloc_init ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
    if (reloc_ && !last_reloc_landmarks_.empty()) {
        for (const auto& landmark : last_reloc_landmarks_) {
            const Eigen::Vector3d p_LinC = T_MtoC.transform(landmark);
            Eigen::Vector2d keypoint;
            const aslam::ProjectionResult projection_result =
                    cameras_->getCamera(0).project3(p_LinC, &keypoint);
            if ((projection_result == aslam::ProjectionResult::KEYPOINT_VISIBLE) ||
                    (projection_result == aslam::ProjectionResult::KEYPOINT_OUTSIDE_IMAGE_BOX)) {
                cv::Point2d pt_l(keypoint(0), keypoint(1));
                cv::circle(tracking_img, pt_l, 5, loop_circle_color, 2);
            }
        }
    }

    double avg_track_length = static_cast<double>(total_track_length) /
            static_cast<double>(tracked_counter);
    double avg_reproj_error = total_error / static_cast<double>(opt_counter);

    // NOTE: Visual keyframe score insert. Do not remove
    key_frames_.back().score = avg_reproj_error;
    
    const auto txtpt = cv::Point(10, 30);
    std::string txt_tracking = " TL:" + std::to_string(avg_track_length) +
                               " OPT:" + std::to_string(opt_counter) +
                               " VIO_E:" + std::to_string(avg_reproj_error);
    if (reloc_avg_error_pixel_ptr != nullptr) {
        txt_tracking += " RELOC_E:" + std::to_string(*reloc_avg_error_pixel_ptr);
    }

    cv::putText(tracking_img, txt_tracking, txtpt,
               cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 2);
    last_viz_imgs_tmp.push_back(std::make_shared<const cv::Mat>(tracking_img));

    // Third. reproject landmark and show depth.
    cv::Mat depth_img;
    if (key_frames_.back().visual_datas.at(kCamIdxShow)->image_ptr->type() == CV_8UC1) {
        cv::cvtColor(*(key_frames_.back().visual_datas.at(kCamIdxShow)->image_ptr),
                     depth_img, cv::COLOR_GRAY2RGB);
    } else {
        key_frames_.back().visual_datas.at(kCamIdxShow)->image_ptr->copyTo(depth_img);
    }

    int projection_visible_counter = 0;
    double depth_sum = 0.;
    for (const Eigen::Vector4d& pc_m : last_live_cloud_) {
        Eigen::Vector3d pc_l = T_MtoC.transform(pc_m.head<3>());
        Eigen::Vector2d keypoint;
        const aslam::ProjectionResult projection_result =
                cameras_->getCamera(kCamIdxShow).project3(pc_l, &keypoint);
        if (projection_result == aslam::ProjectionResult::KEYPOINT_VISIBLE) {
            const double depth = pc_l(2);
            if (depth < 0.0)  {
                continue;
            }
            int color_r = r2 - static_cast<int>(r1 / common::kMaxRangeVisual * depth);
            int color_g = g2 - static_cast<int>(g1 / common::kMaxRangeVisual * depth);
            int color_b = b2 - static_cast<int>(b1 / common::kMaxRangeVisual * depth);
            cv::Point2d pt(keypoint(0), keypoint(1));
            cv::circle(depth_img, pt, 5, cv::Scalar(color_g, color_r, color_b), cv::FILLED);
            depth_sum += depth;
            projection_visible_counter++;
        }
    }
    const double avg_depth = depth_sum / static_cast<double>(projection_visible_counter);
    const std::string txt_state = " VISIBLE:" + std::to_string(projection_visible_counter) +
                                " AVG_DEPTH:" + std::to_string(avg_depth);
    cv::putText(depth_img, txt_state, txtpt,
               cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,255,0), 2);
    last_viz_imgs_tmp.push_back(std::make_shared<const cv::Mat>(depth_img));

    std::unique_lock<std::mutex> lock(viz_mutex_);
    last_viz_imgs_ = last_viz_imgs_tmp;
}

void VinsHandler::UpdateFeatures(const int pop_frame_id) {
    // LUT.
    std::unordered_map<int, size_t> frame_id_to_idx;
    for (size_t i = 0u; i < key_frames_.size(); ++i) {
        frame_id_to_idx[key_frames_[i].keyframe_id] = i;
    }

    for (common::FeaturePointPtr& feature : features_) {
        common::ObservationDeq& observations = feature->observations;
        const int anchor_frame_idx = feature->anchor_frame_idx;

        if (anchor_frame_idx != -1) {
            const int anchor_frame_id = observations[anchor_frame_idx].keyframe_id;
            // Feature re-anchor.
            if (anchor_frame_id == pop_frame_id && feature->observations.size() > 1u) {
                CHECK(frame_id_to_idx.find(observations[anchor_frame_idx].keyframe_id) != frame_id_to_idx.end());
                // << observations[anchor_frame_idx].keyframe_id;
                CHECK(frame_id_to_idx.find(observations[anchor_frame_idx + 1].keyframe_id) != frame_id_to_idx.end());
                // << observations[anchor_frame_idx + 1].keyframe_id;
                const int old_frame_idx = frame_id_to_idx.at(observations[anchor_frame_idx].keyframe_id);
                const int new_frame_idx = frame_id_to_idx.at(observations[anchor_frame_idx + 1].keyframe_id);
                const int old_cam_idx = observations[anchor_frame_idx].camera_idx;
                const int new_cam_idx = observations[anchor_frame_idx + 1].camera_idx;
                const aslam::Transformation& old_T_OtoG = key_frames_[old_frame_idx].state.T_OtoG;
                const aslam::Transformation& new_T_OtoG = key_frames_[new_frame_idx].state.T_OtoG;
                const aslam::Transformation& old_T_CtoG = old_T_OtoG * cameras_->get_T_BtoC(old_cam_idx).inverse();
                const aslam::Transformation& new_T_CtoG = new_T_OtoG * cameras_->get_T_BtoC(new_cam_idx).inverse();
                const double old_inv_depth = feature->inv_depth;
                Eigen::Vector3d old_bearing_3d;
                cameras_->getCamera(old_cam_idx).backProject3(
                    observations[anchor_frame_idx].key_point, &old_bearing_3d);
                old_bearing_3d << old_bearing_3d(0) / old_bearing_3d(2),
                                  old_bearing_3d(1) / old_bearing_3d(2),
                                  1.0;
                const Eigen::Vector3d& old_p_LinC = old_bearing_3d / old_inv_depth;
                Eigen::Vector3d new_p_LinC = feature->ReAnchor(old_T_CtoG, new_T_CtoG, old_p_LinC);
                Eigen::Vector2d new_keypoint;
                cameras_->getCamera(new_cam_idx).project3(new_p_LinC, &new_keypoint);
                feature->inv_depth = 1.0 / new_p_LinC(2);
            }
        }
        if (observations.front().keyframe_id == pop_frame_id) {
            auto it = observations.begin();
            it = observations.erase(it);
        }
    }

    // Remove empty observation feature.
    for (auto it = features_.begin(); it != features_.end();) {
        common::FeaturePointPtr& feature = *it;
        if (feature->observations.empty()) {
            it = features_.erase(it);
        } else {
            ++it;
        }
    }
}

void VinsHandler::UpdatePosesForViz() {
    pg_poses_.resize(key_frames_ba_.size());
    for (size_t i = 0u; i < key_frames_ba_.size(); ++i) {
        pg_poses_[i] = key_frames_ba_[i].state.T_OtoG.getTransformationMatrix();
    }

    last_viz_edges_.clear();
    for (size_t i = 0u; i < loop_results_.size(); ++i) {
        const int keyframes_id_query = loop_results_[i].keyframe_id_query;
        const int keyframes_id_loop = loop_results_[i].keyframe_id_result;
        if (keyframes_id_loop == -1) {
            continue;
        }
        CHECK(keyframe_id_to_idx_ba_.find(keyframes_id_query) != keyframe_id_to_idx_ba_.end());
        CHECK(keyframe_id_to_idx_ba_.find(keyframes_id_loop) != keyframe_id_to_idx_ba_.end());
        const int keyframe_idx_query = keyframe_id_to_idx_ba_.at(keyframes_id_query);
        const int keyframe_idx_loop = keyframe_id_to_idx_ba_.at(keyframes_id_loop);
        last_viz_edges_.emplace_back(key_frames_ba_[keyframe_idx_query].state.T_OtoG.getPosition(),
                                     key_frames_ba_[keyframe_idx_loop].state.T_OtoG.getPosition());
    }
}

void VinsHandler::SelectFeatures(const std::unordered_map<int, size_t>& keyframe_id_to_idx) {
    typedef std::pair<int, size_t> IdAndTrackLength;
    const int last_frame_id = key_frames_.back().keyframe_id;
    // LUT.
    std::vector<IdAndTrackLength> track_id_and_track_length;
    std::unordered_map<int, int> track_id_to_idx;
    track_id_and_track_length.resize(features_.size());
    int in_optimization_counter = 0;
    for (int i = 0; i < static_cast<int>(features_.size()); ++i) {
        track_id_and_track_length[i] =
                std::make_pair(features_[i]->track_id, features_[i]->observations.size());
        track_id_to_idx[features_[i]->track_id] = i;
        if (features_[i]->using_in_optimization) {
            if (features_[i]->observations.back().keyframe_id != last_frame_id
                /*|| features_[i]->observations.back().depth == common::kInValidDepth*/) {
                features_[i]->using_in_optimization = false;
            } else {
                bool check_pass = true;
                if (config_->do_outlier_rejection) {
                    Eigen::VectorXd dists;
                    vins_core::GetVisualReprojectionError(cameras_,
                                                          key_frames_,
                                                          keyframe_id_to_idx,
                                                          *features_[i],
                                                          &dists);

                    if (dists.maxCoeff() > config_->outlier_rejection_scale *
                            config_->visual_sigma_pixel) {
                        check_pass = false;
                    }
                }
                if (check_pass) {
                    in_optimization_counter++;
                } else {
                    features_[i]->using_in_optimization = false;
                }
            }
        }
    }

    int in_optimization_residual =
            config_->max_feature_size_in_opt - in_optimization_counter;
    CHECK_GE(in_optimization_residual, 0);
    std::sort(track_id_and_track_length.begin(), track_id_and_track_length.end(),
        [](const IdAndTrackLength& a, const IdAndTrackLength& b) {
                return a.second > b.second;});
    for (int i = 0u;  i < static_cast<int>(track_id_and_track_length.size()); ++i) {
        if (in_optimization_residual == 0) {
            break;
        }
        const int track_id = track_id_and_track_length[i].first;
        const int anchor_frame_idx = features_[track_id_to_idx.at(track_id)]->anchor_frame_idx;
        if (anchor_frame_idx == -1) {
            continue;
        }
        const double inv_depth = features_[track_id_to_idx.at(track_id)]->inv_depth;
        if (features_[track_id_to_idx.at(track_id)]->anchor_frame_idx != -1 && inv_depth > 0 &&
            features_[track_id_to_idx.at(track_id)]->using_in_optimization == false &&
            features_[track_id_to_idx.at(track_id)]->observations.back().keyframe_id == last_frame_id &&
            track_id_and_track_length[i].second > 1u) {
            bool check_pass = true;
            if (config_->do_outlier_rejection) {
                Eigen::VectorXd dists;
                vins_core::GetVisualReprojectionError(cameras_,
                                                      key_frames_,
                                                      keyframe_id_to_idx,
                                                      *features_[track_id_to_idx.at(track_id)],
                                                      &dists);

                if (dists.maxCoeff() > config_->outlier_rejection_scale * config_->visual_sigma_pixel) {
                    check_pass = false;
                }
            }
            if (check_pass) {
                features_[track_id_to_idx.at(track_id)]->using_in_optimization = true;
                in_optimization_residual--;
            }
        }
    }
    const int total_in_optimization_size =
            config_->max_feature_size_in_opt - in_optimization_residual;
    VLOG(1) << total_in_optimization_size
            << " feature in optimization at frame: " << last_frame_id
            << " of total feature " << features_.size() << " in window.";
}

void VinsHandler::LoopQuery(loop_closure::LoopCandidate* candidate_ptr) {
    loop_closure::LoopCandidate& candidate = *CHECK_NOTNULL(candidate_ptr);
    const common::KeyFrame& key_frame = candidate.first;
    common::LoopResult& loop_result = candidate.second;
    CHECK_EQ(key_frame.keyframe_id, loop_result.keyframe_id_query);

    std::function<bool(const common::KeyFrame&, const common::LoopResult&)> loop_verify_helper =
        [&](const common::KeyFrame& key_frame,
            const common::LoopResult& loop_result) {
        if (last_frame_to_frame_loop_candidate_ptr_ == nullptr) {
            last_frame_to_frame_loop_candidate_ptr_ = std::make_unique<
                loop_closure::LoopCandidate>(std::make_pair(
                    key_frame, loop_result));
            return false;
        } else {
            // Check frame to frame loop candidate by slam pose.
            const aslam::Transformation T_O1toO2_prior = 
                last_frame_to_frame_loop_candidate_ptr_->first.state.T_OtoG.inverse() *
                    key_frame.state.T_OtoG;
            const aslam::Transformation T_O1toO2_estimate =
                last_frame_to_frame_loop_candidate_ptr_->second.T_estimate.inverse() *
                    loop_result.T_estimate;

            constexpr double kMaxPositionError = 1.0; // in meters.
            constexpr double kMaxRotationError = 12.0 * common::kDegToRad; // in rad.

            const bool check_success = common::CheckPoseSimilar(T_O1toO2_prior,
                                                                T_O1toO2_estimate,
                                                                kMaxPositionError,
                                                                kMaxRotationError);

            last_frame_to_frame_loop_candidate_ptr_ = std::make_unique<
                loop_closure::LoopCandidate>(std::make_pair(
                    key_frame, loop_result));

            if (!check_success) {
                VLOG(1) << "Delta pose is not similar enough between current and last loop candidate, "
                        << "not use current loop candidate for mapping.";
            } else {
                VLOG(1) << "Find a frame to frame loop in keyframe: " << key_frame.keyframe_id;
            }
            return check_success;
        }
    };

    bool loop_success = false;
    if (loop_result.loop_sensor == loop_closure::LoopSensor::kVisual) {
        TIME_TIC(VISUAL_LOOP_CLOSURE);
        std::unique_ptr<loop_closure::VertexKeyPointToStructureMatchList> loop_matches_ptr = nullptr;
        if (config_->mapping) {
            loop_matches_ptr.reset(new loop_closure::VertexKeyPointToStructureMatchList);
        }

        loop_success = visual_loop_interface_ptr_->Query(key_frame.visual_datas, T_GtoM_,
                                                         &loop_result, loop_matches_ptr.get());

        if (loop_success) {
            VLOG(1) << "Detected visual loop candidate in frame: "
                    << key_frame.keyframe_id;
            if (loop_matches_ptr != nullptr) {
                const bool pass_verify = loop_verify_helper(key_frame, loop_result);
                if (!pass_verify) {
                    return;
                }

                loop_matches_ba_.emplace_back(key_frame.keyframe_id,
                                              *loop_matches_ptr);

                const int keyframes_id_loop = loop_result.keyframe_id_result;
                CHECK_NE(keyframes_id_loop, -1);
                CHECK(keyframe_id_to_idx_ba_.find(keyframes_id_loop) != keyframe_id_to_idx_ba_.end());
                const int keyframe_idx_loop = keyframe_id_to_idx_ba_.at(keyframes_id_loop);
                last_viz_edges_.emplace_back(key_frame.state.T_OtoG.getPosition(),
                                             key_frames_ba_[keyframe_idx_loop].state.T_OtoG.getPosition());
                if (tuning_mode_ == common::TuningMode::Online) {
                    new_online_loop_counter_++;
                }
            }

            std::unique_lock<std::mutex> lock(loop_result_mutex_);
            loop_results_.push_back(loop_result);

            last_loop_result_ = loop_result;
            have_new_loop_ = true;
            // NOTE(chien): Do not update slam state in there,
            // must check visual repojection state after fusion.
        } else if (reloc_) {
            loop_result.pnp_inliers.resize(0u);
        }
        TIME_TOC(VISUAL_LOOP_CLOSURE);
    } else if (loop_result.loop_sensor == loop_closure::LoopSensor::kScan) {
        if (config_->mapping && tuning_mode_ == common::TuningMode::Online) {
            TIME_TIC(SCAN_LOOP_CLOSURE);
            //TODO: do online scan loop query.
            common::LoopResults tmp_loop_results;
            scan_loop_interface_ptr_->DetectScanInterLoopOnline(
                key_frame,
                key_frames_show_,
                config_->scan_loop_inter_distance,
                &tmp_loop_results);
            if (!tmp_loop_results.empty()) {
                for (const auto& loop_result : tmp_loop_results) {
                    VLOG(1) << "Detected scan loop candidate in frame: "
                            << key_frame.keyframe_id;
                    const bool pass_verify = loop_verify_helper(key_frame, loop_result);
                    if (!pass_verify) {
                        continue;
                    }

                    const int keyframes_id_loop = loop_result.keyframe_id_result;
                    CHECK_NE(keyframes_id_loop, -1);
                    CHECK(keyframe_id_to_idx_ba_.find(keyframes_id_loop) != keyframe_id_to_idx_ba_.end());
                    const int keyframe_idx_loop = keyframe_id_to_idx_ba_.at(keyframes_id_loop);
                    last_viz_edges_.emplace_back(key_frame.state.T_OtoG.getPosition(),
                                                 key_frames_ba_[keyframe_idx_loop].state.T_OtoG.getPosition());
                    std::unique_lock<std::mutex> lock(loop_result_mutex_);
                    loop_results_.push_back(loop_result);

                    last_loop_result_ = loop_result;
                    have_new_loop_ = true;

                    new_online_loop_counter_++;
                }
            }
            TIME_TOC(SCAN_LOOP_CLOSURE);
        } else if (reloc_) {
            if (live_global_submaps_ptr_ == nullptr) {
                return;
            }
            if (!hybrid_optimizer_ptr_->HasRelocInited()) {
                TIME_TIC(SCAN_BF_SEARCHING);
                loop_success = FindCandidateBF(key_frame, &loop_result);
                if (loop_success) {
                    VLOG(1) << "Find scan loop candidate success by brute-force search,"
                            << " in keyframe: " << key_frame.keyframe_id;
                    std::unique_lock<std::mutex> lock(loop_result_mutex_);
                    loop_results_.push_back(loop_result);

                    last_loop_result_ = loop_result;
                    have_new_loop_ = true;
                }
                TIME_TOC(SCAN_BF_SEARCHING);
            } else {
                TIME_TIC(TUNE_MAP_POSE);
#if 0
                TuneMapPose(live_global_submaps_ptr_->submaps().front(),
                            key_frame,
                            &T_GtoM_);
#else
                ScanMatching(live_global_submaps_ptr_->submaps().front(),
                            key_frame,
                            &T_GtoM_);
#endif
                TIME_TOC(TUNE_MAP_POSE);
            }
        }
    }

    if (new_online_loop_counter_ > 5 || data_finish_) {
        call_posegraph_ = true;
        new_online_loop_counter_ = 0;
    }

    // If run on reloc mode, update SLAM state by re-projection checking.
    if (reloc_ && !hybrid_optimizer_ptr_->HasRelocInited() && loop_success) {
        reloc_init_mutex_.lock();
        reloc_init_candidates_.emplace_back(key_frame, loop_result);
        if (reloc_init_candidates_.size() > kMaxLoopCandidateSize) {
            reloc_init_candidates_.pop_front();
        }
        reloc_init_mutex_.unlock();
        call_reloc_init_ = true;
    }
}

void VinsHandler::LoopQuery(const loop_closure::LoopCandidatePtrOneFrame& candidates) {
    for (const loop_closure::LoopCandidatePtr& candidate_ptr : candidates) {
        LoopQuery(candidate_ptr.get());
    }
}

void VinsHandler::TryInitRelocWithPrior() {
    if (key_frames_.empty()) {
        VLOG(0) << "Keyframes is empty in now...!";
        // Wait for SLAM keyframe comming.
        // NOTE(chien): SLAM reloc init run on thread. 
        while (key_frames_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    const common::KeyFrameType this_keyframe_type = key_frames_.back().GetType();
    const common::KeyFrame this_keyframe = key_frames_.back();
    common::LoopResult loop_result;
    bool reloc_success = false;
    if (this_keyframe_type == common::KeyFrameType::Scan ||
        this_keyframe_type == common::KeyFrameType::ScanAndVisual) {
        VLOG(0) << "Get new scan keyframe, try init reloc with prior.";
        if (FindCandidateByPrior(this_keyframe, *T_GtoM_prior_ptr_, &loop_result)) {
            if (loop_result.score > config_-> reloc_accept_score) {
                const auto& global_grid = live_global_submaps_ptr_->submaps().front()->grid();
                const auto& cell_idx_estimate =
                    global_grid->limits().GetCellIndex(
                        loop_result.T_estimate.getPosition().head<2>());
                if (!global_grid->IsKnown(cell_idx_estimate)) {
                    VLOG(0) << "The initialization result cannot pass the final verification. "
                            << " because the pose estimated is not in know area. ";
                } else {
                    reloc_success = true;
                    VLOG(0) << "Scan prior reloc success, RCSM score: "
                            << loop_result.score;
                }
            } else {
                VLOG(0) << "Loop candidate score smaller than threshold: "
                        << loop_result.score << " vs "
                        << config_->reloc_accept_score;
            }
        } else {
            VLOG(0) << "Failed find loop by prior and scan measurement, matching score: "
                    << loop_result.score;
        }
    }

    if (this_keyframe_type == common::KeyFrameType::Visual ||
        this_keyframe_type == common::KeyFrameType::ScanAndVisual) {
        VLOG(0) << "Get new visual keyframe, try init reloc with prior.";
        loop_result.visual_loop_type = loop_closure::VisualLoopType::kMapTracking;
        if (visual_loop_interface_ptr_->Query(this_keyframe.visual_datas,
                                              *T_GtoM_prior_ptr_,
                                              &loop_result,
                                              nullptr)) {
            if (static_cast<int>(loop_result.pnp_inliers.size()) > config_->min_inlier_count) {
                reloc_success = true;
                VLOG(0) << "Visual prior reloc success, pnp inlier size: "
                        << loop_result.pnp_inliers.size();
            } else {
                VLOG(0) << "Loop inlier size smaller than threshold: "
                        << loop_result.pnp_inliers.size() << " vs "
                        << config_->min_inlier_count;
            }
        } else {
            VLOG(0) << "Failed find loop by prior and visual measurement, PnP inlier: "
                    << loop_result.pnp_inliers.size();
        }
    }

    if (reloc_success) {
        const aslam::Transformation T_GtoM_init =
            loop_result.T_estimate * this_keyframe.state.T_OtoG.inverse();
        SetRelocInitSuccess(T_GtoM_init);
        VLOG(0) << "Successfully reloc by prior";
    }

    T_GtoM_prior_ptr_.reset();
}

void VinsHandler::TryInitReloc() {
    if (reloc_init_candidates_.size() < kMaxLoopCandidateSize) {
        return;
    }

    TIME_TIC(RELOC_INITIALIZATION);
    std::vector<aslam::Transformation> T_GtoM_candidate;

    reloc_init_mutex_.lock();
    std::deque<loop_closure::LoopCandidate> reloc_init_canditates_copy = reloc_init_candidates_;
    reloc_init_mutex_.unlock();

    for (size_t i = 0u; i < reloc_init_canditates_copy.size(); ++i) {
        if (reloc_init_canditates_copy[i].second.loop_sensor == loop_closure::LoopSensor::kVisual) {
            CHECK(!reloc_init_canditates_copy[i].second.pnp_inliers.empty());
        }
        const aslam::Transformation& T_OtoM = reloc_init_canditates_copy[i].second.T_estimate;
        const aslam::Transformation& T_OtoG = reloc_init_canditates_copy[i].first.state.T_OtoG;
        const aslam::Transformation T_GtoM = T_OtoM * T_OtoG.inverse();
        T_GtoM_candidate.push_back(T_GtoM);
    }

    aslam::Transformation T_GtoM_init = T_GtoM_candidate.front();
    for (size_t i = 1u; i < T_GtoM_candidate.size(); ++i) {
        Eigen::Vector3d t_GtoM_interpolated = 0.5 * T_GtoM_init.getPosition() +
                0.5 * T_GtoM_candidate[i].getPosition();
        Eigen::Quaterniond q_GtoM_interpolated = T_GtoM_init.getEigenQuaternion().slerp(
                    0.5, T_GtoM_candidate[i].getEigenQuaternion());
        T_GtoM_init.update(q_GtoM_interpolated, t_GtoM_interpolated);
    }

    constexpr double kMaxPositionError = 0.5; // in meters.
    constexpr double kMaxRotationError = 6.0 * common::kDegToRad; // in rad.

    // All transformation candidate need similar enough.
    for (size_t i = 0u; i < reloc_init_canditates_copy.size(); ++i) {
        if (!common::CheckPoseSimilar(T_GtoM_init, T_GtoM_candidate[i],
                                      kMaxPositionError, kMaxRotationError)) {
            VLOG(0) << "Candidate not similar enough, cancel initialization.";
            return;
        }
    }

    common::KeyFrames key_frames_copy;
    common::LoopResults loop_results_copy;
    common::FeaturePointPtrVec features_empty;
    for (const auto& candidate : reloc_init_canditates_copy) {
        key_frames_copy.push_back(candidate.first);
        loop_results_copy.push_back(candidate.second);
    }

    const common::Grid2D* grid = nullptr;
    if (live_global_submaps_ptr_ != nullptr &&
        !live_global_submaps_ptr_->submaps().empty()) {
        grid = live_global_submaps_ptr_->submaps().front()->grid();
    }
    constexpr bool kDoInitOptimization = true;
    if (kDoInitOptimization) {
        vins_core::ParaBuffer buffer;
        vins_core::KeyframeToBuffer(key_frames_copy, cameras_, features_empty, loop_results_copy, T_GtoM_init, &buffer);
        hybrid_optimizer_ptr_->InitReloc(cameras_,
                                         config_,
                                         key_frames_copy,
                                         loop_results_copy,
                                         grid,
                                         &buffer);
        vins_core::BufferToKeyframe(buffer, cameras_, &key_frames_copy, &features_empty, &T_GtoM_init);
    }

    // All keyframe score must be small than threashold.
    bool check_accept = true;
    for (size_t i = 0u; i < loop_results_copy.size(); ++i) {
        if (loop_results_copy[i].loop_sensor == loop_closure::LoopSensor::kVisual) {
            if (static_cast<int>(loop_results_copy[i].pnp_inliers.size()) < config_->min_inlier_count) {
                VLOG(2) << "The initialization result cannot pass the final verification. "
                        << " because have not enough pnp inlier. ";
            }
            double reprojection_error_avg = 1e9;
            ComputeAvgRelocReprojectionError(loop_results_copy[i],
                                             key_frames_copy[i].state.T_OtoG,
                                             T_GtoM_init,
                                             &reprojection_error_avg);

            VLOG(2) << "Visual reprojection error on map in reloc init check: "
                    << reprojection_error_avg;

            if (reprojection_error_avg > config_->outlier_rejection_scale * 
                    config_->visual_sigma_pixel) {
                VLOG(0) << "The initialization result cannot pass the final verification. "
                        << " reprojection error: " << reprojection_error_avg
                        << " vs " << config_->outlier_rejection_scale * config_->visual_sigma_pixel;
                check_accept = false;
                break;
            }
        } else if (loop_results_copy[i].loop_sensor == loop_closure::LoopSensor::kScan) {
            const double score = ScoreScanLoop(key_frames_copy[i],
                                               T_GtoM_init,
                                               grid);

            if (kShowScanLoopResult) {
                common::ShowScanMatching(live_global_submaps_ptr_->submaps().front(),
                                         T_GtoM_init,
                                         key_frames_copy[i]);
            }

            VLOG(2) << "Scan occupancied score on map in reloc init check: "
                    << score;

            if (score < config_-> reloc_accept_score) {
                VLOG(0) << "The initialization result cannot pass the final verification. "
                        << " score: " << score << " vs " << config_-> reloc_accept_score;
                check_accept = false;
                break;
            } else {
                const aslam::Transformation T_OtoM_estimate =
                    T_GtoM_init * key_frames_copy[i].state.T_OtoG;
                const auto& cell_idx_estimate =
                    grid->limits().GetCellIndex(
                        T_OtoM_estimate.getPosition().head<2>());
                if (!grid->IsKnown(cell_idx_estimate)) {
                    VLOG(0) << "The initialization result cannot pass the final verification, "
                            << "because estimated pose is in the unknown area of map";
                    check_accept = false;
                    break;
                }
            }
        }
    }

    if (check_accept) {
        SetRelocInitSuccess(T_GtoM_init);
    }
    TIME_TOC(RELOC_INITIALIZATION);
}

void VinsHandler::TuneMapPose(const std::shared_ptr<common::Submap2D>& submap,
                              const common::KeyFrame& key_frame,
                              aslam::Transformation* T_GtoM_ptr) {
    aslam::Transformation& T_GtoM = *CHECK_NOTNULL(T_GtoM_ptr);

    vins_core::ParaBuffer buffer;
    common::KeyFrames key_frames_tmp;
    key_frames_tmp.push_back(key_frame);
    common::FeaturePointPtrVec features_empty;
    common::LoopResults loop_results_copy;
    vins_core::KeyframeToBuffer(key_frames_tmp, cameras_, features_empty, loop_results_copy, T_GtoM, &buffer);
    hybrid_optimizer_ptr_->MapPoseTuning(
        config_,
        key_frames_tmp,
        submap->grid(),
        &buffer);
    std::unique_lock<std::mutex> lock(fusion_mutex_);
    vins_core::BufferToKeyframe(buffer, cameras_, &key_frames_tmp, &features_empty, &T_GtoM);
}

void VinsHandler::ComputeAvgRelocReprojectionError(
        const common::LoopResult& loop_result,
        const aslam::Transformation& T_OtoG,
        const aslam::Transformation& T_GtoM,
        double* reprojection_error_avg_ptr) {
    double& reprojection_error_avg = *CHECK_NOTNULL(reprojection_error_avg_ptr);
    Eigen::VectorXd dists;
    vins_core::GetMapVisualReprojectionError(cameras_,
                                             loop_result,
                                             T_OtoG,
                                             T_GtoM,
                                             &dists);
    const double reprojection_error_sum = dists.sum();

    if (!loop_result.pnp_inliers.empty()) {
        reprojection_error_avg = reprojection_error_sum /
                static_cast<double>(loop_result.pnp_inliers.size());
        VLOG(1) << "Reloc avg reprojection error (pixel): " << reprojection_error_avg;
    } else {
        VLOG(1) << "Inlier empty, reloc un-success.";
        reprojection_error_avg = 1e9;
    }
}

bool VinsHandler::FindCandidateByPrior(
        const common::KeyFrame& keyframe,
        const aslam::Transformation& T_OtoM_prior,
        common::LoopResult* loop_result_ptr) {       
    common::LoopResult& loop_result = *CHECK_NOTNULL(loop_result_ptr);
    CHECK(!(live_global_submaps_ptr_->submaps().empty()));
    std::shared_ptr<common::Submap2D> global_map =
            live_global_submaps_ptr_->submaps().front();

    common::KeyFrame temp_keyframe = keyframe;
    // Set T_OtoM_prior as temp for local search.
    temp_keyframe.state.T_OtoG = T_OtoM_prior;
    common::MatchingOption option;
    option.linear_search_window_ = 0.3;
    option.angular_search_window_ = common::kDegToRad * 25.0;
    option.min_score_ = config_->reloc_accept_score;

    common::MatchingResult result = CHECK_NOTNULL(reloc_init_matcher_ptr_)->
        Match(temp_keyframe, option);
    if (result.score < 0) {
        return false;
    }

    loop_result = common::LoopResult(keyframe.state.timestamp_ns,
                                     keyframe.keyframe_id,
                                     -1,
                                     result.score,
                                     loop_closure::LoopSensor::kScan,
                                     loop_closure::VisualLoopType::kGlobal,
                                     result.pose_estimate);

    return true;
}

bool VinsHandler::FindCandidateBF(
        const common::KeyFrame& keyframe,
        common::LoopResult* loop_result_ptr) {
    common::LoopResult& loop_result = *CHECK_NOTNULL(loop_result_ptr);
    CHECK(!(live_global_submaps_ptr_->submaps().empty()));
    std::shared_ptr<common::Submap2D> global_map =
            live_global_submaps_ptr_->submaps().front();

    common::MatchingOption option;
    const int num_x_cells = global_map->grid()->limits().cell_limits().num_x_cells;
    const int num_y_cells = global_map->grid()->limits().cell_limits().num_y_cells;
    option.linear_search_window_ = (std::max(num_x_cells, num_y_cells) / 2) *
            global_map->grid()->limits().resolution();
    option.angular_search_window_ = M_PI;
    option.min_score_ = config_->reloc_accept_score;

    common::MatchingResult result = CHECK_NOTNULL(reloc_init_matcher_ptr_)->
        Match(keyframe, option);
    if (result.score < 0) {
        return false;
    }

    loop_result = common::LoopResult(keyframe.state.timestamp_ns,
                                     keyframe.keyframe_id,
                                     -1,
                                     result.score,
                                     loop_closure::LoopSensor::kScan,
                                     loop_closure::VisualLoopType::kGlobal,
                                     result.pose_estimate);

    return true;
}

void VinsHandler::RemoveDynamicObject(const std::shared_ptr<common::Submap2D>& submap,
                                        common::KeyFrame* key_frame_ptr) {
    common::KeyFrame& key_frame = *CHECK_NOTNULL(key_frame_ptr);
    
    // Cluster point cloud.
    common::EigenVector3dVec global_pc;
    for (size_t i = 0u; i < key_frame.points.points.size(); ++i) {
        global_pc.push_back(key_frame.state.T_OtoG.transform(
        key_frame.points.points[i]));
            }

    constexpr double cluster_threshold = 0.075;
    
    std::vector<std::vector<size_t>> point_clusters;
    std::vector<size_t> new_cluster;
    
    for (size_t index = 0u; index < global_pc.size(); ++index) {
        if (index == 0) {
            new_cluster.push_back(index);
            continue;
        }
        if (new_cluster.size() != 0u) {
            size_t last_index = new_cluster.back();
            if ((global_pc.at(index).head<2>() - global_pc.at(last_index).head<2>()).norm() <
                cluster_threshold) {
                new_cluster.push_back(index);
            } else {
                point_clusters.push_back(new_cluster);
                new_cluster.clear();
                new_cluster.push_back(index);
            }
        } else {
            LOG(ERROR) << "ERROR: New cluster is empty!";
        }
    }
    if (new_cluster.size() > 0) {
        point_clusters.push_back(new_cluster);
        new_cluster.clear();
    }

    if ((global_pc[point_clusters.front().front()].head<2>() - 
            global_pc[point_clusters.back().back()].head<2>()).norm() < cluster_threshold) {
        point_clusters.back().insert(point_clusters.back().end(), 
                point_clusters.front().begin(), point_clusters.front().end());
        point_clusters.erase(point_clusters.begin());
    }
    
    const Eigen::Array2i Pattern[9] = {
            Eigen::Array2i(-2, 0),
            Eigen::Array2i(-1, -1),
            Eigen::Array2i(-1, 1),
            Eigen::Array2i(0, -2),
            Eigen::Array2i(0, 0),
            Eigen::Array2i(0, 2),
            Eigen::Array2i(1, -1),
            Eigen::Array2i(1, 1),
            Eigen::Array2i(2, 0)
    };

    std::unique_lock<std::mutex> lock(matching_grid_mutex_);
    // Statistics the average cost in the cluster.
    std::vector<bool> static_status(global_pc.size(), true);
    const auto grid = submap->grid();
    const common::MapLimits& maplimit = grid->limits();
    for (const auto& cluster : point_clusters) {
        bool is_dynamic = false;
        Eigen::Vector2d cluster_centre = Eigen::Vector2d::Zero();
        for (auto c : cluster) {
            cluster_centre += key_frame.points.points[c].head<2>();
        }
        cluster_centre /= cluster.size();
        double max_dis = 0.0;
        for (auto c : cluster) {
            const double dis =
                (key_frame.points.points[c].head<2>() - cluster_centre).norm();
            if (dis > max_dis) {
                max_dis = dis;
            }
        }
        if (cluster_centre.norm() < 6. && max_dis > 0.15) {
            float cluster_cost_sum = 0.; 
            for (auto c : cluster) {
                const Eigen::Vector3d global_point = global_pc[c];
                const Eigen::Array2i cell_index =
                        maplimit.GetCellIndex(global_point.head<2>());
                float max_cost = 0.;
                for (const auto& p : Pattern) {
                    max_cost = std::max(max_cost,
                            grid->GetValue(cell_index + p));
                }
                cluster_cost_sum += max_cost;
            }
            const float cluster_cost_mean = cluster_cost_sum / cluster.size();
            if (cluster_cost_mean < 0.495) {
                is_dynamic = true;
            }
        }
        if (is_dynamic) {
            for (auto c : cluster) {
                static_status[c] = false;
            }
        }
    }

    common::ReduceVector(static_status, &key_frame.points.points);
}

common::MatchingResult CorrelativeScanMatch(
        const common::KeyFrame& range_data,
        const std::shared_ptr<const common::Submap2D>& matching_submap,
        const bool compute_rectional) {

    common::RealTimeCorrelativeScanMatcher matcher(matching_submap->grid());
    common::MatchingOption option;
    option.linear_search_window_ = 0.15;
    option.angular_search_window_ = common::kDegToRad * 16.0;
    option.min_score_ = 0.7;
    option.compute_scan_rectional = compute_rectional;
    const common::MatchingResult result = matcher.Match(range_data, option);
    VLOG(2) << "RCSM score: " << result.score
            << " in keyframe: " << range_data.keyframe_id;
    return result;
}

void VinsHandler::ScanMatching(const std::shared_ptr<common::Submap2D>& submap,
                               common::KeyFrame* key_frame_ptr) {
    common::KeyFrame& key_frame = *CHECK_NOTNULL(key_frame_ptr);

    VLOG(1) << "Scan cloud size in keyframe " << key_frame.keyframe_id
            << ": points " << key_frame.points.points.size()
            << ", miss_points " << key_frame.points.miss_points.size();

    // Update predict rotation by scan matching.
    const common::MatchingResult match_result = CorrelativeScanMatch(
            key_frame,
            submap,
            false);
    if (match_result.score > 0.) {
        key_frame.state.T_OtoG = match_result.pose_estimate;
        key_frame.score = match_result.score;
    }

    if (kShowMatchingResult) {
        common::ShowScanMatching(submap, aslam::Transformation(), key_frame);
    }
}

void VinsHandler::ScanMatching(const std::shared_ptr<common::Submap2D>& submap,
                               const common::KeyFrame& key_frame,
                               aslam::Transformation* T_GtoM_ptr) {
    aslam::Transformation& T_GtoM = *CHECK_NOTNULL(T_GtoM_ptr);

    // Transfrom robot pose to map frame.
    common::KeyFrame key_frame_tmp = key_frame;
    key_frame_tmp.state.T_OtoG = T_GtoM * key_frame_tmp.state.T_OtoG;

    // Update predict rotation by scan matching.
    const common::MatchingResult match_result = CorrelativeScanMatch(
            key_frame_tmp,
            submap,
            false);

    if (match_result.score > 0.) {
        const aslam::Transformation& T_OtoM_estimate = match_result.pose_estimate;
        const aslam::Transformation& T_OtoG = key_frame.state.T_OtoG;
        std::unique_lock<std::mutex> lock(fusion_mutex_);
        T_GtoM_ = T_OtoM_estimate * T_OtoG.inverse();
    }
}

std::shared_ptr<common::Submap2D> VinsHandler::InsertRangeData(
        const common::KeyFrame& range_data,
        const int max_submap_size,
        common::PointCloud* pc_LinGs_ptr) {
    common::PointCloud& pc_LinGs = *CHECK_NOTNULL(pc_LinGs_ptr);

    pc_LinGs.points.resize(range_data.points.points.size());
    for (size_t i = 0u; i < range_data.points.points.size(); ++i) {
        pc_LinGs.points[i] = range_data.state.T_OtoG.transform(
                    range_data.points.points[i]);
    }
    pc_LinGs.miss_points.resize(range_data.points.miss_points.size());
    for (size_t i = 0u; i < range_data.points.miss_points.size(); ++i) {
        pc_LinGs.miss_points[i] = range_data.state.T_OtoG.transform(
                    range_data.points.miss_points[i]);
    }

    std::unique_lock<std::mutex> lock(matching_grid_mutex_);
    std::shared_ptr<common::Submap2D> finished_submap =
            live_local_submaps_ptr_->InsertRangeData(range_data,
                                                     pc_LinGs,
                                                     T_StoO_,
                                                     max_submap_size);
    return finished_submap;
}


void VinsHandler::InsertRangeDataInGlobalMap(
        const common::KeyFrame& range_data,
        const aslam::Transformation& T_GtoM) {
    common::PointCloud p_LinMs;
    p_LinMs.points.resize(range_data.points.points.size());
    for (size_t i = 0u; i < range_data.points.points.size(); ++i) {
        const Eigen::Vector3d p_LinG = range_data.state.T_OtoG.transform(
                    range_data.points.points[i]);
        p_LinMs.points[i] = T_GtoM.transform(p_LinG);
    }
    p_LinMs.miss_points.resize(range_data.points.miss_points.size());
    for (size_t i = 0u; i < range_data.points.miss_points.size(); ++i) {
        const Eigen::Vector3d p_LinG = range_data.state.T_OtoG.transform(
                    range_data.points.miss_points[i]);
        p_LinMs.miss_points[i] = T_GtoM.transform(p_LinG);
    }

    common::KeyFrame range_data_copy = range_data;
    range_data_copy.state.T_OtoG = T_GtoM * range_data_copy.state.T_OtoG;

    live_global_submaps_ptr_->InsertRangeData(range_data_copy,
                                              p_LinMs,
                                              T_StoO_,
                                              -1);
}

void VinsHandler::Fusion() {
    std::unique_lock<std::mutex> lock(fusion_mutex_);

    if (key_frames_.size() >= static_cast<size_t>(config_->window_size)) {
            // LUT.
        std::unordered_map<int, size_t> keyframe_id_to_idx;
        for (size_t i = 0u; i < key_frames_.size(); ++i) {
            keyframe_id_to_idx[key_frames_[i].keyframe_id] = i;
        }

        // TODO(chien): Check is there have problem if last keyframe is scan frame
        //              when perform first fusion.
        
        const auto last_keyframe_type = key_frames_.back().GetType();
        if (last_keyframe_type == common::KeyFrameType::Visual ||
            last_keyframe_type == common::KeyFrameType::ScanAndVisual) {
            // Select feature for optimization.
            SelectFeatures(keyframe_id_to_idx);
            
            if (reloc_ && hybrid_optimizer_ptr_->HasRelocInited()) {
                std::unique_lock<std::mutex> lock(loop_result_mutex_);
                SelectRelocFeatures(key_frames_,
                                    T_GtoM_,
                                    config_->do_outlier_rejection,
                                    &loop_results_);
            }
        }

        std::unique_lock<std::mutex> lock(matching_grid_mutex_);
        vins_core::KeyframeToBuffer(key_frames_, cameras_, features_, loop_results_, T_GtoM_, &buffer_);
        TIME_TIC(HYBIRD_OPTIMIZATION);
        hybrid_optimizer_ptr_->Solve(cameras_, config_, key_frames_, keyframe_id_to_idx,
                                    features_, loop_results_,
                                    live_local_submaps_ptr_->submaps().empty() ?
                                        nullptr : live_local_submaps_ptr_->submaps().front()->grid(),
                                    live_global_submaps_ptr_ == nullptr ?
                                        nullptr : live_global_submaps_ptr_->submaps().front()->grid(),
                                    &buffer_, nullptr);
        TIME_TOC(HYBIRD_OPTIMIZATION);
        TIME_TIC(MARGINALIZATION);
        hybrid_optimizer_ptr_->Marginalize(cameras_, config_, key_frames_, keyframe_id_to_idx,
                                            features_, loop_results_,
                                            live_local_submaps_ptr_->submaps().empty() ?
                                                nullptr : live_local_submaps_ptr_->submaps().front()->grid(),
                                            live_global_submaps_ptr_ == nullptr ?
                                                nullptr : live_global_submaps_ptr_->submaps().front()->grid(),
                                            T_GtoM_, &buffer_);
        TIME_TOC(MARGINALIZATION);
        vins_core::BufferToKeyframe(buffer_, cameras_, &key_frames_, &features_, &T_GtoM_);
    }
}


double VinsHandler::UpdateSlamState() {
    const auto keyframe_type = key_frames_.back().GetType();
    double score = 0.0;
    int check_result_scan = -1;
    int check_result_visual = -1;
    if (keyframe_type == common::KeyFrameType::Scan ||
        keyframe_type == common::KeyFrameType::ScanAndVisual) {
        if (live_global_submaps_ptr_ != nullptr &&
            !live_global_submaps_ptr_->submaps().empty()) {
            const common::Grid2D* grid = live_global_submaps_ptr_->submaps().front()->grid();
            score = ScoreScanLoop(key_frames_.back(), T_GtoM_, grid);
            if (kShowScanLoopResult) {
                common::ShowScanMatching(live_global_submaps_ptr_->submaps().front(),
                                         T_GtoM_,
                                         key_frames_.back());
            }
            VLOG(1) << "Reloc score is: " << score << " (scan frame)";
            if (score > config_->min_monitor_score + 0.2) {
                check_result_scan = 1;
            } else if (score < config_->min_monitor_score - 0.2) {
                check_result_scan = -1;
            } else {
                check_result_scan = 0;
            }
        }
    }
    if (keyframe_type == common::KeyFrameType::Visual ||
        keyframe_type == common::KeyFrameType::ScanAndVisual) {
        // LUT.
        std::unordered_map<int, size_t> keyframe_id_to_idx;
        for (size_t i = 0; i < key_frames_.size(); ++i) {
            keyframe_id_to_idx[key_frames_[i].keyframe_id] = i;
        }

        for (int i = static_cast<int>(loop_results_.size()) - 1; i > 0; --i) {
            const auto iter = keyframe_id_to_idx.find(loop_results_[i].keyframe_id_query);
            if (iter != keyframe_id_to_idx.end()) {
                std::unique_lock<std::mutex> lock(loop_result_mutex_);
                const aslam::Transformation& T_OtoG = key_frames_[iter->second].state.T_OtoG;
                double reprojection_error_avg = 1e9;
                const auto& loop_result = loop_results_[i];
                ComputeAvgRelocReprojectionError(loop_result,
                                                 T_OtoG,
                                                 T_GtoM_,
                                                 &reprojection_error_avg);
                VLOG(1) << "Reloc score is: " << reprojection_error_avg << " (visual frame)";

                if (reprojection_error_avg < config_->outlier_rejection_scale *
                        config_->visual_sigma_pixel) {
                    check_result_visual = 1;
                } else if (reprojection_error_avg > 2.0 * config_->outlier_rejection_scale *
                        config_->visual_sigma_pixel) {
                    check_result_visual = -1;
                } else {
                    check_result_visual = 0;
                }
                score = reprojection_error_avg;
                break;               
            }
        }
    }

    if (check_result_scan == 1 || check_result_visual == 1) {
        slam_state_monitor_ptr_->UpdateCounter(UpdateType::PLUS);
    } else if (check_result_scan == -1 && check_result_visual == -1) {
        slam_state_monitor_ptr_->UpdateCounter(UpdateType::MINUS);
    }

    if (slam_state_monitor_ptr_->IsRunEnoughGood() &&
        visual_loop_type_ != loop_closure::VisualLoopType::kMapTracking) {
        visual_loop_type_ = loop_closure::VisualLoopType::kMapTracking;
        VLOG(0) << "Up loop mode on map tracking.";
    } else if (slam_state_monitor_ptr_->IsRunNotEnoughGood() &&
            visual_loop_type_ != loop_closure::VisualLoopType::kGlobal) {
        visual_loop_type_ = loop_closure::VisualLoopType::kGlobal;
        VLOG(0) << "Down loop mode on global searching.";
    } else if (slam_state_monitor_ptr_->IsRunEnoughBad()) {
        hybrid_optimizer_ptr_->ResetReloc();
        if (buffer_.last_kept_term != nullptr) {
            delete buffer_.last_kept_term;
            buffer_.last_kept_term = nullptr;
            buffer_.last_kept_blocks.clear();
        }
        std::unique_lock<std::mutex> lock(loop_result_mutex_);
        common::LoopResults().swap(loop_results_);
        visual_loop_type_ = loop_closure::VisualLoopType::kGlobal;
        LOG(WARNING) << "Bad loop state, reset reloc.";
    }

    return score;
}

double VinsHandler::ScoreScanLoop(const common::KeyFrame& keyframe,
                                  const aslam::Transformation& T_GtoM,
                                  const common::Grid2D* grid) {
    if (grid == nullptr) {
        return 0.0;
    }

    common::EigenVector3dVec p_LinMs;
    p_LinMs.resize(keyframe.points.points.size());
    for (size_t i = 0u; i < p_LinMs.size(); ++i) {
        Eigen::Vector3d p_LinG = keyframe.state.T_OtoG.transform(
                    keyframe.points.points[i]);
        p_LinMs[i] = T_GtoM.transform(p_LinG);
    }

    const common::MapLimits& limits = grid->limits();
    double value_sum = 0.0;
    for (size_t i = 0u; i < p_LinMs.size(); ++i) {
        const Eigen::Array2i cell_index = limits.GetCellIndex(p_LinMs[i].head<2>());
        const double value = grid->GetValue(cell_index);
        value_sum += value;
    }
    const double score = value_sum / static_cast<double>(p_LinMs.size());
    return score;
}

void VinsHandler::FeatureMerging() {
    VLOG(0) << features_ba_.size()
            << " features before feature merging.";

    for (const auto& matches : loop_matches_ba_) {
           const int keyframe_id_query = matches.first;
           CHECK(keyframe_id_to_idx_ba_.find(keyframe_id_query) != keyframe_id_to_idx_ba_.end());
           const int keyframe_idx_query = keyframe_id_to_idx_ba_.at(keyframe_id_query);
        for (const auto& match : matches.second) {
            // NOTE(chien): Only suite for mono camera system in now.
            const int keyframe_id_result = match.keyframe_id_result.vertex_id;
            const int cam_idx_result = match.keyframe_id_result.frame_index;
            const int keypoint_idx_query = match.keypoint_index_query;
            const int track_id_result = match.landmark_id_result;
            CHECK(keyframe_id_to_idx_ba_.find(keyframe_id_result) != keyframe_id_to_idx_ba_.end());
            const common::VisualFrameDataPtrVec& frame_datas_query =
                    key_frames_ba_[keyframe_idx_query].visual_datas;
            CHECK_LT(keypoint_idx_query, frame_datas_query[cam_idx_result]->track_ids.rows());
            const int track_id_query =
                    frame_datas_query[cam_idx_result]->track_ids(keypoint_idx_query);
            if (track_id_query == track_id_result) {
                // Time diff between query and result may not enough long.
                continue;
            }

            if (track_id_to_idx_ba_.find(track_id_query) == track_id_to_idx_ba_.end() ||
                    track_id_to_idx_ba_.find(track_id_result) == track_id_to_idx_ba_.end()) {
                continue;
            }
            CHECK(track_id_to_idx_ba_.find(track_id_query) != track_id_to_idx_ba_.end());
            const int feature_idx_query = track_id_to_idx_ba_.at(track_id_query);
            CHECK(track_id_to_idx_ba_.find(track_id_result) != track_id_to_idx_ba_.end());
            const int feature_idx_result = track_id_to_idx_ba_.at(track_id_result);
            common::ObservationDeq& obs_query = features_ba_[feature_idx_query]->observations;
            for (common::Observation& obs_tmp : obs_query) {
                obs_tmp.track_id = track_id_result;
            }
            common::ObservationDeq& obs_result = features_ba_[feature_idx_result]->observations;

            obs_result.insert(obs_result.end(), obs_query.begin(), obs_query.end());
            obs_query.clear();
        }
    }
    // Remove empty observation feature.
    for (auto it = features_ba_.begin(); it != features_ba_.end();) {
        common::FeaturePointPtr& feature = *it;
        if (feature->observations.empty()) {
            it = features_ba_.erase(it);
        } else {
            ++it;
        }
    }

    VLOG(0) << "After feature merging, we get " << features_ba_.size()
            << " features for batch optimization.";

    // Clear loop matches buffer for save memory.
    loop_closure::VertexKeyPointToStructureMatchListVec().swap(loop_matches_ba_);
}

void VinsHandler::OnlinePoseGraph() {
    vins_core::ParaBuffer buffer_batch;

    const auto& keyframes = (main_sensor_type_ == common::KeyFrameType::Visual) ?
        key_frames_ba_ : key_frames_virtual_;
    const auto& keyframes_id_to_idx = (main_sensor_type_ == common::KeyFrameType::Visual) ?
        keyframe_id_to_idx_ba_ : keyframe_id_to_idx_virtual_;

    common::FeaturePointPtrVec empty_features;
    vins_core::KeyframeToBuffer(keyframes, cameras_, empty_features,
                                loop_results_, T_GtoM_, &buffer_batch);

    hybrid_optimizer_ptr_->PoseGraph(keyframes, keyframes_id_to_idx,
                                     loop_results_, true/*fix_current*/,
                                     &buffer_batch, nullptr);

    vins_core::BufferToKeyframe(buffer_batch, cameras_,
                                &key_frames_show_, &empty_features, &T_GtoM_);
#if 0
    if (main_sensor_type_ != common::KeyFrameType::Visual) {
        for (size_t i = 0u; i < key_frames_show_.size(); ++i) {
            scan_loop_interface_ptr_->UpdateSubmapOrigin(
                key_frames_show_[i].state.T_OtoG, i);
        }
    }
#endif
}

void VinsHandler::PoseGraph() {
    vins_core::ParaBuffer buffer_batch;

    common::FeaturePointPtrVec empty_features;
    vins_core::KeyframeToBuffer(key_frames_ba_, cameras_, empty_features, loop_results_, T_GtoM_, &buffer_batch);
    hybrid_optimizer_ptr_->PreSetForBA();
    TIME_TIC(POSEGRAPH_OPTIMIZATION);
    hybrid_optimizer_ptr_->PoseGraph(key_frames_ba_, keyframe_id_to_idx_ba_,
                                     loop_results_, false/*fix_current*/,
                                     &buffer_batch, nullptr);
    TIME_TOC(POSEGRAPH_OPTIMIZATION);
    vins_core::BufferToKeyframe(buffer_batch, cameras_, &key_frames_ba_, &empty_features, &T_GtoM_);

    UpdatePosesForViz();
    VLOG(0) << "Pose graph completed.";
}

void VinsHandler::BatchFusion() {
    // Use all feature for batch optimization.
    for (common::FeaturePointPtr& feature : features_ba_) {
        if (feature->anchor_frame_idx != -1 &&
                feature->observations.size() > 1u) {
            feature->using_in_optimization = true;
        }
    }

    if (config_->do_outlier_rejection) {
        OutlierRejectionByReprojectionError(key_frames_ba_,
                                            keyframe_id_to_idx_ba_,
                                            features_ba_);
    }

    for (int i = 0; i < 2; ++i) {
        // Reset using time in observation.
        for (common::FeaturePointPtr& feature : features_ba_) {
            for (common::Observation& obs : feature->observations) {
                obs.used_counter = 0;
            }
        }

        std::unique_lock<std::mutex> lock(matching_grid_mutex_);
        // Perform fusion.
        vins_core::ParaBuffer buffer_batch;
        vins_core::KeyframeToBuffer(key_frames_ba_, cameras_, features_ba_, loop_results_, T_GtoM_, &buffer_batch);
        hybrid_optimizer_ptr_->PreSetForBA();
        TIME_TIC(BATCH_OPTIMIZATION);
        octomap::Pointcloud empty_point_cloud;
        hybrid_optimizer_ptr_->Solve(cameras_, config_, key_frames_ba_, keyframe_id_to_idx_ba_,
                                     features_ba_, loop_results_,
                                     live_local_submaps_ptr_->submaps().empty() ?
                                        nullptr : live_local_submaps_ptr_->submaps().front()->grid(),
                                     nullptr, &buffer_batch, nullptr);
        TIME_TOC(BATCH_OPTIMIZATION);
        vins_core::BufferToKeyframe(buffer_batch, cameras_, &key_frames_ba_, &features_ba_, &T_GtoM_);

        UpdatePosesForViz();
        VLOG(0) << "Batch fusion completed.";
    }
}

void VinsHandler::AddImage(const common::ImageData& img_data) {
    if (data_finish_ || main_sensor_type_ == common::KeyFrameType::Scan) {
        return;
    }
    
    std::unique_lock<std::mutex> lock(img_mutex_);
    if (main_sensor_type_ == common::KeyFrameType::Visual) {
        img_buffer_.push_back(img_data);
    } else if (main_sensor_type_ == common::KeyFrameType::ScanAndVisual) {
        img_l2_buffer_.push_back(img_data);
    }
}

void VinsHandler::AddDepth(const common::DepthData& depth_data) {
    if (data_finish_ || main_sensor_type_ == common::KeyFrameType::Scan) {
        return;
    }
    std::unique_lock<std::mutex> lock(depth_mutex_);
    depth_l2_buffer_.push_back(depth_data);
}

void VinsHandler::AddImu(const common::ImuData& imu_meas) {
    if (data_finish_) {
        return;
    }
    std::unique_lock<std::mutex> lock(imu_mutex_);
    imu_buffer_.push_back(imu_meas);
    if (imu_meas.timestamp_ns > imu_wait_time_ns_) {
        imu_waiter_.notify_all();
    }
}

void VinsHandler::AddOdom(const common::OdomData& odom_meas) {
    if (data_finish_) {
        return;
    }
    std::unique_lock<std::mutex> lock(odom_mutex_);
    odom_buffer_.push_back(odom_meas);
    if (odom_meas.timestamp_ns > odom_wait_time_ns_) {
        odom_waiter_.notify_all();
    }
}

void VinsHandler::AddScan(const common::SensorDataConstPtr scan_meas) {
    if (data_finish_ || main_sensor_type_ == common::KeyFrameType::Visual) {
        return;
    }
    std::unique_lock<std::mutex> lock(scan_mutex_);
    scan_buffer_.push_back(scan_meas);
}

void VinsHandler::AddGroundTruth(const common::OdomData& gt) {
    if (data_finish_) {
        return;
    }
    std::unique_lock<std::mutex> lock(gt_mutex_);
    gt_status_.push_back(gt);
}

void VinsHandler::SetAllFinish() {
    scan_mutex_.lock();
    scan_buffer_.clear();
    scan_mutex_.unlock();
    img_mutex_.lock();
    img_buffer_.clear();
    img_mutex_.unlock();
    hybrid_buffer_mutex_.lock();
    hybrid_buffer_.clear();
    hybrid_buffer_mutex_.unlock();
    fusion_mutex_.lock();
    waiting_for_fusion_keyframes_.clear();
    fusion_mutex_.unlock();
    loop_buffer_mutex_.lock();
    waiting_for_loop_buffer_.clear();
    loop_buffer_mutex_.unlock();
    data_finish_ = true;
    mapping_completed_ = true;
}

void VinsHandler::SetDataFinished() {
    data_finish_ = true;
}

void VinsHandler::SetRelocInitSuccess(const aslam::Transformation& T_GtoM_init) {
    fusion_mutex_.lock();
    T_GtoM_ = T_GtoM_init;
    hybrid_optimizer_ptr_ ->SetRelocInitSuccess();
    slam_state_monitor_ptr_->ResetCounter();
    visual_loop_type_ = loop_closure::VisualLoopType::kGlobal;
    fusion_mutex_.unlock();
    reloc_init_mutex_.lock();
    std::deque<loop_closure::LoopCandidate>().swap(reloc_init_candidates_);
    reloc_init_mutex_.unlock();
    loop_buffer_mutex_.lock();
    std::deque<loop_closure::LoopCandidatePtrOneFrame>().swap(waiting_for_loop_buffer_);
    loop_buffer_mutex_.unlock();
    LOG(INFO) << "Re-loc initialize success.";
    const Eigen::Vector3d T_GtoM_euler =
            common::QuatToEuler(T_GtoM_.getEigenQuaternion());
    VLOG(0) << "Init reloc result: " << T_GtoM_.getPosition()(0) << ", "
                                     << T_GtoM_.getPosition()(1) << ", "
                                     << T_GtoM_.getPosition()(2) << ", "
                                     << common::kRadToDeg * T_GtoM_euler(0) << ", "
                                     << common::kRadToDeg * T_GtoM_euler(1) << ", "
                                     << common::kRadToDeg * T_GtoM_euler(2);

    if (!first_reloc_init_done_) {
        first_reloc_init_done_ = true;
    }
}

bool VinsHandler::IsDataFinished() const {
    return data_finish_;
}

bool VinsHandler::IsDataAssociationDone() const {
    bool buffer_empty;
    if (main_sensor_type_ == common::KeyFrameType::Visual) {
        buffer_empty = img_buffer_.empty();
    } else {
        buffer_empty = scan_buffer_.empty();
    }
    return data_finish_ && buffer_empty;
}

bool VinsHandler::IsDataProcessingDone() const {
    bool buffer_empty = hybrid_buffer_.empty();
    if (main_sensor_type_ == common::KeyFrameType::Visual) {
        buffer_empty = buffer_empty && img_buffer_.empty();
    } else {
        buffer_empty = buffer_empty && scan_buffer_.empty();
    }
    return data_finish_ && buffer_empty;
}

bool VinsHandler::OtherThreadFinished() const {
    bool ret = true;
    for (const auto& iter : map_thread_done_) {
        if (iter.first == syscall(SYS_gettid)) {
            continue;
        }
        ret = ret && iter.second;
    }

    return ret;
}

bool VinsHandler::AllThreadFinished() const {
    bool ret = true;
    for (const auto& iter : map_thread_done_) {
        ret = ret && iter.second;
    }

    return ret;
}

bool VinsHandler::IsNewPose() {
    return have_new_pose_;
}

bool VinsHandler::GetNewMap(cv::Mat* map_ptr,
                            Eigen::Vector2d* origin_ptr,
                            common::EigenVector3dVec* map_cloud_ptr) {
    cv::Mat& occ_map = *CHECK_NOTNULL(map_ptr);
    Eigen::Vector2d& occ_origin = *CHECK_NOTNULL(origin_ptr);

    if (last_occ_map_show_ptr_ == nullptr) {
        last_occ_map_show_ptr_ = std::make_unique<std::pair<cv::Mat, Eigen::Vector2d>>(
            std::make_pair(cv::Mat(), Eigen::Vector2d()));
    }
    
    if (reloc_ && live_global_submaps_ptr_ != nullptr) {
        if (config_->realtime_update_map || last_occ_map_show_ptr_->first.empty()) {
            last_occ_map_show_ptr_->first = scan_loop_interface_ptr_->CollectLocalSubmaps(
                        live_global_submaps_ptr_->submaps(),
                        common::KeyFrames(),
                        &(last_occ_map_show_ptr_->second));
        }
    } else {
        if (key_frames_show_.empty() || HasOfflinePoseGraphComplete()) {
            last_occ_map_show_ptr_->first = scan_loop_interface_ptr_->CollectLocalSubmaps(
                        live_local_submaps_ptr_->submaps(),
                        common::KeyFrames(),
                        &(last_occ_map_show_ptr_->second));
        } else {
            last_occ_map_show_ptr_->first = scan_loop_interface_ptr_->CollectLocalSubmaps(
                        live_local_submaps_ptr_->submaps(),
                        key_frames_show_,
                        &(last_occ_map_show_ptr_->second));
        }
    }

    std::unique_lock<std::mutex> lock(viz_mutex_);
    if (last_occ_map_show_ptr_ != nullptr &&
        !(last_occ_map_show_ptr_->first.empty())) {
        occ_map = last_occ_map_show_ptr_->first;
        occ_origin = last_occ_map_show_ptr_->second;
    }
    *map_cloud_ptr = map_cloud_;
    return true;
}

bool VinsHandler::GetNewLoop(common::LoopResult* loop_result_ptr) {
    if (have_new_loop_) {
        *loop_result_ptr = last_loop_result_;
        have_new_loop_ = false;
        return true;
    } else {
        return false;
    }
}

void VinsHandler::GetNewKeyFrame(common::KeyFrame* keyframe_ptr) {
    *keyframe_ptr = last_keyframe_;
    have_new_pose_ = false;
}

void VinsHandler::GetNewTdCamera(double* td_s) {
    *td_s = key_frames_.empty() ? 0.0 : key_frames_.back().state.td_camera;
}

void VinsHandler::GetNewTdScan(double* td_s) {
    *td_s = key_frames_.empty() ? 0.0 : key_frames_.back().state.td_scan;
}

void VinsHandler::GetGtStatus(std::deque<common::OdomData>* gt_status_ptr) {
    *gt_status_ptr = gt_status_;
}

void VinsHandler::GetShowImage(common::CvMatConstPtrVec* imgs_ptr) {
    std::unique_lock<std::mutex> lock(viz_mutex_);
    *imgs_ptr = last_viz_imgs_;
}

void VinsHandler::GetLiveScan(common::EigenVector3dVec* live_scan_ptr) {
    *live_scan_ptr = last_live_scan_;
}

void VinsHandler::GetLiveCloud(common::EigenVector4dVec* live_cloud_ptr) {
    *live_cloud_ptr = last_live_cloud_;
}

void VinsHandler::GetRelocLandmarks(common::EigenVector3dVec* reloc_landmarks_ptr) {
    *reloc_landmarks_ptr = last_reloc_landmarks_;
}

void VinsHandler::GetVizEdges(common::EdgeVec* viz_edge_ptr) {
    *viz_edge_ptr = last_viz_edges_;
}

void VinsHandler::GetTGtoM(aslam::Transformation* T_GtoM_ptr) {
    *T_GtoM_ptr = T_GtoM_;
}

void VinsHandler::GetTOtoC(aslam::Transformation* T_OtoC_ptr, size_t id) {
    *T_OtoC_ptr = cameras_->get_T_BtoC(id);
}

void VinsHandler::GetTOtoS(aslam::Transformation* T_OtoS_ptr) {
    *T_OtoS_ptr = T_StoO_.inverse();
}

bool VinsHandler::HasOfflinePoseGraphComplete() {
    return offline_posegraph_completed_;
}

bool VinsHandler::HasMappingComplete() {
    return mapping_completed_;
}

bool VinsHandler::HasRelocInitSuccess() {
    return hybrid_optimizer_ptr_->HasRelocInited();
}

void VinsHandler::GetPGPoses(common::EigenMatrix4dVec* pg_poses_ptr) {
    *pg_poses_ptr = pg_poses_;
}

common::KeyFrame VinsHandler::GetLastKeyFrameCopy() {
    return last_keyframe_;
}

aslam::Transformation VinsHandler::GetTGtoMCopy() {
    return T_GtoM_;
}

void VinsHandler::SetResetPose(const aslam::Transformation& T_OtoM_prior,
                               const bool force_reset) {
    aslam::Transformation T_OtoG_last;
    if (!key_frames_.empty()) {
        T_OtoG_last = key_frames_.back().state.T_OtoG;
    }
    aslam::Transformation T_GtoM_prior = T_OtoM_prior * T_OtoG_last.inverse();
    T_GtoM_prior_ptr_.reset(new aslam::Transformation(T_GtoM_prior));
    if (force_reset) {
        T_GtoM_ = T_GtoM_prior;
    }
}

void VinsHandler::SetDockerPoseCallBack(std::function<void(aslam::Transformation)> fun_ptr) {
    docker_pose_func_ptr_ = fun_ptr;
}

void VinsHandler::SaveTrajectoryTUM(const std::string& saving_path) {
    std::ofstream f;
    f.open(saving_path.c_str());
    f << std::fixed;
    for (const common::KeyFrame& keyframe : key_frames_ba_) {
        const uint64_t timestamp_ns = keyframe.state.timestamp_ns;
        const double timestamp_s = common::NanoSecondsToSeconds(timestamp_ns);
        const Eigen::Quaterniond& Q_OtoG = keyframe.state.T_OtoG.getEigenQuaternion();
        const Eigen::Vector3d& p_OinG = keyframe.state.T_OtoG.getPosition();
        f << std::setprecision(6) << 1e9 * timestamp_s << " "
          << std::setprecision(9) << p_OinG(0) << " "
          << std::setprecision(9) << p_OinG(1) << " "
          << std::setprecision(9) << p_OinG(2) << " "
          << std::setprecision(9) << Q_OtoG.x() << " "
          << std::setprecision(9) << Q_OtoG.y() << " "
          << std::setprecision(9) << Q_OtoG.z() << " "
          << std::setprecision(9) << Q_OtoG.w() << std::endl;
    }
    f.close();
}

void VinsHandler::CreateLastLiveObject(const common::PointCloud& p_LinGs) {
    if (reloc_) {
        common::EigenVector3dVec p_LinMs(p_LinGs.points.size());
        for (size_t i = 0u; i < p_LinGs.points.size(); ++i) {
            p_LinMs[i] = T_GtoM_.transform(p_LinGs.points[i]);
        }
        last_live_scan_ = p_LinMs;
    } else {
        last_live_scan_ = p_LinGs.points;
    }

    if (kShowTrackingSubmap) {
        Eigen::Vector2d origin_tmp;
        cv::Mat tracking_map_32 = scan_loop_interface_ptr_->CollectLocalSubmaps(
                    live_local_submaps_ptr_->submaps(),
                    common::KeyFrames(),
                    &origin_tmp);
        cv::Mat tracking_map(tracking_map_32.rows, tracking_map_32.cols, CV_8UC1);
        for (int i = 0; i < tracking_map_32.rows; ++i) {
            for (int j = 0; j < tracking_map_32.cols; ++j) {
                const float occ = tracking_map_32.at<float>(i, j);
                if (occ < 0.) {
                    tracking_map.at<uchar>(i, j) = 128;
                } else {
                    tracking_map.at<uchar>(i, j) =
                            255 - common::ProbabilityToLogOddsInteger(occ);
                }
            }
        }
        if (!tracking_map.empty()) {
            cv::imshow("tracking_submap", tracking_map);
            cv::waitKey(1);
        }
    }

    // LUT.
    std::unordered_map<int, size_t> frame_id_to_idx;
    for (size_t i = 0; i < key_frames_.size(); ++i) {
        frame_id_to_idx[key_frames_[i].keyframe_id] = i;
    }

    last_live_cloud_.clear();
    for (common::FeaturePointPtr feature : features_) {
        const int anchor_frame_idx = feature->anchor_frame_idx;
        if (anchor_frame_idx == -1) {
            continue;
        }
        const int anchor_frame_id = feature->observations[anchor_frame_idx].keyframe_id;
        const int anchor_cam_idx = feature->observations[anchor_frame_idx].camera_idx;
        const aslam::Transformation& T_OtoG_anchor =
                key_frames_[frame_id_to_idx.at(anchor_frame_id)].state.T_OtoG;
        const aslam::Transformation T_CtoG_anchor =
                T_OtoG_anchor * cameras_->get_T_BtoC(anchor_cam_idx).inverse();
        const double inv_depth = feature->inv_depth;
        const Eigen::Vector2d& anchor_keypoint = feature->observations[anchor_frame_idx].key_point;
        Eigen::Vector3d bearing_3d;
        cameras_->getCamera(anchor_cam_idx).backProject3(anchor_keypoint, &bearing_3d);
        bearing_3d << bearing_3d(0) / bearing_3d(2), bearing_3d(1) / bearing_3d(2), 1.0;
        Eigen::Vector3d p_LinG = T_CtoG_anchor.transform(bearing_3d / inv_depth);
        Eigen::Vector3d p_LinM = T_GtoM_.transform(p_LinG);
        Eigen::Vector4d p_LinM_with_opt_flag;
        p_LinM_with_opt_flag << p_LinM(0),
                                p_LinM(1),
                                p_LinM(2),
                                feature->using_in_optimization ? 1. : 0.;
        // NOTE(chien): live cloud create in map frame.
        last_live_cloud_.push_back(p_LinM_with_opt_flag);
    }
}

void VinsHandler::ProcessScan(const common::SyncedHybridSensorData* hybrid_data_ptr,
                              common::PointCloud* scan_cloud_ptr) {

    common::PointCloud& point_cloud = *CHECK_NOTNULL(scan_cloud_ptr);
    if (hybrid_data_ptr == nullptr) {
        return;
    }

    const common::SyncedHybridSensorData& hybrid_data = *hybrid_data_ptr;
    if (config_->use_scan_pointcloud) {
        point_cloud = *(common::PointCloud*)hybrid_data.scan_data.get();
    } else {
        // Scan meas compensation by motion.
        point_cloud = common::GenerateScanPoints(hybrid_data.scan_data,
                                                hybrid_data.odom_datas.back().linear_velocity.head<2>(),
                                                hybrid_data.odom_datas.back().angular_velocity(2));
    }

    common::LaserLineDetector line_detector;
    common::LinesWithId lines;
    common::InlierPointIndices inlier_indices;
    line_detector.Detect(point_cloud, &lines, &inlier_indices);
    LaserCorrectByLine(lines, inlier_indices, &point_cloud);
    point_cloud = common::AdaptiveVoxelFilter(
            common::AdaptiveVoxelFilterOptions(0.1, 200, common::kMaxRangeScan)).Filter(point_cloud);
    // Transfrom point cloud to odomerty frame.
    for (auto& pt : point_cloud.points) {
        pt = T_StoO_.transform(pt);
    }
}

void VinsHandler::LaserCorrectByLine(const common::LinesWithId& lines,
                                     const common::InlierPointIndices& inlier_indices,
                                     common::PointCloud* scan_cloud_ptr) {
    common::PointCloud& scan_cloud = *CHECK_NOTNULL(scan_cloud_ptr);

    for (size_t i = 0u; i < lines.size(); ++i) {
        // NOTE: Define line function: By + Ax + C = 0
        const Eigen::Vector3d& line = lines[i].data;
        const double A = line(0);
        const double B = line(1);
        const double C = line(2);

        for (size_t iter_p = 0u; iter_p < inlier_indices[i].size(); ++iter_p) {
             size_t idx = inlier_indices[i][iter_p];
             double x = (B * B * scan_cloud.points[idx](0) - A * B *
                     scan_cloud.points[idx](1) - A * C) / (A * A + B * B);
             double y = (A * A * scan_cloud.points[idx](1) - A * B *
                     scan_cloud.points[idx](0) - B * C )/ (A * A + B * B);

             scan_cloud.points[idx](0) = x;
             scan_cloud.points[idx](1) = y;
        }
    }
}

void VinsHandler::SaveColmapModel() {
    const std::string completed_colmap_saving_path = common::ConcatenateFilePathFrom(
        common::getRealPath(config_->map_path), "colmap");
    common::createPath(completed_colmap_saving_path);
    const std::string completed_colmap_images_path = common::ConcatenateFilePathFrom(
        common::getRealPath(config_->map_path), "colmap/images");
    common::createPath(completed_colmap_images_path);
    const std::string completed_colmap_depth_path = common::ConcatenateFilePathFrom(
        common::getRealPath(config_->map_path), "colmap/depth_filtered");
    common::createPath(completed_colmap_depth_path);
    const std::string completed_images_file_path = common::ConcatenateFilePathFrom(
        completed_colmap_saving_path, "images.txt");
    const std::string completed_trainval_poses_file_path = common::ConcatenateFilePathFrom(
        completed_colmap_saving_path, "poses.txt");
    std::ofstream ofs_images(completed_images_file_path.c_str());
    std::ofstream ofs_trainval_poses(completed_trainval_poses_file_path.c_str());
    int img_idx = 1;
    for (const auto& key_frame : key_frames_ba_) {
        if (key_frame.GetType() == common::KeyFrameType::Scan ||
            key_frame.sensor_meas.img_data.depth == nullptr) {
            continue;
        }
        const aslam::Transformation& T_OtoG = key_frame.state.T_OtoG;
        const aslam::Transformation T_CtoG = T_OtoG * cameras_->get_T_BtoC(0).inverse();
        const aslam::Transformation T_GtoC = T_CtoG.inverse();

        const std::string image_file_name = std::to_string(static_cast<int>(10e7) + img_idx) + ".png";
        const std::string completed_image_saving_path = common::ConcatenateFilePathFrom(
        completed_colmap_images_path, image_file_name);
        const std::string completed_depth_saving_path = common::ConcatenateFilePathFrom(
        completed_colmap_depth_path, image_file_name);
        cv::imwrite(completed_image_saving_path, *(key_frame.sensor_meas.img_data.images[0]));
        cv::imwrite(completed_depth_saving_path, *(key_frame.sensor_meas.img_data.depth));

        ofs_images << img_idx << " "
                    << T_GtoC.getEigenQuaternion().w() << " "
                    << T_GtoC.getEigenQuaternion().x() << " "
                    << T_GtoC.getEigenQuaternion().y() << " "
                    << T_GtoC.getEigenQuaternion().z() << " "
                    << T_GtoC.getPosition().x() << " "
                    << T_GtoC.getPosition().y() << " "
                    << T_GtoC.getPosition().z() << " "
                    << 1 << " "
                    << image_file_name << std::endl;
        ofs_images << std::endl;

        ofs_trainval_poses << T_CtoG.getTransformationMatrix() << std::endl;

        img_idx++;
    }
    ofs_images.close();
    ofs_trainval_poses.close();
    VLOG(0) << "Colmap model save completed, model path: "
            << completed_colmap_saving_path;
}

template
void VinsHandler::ConcatSensorData<common::ImuData>(
        const std::vector<common::ImuData>& meas,
        std::vector<common::ImuData>* key_meas);

template
void VinsHandler::ConcatSensorData<common::OdomData>(
        const std::vector<common::OdomData>& meas,
        std::vector<common::OdomData>* key_meas);
}


