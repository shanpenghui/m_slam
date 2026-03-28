#ifndef M_VINS_VISN_HANDLER_H_
#define M_VINS_VISN_HANDLER_H_

#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <termios.h>    
#include <thread>
#include <list>
#include <vector>
#include <deque>

#include "feature_tracker/depth_estimator.h"
#include "feature_tracker/gyro_tracker.h"
#include "hybrid_optimizer/hybrid_optimizer.h"
#include "log_common/logging_tools.h"
#include "loop_interface/scan_loop_interface.h"
#include "loop_interface/visual_loop_interface.h"
#include "occ_common/laser_line_detector.h"
#include "occ_common/live_submaps.h"
#include "octomap_core/octo_interface.h"
#include "octomap_core/octo_mapper.h"
#include "sensor_propagator/motion_checker.h"
#include "sensor_propagator/imu_propagator.h"
#include "sensor_propagator/odom_propagator.h"
#include "vins_handler/vins_monitor.h"

namespace vins_handler {

class VinsHandler
{
public:
    VinsHandler(const common::SlamConfigPtr& slam_config,
                const std::shared_ptr<common::LiveSubmaps>& scan_maps_ptr,
                const std::shared_ptr<loop_closure::SummaryMap>& summary_map_ptr,
                const std::shared_ptr<octomap::ColorOcTree>& octree_map_ptr);
    void LoadCalibParams(const std::string& path);
    bool SaveOccupancyMap();
    void Start();
    void Release();
    void ReleaseJoinableThreads();

    void SyncSensorThread();
    void FrontendThread();
    void BackendThread();
    void LoopThread();
    void PoseGraphThread();
    void RelocInitThread();
    void VocTrainingThread();
    void FeatureTrackingTestThread();

    void AddImage(const common::ImageData& img_data);
    void AddDepth(const common::DepthData& depth_data);
    void AddImu(const common::ImuData& imu_meas);
    void AddOdom(const common::OdomData& odom_meas);
    void AddScan(const common::SensorDataConstPtr scan_meas);
    void AddGroundTruth(const common::OdomData& gt);

    void SetAllFinish();
    void SetDataFinished();
    void SetRelocInitSuccess(const aslam::Transformation& T_GtoM_init);
    bool IsDataFinished() const;
    bool IsDataAssociationDone() const;
    bool IsDataProcessingDone() const;
    bool OtherThreadFinished() const;
    bool AllThreadFinished() const;
    bool IsNewPose();
    bool GetNewMap(cv::Mat* map_ptr,
                   Eigen::Vector2d* origin_ptr,
                   common::EigenVector3dVec* map_cloud_ptr);
    bool GetNewLoop(common::LoopResult* loop_result_ptr);
    void GetNewKeyFrame(common::KeyFrame* keyframe_ptr);
    void GetTGtoM(aslam::Transformation* T_GtoM_ptr);
    void GetTOtoC(aslam::Transformation* T_OtoC_ptr, size_t id);
    void GetTOtoS(aslam::Transformation* T_OtoS_ptr);
    void GetNewTdCamera(double* td_s);
    void GetNewTdScan(double* td_s);
    void GetGtStatus(std::deque<common::OdomData>* gt_status_ptr);
    void GetShowImage(common::CvMatConstPtrVec* imgs_ptr);
    void GetLiveScan(common::EigenVector3dVec* live_scan_ptr);
    void GetLiveCloud(common::EigenVector4dVec* live_cloud_ptr);
    void GetRelocLandmarks(common::EigenVector3dVec* reloc_landmarks_ptr);
    void GetVizEdges(common::EdgeVec* viz_edge_ptr);
    bool HasOfflinePoseGraphComplete();
    bool HasMappingComplete();
    bool HasRelocInitSuccess();
    void GetPGPoses(common::EigenMatrix4dVec* pg_poses_ptr);
    void SaveTrajectoryTUM(const std::string& saving_path);
    void SetResetPose(const aslam::Transformation& T_OtoM_prior,
                      const bool force_reset);
    void SetDockerPoseCallBack(std::function<void(aslam::Transformation)> fun_ptr);
    common::KeyFrame GetLastKeyFrameCopy(); 
    aslam::Transformation GetTGtoMCopy();
    
private:
    template <typename SensorType>
    void GetFrontData(std::deque<SensorType>* sensor_buffer_ptr,
                      std::mutex* mutex_ptr,
                      SensorType* get_data_ptr);
    template <typename SensorType>
    bool FindSyncCameraData(const uint64_t time_query_ns,
                            const double time_diff_threshold_s,
                            std::deque<SensorType>* sensor_buffer_ptr,
                            SensorType* find_data_ptr);
    void SyncSensorData();
    void CollectImageDataOnly();
    bool EnoughMotionCheck(const common::State& prev_state,
                             const common::State& curr_state);
    template <typename DataType>
    void ConcatSensorData(const std::vector<DataType>& meas,
                           std::vector<DataType>* key_meas);
    bool InitializeStateByGt(const uint64_t timestamp_ns,
                             common::State* state_ptr);
    double TrackFeature(const common::SyncedHybridSensorData& hybrid_data,
                        common::KeyFrame* key_frame_k_ptr,
                        common::KeyFrame* key_frame_kp1_ptr,
                        std::vector<common::FrameToFrameMatchesWithScore>* matches_vec_ptr);
    void CreateObservations(const common::KeyFrame& key_frame,
                            common::ObservationDeq* observations_ptr);
    void FeatureTriangulation(common::KeyFrame* key_frame_ptr);
    void FeatureReTriangulation();
    void TriangulateSubVec(const size_t start_idx,
                           const size_t end_idx,
                           int* successful_counter_ptr,
                           std::mutex* mutex_ptr);
    double CheckTrackingDiff(const common::KeyFrame& key_frame_k,
                             const common::KeyFrame& key_frame_kp1);
    void ComputeAvgRelocReprojectionError(const common::LoopResult& loop_result,
                                            const aslam::Transformation& T_OtoG,
                                            const aslam::Transformation& T_GtoM,
                                            double* reprojection_error_avg_ptr);
    void OutlierRejectionByReprojectionError(const common::KeyFrames& key_frames,
                                            const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                            const common::FeaturePointPtrVec& feature_points);
    void SelectRelocFeatures(const common::KeyFrames& key_frames,
                             const aslam::Transformation& T_GtoM,
                             const bool do_outlier_rejection,
                             common::LoopResults* loop_results_ptr);
    void SelectFeatures(const std::unordered_map<int, size_t>& keyframe_id_to_idx);
    void LoopQuery(loop_closure::LoopCandidate* candidate_ptr);
    void LoopQuery(const loop_closure::LoopCandidatePtrOneFrame& candidates);
    bool TryInitImuState(const common::OdomDatas& odom_datas,
                         const common::ImuDatas& imu_datas,
                         common::State* state_ptr);
    void TryInitImuState(common::KeyFrame* keyframe_ptr);
    void TryInitReloc();
    void TryInitRelocWithPrior();
    void TuneMapPose(const std::shared_ptr<common::Submap2D>& submap,
                     const common::KeyFrame& key_frame,
                     aslam::Transformation* T_GtoM_ptr);
    bool FindCandidateByPrior(const common::KeyFrame& keyframe,
                              const aslam::Transformation& T_OtoM_prior,
                              common::LoopResult* loop_result_ptr);
    bool FindCandidateBF(const common::KeyFrame& keyframe,
                         common::LoopResult* loop_result_ptr);
    void ScanMatching(const std::shared_ptr<common::Submap2D>& submap,
                      common::KeyFrame* key_frame_ptr);
    void ScanMatching(const std::shared_ptr<common::Submap2D>& submap,
                      const common::KeyFrame& key_frame,
                      aslam::Transformation* T_GtoM_ptr);
    void RemoveDynamicObject(
            const std::shared_ptr<common::Submap2D>& submap,
            common::KeyFrame* key_frame_ptr);
    void Fusion();
    void ProcessDepthCloud(const common::ImageData& img_data,
                          octomap::OctoMapper* octo_mapper_ptr,
                          std::deque<common::PointCloudWithTimeStamp>* depth_cloud_buffer_ptr);
    void OctoMapping(const common::OctoMappingInput& pose_and_depth,
                    const bool do_castray,
                    octomap::OctoMapper* octo_mapper_ptr,
                    common::OctomapKeySetPairs* octomap_keysets_ptr);
    void OctoMappingMult();
    std::shared_ptr<common::Submap2D> InsertRangeData(
            const common::KeyFrame& range_data,
            const int max_submap_size,
            common::PointCloud* pc_global_ptr);
    void InsertRangeDataInGlobalMap(
            const common::KeyFrame& range_data,
            const aslam::Transformation& T_GtoM);
    double UpdateSlamState();
    double ScoreScanLoop(const common::KeyFrame& keyframe,
                         const aslam::Transformation& T_GtoM,
                         const common::Grid2D* grid);
    void FeatureMerging();
    void OnlinePoseGraph();
    void PoseGraph();
    void BatchFusion();
    void UpdateFeatures(const int pop_frame_id);
    void UpdatePosesForViz();
    void CreateLastLiveObject(const common::PointCloud& pc_global);
    void ProcessScan(const common::SyncedHybridSensorData* hybrid_data_ptr,
                     common::PointCloud* scan_cloud_ptr);
    void LaserCorrectByLine(const common::LinesWithId& lines,
                            const common::InlierPointIndices& inlier_indices,
                            common::PointCloud* scan_cloud_ptr);
    void ReCastrayAllRangeData();
    void ComputeVisualReprojectionErrorAndShow(const double* reloc_avg_error_pixel_ptr);
    void SaveColmapModel();

    common::KeyFrameType main_sensor_type_;

    common::TuningMode tuning_mode_;

    aslam::Transformation T_StoO_;

    aslam::Transformation T_GtoM_;

    std::unique_ptr<std::thread> sync_data_thread_;

    std::unique_ptr<std::thread> frontend_thread_;

    std::unique_ptr<std::thread> backend_thread_;

    std::unique_ptr<std::thread> loop_thread_;

    std::unique_ptr<std::thread> posegraph_thread_;

    std::unique_ptr<std::thread> reloc_init_thread_;

    std::unique_ptr<std::thread> voc_training_thread_;

    std::unique_ptr<std::thread> feature_tracking_testing_thread_;

    std::map<long, bool> map_thread_done_;

    std::unique_ptr<vins_core::ImuPropagator> imu_propagator_ptr_;

    std::unique_ptr<vins_core::OdomPropagator> odom_propagator_ptr_;

    std::unique_ptr<vins_core::MotionChecker> motion_checker_ptr_;

    std::unique_ptr<vins_core::FeatureTrackerBase> feature_tracker_ptr_;

    std::unique_ptr<vins_core::DepthEstimator> depth_estimator_ptr_;

    std::unique_ptr<vins_core::HybridOptimizer> hybrid_optimizer_ptr_;

    std::unique_ptr<common::LiveSubmaps> live_local_submaps_ptr_;

    std::shared_ptr<common::LiveSubmaps> live_global_submaps_ptr_;

    std::shared_ptr<common::FastCorrelativeScanMatcher> reloc_init_matcher_ptr_;

    std::unique_ptr<loop_closure::ScanLoopInterface> scan_loop_interface_ptr_;

    std::unique_ptr<loop_closure::VisualLoopInterface> visual_loop_interface_ptr_;

    std::unique_ptr<vins_handler::SlamStateMonitor> slam_state_monitor_ptr_;

    std::unique_ptr<octomap::OctomapInterface> octomap_interface_ptr_;

    std::unique_ptr<octomap::OctoMapper> full_octo_mapper_ptr_;

    std::unique_ptr<aslam::Transformation> T_GtoM_prior_ptr_;
    
    std::unique_ptr<loop_closure::LoopCandidate> last_frame_to_frame_loop_candidate_ptr_;

    typedef std::pair<cv::Mat, Eigen::Vector2d> OccupancyMapShow;

    std::unique_ptr<OccupancyMapShow> last_occ_map_show_ptr_;

    std::function<void(aslam::Transformation)> docker_pose_func_ptr_;
    
    loop_closure::VisualLoopType visual_loop_type_;

    const common::SlamConfigPtr config_;

    aslam::NCamera::Ptr cameras_;

    std::mutex imu_mutex_;
    std::mutex odom_mutex_;
    std::mutex img_mutex_;
    std::mutex depth_mutex_;
    std::mutex scan_mutex_;
    std::mutex gt_mutex_;
    std::mutex hybrid_buffer_mutex_;
    std::mutex keyframe_buffer_mutex_;
    std::mutex loop_buffer_mutex_;
    std::mutex loop_result_mutex_;
    std::mutex fusion_mutex_;
    std::mutex matching_grid_mutex_;
    std::mutex reloc_init_mutex_;
    std::mutex viz_mutex_;

    uint64_t odom_wait_time_ns_;
    uint64_t imu_wait_time_ns_;
    std::condition_variable odom_waiter_;
    std::condition_variable imu_waiter_;

    std::deque<common::SyncedHybridSensorData> hybrid_buffer_;
    std::deque<common::ImuData> imu_buffer_;
    std::deque<common::OdomData> odom_buffer_;
    std::deque<common::ImageData> img_buffer_;
    std::deque<common::ImageData> img_l2_buffer_;
    std::deque<common::DepthData> depth_buffer_;
    std::deque<common::DepthData> depth_l2_buffer_;
    std::deque<common::SensorDataConstPtr> scan_buffer_;
    std::deque<common::OdomData> gt_status_;
    common::ImuDatas imu_datas_assosiated_;
    common::OdomDatas odom_datas_assosiated_;
    common::KeyFrames waiting_for_fusion_keyframes_;
    std::deque<loop_closure::LoopCandidatePtrOneFrame> waiting_for_loop_buffer_;

    std::vector<common::OdomDatas> odom_datas_for_init_;
    std::vector<common::ImuDatas> imu_datas_for_init_;

    common::KeyFrames key_frames_;

    common::KeyFrames key_frames_virtual_;

    common::KeyFrames key_frames_show_;

    common::KeyFrames key_frames_ba_;

    std::unordered_map<int, size_t> keyframe_id_to_idx_virtual_;

    std::unordered_map<int, size_t> keyframe_id_to_idx_ba_;

    common::FeaturePointPtrVec features_;

    common::FeaturePointPtrVec features_ba_;

    std::unordered_map<int, size_t> track_id_to_idx_ba_;

    common::LoopResults loop_results_;

    vins_core::ParaBuffer buffer_;

    common::State last_state_;

    common::KeyFrame last_keyframe_;

    std::deque<loop_closure::LoopCandidate> reloc_init_candidates_;

    loop_closure::VertexKeyPointToStructureMatchListVec loop_matches_ba_;

    common::CvMatConstPtrVec last_viz_imgs_;

    common::EigenVector3dVec last_live_scan_;

    common::EigenVector4dVec last_live_cloud_;

    common::EigenVector3dVec last_reloc_landmarks_;

    common::EigenVector3dVec map_cloud_;

    common::LoopResult last_loop_result_;

    common::EdgeVec last_viz_edges_;

    common::EigenMatrix4dVec pg_poses_;

    uint64_t init_success_time_ns_;

    double loop_time_interval_s_;

    double last_loop_time_s_;

    int continuous_sleep_time_ms_;

    int track_id_provider_;

    int keyframe_id_provider_;

    int new_online_loop_counter_;

    bool reloc_;

    bool call_reloc_init_;

    bool call_posegraph_;

    bool sys_inited_;

    bool visual_map_loaded_;

    bool first_reloc_init_done_;

    bool have_new_pose_;

    bool have_new_loop_;

    bool offline_posegraph_completed_;

    bool mapping_completed_;

    bool data_finish_;

    bool has_docker_pose_;
};

}

#endif
