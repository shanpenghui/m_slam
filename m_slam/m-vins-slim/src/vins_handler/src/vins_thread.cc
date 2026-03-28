#include "vins_handler/vins_handler.h"
#include <syscall.h>

#include "file_common/binary_serialization.h"
#include "time_common/time.h"
#include "time_common/time_table.h"
#include "summary_map/map_saver.h"
#include "vins_handler/map_tool.h"

namespace vins_handler {

void VinsHandler::Start() {
    sync_data_thread_.reset(new std::thread(&VinsHandler::SyncSensorThread, this));

    if (config_->voc_train_mode) {
        voc_training_thread_.reset(
                    new std::thread(&VinsHandler::VocTrainingThread, this));
    } else if (config_->feature_tracking_test_mode) {
        feature_tracking_testing_thread_.reset(
                    new std::thread(&VinsHandler::FeatureTrackingTestThread, this));
    } else {
        frontend_thread_.reset(new std::thread(&VinsHandler::FrontendThread, this));
        loop_thread_.reset(new std::thread(&VinsHandler::LoopThread, this));
        if (config_->mapping && tuning_mode_ == common::TuningMode::Online) {
            posegraph_thread_.reset(new std::thread(&VinsHandler::PoseGraphThread, this));
        }
        backend_thread_.reset(new std::thread(&VinsHandler::BackendThread, this));
    }

    if (reloc_) {
        reloc_init_thread_.reset(new std::thread(&VinsHandler::RelocInitThread, this));
    }
}

void VinsHandler::Release() {
    ReleaseJoinableThreads();
    VLOG(0) << "All thread in vins handler has been released.";
}

void VinsHandler::SyncSensorThread() {
    LOG(INFO) << "Sync data thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;
    
    while (true) {
        if (config_->voc_train_mode || config_->feature_tracking_test_mode) {
            CollectImageDataOnly();

            if (data_finish_ && img_buffer_.empty()) {
                LOG(INFO) << "Sync data thread can close now.";
                break;
            }
        } else {
            // Sync sensor data from buffer.
            SyncSensorData();

            if ((config_->online && data_finish_) ||
                (!config_->online && IsDataAssociationDone())) {
                LOG(INFO) << "Sync data thread can close now.";
                break;
            }
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

void VinsHandler::LoopThread() {
    LOG(INFO) << "Loop closure thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;

    while (true) {
        if (!waiting_for_loop_buffer_.empty()) {
            loop_closure::LoopCandidatePtrOneFrame candidates_one_frame;
            if (config_->online) {
                std::unique_lock<std::mutex> lock(loop_buffer_mutex_);
                candidates_one_frame = waiting_for_loop_buffer_.back();
                waiting_for_loop_buffer_.clear();
            } else {
                std::unique_lock<std::mutex> lock(loop_buffer_mutex_);
                candidates_one_frame = waiting_for_loop_buffer_.front();
                waiting_for_loop_buffer_.pop_front();
            }
            LoopQuery(candidates_one_frame);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if ((config_->online && data_finish_) ||
            (!config_->online && IsDataProcessingDone())) {
            LOG(INFO) << "Loop closure thread can close now.";
            break;
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

void VinsHandler::PoseGraphThread() {
    LOG(INFO) << "Online pose graph thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;

    while (true) {
        if (call_posegraph_) {
            OnlinePoseGraph();
            call_posegraph_ = false;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if ((config_->online && data_finish_) ||
            (!config_->online && IsDataProcessingDone())) {
            LOG(INFO) << "Online pose graph thread can close now.";
            break;
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

void VinsHandler::RelocInitThread() {
    LOG(INFO) << "Reloc init thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;

    while (true) {
        if (call_reloc_init_) {
            // Try relocalization init.
            if (T_GtoM_prior_ptr_ != nullptr) {
                TryInitRelocWithPrior();
            } else {
                TryInitReloc();
            }
            call_reloc_init_ = false;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if ((config_->online && data_finish_) ||
            (!config_->online && IsDataProcessingDone())) {
            LOG(INFO) << "Reloc init thread can close now.";
            break;
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

void VinsHandler::FrontendThread() {
    LOG(INFO) << "SLAM frontend thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;

    common::State last_state_frontend;
    common::State second_last_state_frontend;
    std::shared_ptr<common::KeyFrame> last_visual_keyframe_frontend_ptr = nullptr;
    bool first_frame = true;
    while (true) {
        if (!hybrid_buffer_.empty() && waiting_for_fusion_keyframes_.empty()) {
            TIME_TIC(PROCESS_FRONTEND);
            // Update last state measurements for propagate in next step.
            if (first_frame) {
                last_state_frontend = last_state_;
                first_frame = false;
            } else {
                if (last_state_.timestamp_ns == last_state_frontend.timestamp_ns) {
                    last_state_frontend = last_state_;
                } else {
                    CHECK_EQ(last_state_.timestamp_ns,
                             second_last_state_frontend.timestamp_ns);
                    const aslam::Transformation delta_T =
                            second_last_state_frontend.T_OtoG.inverse() *
                            last_state_frontend.T_OtoG;
                    const uint64_t last_timestamp_tmp = last_state_frontend.timestamp_ns;
                    last_state_frontend = last_state_;
                    last_state_frontend.timestamp_ns = last_timestamp_tmp;
                    last_state_frontend.T_OtoG = last_state_.T_OtoG * delta_T;
                }
            }

            std::shared_ptr<common::SyncedHybridSensorData> hybrid_data_ptr;
            if (config_->online) {
                std::unique_lock<std::mutex> lock(hybrid_buffer_mutex_);
                common::ImuDatas tmp_imu_data_buffer;
                common::OdomDatas tmp_odom_data_buffer;
                for (const auto& tmp_hybrid_data : hybrid_buffer_) {
                    ConcatSensorData(tmp_hybrid_data.imu_datas, &tmp_imu_data_buffer);
                    ConcatSensorData(tmp_hybrid_data.odom_datas, &tmp_odom_data_buffer);
                }
                hybrid_data_ptr = std::make_shared<common::SyncedHybridSensorData>(hybrid_buffer_.back());
                hybrid_data_ptr->imu_datas = tmp_imu_data_buffer;
                hybrid_data_ptr->odom_datas = tmp_odom_data_buffer;
                hybrid_buffer_.clear();
            } else {
                std::unique_lock<std::mutex> lock(hybrid_buffer_mutex_);
                hybrid_data_ptr = std::make_shared<common::SyncedHybridSensorData>(hybrid_buffer_.front());
                hybrid_buffer_.pop_front();
            }
            const common::SyncedHybridSensorData& hybrid_data = *CHECK_NOTNULL(hybrid_data_ptr);

            const bool have_scan = (hybrid_data.scan_data != nullptr);
            const bool have_img = (!hybrid_data.img_data.images.empty());
            if (main_sensor_type_ == common::KeyFrameType::ScanAndVisual) {
                CHECK(have_scan || have_img) << "Must have scan or image measurements in a keyframe.";
            } else if (main_sensor_type_ == common::KeyFrameType::Scan) {
                CHECK(have_scan && !have_img) << "Must have scan measurements in a keyframe.";
            } else if (main_sensor_type_ == common::KeyFrameType::Visual) {
                CHECK(!have_scan && have_img) << "Must have visual measurements in a keyframe.";
            }  

            // Perform Odom Propagation.
            common::State cur_state = last_state_frontend;
            cur_state.timestamp_ns = hybrid_data.timestamp_ns;
            odom_propagator_ptr_->Propagate(hybrid_data.odom_datas, &cur_state);

            common::PointCloud scan_cloud;
            if (have_scan) {
                ProcessScan(&hybrid_data, &scan_cloud);
            }

            // Create new keyframe measurement.
            common::KeyFrame new_key_frame(keyframe_id_provider_++, hybrid_data,
                                           scan_cloud, cur_state);
            second_last_state_frontend = last_state_frontend;
            last_state_frontend = cur_state;

            if (have_img) {
                // Perform feature detection and tracking.
                std::vector<common::FrameToFrameMatchesWithScore> matches_vec;
                double adjacent_pixel_diff = std::numeric_limits<double>::max();
                adjacent_pixel_diff = TrackFeature(hybrid_data,
                                                  last_visual_keyframe_frontend_ptr.get(),
                                                  &new_key_frame,
                                                  &matches_vec);
                new_key_frame.SetZeroVelocityFlag(
                            adjacent_pixel_diff < config_->zero_velocity_pixel_diff);

                last_visual_keyframe_frontend_ptr = 
                    std::make_shared<common::KeyFrame>(new_key_frame);
            }

            if (config_->use_imu && !hybrid_optimizer_ptr_->HasImuInited()) {
                TryInitImuState(&new_key_frame);
            }

            const common::KeyFrameType this_keyframe_type = new_key_frame.GetType();
            CHECK(this_keyframe_type != common::KeyFrameType::InValid);

            if (this_keyframe_type == common::KeyFrameType::Scan ||
                this_keyframe_type == common::KeyFrameType::ScanAndVisual) {

                if (!(live_local_submaps_ptr_->submaps().empty()) && reloc_) {
                    TIME_TIC(REMOVE_DYNAMIC_OBJECT);
                    RemoveDynamicObject(live_local_submaps_ptr_->submaps().front(),
                                        &new_key_frame);
                    TIME_TOC(REMOVE_DYNAMIC_OBJECT);
                }

                // Perform scan matching.
                TIME_TIC(SCAN_MATCHING);
                if (!(live_local_submaps_ptr_->submaps().empty()) && config_->use_scan_matching) {
                    std::unique_lock<std::mutex> lock(matching_grid_mutex_);
                    ScanMatching(live_local_submaps_ptr_->submaps().front(),
                                &new_key_frame);
                }
                TIME_TOC(SCAN_MATCHING);
            }

            // Perform loop detection if need.
            // NOTE(chien): When tuning mode is online
            // scan loop will be run at submap finished.
            const double curr_keyframe_time_s = common::NanoSecondsToSeconds(
                        new_key_frame.state.timestamp_ns);
            const double loop_delta_time_s = curr_keyframe_time_s - last_loop_time_s_;
            loop_closure::LoopCandidatePtrOneFrame candidates;
            if (((config_->mapping && common::IsTuningOn(tuning_mode_)) || reloc_) &&
                loop_delta_time_s > loop_time_interval_s_) {
                if (this_keyframe_type == common::KeyFrameType::Visual ||
                    (this_keyframe_type == common::KeyFrameType::ScanAndVisual && reloc_)) {
                    common::LoopResult lc_result(new_key_frame.state.timestamp_ns,
                                                 new_key_frame.keyframe_id,
                                                 loop_closure::LoopSensor::kVisual,
                                                 visual_loop_type_,
                                                 new_key_frame.state.T_OtoG);
                    
                    // NOTE(chien): Import for pose grapgh.
                    lc_result.score = 0.1;

                    candidates.push_back(std::make_shared<loop_closure::LoopCandidate>(
                        std::make_pair(new_key_frame, lc_result)));
                }
                if ((this_keyframe_type == common::KeyFrameType::Scan ||
                    this_keyframe_type == common::KeyFrameType::ScanAndVisual) && reloc_) {
                    common::LoopResult lc_result(new_key_frame.state.timestamp_ns,
                                                new_key_frame.keyframe_id,
                                                loop_closure::LoopSensor::kScan,
                                                loop_closure::VisualLoopType::kGlobal,
                                                new_key_frame.state.T_OtoG);

                    candidates.push_back(std::make_shared<loop_closure::LoopCandidate>(
                        std::make_pair(new_key_frame, lc_result)));
                }

                last_loop_time_s_ = curr_keyframe_time_s;
            }
            
            if (!candidates.empty()) {
                if (config_->mapping && tuning_mode_ == common::TuningMode::Offline) {
                    LoopQuery(candidates);
                } else {
                    std::unique_lock<std::mutex> lock(loop_buffer_mutex_);
                    waiting_for_loop_buffer_.push_back(candidates);                    
                }
            }

            std::unique_lock<std::mutex> lock(keyframe_buffer_mutex_);
            waiting_for_fusion_keyframes_.push_back(new_key_frame);
            TIME_TOC(PROCESS_FRONTEND);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if ((config_->online && data_finish_) ||
            (!config_->online && IsDataProcessingDone())) {
            LOG(INFO) << "SLAM frontend thread can close now.";
            break;
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

void VinsHandler::BackendThread() {
    LOG(INFO) << "SLAM backend thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;

    while (true) {
        if (!waiting_for_fusion_keyframes_.empty()) {
            TIME_TIC(PROCESS_BACKEND);

            // Get keyframe measurement from buffer.
            keyframe_buffer_mutex_.lock();
            key_frames_.push_back(waiting_for_fusion_keyframes_.front());
            waiting_for_fusion_keyframes_.pop_front();
            keyframe_buffer_mutex_.unlock();

            const common::KeyFrameType this_keyframe_type = key_frames_.back().GetType();

            if (this_keyframe_type == common::KeyFrameType::Visual ||
                this_keyframe_type == common::KeyFrameType::ScanAndVisual) {
                // Perform feature depth estimate.
                TIME_TIC(FEATURE_TRIANGULATION);
                FeatureTriangulation(&key_frames_.back());
                TIME_TOC(FEATURE_TRIANGULATION);
            }

            // Perform hybrid sensor fusion.
            Fusion();

            // Print keyframe state.
            const uint64_t curr_time_ns = key_frames_.back().state.timestamp_ns;
            const double dt_s = common::NanoSecondsToSeconds(
                        curr_time_ns - init_success_time_ns_);
            key_frames_.back().state.Print(1, dt_s);

            if (reloc_) {
                const Eigen::Vector3d euler_GtoM =
                        common::QuatToEuler(T_GtoM_.getEigenQuaternion());
                VLOG(2) << "T_GtoM: " << dt_s << ", "
                        << T_GtoM_.getPosition()(0) << ", "
                        << T_GtoM_.getPosition()(1) << ", "
                        << T_GtoM_.getPosition()(2) << ", "
                        << common::kRadToDeg * euler_GtoM(0) << ", "
                        << common::kRadToDeg * euler_GtoM(1) << ", "
                        << common::kRadToDeg * euler_GtoM(2);
                const aslam::Transformation T_OtoM = T_GtoM_ * key_frames_.back().state.T_OtoG;
                const Eigen::Vector3d euler_OtoM =
                        common::QuatToEuler(T_OtoM.getEigenQuaternion());
                VLOG(1) << "PoseInMap: " << dt_s << ", "
                        << T_OtoM.getPosition()(0) << ", "
                        << T_OtoM.getPosition()(1) << ", "
                        << T_OtoM.getPosition()(2) << ", "
                        << common::kRadToDeg * euler_OtoM(0) << ", "
                        << common::kRadToDeg * euler_OtoM(1) << ", "
                        << common::kRadToDeg * euler_OtoM(2);
            }
            if (config_->do_ex_online_calib) {
                const Eigen::Vector3d euler_OtoC =
                        common::QuatToEuler(cameras_->get_T_BtoC(0).getEigenQuaternion());
                VLOG(2) << "T_OtoC: " << dt_s << ", "
                        << cameras_->get_T_BtoC(0).getPosition()(0) << ", "
                        << cameras_->get_T_BtoC(0).getPosition()(1) << ", "
                        << cameras_->get_T_BtoC(0).getPosition()(2) << ", "
                        << common::kRadToDeg * euler_OtoC(0) << ", "
                        << common::kRadToDeg * euler_OtoC(1) << ", "
                        << common::kRadToDeg * euler_OtoC(2);                    
            }

            std::unique_ptr<double> reloc_avg_error_pixel_ptr = nullptr;
            if (reloc_ && hybrid_optimizer_ptr_->HasRelocInited()) {
                    reloc_avg_error_pixel_ptr.reset(new double(0.0));
                    // NOTE(chien): If is scan keyframe, score is occupied rate,
                    // else if is visual keyframe, score is reprojection error.
                    *reloc_avg_error_pixel_ptr = UpdateSlamState();
            }

            // Drop state out of window.
            if (key_frames_.size() == static_cast<size_t>(config_->window_size)) {
                // Update features.
                const int pop_frame_id = key_frames_.front().keyframe_id;
                UpdateFeatures(pop_frame_id);
                
                if (reloc_ && !loop_results_.empty()) {
                    if (loop_results_.front().keyframe_id_query == key_frames_.front().keyframe_id) {
                        std::unique_lock<std::mutex> lock(loop_result_mutex_);
                        loop_results_.pop_front();
                    }
                }

                key_frames_.pop_front();
            }

            common::PointCloud p_LinGs;
            if (this_keyframe_type == common::KeyFrameType::Scan ||
                this_keyframe_type == common::KeyFrameType::ScanAndVisual) {
                // Insert range data into submap.
                if (config_->mapping && tuning_mode_ == common::TuningMode::Off) {
                    InsertRangeData(key_frames_.back(), 1/*max_submap_size*/, &p_LinGs);
                } else if (config_->mapping) {
                    std::shared_ptr<common::Submap2D> new_finished_submap =
                        InsertRangeData(key_frames_.back(), config_->max_submap_size, &p_LinGs);
                    // NOTE(chien): Important for loop.
                    const auto& submaps = live_local_submaps_ptr_->submaps();
                    for (const auto& submap : submaps) {
                        key_frames_.back().submap_id.insert(submap->submap_id_);
                    }
                    if (new_finished_submap != nullptr) {
                        scan_loop_interface_ptr_->AddFinishedSubMap(new_finished_submap);
                        if (tuning_mode_ == common::TuningMode::Online) {
                            common::KeyFrame virtual_keyframe = common::CreateVirtualKeyframe(
                                new_finished_submap, key_frames_ba_, keyframe_id_to_idx_ba_);
                            common::LoopResult lc_result(virtual_keyframe.state.timestamp_ns,
                                                        virtual_keyframe.keyframe_id,
                                                        loop_closure::LoopSensor::kScan,
                                                        loop_closure::VisualLoopType::kGlobal,
                                                        virtual_keyframe.state.T_OtoG);

                            key_frames_virtual_.push_back(virtual_keyframe);
                            common::KeyFrame tmp_keyframe(virtual_keyframe.keyframe_id, virtual_keyframe.state);
                            key_frames_show_.push_back(tmp_keyframe);
                            keyframe_id_to_idx_virtual_[key_frames_virtual_.back().keyframe_id] =
                                key_frames_virtual_.size() - 1u;

                            loop_closure::LoopCandidatePtrOneFrame candidates_one_frame;
                            candidates_one_frame.push_back(std::make_shared<loop_closure::LoopCandidate>(
                                std::make_pair(virtual_keyframe, lc_result)));
                            std::unique_lock<std::mutex> lock(loop_buffer_mutex_);
                            waiting_for_loop_buffer_.push_back(candidates_one_frame);
                        }
                    }
                } else {
                    InsertRangeData(key_frames_.back(), config_->max_submap_size, &p_LinGs);
                }

                // Update global map.
                if (reloc_ && config_->realtime_update_map) {
                    InsertRangeDataInGlobalMap(key_frames_.back(), T_GtoM_);
                }
            }

            // Visualization.
            TIME_TIC(VISULIZATION);
            CreateLastLiveObject(p_LinGs);
            ComputeVisualReprojectionErrorAndShow(reloc_avg_error_pixel_ptr.get());
            TIME_TOC(VISULIZATION);

            // Insert frame data for mapping.
            if (config_->mapping) {
                if (common::IsTuningOn(tuning_mode_) && (
                    // In now, if run on scan&visual mode, we dont use
                    // visual data to loop closure.
                    /*this_keyframe_type == common::KeyFrameType::ScanAndVisual ||*/
                    this_keyframe_type == common::KeyFrameType::Visual)) {    
                    // Insert frame data into loop interface.
                    TIME_TIC(INSERT_LOOP_DATA);
                                                                key_frames_.back().visual_datas,
                                                                features_);
                    TIME_TOC(INSERT_LOOP_DATA);
                }

                const bool copy_img = config_->do_octo_mapping;
                key_frames_ba_.push_back(common::KeyFrame(key_frames_.back(), copy_img));
                keyframe_id_to_idx_ba_[key_frames_ba_.back().keyframe_id] =
                    key_frames_ba_.size() - 1u;
                if (main_sensor_type_ == common::KeyFrameType::Visual) {
                    key_frames_show_.push_back(key_frames_ba_.back());
                }
            }

            last_state_ = key_frames_.back().state;
            last_keyframe_ = key_frames_.back();
            // NOTE(chien): Do not pub pose before first reloc init complete.
            if (first_reloc_init_done_) {
                have_new_pose_ = true;
            }

            TIME_TOC(PROCESS_BACKEND);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (OtherThreadFinished()) {
            // Perform mapping.
            if (config_->mapping && !mapping_completed_) {
                VLOG(0) << "All data processing finished, start mapping.";
                if (key_frames_ba_.empty()) {
                    LOG(ERROR) << "ERROR: Mapping data empty, save map failed!";
                    mapping_completed_ = true;
                    break;
                };

                if (common::IsTuningOn(tuning_mode_)) {
                    // Close unfinished submap.
                    for (auto& submap : live_local_submaps_ptr_->submaps()) {
                        if (!(submap->insertion_finished())) {
                            std::unique_lock<std::mutex> lock(matching_grid_mutex_);
                            submap->Finish();
                            scan_loop_interface_ptr_->AddFinishedSubMap(submap);
                        }
                    }

                    if (tuning_mode_ == common::TuningMode::Offline) {
                        if (main_sensor_type_ == common::KeyFrameType::Scan ||
                            main_sensor_type_ == common::KeyFrameType::ScanAndVisual) {
                            TIME_TIC(SCAN_BATCH_LOOP_CLOSURE);
                            // First. Detect scan loop closure.
                            VLOG(0) << "Start scan loop detecting...";                
                            scan_loop_interface_ptr_->DetectScanIntraLoop(key_frames_ba_,
                                                                        &loop_results_);
                            scan_loop_interface_ptr_->DetectScanInterLoop(key_frames_ba_,
                                                                        keyframe_id_to_idx_ba_,
                                                                        config_->scan_loop_inter_distance,
                                                                        &loop_results_);
                            scan_loop_interface_ptr_->VerifyScanLoop(key_frames_ba_,
                                                                    keyframe_id_to_idx_ba_,
                                                                    &loop_results_);
                            VLOG(0) << "Scan loop detection completed.";
                            TIME_TOC(SCAN_BATCH_LOOP_CLOSURE);
                        }
                    }
                    // Second. pose graph.
                    PoseGraph();
                    offline_posegraph_completed_ = true;

                    if (main_sensor_type_ == common::KeyFrameType::Visual) {
                        if (tuning_mode_ == common::TuningMode::Offline) {
                            // Feature re-triangulation.
                            FeatureReTriangulation();

                            // Merge feature observation.
                            FeatureMerging();

                            // Batch optimization.
                            BatchFusion();

                            if (config_->save_colmap_model) {
                                SaveColmapModel();
                            }
                        }
                    } else {
                        // Re-castray after pose graph.
                        ReCastrayAllRangeData();
                    }
#if 0
                    for (const auto& key_frame : key_frames_ba_) {
                        const double dt_s = common::NanoSecondsToSeconds(
                                    key_frame.state.timestamp_ns - init_success_time_ns_);
                        key_frame.state.Print(0, dt_s, "BA");
                    }
#endif
                    int feature_size_after_check = 0;
                    for (size_t i = 0u; i < features_ba_.size(); ++i) {
                        if (!depth_estimator_ptr_->CheckReprojectionError(cameras_,
                                                                          key_frames_ba_,
                                                                          keyframe_id_to_idx_ba_,
                                                                          *features_ba_[i])) {
                            features_ba_[i]->anchor_frame_idx = -1;
                        } else {
                            feature_size_after_check++;
                        }
                    }
                    VLOG(0) << "After visual reprojection checking, "
                            << feature_size_after_check
                            << " feature remaining.";
                }

                if (!live_local_submaps_ptr_->submaps().empty()) {
                    // NOTE: keyframes_ba will be rotated in below function.
                    if (SaveOccupancyMap()) {
                        const std::string complete_map_yaml_name =
                            common::ConcatenateFilePathFrom(
                                common::getRealPath(config_->map_path), common::kMapYamlFileName);
                        const std::string complete_scan_map_name =
                            common::ConcatenateFilePathFrom(
                                common::getRealPath(config_->map_path), common::kMapFileName);
                        CHECK(common::fileExists(complete_scan_map_name) &&
                                common::fileExists(complete_map_yaml_name));
                        VLOG(0) << "Occ map saving completed, map path: "
                                << config_->map_path;
                    }

                    common::PointCloud empty_pc;
                    CreateLastLiveObject(empty_pc);
                }

                // Save visual map.
                if (!features_ba_.empty()) {
                    const size_t start_idx = 0u;
                    const size_t end_idx = key_frames_ba_.size() - 1u;
                    loop_closure::SummaryMap summary_map;
                    summary_map.AddNewFrame(cameras_, key_frames_ba_,
                                            start_idx, end_idx, features_ba_);
                    loop_closure::MapSaver map_saver(config_->map_path);
                    map_saver.SaveMap(summary_map);
                    const std::string complete_map_ply_name =
                        common::ConcatenateFilePathFrom(
                            common::getRealPath(config_->map_path), common::kVisualMapPlyFileName);
                    summary_map.SaveMapAsPly(complete_map_ply_name);
                    const std::string complete_visual_map_name =
                        common::ConcatenateFilePathFrom(
                            common::getRealPath(config_->map_path), common::kVisualMapFileName);
                    CHECK(common::fileExists(complete_visual_map_name));
                    VLOG(0) << "Visual map saving completed, map path: "
                            << config_->map_path;
                    map_cloud_ = summary_map.GetLandmarkPositionAll();
                }

                // Octomap mapping if need.
                if (config_->do_octo_mapping) {
                    OctoMappingMult();

                    const std::string complete_octo_map_name =
                            common::ConcatenateFilePathFrom(config_->map_path,
                                                            common::kOctoMapFileName);
                    full_octo_mapper_ptr_->SaveOctoMap(complete_octo_map_name);
                    CHECK(common::fileExists(complete_octo_map_name));
                    VLOG(0) << "Octomap map saving completed.";
                }

                const std::string complete_tum_trajectory =
                        common::ConcatenateFilePathFrom(config_->log_path,
                                                        "slam.txt");
                SaveTrajectoryTUM(complete_tum_trajectory);

                std::this_thread::sleep_for(std::chrono::milliseconds(2000));

                mapping_completed_ = true;
            }
            LOG(INFO) << "SLAM backend thread can close now.";
            break;
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

void VinsHandler::VocTrainingThread() {
    LOG(INFO) << "Vocabulary training thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;

#ifdef USE_CNN_FEATURE
    int target_dimensionality = 256;
    int half_target_dimensionality = 128;
#else
    int target_dimensionality = 10;
    int half_target_dimensionality = 5;
#endif

    cv::Mat all_desc(0, target_dimensionality, CV_32FC1);
    while (true) {
        if (!hybrid_buffer_.empty()) {
            const common::SyncedHybridSensorData hybrid_data =
                    hybrid_buffer_.front();
            std::unique_lock<std::mutex> lock(hybrid_buffer_mutex_);
            hybrid_buffer_.pop_front();
            common::VisualFrameData visual_frame_data;

#ifdef USE_CNN_FEATURE
            feature_tracker_ptr_->InferFeature(hybrid_data.img_data.timestamp_ns,
                                               hybrid_data.img_data.images[0],
                                               &visual_frame_data,
                                               &track_id_provider_);
#else 
            feature_tracker_ptr_->DetectAndExtractFeature(hybrid_data.img_data.timestamp_ns,
                                                hybrid_data.img_data.images[0],
                                                &visual_frame_data,
                                                &track_id_provider_);
                                                           &visual_frame_data.projected_descriptors);
#endif

            VLOG(0) << "Detected keypoints size: " << visual_frame_data.key_points.cols()
                    << " descriptors size: " << visual_frame_data.projected_descriptors.cols()
                    << " dims: " << visual_frame_data.projected_descriptors.rows();

            cv::Mat descriptors_cv(cv::Size(visual_frame_data.projected_descriptors.rows(),
                                            visual_frame_data.projected_descriptors.cols()), CV_32FC1,
                                            visual_frame_data.projected_descriptors.data());
            
            cv::vconcat(all_desc, descriptors_cv, all_desc);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (data_finish_ && img_buffer_.empty() && hybrid_buffer_.empty()) {
            constexpr int K = 1000;
            cv::Mat all_desc_left = all_desc(cv::Rect(0, 0, half_target_dimensionality, all_desc.rows));
            cv::Mat labels_left, clusters_left;
            VLOG(0) << "Start to train left half words.";
            cv::kmeans(all_desc_left, K, labels_left, cv::TermCriteria(), 2, cv::KMEANS_PP_CENTERS, clusters_left);
            VLOG(0) << "End to train left half words.";

            cv::Mat all_desc_right = all_desc(cv::Rect(half_target_dimensionality, 
                0, half_target_dimensionality, all_desc.rows));
            cv::Mat labels_right, clusters_right;
            VLOG(0) << "Start to train right half words.";
            cv::kmeans(all_desc_right, K, labels_right, cv::TermCriteria(), 2, cv::KMEANS_PP_CENTERS, clusters_right);
            VLOG(0) << "End to train right half words.";

#ifdef USE_CNN_FEATURE
            std::string complete_new_asset_file_name = 
                common::ConcatenateFilePathFrom(config_->assets_path,
                "inverted_multi_index_quantizer_superpoint_new.dat");

            std::ofstream out_stream(complete_new_asset_file_name,
                                     std::ios_base::binary);
#else
            std::string complete_new_asset_file_name = 
                common::ConcatenateFilePathFrom(config_->assets_path,
                "inverted_multi_index_quantizer_superpoint_new.dat");

            std::ofstream out_stream(complete_new_asset_file_name,
                                     std::ios_base::binary);
#endif

            int serialized_version = 100;
            common::Serialize(serialized_version, &out_stream);
            common::Serialize(target_dimensionality, &out_stream);   
            Eigen::MatrixXf words_left =
                    Eigen::Map<common::DescriptorsMatF32>(
                        (float*)clusters_left.data, half_target_dimensionality, K);
            common::Serialize(words_left, &out_stream);
            Eigen::MatrixXf words_right =
                    Eigen::Map<common::DescriptorsMatF32>(
                        (float*)clusters_right.data, half_target_dimensionality, K);
            common::Serialize(words_right, &out_stream);
            out_stream.close();

            LOG(INFO) << "Vocabulary training thread can close now.";
            break;
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

void VinsHandler::FeatureTrackingTestThread() {
    LOG(INFO) << "Feature tracking test thread pid: " << syscall(SYS_gettid);
    map_thread_done_[syscall(SYS_gettid)] = false;

    std::unique_ptr<common::VisualFrameData> last_visual_frame_data_ptr = nullptr;
    size_t count = 0, frame_gap = 10;
    double matches_rank = 0, good_matches_size = 0;
    
    while (true) {
        if (!hybrid_buffer_.empty()) {
            // Match every frame_gap frame 
            if ((count ++) % frame_gap != 0) {
                hybrid_buffer_.pop_front();
                continue;
            }
            const common::SyncedHybridSensorData hybrid_data = hybrid_buffer_.front();
            common::VisualFrameData visual_frame_data_kp1;

            LOG(INFO) << "----------------------------------------";
            TIME_TIC(FEATURE_TRACKING_TEST);
#ifdef USE_CNN_FEATURE
            LOG(INFO) << "Superpoint Mode loop number: " << matches_rank;
            feature_tracker_ptr_->InferFeature(hybrid_data.img_data.timestamp_ns,
                                               hybrid_data.img_data.images[0],
                                               &visual_frame_data_kp1,
                                               &track_id_provider_);
#else
            LOG(INFO) << "Freak Mode loop number: " << matches_rank;
            feature_tracker_ptr_->DetectAndExtractFeature(hybrid_data.img_data.timestamp_ns,
                                                hybrid_data.img_data.images[0],
                                                &visual_frame_data_kp1,
                                                &track_id_provider_);
                                                           &visual_frame_data_kp1.projected_descriptors);
#endif
            VLOG(0) << "Detected keypoints size: " << visual_frame_data_kp1.key_points.cols()
                    << " descriptors size: " << visual_frame_data_kp1.projected_descriptors.cols()
                    << " dims: " << visual_frame_data_kp1.projected_descriptors.rows();
            
            if (last_visual_frame_data_ptr == nullptr) {
                last_visual_frame_data_ptr.reset(new common::VisualFrameData(visual_frame_data_kp1));
                hybrid_buffer_.pop_front();
                continue;
            }         
            matches_rank++;

            cv::Mat descriptors_cv_kp1(cv::Size(visual_frame_data_kp1.projected_descriptors.rows(),
                                            visual_frame_data_kp1.projected_descriptors.cols()), CV_32FC1,
                                            visual_frame_data_kp1.projected_descriptors.data());

            cv::Mat descriptors_cv_k(cv::Size(last_visual_frame_data_ptr.get()->projected_descriptors.rows(),
                                last_visual_frame_data_ptr.get()->projected_descriptors.cols()), CV_32FC1,
                                last_visual_frame_data_ptr.get()->projected_descriptors.data());

            std::vector<cv::KeyPoint> key_points_k, key_points_kp1;

            // assemble current frame
            for (int i = 0; i < visual_frame_data_kp1.key_points.cols(); ++i) {
                cv::KeyPoint cv_keypoint;
                const auto& keypoints = visual_frame_data_kp1.key_points;
                cv_keypoint.pt = cv::Point2f(static_cast<float>(keypoints(X, i)),
                                             static_cast<float>(keypoints(Y, i)));
                cv_keypoint.angle = keypoints(3, i);
                cv_keypoint.size = keypoints(2, i);
                cv_keypoint.response = keypoints(4, i);
                key_points_kp1.push_back(cv_keypoint);
            }

            // assemble last frame
            for (int i = 0; i < last_visual_frame_data_ptr.get()->key_points.cols(); ++i) {
                cv::KeyPoint cv_keypoint;
                const auto& keypoints = last_visual_frame_data_ptr.get()->key_points;
                cv_keypoint.pt = cv::Point2f(static_cast<float>(keypoints(X, i)),
                                             static_cast<float>(keypoints(Y, i)));
                cv_keypoint.angle = keypoints(3, i);
                cv_keypoint.size = keypoints(2, i);
                cv_keypoint.response = keypoints(4, i);
                key_points_k.push_back(cv_keypoint);
            }

            // match
            std::vector<cv::DMatch> matches;
            cv::BFMatcher matcher;
            matcher.match(descriptors_cv_kp1, descriptors_cv_k, matches);

            // Calculate the fundamental matrix to 
            // filter the matches
            std::vector<cv::Point2f> queryPoints, refPoints;

            for (size_t i = 0; i < matches.size(); i++) {
                queryPoints.push_back(key_points_kp1[matches[i].queryIdx].pt);
                refPoints.push_back(key_points_k[matches[i].trainIdx].pt);
            }

            std::vector<uchar> inliers;
            cv::findFundamentalMat(queryPoints, refPoints, inliers, cv::FM_RANSAC);
            std::vector<cv::DMatch> goodMatches;
            for (size_t i = 0u; i < inliers.size(); i++) {
                if (inliers[i]) {
                    goodMatches.push_back(matches[i]);
                }
            } 
            good_matches_size += goodMatches.size();
            LOG(INFO) << "Average good matches : " << good_matches_size / matches_rank;

            // Visualization results
            cv::Mat result;
            cv::Mat img_k = *(last_visual_frame_data_ptr.get()->image_ptr.get());
            cv::Mat img_kp1 = *(visual_frame_data_kp1.image_ptr.get()); 
            cv::drawMatches(img_kp1, key_points_kp1, img_k, key_points_k, goodMatches, result);
            std::string completed_feature_tracking_test_path = 
                common::ConcatenateFilePathFrom(config_->feature_tracking_test_path, std::to_string(matches_rank));
            cv::imwrite(completed_feature_tracking_test_path + ".png", result);
            
            // refresh last visual frame data
            last_visual_frame_data_ptr.reset(new common::VisualFrameData(visual_frame_data_kp1));

            std::unique_lock<std::mutex> lock(hybrid_buffer_mutex_);
            hybrid_buffer_.pop_front();
            TIME_TOC(FEATURE_TRACKING_TEST);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        if (data_finish_ && img_buffer_.empty() && hybrid_buffer_.empty()) {
            LOG(INFO) << "Feature tracking test thread can close now.";
            break;
        }
    }
    map_thread_done_[syscall(SYS_gettid)] = true;
}

}