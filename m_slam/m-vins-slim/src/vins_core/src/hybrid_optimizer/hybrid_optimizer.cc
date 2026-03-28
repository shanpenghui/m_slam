#include "hybrid_optimizer/hybrid_optimizer.h"

#include "cost_function/camera_reprojection_cost.h"
#include "cost_function/imu_propagation_cost.h"
#include "cost_function/marginalization_cost_function.h"
#include "cost_function/occupied_space_cost_function.h"
#include "cost_function/odom_propagation_cost.h"
#include "cost_function/pose_in_plane_cost.h"
#include "cost_function/pose_local_parameterization.h"
#include "cost_function/relative_pose_cost.h"
#include "cost_function/switch_prior_cost.h"
#include "cost_function/zero_velocity_cost.h"

namespace vins_core {

void KeyframeToBuffer(const common::KeyFrames& key_frames,
                      const aslam::NCamera::Ptr& cameras,
                      const common::FeaturePointPtrVec& features,
                      const common::LoopResults& loop_results,
                      const aslam::Transformation& T_GtoM,
                      ParaBuffer* buffer_ptr) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    buffer.poses_estimate.resize(key_frames.size());
    buffer.velocity_estimate.resize(key_frames.size());
    buffer.bg_estimate.resize(key_frames.size());
    buffer.ba_estimate.resize(key_frames.size());
    buffer.bt_estimate.resize(key_frames.size());
    buffer.br_estimate.resize(key_frames.size());
    buffer.td_camera_estimate.resize(key_frames.size());
    buffer.td_scan_estimate.resize(key_frames.size());

    for (size_t i = 0u; i < key_frames.size(); ++i) {
            const Eigen::Matrix<double, common::kGlobalPoseSize, 1> pose_vec =
            key_frames[i].state.T_OtoG.asVector();
        buffer.poses_estimate[i] = pose_vec;
        buffer.velocity_estimate[i] = key_frames[i].state.velocity;
        buffer.bg_estimate[i] = key_frames[i].state.bg;
        buffer.ba_estimate[i] = key_frames[i].state.ba;
        buffer.bt_estimate[i] = key_frames[i].state.bt;
        buffer.br_estimate[i] = key_frames[i].state.br;
        buffer.td_camera_estimate[i] = key_frames[i].state.td_camera;
        buffer.td_scan_estimate[i] = key_frames[i].state.td_scan;
    }

    buffer.inv_depth_estimate.resize(features.size());
    for (size_t i = 0u; i < features.size(); ++i) {
        const common::FeaturePointPtr& feature = features[i];
        buffer.inv_depth_estimate[i] = feature->inv_depth;
    }

    buffer.pose_GtoM_estimate = T_GtoM.asVector();

    buffer.exs_OtoC_estimate.resize(cameras->getNumCameras());
    for (size_t i = 0u; i < cameras->getNumCameras(); ++i) {
        buffer.exs_OtoC_estimate[i] = cameras->get_T_BtoC(i).asVector();
    }

    buffer.switch_variables.resize(loop_results.size());
    for (size_t i = 0u; i < loop_results.size(); ++i) {
        buffer.switch_variables[i] = 1.0;
    }
}

void BufferToKeyframe(const ParaBuffer& buffer,
                      const aslam::NCamera::Ptr& cameras,
                      common::KeyFrames* key_frames_ptr,
                      common::FeaturePointPtrVec* features_ptr,
                      aslam::Transformation* T_GtoM_ptr) {
    common::KeyFrames& key_frames = *CHECK_NOTNULL(key_frames_ptr);
    common::FeaturePointPtrVec& features = *CHECK_NOTNULL(features_ptr);
    aslam::Transformation& T_GtoM = *CHECK_NOTNULL(T_GtoM_ptr);

    for (size_t i = 0u; i < buffer.poses_estimate.size(); ++i) {
        key_frames[i].state.T_OtoG.update(buffer.poses_estimate[i]);
        key_frames[i].state.velocity = buffer.velocity_estimate[i];
        key_frames[i].state.bg = buffer.bg_estimate[i];
        key_frames[i].state.ba = buffer.ba_estimate[i];
        key_frames[i].state.bt = buffer.bt_estimate[i];
        key_frames[i].state.br = buffer.br_estimate[i];
        key_frames[i].state.td_camera = buffer.td_camera_estimate[i];
        key_frames[i].state.td_scan = buffer.td_scan_estimate[i];
    }

    for (size_t i = 0u; i < buffer.inv_depth_estimate.size(); ++i) {
        features[i]->inv_depth = buffer.inv_depth_estimate[i];
    }

    T_GtoM.update(buffer.pose_GtoM_estimate);

    for (size_t i = 0u; i < buffer.exs_OtoC_estimate.size(); ++i) {
        aslam::Transformation T_BtoCi;
        T_BtoCi.update(buffer.exs_OtoC_estimate[i]);
        cameras->set_T_BtoC(i, T_BtoCi);
    }
}

HybridOptimizer::HybridOptimizer(
        const common::SlamConfigPtr& slam_config)
    : config_(slam_config) {
    Reset();
}

void HybridOptimizer::Reset() {
    ceres_solver_options_.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    ceres_solver_options_.jacobi_scaling = false;
    ceres_solver_options_.minimizer_progress_to_stdout = true;
    ceres_solver_options_.minimizer_type = ceres::TRUST_REGION;
    ceres_solver_options_.use_nonmonotonic_steps = false;
    ceres_solver_options_.max_num_iterations = 10;
    ceres_solver_options_.function_tolerance = 5e-3;
    ceres_solver_options_.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    ceres_solver_options_.logging_type = ceres::SILENT;

    reloc_inited_ = false;
    imu_inited_ = false;
    T_GtoM_prior_seted_ = false;
    first_optimization_ = true;
    first_marginalization_ = true;
}

void HybridOptimizer::SetRelocInitSuccess() {
    reloc_inited_ = true;
}

void HybridOptimizer::SetImuInitSuccess() {
    imu_inited_ = true;
}

void HybridOptimizer::Solve(const aslam::NCamera::Ptr& cameras,
                            const common::SlamConfigPtr& config,
                            const common::KeyFrames& key_frames,
                            const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                            const common::FeaturePointPtrVec& features,
                            const common::LoopResults& loop_results,
                            const common::Grid2D* grid_local,
                            const common::Grid2D* grid_global,
                            ParaBuffer* para_buffer_ptr,
                            common::EdgeVec* edges_ptr) {
    ParaBuffer& buffer = *CHECK_NOTNULL(para_buffer_ptr);
    CHECK_EQ(buffer.poses_estimate.size(), key_frames.size());
    std::unique_ptr<ceres::Problem> problem(new ceres::Problem);

    SetSlidingWindowProblem(cameras, config, key_frames, keyframe_id_to_idx,
                            features, loop_results, grid_local, grid_global,
                            &buffer, edges_ptr, problem.get());

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_solver_options_, problem.get(), &summary);
    VLOG(5) << summary.FullReport();
}

void HybridOptimizer::Marginalize(const aslam::NCamera::Ptr& cameras,
                                  const common::SlamConfigPtr& config,
                                  const common::KeyFrames& key_frames,
                                  const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                  const common::FeaturePointPtrVec& features,
                                  const common::LoopResults& loop_results,                         
                                  const common::Grid2D* grid_local,
                                  const common::Grid2D* grid_global,
                                  const aslam::Transformation& T_GtoM,
                                  ParaBuffer* para_buffer_ptr) {
    ParaBuffer& buffer = *CHECK_NOTNULL(para_buffer_ptr);
    CHECK_EQ(buffer.poses_estimate.size(), key_frames.size());

    schur::Problem* problem = new schur::Problem();
    std::vector<double*> blocks_to_marginalize;
    const bool have_reloc = SetSlidingWindowProblem(
        cameras, config, key_frames, keyframe_id_to_idx,
        features, loop_results, grid_local, grid_global,
        &buffer, nullptr, problem);

    if (!features.empty()) {
        for (size_t i = 0u; i < buffer.inv_depth_estimate.size(); ++i) {
            if (features[i]->using_in_optimization) {
               if (!config->fix_depth) {
                  blocks_to_marginalize.emplace_back(&(buffer.inv_depth_estimate[i]));
               }
            }
        }
    }

    if (HasRelocInited() && !T_GtoM_prior_seted_) {
        const double sigma_p = 0.05;
        const double sigma_q = 1.5 * common::kDegToRad;
        Eigen::Matrix<double, 6, 6> pose_prior_sqrt_info = Eigen::Matrix<double, 6, 6>::Identity();
        pose_prior_sqrt_info.diagonal() << 1. / sigma_p,
                                           1. / sigma_p,
                                           1. / sigma_p,
                                           1. / sigma_q,
                                           1. / sigma_q,
                                           1. / sigma_q;
        PosePriorCost* T_GtoM_prior_cost = new PosePriorCost(T_GtoM.getEigenQuaternion(),
                                                             T_GtoM.getPosition(),
                                                             pose_prior_sqrt_info);
        problem->AddResidualBlock(T_GtoM_prior_cost, nullptr, buffer.pose_GtoM_estimate.data());   
        T_GtoM_prior_seted_ = true;
    } else if (HasRelocInited() && have_reloc) {
        blocks_to_marginalize.emplace_back(buffer.pose_GtoM_estimate.data());
    }

    std::unordered_map<int64_t, double*> addr_shift;

    for (size_t i = 1u; i < buffer.poses_estimate.size(); ++i) {
        addr_shift[reinterpret_cast<int64_t>(buffer.poses_estimate[i].data())] =
            buffer.poses_estimate[i - 1u].data();
    }
    blocks_to_marginalize.emplace_back(
        buffer.poses_estimate[0].data());

    size_t start_idx = 0u;
    size_t end_idx = buffer.poses_estimate.size() - 1u;
    if (!first_marginalization_) {
        start_idx = buffer.poses_estimate.size() - 2u;
    }
    if (config->use_imu && imu_inited_) {
        addr_shift[reinterpret_cast<int64_t>(buffer.velocity_estimate[key_frames.size()-1u].data())] =
            buffer.velocity_estimate[key_frames.size()-2u].data();
        for (size_t i = start_idx; i < end_idx; ++i) {
            blocks_to_marginalize.emplace_back(
                buffer.velocity_estimate[i].data());
        }

        addr_shift[reinterpret_cast<int64_t>(buffer.bg_estimate[key_frames.size()-1u].data())] =
            buffer.bg_estimate[key_frames.size()-2u].data();
        for (size_t i = start_idx; i < end_idx; ++i) {
            blocks_to_marginalize.emplace_back(
                buffer.bg_estimate[i].data());
        }

        addr_shift[reinterpret_cast<int64_t>(buffer.ba_estimate[key_frames.size()-1u].data())] =
            buffer.ba_estimate[key_frames.size()-2u].data();
        for (size_t i = start_idx; i < end_idx; ++i) {
            blocks_to_marginalize.emplace_back(
                buffer.ba_estimate[i].data());
        }
    }

    addr_shift[reinterpret_cast<int64_t>(buffer.bt_estimate[key_frames.size()-1u].data())] =
        buffer.bt_estimate[key_frames.size()-2u].data();
    for (size_t i = start_idx; i < end_idx; ++i) {
        blocks_to_marginalize.emplace_back(
            buffer.bt_estimate[i].data());
    }

    addr_shift[reinterpret_cast<int64_t>(&buffer.br_estimate[key_frames.size()-1u])] =
        &buffer.br_estimate[key_frames.size()-2u];
    for (size_t i = start_idx; i < end_idx; ++i) {
        blocks_to_marginalize.emplace_back(
            &buffer.br_estimate[i]);
    }

   problem->CopyBlockData();
   problem->AddMarginalizeAddress(blocks_to_marginalize);
   problem->Evaluate();
   constexpr bool kLogCov = false;
   problem->Marginalize(kLogCov);
   if (kLogCov) {
       problem->PrintCov(6,
                         key_frames.back().state.timestamp_ns,
                         reinterpret_cast<int64_t>(buffer.poses_estimate[1].data()));
   }

   if (buffer.last_kept_term != nullptr) {
       delete buffer.last_kept_term;
   }
   buffer.last_kept_term = problem;
   buffer.last_kept_blocks = problem->GetParameterBlocks(addr_shift);

   if (first_marginalization_) {
       first_marginalization_ = false;
   }
}

void HybridOptimizer::PoseGraph(const common::KeyFrames& key_frames,
                                const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                const common::LoopResults& loop_results,
                                const bool fix_current,
                                ParaBuffer* buffer_ptr,
                                common::EdgeVec* edges_ptr) {
    std::unique_ptr<ceres::Problem> problem(new ceres::Problem);
    SetPoseGraphProblem(key_frames,
                        keyframe_id_to_idx,
                        loop_results,
                        fix_current,
                        buffer_ptr,
                        edges_ptr,
                        problem.get());

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_solver_options_, problem.get(), &summary);
}

void HybridOptimizer::SetPoseGraphProblem(const common::KeyFrames& key_frames,
                                           const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                           const common::LoopResults& loop_results,
                                           const bool fix_current,
                                           ParaBuffer* buffer_ptr,
                                           common::EdgeVec* edges_ptr,
                                           ceres::Problem* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    CHECK_EQ(buffer.poses_estimate.size(), key_frames.size());
    if (key_frames.empty()) {
        return;
    }

    for (size_t i = 0u; i < buffer.poses_estimate.size(); ++i) {
        ceres::LocalParameterization* local_parameterization;
        const auto& curr_keyframe_type = key_frames[i].GetType();
        local_parameterization = new Pose2DLocalParameterization();
        problem->AddParameterBlock(buffer.poses_estimate[i].data(),
                                   common::kGlobalPoseSize,
                                   local_parameterization);
    }

    const Eigen::Matrix<double, 6, 6> sqrt_info =
            Eigen::Matrix<double, 6, 6>::Identity();
    for (size_t i = 0u; i < buffer.poses_estimate.size() - 1u; ++i) {
        const aslam::Transformation& T_curr = key_frames[i].state.T_OtoG;
        const aslam::Transformation& T_next = key_frames[i + 1u].state.T_OtoG;
        const aslam::Transformation delta_T = T_curr.inverse() * T_next;
        RelativePoseCost* relative_pose_cost = new RelativePoseCost(
                    delta_T.getEigenQuaternion(),
                    delta_T.getPosition(),
                    sqrt_info);
        problem->AddResidualBlock(relative_pose_cost, nullptr,
                                  buffer.poses_estimate[i].data(),
                                  buffer.poses_estimate[i + 1u].data());
    }

    for (size_t i = 0u; i < loop_results.size(); ++i) {
        const int keyframes_id_query = loop_results[i].keyframe_id_query;
        const int keyframes_id_loop = loop_results[i].keyframe_id_result;
        if (keyframes_id_loop == -1) {
            continue;
        }
        const auto& iter_query = keyframe_id_to_idx.find(keyframes_id_query);
        const auto& iter_loop = keyframe_id_to_idx.find(keyframes_id_loop);
        if (iter_query == keyframe_id_to_idx.end()) {
            LOG(WARNING) << "Query keyframe id not find in map."
                         << "query id: " << keyframes_id_query
                         << " vs last id: " << keyframe_id_to_idx.cend()->first;
            continue;
        }
        CHECK(iter_loop != keyframe_id_to_idx.end());
        const int keyframe_idx_query = keyframe_id_to_idx.at(keyframes_id_query);
        const int keyframe_idx_loop = keyframe_id_to_idx.at(keyframes_id_loop);

        const aslam::Transformation& T_curr_lc = loop_results[i].T_estimate;
        const aslam::Transformation& T_loop = key_frames[keyframe_idx_loop].state.T_OtoG;
        const aslam::Transformation delta_T = T_curr_lc.inverse() * T_loop;
        RelativePoseCost* loop_closure_cost = new RelativePoseCost(
                    delta_T.getEigenQuaternion(),
                    delta_T.getPosition(),
                    loop_results[i].score * sqrt_info);
        problem->AddResidualBlock(loop_closure_cost, nullptr,
                                  buffer.poses_estimate[keyframe_idx_query].data(),
                                  buffer.poses_estimate[keyframe_idx_loop].data());

        auto* switch_prior_cost = CreateSwitchPriorCost(1.0 ,1.0);
        problem->AddResidualBlock(switch_prior_cost, nullptr,
                                  &buffer.switch_variables[i]);

        constexpr double kSwitchVariableMinValue = 0.0;
        constexpr double kSwitchVariableMaxValue = 1.0;
        constexpr int kSwitchVariableIndexIntoParameterBlock = 0;
        problem->SetParameterLowerBound(
            &buffer.switch_variables[i], kSwitchVariableIndexIntoParameterBlock,
            kSwitchVariableMinValue);
        problem->SetParameterUpperBound(
            &buffer.switch_variables[i], kSwitchVariableIndexIntoParameterBlock,
            kSwitchVariableMaxValue);
        
        if (edges_ptr != nullptr) {
            edges_ptr->emplace_back(buffer.poses_estimate[keyframe_idx_query].head<3>(),
                                    buffer.poses_estimate[keyframe_idx_loop].head<3>());
        }
    }

    if (fix_current) {
        problem->SetParameterBlockConstant(buffer.poses_estimate.back().data());
    } else {
        problem->SetParameterBlockConstant(buffer.poses_estimate[0].data());
    }
}

void HybridOptimizer::InitReloc(const aslam::NCamera::Ptr& cameras,
                                const common::SlamConfigPtr& config,
                                const common::KeyFrames& key_frames,
                                const common::LoopResults& reloc_results,              
                                const common::Grid2D* grid_global,
                                ParaBuffer* para_buffer_ptr) {
    ParaBuffer& buffer = *CHECK_NOTNULL(para_buffer_ptr);

    CHECK(!HasRelocInited());

    // LUT
    std::unordered_map<int, size_t> keyframe_id_to_idx;
    for (size_t i = 0u; i < key_frames.size(); ++i) {
        keyframe_id_to_idx[key_frames[i].keyframe_id] = i;
    }

    std::unique_ptr<ceres::Problem> problem(new ceres::Problem);

    for (size_t i = 0u; i < buffer.poses_estimate.size(); ++i) {
        ceres::LocalParameterization* local_parameterization;
        const auto& curr_keyframe_type = key_frames[i].GetType();
        local_parameterization = new Pose2DLocalParameterization();
        problem->AddParameterBlock(buffer.poses_estimate[i].data(),
                                   common::kGlobalPoseSize,
                                   local_parameterization);
    }
    // Now we assume that both the map and SLAM frame are on the XY plane.
    problem->AddParameterBlock(buffer.pose_GtoM_estimate.data(),
                               common::kGlobalPoseSize,
                               new Pose2DLocalParameterization());

    for (size_t i = 0u; i < reloc_results.size(); ++i) {
        if (reloc_results[i].loop_sensor == loop_closure::LoopSensor::kScan &&
            grid_global != nullptr) {
            size_t start_idx = i;
            size_t end_idx = i;
            SetGlobalOccupiedGridCost(config, key_frames, grid_global,
                                      start_idx, end_idx,
                                      &buffer, problem.get());
        }
    }

    common::EdgeVec* edges_ptr = nullptr;
    SetMapVisualReprojectionCost(cameras, config, keyframe_id_to_idx, reloc_results,
                                 &buffer, edges_ptr, problem.get());

    Eigen::Matrix2d sqrt_info_r = Eigen::Matrix2d::Zero();
    sqrt_info_r.diagonal() << 1.0 / 1e-6,
                              1.0 / 1e-6;
    const double sqrt_info_z = 1.0 / 1e-6;
    PoseInPlaneCost* pose_in_plane_cost =
            new PoseInPlaneCost(sqrt_info_r, sqrt_info_z);
    problem->AddResidualBlock(pose_in_plane_cost, nullptr,
                              buffer.pose_GtoM_estimate.data());

    for (size_t i = 0u; i < buffer.poses_estimate.size(); ++i) {
        problem->SetParameterBlockConstant(
                    buffer.poses_estimate[i].data());
    }

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_solver_options_, problem.get(), &summary);
}

void HybridOptimizer::MapPoseTuning(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const common::Grid2D* grid_global,
    ParaBuffer* para_buffer_ptr) {
    ParaBuffer& buffer = *CHECK_NOTNULL(para_buffer_ptr);

    std::unique_ptr<ceres::Problem> problem(new ceres::Problem);

    ceres::LocalParameterization* local_parameterization =
        new Pose2DLocalParameterization();
    problem->AddParameterBlock(buffer.poses_estimate.back().data(),
                               common::kGlobalPoseSize,
                               local_parameterization);
    problem->AddParameterBlock(buffer.pose_GtoM_estimate.data(),
                               common::kGlobalPoseSize,
                               local_parameterization);

    size_t start_idx = key_frames.size() - 1u;
    size_t end_idx = key_frames.size() - 1u;
    SetGlobalOccupiedGridCost(config, key_frames, grid_global,
                              start_idx, end_idx,
                              &buffer, problem.get());

    problem->SetParameterBlockConstant(buffer.poses_estimate.back().data());

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_solver_options_, problem.get(), &summary);
}

template<class ProblemType>
bool HybridOptimizer::SetSlidingWindowProblem(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx,
    const common::FeaturePointPtrVec& features,
    const common::LoopResults& loop_results,
    const common::Grid2D* grid_local,
    const common::Grid2D* grid_global,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    ProblemType* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    CHECK_EQ(buffer.poses_estimate.size(), key_frames.size());

    bool have_reloc = false;

    for (size_t i = 0u; i < buffer.poses_estimate.size(); ++i) {
        const auto& curr_keyframe_type = key_frames[i].GetType();
        ceres::LocalParameterization* local_parameterization;
        local_parameterization = new Pose2DLocalParameterization();
        problem->AddParameterBlock(buffer.poses_estimate[i].data(),
                                   common::kGlobalPoseSize,
                                   local_parameterization);
    }

    // Add odom propagation cost.
    SetOdomPropagationCost(config, key_frames,
                            buffer_ptr, edges_ptr, problem);

    if (config->use_imu && imu_inited_) {
        // Add IMU propagation cost.
        SetImuPropagationCost(config, key_frames,
                              buffer_ptr, edges_ptr, problem);
    }

    const common::KeyFrameType last_keyframe_type = key_frames.back().GetType();

    if (last_keyframe_type == common::KeyFrameType::Visual ||
        last_keyframe_type == common::KeyFrameType::ScanAndVisual) {
        // Add visual re-projection cost.
        const int visual_cost_num = SetVisualReprojectionCost(
                                cameras, config, key_frames,
                                keyframe_id_to_idx, features,
                                buffer_ptr, edges_ptr, problem);
 
        ceres::LocalParameterization* local_parameterization =
            new Pose2DLocalParameterization();
        for (size_t i = 0u; i < buffer.exs_OtoC_estimate.size(); ++i) {
            problem->AddParameterBlock(buffer.exs_OtoC_estimate[i].data(),
                                       common::kGlobalPoseSize,
                                       local_parameterization);
            if (!config_->do_ex_online_calib && visual_cost_num) {
                problem->SetParameterBlockConstant(buffer.exs_OtoC_estimate[i].data());
            }
        }
        if (HasRelocInited() && !loop_results.empty()) {
            // Add reloc cost.
            const int visual_reloc_cost_num = SetMapVisualReprojectionCost(
                                cameras, config, keyframe_id_to_idx,
                                loop_results, buffer_ptr, edges_ptr, problem);
            have_reloc = (visual_reloc_cost_num > 0);
        }
    }

    if (last_keyframe_type == common::KeyFrameType::Scan ||
        last_keyframe_type == common::KeyFrameType::ScanAndVisual) {
        if (grid_local != nullptr) {
            // Add scan occupied grid cost.
            SetLocalOccupiedGridCost(config, key_frames, grid_local,
                                     buffer_ptr, problem);
        }
        if (grid_global != nullptr && HasRelocInited()) {
            // Add scan occupied grid cost in map.
            size_t start_idx = buffer.poses_estimate.size() - 1u;
            size_t end_idx = buffer.poses_estimate.size() - 1u;
            SetGlobalOccupiedGridCost(config, key_frames, grid_global,
                                      start_idx, end_idx,
                                      buffer_ptr, problem);
            have_reloc = true;
        }
    }

    if (HasRelocInited()) {
        // Now we assume that both the map and SLAM frame are on the XY plane.
        ceres::LocalParameterization* local_parameterization =
            new Pose2DLocalParameterization();
        problem->AddParameterBlock(buffer.pose_GtoM_estimate.data(),
                                   common::kGlobalPoseSize,
                                   local_parameterization);
    }

    // Add pose in plane cost.
    Eigen::Matrix2d sqrt_info_r = Eigen::Matrix2d::Zero();
    sqrt_info_r.diagonal() << 1.0 / 1e-6,
                              1.0 / 1e-6;
    const double sqrt_info_z = 1.0 / 1e-6;
    PoseInPlaneCost* pose_in_plane_cost =
            new PoseInPlaneCost(sqrt_info_r, sqrt_info_z);
    problem->AddResidualBlock(pose_in_plane_cost, nullptr,
                              buffer.poses_estimate.back().data());

    // Add marginalization cost.
    if (!first_optimization_ && buffer.last_kept_term != nullptr) {
        MarginalizationCost* marginalization_cost_function =
                new MarginalizationCost(buffer.last_kept_term);
        problem->AddResidualBlock(marginalization_cost_function, nullptr,
                                  buffer.last_kept_blocks);
    }

    if (first_optimization_) {
        problem->SetParameterBlockConstant(buffer.poses_estimate[0u].data());
        problem->SetParameterBlockConstant(buffer.bt_estimate[0u].data());
        problem->SetParameterBlockConstant(&buffer.br_estimate[0u]);
        if (config->use_imu && imu_inited_) {
            problem->SetParameterBlockConstant(buffer.velocity_estimate[0u].data());
            problem->SetParameterBlockConstant(buffer.bg_estimate[0u].data());
            problem->SetParameterBlockConstant(buffer.ba_estimate[0u].data());
        }
        first_optimization_ = false;
    }

    return have_reloc;
}

template<class ProblemType>
void HybridOptimizer::SetImuPropagationCost(const common::SlamConfigPtr& config,
                                            const common::KeyFrames& key_frames,
                                            ParaBuffer* buffer_ptr,
                                            common::EdgeVec* edges_ptr,
                                            ProblemType* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    CHECK_EQ(buffer.poses_estimate.size(), key_frames.size());

    size_t start_idx = 0u;
    size_t end_idx = buffer.poses_estimate.size() - 1u;
    if (!first_marginalization_) {
        start_idx = buffer.poses_estimate.size() - 2u;
    }
    int imu_cost_counter = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
        CHECK(!(key_frames[i+1].sensor_meas.imu_datas.empty()));
        CHECK_EQ(key_frames[i].state.timestamp_ns,
                 key_frames[i+1].sensor_meas.imu_datas.front().timestamp_ns);
        CHECK_EQ(key_frames[i+1].state.timestamp_ns,
                 key_frames[i+1].sensor_meas.imu_datas.back().timestamp_ns);
        ImuPropagationCost* imu_propagation_cost =
                new ImuPropagationCost(config,
                                       key_frames[i + 1u].sensor_meas.imu_datas);
        problem->AddResidualBlock(imu_propagation_cost, nullptr,
                                  buffer.poses_estimate[i].data(),
                                  buffer.poses_estimate[i + 1u].data(),
                                  buffer.velocity_estimate[i].data(),
                                  buffer.velocity_estimate[i + 1u].data(),
                                  buffer.bg_estimate[i].data(),
                                  buffer.bg_estimate[i + 1u].data(),
                                  buffer.ba_estimate[i].data(),
                                  buffer.ba_estimate[i + 1u].data());
        imu_cost_counter++;
        if (edges_ptr != nullptr) {
            edges_ptr->emplace_back(buffer.poses_estimate[i].head<3>(),
                                    buffer.poses_estimate[i + 1u].head<3>());
        }

        if (key_frames[i + 1u].zero_vm) {
            Eigen::Matrix<double, 3, 3> sqrt_info = Eigen::Matrix<double, 3, 3>::Identity();
            sqrt_info.diagonal() << 1, 1., 1.;
            ZeroVelocityCost* velocity_static_cost = new ZeroVelocityCost(sqrt_info);
            problem->AddResidualBlock(velocity_static_cost, nullptr,
                                       buffer.velocity_estimate[i].data());
            VLOG(2) << "Suspected static state, add zero velocity cost in frame: "
                    << key_frames[i].keyframe_id;
        }
    }
    VLOG(2) << "IMU cost num: " << imu_cost_counter;
}

template<class ProblemType>
void HybridOptimizer::SetOdomPropagationCost(const common::SlamConfigPtr& config,
                                              const common::KeyFrames& key_frames,
                                              ParaBuffer* buffer_ptr,
                                              common::EdgeVec* edges_ptr,
                                              ProblemType* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    CHECK_EQ(buffer.poses_estimate.size(), key_frames.size());

    size_t start_idx = 0u;
    size_t end_idx = buffer.poses_estimate.size() - 1u;
    if (!first_marginalization_) {
        start_idx = buffer.poses_estimate.size() - 2u;
    }
    int odom_cost_counter = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
        CHECK(!(key_frames[i+1].sensor_meas.odom_datas.empty()));
        CHECK_EQ(key_frames[i].state.timestamp_ns,
                 key_frames[i+1].sensor_meas.odom_datas.front().timestamp_ns);
        CHECK_EQ(key_frames[i+1].state.timestamp_ns,
                 key_frames[i+1].sensor_meas.odom_datas.back().timestamp_ns);
        OdomPropagationCost* odom_propagation_cost =
                new OdomPropagationCost(config,
                                        key_frames[i + 1u].sensor_meas.odom_datas);
        problem->AddResidualBlock(odom_propagation_cost, nullptr,
                                  buffer.poses_estimate[i].data(),
                                  buffer.poses_estimate[i + 1u].data(),
                                  buffer.bt_estimate[i].data(),
                                  buffer.bt_estimate[i + 1u].data(),
                                  &buffer.br_estimate[i],
                                  &buffer.br_estimate[i + 1u]);
        odom_cost_counter++;

        if (edges_ptr != nullptr) {
            edges_ptr->emplace_back(buffer.poses_estimate[i].head<3>(),
                                    buffer.poses_estimate[i + 1u].head<3>());
        }
    }
    VLOG(2) << "Odom cost num: " << odom_cost_counter;
}

template<class ProblemType>
int HybridOptimizer::SetVisualReprojectionCost(const aslam::NCamera::Ptr& cameras,
                                               const common::SlamConfigPtr& config,
                                               const common::KeyFrames& key_frames,
                                               const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                               const common::FeaturePointPtrVec& features,
                                               ParaBuffer* buffer_ptr,
                                               common::EdgeVec* edges_ptr,
                                               ProblemType* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    CHECK_EQ(buffer.poses_estimate.size(), key_frames.size());

    int visual_cost_counter = 0;
    for (size_t i = 0u; i < features.size(); ++i) {
        if (features[i]->using_in_optimization) {
            bool check_using_in_optimization = false;
            common::ObservationDeq& observations = features[i]->observations;
            const int anchor_frame_idx = features[i]->anchor_frame_idx;
            if (anchor_frame_idx == -1) {
                continue;
            }

            // Sliding window BA, use relative position for optimization.
            const int anchor_frame_id = observations[anchor_frame_idx].keyframe_id;
            const int anchor_cam_idx = observations[anchor_frame_idx].camera_idx;
            auto iter_anchor = keyframe_id_to_idx.find(anchor_frame_id);
            CHECK(iter_anchor != keyframe_id_to_idx.end());
            const aslam::Transformation& T_OtoC = cameras->get_T_BtoC(anchor_cam_idx);
            const aslam::Transformation T_CtoG_anchor =
                    key_frames[iter_anchor->second].state.T_OtoG * T_OtoC.inverse();
            const double inv_depth = features[i]->inv_depth;
            Eigen::Vector3d bearing_3d;
            const Eigen::Vector2d& anchor_keypoint = observations[anchor_frame_idx].key_point;
            cameras->getCamera(anchor_cam_idx).backProject3(anchor_keypoint, &bearing_3d);
            bearing_3d << bearing_3d(0) / bearing_3d(2), bearing_3d(1) / bearing_3d(2), 1.0;
            const Eigen::Vector3d p_LinG = T_CtoG_anchor.transform(bearing_3d / inv_depth);
            const Eigen::Vector2d& anchor_velocity = observations[anchor_frame_idx].velocity;
            if (edges_ptr != nullptr) {
                edges_ptr->emplace_back(p_LinG,
                                        buffer.poses_estimate[iter_anchor->second].head<3>());
            }
            std::unordered_set<int> used_keyframe_id;
            for (size_t j = 0u; j < observations.size(); ++j) {
                const int curr_frame_id = observations[j].keyframe_id;
                if (observations[j].used_counter == 2 || curr_frame_id == anchor_frame_id ||
                        used_keyframe_id.count(curr_frame_id)) {
                    continue;
                }
                const int curr_cam_idx = observations[j].camera_idx;
                const Eigen::Vector2d& curr_keypoint = observations[j].key_point;
                const Eigen::Vector2d& curr_velocity = observations[j].velocity;
                const auto iter_current = keyframe_id_to_idx.find(curr_frame_id);
                CHECK(iter_current != keyframe_id_to_idx.end());
                CHECK_NE(iter_anchor->second, iter_current->second);

                const bool use_depth = observations[j].depth != common::kInValidDepth;
                double depth_meas = 0.0;
                double depth_sigma = 0.0;
                if (use_depth) {
                    depth_meas = observations[j].depth;
                    depth_sigma = depth_meas * depth_meas * config->depth_noise_params(0) +
                                  depth_meas * config->depth_noise_params(1) +
                                  config->depth_noise_params(2);
                }

                CHECK_EQ(anchor_cam_idx, curr_cam_idx)
                    << "Only support visual reprojection computation on the same camera in now.";

                const double meas_td_anchor = buffer.td_camera_estimate[iter_anchor->second];
                const double meas_td_current = buffer.td_camera_estimate[iter_current->second];
                CameraReprojectionCost* cam_reprojection_cost =
                        new CameraReprojectionCost(cameras, anchor_keypoint, curr_keypoint,
                                                  anchor_velocity, curr_velocity, use_depth,
                                                  anchor_cam_idx, curr_cam_idx, depth_meas,
                                                  meas_td_anchor, meas_td_current,
                                                  config_->visual_sigma_pixel, depth_sigma);
                if (anchor_cam_idx == curr_cam_idx) {
                    problem->AddResidualBlock(cam_reprojection_cost, new ceres::CauchyLoss(1.0),
                                            &buffer.inv_depth_estimate[i],
                                            buffer.poses_estimate[iter_anchor->second].data(),
                                            buffer.poses_estimate[iter_current->second].data(),
                                            &buffer.td_camera_estimate[iter_current->second],
                                            buffer.exs_OtoC_estimate[anchor_cam_idx].data());
                } else {
                    problem->AddResidualBlock(cam_reprojection_cost, new ceres::CauchyLoss(1.0),
                                            &buffer.inv_depth_estimate[i],
                                            buffer.poses_estimate[iter_anchor->second].data(),
                                            buffer.poses_estimate[iter_current->second].data(),
                                            &buffer.td_camera_estimate[iter_current->second],
                                            buffer.exs_OtoC_estimate[anchor_cam_idx].data(),
                                            buffer.exs_OtoC_estimate[curr_cam_idx].data());
                }

                if (!config->do_time_online_calib) {
                    problem->SetParameterBlockConstant(&buffer.td_camera_estimate[iter_current->second]);
                }

                visual_cost_counter++;
                if (edges_ptr != nullptr) {
                    edges_ptr->emplace_back(p_LinG,
                                            buffer.poses_estimate[iter_current->second].head<3>());
                }

                observations[j].used_counter++;
                used_keyframe_id.emplace(observations[j].keyframe_id);
                check_using_in_optimization = true;
            }

            if (config->fix_depth) {
                problem->SetParameterBlockConstant(&buffer.inv_depth_estimate[i]);
            }
            CHECK_EQ(check_using_in_optimization, true);
        }
    }

    VLOG(2) << "Visual cost num: " << visual_cost_counter;

    return visual_cost_counter;
}

template<class ProblemType>
int HybridOptimizer::SetMapVisualReprojectionCost(const aslam::NCamera::Ptr& cameras,
                                                   const common::SlamConfigPtr& config,
                                                   const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                                   const common::LoopResults& reloc_results,
                                                   ParaBuffer* buffer_ptr,
                                                   common::EdgeVec* edges_ptr,
                                                   ProblemType* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);

    int reloc_cost_counter = 0;
    for (const common::LoopResult& reloc_result : reloc_results) {
        auto itor_query = keyframe_id_to_idx.find(reloc_result.keyframe_id_query);
        if (itor_query == keyframe_id_to_idx.end()) {
            continue;
        }
        const std::vector<std::pair<int, bool>>& inlier_indices = reloc_result.pnp_inliers;
        const Eigen::Matrix2Xd& keypoints = reloc_result.keypoints;
        const Eigen::Matrix3Xd& p_LinMs = reloc_result.positions;
        const Eigen::VectorXd& depths = reloc_result.depths;
        const Eigen::VectorXi& cam_indices = reloc_result.cam_indices;
        CHECK_EQ(keypoints.cols(), p_LinMs.cols());
        CHECK_EQ(keypoints.cols(), cam_indices.rows());
        CHECK_EQ(keypoints.cols(), depths.rows());
        for (const auto& inlier_idx : inlier_indices) {
            const bool is_using_for_opt = inlier_idx.second;
            if (!is_using_for_opt) {
                continue;
            }
            CHECK_LT(inlier_idx.first, keypoints.cols());
            const Eigen::Vector2d keypoint = keypoints.col(inlier_idx.first);
            const Eigen::Vector3d p_LinM = p_LinMs.col(inlier_idx.first);
            const int cam_idx = cam_indices(inlier_idx.first);
            const double reloc_sigma = config->visual_sigma_pixel;
            constexpr double kUnuseDepthDefaultMeas = -1.0;
            bool use_depth = false;
            double depth_meas = kUnuseDepthDefaultMeas;
            double depth_sigma = 0.0;
            if (depths(inlier_idx.first) != common::kInValidDepth) {
                use_depth = true;
                depth_meas = depths(inlier_idx.first);
                depth_sigma = depth_meas * depth_meas * config->depth_noise_params(0) +
                              depth_meas * config->depth_noise_params(1) +
                              config->depth_noise_params(2);
            }
            CameraReprojectionFromMapCost* cam_reprojection_from_map_cost =
                    new CameraReprojectionFromMapCost(cameras,
                                                      p_LinM,
                                                      keypoint,
                                                      cam_idx,
                                                      use_depth,
                                                      depth_meas,
                                                      reloc_sigma,
                                                      depth_sigma);
            problem->AddResidualBlock(cam_reprojection_from_map_cost, new ceres::CauchyLoss(1.0),
                                      buffer.pose_GtoM_estimate.data(),
                                      buffer.poses_estimate[itor_query->second].data());
            if (edges_ptr != nullptr) {
                edges_ptr->emplace_back(p_LinM,
                                        buffer.poses_estimate[itor_query->second].head<3>());
            }
            reloc_cost_counter++;
        }

    }
    VLOG(2) << "Reloc cost num: " <<reloc_cost_counter;
    return reloc_cost_counter;
}

template<class ProblemType>
void HybridOptimizer::SetLocalOccupiedGridCost(
        const common::SlamConfigPtr& config,
        const common::KeyFrames& key_frames,
        const common::Grid2D* grid,
        ParaBuffer* buffer_ptr,
        ProblemType* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    CHECK_NOTNULL(grid);
    CHECK_EQ(key_frames.size(), buffer.poses_estimate.size());
    size_t start_idx = 0u;
    size_t end_idx = buffer.poses_estimate.size() - 1u;
    if (!first_marginalization_) {
        start_idx = buffer.poses_estimate.size() - 1u;
    }
    for (size_t i = start_idx; i <= end_idx; ++i) {
        if (!key_frames[i].points.points.empty()) {
            auto* scan_cost = CreateLocalOccupiedSpace2D(
                        key_frames[i].points,
                        *grid,
                        config->scan_sigma);
            problem->AddResidualBlock(
                        scan_cost,
                        nullptr,
                        buffer.poses_estimate[i].data());
            VLOG(2) << "Scan cost num: " << key_frames[i].points.points.size();
        }
    }
}

template<class ProblemType>
void HybridOptimizer::SetGlobalOccupiedGridCost(
        const common::SlamConfigPtr& config,
        const common::KeyFrames& key_frames,
        const common::Grid2D* grid,
        const size_t start_idx,
        const size_t end_idx,
        ParaBuffer* buffer_ptr,
        ProblemType* problem) {
    ParaBuffer& buffer = *CHECK_NOTNULL(buffer_ptr);
    for (size_t i = start_idx; i <= end_idx; ++i) {
        if (!key_frames[i].points.points.empty()) {
            auto* scan_cost = CreateGlobalOccupiedSpace2D(
                        key_frames[i].points,
                        *grid,
                        config->scan_sigma);
            problem->AddResidualBlock(
                        scan_cost,
                        nullptr,
                        buffer.poses_estimate[i].data(),
                        buffer.pose_GtoM_estimate.data());
        }
    }
}

void HybridOptimizer::PreSetForBA() {
    Reset();
    ceres_solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres_solver_options_.max_num_iterations = 100;
    ceres_solver_options_.function_tolerance = 1e-10;
    ceres_solver_options_.logging_type = ceres::PER_MINIMIZER_ITERATION;
}

bool HybridOptimizer::HasRelocInited() const {
    return reloc_inited_;
}

bool HybridOptimizer::HasImuInited() const {
    return imu_inited_;
}

void HybridOptimizer::ResetReloc() {
    reloc_inited_ = false;
    T_GtoM_prior_seted_ = false;
    first_marginalization_ = true;
}

template
bool HybridOptimizer::SetSlidingWindowProblem<schur::Problem>(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx,
    const common::FeaturePointPtrVec& features,
    const common::LoopResults& loop_results,
    const common::Grid2D* grid_local,
    const common::Grid2D* grid_global,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    schur::Problem* problem);

template
bool HybridOptimizer::SetSlidingWindowProblem<ceres::Problem>(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx,
    const common::FeaturePointPtrVec& features,
    const common::LoopResults& loop_results,
    const common::Grid2D* grid_local,
    const common::Grid2D* grid_global,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    ceres::Problem* problem);

template
void HybridOptimizer::SetImuPropagationCost<schur::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    schur::Problem* problem);

template
void HybridOptimizer::SetImuPropagationCost<ceres::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    ceres::Problem* problem);

template
void HybridOptimizer::SetOdomPropagationCost<schur::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    schur::Problem* problem);

template
void HybridOptimizer::SetOdomPropagationCost<ceres::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    ceres::Problem* problem);

template
int HybridOptimizer::SetVisualReprojectionCost<schur::Problem>(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx,
    const common::FeaturePointPtrVec& features,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    schur::Problem* problem);

template
int HybridOptimizer::SetVisualReprojectionCost<ceres::Problem>(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx,
    const common::FeaturePointPtrVec& features,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    ceres::Problem* problem);

template
int HybridOptimizer::SetMapVisualReprojectionCost<schur::Problem>(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx,
    const common::LoopResults& reloc_results,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    schur::Problem* problem);

template
int HybridOptimizer::SetMapVisualReprojectionCost<ceres::Problem>(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx,
    const common::LoopResults& reloc_results,
    ParaBuffer* buffer_ptr,
    common::EdgeVec* edges_ptr,
    ceres::Problem* problem);                                     
template
void HybridOptimizer::SetLocalOccupiedGridCost<schur::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const common::Grid2D* grid,
    ParaBuffer* buffer_ptr,
    schur::Problem* problem);

template
void HybridOptimizer::SetLocalOccupiedGridCost<ceres::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const common::Grid2D* grid,
    ParaBuffer* buffer_ptr,
    ceres::Problem* problem);

template
void HybridOptimizer::SetGlobalOccupiedGridCost<schur::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const common::Grid2D* grid,
    const size_t start_idx,
    const size_t end_idx,
    ParaBuffer* buffer_ptr,
    schur::Problem* problem);

template
void HybridOptimizer::SetGlobalOccupiedGridCost<ceres::Problem>(
    const common::SlamConfigPtr& config,
    const common::KeyFrames& key_frames,
    const common::Grid2D* grid,
    const size_t start_idx,
    const size_t end_idx,
    ParaBuffer* buffer_ptr,
    ceres::Problem* problem);
}
