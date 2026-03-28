#ifndef MVINS_HYBRID_OPTIMIZER_HYBRID_OPTIMIZER_H_
#define MVINS_HYBRID_OPTIMIZER_HYBRID_OPTIMIZER_H_

#include <aslam/cameras/ncamera.h>
#include <ceres/ceres.h>

#include "cfg_common/slam_config.h"
#include "cost_function/schur_complement_problem.h"
#include "data_common/state_structures.h"
#include "occ_common/grid_2d.h"

namespace vins_core {

struct ParaBuffer {
    common::EigenVector7dVec poses_estimate;

    common::EigenVector3dVec velocity_estimate;

    std::vector<double> td_camera_estimate;
    std::vector<double> td_scan_estimate;

    common::EigenVector3dVec bg_estimate;
    common::EigenVector3dVec ba_estimate;

    common::EigenVector2dVec bt_estimate;
    std::vector<double> br_estimate;

    std::vector<double> inv_depth_estimate;

    common::EigenVector7d pose_GtoM_estimate;

    common::EigenVector7dVec exs_OtoC_estimate;

    std::vector<double> switch_variables;

    schur::Problem* last_kept_term = nullptr;
    std::vector<double*> last_kept_blocks;
};

void KeyframeToBuffer(const common::KeyFrames& key_frames,
                      const aslam::NCamera::Ptr& cameras,
                      const common::FeaturePointPtrVec& features,
                      const common::LoopResults& loop_results,
                      const aslam::Transformation& T_GtoM,
                      ParaBuffer* buffer_ptr);
void BufferToKeyframe(const ParaBuffer& buffer,
                      const aslam::NCamera::Ptr& cameras,
                      common::KeyFrames* key_frames_ptr,
                      common::FeaturePointPtrVec* features_ptr,
                      aslam::Transformation* T_GtoM_ptr);

class HybridOptimizer{
public:
    explicit HybridOptimizer(const common::SlamConfigPtr& slam_config);
    virtual ~HybridOptimizer() = default;

    void Reset();

    void Solve(const aslam::NCamera::Ptr& cameras,
               const common::SlamConfigPtr& config,
               const common::KeyFrames& key_frames,
               const std::unordered_map<int, size_t>& keyframe_id_to_idx,
               const common::FeaturePointPtrVec& features,
               const common::LoopResults& loop_results,
               const common::Grid2D* grid_local,
               const common::Grid2D* grid_global,
               ParaBuffer* para_buffer_ptr,
               common::EdgeVec* edges_ptr);

    void Marginalize(const aslam::NCamera::Ptr& cameras,
                     const common::SlamConfigPtr& config,
                     const common::KeyFrames& key_frames,
                     const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                     const common::FeaturePointPtrVec& features,
                     const common::LoopResults& loop_results,
                     const common::Grid2D* grid_local,
                     const common::Grid2D* grid_global,
                     const aslam::Transformation& T_GtoM,
                     ParaBuffer* para_buffer_ptr);

    void PoseGraph(const common::KeyFrames& key_frames,
                   const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                   const common::LoopResults& loop_results,
                   const bool fix_current,
                   ParaBuffer* para_buffer_ptr,
                   common::EdgeVec* edges_ptr);

    void InitReloc(const aslam::NCamera::Ptr& cameras,
                   const common::SlamConfigPtr& config,
                   const common::KeyFrames& key_frames,
                   const common::LoopResults& reloc_results,              
                   const common::Grid2D* grid_global,
                   ParaBuffer* para_buffer_ptr);

    void MapPoseTuning(const common::SlamConfigPtr& config,
                       const common::KeyFrames& key_frames,
                       const common::Grid2D* grid_global,
                       ParaBuffer* para_buffer_ptr);

    void PreSetForBA();

    bool HasRelocInited() const;

    bool HasImuInited() const;

    void ResetReloc();

    void SetRelocInitSuccess();

    void SetImuInitSuccess();

private:
    template<class ProblemType>
    bool SetSlidingWindowProblem(const aslam::NCamera::Ptr& cameras,
                                 const common::SlamConfigPtr& config,
                                 const common::KeyFrames& key_frames,
                                 const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                 const common::FeaturePointPtrVec& features,
                                 const common::LoopResults& loop_results,
                                 const common::Grid2D* grid_local,
                                 const common::Grid2D* grid_global,
                                 ParaBuffer* buffer_ptr,
                                 common::EdgeVec* edges_ptr,
                                 ProblemType* problem);

    template<class ProblemType>
    void SetImuPropagationCost(const common::SlamConfigPtr& config,
                               const common::KeyFrames& key_frames,
                               ParaBuffer* buffer_ptr,
                               common::EdgeVec* edges_ptr,
                               ProblemType* problem);

    template<class ProblemType>
    void SetOdomPropagationCost(const common::SlamConfigPtr& config,
                                const common::KeyFrames& key_frames,
                                ParaBuffer* buffer_ptr,
                                common::EdgeVec* edges_ptr,
                                ProblemType* problem);

    template<class ProblemType>
    int SetVisualReprojectionCost(const aslam::NCamera::Ptr& cameras,
                                  const common::SlamConfigPtr& config,
                                  const common::KeyFrames& key_frames,
                                  const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                  const common::FeaturePointPtrVec& features,
                                  ParaBuffer* buffer_ptr,
                                  common::EdgeVec* edges_ptr,
                                  ProblemType* problem);

    template<class ProblemType>
    int SetMapVisualReprojectionCost(const aslam::NCamera::Ptr& cameras,
                                     const common::SlamConfigPtr& config,
                                     const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                     const common::LoopResults& reloc_results,
                                     ParaBuffer* buffer_ptr,
                                     common::EdgeVec* edges_ptr,
                                     ProblemType* problem);

    template<class ProblemType>
    void SetLocalOccupiedGridCost(const common::SlamConfigPtr& config,
                                  const common::KeyFrames& key_frames,
                                  const common::Grid2D* grid,
                                  ParaBuffer* buffer_ptr,
                                  ProblemType* problem);

    template<class ProblemType>
    void SetGlobalOccupiedGridCost(const common::SlamConfigPtr& config,
                                   const common::KeyFrames& key_frames,
                                   const common::Grid2D* grid,
                                   const size_t start_idx,
                                   const size_t end_idx,
                                   ParaBuffer* buffer_ptr,
                                   ProblemType* problem);

    void SetPoseGraphProblem(const common::KeyFrames& key_frames,
                             const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                             const common::LoopResults& loop_results,
                             const bool fix_current,
                             ParaBuffer* para_buffer_ptr,
                             common::EdgeVec* edges_ptr,
                             ceres::Problem* problem);

    const common::SlamConfigPtr config_;

    ceres::Solver::Options ceres_solver_options_;
    bool reloc_inited_;
    bool imu_inited_;
    bool T_GtoM_prior_seted_;
    bool first_optimization_;
    bool first_marginalization_;
};

}

#endif
