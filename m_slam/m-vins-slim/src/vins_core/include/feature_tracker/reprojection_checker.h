#ifndef MVINS_REPROJECTION_CHECKER_H_
#define MVINS_REPROJECTION_CHECKER_H_

#include <Eigen/Core>

#include "aslam/cameras/ncamera.h"
#include "data_common/state_structures.h"

namespace vins_core {

void GetVisualReprojectionError(const aslam::NCamera::Ptr& cameras,
                               const common::KeyFrames& key_frames,
                               const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                               const common::FeaturePoint& feature_point,
                               Eigen::VectorXd* dists_ptr);

void GetMapVisualReprojectionError(const aslam::NCamera::Ptr& cameras,
                                   const common::LoopResult& loop_result,
                                   const aslam::Transformation& T_OtoG,
                                   const aslam::Transformation& T_GtoM,
                                   Eigen::VectorXd* dists_ptr);
}
#endif //  MVINS_REPROJECTION_CHECKER_H_
