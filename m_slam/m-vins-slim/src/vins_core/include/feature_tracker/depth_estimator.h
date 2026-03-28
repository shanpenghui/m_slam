#ifndef FEATURE_TRACKER_FEATURE_DEPTH_ESTIMATOR_H_
#define FEATURE_TRACKER_FEATURE_DEPTH_ESTIMATOR_H_

#include <aslam/cameras/ncamera.h>

#include "data_common/state_structures.h"
#include "data_common/visual_structures.h"

namespace vins_core {
struct DepthEstimatorOptions {
    /// If we should perform Levenberg-Marquardt refinment
    bool refine_features = true;

    /// Max runs for Levenberg-Marquardt
    int max_runs = 5;

    /// Init lambda for Levenberg-Marquardt optimization
    double init_lamda = 1e-3;

    /// Max lambda for Levenberg-Marquardt optimization
    double max_lamda = 1e10;

    /// Cutoff for dx increment to consider as converged
    double min_dx = 1e-6;

    /// Cutoff for cost decrement to consider as converged
    double min_dcost = 1e-6;

    /// Multiplier to increase/decrease lambda
    double lam_mult = 10;

    /// Minimum distance to accept features
    double min_dist = 0.10;

    /// Minimum distance to accept features
    double max_dist = 10;

    /// Max baseline ratio to accept features
    double max_baseline = 40;

    /// Max condition number of linear matrix accept features
    double max_cond_number = 10000;

    // Outlier rejection threshold for accept observation.
    double outlier_rejection_threshold = 4.0;
};

class DepthEstimator {
public:
    DepthEstimator(const DepthEstimatorOptions& options);
    bool Triangulation(const aslam::NCamera::Ptr& cameras,
                       const common::KeyFrames& key_frames,
                       const common::ObservationDeq& observations,
                       common::FeaturePoint* feature_point_ptr);
    bool EstimateDepth(const aslam::NCamera::Ptr& cameras,
                       const common::KeyFrames& key_frames,
                       const common::ObservationDeq& observations,
                       common::FeaturePoint* feature_point_ptr);
    bool CheckReprojectionError(const aslam::NCamera::Ptr& cameras,
                                const common::KeyFrames& key_frames,
                                const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                const common::FeaturePoint& feature_point);
private:
    bool TriangulationOptimization(const aslam::NCamera::Ptr& cameras,
                                   const common::EigenMatrix3dVec R_AtoCis,
                                   const common::EigenVector3dVec p_CiinAs,
                                   const common::EigenVector3dVec p_AinCis,
                                   const common::ObservationDeq& observations,
                                   Eigen::Vector3d* p_f_ptr);
    double ComputeError(const aslam::NCamera::Ptr& cameras,
                        const common::EigenMatrix3dVec R_AtoCis,
                        const common::EigenVector3dVec p_AinCis,
                        const common::ObservationDeq& observations,
                        const double alpha,
                        const double beta,
                        const double rho,
                        Eigen::Matrix<double, 3, 3>* Hess_ptr = nullptr,
                        Eigen::Matrix<double, 3, 1>* grad_ptr = nullptr);
    const DepthEstimatorOptions options_;
};
}

#endif
