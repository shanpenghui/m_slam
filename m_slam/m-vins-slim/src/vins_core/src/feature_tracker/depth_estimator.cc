#include "feature_tracker/depth_estimator.h"

#include <Eigen/QR>

#include "cost_function/pose_local_parameterization.h"
#include "feature_tracker/reprojection_checker.h"
#include "math_common/math.h"

namespace vins_core {

constexpr int kAnchorObsIdx = 0;

DepthEstimator::DepthEstimator(const DepthEstimatorOptions& options)
    : options_(options) {}

bool DepthEstimator::Triangulation(const aslam::NCamera::Ptr& cameras,
                                   const common::KeyFrames& key_frames,
                                   const common::ObservationDeq& observations,
                                   common::FeaturePoint* feature_point_ptr) {
    common::FeaturePoint& feature_point = *CHECK_NOTNULL(feature_point_ptr);

    // LUT.
    std::unordered_map<int, size_t> keyframe_id_to_idx;
    for (size_t i = 0; i < key_frames.size(); ++i) {
        keyframe_id_to_idx[key_frames[i].keyframe_id] = i;
    }

    // Our linear system matrices.
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    const int anchor_frame_id = observations[kAnchorObsIdx].keyframe_id;
    const aslam::Transformation& T_BtoA =
            cameras->get_T_BtoC(observations[kAnchorObsIdx].camera_idx);
    const aslam::Transformation& T_BtoG =
            key_frames[keyframe_id_to_idx.at(anchor_frame_id)].state.T_OtoG;
    const aslam::Transformation T_AtoG = T_BtoG * T_BtoA.inverse();
    const Eigen::Matrix3d R_GtoA = T_AtoG.getRotationMatrix().transpose();
    const Eigen::Vector3d p_AinG = T_AtoG.getPosition();

    common::EigenMatrix3dVec R_AtoCis;
    common::EigenVector3dVec p_CiinAs;
    common::EigenVector3dVec p_AinCis;
    for (const auto& observation : observations) {
        // Get the position of this frame in the global.
        const Eigen::Vector2d& obs_key_point = observation.key_point;
        const int obs_cam_idx = observation.camera_idx;
        const int obs_frame_id = observation.keyframe_id;
        const common::KeyFrame& obs_key_frame =
                key_frames[keyframe_id_to_idx.at(obs_frame_id)];

        const aslam::Transformation& T_BtoCi = cameras->get_T_BtoC(obs_cam_idx);
        const aslam::Transformation& T_BtoG = obs_key_frame.state.T_OtoG;
        const aslam::Transformation T_CitoG = T_BtoG * T_BtoCi.inverse();
        const Eigen::Matrix3d R_GtoCi = T_CitoG.getRotationMatrix().transpose();
        const Eigen::Vector3d p_CiinG = T_CitoG.getPosition();

        // Convert current position relative to anchor
        Eigen::Matrix<double,3,3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
        R_AtoCis.push_back(R_AtoCi);
        Eigen::Matrix<double,3,1> p_CiinA;
        p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
        p_CiinAs.push_back(p_CiinA);
        Eigen::Matrix<double,3,1> p_AinCi;
        p_AinCi.noalias() = -R_AtoCi*p_CiinA;
        p_AinCis.push_back(p_AinCi);

        // Get the UV coordinate normal
        Eigen::Vector3d b_i;
        cameras->getCamera(obs_cam_idx).backProject3(obs_key_point, &b_i);
        b_i << b_i(0) / b_i(2), b_i(1) / b_i(2), 1.0;
        b_i = R_AtoCi.transpose() * b_i;
        b_i = b_i / b_i.norm();
        Eigen::Matrix3d Bperp = common::skew_x(b_i);

        // Append to our linear system
        Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
        A += Ai;
        b += Ai * p_CiinA;
    }

    // Solve the linear system
    Eigen::Vector3d p_f = A.colPivHouseholderQr().solve(b);

    // Check A and p_f
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    if (std::abs(condA) > options_.max_cond_number || p_f(2,0) < options_.min_dist ||
            p_f(2,0) > options_.max_dist || std::isnan(p_f.norm())) {
        return false;
    }

    // Fine tuning by optmization.
    if (TriangulationOptimization(cameras, R_AtoCis, p_CiinAs, p_AinCis, observations, &p_f)) {
        // Store it in our feature object
        feature_point.anchor_frame_idx = kAnchorObsIdx;
        feature_point.inv_depth = 1.0 / p_f(2);
        feature_point.observations = observations;

        return CheckReprojectionError(cameras,
                                      key_frames,
                                      keyframe_id_to_idx,
                                      feature_point);
    }
    return false;
}

bool DepthEstimator::TriangulationOptimization(const aslam::NCamera::Ptr& cameras,
                                               const common::EigenMatrix3dVec R_AtoCis,
                                               const common::EigenVector3dVec p_CiinAs,
                                               const common::EigenVector3dVec p_AinCis,
                                               const common::ObservationDeq& observations,
                                               Eigen::Vector3d* p_f_ptr) {
    Eigen::Vector3d& p_f = *CHECK_NOTNULL(p_f_ptr);

    // Get into inverse depth.
    double rho = 1.0 / p_f(2);
    double alpha = p_f(0) / p_f(2);
    double beta = p_f(1) / p_f(2);

    // Optimization parameters
    double lam = options_.init_lamda;
    double eps = 10000;
    int runs = 0;

    // Variables used in the optimization
    bool recompute = true;
    Eigen::Matrix<double,3,3> Hess = Eigen::Matrix<double,3,3>::Zero();
    Eigen::Matrix<double,3,1> grad = Eigen::Matrix<double,3,1>::Zero();

    double cost_old = std::numeric_limits<double>::max();
    // Loop till we have either
    // 1. Reached our max iteration count
    // 2. System is unstable
    // 3. System has converged
    while (runs < options_.max_runs && lam < options_.max_lamda && eps > options_.min_dx) {

        // Triggers a recomputation of jacobians/information/gradients
        if (recompute) {
            Hess.setZero();
            grad.setZero();

            // Cost at the last iteration
            cost_old = ComputeError(cameras, R_AtoCis, p_AinCis, observations, alpha, beta, rho,
                                     &Hess, &grad);
        }

        // Solve Levenberg iteration
        Eigen::Matrix<double,3,3> Hess_l = Hess;
        for (size_t r=0; r < (size_t)Hess.rows(); r++) {
            Hess_l(r,r) *= (1.0+lam);
        }

        Eigen::Matrix<double,3,1> dx = Hess_l.colPivHouseholderQr().solve(grad);
        // Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

        // Check if error has gone down
        double cost = ComputeError(cameras, R_AtoCis, p_AinCis, observations,
                                     alpha+dx(0,0), beta+dx(1,0), rho+dx(2,0));

        // Debug print
        VLOG(10) << "run = " << runs << " | cost = " << dx.norm()
                 << " | lamda = " << lam << " | depth = " << 1/rho << std::endl;

        // Check if converged
        if (cost <= cost_old && (cost_old-cost)/cost_old < options_.min_dcost) {
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            break;
        }

        // If cost is lowered, accept step
        // Else inflate lambda (try to make more stable)
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            runs++;
            lam = lam / options_.lam_mult;
            eps = dx.norm();
        } else {
            recompute = false;
            lam = lam * options_.lam_mult;
            continue;
        }
    }

    // Revert to standard, and set to all
    p_f(0) = alpha/rho;
    p_f(1) = beta/rho;
    p_f(2) = 1/rho;

    // Get tangent plane to x_hat
     Eigen::HouseholderQR<Eigen::MatrixXd> qr(p_f);
     Eigen::MatrixXd Q = qr.householderQ();

     // Max baseline we have between poses
     double base_line_max = 0.0;

     // Check maximum baseline
     for (size_t m = 0; m < p_CiinAs.size(); m++) {
         const auto& p_CiinA = p_CiinAs[m];
         // Dot product camera pose and nullspace
         double base_line = ((Q.block(0,1,3,2)).transpose() * p_CiinA).norm();
         if (base_line > base_line_max) base_line_max = base_line;
     }

     // Check if this feature is bad or not
     // 1. If the feature is too close
     // 2. If the feature is invalid
     // 3. If the baseline ratio is large
     if (p_f(2) < options_.min_dist
         || p_f(2) > options_.max_dist
         || (p_f.norm() / base_line_max) > options_.max_baseline
         || std::isnan(p_f.norm())) {
         return false;
     }

    return true;
}

bool DepthEstimator::EstimateDepth(const aslam::NCamera::Ptr& cameras,
                                   const common::KeyFrames& key_frames,
                                   const common::ObservationDeq& observations,
                                   common::FeaturePoint* feature_point_ptr) {
    common::FeaturePoint& feature_point = *CHECK_NOTNULL(feature_point_ptr);

    // LUT.
    std::unordered_map<int, size_t> keyframe_id_to_idx;
    for (size_t i = 0; i < key_frames.size(); ++i) {
        keyframe_id_to_idx[key_frames[i].keyframe_id] = i;
    }

    Eigen::Vector3d bearing_3d_anchor;
    cameras->getCamera(observations[kAnchorObsIdx].camera_idx).backProject3(
                observations[kAnchorObsIdx].key_point, &bearing_3d_anchor);
    const double z = bearing_3d_anchor(2);
    bearing_3d_anchor << bearing_3d_anchor(0) / z, bearing_3d_anchor(1) / z, 1.0;
    Eigen::Vector3d p_f = observations[kAnchorObsIdx].depth * bearing_3d_anchor;

    if (!(p_f(2) < options_.min_dist || p_f(2) > options_.max_dist || std::isnan(p_f.norm()))) {
        // Store it in our feature object.
        feature_point.anchor_frame_idx = kAnchorObsIdx;
        feature_point.inv_depth = 1.0 / p_f(2);

        feature_point.observations = observations;

        return CheckReprojectionError(cameras,
                                      key_frames,
                                      keyframe_id_to_idx,
                                      feature_point);
    }

    return false;
}

bool DepthEstimator::CheckReprojectionError(const aslam::NCamera::Ptr& cameras,
                                            const common::KeyFrames& key_frames,
                                            const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                                            const common::FeaturePoint& feature_point) {
    if (feature_point.anchor_frame_idx == -1) {
        return false;
    }

    Eigen::VectorXd dists;
    GetVisualReprojectionError(cameras,
                              key_frames,
                              keyframe_id_to_idx,
                              feature_point,
                              &dists);
    bool is_success = false;
    if (dists.maxCoeff() <= options_.outlier_rejection_threshold) {
        is_success = true;
    }

    return is_success;
}

double DepthEstimator::ComputeError(const aslam::NCamera::Ptr& cameras,
                                    const common::EigenMatrix3dVec R_AtoCis,
                                    const common::EigenVector3dVec p_AinCis,
                                    const common::ObservationDeq& observations,
                                    const double alpha,
                                    const double beta,
                                    const double rho,
                                    Eigen::Matrix<double, 3, 3>* Hess_ptr,
                                    Eigen::Matrix<double, 3, 1>* grad_ptr) {
    // Total error
    double err = 0;

    for (size_t m = 0; m < R_AtoCis.size(); m++) {
        const Eigen::Matrix3d& R_AtoCi = R_AtoCis[m];
        const Eigen::Vector3d& p_AinCi = p_AinCis[m];
        // Middle variables of the system
        double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
        double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
        double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
        Eigen::Matrix<double, 2, 3> H = Eigen::Matrix<double, 2, 3>::Zero();
        if (grad_ptr != nullptr && Hess_ptr != nullptr) {
            // Calculate jacobian
            double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
            double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
            double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
            double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
            double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
            double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
            H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
        }
        // Calculate residual
        Eigen::Matrix<double, 2, 1> z;
        z << hi1 / hi3, hi2 / hi3;
        Eigen::Vector3d bearing_3d;
        cameras->getCamera(observations.at(m).camera_idx).backProject3(
                    observations.at(m).key_point, &bearing_3d);
        bearing_3d << bearing_3d(0) / bearing_3d(2), bearing_3d(1) / bearing_3d(2), 1.0;
        Eigen::Matrix<double, 2, 1> res = bearing_3d.head<2>() - z;
        // Append to our summation variables
        err += pow(res.norm(), 2);
        if (grad_ptr != nullptr && Hess_ptr != nullptr) {
            auto& grad = *grad_ptr;
            auto& Hess = *Hess_ptr;
            grad.noalias() += H.transpose() * res;
            Hess.noalias() += H.transpose() * H;
        }
    }

    return err;
}
}
