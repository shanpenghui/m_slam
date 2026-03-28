#include "cost_function/relative_pose_cost.h"

#include "math_common/math.h"

namespace vins_core {
RelativePoseCost::RelativePoseCost(
        const Eigen::Quaterniond& delta_q,
        const Eigen::Vector3d& delta_p,
        const Eigen::Matrix<double, 6, 6>& sqrt_info)
    : delta_q_measured_(delta_q),
      delta_p_measured_(delta_p),
      sqrt_info_(sqrt_info) {
    // 1) Current keyframe pose.
    // 2) Next keyframe pose.
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);

    set_num_residuals(6);
}

bool RelativePoseCost::Evaluate(double const * const *parameters, double *residuals,
                                double **jacobians) const {
    Eigen::Map<const Eigen::Vector3d> p_OinG_current(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_current(parameters[0] + 3);
    const Eigen::Matrix3d R_OtoG_current = q_OtoG_current.toRotationMatrix();
    Eigen::Map<const Eigen::Vector3d> p_OinG_next(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_next(parameters[1] + 3);

    const Eigen::Quaterniond delta_q_estimated =
            q_OtoG_current.conjugate() * q_OtoG_next;
    const Eigen::Vector3d delta_p_estimated =
            q_OtoG_current.conjugate() * (p_OinG_next - p_OinG_current);

    const Eigen::Quaterniond diff_delta_q =
            delta_q_measured_.conjugate() * delta_q_estimated;

    Eigen::Map<Eigen::Matrix<double, 6, 1> > residual(residuals);
    residual.head<3>() = delta_p_estimated - delta_p_measured_;
    const double sign_q = diff_delta_q.w() > 0 ? 1. : -1.;
    residual.tail<3>() = sign_q * 2.0 * diff_delta_q.vec();
    residual = sqrt_info_ * residual;

    if (jacobians) {
        const Eigen::Matrix3d dleta_R_estimated = delta_q_estimated.toRotationMatrix();
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_current(
                    jacobians[0]);
            dr_dT_current.setZero();

            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> drp_dT_current;
            drp_dT_current.setZero();
            drp_dT_current.leftCols<3>() = -R_OtoG_current.transpose();
            drp_dT_current.rightCols<3>() = common::skew_x(delta_p_estimated);

            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> drr_dT_current;
            drr_dT_current.setZero();
            drr_dT_current.rightCols<3>() = -dleta_R_estimated.transpose();

            dr_dT_current.topLeftCorner<3, common::kLocalPoseSize>() = drp_dT_current;
            dr_dT_current.bottomLeftCorner<3, common::kLocalPoseSize>() = drr_dT_current;

            dr_dT_current = sqrt_info_ * dr_dT_current;
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 6, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_next(
                    jacobians[1]);
            dr_dT_next.setZero();

            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> drp_dT_next;
            drp_dT_next.setZero();
            drp_dT_next.leftCols<3>() = R_OtoG_current.transpose();

            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> drr_dT_next;
            drr_dT_next.setZero();
            drr_dT_next.rightCols<3>() = Eigen::Matrix3d::Identity();

            dr_dT_next.topLeftCorner<3, common::kLocalPoseSize>() = drp_dT_next;
            dr_dT_next.bottomLeftCorner<3, common::kLocalPoseSize>() = drr_dT_next;

            dr_dT_next = sqrt_info_ * dr_dT_next;
        }
    }

    return true;
}

PosePriorCost::PosePriorCost(
        const Eigen::Quaterniond& q,
        const Eigen::Vector3d& p,
        const Eigen::Matrix<double, 6, 6>& sqrt_info)
    : q_measured_(q),
      p_measured_(p),
      sqrt_info_(sqrt_info) {
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);

    set_num_residuals(6);
}

bool PosePriorCost::Evaluate(double const * const *parameters, double *residuals,
                                double **jacobians) const {
    Eigen::Map<const Eigen::Vector3d> p_OinG(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG(parameters[0] + 3);

    const Eigen::Quaterniond diff_q = q_measured_.conjugate() * q_OtoG;

    Eigen::Map<Eigen::Matrix<double, 6, 1> > residual(residuals);
    residual.head<3>() = p_OinG - p_measured_;
    const double sign_q = diff_q.w() > 0 ? 1. : -1.;
    residual.tail<3>() = sign_q * 2.0 * diff_q.vec();
    residual = sqrt_info_ * residual;
    CHECK(!residual.hasNaN()) << residual.transpose();

    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT(
                    jacobians[0]);
            dr_dT.setZero();

            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> drp_dT;
            drp_dT.setZero();
            drp_dT.leftCols<3>() = Eigen::Matrix3d::Identity();

            Eigen::Matrix<double, 3, common::kLocalPoseSize, Eigen::RowMajor> drr_dT;
            drr_dT.setZero();
            drr_dT.rightCols<3>() = Eigen::Matrix3d::Identity();

            dr_dT.topLeftCorner<3, common::kLocalPoseSize>() = drp_dT;
            dr_dT.bottomLeftCorner<3, common::kLocalPoseSize>() = drr_dT;

            dr_dT = sqrt_info_ * dr_dT;
        }
    }

    return true;
}
}
