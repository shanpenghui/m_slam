#include "cost_function/pose_in_plane_cost.h"

#include "data_common/constants.h"
#include "math_common/math.h"

namespace vins_core {

PoseInPlaneCost::PoseInPlaneCost(const Eigen::Matrix2d& sqrt_info_R,
                                 const double sqrt_info_t) {

    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    set_num_residuals(3);

    sqrt_info_.setZero();
    sqrt_info_.topLeftCorner<2, 2>() = sqrt_info_R;
    sqrt_info_(2, 2) = sqrt_info_t;
}

bool PoseInPlaneCost::Evaluate(double const * const *parameters, double *residuals,
                               double **jacobians) const {
    Eigen::Map<const Eigen::Vector3d> p_OinG(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG(parameters[0] + 3);
    const Eigen::Matrix3d R_OtoG = q_OtoG.toRotationMatrix();

    Eigen::Vector3d e3(0.0, 0.0, 1.0);
    Eigen::Matrix<double, 2, 3, Eigen::RowMajor> gama;
    gama << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0;

    Eigen::Map<Eigen::Vector3d> residual(residuals);
    residual.head<2>() = gama * R_OtoG * e3;
    residual(2) = e3.transpose() * p_OinG;
    residual = sqrt_info_ * residual;

    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, common::kGlobalPoseSize>> dr_dT(jacobians[0]);
            dr_dT.setZero();
            dr_dT.topLeftCorner<2, 3>() = Eigen::Matrix<double, 2, 3>::Zero();
            dr_dT.bottomLeftCorner<1, 3>() = e3.transpose();
            dr_dT.block<2, 3>(0, 3) = - gama * R_OtoG * common::skew_x(e3);
            dr_dT.block<1, 3>(2, 3) = Eigen::Matrix<double, 1, 3>::Zero();
            dr_dT = sqrt_info_ * dr_dT;
        }
    }
    return true;
}

}