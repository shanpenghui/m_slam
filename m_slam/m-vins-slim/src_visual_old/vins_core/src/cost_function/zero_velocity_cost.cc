#include "cost_function/zero_velocity_cost.h"

namespace vins_core {
ZeroVelocityCost::ZeroVelocityCost(
        const Eigen::Matrix<double, 3, 3>& sqrt_info)
    : sqrt_info_(sqrt_info) {
    mutable_parameter_block_sizes()->push_back(3);

    set_num_residuals(3);
}

bool ZeroVelocityCost::Evaluate(double const * const *parameters, double *residuals,
                                double **jacobians) const {
    Eigen::Map<const Eigen::Vector3d> velocity(parameters[0]);

    Eigen::Map<Eigen::Matrix<double, 3, 1> > residual(residuals);
    residual = sqrt_info_ * velocity;

    if (jacobians) {
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > dr_dv(jacobians[0]);
            dr_dv.setZero();

            dr_dv = sqrt_info_ * Eigen::Matrix3d::Identity();
        }
    }

    return true;
}
}
