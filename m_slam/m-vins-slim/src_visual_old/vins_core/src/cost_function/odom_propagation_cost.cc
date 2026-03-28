#include "cost_function/odom_propagation_cost.h"

namespace vins_core {

OdomPropagationCost::OdomPropagationCost(
        const common::SlamConfigPtr& config,
        const common::OdomDatas& propa_data)
    : config_(config), propa_data_(propa_data) {
    // Pose
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    // bt
    mutable_parameter_block_sizes()->push_back(2);
    mutable_parameter_block_sizes()->push_back(2);
    // br
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);

    set_num_residuals(9);
}

bool OdomPropagationCost::Evaluate(double const *const *parameters,
                                    double *residuals_ptr,
                                    double **jacobians) const {
    Eigen::Map<const Eigen::Vector3d> p_OinG_curr(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_curr(parameters[0] + 3);
    Eigen::Map<const Eigen::Vector3d> p_OinG_next(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_next(parameters[1] + 3);
    Eigen::Map<const Eigen::Vector2d> bt_curr(parameters[2]);
    Eigen::Map<const Eigen::Vector2d> bt_next(parameters[3]);
    const double br_curr(parameters[4][0]);
    const double br_next(parameters[5][0]);

    vins_core::OdomPropagator odom_propagator;
    aslam::Transformation T_OtoG_curr(q_OtoG_curr, p_OinG_curr);
    aslam::Transformation T_OtoG_next(q_OtoG_next, p_OinG_next);
    common::State state_curr(T_OtoG_curr, bt_curr, br_curr);
    common::State state_next(T_OtoG_next, bt_next, br_next);

    const double pose_diff = (state_next - state_curr).norm();

    common::State state_next_estimated;
    Eigen::Matrix<double, 9, 9> Phi = Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix<double, 9, 9> Q = Eigen::Matrix<double, 9, 9>::Zero();
    if (pose_diff < 1e-6) {
        state_next_estimated = state_next;
    } else {
        odom_propagator.Propagate(propa_data_, &state_curr, &Phi, &Q);
        state_next_estimated = state_curr;
    }

    if (Phi.maxCoeff() > 1e8 || Phi.minCoeff() < -1e8) {
        LOG(FATAL) << "Numerical unstable in pre-integration.";
    }

    Eigen::LLT<Eigen::Matrix<double, 9, 9>> cholesky_solver;
    cholesky_solver.compute(Q);
    Eigen::Matrix<double, 9, 9> sqrt_info = Eigen::Matrix<double, 9, 9>::Identity();
    cholesky_solver.matrixL().solveInPlace(sqrt_info);
#ifdef DEBUG
    CHECK(!sqrt_info.hasNaN()) << "Odom meas size: " << propa_data_.size() << " Q: "
                               << std::endl << Q;
#endif
    Eigen::Matrix<double, 18, 1> state_residual = state_next - state_next_estimated;
    Eigen::Map<Eigen::Matrix<double, 9, 1>> residuals_map(residuals_ptr);
    residuals_map.head<6>() = state_residual.head<6>();
    residuals_map.tail<3>() = state_residual.tail<3>();
    residuals_map = sqrt_info * residuals_map;
#ifdef DEBUG
    CHECK(!residuals_map.hasNaN()) << residuals_map.transpose();
#endif

    if (jacobians) {
        Eigen::Matrix<double, 9, 9> identity = Eigen::Matrix<double, 9, 9>::Identity();
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 9, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_curr(jacobians[0]);
            dr_dT_curr.setZero();

            dr_dT_curr.topLeftCorner<9, common::kLocalPoseSize>() =
                    -Phi.topLeftCorner<9, common::kLocalPoseSize>();
            dr_dT_curr = sqrt_info * dr_dT_curr;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 9, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_next(jacobians[1]);
            dr_dT_next.setZero();

            dr_dT_next.topLeftCorner<9, common::kLocalPoseSize>() =
                    identity.topLeftCorner<9, common::kLocalPoseSize>();
            dr_dT_next = sqrt_info * dr_dT_next;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> dr_dbt_curr(jacobians[2]);
            dr_dbt_curr.setZero();

            dr_dbt_curr = -Phi.block<9, 2>(0, 6);
            dr_dbt_curr = sqrt_info * dr_dbt_curr;
        }
        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 9, 2, Eigen::RowMajor>> dr_dbt_next(jacobians[3]);
            dr_dbt_next.setZero();

            dr_dbt_next = identity.block<9, 2>(0, 6);
            dr_dbt_next = sqrt_info * dr_dbt_next;
        }
        if (jacobians[4]) {
            Eigen::Map<Eigen::Matrix<double, 9, 1, Eigen::ColMajor>> dr_dbr_curr(jacobians[4]);
            dr_dbr_curr.setZero();

            dr_dbr_curr = -Phi.block<9, 1>(0, 8);
            dr_dbr_curr = sqrt_info * dr_dbr_curr;
        }
        if (jacobians[5]) {
            Eigen::Map<Eigen::Matrix<double, 9, 1, Eigen::ColMajor>> dr_dbr_next(jacobians[5]);
            dr_dbr_next.setZero();

            dr_dbr_next = identity.block<9, 1>(0, 8);
            dr_dbr_next = sqrt_info * dr_dbr_next;
        }
    }

    return true;
}

}
