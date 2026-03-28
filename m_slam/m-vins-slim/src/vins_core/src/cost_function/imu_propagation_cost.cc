#include "cost_function/imu_propagation_cost.h"

namespace vins_core {

ImuPropagationCost::ImuPropagationCost(
        const common::SlamConfigPtr& config,
        const common::ImuDatas& propa_data)
    : config_(config),
      propa_data_(propa_data) {
    // Pose
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    mutable_parameter_block_sizes()->push_back(common::kGlobalPoseSize);
    // v
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);
    // bg
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);
    // ba
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);

    set_num_residuals(15);
}

bool ImuPropagationCost::Evaluate(double const *const *parameters,
                                   double *residuals_ptr,
                                   double **jacobians) const {
    Eigen::Map<const Eigen::Vector3d> p_OinG_curr(parameters[0]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_curr(parameters[0] + 3);
    Eigen::Map<const Eigen::Vector3d> p_OinG_next(parameters[1]);
    Eigen::Map<const Eigen::Quaterniond> q_OtoG_next(parameters[1] + 3);
    Eigen::Map<const Eigen::Vector3d> v_curr(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> v_next(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> bg_curr(parameters[4]);
    Eigen::Map<const Eigen::Vector3d> bg_next(parameters[5]);
    Eigen::Map<const Eigen::Vector3d> ba_curr(parameters[6]);
    Eigen::Map<const Eigen::Vector3d> ba_next(parameters[7]);

    vins_core::ImuPropagator::NoiseManager noise_manager(config_);
    vins_core::ImuPropagator imu_propagator(noise_manager);

    aslam::Transformation T_OtoG_curr(q_OtoG_curr, p_OinG_curr);
    aslam::Transformation T_OtoG_next(q_OtoG_next, p_OinG_next);
    common::State state_curr(T_OtoG_curr, v_curr, bg_curr, ba_curr);
    common::State state_next(T_OtoG_next, v_next, bg_next, ba_next);

    const double pose_diff = (state_next - state_curr).norm();

    common::State state_next_estimated;
    Eigen::Matrix<double, 15, 15> Phi = Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix<double, 15, 15> Q = Eigen::Matrix<double, 15, 15>::Zero();
    if (pose_diff < 1e-6) {
        state_next_estimated = state_next;
    } else{
        imu_propagator.Propagate(propa_data_, &state_curr, &Phi, &Q);
        state_next_estimated = state_curr;
    }

    if (Phi.maxCoeff() > 1e8 || Phi.minCoeff() < -1e8) {
        LOG(FATAL) << "Numerical unstable in pre-integration.";
    }

    Eigen::LLT<Eigen::Matrix<double, 15, 15>> cholesky_solver;
    cholesky_solver.compute(Q);
    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::Matrix<double, 15, 15>::Identity();
    cholesky_solver.matrixL().solveInPlace(sqrt_info);
#ifdef DEBUG
    CHECK(!sqrt_info.hasNaN()) << "IMU meas size: " << propa_data_.size() << " Q: "
                               << std::endl << Q;
#endif

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residuals_map(residuals_ptr);
    residuals_map = (state_next - state_next_estimated).head<15>();
    residuals_map = sqrt_info * residuals_map;
#ifdef DEBUG
    CHECK(!residuals_map.hasNaN()) << residuals_map.transpose();
#endif

    if (jacobians) {
        Eigen::Matrix<double, 15, 15> identity = Eigen::Matrix<double, 15, 15>::Identity();
        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 15, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_curr(jacobians[0]);
            dr_dT_curr.setZero();

            dr_dT_curr.topLeftCorner<15, common::kLocalPoseSize>() =
                    -Phi.topLeftCorner<15, common::kLocalPoseSize>();
            dr_dT_curr = sqrt_info * dr_dT_curr;
        }
        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 15, common::kGlobalPoseSize, Eigen::RowMajor>> dr_dT_next(jacobians[1]);
            dr_dT_next.setZero();

            dr_dT_next.topLeftCorner<15, common::kLocalPoseSize>() =
                    identity.topLeftCorner<15, common::kLocalPoseSize>();
            dr_dT_next = sqrt_info * dr_dT_next;
        }
        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dv_curr(jacobians[2]);
            dr_dv_curr.setZero();

            dr_dv_curr = -Phi.block<15, 3>(0, 6);
            dr_dv_curr = sqrt_info * dr_dv_curr;
        }
        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dv_next(jacobians[3]);
            dr_dv_next.setZero();

            dr_dv_next = identity.block<15, 3>(0, 6);
            dr_dv_next = sqrt_info * dr_dv_next;
        }
        if (jacobians[4]) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dbg_curr(jacobians[4]);
            dr_dbg_curr.setZero();

            dr_dbg_curr = -Phi.block<15, 3>(0, 9);
            dr_dbg_curr = sqrt_info * dr_dbg_curr;
        }
        if (jacobians[5]) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dbg_next(jacobians[5]);
            dr_dbg_next.setZero();

            dr_dbg_next = identity.block<15, 3>(0, 9);
            dr_dbg_next = sqrt_info * dr_dbg_next;
        }
        if (jacobians[6]) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dba_curr(jacobians[6]);
            dr_dba_curr.setZero();

            dr_dba_curr = -Phi.block<15, 3>(0, 12);
            dr_dba_curr = sqrt_info * dr_dba_curr;
        }
        if (jacobians[7]) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> dr_dba_next(jacobians[7]);
            dr_dba_next.setZero();

            dr_dba_next = identity.block<15, 3>(0, 12);
            dr_dba_next = sqrt_info * dr_dba_next;
        }
    }
    return true;
}
}
