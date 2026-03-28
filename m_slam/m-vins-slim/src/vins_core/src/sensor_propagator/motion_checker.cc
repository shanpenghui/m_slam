#include "sensor_propagator/motion_checker.h"

#include "data_common/constants.h"
#include "math_common/math.h"

namespace vins_core {

MotionChecker::MotionChecker(const common::SlamConfigPtr& config,
                             const double min_trans,
                             const double min_rot)
    : min_trans_(min_trans),
      min_rot_(min_rot) {
    odom_propagator_ptr_.reset(new OdomPropagator);
    vins_core::ImuPropagator::NoiseManager noise_manager(config);
    imu_propagator_ptr_.reset(
                new vins_core::ImuPropagator(noise_manager));
}

bool MotionChecker::MotionChecking(const common::State& state) {
    const Eigen::Vector3d& p = state.T_OtoG.getPosition();
    const Eigen::Quaterniond& q = state.T_OtoG.getEigenQuaternion();
    const double trans_norm = p.norm();
    Eigen::Vector3d euler = common::QuatToEuler(q);
    const double euler_norm = euler.norm();
    const bool is_motion_enough = (trans_norm > min_trans_) ||
             (common::kRadToDeg * euler_norm > min_rot_);

    return is_motion_enough;
}

bool MotionChecker::CheckIsMotionEnough(const common::OdomDatas& odom_data) {
    CHECK_NOTNULL(odom_propagator_ptr_);
    if (odom_data.size() < 2u) {
        return false;
    }
    common::State state;
    odom_propagator_ptr_->Propagate(odom_data, &state);
    return (MotionChecking(state));
}

bool MotionChecker::CheckIsMotionEnough(const common::ImuDatas& imu_data) {
    CHECK_NOTNULL(imu_propagator_ptr_);
    if (imu_data.size() < 2u) {
        return false;
    }
    common::State state;
    imu_propagator_ptr_->Propagate(imu_data, &state);
    return (MotionChecking(state));
}

}
