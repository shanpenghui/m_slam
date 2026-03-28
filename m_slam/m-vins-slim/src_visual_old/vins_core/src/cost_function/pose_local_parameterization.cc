#include "cost_function/pose_local_parameterization.h"

#include "data_common/constants.h"

namespace vins_core {
Eigen::Quaterniond QuatermionPlusHamilton(const Eigen::Quaterniond& q,
                                          const Eigen::Vector3d& delta) {
    Eigen::Quaterniond q_plus;

    const double delta_norm = delta.norm();
    if (delta_norm > 0.0) {
        Eigen::Vector3d half_delta = delta;
        half_delta /= 2.0;
        Eigen::Quaterniond delta_q;
        delta_q.w() = 1.0;
        delta_q.x() = half_delta.x();
        delta_q.y() = half_delta.y();
        delta_q.z() = half_delta.z();
                                   
        q_plus = q * delta_q;
    } else {
        q_plus = q;
    }
    q_plus.normalize();
    q_plus = common::Positify(q_plus);

    return q_plus;
}

bool Pose3DLocalParameterization::Plus(
        const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);
     Eigen::Map<const Eigen::Vector3d> dtheta(delta + 3);

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);


    q = QuatermionPlusHamilton(_q, dtheta);
    p = _p + dp;

    return true;
}

bool Pose3DLocalParameterization::ComputeJacobian(
        const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, common::kGlobalPoseSize,
                                     common::kLocalPoseSize,
                                     Eigen::RowMajor>> J(jacobian);
    J.topRows<common::kLocalPoseSize>().setIdentity();
    J.bottomRows<1>().setZero();

    return true;
}

bool Pose2DLocalParameterization::Plus(
        const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    const Eigen::Vector3d dp(delta[0], delta[1], 0);
    const Eigen::Vector3d dtheta(0, 0, delta[2]);

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = QuatermionPlusHamilton(_q, dtheta);

    return true;
}

bool Pose2DLocalParameterization::ComputeJacobian(
        const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 3, Eigen::RowMajor>> j(jacobian);
    j.setZero();

    j.topLeftCorner<2, 2>().setIdentity();
    j(5, 2) = 1.0;

    return true;
}

}
