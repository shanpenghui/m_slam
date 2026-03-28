#ifndef MATH_COMMON_NANO_EKF_H_
#define MATH_COMMON_NANO_EKF_H_

#include <Eigen/Core>
#include <aslam/common/pose-types.h>

#include "data_common/sensor_structures.h"
#include "math_common/math.h"
#include "sensor_propagator/odom_propagator.h"

namespace common {

typedef Eigen::Matrix<double, 6, 1> StateVector;
typedef Eigen::Matrix<double, 6, 6> StateTransitionMatrix;
typedef Eigen::Matrix<double, 6, 6> MeasurementNoiseMatrix;
typedef Eigen::Matrix<double, 6, 6> ProcessNoiseMatrix;

class NanoEKF {
public:
NanoEKF(const aslam::Transformation& init) {
    x_ = init.log();

    Q_ = ProcessNoiseMatrix::Identity() * 1e-4;

    R_ = MeasurementNoiseMatrix::Identity() * 1e-2;

    P_ = ProcessNoiseMatrix::Identity() * 1e-2;
}

void Predict() {
    StateTransitionMatrix F = StateTransitionMatrix::Identity();

    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q_;
}

void Predict(const aslam::Transformation& x) {
    StateTransitionMatrix F = StateTransitionMatrix::Identity();

    x_ = F * x.log();
    P_ = F * P_ * F.transpose() + Q_;
}

void Predict(const common::OdomDatas& odoms) {
    aslam::Transformation old_x(x_);
    common::State state(old_x, Eigen::Vector2d::Zero(), 0.0);

    vins_core::OdomPropagator odom_propagator;
    odom_propagator.Propagate(odoms, &state, nullptr, nullptr);

    const aslam::Transformation& new_x = state.T_OtoG;

    StateTransitionMatrix F = StateTransitionMatrix::Identity();

    x_ = F * new_x.log();
    P_ = F * P_ * F.transpose() + Q_;
}

void Predict(const Eigen::Vector3d& w, const Eigen::Vector3d& v, const double dt) {
    Eigen::Quaterniond q_plus = common::EulerToQuat(w * dt);
    Eigen::Vector3d t_plus = v * dt;
    aslam::Transformation old_x(x_);
    Eigen::Quaternion new_q = old_x.getEigenQuaternion() * q_plus;
    Eigen::Vector3d new_t = old_x.getPosition() + t_plus;
    const aslam::Transformation new_x(new_q, new_t);

    StateTransitionMatrix F = StateTransitionMatrix::Identity();

    x_ = F * new_x.log();
    P_ = F * P_ * F.transpose() + Q_;
}

 aslam::Transformation Update(const aslam::Transformation& pose_z, const Eigen::MatrixXd& H) {
    StateVector z =  pose_z.log();
    StateVector y = z - x_;
    for (int i = 3; i < 6; ++i) {
        y(i) = common::NormalizeRad(y(i));
    }
    Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + R_).inverse();

    x_ += K * y;
    P_ = (Eigen::Matrix<double, 6, 6>::Identity() - K * H) * P_;

    return common::Exp(x_);
}

private:
    StateVector x_;
    ProcessNoiseMatrix Q_;
    MeasurementNoiseMatrix R_;
    ProcessNoiseMatrix P_;
};

}

#endif