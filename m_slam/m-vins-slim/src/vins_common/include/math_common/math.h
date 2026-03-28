#ifndef MATH_COMMON_MATH_H_
#define MATH_COMMON_MATH_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <aslam/common/pose-types.h>

namespace common {
// Clamps 'value' to be in the range ['min', 'max'].
template <typename T>
T Clamp(const T value, const T min, const T max) {
  if (value > max) {
    return max;
  }
  if (value < min) {
    return min;
  }
  return value;
}

// Calculates 'base'^'exponent'.
template <typename T>
constexpr T Power(T base, int exponent) {
  return (exponent != 0) ? base * Power(base, exponent - 1) : T(1);
}

// Calculates a^2.
template <typename T>
constexpr T Pow2(T a) {
  return Power(a, 2);
}


template <typename T>
static T NormalizeRad(T difference) {
  while (difference > M_PI) {
    difference -= T(2. * M_PI);
  }
  while (difference < -M_PI) {
    difference += T(2. * M_PI);
  }
  return difference;
}

template <typename T>
static T NormalizeDeg(T difference) {
  while (difference > 180.) {
    difference -= T(2. * 180.);
  }
  while (difference < -180.) {
    difference += T(2. * 180.);
  }
  return difference;
}

static inline bool IsLessThanEpsilons4thRoot(double x) {
    static const double epsilon4thRoot =
        std::pow(std::numeric_limits<double>::epsilon(), 1.0/4.0);
    return x < epsilon4thRoot;
}

// Note: this is the exp map of SO(3)xR(3) and not SE(3).
static inline aslam::Transformation Exp(const Eigen::Matrix<double, 6, 1>& vec) {
    // Translation component, J*rho is translation.
    const Eigen::Vector3d& rho = vec.head<3>();
    // Rotation component, in angle-axis form.
    const Eigen::Vector3d& phi = vec.tail<3>();

    // Method of implementing this function that is accurate to
    // numerical precision from Grassia, F. S. (1998).
    // Practical parameterization of rotations using the exponential map.
    // Journal of graphics, gpu, and game tools, 3(3):29–48.
    const double theta = phi.norm();
    // na is 1/theta sin(theta/2).
    double na;
    if (IsLessThanEpsilons4thRoot(theta)) {
        constexpr double one_over_48 = 0.0208333;
        na = 0.5 + (theta * theta) * one_over_48;
    } else {
        na = std::sin(theta*0.5) / theta;
    }
    const double ct = std::cos(theta*0.5);
    Eigen::Quaterniond q(ct, phi[0]*na, phi[1]*na, phi[2]*na);
    return aslam::Transformation(q, rho);
}

inline Eigen::Matrix<double, 3, 3> skew_x(const Eigen::Matrix<double, 3, 1> &w) {
    Eigen::Matrix<double, 3, 3> w_x;
    w_x << 0, -w(2), w(1),
            w(2), 0, -w(0),
            -w(1), w(0), 0;
    return w_x;
}

inline Eigen::Vector3d RotToEuler(
    const Eigen::Matrix3d& R) {
    // Rotation order: yaw first, pitch second, roll third.
    const double cosy =
            std::sqrt(R(0, 0) * R(0, 0) + R(0, 1) * R(0, 1));
    const bool singular = cosy < 1e-6;
    // rpy: [roll, pitch, yaw]
    Eigen::Vector3d rpy;
    if (!singular) {
        rpy(0) = std::atan2(-R(1, 2), R(2, 2));
        rpy(1) = std::atan2(R(0, 2), cosy);
        rpy(2) = std::atan2(-R(0, 1), R(0, 0));
    } else {  // Gimbal lock when pitch == ±pi/2.
        rpy(0) = 0;
        rpy(1) = std::asin(R(0, 2));
        rpy(2) = std::atan2(R(1, 0), R(1, 1));
    }
    return rpy;
}

inline Eigen::Matrix3d EulerToRot(
        const Eigen::Vector3d& euler) {
    Eigen::Matrix3d r_x;
    r_x << 1.0, 0.0, 0.0,
            0.0, cos(euler(0)), -sin(euler(0)),
            0.0, sin(euler(0)), cos(euler(0));

    Eigen::Matrix3d r_y;
    r_y << cos(euler(1)), 0.0, sin(euler(1)),
            0.0, 1.0, 0.0,
            -sin(euler(1)), 0, cos(euler(1));

    Eigen::Matrix3d r_z;
    r_z << cos(euler(2)), -sin(euler(2)), 0.0,
            sin(euler(2)), cos(euler(2)), 0.0,
            0.0, 0.0, 1.0;

    return r_x * r_y * r_z;
}

inline Eigen::Vector3d QuatToEuler(
    const Eigen::Quaterniond& q) {
    return RotToEuler(q.matrix());
}

inline Eigen::Quaterniond EulerToQuat(
    const Eigen::Vector3d& e) {
    return Eigen::Quaterniond(EulerToRot(e));
}
}

#endif
