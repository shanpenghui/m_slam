// Copyright (c) Alibaba Inc. All rights reserved.

#ifndef ASLAM_PLUMB_BOB_DISTORTION_H_
#define ASLAM_PLUMB_BOB_DISTORTION_H_

#include <string>

#include <aslam/common/crtp-clone.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/macros.h>
#include <Eigen/Core>
#include <glog/logging.h>

namespace aslam {

// An implementation of the standard plumb-bob
// distortion model for pinhole cameras.
// Three radial (k1, k2, k3) and tangential (p1, p2)
// parameters are used in this implementation.
// The ordering of the parameter vector is:
// k1 k2 p1 p2 k3
// NOTE: The inverse transformation (undistort)
// in this case is not available in
// closed form and so it is computed iteratively!
class PlumbBobDistortion :
        public aslam::Cloneable<Distortion, PlumbBobDistortion> {
 public:
  // Number of parameters used for this distortion model.
  enum {kNumOfParams = 5};

  enum {CLASS_SERIALIZATION_VERSION = 1};
  ASLAM_POINTER_TYPEDEFS(PlumbBobDistortion);

  // distortionParams Vector containing the distortion parameter.
  // (dim=5: k1, k2, p1, p2, k3)
  explicit PlumbBobDistortion(
        const Eigen::VectorXd& distortionParams);

  // Convenience function to print the state using streams.
  friend std::ostream& operator<<(
        std::ostream& out,
        const PlumbBobDistortion& distortion);

  // Copy constructor for clone operation.
  PlumbBobDistortion(const PlumbBobDistortion&) = default;
  void operator=(const PlumbBobDistortion&) = delete;

  // Apply distortion to a point in the normalized image plane
  // using provided distortion coefficients.
  // External distortion coefficients can be
  // specified using this function.
  // Ignores the internally stored parameters.
  // dist_coeffs is the vector containing the
  // coefficients for the distortion model.
  // NOTE: If dist_coeffs is nullptr, use internal distortion parameters.
  // The point in the normalized image plane. After the function,
  // this point is distorted.
  // out_jacobian is the Jacobian of the distortion
  // function with respect to small changes in the input point.
  // If NULL is passed, the Jacobian calculation is skipped.
  virtual void distortUsingExternalCoefficients(
        const Eigen::VectorXd* dist_coeffs,
        Eigen::Vector2d* point,
        Eigen::Matrix2d* out_jacobian) const;

  // Apply distortion to the point and provide
  // the Jacobian of the distortion with respect
  // to small changes in the distortion parameters.
  // dist_coeffs id the vector containing the coefficients
  // for the distortion model.
  // NOTE: If dist_coeffs is nullptr, use internal distortion parameters.
  // point is the point in the normalized image plane.
  // out_jacobian is the Jacobian of the distortion
  // with respect to small changes in the distortion parameters.
  virtual void distortParameterJacobian(
        const Eigen::VectorXd* dist_coeffs,
        const Eigen::Vector2d& point,
        Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const;

  // Apply undistortion to recover a point in the normalized image plane
  // using provided distortion coefficients.
  // External distortion coefficients can be specified using this function.
  // Ignores the internally  stored parameters.
  // dist_coeffs is the vector containing the coefficients
  // for the distortion model.
  // After the function, point is in the normalized image plane.
  virtual void undistortUsingExternalCoefficients(
        const Eigen::VectorXd& dist_coeffs,
        Eigen::Vector2d* point) const;

  // Methods to support unit testing.

  // Create a test distortion object for unit testing.
  PlumbBobDistortion::UniquePtr createTestDistortion() {
      Eigen::VectorXd params(5); params
              << -0.28,  0.08, -0.00026, -0.00024, -0.0002;
      return PlumbBobDistortion::UniquePtr(
              new PlumbBobDistortion(params));
  }

  // Create a test distortion object for
  // unit testing with null distortion.
  PlumbBobDistortion::UniquePtr createZeroTestDistortion() {
      Eigen::VectorXd params(4); params
              << 0.0, 0.0, 0.0, 0.0, 0.0;
      return PlumbBobDistortion::UniquePtr(
              new PlumbBobDistortion(params));
  }

  // Methods to set/get distortion parameters

  // Check whether the given intrinsic parameters
  // are valid for this model.
  static bool areParametersValid(
          const Eigen::VectorXd& parameters);

  // Check the validity of distortion parameters.
  virtual bool distortionParametersValid(
          const Eigen::VectorXd& dist_coeffs) const;

  // Returns the number of parameters
  // used in this distortion model.
  inline size_t parameterCount() {
      return kNumOfParams;
  }

  // Returns the number of parameters used in the distortion model.
  // NOTE: Use the constexpr function parameterCount
  // if you know the exact distortion type.
  virtual int getParameterSize() const {
      return kNumOfParams;
  }

  // Print the internal parameters of the distortion
  // in a human-readable form
  // Print to the ostream that is passed in.
  // The text is extra text used by the
  // calling function to distinguish cameras.
  virtual void printParameters(
          std::ostream& out, const std::string& text) const;

};
}  // namespace aslam

#endif  //ASLAM_PLUMB_BOB_DISTORTION_H_
