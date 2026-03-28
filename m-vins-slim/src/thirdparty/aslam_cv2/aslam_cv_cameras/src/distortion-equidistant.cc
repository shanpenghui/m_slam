#include <aslam/cameras/distortion-equidistant.h>
#include <cmath>

namespace aslam {
std::ostream& operator<<(std::ostream& out, const EquidistantDistortion& distortion) {
  distortion.printParameters(out, std::string(""));
  return out;
}

EquidistantDistortion::EquidistantDistortion(const Eigen::VectorXd& dist_coeffs)
: Base(dist_coeffs, Distortion::Type::kEquidistant) {
  CHECK(distortionParametersValid(dist_coeffs)) << dist_coeffs.transpose();
}

void EquidistantDistortion::distortUsingExternalCoefficients(
    const Eigen::VectorXd* dist_coeffs,
    Eigen::Vector2d* point,
    Eigen::Matrix2d* out_jacobian) const {
  CHECK_NOTNULL(point);

  double& x = (*point)(0);
  double& y = (*point)(1);

  // Use internal params if dist_coeffs==nullptr
  if(!dist_coeffs)
    dist_coeffs = &distortion_coefficients_;
  CHECK_EQ(dist_coeffs->size(), kNumOfParams) << "dist_coeffs: invalid size!";

  const double& k1 = (*dist_coeffs)(0);
  const double& k2 = (*dist_coeffs)(1);
  const double& k3 = (*dist_coeffs)(2);
  const double& k4 = (*dist_coeffs)(3);

  double x2 = x*x;
  double y2 = y*y;
  double r = std::sqrt(x2 + y2);

  // Handle special case around image center.
  if (r < 1e-10) {
    // Keypoint remains unchanged.
    if(out_jacobian)
      out_jacobian->setZero();
    return;
  }

  double theta = std::atan(r);
  double theta2 = theta * theta;
  double theta4 = theta2 * theta2;
  double theta6 = theta2 * theta4;
  double theta8 = theta4 * theta4;
  double term3_upper =
      1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
  double thetad = theta * term3_upper;

  if (out_jacobian) {
    double theta3 = theta2 * theta;
    double theta5 = theta4 * theta;
    double theta7 = theta6 * theta;

      // upper as the numerator and lower as the denominator.
      double xy = x * y;
      double x2_y2 = x2 + y2;

      double term1 = thetad / r;

      double term2_lower = (x2 + y2 + 1.0) * r * r;
      double term2_k1 = (k1 * theta * 2.0);
      double term2_k2 = (k2 * theta3 * 4.0);
      double term2_k3 = (k3 * theta5 * 6.0);
      double term2_k4 = (k4 * theta7 * 8.0);
      double term2_sum = (term2_k1 + term2_k2 + term2_k3 + term2_k4);
      double term2_upper = theta * term2_sum;

      double term3_lower = (x2_y2 * (x2_y2 + 1.0));

      double term4_lower = std::sqrt(x2_y2 * x2_y2 * x2_y2);
      double term4_upper = thetad;
      double common_factor =
          term2_upper / term2_lower +
          term3_upper / term3_lower -
          term4_upper / term4_lower;

      // Take division in the final calculation to avoid error expansion.
      const double duf_du = term1 + x2 * common_factor;

      const double duf_dv = xy * common_factor;

      const double dvf_du = duf_dv;

      const double dvf_dv = term1 + y2 * common_factor;

    *out_jacobian << duf_du, duf_dv,
                     dvf_du, dvf_dv;
  }

  double scaling = (r > 1e-8) ? thetad / r : 1.0;
  x *= scaling;
  y *= scaling;
}

void EquidistantDistortion::distortParameterJacobian(
    const Eigen::VectorXd* dist_coeffs,
    const Eigen::Vector2d& point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_EQ(dist_coeffs->size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(out_jacobian);

  const double& x = point(0);
  const double& y = point(1);

  double r = std::sqrt(x*x + y*y);
  double theta = std::atan(r);

  // Handle special case around image center.
  if (r < 1e-10) {
    out_jacobian->resize(2, kNumOfParams);
    out_jacobian->setZero();
    return;
  }

  const double duf_dk1 = x * theta * theta * theta * 1.0 / r;
  const double duf_dk2 = duf_dk1 * theta * theta;
  const double duf_dk3 = duf_dk2 * theta * theta;
  const double duf_dk4 = duf_dk3 * theta * theta;

  const double dvf_dk1 = y * theta * theta * theta * 1.0 / r;
  const double dvf_dk2 = dvf_dk1 * theta * theta;
  const double dvf_dk3 = dvf_dk2 * theta * theta;
  const double dvf_dk4 = dvf_dk3 * theta * theta;

  out_jacobian->resize(2, kNumOfParams);
  *out_jacobian << duf_dk1, duf_dk2, duf_dk3, duf_dk4,
                   dvf_dk1, dvf_dk2, dvf_dk3, dvf_dk4;
}

void EquidistantDistortion::undistortUsingExternalCoefficients(const Eigen::VectorXd& dist_coeffs,
                                                               Eigen::Vector2d* point) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(point);

  const int n = 30;  // Max. number of iterations

  Eigen::Vector2d& y = *point;
  Eigen::Vector2d ybar = y;
  Eigen::Matrix2d F;
  Eigen::Vector2d y_tmp;

  // Handle special case around image center.
  if (y.squaredNorm() < 1e-6)
    return; // Point remains unchanged.

  int i;
  for (i = 0; i < n; ++i) {
    y_tmp = ybar;
    distortUsingExternalCoefficients(&dist_coeffs, &y_tmp, &F);
    Eigen::Vector2d e(y - y_tmp);
    Eigen::Vector2d du = (F.transpose() * F).inverse() * F.transpose() * e;
    ybar += du;
    if (e.dot(e) <= aslam_cv_inv_distortion_tolerance)
      break;
  }
  LOG_IF(WARNING, i >= n) << "Did not converge with max. iterations.";

  y = ybar;
}

bool EquidistantDistortion::areParametersValid(const Eigen::VectorXd& parameters) {
  // Check the vector size.
  if (parameters.size() != kNumOfParams)
    return false;

  return true;
}

bool EquidistantDistortion::distortionParametersValid(const Eigen::VectorXd& dist_coeffs) const {
  return areParametersValid(dist_coeffs);
}

void EquidistantDistortion::printParameters(std::ostream& out, const std::string& text) const {
  Eigen::VectorXd distortion_coefficients = getParameters();
  CHECK_EQ(distortion_coefficients.size(), kNumOfParams) << "dist_coeffs: invalid size!";

  out << text << std::endl;
  out << "Distortion: (EquidistantDistortion) " << std::endl;
  out << "  k1: " << distortion_coefficients(0) << std::endl;
  out << "  k2: " << distortion_coefficients(1) << std::endl;
  out << "  k3: " << distortion_coefficients(2) << std::endl;
  out << "  k4: " << distortion_coefficients(3) << std::endl;
}

} // namespace aslam
