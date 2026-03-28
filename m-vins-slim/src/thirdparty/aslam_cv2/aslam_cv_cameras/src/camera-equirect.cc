#include <memory>
#include <utility>

#include <aslam/cameras/camera-equirect.h>

#include <aslam/cameras/camera-factory.h>
#include <aslam/common/types.h>

namespace aslam {
std::ostream& operator<<(std::ostream& out,
                         const EquirectCamera& camera) {
    camera.printParameters(out, std::string(""));
    return out;
}

EquirectCamera::EquirectCamera()
    : Base(Eigen::Matrix<double, 0, 1>::Zero(),
           0, 0, Camera::Type::kEquirectangular) {
    CHECK_EQ(distortion_->getType(), Distortion::Type::kNoDistortion)
        << "Equirectangular model does not support distortion currently!";
}

EquirectCamera::EquirectCamera(uint32_t image_width, uint32_t image_height,
                               aslam::Distortion::UniquePtr& distortion)
    : Base(Eigen::Matrix<double, 0, 1>::Zero(), distortion,
           image_width, image_height, Camera::Type::kEquirectangular) {
    CHECK_EQ(distortion_->getType(), Distortion::Type::kNoDistortion)
        << "Equirectangular model does not support distortion currently!";
}

EquirectCamera::EquirectCamera(uint32_t image_width, uint32_t image_height)
    : Base(Eigen::Matrix<double, 0, 1>::Zero(), image_width, image_height,
           Camera::Type::kEquirectangular) {
    CHECK_EQ(distortion_->getType(), Distortion::Type::kNoDistortion)
        << "Equirectangular model does not support distortion currently!";
}

bool EquirectCamera::operator==(const Camera& other) const {
    // Check that the camera models are the same.
    const EquirectCamera* rhs = dynamic_cast<const EquirectCamera*>(&other);
    if (!rhs) {
        return false;
    }

    // Verify that the base members are equal.
    if (!Camera::operator==(other)) {
        return false;
    }

    // Compare the distortion model (if distortion is set for both).
    if (!(*(this->distortion_) == *(rhs->distortion_))) {
        return false;
    }

    return true;
}

bool EquirectCamera::backProject3(
    const Eigen::Ref<const Eigen::Vector2d>& kp,
    Eigen::Vector3d* out_point_3d) const {
    CHECK_NOTNULL(out_point_3d);

    // To angles.
    const double rad_x = ((kp[0] + 0.5) / imageWidth() - 1.0) * M_PI;
    const double rad_y = (1.0 - (kp[1] + 0.5) * 2.0 / imageHeight()) * M_PI_2;

    // To euclidean.
    (*out_point_3d)[0] = std::cos(rad_y) * std::sin(rad_x);
    (*out_point_3d)[1] = std::sin(rad_y) * -1.0;
    (*out_point_3d)[2] = std::cos(rad_y) * std::cos(rad_x);

    // Always valid for the equirect model.
    return true;
}

const ProjectionResult EquirectCamera::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jac_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jac_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jac_distortion) const {
    CHECK_NOTNULL(out_keypoint);

    // Nothing to do with intrinsics and distortion coefficients.
    if (out_jac_intrinsics) {
        out_jac_intrinsics->setZero(2, kNumOfParams);
    }
    if (out_jac_distortion) {
        out_jac_distortion->setZero(2, distortion_->getParameterSize());
    }

    // Start projection.
    const double& x = point_3d[0];
    const double& y = point_3d[1];
    const double& z = point_3d[2];

    const double xx = x * x;
    const double yy = y * y;
    const double zz = z * z;

    const double dxz_2 = xx + zz;
    const double dxz = std::sqrt(dxz_2);
    const double d1_2 = dxz_2 + yy;
    const double d1 = std::sqrt(d1_2);

    // To angles.
    const double h = -y / d1;
    const double rad_x = std::atan2(x, z);
    const double rad_y = std::asin(h);

    // To pixels.
    (*out_keypoint)[0] = (rad_x * M_1_PI + 1.0) * imageWidth() * 0.5 - 0.5;
    (*out_keypoint)[1] = (1.0 - rad_y * M_2_PI) * imageHeight() * 0.5 - 0.5;

    // Calculate the Jacobian w.r.t to the 3d point, if requested.
    if (out_jac_point) {
        const double d_u_d_radx = 0.5 * imageWidth() * M_1_PI;
        const double d_v_d_rady = -0.5 * imageHeight() * M_2_PI;

        // Derivation of atan2. https://www.liquisearch.com/atan2/derivative.
        const double frac_1_dxz_2 = 1.0 / dxz_2;
        const double d_radx_d_x = z * frac_1_dxz_2;
        const double d_radx_d_y = 0.0;
        const double d_radx_d_z = -x * frac_1_dxz_2;

        // Derivation of asin. https://themathpage.com/aCalc/inverse-trig.htm.
        const double frac_1_d1_2 = 1.0 / d1_2;
        const double d_rady_d_x = x * y / dxz * frac_1_d1_2;
        const double d_rady_d_y = -dxz * frac_1_d1_2;
        const double d_rady_d_z = z * y / dxz * frac_1_d1_2;

        // Finally, d_u&v / d_x&y&z.
        Eigen::Matrix<double, 2, 3>& J = *out_jac_point;
        J(0, 0) = d_u_d_radx * d_radx_d_x;
        J(1, 0) = d_v_d_rady * d_rady_d_x;

        J(0, 1) = d_u_d_radx * d_radx_d_y;
        J(1, 1) = d_v_d_rady * d_rady_d_y;

        J(0, 2) = d_u_d_radx * d_radx_d_z;
        J(1, 2) = d_v_d_rady * d_rady_d_z;
    }

    return evaluateProjectionResult(*out_keypoint, point_3d);
}

inline const ProjectionResult EquirectCamera::evaluateProjectionResult(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint,
    const Eigen::Vector3d& point_3d) const {

    // Check the validity more strictly if required.
    constexpr bool kDisableLoop = true;
    bool visibility = false;
    if (kDisableLoop) {
        visibility = keypoint[0] >= 0.0 &&
                     keypoint[0] <= imageWidth() - 1.0 &&
                     keypoint[1] >= 0.0 &&
                     keypoint[1] <= imageHeight() - 1.0;
    } else {
        visibility = isKeypointVisible(keypoint);
    }

    const double d2 = point_3d.squaredNorm();
    const double minDepth2 = kMinimumDepth*kMinimumDepth;

    if (d2 > minDepth2) {
        if (visibility) {
            return ProjectionResult(
                   ProjectionResult::Status::KEYPOINT_VISIBLE);
        } else {
            return ProjectionResult(
                   ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX);
        }
    }

    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}

Eigen::Vector2d EquirectCamera::createRandomKeypoint() const {
    Eigen::Vector2d out;
    out.setRandom();
    // Keep the point away from the border.
    double border = std::min(imageWidth(), imageHeight()) * 0.1;
    out(0) = border + std::abs(out(0)) * (imageWidth() - border * 2.0);
    out(1) = border + std::abs(out(1)) * (imageHeight() - border * 2.0);
    return out;
}

Eigen::Vector3d EquirectCamera::createRandomVisiblePoint(double depth) const {
    CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
    Eigen::Vector3d point_3d;

    Eigen::Vector2d y = createRandomKeypoint();
    backProject3(y, &point_3d);
    point_3d /= point_3d.norm();

    // Muck with the depth. This doesn't change the pointing direction.
    return point_3d * depth;
}

bool EquirectCamera::areParametersValid(const Eigen::VectorXd& parameters) {
    // Size should be empty.
    return parameters.size() == parameterCount();
}

bool EquirectCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) {
    return areParametersValid(intrinsics);
}

void EquirectCamera::printParameters(
    std::ostream& out, const std::string& text) const {
    Camera::printParameters(out, text);
    out << "  distortion: ";
    distortion_->printParameters(out, text);
}
const double EquirectCamera::kMinimumDepth = 1e-10;
}  // namespace aslam
