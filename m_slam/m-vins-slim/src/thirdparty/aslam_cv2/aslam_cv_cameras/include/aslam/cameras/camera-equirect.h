#ifndef ASLAM_CAMERAS_EQUIRECT_CAMERA_H_
#define ASLAM_CAMERAS_EQUIRECT_CAMERA_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/crtp-clone.h>
#include <aslam/common/macros.h>

namespace aslam {

/// \class EquirectCamera
/// \brief An implementation of the equirect camera model. No distortion.
///
/// The usual model of a equirect camera follows these steps:
///    - Transformation: Transform the point into a coordinate w.r.t. camera.
///    - Normalization:  Project the point onto the normalized image SPHERE.
///    - Distortion:     Not applied.
///    - Projection:     Project the point into the image using a FUNC.
class EquirectCamera : public aslam::Cloneable<Camera, EquirectCamera> {
  enum { kNumOfParams = 0 };
 public:
  ASLAM_POINTER_TYPEDEFS(EquirectCamera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // TODO(slynen) Enable commented out PropertyTree support.
  // EquirectCamera(const sm::PropertyTree& config);

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators.
  /// @{

 protected:
  /// \brief Empty constructor for serialization interface.
  EquirectCamera();

 public:
  /// Copy constructor for clone operation.
  EquirectCamera(const EquirectCamera& other) = default;
  void operator=(const EquirectCamera&) = delete;

 public:
  /// \brief Construct a EquirectCamera with distortion.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  /// @param[in] distortion   Pointer to the distortion model.
  EquirectCamera(uint32_t image_width, uint32_t image_height,
                aslam::Distortion::UniquePtr& distortion);

  /// \brief Construct a EquirectCamera without distortion.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  EquirectCamera(uint32_t image_width, uint32_t image_height);

  virtual ~EquirectCamera() = default;

  /// \brief Compare this camera to another camera object.
  virtual bool operator==(const Camera& other) const;

  /// \brief Convenience function to print the state using streams.
  friend std::ostream& operator<<(
      std::ostream& out, const EquirectCamera& camera);

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project euclidean points.
  /// @{

  /// \brief Compute the 3d bearing vector in euclidean coordinates given
  ///        a keypoint in image coordinates. Uses the projection model.
  ///        The result IS in normalized image SPHERE for the general case.
  /// @param[in]  keypoint     Keypoint in image coordinates.
  /// @param[out] out_point_3d Bearing vector in euclidean coordinates.
  virtual bool backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                            Eigen::Vector3d* out_point_3d) const;

  /// \brief Checks the success of a projection operation and returns
  ///        the result in a ProjectionResult object.
  /// @param[in] keypoint Keypoint in image coordinates.
  /// @param[in] point_3d Projected point in euclidean.
  /// @return The ProjectionResult object contains details about projection.
  const ProjectionResult evaluateProjectionResult(
      const Eigen::Ref<const Eigen::Vector2d>& keypoint,
      const Eigen::Vector3d& point_3d) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Functional methods to project and back-project points.
  /// @{

  // Get the overloaded non-virtual project3Functional(..) from base into scope.
  using Camera::project3Functional;

  /// \brief Template version of project3Functional.
  template <typename ScalarType, typename DistortionType,
            typename MIntrinsics, typename MDistortion>
  const ProjectionResult project3Functional(
      const Eigen::Matrix<ScalarType, 3, 1>& point_3d,
      const Eigen::MatrixBase<MIntrinsics>& intrinsics_external,
      const Eigen::MatrixBase<MDistortion>& distortion_coefficients_external,
      Eigen::Matrix<ScalarType, 2, 1>* out_keypoint) const;

  /// \brief This function projects a point into the image using the intrinsic
  ///        parameters that are passed in as arguments. If any of the Jacobians
  ///        are nonnull, they should be filled in with the Jacobian wrt. small
  ///        changes in the argument.
  /// @param[in]  point_3d                The point in euclidean coordinates.
  /// @param[in]  intrinsics_external     External intrinsic parameter vector.
  ///                                     NOTE: If nullptr, use internal params.
  /// @param[in]  distortion_coefficients_external External distortion parameter
  ///                                     vector. Ignored if no act distortion.
  ///                                     NOTE: If nullptr, use internal params.
  /// @param[out] out_keypoint            The keypoint in image coordinates.
  /// @param[out] out_jacobian_point      The Jacobian wrt. to changes
  ///                                     in the euclidean point.
  ///                                     NOTE: If nullptr, skip calc.
  /// @param[out] out_jacobian_intrinsics The Jacobian wrt. to changes
  ///                                     in the intrinsics.
  ///                                     NOTE: If nullptr, skip calc.
  /// @param[out] out_jacobian_distortion The Jacobian wrt. to changes
  ///                                     in the distortion parameters.
  ///                                     NOTE: If nullptr, skip calc.
  /// @return Contains information about the success of the projection. Check
  ///         \ref ProjectionResult for more information.
  virtual const ProjectionResult project3Functional(
      const Eigen::Ref<const Eigen::Vector3d>& point_3d,
      const Eigen::VectorXd* intrinsics_external,
      const Eigen::VectorXd* distortion_coefficients_external,
      Eigen::Vector2d* out_keypoint,
      Eigen::Matrix<double, 2, 3>* out_jacobian_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to support unit testing.
  /// @{

  /// \brief Creates a random valid keypoint.
  virtual Eigen::Vector2d createRandomKeypoint() const;

  /// \brief Creates a random visible point. Negative depth means random between
  ///        0 and 100 meters.
  virtual Eigen::Vector3d createRandomVisiblePoint(double depth) const;

  /// \brief Get a set of border rays.
  void getBorderRays(Eigen::MatrixXd& rays) const;

  /// \brief Create a test camera object for unit testing.
  template<typename DistortionType>
  static EquirectCamera::Ptr createTestCamera() {
      aslam::Distortion::UniquePtr dis = DistortionType::createTestDistortion();
      aslam::EquirectCamera::Ptr cam(new EquirectCamera(1920, 960, dis));
      aslam::CameraId id;
      id.randomize();
      cam->setId(id);
      return cam;
  }

  /// \brief Create a test camera object for unit testing. (without distortion)
  static EquirectCamera::Ptr createTestCamera() {
      aslam::EquirectCamera::Ptr cam(new EquirectCamera(1920, 960));
      aslam::CameraId id;
      id.randomize();
      cam->setId(id);
      return cam;
  }

  /// @}

 public:
  //////////////////////////////////////////////////////////////
  /// \name Methods to access intrinsics.
  /// @{

  /// \brief Returns the camera matrix for the pinhole projection.
  Eigen::Matrix3d getCameraMatrix() const {
    Eigen::Matrix3d K;
    K.setConstant(std::numeric_limits<double>::quiet_NaN());
    LOG(FATAL) << "You shouldn't get the camera matrix of a equirect camera!";
    return K;
  }

  /// \brief Returns number of intrinsic parameters used in this camera model.
  inline static constexpr int parameterCount() {
      return kNumOfParams;
  }

  /// \brief Returns number of intrinsic parameters used in this camera model.
  inline virtual int getParameterSize() const {
      return kNumOfParams;
  }

  /// Static function checks whether the given intrinsic parameters are valid.
  static bool areParametersValid(const Eigen::VectorXd& parameters);

  /// Function checks whether the given intrinsic parameters are valid.
  virtual bool intrinsicsValid(const Eigen::VectorXd& intrinsics);

  /// Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras
  virtual void printParameters(
      std::ostream& out, const std::string& text) const;

  /// @}

 private:
  /// \brief Minimal depth for a valid projection.
  static const double kMinimumDepth;
};

}  // namespace aslam

#endif  // ASLAM_CAMERAS_EQUIRECT_CAMERA_H_
