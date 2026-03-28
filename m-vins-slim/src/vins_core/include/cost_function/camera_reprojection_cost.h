#ifndef MVINS_COST_FUNCTION_CAMERA_REPROJECTION_COST_H_
#define MVINS_COST_FUNCTION_CAMERA_REPROJECTION_COST_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <ceres/ceres.h>

#include "data_common/constants.h"

namespace vins_core {
class CameraReprojectionCost : public ceres::CostFunction {
public:
   CameraReprojectionCost(
           const aslam::NCamera::Ptr& cameras,
           const Eigen::Vector2d& keypoint_anchor,
           const Eigen::Vector2d& keypoint_current,
           const Eigen::Vector2d& velocity_anchor,
           const Eigen::Vector2d& velocity_current,
           const double use_depth,
           const int cam_idx_anchor,
           const int cam_idx_currnet,
           const double depth_current,
           const double td_anchor,
           const double td_current,
           const double sigma_pixel,
           const double sigma_depth);

   virtual bool Evaluate(double const * const *parameters, double *residuals,
                         double **jacobians) const;
private:
   const aslam::NCamera::Ptr cameras_;
   Eigen::Vector2d meas_anchor_pixel_;
   Eigen::Vector2d meas_current_pixel_;
   Eigen::Vector2d velocity_anchor_pixel_;
   Eigen::Vector2d velocity_current_pixel_;
   const bool use_depth_;
   const int cam_idx_anchor_;
   const int cam_idx_current_;
   const double meas_depth_current_;
   const double meas_td_anchor_;
   const double meas_td_current_;
   Eigen::MatrixXd sqrt_info_;
};

class CameraReprojectionFromMapCost : public ceres::CostFunction {
public:
   CameraReprojectionFromMapCost(
           const aslam::NCamera::Ptr& cameras,
           const Eigen::Vector3d& meas_p_LinM,
           const Eigen::Vector2d& meas_pixel,
           const int cam_idx,
           const bool use_depth,
           const double meas_depth,
           const double sigma_pixel,
           const double sigma_depth);

   virtual bool Evaluate(double const * const *parameters, double *residuals,
                         double **jacobians) const;
private:
   const aslam::NCamera::Ptr cameras_;
   const Eigen::Vector3d meas_p_LinM_;
   const Eigen::Vector2d meas_pixel_;
   const int cam_idx_;
   const bool use_depth_;
   const double meas_depth_;
   Eigen::MatrixXd sqrt_info_;
};

class BearingEuclideanCost : public ceres::CostFunction {
public:
    BearingEuclideanCost(const aslam::NCamera::Ptr& cameras,
                         const Eigen::Vector2d& meas_bearing,
                         const int cam_idx,
                         const double sigma_pixel);

    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;

private:
   const aslam::NCamera::Ptr cameras_;
   const Eigen::Vector2d meas_bearing_;
   const int cam_idx_;
   Eigen::Matrix2d sqrt_info_;
};

}

#endif
