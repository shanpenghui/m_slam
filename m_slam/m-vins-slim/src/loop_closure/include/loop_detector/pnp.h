#ifndef LOOP_CLOSURE_PNP_H_
#define LOOP_CLOSURE_PNP_H_

#include <Eigen/Core>
#include <aslam/cameras/ncamera.h>
#include <aslam/cameras/camera.h>

namespace loop_closure {
bool RansacP3P(
    const Eigen::Matrix2Xd& keypoints,
    const Eigen::VectorXi& cam_indices,
    const Eigen::Matrix3Xd& p_LinMs,
    const double loop_closure_sigma_pixel,
    const int pnp_num_ransac_iters,
    const aslam::NCamera::ConstPtr& cameras,
    aslam::Transformation* T_OtoM_pnp_ptr,
    std::vector<int>* inliers_ptr,
    std::vector<double>* inlier_distances_to_model_ptr,
    int* num_iters_ptr);

bool OptimizePnP(
    const Eigen::Matrix2Xd& keypoints,
    const Eigen::VectorXi& cam_indices,
    const std::vector<aslam::Transformation>& T_C0toCi,
    const Eigen::Matrix3Xd& p_LinMs,
    const std::vector<int>& inliers_init,
    const aslam::NCamera::ConstPtr& ncameras,
    const double converge_tolerance,
    const int max_num_iterations,
    const aslam::Transformation& T_OtoM_init,
    aslam::Transformation* T_OtoM_final,
    double* cost_init_ptr,
    double* cost_final_ptr,
    int* num_iterations_ptr);

void CheckInliers(
    const Eigen::Matrix2Xd& keypoints,
    const Eigen::VectorXi& cam_indices,
    const Eigen::Matrix3Xd& p_LinMs,
    const aslam::NCamera::ConstPtr& cameras,
    const std::vector<aslam::Transformation>& T_MtoC,
    const double loop_closure_sigma_pixel,
    std::vector<int>* inliers_ptr,
    std::vector<double>* inlier_distances_to_model_ptr);
}
#endif
