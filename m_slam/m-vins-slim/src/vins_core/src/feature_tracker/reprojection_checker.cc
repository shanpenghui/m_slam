#include "feature_tracker/reprojection_checker.h"

namespace vins_core {

void GetVisualReprojectionError(const aslam::NCamera::Ptr& cameras,
                               const common::KeyFrames& key_frames,
                               const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                               const common::FeaturePoint& feature_point,
                               Eigen::VectorXd* dists_ptr) {
    Eigen::VectorXd& dists = *CHECK_NOTNULL(dists_ptr);
    const common::ObservationDeq& observation = feature_point.observations;
    const int anchor_frame_idx = feature_point.anchor_frame_idx;
    const common::Observation& obs_anchor = observation[anchor_frame_idx];
    const int anchor_keyframe_id = obs_anchor.keyframe_id;
    const int anchor_cam_id = obs_anchor.camera_idx;
    const int anchor_keyframe_idx = keyframe_id_to_idx.at(anchor_keyframe_id);
    const aslam::Transformation& T_OtoG_anchor = key_frames[anchor_keyframe_idx].state.T_OtoG;
    const aslam::Transformation& T_OtoC_anchor = cameras->get_T_BtoC(anchor_cam_id);
    const double inv_depth = feature_point.inv_depth;
    Eigen::Vector3d bearing_3d;
    cameras->getCamera(anchor_cam_id).backProject3(
        feature_point.observations[feature_point.anchor_frame_idx].key_point, &bearing_3d);
    bearing_3d << bearing_3d(0) / bearing_3d(2), bearing_3d(1) / bearing_3d(2), 1.0;
    const Eigen::Vector3d& p_LinC_anchor = bearing_3d / inv_depth;
    const Eigen::Vector3d p_LinO_anchor = T_OtoC_anchor.inverse().transform(p_LinC_anchor);
    const Eigen::Vector3d p_LinG = T_OtoG_anchor.transform(p_LinO_anchor);
    dists.resize(observation.size());
    for (int i = 0; i < static_cast<int>(observation.size()); ++i) {
        const common::Observation& obs_current = observation[i];
        const int current_keyframe_id = obs_current.keyframe_id;
        const int current_cam_id = obs_current.camera_idx;
        const int current_keyframe_idx = keyframe_id_to_idx.at(current_keyframe_id);
        const aslam::Transformation& T_OtoG_current = key_frames[current_keyframe_idx].state.T_OtoG;
        const aslam::Transformation& T_OtoC_current = cameras->get_T_BtoC(current_cam_id);
        const Eigen::Vector3d p_LinO_current = T_OtoG_current.inverse().transform(p_LinG);
        const Eigen::Vector3d p_LinC_current = T_OtoC_current.transform(p_LinO_current);
        Eigen::Vector2d predict_keypoint;
        cameras->getCamera(current_cam_id).project3(
                    p_LinC_current, &predict_keypoint, nullptr);
        const Eigen::Vector2d meas_keypoint = obs_current.key_point;
        const double dist = (predict_keypoint - meas_keypoint).norm() / std::sqrt(2.);
        dists(i) = dist;
    }
}

void GetMapVisualReprojectionError(const aslam::NCamera::Ptr& cameras,
                                   const common::LoopResult& loop_result,
                                   const aslam::Transformation& T_OtoG,
                                   const aslam::Transformation& T_GtoM,
                                   Eigen::VectorXd* dists_ptr) {
    Eigen::VectorXd& dists = *CHECK_NOTNULL(dists_ptr);
    const std::vector<std::pair<int, bool>>& inlier_indices = loop_result.pnp_inliers;
    const Eigen::Matrix2Xd& keypoints = loop_result.keypoints;
    const Eigen::Matrix3Xd& p_LinMs = loop_result.positions;
    const Eigen::VectorXi& cam_indices = loop_result.cam_indices;
    CHECK_EQ(keypoints.cols(), p_LinMs.cols());
    CHECK_EQ(static_cast<size_t>(keypoints.cols()), cam_indices.rows());
    dists.resize(inlier_indices.size());
    const aslam::Transformation T_MtoG = T_GtoM.inverse();
    const aslam::Transformation T_GtoO = T_OtoG.inverse();
    for (size_t i = 0u; i < inlier_indices.size(); ++i) {
        const int inlier_idx = inlier_indices[i].first;
        CHECK_LT(inlier_idx, keypoints.cols());
        const Eigen::Vector2d keypoint = keypoints.col(inlier_idx);
        const Eigen::Vector3d p_LinM = p_LinMs.col(inlier_idx);
        const int cam_idx = cam_indices(inlier_idx);
        const aslam::Transformation T_MtoC = cameras->get_T_BtoC(cam_idx) *
                T_GtoO * T_MtoG;
        const Eigen::Vector3d p_LinC = T_MtoC.transform(p_LinM);

        Eigen::Vector2d reprojection_pixel;
        const aslam::Camera& camera = cameras->getCamera(cam_idx);
        camera.project3(p_LinC, &reprojection_pixel, nullptr);

        const double reprojection_error = (keypoint - reprojection_pixel).norm() / std::sqrt(2.);
        dists(i) = reprojection_error;
    }
}

}
