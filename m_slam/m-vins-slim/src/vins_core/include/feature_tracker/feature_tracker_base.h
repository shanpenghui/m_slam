#ifndef MVINS_FEATURE_TRECKER_H_
#define MVINS_FEATURE_TRECKER_H_

#include <aslam/cameras/ncamera.h>

#include "data_common/visual_structures.h"
#ifdef USE_CNN_FEATURE
#include "feature_tracker/super_point_infer.h"
#endif
#include "feature_tracker/feature_detector.h"
#include "feature_tracker/feature_extractor.h"

namespace vins_core {

class FeatureTrackerBase {
public:
    virtual ~FeatureTrackerBase() = default;

#ifdef USE_CNN_FEATURE
    SuperPointConfig CreateSuperPointConfig(
            const common::SlamConfigPtr& config);

    void RestoreKeypoints(common::VisualFrameData* frame_data, 
                          const int infer_img_width,
                          const int infer_img_height);

    virtual void InferFeature(
            const uint64 timestamp_ns,
            const common::CvMatConstPtr& img,
            common::VisualFrameData* visual_frame_data_ptr,
            int* track_id_provider_ptr);
#endif
    virtual void DetectAndExtractFeature(
            const uint64 timestamp_ns,
            const common::CvMatConstPtr& img,
            common::VisualFrameData* visual_frame_data_ptr,
            int* track_id_provider_ptr);

    virtual void TrackFeature(
            const int cam_idx,
            const Eigen::Quaterniond& q_kp1_k,
            const common::VisualFrameData& frame_data_k,
            common::VisualFrameData* frame_data_kp1_ptr,
            common::FrameToFrameMatchesWithScore* matches_ptr,
            int* track_id_provider_ptr) {
        LOG(FATAL) << "We have no default implementation!";
    }

    void LengthFiltering(
            const int cam_idx,
            const common::VisualFrameData& frame_data_k,
            const common::VisualFrameData& frame_data_kp1,
            common::FrameToFrameMatchesWithScore* matches_ptr);

    void RansacFiltering(
            const aslam::Camera& camera,
            const Eigen::Matrix3d& R_kp1_k,
            const Eigen::Matrix6Xd& keypoints_k,
            const Eigen::Matrix6Xd& keypoints_kp1,
            common::FrameToFrameMatchesWithScore* matches_kp1_k_ptr);
protected:
    explicit FeatureTrackerBase(
            const aslam::NCamera::Ptr& cameras,
            const common::SlamConfigPtr& config);

    void PredictKeypointsByRotation(
        const aslam::Camera& camera,
        const Eigen::Matrix2Xd keypoints_k,
        const Eigen::Quaterniond& q_kp1_k,
        Eigen::Matrix2Xd* predicted_keypoints_kp1,
        std::vector<unsigned char>* prediction_success);

    template <typename Type>
    void EraseVectorElementsByIndex(
            const std::unordered_set<size_t>& indices_to_erase,
            std::vector<Type>* vec) const;

    const aslam::NCamera::Ptr cameras_;
    const common::SlamConfigPtr config_;
#ifdef USE_CNN_FEATURE
    std::unique_ptr<SuperPoint> superpoint_infer_ptr_;
#endif
    std::unique_ptr<FeatureDetector> feature_detector_;
    std::unique_ptr<FeatureExtractor> feature_extractor_;

private:
    void DetectAndExtractFeatureImpl(
        const common::CvMatConstPtr& img,
        std::vector<cv::KeyPoint>* keypoints_ptr,
        cv::Mat* descriptors_ptr);

    bool TwoPointRansac(
        const aslam::Camera& camera,
        const Eigen::Matrix3d& R_kp1_k,
        const Eigen::Matrix2Xd& keypoints_kp1_2d,
        const Eigen::Matrix2Xd& keypoints_k_2d,
        const bool enough_rotation,
        const int ransac_max_iterations,
        const double ransac_threshold,
        std::vector<unsigned char>* is_valid_ptr);
};
}  // namespace vins_core
#endif  // MVINS_FEATURE_TRECKER_H_
