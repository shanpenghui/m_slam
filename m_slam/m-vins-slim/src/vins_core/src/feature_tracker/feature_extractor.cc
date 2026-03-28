#include "feature_tracker/feature_extractor.h"

#include "opencv2/xfeatures2d.hpp"

namespace vins_core {
FeatureExtractor::FeatureExtractor(
        const common::SlamConfigPtr& config)
    : config_(config) {
    cv_extractor_ = cv::xfeatures2d::FREAK::create(
        config_->freak_orientation_normalized,
        config_->freak_scale_normalized,
        config_->freak_pattern_scale,
        config_->freak_num_octaves,
        std::vector<int>());
}

void FeatureExtractor::Extract(
        const cv::Mat& img,
        std::vector<cv::KeyPoint>* keypoints_ptr,
        cv::Mat* descriptors_ptr) {
    std::vector<cv::KeyPoint>& keypoints = *CHECK_NOTNULL(keypoints_ptr);
    cv::Mat& descriptors = *CHECK_NOTNULL(descriptors_ptr);

    if (keypoints.empty()) {
        return;
    }

    cv_extractor_->compute(img,
                           keypoints,
                           descriptors);
}
}  // namespace visual_feature
