

#ifndef VISUAL_FEATURE_EXTRACTOR_H_
#define VISUAL_FEATURE_EXTRACTOR_H_

#include <opencv2/opencv.hpp>

#include "cfg_common/slam_config.h"
#include "data_common/constants.h"

namespace vins_core {
class FeatureExtractor {
public:
    explicit FeatureExtractor(
            const common::SlamConfigPtr& config);

    void Extract(const cv::Mat& img,
                std::vector<cv::KeyPoint>* keypoints_ptr,
                cv::Mat* descriptors_ptr);
private:
    const common::SlamConfigPtr config_;

    // OpenCV base feature extractor.
    cv::Ptr<cv::DescriptorExtractor> cv_extractor_;
};
}  // namespace visual_feature
#endif  // VISUAL_FEATURE_EXTRACTOR_H_
