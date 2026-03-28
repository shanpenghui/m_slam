#ifndef VISUAL_FEATURE_DETECTOR_H_
#define VISUAL_FEATURE_DETECTOR_H_

#include <opencv2/opencv.hpp>

#include "cfg_common/slam_config.h"
#include "data_common/constants.h"

namespace vins_core {

static std::string GFTTString = "gftt";
static std::string FASTString = "fast";

class FeatureDetector {
public:
    explicit FeatureDetector(
            const common::SlamConfigPtr& config,
            const cv::Mat& mask);

    std::vector<cv::KeyPoint> Detect(const cv::Mat& img);

    enum class FeatureDetectorType {
        GFTT, 
        FAST
    };

    void SetFeatureDetectorType(
        const std::string& feature_detector_type);

private:
    void DetectKeyPointsInGrids(
        const cv::Ptr<cv::FeatureDetector>& cv_detector,
        const cv::Mat& image,
        const int num_grids_vertical,
        const int num_grids_horizontal,
        const int max_features_per_grid,
        std::vector<cv::KeyPoint>* keypoints_ptr);

    void DoNonMaximumSuppression(
        const cv::Size& image_size,
        const int suppression_radius,
        std::vector<cv::KeyPoint>* keypoints_ptr);

    const common::SlamConfigPtr config_;

    int max_feature_per_grid_ = 0;

    FeatureDetectorType feature_detector_type_;

    std::string feature_detector_type_string_;
    // OpenCV base feature detector.
    cv::Ptr<cv::FeatureDetector> cv_detector_;

    cv::Mat mask_;
};
}  // namespace visual_feature
#endif  // VISUAL_FEATURE_DETECTOR_H_
