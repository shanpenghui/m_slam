#include "feature_tracker/feature_detector.h"

#include <opencv2/xfeatures2d.hpp>

namespace vins_core {
FeatureDetector::FeatureDetector(
        const common::SlamConfigPtr& config,
        const cv::Mat& mask)
    : config_(config),
      mask_(mask) {
    // Compute max num of features to detect for each grid.
    if (config_->use_grids) {
        // If grid detection is enabled, we will try to detect more features in
        // each grid and then eliminate the weak features based on responses.
        // In this way, we can end up with closer to desired number of features.
        const int num_grids =
                config_->num_grids_horizontal * config_->num_grids_vertical;
        CHECK_GE(num_grids, 1);
        max_feature_per_grid_ = static_cast<int>(
            config_->num_feature_to_detect / num_grids) + 1;
    } else {
        max_feature_per_grid_ = static_cast<int>(
            config_->num_feature_to_detect);
    }
    SetFeatureDetectorType(config_->feature_detector_type);
    
    // Init cv_detector_.
    switch (feature_detector_type_)
    {
        case FeatureDetectorType::FAST: {
            cv_detector_ = cv::ORB::create(
                max_feature_per_grid_,
                config_->fast_scale_factor,
                config_->fast_pyramid_levels,
                0,  // size of no search border
                0,  // first level, always 0
                2,  // WTA_K, pts in each comparison
                cv::ORB::FAST_SCORE,
                config_->fast_patch_size,
                config_->fast_circle_threshold);
            break;
        }
        case FeatureDetectorType::GFTT: {
            cv_detector_ = cv::GFTTDetector::create(
                max_feature_per_grid_,
                config_->gftt_quality_level,
                config_->gftt_min_distance,
                config_->gftt_block_size,
                config_->gftt_use_harris_detector,
                config_->gftt_k);
            break;
        }
        default: { 
            LOG(FATAL) << "Invalid feature detector type: " 
                        << feature_detector_type_string_;
            break;
        }
    }
}

std::vector<cv::KeyPoint> FeatureDetector::Detect(const cv::Mat& img) {
    std::vector<cv::KeyPoint> keypoints;

    if (config_->use_grids) {
        DetectKeyPointsInGrids(
            cv_detector_,
            img,
            config_->num_grids_vertical,
            config_->num_grids_horizontal,
            max_feature_per_grid_,
            &keypoints);
        
    } else {
        cv_detector_->detect(img, keypoints, mask_);
    }

    if (feature_detector_type_ == FeatureDetectorType::FAST) {
        // Remove the keypoints scored low.
        static const double kFastScoreLowerBound =
        config_->fast_score_lower_bound;
        auto iter = std::remove_if(keypoints.begin(), keypoints.end(),
            [](const cv::KeyPoint& kp) {
                return kp.response < kFastScoreLowerBound;});
        keypoints.erase(iter, keypoints.end());

        // Do non-maximum suppression.
        DoNonMaximumSuppression(
            img.size(),
            config_->non_max_suppression_radius,
            &keypoints);
    }
    
    return keypoints;
}

void FeatureDetector::DetectKeyPointsInGrids(
    const cv::Ptr<cv::FeatureDetector>& cv_detector,
    const cv::Mat& image,
    const int num_grids_vertical,
    const int num_grids_horizontal,
    const int max_features_per_grid,
    std::vector<cv::KeyPoint>* keypoints_ptr) {
    std::vector<cv::KeyPoint>& keypoints = *CHECK_NOTNULL(keypoints_ptr);

    if (image.empty()) {
        return;
    }
    
    // Allocate memory.
    const size_t max_num =
        num_grids_vertical * num_grids_horizontal * max_features_per_grid;
    keypoints.reserve(max_num);

    const int image_height = image.rows;
    const int image_width = image.cols;
    const int grid_height = image_height / num_grids_vertical + 1;
    const int grid_width = image_width / num_grids_horizontal + 1;
    
    // Detect key points in each grid.
    std::vector<cv::KeyPoint> sub_keypoints;
    for (int n = 0; n < num_grids_vertical; ++n) {
        for (int m = 0; m < num_grids_horizontal; ++m) {
            // Starting and ending column numbers for each grid.
            const int grid_col_start = m * grid_width;
            const int grid_col_end = std::min(
                (m + 1) * grid_width,
                image_width);

            // Starting and ending row numbers for each grid.
            const int grid_row_start = n * grid_height;
            const int grid_row_end = std::min(
                (n + 1) * grid_height,
                image_height);
            
            cv::Range row_range(grid_row_start, grid_row_end);
            cv::Range col_range(grid_col_start, grid_col_end);
            cv::Mat sub_image = image(row_range, col_range);
            cv::Mat sub_mask = mask_(row_range, col_range);
            
            sub_keypoints.clear();
            cv_detector->detect(sub_image, sub_keypoints, sub_mask);

            // Compensate for the offset.
            for (auto& keypoint : sub_keypoints) {
                keypoint.pt.x += col_range.start;
                keypoint.pt.y += row_range.start;
            }

            // Append the key points.
            keypoints.insert(
                keypoints.end(),
                sub_keypoints.begin(),
                sub_keypoints.end());
        }
    }
}

void FeatureDetector::DoNonMaximumSuppression(
    const cv::Size& image_size,
    const int suppression_radius,
    std::vector<cv::KeyPoint>* keypoints_ptr) {
    CHECK_GT(suppression_radius, 0);
    std::vector<cv::KeyPoint>& keypoints = *CHECK_NOTNULL(keypoints_ptr);

    if (keypoints.empty()) {
        return;
    }

    // Prefer key points with higher response.
    std::sort(keypoints.begin(), keypoints.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;});

    std::vector<cv::KeyPoint> keypoints_selected;
    keypoints_selected.reserve(keypoints.size());

    cv::Mat mask = cv::Mat(image_size, CV_8UC1, cv::Scalar(0));
    for (const auto& keypoint : keypoints) {
        if (mask.at<unsigned char>(keypoint.pt) == 0) {
            keypoints_selected.push_back(keypoint);

            // Add mask here.
            cv::circle(mask, keypoint.pt, suppression_radius,
                cv::Scalar(255), -1);
        }
    }

    keypoints = keypoints_selected;
}

void FeatureDetector::SetFeatureDetectorType(
    const std::string& feature_detector_type) {
    feature_detector_type_string_ = feature_detector_type;
    if (feature_detector_type == FASTString) {
        feature_detector_type_ = FeatureDetectorType::FAST;
    } else if (feature_detector_type == GFTTString) {
        feature_detector_type_ = FeatureDetectorType::GFTT;
    } else {
        LOG(FATAL) << "Unknown feature detector type: " << feature_detector_type;
    }
}
}  // namespace visual_feature
