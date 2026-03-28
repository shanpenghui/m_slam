#ifndef MVINS_FEATURE_TRACKER_SUPER_POINT_H_
#define MVINS_FEATURE_TRACKER_SUPER_POINT_H_

#include <string>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "rknn/rknn_api.h"

namespace vins_core {

struct SuperPointConfig {
    int max_keypoints;
    int remove_borders;
    double raw_input_img_height;
    double raw_input_img_width;
    float keypoint_threshold;
    std::string model_name;
};

struct KeypointWithDescriptor {
    cv::KeyPoint keypoint;
    std::vector<float> descriptor;

    KeypointWithDescriptor() = delete;

    KeypointWithDescriptor(const cv::KeyPoint& _keypoint,
                            const std::vector<float> _descriptor) {
        keypoint = _keypoint;
        descriptor = _descriptor;
    }
};

enum class SuperPointMode {
    SuperPoint,
    CombinedSuperPoint,
    NoInitialized
};

class SuperPoint {
public:
    explicit SuperPoint(SuperPointConfig super_point_config);

    bool build();

    bool infer(const cv::Mat &image, 
               Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &features,
               std::vector<cv::KeyPoint>* keypoints_ptr = nullptr);

    void visualization(const std::string &image_name, const cv::Mat &image);

    bool debuild();

    int Width() {
        return width_;
    }

    int Height() {
        return height_;
    }

    SuperPointMode GetSuperPointMode() {
        return super_point_mode_;
    }

private:
    SuperPointConfig super_point_config_;

    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<float>> descriptors_;

    int ret_;
    rknn_context ctx_;
    rknn_input_output_num io_num_;
    rknn_tensor_attr input_attrs_[1];
    rknn_tensor_attr output_attrs_[2];

    int channel_ = 1;
    int width_ = 0;
    int height_ = 0;

    rknn_input inputs_[1];
    rknn_output outputs_[2];

    SuperPointMode super_point_mode_;

    bool process_output(float* output_desc, 
                        std::vector<cv::KeyPoint>& keypoints, 
                        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &features,
                        const double raw_img_height,
                        const double raw_img_width);

    bool process_output(float* output_score, 
                        float* output_desc, 
                        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &features,
                        const double raw_img_height,
                        const double raw_img_width);

    void remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border, int height,
                        int width);

    std::vector<size_t> sort_indexes(std::vector<float> &data);

    void top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k);

    void find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints, int h, int w,
                               float threshold);

    void sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                            std::vector<std::vector<float>> &dest_descriptors, int dim, int h, int w, int s = 8);

    void RemoveNaNKeypoints(std::vector<cv::KeyPoint>* keypoints_ptr, 
            std::vector<std::vector<float>>* descriptors_ptr);
};
}
#endif
