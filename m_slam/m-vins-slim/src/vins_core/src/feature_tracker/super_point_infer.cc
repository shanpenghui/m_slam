#include "feature_tracker/super_point_infer.h"

#include <numeric>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace vins_core {

SuperPoint::SuperPoint(SuperPointConfig super_point_config)
        : super_point_config_(std::move(super_point_config)),
          super_point_mode_(SuperPointMode::NoInitialized) {
}

static void dump_tensor_attr(rknn_tensor_attr* attr) {
    LOG(INFO) << "----------------------------------------------------------------";
    LOG(INFO) << "Name = " << attr->name 
              << ", Index = " << attr->index 
              << ", Number of dimension = " << attr->n_dims << ",";
    LOG(INFO) << "Dims = [" << attr->dims[0] << " " << attr->dims[1] << " " << attr->dims[2] << " " << attr->dims[3] << "]"
              << ", Number of elems = " << attr->n_elems 
              << ", Size = " << attr->size 
              << ", Format = " << get_format_string(attr->fmt) << ",";
    LOG(INFO) << "Type = " << get_type_string(attr->type)
              << ", Quantification Type = " << get_qnt_type_string(attr->qnt_type)
              << ", ZP = " << attr->zp
              << ", Scale = " << attr->scale;
}

float __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz) {
  unsigned char* data;
  int ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size) {
  FILE* fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

bool SuperPoint::build() {
    /* Create the neural network */
    LOG(INFO) << "Loading mode " << super_point_config_.model_name;
    int model_data_size = 0;
    unsigned char* model_data = load_model(super_point_config_.model_name.c_str(),
                                           &model_data_size);
    ret_ = rknn_init(&ctx_, model_data, model_data_size, 0, NULL);
    
    if (model_data) 
        free(model_data);

    if (ret_ < 0) {
        printf("rknn_init error ret=%d\n", ret_);
        return false;
    }

    rknn_sdk_version version;
    ret_ = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret_ < 0) {
        printf("rknn_init error ret=%d\n", ret_);
        return false;
    }
    LOG(INFO) << "SDK version: " << version.api_version << " driver version: " << version.drv_version;

    ret_ = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret_ < 0) {
        printf("rknn_init error ret=%d\n", ret_);
        return false;
    }
    LOG(INFO) << "Model input num: " << io_num_.n_input << " output num: " << io_num_.n_output;
    
    if (io_num_.n_output == 1) {
        super_point_mode_ = SuperPointMode::CombinedSuperPoint;
        LOG(WARNING) << "Use Combined SuperPoint.";
    } else if (io_num_.n_output == 2) {
        super_point_mode_ = SuperPointMode::SuperPoint;
        LOG(WARNING) << "Use Pure SuperPoint.";
    } else {
        LOG(FATAL) << "Wrong Superpoint mode file, please check!";
    }

    memset(input_attrs_, 0, sizeof(input_attrs_));
    for (int i = 0; i < io_num_.n_input; i++) {
        input_attrs_[i].index = i;
        ret_ = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret_ < 0) {
            printf("rknn_init error ret=%d\n", ret_);
            return false;
        }
        dump_tensor_attr(&(input_attrs_[i]));
    }

    memset(output_attrs_, 0, sizeof(output_attrs_));
    for (int i = 0; i < io_num_.n_output; i++) {
        output_attrs_[i].index = i;
        ret_ = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs_[i]));
    }

    if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
        LOG(INFO) << "Model is NCHW input format.";
        channel_ = input_attrs_[0].dims[1];
        height_ = input_attrs_[0].dims[2];
        width_ = input_attrs_[0].dims[3];
    } else {
        LOG(INFO) << "Model is NHWC input format.";
        height_ = input_attrs_[0].dims[1];
        width_ = input_attrs_[0].dims[2];
        channel_ = input_attrs_[0].dims[3];
    }
    LOG(INFO) << "Model input height = " << height_ << " width = " << width_ 
              << " channel = " << channel_;
    
    memset(inputs_, 0, sizeof(inputs_));
    inputs_[0].index = 0;
    inputs_[0].type = RKNN_TENSOR_UINT8;
    inputs_[0].size = width_ * height_ * channel_;
    inputs_[0].fmt = RKNN_TENSOR_NHWC;
    inputs_[0].pass_through = 0;

    memset(outputs_, 0, sizeof(outputs_));
    for (int i = 0; i < io_num_.n_output; i++) {
        outputs_[i].want_float = 1;
    }

    return true;
}

bool SuperPoint::infer(const cv::Mat& orig_img, 
                       Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &features,
                       std::vector<cv::KeyPoint>* keypoints_ptr) {
    cv::Mat img_resized;
    cv::Size dsize = cv::Size(width_, height_);
    cv::resize(orig_img, img_resized, dsize, 0, 0, cv::INTER_AREA);

    inputs_[0].buf = (void*)img_resized.data;
    rknn_inputs_set(ctx_, io_num_.n_input, inputs_);
    ret_ = rknn_run(ctx_, NULL);
    ret_ = rknn_outputs_get(ctx_, io_num_.n_output, outputs_, NULL);

    if (keypoints_ptr != nullptr) {
        std::vector<cv::KeyPoint>& keypoints = *keypoints_ptr;
        if (!process_output((float *)(outputs_[0].buf), 
                            keypoints, 
                            features, 
                            orig_img.rows, 
                            orig_img.cols)) {
            return false;
        }
    } else {
        if (!process_output((float *)(outputs_[0].buf), 
                            (float *)(outputs_[1].buf), 
                            features, 
                            orig_img.rows, 
                            orig_img.cols)) {
            return false;
        }
    }
    rknn_outputs_release(ctx_, io_num_.n_output, outputs_);     
    return true;
}

void SuperPoint::find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints,
                                       int h, int w, float threshold) {
    std::vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            std::vector<int> location = {int(i / w), i % w};
            keypoints.emplace_back(location);
            new_scores.emplace_back(scores[i]);
        }
    }
    scores.swap(new_scores);
}


void SuperPoint::remove_borders(std::vector<std::vector<int>> &keypoints, 
                                std::vector<float> &scores, 
                                int border,
                                int semi_height,
                                int semi_width) {
    std::vector<std::vector<int>> keypoints_selected;
    std::vector<float> scores_selected;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (semi_height - border));
        bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (semi_width - border));
        if (flag_h && flag_w) {
            keypoints_selected.emplace_back(std::vector<int>{keypoints[i][1], keypoints[i][0]});
            scores_selected.emplace_back(scores[i]);
        }
    }
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

std::vector<size_t> SuperPoint::sort_indexes(std::vector<float> &data) {
    std::vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SuperPoint::top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k) {
    if (k < keypoints.size() && k != -1) {
        std::vector<std::vector<int>> keypoints_top_k;
        std::vector<float> scores_top_k;
        std::vector<size_t> indexes = sort_indexes(scores);
        for (int i = 0; i < k; ++i) {
            keypoints_top_k.emplace_back(keypoints[indexes[i]]);
            scores_top_k.emplace_back(scores[indexes[i]]);
        }
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}

void normalize_keypoints(const std::vector<std::vector<int>> &keypoints,
                         std::vector<std::vector<float>> &keypoints_norm,
                         int h, int w, int s) {
    float temp1 = - s / 2 + 0.5;
    float temp2 = - s / 2 - 0.5;
    for (auto &keypoint : keypoints) {
        std::vector<float> kp = {keypoint[0] + temp1, keypoint[1] + temp1};
        kp[0] = kp[0] / (w * s + temp2);
        kp[1] = kp[1] / (h * s + temp2);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.emplace_back(kp);
    }
}

int clip(int val, int max) {
    if (val < 0) {
        return 0;
    }
    return std::min(val, max - 1);
}

void grid_sample(const float *input, std::vector<std::vector<float>> &grid,
                 std::vector<std::vector<float>> &output, int dim, int h, int w) {
    // descriptors 1, 256, image_height/8, image_width/8
    // keypoints 1, 1, number, 2
    // out 1, 256, 1, number
    for (auto &g : grid) {
        float ix = ((g[0] + 1) / 2) * (w - 1);
        float iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = clip(std::floor(ix), w);
        int iy_nw = clip(std::floor(iy), h);

        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);

        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);

        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        float nw = (ix_se - ix) * (iy_se - iy);
        float ne = (ix - ix_sw) * (iy_sw - iy);
        float sw = (ix_ne - ix) * (iy - iy_ne);
        float se = (ix - ix_nw) * (iy - iy_nw);

        std::vector<float> descriptor;
        int area = h * w;
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 dhw
            // x * height * depth + y * depth + z
            int vol = i * area;
            float nw_val = input[vol + iy_nw * w + ix_nw];
            float ne_val = input[vol + iy_ne * w + ix_ne];
            float sw_val = input[vol + iy_sw * w + ix_sw];
            float se_val = input[vol + iy_se * w + ix_se];
            descriptor.emplace_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.emplace_back(descriptor);
    }
}

template<typename Iter_T>
float vector_normalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
}

void normalize_descriptors(std::vector<std::vector<float>> &dest_descriptors) {
    for (auto& descriptor : dest_descriptors) {    
        float norm_inv = 1.0 / vector_normalize(descriptor.begin(), descriptor.end());
        std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                       std::bind1st(std::multiplies<float>(), norm_inv));
    }    
}

void SuperPoint::sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                                    std::vector<std::vector<float>> &dest_descriptors, int dim, int h, int w, int s) {
    std::vector<std::vector<float>> keypoints_norm;
    normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    grid_sample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    normalize_descriptors(dest_descriptors);
}

bool SuperPoint::process_output(float* output_desc, 
                                std::vector<cv::KeyPoint>& keypoints,
                                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &features,
                                const double raw_img_height, 
                                const double raw_img_width) {
    keypoints_.clear();
    descriptors_.clear();
    int semi_feature_map_h = height_;
    int semi_feature_map_w = width_;
    // superpoint resolution / raw resolution
    const double height_scalar = static_cast<double>(height_) / raw_img_height; 
    const double width_scalar = static_cast<double>(width_) / raw_img_width;

    // Note(longyb7): keypoints format: (y,x), but (x,y) performs better.
    for (int i = 0; i < keypoints.size(); ++i) {
        int projected_x = static_cast<int>(std::round(keypoints[i].pt.x * width_scalar));
        int projected_y = static_cast<int>(std::round(keypoints[i].pt.y * height_scalar));
        keypoints_.push_back(std::vector<int>{projected_x, projected_y});
    }

    int desc_feature_dim = 256;
    int desc_feature_map_h = height_ / 8;
    int desc_feature_map_w = width_ / 8;
    sample_descriptors(keypoints_, output_desc, descriptors_, desc_feature_dim, desc_feature_map_h, desc_feature_map_w);

    RemoveNaNKeypoints(&keypoints, &descriptors_);
            
    features.resize(256, keypoints.size());

    for (int m = 0; m < 256; ++m) {
        for (int n = 0; n < descriptors_.size(); ++n) {
            features(m, n) = descriptors_[n][m];
        }
    }
    return true;
}

bool SuperPoint::process_output(float* output_score,
                                float* output_desc, 
                                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &features,
                                const double raw_img_height, 
                                const double raw_img_width) {
    keypoints_.clear();
    descriptors_.clear();
    int semi_feature_map_h = height_;
    int semi_feature_map_w = width_;
    std::vector<float> scores_vec(output_score, output_score + semi_feature_map_h * semi_feature_map_w);
    
    find_high_score_index(scores_vec, keypoints_, semi_feature_map_h, semi_feature_map_w, super_point_config_.keypoint_threshold);
    remove_borders(keypoints_, scores_vec, super_point_config_.remove_borders, semi_feature_map_h, semi_feature_map_w);
    top_k_keypoints(keypoints_, scores_vec, super_point_config_.max_keypoints);

    features.resize(259, scores_vec.size());
    int desc_feature_dim = 256;
    int desc_feature_map_h = height_ / 8;
    int desc_feature_map_w = width_ / 8;
    sample_descriptors(keypoints_, output_desc, descriptors_, desc_feature_dim, desc_feature_map_h, desc_feature_map_w);
    
    for (int i = 0; i < scores_vec.size(); i++){
        features(0, i) = scores_vec[i];
    }
    for (int i = 1; i < 3; ++i) {
        for (int j = 0; j < keypoints_.size(); ++j) {
            features(i, j) = keypoints_[j][i-1];
        }
    }
    for (int m = 3; m < 259; ++m) {
        for (int n = 0; n < descriptors_.size(); ++n) {
            features(m, n) = descriptors_[n][m-3];
        }
    }
    return true;
}

void SuperPoint::visualization(const std::string &image_name, const cv::Mat &image) {
    cv::Mat image_display;
    if(image.channels() == 1)
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    else
        image_display = image.clone();
    for (auto &keypoint : keypoints_) {
        cv::circle(image_display, cv::Point(int(keypoint[0]), int(keypoint[1])), 1, cv::Scalar(255, 0, 0), -1, 16);
    }
    cv::imwrite(image_name, image_display);
}

bool SuperPoint::debuild() {
    rknn_outputs_release(ctx_, io_num_.n_output, outputs_);    
    ret_ = rknn_destroy(ctx_);
    return true;
}

void SuperPoint::RemoveNaNKeypoints(std::vector<cv::KeyPoint>* keypoints_ptr, 
                                    std::vector<std::vector<float>>* descriptors_ptr) {
    std::vector<cv::KeyPoint>& keypoints = *CHECK_NOTNULL(keypoints_ptr);
    std::vector<std::vector<float>>& descriptors = *CHECK_NOTNULL(descriptors_ptr);
    CHECK(keypoints.size() == descriptors.size());

    std::vector<KeypointWithDescriptor> keypoints_descriptors;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        KeypointWithDescriptor keypoint_descriptor(keypoints[i], descriptors[i]);
        keypoints_descriptors.push_back(keypoint_descriptor);
    }

    keypoints_descriptors.erase(
        std::remove_if(keypoints_descriptors.begin(),
                       keypoints_descriptors.end(),
            [](KeypointWithDescriptor& key_des) -> bool {
                for (const auto& each_des: key_des.descriptor) {
                    if (std::isnan(each_des)) {
                        return true;
                    }
                }
                return false;
            }), 
        keypoints_descriptors.end());

    keypoints.clear();
    descriptors.clear();

    for (const auto& keypoint_with_descriptor : keypoints_descriptors) {
        keypoints.push_back(keypoint_with_descriptor.keypoint);
        descriptors.push_back(keypoint_with_descriptor.descriptor);
    }
}

}