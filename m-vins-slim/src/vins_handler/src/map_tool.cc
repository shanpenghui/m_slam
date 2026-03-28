#include "vins_handler/map_tool.h"

#include "cost_function/line_rectional_cost.h"

namespace vins_handler {

double GetAngle(cv::Vec4i line) {
    double dx = line[2] - line[0];
    double dy = line[3] - line[1];
    return std::atan2(dy, dx);
}

double distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0u; i < a.size(); ++i) {
        sum += std::pow(a[i] - b[i], 2.0);
    }
    return std::sqrt(sum);
}

std::vector<int> assign_clusters(
    const std::vector<double>& data,
    const std::vector<std::vector<double>>& centers) {
    std::vector<int> clusters(data.size());
    for (size_t i = 0u; i < data.size(); ++i) {
        double min_distance = INFINITY;
        int min_index = -1;
        for (size_t j = 0u; j < centers.size(); ++j) {
            double d = distance({data[i]}, centers[j]);
            if (d < min_distance) {
                min_distance = d;
                min_index = j;
            }
        }
        clusters[i] = min_index;
    }
    return clusters;
}

std::vector<std::vector<double>> update_centers(
    const std::vector<double>& data,
    const std::vector<int>& clusters,
    int k) {
    std::vector<std::vector<double>> centers(k, std::vector<double>(1, 0.0));
    std::vector<int> counts(k, 0);
    for (size_t i = 0u; i < data.size(); ++i) {
        int cluster_index = clusters[i];
        counts[cluster_index]++;
        centers[cluster_index][0] += data[i];
    }
    for (int i = 0; i < k; ++i) {
        if (counts[i] > 0) {
            centers[i][0] /= counts[i];
        }
    }
    return centers;
}

std::vector<std::vector<int>> k_means(
    const std::vector<double>& data,
    int k,
    int max_iterations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);
    std::vector<std::vector<double>> centers(k, std::vector<double>(1));
    for (int i = 0; i < k; ++i) {
        centers[i][0] = data[dis(gen)];
    }

    std::vector<int> clusters(data.size());
    for (int i = 0; i < max_iterations; ++i) {
        clusters = assign_clusters(data, centers);
        std::vector<std::vector<double>> new_centers = update_centers(data, clusters, k);
        if (centers == new_centers) {
            break;
        }
        centers = new_centers;
    }

    std::vector<std::vector<int>> result(k);
    for (size_t i = 0u; i < data.size(); ++i) {
        result[clusters[i]].push_back(i);
    }
    return result;
}

void ComputeDegree(const cv::Mat& img,
                   std::vector<cv::Vec4i>* lines_ptr,
                   std::vector<int>* inlier_indices_ptr,
                   double* angle_ptr) {
    std::vector<cv::Vec4i>& lines = *CHECK_NOTNULL(lines_ptr);
    std::vector<int>& inlier_indices = *CHECK_NOTNULL(inlier_indices_ptr);
    double& angle = *CHECK_NOTNULL(angle_ptr);

    cv::Mat raw;
    img.copyTo(raw);
    for (int idx = 0; idx < raw.rows * raw.cols; ++idx) {
        uchar& value = raw.data[idx];
        if (value == static_cast<uchar>(127)) {
            value = static_cast<uchar>(255);
        }
    }
    raw = ~raw;
    cv::HoughLinesP(raw, lines, 1, CV_PI / 180, 10, 15, 4);

    std::vector<double> lines_angle;
    lines_angle.resize(lines.size());
    for (size_t i = 0u; i < lines.size(); ++i) {
        lines_angle[i] = std::abs(GetAngle(lines[i]) * common::kRadToDeg);
        if (lines_angle[i] > 90.) {
            lines_angle[i] = 180. - lines_angle[i];
        }
    }

    std::function<void(const std::vector<double>&, std::vector<int>*)> check_inliers_helper =
        [&](const std::vector<double>& data,
            std::vector<int>* inliers_ptr) {
        std::vector<int>& inliers = *CHECK_NOTNULL(inliers_ptr);
        // Brute force search.
        std::vector<int> left_max_indices, right_max_indices;
        for (double deg = 0.0; deg < 45.0; ++deg) {
            std::vector<int> tmp_indices;
            for (size_t i = 0u; i < data.size(); ++i) {
                if (std::abs(deg - data[i]) <= 5.0) {
                    tmp_indices.push_back(i);
                }
            }
            if (tmp_indices.size() > left_max_indices.size()) {
                left_max_indices = tmp_indices;
            }
        }
        for (double deg = 45.0; deg <= 90.0; ++deg) {
            std::vector<int> tmp_indices;
            for (size_t i = 0u; i < data.size(); ++i) {
                if (std::abs(deg - data[i]) <= 5.0) {
                    tmp_indices.push_back(i);
                }
            }
            if (tmp_indices.size() > right_max_indices.size()) {
                right_max_indices = tmp_indices;
            }           
        }

        for (size_t i = 0u; i < left_max_indices.size(); ++i) {
            inliers[left_max_indices[i]] = 1;
        }
        for (size_t i = 0u; i < right_max_indices.size(); ++i) {
            inliers[right_max_indices[i]] = 2;
        }
    };

    inlier_indices.resize(lines.size(), 0);
    check_inliers_helper(lines_angle, &inlier_indices);

    std::vector<cv::Vec4i> inlier_lines;
    for (size_t i = 0u; i < lines.size(); ++i) {
        if (inlier_indices[i]) {
            inlier_lines.push_back(lines[i]);
        }
    }

    ceres::Problem problem;

    ceres::CostFunction* line_rectional_cost =
        vins_core::CreateLineRectionalCost(inlier_lines);
    problem.AddResidualBlock(line_rectional_cost,
                             nullptr,
                             &angle);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

cv::Mat RotateImage(const double degree, cv::Mat* img_ptr) {
    cv::Mat& img = *CHECK_NOTNULL(img_ptr);

    cv::Point2f center;
    center.x = static_cast<float>(img.cols / 2.0);
    center.y = static_cast<float>(img.rows / 2.0);

    const double a = std::sin(common::kDegToRad * degree);
    const double b = std::cos(common::kDegToRad * degree);
    const int width_rotate = int(img.rows * std::abs(a) + img.cols * std::abs(b));
    const int height_rotate = int(img.cols * std::abs(a) + img.rows * std::abs(b));

    cv::Mat M1 = cv::getRotationMatrix2D(center, degree, 1);

    cv::Point2f from_pts[3];
    cv::Point2f to_pts[3];

    from_pts[0] = cv::Point2i(0, 0);
    from_pts[1] = cv::Point2i(0, img.rows);
    from_pts[2] = cv::Point2i(img.cols, 0);

    to_pts[0] = cv::Point2i((width_rotate - img.cols) / 2 , (height_rotate - img.rows) / 2);
    to_pts[1] = cv::Point2i((width_rotate - img.cols) / 2 , img.rows + (height_rotate - img.rows) / 2);
    to_pts[2] = cv::Point2i(img.cols + (width_rotate - img.cols) / 2, (height_rotate - img.rows) / 2);

    cv::Mat M2 = cv::getAffineTransform(from_pts, to_pts);

    M1.at<double>(0, 2) = M1.at<double>(0, 2) + M2.at<double>(0, 2);
    M1.at<double>(1, 2) = M1.at<double>(1, 2) + M2.at<double>(1, 2);

    cv::warpAffine(img, img, M1, cv::Size(width_rotate, height_rotate),
                  cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(128,0,0));

    return M1;
}

void MapPoseRectangulate(const cv::Mat& raw_map,
                         common::KeyFrames* key_frames_ptr,
                         std::vector<cv::Vec4i>* lines_ptr,
                         std::vector<int>* inlier_indices_ptr) {
    common::KeyFrames& key_frames = *CHECK_NOTNULL(key_frames_ptr);
    std::vector<cv::Vec4i>& lines = *CHECK_NOTNULL(lines_ptr);
    std::vector<int>& inlier_indices = *CHECK_NOTNULL(inlier_indices_ptr);

    double angle;
    
    ComputeDegree(raw_map, &lines, &inlier_indices, &angle);
    VLOG(0) << "Detected angle: " << angle * common::kRadToDeg << "(deg) in map rectangulation.";

    Eigen::Matrix3d R = common::EulerToRot(
                Eigen::Vector3d(0., 0., angle));

    // Update keyframe pose.
    for (size_t i = 0u; i < key_frames.size(); ++i) {
        const Eigen::Matrix3d R_trans = R * key_frames[i].state.T_OtoG.getRotationMatrix();
        const Eigen::Vector3d p_trans = R * key_frames[i].state.T_OtoG.getPosition();
        key_frames[i].state.T_OtoG.update(Eigen::Quaterniond(R_trans), p_trans);
    }
}

cv::Mat CreateShowMapMat(const cv::Mat& raw_map, 
                         const bool do_remove_obstacles,
                         const int contour_length) {   
    CHECK_EQ(raw_map.type(), CV_32FC1);

    cv::Mat map_show = cv::Mat(raw_map.rows, raw_map.cols, CV_8UC1);
    
    for (int i = 0; i < raw_map.rows; ++i) {
        for (int j = 0; j < raw_map.cols; ++j) {
            const float occ = raw_map.at<float>(i, j);
            uchar& pixel_show = map_show.at<uchar>(i, j);
            if (occ < 0.) {
                pixel_show = 127;
            } else {
                if(occ > 0.65) {
                    pixel_show = 0;
                }
                else if (occ < 0.43) {
                    pixel_show = 255;
                }
                else {
                    pixel_show = 127;
                }
            }
        }
    }

    cv::Mat kernel = cv::Mat::ones(3,3,CV_8U);
    cv::Mat temp = cv::Mat(raw_map.rows, raw_map.cols, CV_8UC1);
    cv::threshold(map_show, temp, 150, 255, cv::THRESH_BINARY);
    cv::filter2D(temp, temp, CV_8UC1, kernel);
    cv::Mat mask = ((temp == 0) & (map_show == 0));
    map_show.setTo(127,mask); 
    
    for (int i = 0; i < map_show.rows; ++i) {
        for (int j = 0; j < map_show.cols; ++j) {
            uchar& pixel_show = map_show.at<uchar>(i, j);
            if (pixel_show == 0  || pixel_show == 255) {
                bool top = false;
                bool bottom = false;
                bool left = false;
                bool right = false;
                for(int step = 1; step <= 5; step++) {
                    if (i + step < raw_map.rows && !bottom) {
                        if (map_show.at<uchar>(i + step , j) != 255) {
                           bottom = true;
                        }
                    }
                    if (i - step >= 0 && !top) {
                        if (map_show.at<uchar>(i - step , j) != 255) {
                            top = true;
                        }
                    }
                    if (j + step < raw_map.cols && !right) {
                        if (map_show.at<uchar>(i , j + step) != 255) {
                            right = true;
                        }
                    }
                    if (j - step >= 0 && !left) {
                        if (map_show.at<uchar>(i , j - step) != 255) {
                            left = true;
                        }
                    }
                }
                if (left && right && top && bottom) {
                    pixel_show = 127;
                }
            }
        }
    }  
    
    if (do_remove_obstacles) {
        RemoveSmallObstacles(&map_show, contour_length);
    }

    return map_show;
}

void RemoveSmallObstacles(cv::Mat* input_map_ptr, const int contour_length) {
    cv::Mat& input_map = *CHECK_NOTNULL(input_map_ptr);
    CHECK_EQ(input_map.type(), CV_8UC1);

    cv::Mat binary_map;
    cv::threshold(input_map, binary_map, 254, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierachy;
    cv::findContours(binary_map, contours, hierachy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    // if width or height of the obstacle boundingbox lower than contour_length(pixel_value),
    // this obstacle will be removed.
    if (contours.size() != 0) {
        for (size_t contour_index = 0; contour_index < contours.size(); contour_index++) {
            cv::Rect boundRect = cv::boundingRect(contours[contour_index]);
            if (boundRect.height <= contour_length && boundRect.width <= contour_length) {
                cv::drawContours(input_map, contours, static_cast<int>(contour_index), 
                    cv::Scalar(255,255,255), cv::FILLED, 8, hierachy, 2);
			}
        }
    }
}

}
