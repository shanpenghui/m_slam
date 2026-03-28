#include "feature_tracker/feature_tracker_base.h"

#include "cfg_common/slam_config.h"
#include "file_common/file_system_tools.h"
#include "feature_tracker/delaunay_mesher.h"

namespace vins_core {

FeatureTrackerBase::FeatureTrackerBase(
        const aslam::NCamera::Ptr& cameras,
        const common::SlamConfigPtr& config)
    : cameras_(cameras),
      config_(config) {
#ifdef USE_CNN_FEATURE
    SuperPointConfig superpoint_config = CreateSuperPointConfig(config);
    superpoint_infer_ptr_.reset(new SuperPoint(superpoint_config));
    LOG(WARNING) << "Start building SuperPoint mode...";
    if (!superpoint_infer_ptr_->build()) {
        LOG(FATAL) << "Error in SuperPoint building engine, model file path: "
                   << superpoint_config.model_name;
    } else {
        LOG(WARNING) << "SuperPoint building success.";
    }
#endif
    VLOG(0) << "Load mask from path: " << config_->mask_path;
    cv::Mat mask;
    mask = cv::imread(config_->mask_path, cv::IMREAD_GRAYSCALE);
    cv::threshold(mask, mask, 100, 255, cv::THRESH_BINARY);
    feature_detector_.reset(new FeatureDetector(config_, mask));
    feature_extractor_.reset(new FeatureExtractor(config_));
}

#ifdef USE_CNN_FEATURE
SuperPointConfig FeatureTrackerBase::CreateSuperPointConfig(
    const common::SlamConfigPtr& config) {
    SuperPointConfig superpoint_config;
    superpoint_config.model_name = common::ConcatenateFilePathFrom(
        config->assets_path, config->superpoint_model_file);
    superpoint_config.max_keypoints = config->num_feature_to_detect;
    superpoint_config.remove_borders = 4;
    superpoint_config.keypoint_threshold = 0.002;
    return superpoint_config;
}

void FeatureTrackerBase::InferFeature(
    const uint64 timestamp_ns,
    const common::CvMatConstPtr& img,
    common::VisualFrameData* visual_frame_data_ptr,
    int* track_id_provider_ptr) {
    common::VisualFrameData& visual_frame_data =
            *CHECK_NOTNULL(visual_frame_data_ptr);
    int& track_id_provider =
            *CHECK_NOTNULL(track_id_provider_ptr);  

    cv::Mat image_gray;
    if (img->type() == CV_8UC3) {
        cv::cvtColor(*img, image_gray, cv::COLOR_BGR2GRAY);
    } else if (img->type() == CV_8UC1) {
        image_gray = *img;
    } else {
        LOG(FATAL) << "Unsupport image type.";
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> feature_points;
    if (superpoint_infer_ptr_->GetSuperPointMode() == SuperPointMode::CombinedSuperPoint) {
        std::vector<cv::KeyPoint> keypoints;
        keypoints = feature_detector_->Detect(image_gray);
        if (!superpoint_infer_ptr_->infer(image_gray, feature_points, &keypoints)) {
            LOG(WARNING) << "SuperPoint inference failure in time: " << timestamp_ns;
            return;
        }
        visual_frame_data.SetKeyPoints(keypoints,
                                       config_->keypoint_uncertainty_px);
    } else if (superpoint_infer_ptr_->GetSuperPointMode() == SuperPointMode::SuperPoint) {
        if (!superpoint_infer_ptr_->infer(image_gray, feature_points)) {
            LOG(WARNING) << "SuperPoint inference failure in time: " << timestamp_ns;
            return;
        }
        visual_frame_data.SetKeyPoints(feature_points,
                                       config_->keypoint_uncertainty_px);
    } else {
        LOG(ERROR) << "ERROR: SuperPoint mode has not initialized";
        return;
    }

    // Fill visual frame data.
    visual_frame_data.timestamp_ns = timestamp_ns;
    // NOTE. Do not initialize track_lengths status in there,
    // because of the keypoint size maybe changed in feature track step.
    visual_frame_data.SetDescriptors(feature_points);
    visual_frame_data.GenerateTrackIds(&track_id_provider);
    visual_frame_data.image_ptr = img;

    if (superpoint_infer_ptr_->GetSuperPointMode() == SuperPointMode::SuperPoint) {    
        RestoreKeypoints(&visual_frame_data, 
                         superpoint_infer_ptr_->Width(),
                         superpoint_infer_ptr_->Height());
    }
}

void FeatureTrackerBase::RestoreKeypoints(
    common::VisualFrameData* frame_data_ptr,
    const int infer_img_width,
    const int infer_img_height) {
    common::VisualFrameData& frame_data =
            *CHECK_NOTNULL(frame_data_ptr);
    const int raw_width = frame_data.image_ptr->cols;
    const int raw_height = frame_data.image_ptr->rows;
    const double down_rate_w = static_cast<double>(raw_width) /
        static_cast<double>(infer_img_width);
    const double down_rate_h = static_cast<double>(raw_height) /
        static_cast<double>(infer_img_height);
    
    for (int i = 0; i < frame_data.key_points.cols(); ++i) {
        frame_data.key_points(X, i) = frame_data.key_points(X, i) * down_rate_w;
        frame_data.key_points(Y, i) = frame_data.key_points(Y, i) * down_rate_h;
    }
}
#endif

void FeatureTrackerBase::DetectAndExtractFeatureImpl(
        const common::CvMatConstPtr& img,
        std::vector<cv::KeyPoint>* keypoints_ptr,
        cv::Mat* descriptors_ptr) {
    std::vector<cv::KeyPoint>& keypoints = *CHECK_NOTNULL(keypoints_ptr);
    cv::Mat& descriptors = *CHECK_NOTNULL(descriptors_ptr);

    cv::Mat image_gray;
    if (img->type() == CV_8UC3) {
        cv::cvtColor(*img, image_gray, cv::COLOR_BGR2GRAY);
    } else if (img->type() == CV_8UC1) {
        image_gray = *img;
    } else {
        LOG(FATAL) << "Unsupport image type.";
    }

    // Detect keypoints.
    keypoints = feature_detector_->Detect(image_gray);
    // Extract descriptors.
    feature_extractor_->Extract(image_gray, &keypoints, &descriptors);
    CHECK_EQ(static_cast<int>(keypoints.size()), descriptors.rows);
}

void FeatureTrackerBase::DetectAndExtractFeature(
        const uint64 timestamp_ns,
        const common::CvMatConstPtr& img,
        common::VisualFrameData* visual_frame_data_ptr,
        int* track_id_provider_ptr) {
    common::VisualFrameData& visual_frame_data =
            *CHECK_NOTNULL(visual_frame_data_ptr);
    int& track_id_provider =
            *CHECK_NOTNULL(track_id_provider_ptr);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    DetectAndExtractFeatureImpl(img, &keypoints, &descriptors);

    // Fill visual frame data.
    visual_frame_data.timestamp_ns = timestamp_ns;
    visual_frame_data.SetKeyPoints(keypoints,
                                   config_->keypoint_uncertainty_px);
    // NOTE. Do not initialize track_lengths status in there,
    // because of the keypoint size maybe changed in feature track step.
    visual_frame_data.SetDescriptors(descriptors);
    visual_frame_data.GenerateTrackIds(&track_id_provider);
    visual_frame_data.image_ptr = img;
}

void FeatureTrackerBase::PredictKeypointsByRotation(
    const aslam::Camera& camera,
    const Eigen::Matrix2Xd keypoints_k,
    const Eigen::Quaterniond& q_kp1_k,
    Eigen::Matrix2Xd* predicted_keypoints_kp1,
    std::vector<unsigned char>* prediction_success) {
  CHECK_NOTNULL(predicted_keypoints_kp1);
  CHECK_NOTNULL(prediction_success)->clear();
  if (keypoints_k.cols() == 0u) {
    return;
  }

  // Early exit for identity rotation.
  if (std::abs(q_kp1_k.w() - 1.0) < 1e-8) {
    *predicted_keypoints_kp1 = keypoints_k;
    prediction_success->resize(predicted_keypoints_kp1->size(), true);
  }

  // Backproject the keypoints to bearing vectors.
  Eigen::Matrix3Xd bearing_vectors_k;
  camera.backProject3Vectorized(keypoints_k, &bearing_vectors_k,
                                prediction_success);
  if (camera.getType() == aslam::Camera::Type::kUnifiedProjection) {
    for (int i = 0; i < bearing_vectors_k.cols(); ++i) {
        const double z = bearing_vectors_k(2, i);
        bearing_vectors_k(0, i) = bearing_vectors_k(0, i) / z;
        bearing_vectors_k(1, i) = bearing_vectors_k(1, i) / z;
        bearing_vectors_k(2, i) = 1.0;
    }
  }
  CHECK_EQ(static_cast<int>(prediction_success->size()), bearing_vectors_k.cols());
  CHECK_EQ(keypoints_k.cols(), bearing_vectors_k.cols());

  // Rotate the bearing vectors into the keypoints_kp1 coordinates.
  const Eigen::Matrix3Xd bearing_vectors_kp1 =
          q_kp1_k.toRotationMatrix() * bearing_vectors_k;

  // Project the bearing vectors to the keypoints_kp1.
  std::vector<aslam::ProjectionResult> projection_results;
  camera.project3Vectorized(bearing_vectors_kp1,
                           predicted_keypoints_kp1,
                           &projection_results);
  CHECK_EQ(predicted_keypoints_kp1->cols(), bearing_vectors_k.cols());
  CHECK_EQ(static_cast<int>(projection_results.size()), bearing_vectors_k.cols());

  // Set the success based on the backprojection and projection results
  // and output the initial unrotated keypoint for failed predictions.
  CHECK_EQ(keypoints_k.cols(), predicted_keypoints_kp1->cols());

  for (size_t idx = 0u; idx < projection_results.size(); ++idx) {
    (*prediction_success)[idx] = (*prediction_success)[idx] &&
                                 projection_results[idx].isKeypointVisible();

    // Set the initial keypoint location for failed predictions.
    if (!(*prediction_success)[idx]) {
      predicted_keypoints_kp1->col(idx) = keypoints_k.col(idx);
    }
  }
}

void FeatureTrackerBase::RansacFiltering(
    const aslam::Camera& camera,
    const Eigen::Matrix3d& R_kp1_k,
    const Eigen::Matrix6Xd& keypoints_k,
    const Eigen::Matrix6Xd& keypoints_kp1,
    common::FrameToFrameMatchesWithScore* matches_kp1_k_ptr) {
    auto& matches_kp1_k = *CHECK_NOTNULL(matches_kp1_k_ptr);

    // Convert bearings to Eigen type.
    Eigen::Matrix2Xd keypoints_k_eigen, keypoints_kp1_eigen;
    keypoints_k_eigen.resize(Eigen::NoChange, matches_kp1_k.size());
    keypoints_kp1_eigen.resize(Eigen::NoChange, matches_kp1_k.size());

    for (size_t i = 0u; i < matches_kp1_k.size(); ++i) {
        const int idx_kp1 = matches_kp1_k[i].GetKeypointIndexAppleFrame();
        const int idx_k = matches_kp1_k[i].GetKeypointIndexBananaFrame();

        keypoints_k_eigen.col(i) = keypoints_k.col(idx_k).head<2>();
        keypoints_kp1_eigen.col(i) = keypoints_kp1.col(idx_kp1).head<2>();
    }

    std::vector<unsigned char> is_valid = std::vector<unsigned char>(
        matches_kp1_k.size(), 1);

    constexpr bool kEnoughRotation = true;
    const double threshold =
            std::cos((90. - config_->ransac_angle_tolerance) * common::kDegToRad);
    const int max_iteration =
            std::ceil(std::log(1. - config_->ransac_success_probability) /
                    std::log(1. - std::pow(1. - config_->ransac_outlier_percentage,
                                        config_->ransac_sample_point_size)));
    bool success = TwoPointRansac(
        camera,
        R_kp1_k,
        keypoints_kp1_eigen,
        keypoints_k_eigen,
        kEnoughRotation,
        max_iteration,
        threshold,
        &is_valid);

    // Remove invalid matches.
    common::FrameToFrameMatchesWithScore valid_matches;
    valid_matches.reserve(matches_kp1_k.size());

    if (success) {
        CHECK(is_valid.size() == matches_kp1_k.size());
        for (size_t i = 0u; i < is_valid.size(); ++i) {
            if (is_valid[i] == 1) {
                valid_matches.emplace_back(matches_kp1_k[i]);
            }
        }
    }

    matches_kp1_k = valid_matches;
}

bool FeatureTrackerBase::TwoPointRansac(
    const aslam::Camera& camera,
    const Eigen::Matrix3d& R_kp1_k,
    const Eigen::Matrix2Xd& keypoints_kp1_2d,
    const Eigen::Matrix2Xd& keypoints_k_2d,
    const bool enough_rotation,
    const int ransac_max_iterations,
    const double ransac_threshold,
    std::vector<unsigned char>* is_valid_ptr) {
    std::vector<unsigned char>& is_valid = *CHECK_NOTNULL(is_valid_ptr);

    const int num_measurements = static_cast<int>(keypoints_k_2d.cols());

    if (num_measurements < 3 ||
        std::count(is_valid.begin(), is_valid.end(), 1) < 3) {
        return false;
    }

    size_t max_inlier_num = 0u;
    std::vector<int> best_inlier_indices;

    int iteration_num = 0;
    constexpr size_t ransac_min_set = 2u;
    int idx_1, idx_2;

    constexpr unsigned seed = 12345u;
    std::mt19937 mt(seed);
    std::uniform_int_distribution<> dis(0, num_measurements - 1);

    Eigen::Vector3d bearing_kp1;
    Eigen::Vector3d bearing_k;
    Eigen::Matrix3Xd epipolar_plane_normals;
    epipolar_plane_normals.resize(Eigen::NoChange, num_measurements);

    // Buffer for saving computation.
    for (int i = 0; i < num_measurements; ++i) {
        // NOTE(chien): only suite mono case.
        camera.backProject3(keypoints_kp1_2d.col(i), &bearing_kp1);
        bearing_kp1 << bearing_kp1(0) /bearing_kp1(2),
                       bearing_kp1(1) /bearing_kp1(2),
                       1.0;
        camera.backProject3(keypoints_k_2d.col(i), &bearing_k);
        bearing_kp1 << bearing_k(0) /bearing_k(2),
                       bearing_k(1) /bearing_k(2),
                       1.0;
        if (enough_rotation) {
            // Align curr to the orientation of "k"
            bearing_k = R_kp1_k * bearing_k;
        }

        epipolar_plane_normals.col(i) = bearing_kp1.normalized().cross(
                bearing_k.normalized());
    }

    while (iteration_num < ransac_max_iterations) {
        ++iteration_num;

        // Get min set of points.
        std::vector<int> index;
        while (true) {
            int idx = dis(mt);
            if (is_valid[idx] != 0u) {
                index.push_back(idx);
                if (index.size() == ransac_min_set) {
                    break;
                }
            }
        }

        idx_1 = index[0];
        idx_2 = index[1];

        // Calculate model: direction of camera translation
        // (sign checking not necessary).
        Eigen::Vector3d translation =
            epipolar_plane_normals.col(idx_1).cross(
                epipolar_plane_normals.col(idx_2));
        translation.normalize();

        // Bad two point selection, select again.
        if (std::abs(translation.norm() - 1.) > 1e-10) {
            continue;
        }

        // Check inliers.
        std::vector<int> inlier_indices;
        for (int i = 0; i < num_measurements; ++i) {
            if (is_valid[i] == 0u) {
                continue;
            }

            const double error = std::abs(
                epipolar_plane_normals.col(i).dot(translation));

            if (error < ransac_threshold) {
                inlier_indices.push_back(i);
            }
        }

        // Update best result.
        if (inlier_indices.size() > max_inlier_num) {
            max_inlier_num = inlier_indices.size();
            best_inlier_indices = inlier_indices;
        }
    }

    // Make sure the A matrix below has at least
    // 3 columns and can be SVD decomposed.
    if (best_inlier_indices.size() < 3u) {
        return false;
    }

    // Final translation estimation.
    Eigen::Matrix3Xd A;
    A.resize(Eigen::NoChange, best_inlier_indices.size());
    for (size_t i = 0u; i < best_inlier_indices.size(); ++i) {
        A.col(i) = epipolar_plane_normals.col(best_inlier_indices[i]);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        A.transpose(),
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector3d translation_final =
        svd.matrixV().rightCols<1>().normalized();

    // Final inlier check.
    best_inlier_indices.clear();
    for (int i = 0; i < num_measurements; ++i) {
        if (is_valid[i] == 0u) {
            continue;
        }

        const double error = std::abs(
            epipolar_plane_normals.col(i).dot(translation_final));

        if (error < ransac_threshold) {
            best_inlier_indices.push_back(i);
        }
    }

    VLOG(2) << "Num. Features " << num_measurements
            << ", num. inliers " << best_inlier_indices.size();
    CHECK_GT(num_measurements, 0);
    VLOG(3) << "Inlier ratio of 2-Point-RANSAC: "
            << static_cast<double>(best_inlier_indices.size()) /
                   static_cast<double>(num_measurements);

    is_valid = std::vector<unsigned char>(num_measurements, 0u);
    for (const auto idx : best_inlier_indices) {
        is_valid[idx] = 1u;
    }

    return true;
}

void FeatureTrackerBase::LengthFiltering(
        const int cam_idx,
        const common::VisualFrameData& frame_data_k,
        const common::VisualFrameData& frame_data_kp1,
        common::FrameToFrameMatchesWithScore* matches_ptr) {
    common::FrameToFrameMatchesWithScore& matches
            = *CHECK_NOTNULL(matches_ptr);

    // Too few feature points to be meshed.
    if (matches.size() < 3u) {
        return;
    }

    // Create vertices and compute optical flows.
    std::vector<Vertex> vertices;
    common::EigenVector2dVec optical_flows;
    vertices.reserve(matches.size());
    optical_flows.reserve(matches.size());
    for (size_t i = 0u; i < matches.size(); ++i) {
        const auto& match = matches[i];
        const int idx_kp1 = match.GetKeypointIndexAppleFrame();
        const Eigen::Vector2d& pt_kp1_eigen =
                frame_data_kp1.key_points.col(idx_kp1).head<2>();
        vertices.emplace_back(pt_kp1_eigen(X), pt_kp1_eigen(Y), i);

        const int idx_k = match.GetKeypointIndexBananaFrame();
        const Eigen::Vector2d& pt_k_eigen =
                frame_data_k.key_points.col(idx_k).head<2>();
        optical_flows.emplace_back(
            pt_kp1_eigen(X) - pt_k_eigen(X),
            pt_kp1_eigen(Y) - pt_k_eigen(Y));
    }

    // Generate mesh.
    DelaunayMeshGenerator mesh_generator;
    mesh_generator.GenerateMesh(vertices);
    const std::vector<Edge>& edges = mesh_generator.GetEdges();

    // Convert the mesh into adjacency matrix and check for local support.
    Eigen::MatrixXi adjacency_matrix(
        matches.size(),
        matches.size());
    adjacency_matrix.setZero();
    std::vector<int> support_count(matches.size(), 0);

    for (const auto& edge : edges) {
        const int vertex_index_1 = edge.v1->index;
        const int vertex_index_2 = edge.v2->index;

        if (adjacency_matrix(vertex_index_1, vertex_index_2) == 0) {
            // 1 represents an edge.
            adjacency_matrix(vertex_index_1, vertex_index_2) = 1;
            adjacency_matrix(vertex_index_2, vertex_index_1) = 1;

            // Check edge length.
            const double dist = edge.v1->Dist(*edge.v2);
            // Neighborhood size is set to one fifth of image height.
            const int image_width =
                    cameras_->getCamera(cam_idx).imageWidth();
            const int image_height =
                    cameras_->getCamera(cam_idx).imageHeight();
            if (dist > (image_width + image_height) * 0.5 * 0.3) {
                // Too far to support.
                continue;
            }

            const Eigen::Vector2d& flow_1 = optical_flows[vertex_index_1];
            const Eigen::Vector2d& flow_2 = optical_flows[vertex_index_2];

            const double flow_length_1 = flow_1.norm();
            const double flow_length_2 = flow_2.norm();

            // If one of the flows is too small, it is probably static.
            if (flow_length_1 < static_cast<double>(image_height) * 2e-3 ||
                flow_length_2 < static_cast<double>(image_height) * 2e-3) {
                if (flow_length_1 > static_cast<double>(image_height) * 4e-3 ||
                    flow_length_2 > static_cast<double>(image_height) * 4e-3) {
                    continue;
                } else {
                    // 2 represents an supporting edge.
                    adjacency_matrix(vertex_index_1, vertex_index_2) = 2;
                    adjacency_matrix(vertex_index_2, vertex_index_1) = 2;

                    ++support_count[vertex_index_1];
                    ++support_count[vertex_index_2];
                    continue;
                }
            }

            // Compute the relative length ratio of the two flows.
            CHECK_GT(flow_length_1, std::numeric_limits<double>::min());
            CHECK_GT(flow_length_2, std::numeric_limits<double>::min());
            const double length_ratio = flow_length_1 < flow_length_2 ?
                                        flow_length_2 / flow_length_1 :
                                        flow_length_1 / flow_length_2;

            // Compute the cosine of the angle of the two flows.
            const double cos_theta =
                flow_1.dot(flow_2) / (flow_length_1 * flow_length_2);

            constexpr double kOpticalFlowLengthRatioThresh = 4;
            constexpr double kOpticalFlowAngleCosineThresh = 0.5;
            if (length_ratio < kOpticalFlowLengthRatioThresh &&
                cos_theta > kOpticalFlowAngleCosineThresh) {
                adjacency_matrix(vertex_index_1, vertex_index_2) = 2;
                adjacency_matrix(vertex_index_2, vertex_index_1) = 2;

                ++support_count[vertex_index_1];
                ++support_count[vertex_index_2];
            }
        }
    }

    common::FrameToFrameMatchesWithScore matches_new;
    for (size_t i = 0u; i < matches.size(); ++i) {
        // We only accept flows with at least 2 supports.
        if (support_count[i] >= 2) {
            matches_new.emplace_back(matches[i]);
        }
    }

    matches = matches_new;
}

template <typename Type>
void FeatureTrackerBase::EraseVectorElementsByIndex(
        const std::unordered_set<size_t>& indices_to_erase,
        std::vector<Type>* vec) const {
  CHECK_NOTNULL(vec);
    std::vector<bool> erase_index(vec->size(), false);
    for (const size_t i: indices_to_erase) {
        erase_index[i] = true;
    }
    std::vector<bool>::const_iterator it_to_erase = erase_index.begin();
    typename std::vector<Type>::iterator it_erase_from = std::remove_if(
        vec->begin(), vec->end(),
        [&it_to_erase](const Type& /*whatever*/) -> bool {
          return *it_to_erase++ == true;
        }
    );
    vec->erase(it_erase_from, vec->end());
    vec->shrink_to_fit();
}

template
void FeatureTrackerBase::EraseVectorElementsByIndex<cv::Point2f>(
        const std::unordered_set<size_t>& indices_to_erase,
        std::vector<cv::Point2f>* vec) const;
        template
void FeatureTrackerBase::EraseVectorElementsByIndex<int>(
        const std::unordered_set<size_t>& indices_to_erase,
        std::vector<int>* vec) const;
}  // namespace vins_core
