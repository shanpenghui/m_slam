#include "feature_tracker/gyro_tracker.h"

namespace vins_core {

GyroTracker::GyroTracker(
    const aslam::NCamera::Ptr& cameras,
    const common::SlamConfigPtr& config)
    : FeatureTrackerBase(cameras, config),
      initialized_(false) {
}

void GyroTracker::TrackFeature(
    const int cam_idx,
    const Eigen::Quaterniond& q_kp1_k,
    const common::VisualFrameData& frame_k,
    common::VisualFrameData* frame_kp1_ptr,
    common::FrameToFrameMatchesWithScore* matches_kp1_k_ptr,
    int* track_id_provider_ptr) {
    common::VisualFrameData& frame_kp1 =
            *CHECK_NOTNULL(frame_kp1_ptr);
    common::FrameToFrameMatchesWithScore& matches_kp1_k =
            *CHECK_NOTNULL(matches_kp1_k_ptr);
    int& track_id_provider = *CHECK_NOTNULL(track_id_provider_ptr);

    if (frame_k.key_points.cols() == 0 ||
            frame_kp1.key_points.cols() == 0) {
        return;
    }

    if (config_->lk_candidates_ratio > 0.0) {
        UpdateTrackIdDeque(frame_k);
        if (!initialized_) {
            InitializeFeatureStatusDeque();
        }
    }

    // Predict keypoint positions for all keypoints in current frame k.
    Eigen::Matrix2Xd predicted_keypoint_positions_kp1;
    std::vector<unsigned char> prediction_success;
    PredictKeypointsByRotation(cameras_->getCamera(cam_idx),
                               frame_k.key_points.block(
                                   0, 0, 2, frame_k.key_points.cols()),
                               q_kp1_k,
                               &predicted_keypoint_positions_kp1,
                               &prediction_success);
                          
    CHECK_EQ(static_cast<int>(prediction_success.size()),
               predicted_keypoint_positions_kp1.cols());

    // Match descriptors of frame k with those of frame (k+1).
    GyroTwoFrameMatcher matcher(
                config_,
                frame_kp1, frame_k,
                cameras_->getCamera(cam_idx).imageHeight(),
                predicted_keypoint_positions_kp1,
                prediction_success,
                matches_kp1_k_ptr);
    matcher.Match();

    if (config_->lk_candidates_ratio > 0.0) {
      // Compute LK candidates and track them.
        FrameStatusTrackLength status_track_length_k;
        std::vector<TrackedMatch> tracked_matches;
        std::vector<int> lk_candidate_indices_k;

        ComputeTrackedMatches(&tracked_matches);
        ComputeStatusTrackLengthOfFrameK(tracked_matches, &status_track_length_k);
        ComputeLKCandidates(matches_kp1_k, status_track_length_k,
                            frame_k, frame_kp1, &lk_candidate_indices_k);
        LkTracking(cam_idx, predicted_keypoint_positions_kp1, prediction_success,
                    lk_candidate_indices_k, frame_k, &frame_kp1, &matches_kp1_k);

        frame_kp1.UpdateTrackIds(&track_id_provider);

        status_track_length_km1_.swap(status_track_length_k);
        initialized_ = true;
    }
}

void GyroTracker::UpdateTrackIdDeque(
    const common::VisualFrameData& frame_data_k) {
    const Eigen::VectorXi& track_ids_k = frame_data_k.track_ids;

    track_ids_k_km1_.push_front(track_ids_k);
    if (track_ids_k_km1_.size() == 3u) {
    track_ids_k_km1_.pop_back();
    }
}

void GyroTracker::InitializeFeatureStatusDeque() {
    CHECK_EQ(track_ids_k_km1_.size(), 1u);
    const size_t num_points_k = track_ids_k_km1_[0].size();
    FrameFeatureStatus frame_feature_status_k(num_points_k, FeatureStatus::kDetected);
    UpdateFeatureStatusDeque(frame_feature_status_k);
}

void GyroTracker::UpdateFeatureStatusDeque(
    const FrameFeatureStatus& frame_feature_status_kp1) {
    feature_status_k_km1_.emplace_front(
        frame_feature_status_kp1.begin(), frame_feature_status_kp1.end());
    if (feature_status_k_km1_.size() == 3u) {
        feature_status_k_km1_.pop_back();
    }
}

void GyroTracker::ComputeTrackedMatches(
    std::vector<TrackedMatch>* tracked_matches) const {
    CHECK_NOTNULL(tracked_matches)->clear();
    if (!initialized_) {
        return;
    }
    CHECK_EQ(track_ids_k_km1_.size(), 2u);

    // Get the index of an integer value in a vector of unique elements.
    // Return -1 if there is no such value in the vector.
    std::function<int(const std::vector<int>&, const int)> GetIndexOfValue =
      [](const std::vector<int>& vec, const int value) -> int {
        std::vector<int>::const_iterator iter = std::find(vec.begin(), vec.end(), value);
        if (iter == vec.end()) {
            return -1;
        } else {
            return static_cast<int>(std::distance(vec.begin(), iter));
        }
    };

    const Eigen::VectorXi& track_ids_k_eigen = track_ids_k_km1_[0];
    const Eigen::VectorXi& track_ids_km1_eigen = track_ids_k_km1_[1];
    const std::vector<int> track_ids_km1(
        track_ids_km1_eigen.data(),
        track_ids_km1_eigen.data() + track_ids_km1_eigen.size());
    for (int index_k = 0; index_k < track_ids_k_eigen.size(); ++index_k) {
        const int track_id_k = track_ids_k_eigen(index_k);
        // Skip invalid track IDs.
        if (track_id_k == -1) {
            continue;
        }
        const int index_km1 = GetIndexOfValue(track_ids_km1, track_ids_k_eigen(index_k));
        if (index_km1 >= 0) {
            tracked_matches->emplace_back(index_k, index_km1);
        }
    }
}

void GyroTracker::ComputeStatusTrackLengthOfFrameK(
    const std::vector<TrackedMatch>& tracked_matches,
    FrameStatusTrackLength* status_track_length_k) {
    CHECK_NOTNULL(status_track_length_k)->clear();
    CHECK_GT(track_ids_k_km1_.size(), 0u);

    const int kNumPointsK = track_ids_k_km1_[0].size();
    status_track_length_k->assign(kNumPointsK, 0u);

    if (!initialized_) {
        return;
    }
    CHECK_EQ(feature_status_k_km1_.size(), 2u);

    for (const TrackedMatch& match: tracked_matches) {
        const int match_index_k = match.first;
        const int match_index_km1 = match.second;
        if (feature_status_k_km1_[1][match_index_km1] !=
            feature_status_k_km1_[0][match_index_k]) {
        // Reset the status track length to 1 because the status of this
        // particular tracked keypoint has changed from frame (k-1) to k.
            (*status_track_length_k)[match_index_k] = 1u;
        } else {
            (*status_track_length_k)[match_index_k] =
            status_track_length_km1_[match_index_km1] + 1u;
        }
    }
}

void GyroTracker::ComputeUnmatchedIndicesOfFrameK(
    const common::FrameToFrameMatchesWithScore& matches_kp1_k,
    std::vector<int>* unmatched_indices_k) const {
    CHECK_GT(track_ids_k_km1_.size(), 0u);
    CHECK_GE(track_ids_k_km1_[0].size(), static_cast<int>(matches_kp1_k.size()));
    CHECK_NOTNULL(unmatched_indices_k)->clear();

    const size_t kNumPointsK = track_ids_k_km1_[0].size();
    const size_t kNumMatchesK = matches_kp1_k.size();
    const size_t kNumUnmatchedK = kNumPointsK - kNumMatchesK;

    unmatched_indices_k->reserve(kNumUnmatchedK);
    std::vector<bool> is_unmatched(kNumPointsK, true);

    for (const common::FrameToFrameMatchWithScore& match: matches_kp1_k) {
        is_unmatched[match.GetKeypointIndexBananaFrame()] = false;
    }

    for (int i = 0; i < static_cast<int>(kNumPointsK); ++i) {
        if (is_unmatched[i]) {
            unmatched_indices_k->push_back(i);
        }
    }

    CHECK_EQ(unmatched_indices_k->size(), kNumUnmatchedK);
    CHECK_EQ(kNumMatchesK + unmatched_indices_k->size(),
                kNumPointsK);
}

void GyroTracker::ComputeLKCandidates(
    const common::FrameToFrameMatchesWithScore& matches_kp1_k,
    const FrameStatusTrackLength& status_track_length_k,
    const common::VisualFrameData& frame_k,
    const common::VisualFrameData& frame_kp1,
    std::vector<int>* lk_candidate_indices_k) const {
    CHECK_NOTNULL(lk_candidate_indices_k)->clear();
    CHECK_EQ(
    static_cast<int>(status_track_length_k.size()), track_ids_k_km1_[0].size());

    std::vector<int> unmatched_indices_k;
    ComputeUnmatchedIndicesOfFrameK(
        matches_kp1_k, &unmatched_indices_k);

    typedef std::pair<int, size_t> IndexTrackLengthPair;

    std::vector<IndexTrackLengthPair> indices_detected_and_tracked;
    std::vector<IndexTrackLengthPair> indices_lktracked;
    for (const int unmatched_index_k: unmatched_indices_k) {
        const int current_status_track_length =
            status_track_length_k[unmatched_index_k];
        const FeatureStatus current_feature_status =
            feature_status_k_km1_[0][unmatched_index_k];
        if (current_feature_status == FeatureStatus::kDetected) {
            if (current_status_track_length >= config_->lk_track_detected_threshold) {
                // These candidates have the highest priority as lk candidates.
                // The most valuable candidates have the longest status track length.
                indices_detected_and_tracked.emplace_back(
                unmatched_index_k, current_status_track_length);
            }
        } else if (current_feature_status == FeatureStatus::kLkTracked) {
            if (current_status_track_length < config_->lk_max_status_track_length) {
                // These candidates have the lowest priority as lk candidates.
                // The most valuable candidates have the shortest status track length.
                indices_lktracked.emplace_back(
                    unmatched_index_k, current_status_track_length);
            }
        } else {
            VLOG(2) << "Unknown feature status.";
        }
    }

    const size_t kNumPointsKp1 = frame_kp1.track_ids.rows();
    CHECK_EQ(static_cast<int>(kNumPointsKp1), frame_kp1.descriptors.cols());
    const size_t kLkNumCandidatesBeforeCutoff =
        indices_detected_and_tracked.size() + indices_lktracked.size();
    const size_t kLkNumMaxCandidates = static_cast<size_t>(
        kNumPointsKp1*config_->lk_candidates_ratio);
    const size_t kNumLkCandidatesAfterCutoff = std::min(
        kLkNumCandidatesBeforeCutoff, kLkNumMaxCandidates);
    lk_candidate_indices_k->reserve(kNumLkCandidatesAfterCutoff);

    // Only sort indices that are possible candidates.
    if (kLkNumCandidatesBeforeCutoff > kLkNumMaxCandidates) {
        std::sort(indices_detected_and_tracked.begin(),
              indices_detected_and_tracked.end(),
              [](const IndexTrackLengthPair& lhs,
                  const IndexTrackLengthPair& rhs) -> bool {
                    return lhs.second > rhs.second;
                });
        if (indices_detected_and_tracked.size() < kLkNumMaxCandidates) {
            std::sort(indices_lktracked.begin(),
                    indices_lktracked.end(),
                    [](const IndexTrackLengthPair& lhs,
                        const IndexTrackLengthPair& rhs) -> bool {
                        return lhs.second < rhs.second;
                    });
        }
    }

    // Construct candidate vector based on sorted candidate indices
    // until max number of candidates is reached.
    size_t counter = 0u;
    for (const IndexTrackLengthPair& pair: indices_detected_and_tracked) {
        if (counter == kLkNumMaxCandidates) {
            break;
        } 
        lk_candidate_indices_k->push_back(pair.first);
        ++counter;
    }
    for (const IndexTrackLengthPair& pair: indices_lktracked) {
        if (counter == kLkNumMaxCandidates) {
            break;
        } 
        lk_candidate_indices_k->push_back(pair.first);
        ++counter;
    }

    VLOG(4) << "(num LK candidates before cut-off)/"
        "(num detected features in frame k+1): " <<
        kLkNumCandidatesBeforeCutoff/static_cast<float>(kNumPointsKp1) <<
        ". Cut-off ratio: " << config_->lk_candidates_ratio;
}

void GyroTracker::LkTracking(
    const int cam_idx,
    const Eigen::Matrix2Xd& predicted_keypoint_positions_kp1,
    const std::vector<unsigned char>& prediction_success,
    const std::vector<int>& lk_candidate_indices_k,
    const common::VisualFrameData& frame_k,
    common::VisualFrameData* frame_kp1,
    common::FrameToFrameMatchesWithScore* matches_kp1_k) {
    CHECK_NOTNULL(frame_kp1);
    CHECK_NOTNULL(matches_kp1_k);
    CHECK_EQ(static_cast<int>(prediction_success.size()),
        predicted_keypoint_positions_kp1.cols());
    CHECK_LE(lk_candidate_indices_k.size(), prediction_success.size());

    const int kInitialSizeKp1 = frame_kp1->track_ids.rows();

    // Definite lk indices are the subset of lk candidate indices with
    // successfully predicted keypoint locations in frame (k+1).
    std::vector<int> lk_definite_indices_k;
    for (const int candidate_index_k: lk_candidate_indices_k) {
        if (prediction_success[candidate_index_k] == 1) {
            lk_definite_indices_k.push_back(candidate_index_k);
        }
    }

    if (lk_definite_indices_k.empty()) {
        // This means that we won't insert any new keypoints into frame (k+1).
        // Since only inserted keypoints are those that are lk-tracked, all
        // keypoints in frame (k+1) were detected.
        // Update feature status for next iteration.
        FrameFeatureStatus frame_feature_status_kp1(kInitialSizeKp1);
        std::fill(frame_feature_status_kp1.begin(), frame_feature_status_kp1.end(),
                    FeatureStatus::kDetected);
        UpdateFeatureStatusDeque(frame_feature_status_kp1);
        VLOG(4) << "No LK candidates to track.";
        return;
    }

  // Get definite lk keypoint locations in OpenCV format.
    std::vector<cv::Point2f> lk_cv_points_k;
    std::vector<cv::Point2f> lk_cv_points_kp1;
    lk_cv_points_k.reserve(lk_definite_indices_k.size());
    lk_cv_points_kp1.reserve(lk_definite_indices_k.size());
    for (const int lk_definite_index_k: lk_definite_indices_k) {
        // Compute Cv points in frame k.
        const Eigen::Vector2d& lk_keypoint_location_k =
        frame_k.key_points.col(lk_definite_index_k).head<2>();
        lk_cv_points_k.emplace_back(
            static_cast<float>(lk_keypoint_location_k(X)),
            static_cast<float>(lk_keypoint_location_k(Y)));
    // Compute predicted locations in frame (k+1).
        lk_cv_points_kp1.emplace_back(
            static_cast<float>(predicted_keypoint_positions_kp1(X, lk_definite_index_k)),
            static_cast<float>(predicted_keypoint_positions_kp1(Y, lk_definite_index_k)));
    }

    std::vector<unsigned char> lk_tracking_success;
    std::vector<float> lk_tracking_errors;
    CHECK_NOTNULL(frame_kp1->image_ptr);
    CHECK_NOTNULL(frame_k.image_ptr);
    cv::calcOpticalFlowPyrLK(
        *(frame_k.image_ptr), *(frame_kp1->image_ptr), lk_cv_points_k,
        lk_cv_points_kp1, lk_tracking_success, lk_tracking_errors,
        cv::Size(config_->lk_window_size, config_->lk_window_size),
        config_->lk_max_pyramid_levels,
        cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW,
        config_->lk_min_eigenvalue_threshold);

    CHECK_EQ(lk_tracking_success.size(), lk_definite_indices_k.size());
    CHECK_EQ(lk_cv_points_kp1.size(), lk_tracking_success.size());
    CHECK_EQ(lk_cv_points_k.size(), lk_cv_points_kp1.size());

    std::function<bool(const cv::Point2f&, const aslam::Camera&)> is_outside_roi =
        [this](const cv::Point2f& point, const aslam::Camera& camera) -> bool {
        return point.x < config_->min_distance_to_image_border_px ||
            point.x >= (camera.imageWidth() -
                        config_->min_distance_to_image_border_px) ||
            point.y < config_->min_distance_to_image_border_px ||
            point.y >= (camera.imageHeight() -
                        config_->min_distance_to_image_border_px);
    };

    std::unordered_set<size_t> indices_to_erase;
    for (size_t i = 0u; i < lk_tracking_success.size(); ++i) {
        if (lk_tracking_success[i] == 0u ||
                is_outside_roi(lk_cv_points_kp1[i], cameras_->getCamera(cam_idx))) {
        indices_to_erase.insert(i);
        }
    }
    EraseVectorElementsByIndex(indices_to_erase, &lk_definite_indices_k);
    EraseVectorElementsByIndex(indices_to_erase, &lk_cv_points_kp1);

    const size_t kNumPointsSuccessfullyTracked = lk_cv_points_kp1.size();

    // Convert Cv points to Cv keypoints because this format is
    // required for descriptor extraction. Take relevant keypoint information
    // (such as score and size) from frame k.
    // Assign unique class_id to keypoints because some of them will get removed
    // during the extraction phase and we want to be able to identify them.
    std::vector<cv::KeyPoint> lk_cv_keypoints_kp1;
    lk_cv_keypoints_kp1.reserve(kNumPointsSuccessfullyTracked);
    for (size_t i = 0u; i < kNumPointsSuccessfullyTracked; ++i) {
        const size_t channel_idx = lk_definite_indices_k[i];
        const int class_id = static_cast<int>(i);
        lk_cv_keypoints_kp1.emplace_back(
            lk_cv_points_kp1[i], frame_k.key_points.col(channel_idx)(SIZE),
            frame_k.key_points.col(channel_idx)(ANGLE),
            frame_k.key_points.col(channel_idx)(RESPONSE),
            0 /* Octave info not used by extractor */, class_id);
    }

    cv::Mat image_gray;
    if (frame_kp1->image_ptr->type() == CV_8UC3) {
        cv::cvtColor(*(frame_kp1->image_ptr), image_gray, cv::COLOR_BGR2GRAY);
    } else if (frame_kp1->image_ptr->type() == CV_8UC1) {
        image_gray = *(frame_kp1->image_ptr);
    } else {
        LOG(FATAL) << "Unsupport image type.";
    }

    cv::Mat lk_descriptors_kp1;
    feature_extractor_->Extract(image_gray,
                                &lk_cv_keypoints_kp1,
                                &lk_descriptors_kp1);
    CHECK_EQ(lk_descriptors_kp1.type(), CV_8UC1);

    const size_t kNumPointsAfterExtraction = lk_cv_keypoints_kp1.size();

    for (int i = 0; i < static_cast<int>(kNumPointsAfterExtraction); ++i) {
        matches_kp1_k->emplace_back(
            kInitialSizeKp1 + i, lk_definite_indices_k[lk_cv_keypoints_kp1[i].class_id],
            0.0 /* We don't have scores for lk tracking */);
    }

    // Update feature status for next iteration.
    const size_t extended_size_pk1 = static_cast<size_t>(kInitialSizeKp1) +
        kNumPointsAfterExtraction;
    FrameFeatureStatus frame_feature_status_kp1(extended_size_pk1);
    std::fill(frame_feature_status_kp1.begin(), frame_feature_status_kp1.begin() +
                kInitialSizeKp1, FeatureStatus::kDetected);
    std::fill(frame_feature_status_kp1.begin() + kInitialSizeKp1,
                frame_feature_status_kp1.end(), FeatureStatus::kLkTracked);
    UpdateFeatureStatusDeque(frame_feature_status_kp1);

    if (lk_descriptors_kp1.empty()) {
        return;
    }
    CHECK(lk_descriptors_kp1.isContinuous());

  // Add keypoints and descriptors to frame (k+1).
    common::InsertAdditionalCvKeypointsAndDescriptorsToVisualFrame(
        lk_cv_keypoints_kp1, lk_descriptors_kp1,
        config_->keypoint_uncertainty_px,
        frame_kp1);
}
}  // namespace vins_core
