#include "feature_tracker/gyro_two_frame_matcher.h"

#include <aslam/common/statistics/statistics.h>

namespace vins_core
{

inline float GetL1Different(
    const Eigen::VectorXf& descriptor1,
    const Eigen::VectorXf& descriptor2) {
    const uint32_t descriptor_size = descriptor1.size();
    const uint32_t descriptor2_size = descriptor2.size();
    CHECK_EQ(descriptor_size, descriptor2_size)
        << "Cannot compare descriptors of unequal size.";
    float l1_dis_avg = 0.f;
    for (uint32_t i = 0; i < descriptor_size; ++i) {
        l1_dis_avg += std::abs(descriptor1(i) - descriptor2(i)) /
                        static_cast<float>(descriptor_size);
    }
    return l1_dis_avg;
}

GyroTwoFrameMatcher::GyroTwoFrameMatcher(
    const common::SlamConfigPtr &config,
    const common::VisualFrameData &frame_kp1,
    const common::VisualFrameData &frame_k,
    const uint32_t image_height,
    const Eigen::Matrix2Xd &predicted_keypoint_positions_kp1,
    const std::vector<unsigned char> &prediction_success,
    common::FrameToFrameMatchesWithScore *matches_kp1_k_)
    : config_(config),
      frame_kp1_(frame_kp1), frame_k_(frame_k),
      predicted_keypoint_positions_kp1_(predicted_keypoint_positions_kp1),
      prediction_success_(prediction_success),
#ifdef USE_CNN_FEATURE
      kDescriptorSizeBytes_(frame_kp1.GetProjectedDescriptorSizeBytes()),
#else
      kDescriptorSizeBytes_(frame_kp1.GetDescriptorSizeBytes()),
#endif
      kNumPointsKp1_(static_cast<int>(frame_kp1.key_points.cols())),
      kNumPointsK_(static_cast<int>(frame_k.key_points.cols())),
      kImageHeight_(image_height),
      matches_kp1_k_(matches_kp1_k_),
      is_keypoint_kp1_matched_(kNumPointsKp1_, false),
      iteration_processed_keypoints_kp1_(kNumPointsKp1_, false),
      small_search_distance_px_(config->lk_small_search_distance_px),
      large_search_distance_px_(config->lk_large_search_distance_px) {
    CHECK_GT(frame_kp1.key_points.cols(), 0u);
    CHECK_GT(frame_k.key_points.cols(), 0u);
#ifdef USE_CNN_FEATURE
    CHECK_GT(frame_kp1.projected_descriptors.cols(), 0);
    CHECK_GT(frame_k.projected_descriptors.cols(), 0);
#else
    CHECK_GT(frame_kp1.descriptors.cols(), 0);
    CHECK_GT(frame_k.descriptors.cols(), 0);
#endif
    CHECK_GT(frame_kp1.key_points.cols(), 0u);
    CHECK_GT(frame_k.key_points.cols(), 0u);
    CHECK_NOTNULL(matches_kp1_k_)->clear();
#ifdef USE_CNN_FEATURE
    CHECK_EQ(kNumPointsKp1_, frame_kp1.projected_descriptors.cols()) <<
        "Number of keypoints and descriptors in frame k+1 is not the same.";
    CHECK_EQ(kNumPointsK_, frame_k.projected_descriptors.cols()) <<
        "Number of keypoints and descriptors in frame k is not the same.";
#else
    CHECK_EQ(kNumPointsKp1_, frame_kp1.descriptors.cols()) <<
        "Number of keypoints and descriptors in frame k+1 is not the same.";
    CHECK_EQ(kNumPointsK_, frame_k.descriptors.cols()) <<
        "Number of keypoints and descriptors in frame k is not the same.";
    CHECK_LE(kDescriptorSizeBytes_ * 8, 512u) << "Usually binary descriptors' size "
            "is less or equal to 512 bits. Adapt the following check if this "
            "framework uses larger binary descriptors.";
#endif
    CHECK_GT(kImageHeight_, 0u);
    CHECK_EQ(static_cast<int>(iteration_processed_keypoints_kp1_.size()),
                kNumPointsKp1_);
    CHECK_EQ(static_cast<int>(is_keypoint_kp1_matched_.size()),
                kNumPointsKp1_);
    CHECK_EQ(static_cast<int>(prediction_success_.size()),
                predicted_keypoint_positions_kp1_.cols());
    CHECK_GT(small_search_distance_px_, 0);
    CHECK_GT(large_search_distance_px_, 0);
    CHECK_GE(large_search_distance_px_, small_search_distance_px_);

    descriptors_kp1_wrapped_.reserve(kNumPointsKp1_);
    descriptors_k_wrapped_.reserve(kNumPointsK_);
    keypoints_kp1_sorted_by_y_.reserve(kNumPointsKp1_);
    matches_kp1_k_->reserve(kNumPointsK_);
    corner_row_LUT_.reserve(kImageHeight_);
}

void GyroTwoFrameMatcher::Initialize() {
    // Prepare descriptors for efficient matching.
#ifdef USE_CNN_FEATURE
    const common::DescriptorsMatF32& descriptors_kp1 =
        frame_kp1_.projected_descriptors;
    const common::DescriptorsMatF32& descriptors_k =
        frame_k_.projected_descriptors;
#else
    const common::DescriptorsMatUint8& descriptors_kp1 =
        frame_kp1_.descriptors;
    const common::DescriptorsMatUint8& descriptors_k =
        frame_k_.descriptors;
#endif

    for (int descriptor_kp1_idx = 0; descriptor_kp1_idx < kNumPointsKp1_;
            ++descriptor_kp1_idx) {
#ifdef USE_CNN_FEATURE
        const Eigen::VectorXf& desc = descriptors_kp1.col(descriptor_kp1_idx);
        descriptors_kp1_wrapped_.emplace_back(desc);
#else
        descriptors_kp1_wrapped_.emplace_back(
            &(descriptors_kp1.coeffRef(0, descriptor_kp1_idx)), kDescriptorSizeBytes_);
#endif
    }

    for (int descriptor_k_idx = 0; descriptor_k_idx < kNumPointsK_;
            ++descriptor_k_idx) {
#ifdef USE_CNN_FEATURE
        const Eigen::VectorXf& desc = descriptors_k.col(descriptor_k_idx);
        descriptors_k_wrapped_.push_back(desc);
#else
        descriptors_k_wrapped_.emplace_back(
            &(descriptors_k.coeffRef(0, descriptor_k_idx)), kDescriptorSizeBytes_);
#endif
    }

    // Sort keypoints of frame (k+1) from small to large y coordinates.
    for (int i = 0; i < kNumPointsKp1_; ++i) {
        keypoints_kp1_sorted_by_y_.emplace_back(
            frame_kp1_.key_points.col(i).head<2>(), i);
    }

    std::sort(keypoints_kp1_sorted_by_y_.begin(), keypoints_kp1_sorted_by_y_.end(),
            [](const KeypointData &lhs, const KeypointData &rhs) -> bool {
                    return lhs.measurement(1) < rhs.measurement(1);
            });

    // Lookup table construction.
    int v = 0;
    for (size_t y = 0u; y < kImageHeight_; ++y) {
        while (v < kNumPointsKp1_ &&
                y > static_cast<size_t>(keypoints_kp1_sorted_by_y_[v].measurement(1))) {
            ++v;
        }
        corner_row_LUT_.push_back(v);
    }
    CHECK_EQ(corner_row_LUT_.size(), kImageHeight_);
}

void GyroTwoFrameMatcher::Match() {
    Initialize();

    if (kNumPointsK_ == 0 || kNumPointsKp1_ == 0) {
        return;
    }

    for (int i = 0; i < kNumPointsK_; ++i) {
        MatchKeypoint(i);
    }

    std::vector<bool> is_inferior_keypoint_kp1_matched(
        is_keypoint_kp1_matched_);
    for (size_t i = 0u; i < kMaxNumInferiorIterations; ++i) {
        if (!MatchInferiorMatches(&is_inferior_keypoint_kp1_matched)) {
            return;
        }
    }
}

void GyroTwoFrameMatcher::MatchKeypoint(const int idx_k) {
    if (!prediction_success_[idx_k]) {
        return;
    }

    std::fill(iteration_processed_keypoints_kp1_.begin(),
              iteration_processed_keypoints_kp1_.end(),
              false);

    bool found = false;
    bool passed_ratio_test = false;
    int n_processed_corners = 0;
    KeyPointIterator it_best;

#ifdef USE_CNN_FEATURE
    const static float kMaxDistance = 1.f;
    float best_score = kMaxDistance * kMatchingThresholdBitsRatioRelaxed;
    float distance_best = kMaxDistance;
    float distance_second_best = kMaxDistance;
    const Eigen::VectorXf &descriptor_k = descriptors_k_wrapped_[idx_k];
#else
    const static unsigned int kMaxDistance = 8 * kDescriptorSizeBytes_;
    int best_score = static_cast<int>(
        kMaxDistance * kMatchingThresholdBitsRatioRelaxed);
    unsigned int distance_best = kMaxDistance + 1;
    unsigned int distance_second_best = kMaxDistance + 1;
    const aslam::common::FeatureDescriptorConstRef &descriptor_k =
        descriptors_k_wrapped_[idx_k];
#endif

    Eigen::Vector2d predicted_keypoint_position_kp1 =
        predicted_keypoint_positions_kp1_.block<2, 1>(0, idx_k);
    KeyPointIterator nearest_corners_begin, nearest_corners_end;
    GetKeypointIteratorsInWindow(
        predicted_keypoint_position_kp1,
        small_search_distance_px_,
        &nearest_corners_begin,
        &nearest_corners_end);

    const int bound_left_nearest =
        predicted_keypoint_position_kp1(0) - small_search_distance_px_;
    const int bound_right_nearest =
        predicted_keypoint_position_kp1(0) + small_search_distance_px_;

    MatchData current_match_data;

    // First search small window.
    for (KeyPointIterator it = nearest_corners_begin; it != nearest_corners_end; ++it) {
        if (it->measurement(0) < bound_left_nearest ||
            it->measurement(0) > bound_right_nearest) {
            continue;
        }

        CHECK_LT(it->channel_index, kNumPointsKp1_);
        CHECK_GE(it->channel_index, 0);

#ifdef USE_CNN_FEATURE
        const Eigen::VectorXf &descriptor_kp1 =
            descriptors_kp1_wrapped_[it->channel_index];

        float distance = GetL1Different(descriptor_k, descriptor_kp1);
        float current_score = kMaxDistance - distance;        
#else
        const aslam::common::FeatureDescriptorConstRef &descriptor_kp1 =
            descriptors_kp1_wrapped_[it->channel_index];
        unsigned int distance =
            aslam::common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);
        int current_score = kMaxDistance - distance;
#endif

        if (current_score > best_score) {
            best_score = current_score;
            distance_second_best = distance_best;
            distance_best = distance;
            it_best = it;
            found = true;
        } else if (distance < distance_second_best) {
            // The second best distance can also belong
            // to two descriptors that do not qualify as match.
            distance_second_best = distance;
        }
        iteration_processed_keypoints_kp1_[it->channel_index] = true;
        ++n_processed_corners;
        const double current_matching_score =
            ComputeMatchingScore(static_cast<float>(current_score),
                                    static_cast<float>(kMaxDistance));
        current_match_data.AddCandidate(it, current_matching_score);
    }

    // If no match in small window, increase window and search again.
    if (!found) {
        const int bound_left_near =
            predicted_keypoint_position_kp1(0) - large_search_distance_px_;
        const int bound_right_near =
            predicted_keypoint_position_kp1(0) + large_search_distance_px_;

        KeyPointIterator near_corners_begin, near_corners_end;
        GetKeypointIteratorsInWindow(
            predicted_keypoint_position_kp1,
            large_search_distance_px_,
            &near_corners_begin,
            &near_corners_end);

        for (KeyPointIterator it = near_corners_begin; it != near_corners_end; ++it) {
            if (iteration_processed_keypoints_kp1_[it->channel_index]) {
                continue;
            }
            if (it->measurement(0) < bound_left_near ||
                it->measurement(0) > bound_right_near) {
                continue;
            }
            CHECK_LT(it->channel_index, kNumPointsKp1_);
            CHECK_GE(it->channel_index, 0);
#ifdef USE_CNN_FEATURE
            const Eigen::VectorXf &descriptor_kp1 =
                descriptors_kp1_wrapped_[it->channel_index];
            float distance = GetL1Different(descriptor_k, descriptor_kp1);
            float current_score = kMaxDistance - distance;
#else
            const aslam::common::FeatureDescriptorConstRef &descriptor_kp1 =
                descriptors_kp1_wrapped_[it->channel_index];
            unsigned int distance =
                aslam::common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);
            int current_score = kMaxDistance - distance;
#endif
            if (current_score > best_score) {
                best_score = current_score;
                distance_second_best = distance_best;
                distance_best = distance;
                it_best = it;
                found = true;
            } else if (distance < distance_second_best) {
                // The second best distance can also belong
                // to two descriptors that do not qualify as match.
                distance_second_best = distance;
            }
            ++n_processed_corners;
            const double current_matching_score =
                ComputeMatchingScore(static_cast<float>(current_score),
                                        static_cast<float>(kMaxDistance));
            current_match_data.AddCandidate(it, current_matching_score);
        }
    }

    if (found) {
        passed_ratio_test = RatioTest(static_cast<float>(kMaxDistance),
                                        static_cast<float>(distance_best),
                                        static_cast<float>(distance_second_best));
    }

    if (passed_ratio_test) {
        CHECK(idx_k_to_attempted_match_data_map_.insert(
                                                    std::make_pair(idx_k, current_match_data))
                    .second);
        const int best_match_keypoint_idx_kp1 = it_best->channel_index;
        const double matching_score =
            ComputeMatchingScore(static_cast<float>(best_score),
                                    static_cast<float>(kMaxDistance));
        if (is_keypoint_kp1_matched_[best_match_keypoint_idx_kp1])
        {
            if (matching_score > kp1_idx_to_matches_iterator_map_[best_match_keypoint_idx_kp1]->GetScore()) {
                // The current match is better than a previous match associated with the
                // current keypoint of frame (k+1). Hence, the inferior match is the
                // previous match associated with the current keypoint of frame (k+1).
                const int inferior_keypoint_idx_k = 
                    kp1_idx_to_matches_iterator_map_[best_match_keypoint_idx_kp1]->GetKeypointIndexBananaFrame();
                
                inferior_match_keypoint_idx_k_.push_back(inferior_keypoint_idx_k);

                kp1_idx_to_matches_iterator_map_[best_match_keypoint_idx_kp1]->SetScore(matching_score);
                kp1_idx_to_matches_iterator_map_[best_match_keypoint_idx_kp1]->SetIndexApple(best_match_keypoint_idx_kp1);
                kp1_idx_to_matches_iterator_map_[best_match_keypoint_idx_kp1]->SetIndexBanana(idx_k);
            } else {
                // The current match is inferior to a previous match associated with the
                // current keypoint of frame (k+1).
                inferior_match_keypoint_idx_k_.push_back(idx_k);
            }
        } else {
            is_keypoint_kp1_matched_[best_match_keypoint_idx_kp1] = true;
            matches_kp1_k_->emplace_back(best_match_keypoint_idx_kp1, idx_k, matching_score);

            CHECK(matches_kp1_k_->end() != matches_kp1_k_->begin())
                << "Match vector should not be empty.";
            CHECK(kp1_idx_to_matches_iterator_map_.emplace(
                                                        best_match_keypoint_idx_kp1, matches_kp1_k_->end() - 1)
                        .second);
        }
    }
}

bool GyroTwoFrameMatcher::MatchInferiorMatches(
    std::vector<bool>* is_inferior_keypoint_kp1_matched) {
    CHECK_NOTNULL(is_inferior_keypoint_kp1_matched);
    CHECK_EQ(is_inferior_keypoint_kp1_matched->size(), is_keypoint_kp1_matched_.size());

    bool found_inferior_match = false;

    std::unordered_set<int> erase_inferior_match_keypoint_idx_k;
    for (const int inferior_keypoint_idx_k : inferior_match_keypoint_idx_k_) {
        const MatchData& match_data =
            idx_k_to_attempted_match_data_map_[inferior_keypoint_idx_k];
        bool found = false;
        double best_matching_score = static_cast<double>(kMatchingThresholdBitsRatioStrict);
        KeyPointIterator it_best;

        for (size_t i = 0u; i < match_data.keypoint_match_candidates_kp1.size(); ++i) {
            const KeyPointIterator& keypoint_kp1 = match_data.keypoint_match_candidates_kp1[i];
            const double matching_score = match_data.match_candidate_matching_scores[i];
            // Make sure that we don't try to match with already matched keypoints
            // of frame (k+1) (also previous inferior matches).
            if (is_keypoint_kp1_matched_[keypoint_kp1->channel_index]) {
                continue;
            }
            if (matching_score > best_matching_score) {
                it_best = keypoint_kp1;
                best_matching_score = matching_score;
                found = true;
            }
        }

        if (found) {
            found_inferior_match = true;
            const int best_match_keypoint_idx_kp1 = it_best->channel_index;
            if ((*is_inferior_keypoint_kp1_matched)[best_match_keypoint_idx_kp1]) {
                if (best_matching_score > kp1_idx_to_matches_iterator_map_
                                                [best_match_keypoint_idx_kp1]
                                                    ->GetScore()) {
                    // The current match is better than a previous match associated with the
                    // current keypoint of frame (k+1). Hence, the revoked match is the
                    // previous match associated with the current keypoint of frame (k+1).
                    const int revoked_inferior_keypoint_idx_k =
                        kp1_idx_to_matches_iterator_map_
                            [best_match_keypoint_idx_kp1]
                                ->GetKeypointIndexBananaFrame();
                    // The current keypoint k does not have to be matched anymore
                    // in the next iteration.
                    erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);
                    // The keypoint k that was revoked. That means that it can be matched
                    // again in the next iteration.
                    erase_inferior_match_keypoint_idx_k.erase(revoked_inferior_keypoint_idx_k);

                    kp1_idx_to_matches_iterator_map_
                        [best_match_keypoint_idx_kp1]->SetScore(best_matching_score);
                    kp1_idx_to_matches_iterator_map_
                        [best_match_keypoint_idx_kp1]->SetIndexApple(best_match_keypoint_idx_kp1);
                    kp1_idx_to_matches_iterator_map_
                        [best_match_keypoint_idx_kp1]->SetIndexBanana(inferior_keypoint_idx_k);
                }
            }
            else
            {
                (*is_inferior_keypoint_kp1_matched)[best_match_keypoint_idx_kp1] = true;
                matches_kp1_k_->emplace_back(
                    best_match_keypoint_idx_kp1, inferior_keypoint_idx_k, best_matching_score);
                erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);

                CHECK(matches_kp1_k_->end() != matches_kp1_k_->begin())
                    << "Match vector should not be empty.";
                CHECK(kp1_idx_to_matches_iterator_map_.emplace(
                                                            best_match_keypoint_idx_kp1, matches_kp1_k_->end() - 1)
                            .second);
            }
        }
    }

    if (erase_inferior_match_keypoint_idx_k.size() > 0u) {
        // Do not iterate again over newly matched keypoints of frame k.
        // Hence, remove the matched keypoints.
        std::vector<int>::iterator iter_erase_from = std::remove_if(
            inferior_match_keypoint_idx_k_.begin(), inferior_match_keypoint_idx_k_.end(),
            [&erase_inferior_match_keypoint_idx_k](const int element) -> bool {
                return erase_inferior_match_keypoint_idx_k.count(element) == 1u;
            });
        inferior_match_keypoint_idx_k_.erase(
            iter_erase_from, inferior_match_keypoint_idx_k_.end());
    }

    // Subsequent iterations should not mess with the current matches.
    is_keypoint_kp1_matched_ = *is_inferior_keypoint_kp1_matched;

    return found_inferior_match;
}
} // namespace vins_core
