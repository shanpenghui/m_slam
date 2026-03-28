#ifndef MVINS_GYRO_TRECKER_H_
#define MVINS_GYRO_TRECKER_H_

#include "feature_tracker/feature_tracker_base.h"

#include "feature_tracker/gyro_two_frame_matcher.h"

namespace vins_core {
enum class FeatureStatus {
  kDetected,
  kLkTracked
};

// first: index_k, second: index_km1.
typedef std::pair<int, int> TrackedMatch;
typedef std::vector<FeatureStatus> FrameFeatureStatus;
typedef std::vector<int> FrameStatusTrackLength;
typedef Eigen::VectorXi TrackIds;

class GyroTracker : public FeatureTrackerBase {
public:
    explicit GyroTracker(
            const aslam::NCamera::Ptr& cameras,
            const common::SlamConfigPtr& config);

    virtual ~GyroTracker() = default;

    void TrackFeature(
            const int cam_idx,
            const Eigen::Quaterniond& q_kp1_k,
            const common::VisualFrameData& frame_data_k,
            common::VisualFrameData* frame_data_kp1_ptr,
            common::FrameToFrameMatchesWithScore* matches_ptr,
            int* track_id_provider_ptr) override;
private:
    void UpdateTrackIdDeque(
        const common::VisualFrameData& frame_data_k);

    void InitializeFeatureStatusDeque();

    void UpdateFeatureStatusDeque(
        const FrameFeatureStatus& frame_feature_status_kp1);

    void ComputeTrackedMatches(
          std::vector<TrackedMatch>* tracked_matches) const;

    void ComputeStatusTrackLengthOfFrameK(
        const std::vector<TrackedMatch>& tracked_matches,
        FrameStatusTrackLength* status_track_length_k);

    void ComputeUnmatchedIndicesOfFrameK(
        const common::FrameToFrameMatchesWithScore& matches_kp1_k,
        std::vector<int>* unmatched_indices_k) const;

    void ComputeLKCandidates(
        const common::FrameToFrameMatchesWithScore& matches_kp1_k,
        const FrameStatusTrackLength& status_track_length_k,
        const common::VisualFrameData& frame_k,
        const common::VisualFrameData& frame_kp1,
        std::vector<int>* lk_candidate_indices_k) const;

    void LkTracking(
          const int cam_idx,
          const Eigen::Matrix2Xd& predicted_keypoint_positions_kp1,
          const std::vector<unsigned char>& prediction_success,
          const std::vector<int>& lk_candidate_indices_k,
          const common::VisualFrameData& frame_k,
          common::VisualFrameData* frame_kp1,
          common::FrameToFrameMatchesWithScore* matches_kp1_k);

    //! Remember if we have initialized already.
    bool initialized_;
    //! Store track IDs of frame k and (k-1) in that order.
    std::deque<TrackIds> track_ids_k_km1_;
    //! Keep feature status for every index.
    //! For frames k and km1 in that order.
    std::deque<FrameFeatureStatus> feature_status_k_km1_;
    /// Keep status track length of frame (k-1) for every index.
    /// Status track length refers to the track length
    /// since the status of the feature has changed.
    FrameStatusTrackLength status_track_length_km1_;
};
}  // namespace vins_core
#endif  // MVINS_GYRO_TRECKER_H_
