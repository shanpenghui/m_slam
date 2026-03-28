#ifndef OCCUPANCY_GRID_REALTIME_CSM_H_
#define OCCUPANCY_GRID_REALTIME_CSM_H_

#include "occ_common/fast_correlative_scan_matcher.h"


namespace common {
class RealTimeCorrelativeScanMatcher {
 public:
    explicit RealTimeCorrelativeScanMatcher(const common::Grid2D* grid);

    // Aligns 'point_cloud' within the 'grid' given an 'initial_pose_estimate'
    // then updates 'pose_estimate' with the result and returns the score.
    MatchingResult Match(const common::KeyFrame& scan,
                         const MatchingOption& options) const;

    // Computes the score for each Candidate2D in a collection.
    // The cost is computed as the sum of probabilities or normalized TSD.
    void ScoreCandidates(const std::vector<DiscreteScan2D>& discrete_scans,
                         const MatchingOption& options,
                         std::vector<Candidate2D>* candidates) const;

 private:
    std::vector<Candidate2D> GenerateExhaustiveSearchCandidates(
            const SearchParameters& search_parameters) const;

    const common::Grid2D* grid_;
};
}

#endif
