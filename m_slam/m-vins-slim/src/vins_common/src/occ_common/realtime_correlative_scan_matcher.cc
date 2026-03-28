#include "occ_common/realtime_correlative_scan_matcher.h"

#include "occ_common/probability_grid.h"
#include "occ_common/voxel_filter.h"

namespace common {
float ComputeCandidateScore(const common::ProbabilityGrid* probability_grid,
                             const DiscreteScan2D& discrete_scan,
                             const bool compute_rectional,
                             int x_index_offset,
                             int y_index_offset) {
    float rectional_score = 0.f;
    int temp_counter = 0;
    float candidate_score = 0.f;
    Eigen::Array2i start = discrete_scan.back();
    for (const Eigen::Array2i& xy_index : discrete_scan) {
        const Eigen::Array2i proposed_xy_index(xy_index.x() + x_index_offset,
                                               xy_index.y() + y_index_offset);
        const float probability =
                probability_grid->GetValue(proposed_xy_index);
        candidate_score += probability;
        const Eigen::Array2i diff = xy_index - start;
        if (diff.x() == 0 || diff.y() == 0) {
                temp_counter++;
        } else {
                if (temp_counter > 15) {
                        rectional_score += temp_counter;
                }
                start = xy_index;
                temp_counter = 0;
        }
    }
    rectional_score /= static_cast<float>(discrete_scan.size());
    candidate_score /= static_cast<float>(discrete_scan.size());
    CHECK_GT(candidate_score, 0.f);
    float mix_score;
    if (compute_rectional) {
        mix_score = 0.5f * candidate_score + 0.5 * rectional_score;
    } else {
        mix_score = candidate_score;
    }
    return mix_score;
}

RealTimeCorrelativeScanMatcher::RealTimeCorrelativeScanMatcher(
        const common::Grid2D* grid)
        : grid_(grid) {}

std::vector<Candidate2D>
RealTimeCorrelativeScanMatcher::GenerateExhaustiveSearchCandidates(
        const SearchParameters& search_parameters) const {
    int num_candidates = 0;
    for (int scan_index = 0; scan_index != search_parameters.num_scans;
        ++scan_index) {
        const int num_linear_x_candidates =
                (search_parameters.linear_bounds[scan_index].max_x -
                 search_parameters.linear_bounds[scan_index].min_x + 1);
        const int num_linear_y_candidates =
                (search_parameters.linear_bounds[scan_index].max_y -
                 search_parameters.linear_bounds[scan_index].min_y + 1);
        num_candidates += num_linear_x_candidates * num_linear_y_candidates;
    }
    std::vector<Candidate2D> candidates;
    candidates.reserve(num_candidates);
    for (int scan_index = 0; scan_index != search_parameters.num_scans;
        ++scan_index) {
        for (int x_index_offset =
                search_parameters.linear_bounds[scan_index].min_x;
            x_index_offset <= search_parameters.linear_bounds[scan_index].max_x;
            ++x_index_offset) {
            for (int y_index_offset =
                    search_parameters.linear_bounds[scan_index].min_y;
                y_index_offset <=
                    search_parameters.linear_bounds[scan_index].max_y;
                ++y_index_offset) {

                candidates.emplace_back(
                        scan_index, x_index_offset,
                        y_index_offset, search_parameters);
            }
        }
    }
    CHECK_EQ(candidates.size(), static_cast<size_t>(num_candidates));
    return candidates;
}


MatchingResult RealTimeCorrelativeScanMatcher::Match(
        const common::KeyFrame& scan,
        const MatchingOption& options) const {

    const Eigen::Vector3d& initial_translation = scan.state.T_OtoG.getPosition();
    const Eigen::Quaterniond& initial_rotation = scan.state.T_OtoG.getEigenQuaternion();

    common::AdaptiveVoxelFilterOptions filter_options;
    common::AdaptiveVoxelFilter voxel_filter(filter_options);
    common::PointCloud points_filted = voxel_filter.Filter(scan.points);

    common::KeyFrame rotated_range_data;
    for (const auto& pt : points_filted.points) {
        rotated_range_data.points.points.push_back(initial_rotation * pt);
    }
    const SearchParameters search_parameters(
            options.linear_search_window_,
            options.angular_search_window_,
            rotated_range_data,
            grid_->limits().resolution());

    const std::vector<std::shared_ptr<common::KeyFrame>> rotated_scans =
            GenerateRotatedScans(rotated_range_data.points.points,
                                    search_parameters);
    const std::vector<DiscreteScan2D> discrete_scans = DiscretizeScans(
            grid_->limits(), rotated_scans, initial_translation.head<2>());

    std::vector<Candidate2D> candidates =
        GenerateExhaustiveSearchCandidates(search_parameters);
    ScoreCandidates(discrete_scans, options, &candidates);

    const Candidate2D& best_candidate =
            *std::max_element(candidates.begin(), candidates.end());

    const double best_yaw = best_candidate.orientation;
    Eigen::Quaterniond q_estimate =
            common::EulerToQuat(Eigen::Vector3d(0.0, 0.0, best_yaw)) * initial_rotation;
    q_estimate.normalize();
    const Eigen::Vector3d p_estimate =
            initial_translation + Eigen::Vector3d(best_candidate.x, best_candidate.y, 0);

    aslam::Transformation pose_estimate(p_estimate, q_estimate);
    return MatchingResult(pose_estimate, best_candidate.score,
                          best_candidate.valid_rate);
}

void RealTimeCorrelativeScanMatcher::ScoreCandidates(
        const std::vector<DiscreteScan2D>& discrete_scans,
        const MatchingOption& options,
        std::vector<Candidate2D>* const candidates) const {
    for (Candidate2D& candidate : *candidates) {
        candidate.score = ComputeCandidateScore(
                CHECK_NOTNULL(dynamic_cast<const common::ProbabilityGrid*>(grid_)),
                discrete_scans[candidate.scan_index],
                options.compute_scan_rectional,
                candidate.x_index_offset,
                candidate.y_index_offset);
        candidate.score *= std::exp(-common::Pow2(
                std::hypot(candidate.x, candidate.y) * options.translation_delta_cost_weight_ +
                std::abs(candidate.orientation) * options.rotation_delta_cost_weight_));
    }
}

}
