#include "occ_common/fast_correlative_scan_matcher.h"

#include "occ_common/voxel_filter.h"

namespace common {

SearchParameters::SearchParameters(double linear_search_window,
                                       double angular_search_window,
                                       const common::KeyFrame& range_data,
                                       double resolution)
        : resolution(resolution) {
    double max_scan_range = 3.f * resolution;
    for (const Eigen::Vector3d& point : range_data.points.points) {
        const double range = point.norm();
        max_scan_range = std::max(range, max_scan_range);
    }
    constexpr double kSafetyMargin = 1. - 1e-3;
#if 0
    // We set this value to something on the order of resolution to make sure
    // that the std::acos() below is defined.
    angular_perturbation_step_size =
            kSafetyMargin * std::acos(1. - common::Pow2(resolution) /
            (2. * common::Pow2(max_scan_range)));
#else
    angular_perturbation_step_size =
            kSafetyMargin * std::acos(1. - common::Pow2(resolution) / max_scan_range);
#endif
    angular_perturbation_step_size = std::max(0.005, angular_perturbation_step_size);
    init(linear_search_window, linear_search_window, angular_search_window);
}

SearchParameters::SearchParameters(
        double linear_x_search_window,
        double linear_y_search_window,
        double angular_search_window,
        double angular_step,
        double resolution)
    : angular_perturbation_step_size(angular_step),
      resolution(resolution) {

    init(linear_x_search_window, linear_y_search_window, angular_search_window);
}

void SearchParameters::init(
        double linear_x_search_window,
        double linear_y_search_window,
        double angular_search_window) {

    num_angular_perturbations =
            std::ceil(angular_search_window / angular_perturbation_step_size);
    num_scans = 2 * num_angular_perturbations + 1;

    const int num_linear_x_perturbations =
            std::ceil(linear_x_search_window / resolution);
    const int num_linear_y_perturbations =
            std::ceil(linear_y_search_window / resolution);
    linear_bounds.reserve(num_scans);
    for (int i = 0; i != num_scans; ++i) {
        linear_bounds.push_back(
                LinearBounds{-num_linear_x_perturbations,
                             num_linear_x_perturbations,
                             -num_linear_y_perturbations,
                             num_linear_y_perturbations});
    }
}

void SearchParameters::ShrinkToFit(const std::vector<DiscreteScan2D>& scans,
                                   const common::CellLimits& cell_limits) {
    CHECK_EQ(scans.size(), static_cast<size_t>(num_scans));
    CHECK_EQ(linear_bounds.size(), static_cast<size_t>(num_scans));
    for (int i = 0; i != num_scans; ++i) {
        Eigen::Array2i min_bound = Eigen::Array2i::Zero();
        Eigen::Array2i max_bound = Eigen::Array2i::Zero();
        for (const Eigen::Array2i& xy_index : scans[i]) {
            min_bound = min_bound.min(-xy_index);
            max_bound = max_bound.max(
                    Eigen::Array2i(cell_limits.num_x_cells - 1,
                            cell_limits.num_y_cells - 1) - xy_index);
        }
        linear_bounds[i].min_x =
                std::max(linear_bounds[i].min_x, min_bound.x());
        linear_bounds[i].min_y =
                std::max(linear_bounds[i].min_y, min_bound.y());

        linear_bounds[i].max_x =
                std::min(linear_bounds[i].max_x, max_bound.x());
        linear_bounds[i].max_y =
                std::min(linear_bounds[i].max_y, max_bound.y());
    }
}

std::vector<std::shared_ptr<common::KeyFrame>> GenerateRotatedScans(
        const common::EigenVector3dVec& range_data,
        const SearchParameters& search_parameters) {
    std::vector<std::shared_ptr<common::KeyFrame>> rotated_scans;
    rotated_scans.reserve(search_parameters.num_scans);

    double delta_theta = -search_parameters.num_angular_perturbations *
                           search_parameters.angular_perturbation_step_size;
    for (int scan_index = 0; scan_index < search_parameters.num_scans;
         ++scan_index, delta_theta +=
            search_parameters.angular_perturbation_step_size) {

        const Eigen::Matrix3d delta_yaw =
                Eigen::AngleAxisd(delta_theta, Eigen::Vector3d::UnitZ()).
                toRotationMatrix();
        const aslam::Transformation delta_pose(
                Eigen::Vector3d::Zero(), Eigen::Quaterniond(delta_yaw));
        std::shared_ptr<common::KeyFrame> rotated_scan =
                std::make_shared<common::KeyFrame>();
        for (const auto& pt : range_data) {
            rotated_scan->points.points.push_back(delta_pose.transform(pt));
        }

        rotated_scans.push_back(rotated_scan);
    }
    return rotated_scans;
}

std::vector<std::shared_ptr<common::KeyFrame>> GenerateRotatedScans(
        const common::EigenVector3dVec& range_data,
        const MatchingOption option,
        const double angle_search_resolution) {
    std::vector<std::shared_ptr<common::KeyFrame>> rotated_scans;

    for (double delta_theta = 0.0; delta_theta < 2 * M_PI;
         delta_theta += angle_search_resolution) {
        const Eigen::Matrix3d delta_yaw =
                Eigen::AngleAxisd(delta_theta, Eigen::Vector3d::UnitZ()).
                toRotationMatrix();
        const aslam::Transformation delta_pose(
                Eigen::Vector3d::Zero(), Eigen::Quaterniond(delta_yaw));
        std::shared_ptr<common::KeyFrame> rotated_scan =
                std::make_shared<common::KeyFrame>();
        for (const auto& pt : range_data) {
            rotated_scan->points.points.push_back(delta_pose.transform(pt));
        }

        rotated_scans.push_back(rotated_scan);
    }
    return rotated_scans;
}

std::vector<Candidate2D> GenerateLowestResolutionCandidates(
        const int linear_step_size,
        const SearchParameters& search_parameters)  {
    int num_candidates = 0;
    for (int scan_index = 0; scan_index != search_parameters.num_scans;
         ++scan_index) {
        const int num_lowest_resolution_linear_x_candidates =
                (search_parameters.linear_bounds[scan_index].max_x -
                 search_parameters.linear_bounds[scan_index].min_x +
                 linear_step_size) / linear_step_size;
        const int num_lowest_resolution_linear_y_candidates =
                (search_parameters.linear_bounds[scan_index].max_y -
                 search_parameters.linear_bounds[scan_index].min_y +
                 linear_step_size) / linear_step_size;
        num_candidates += num_lowest_resolution_linear_x_candidates *
                          num_lowest_resolution_linear_y_candidates;
    }

    std::vector<Candidate2D> candidates;
    candidates.reserve(num_candidates);
    for (int scan_index = 0; scan_index != search_parameters.num_scans;
         ++scan_index) {
        for (int x_index_offset =
                search_parameters.linear_bounds[scan_index].min_x;
             x_index_offset <=
                search_parameters.linear_bounds[scan_index].max_x;
             x_index_offset += linear_step_size) {
            for (int y_index_offset =
                    search_parameters.linear_bounds[scan_index].min_y;
                 y_index_offset <=
                    search_parameters.linear_bounds[scan_index].max_y;
                 y_index_offset += linear_step_size) {
                candidates.emplace_back(scan_index, x_index_offset,
                        y_index_offset, search_parameters);
            }
        }
    }

    CHECK_EQ(candidates.size(), static_cast<size_t>(num_candidates));
    return candidates;
}


std::vector<DiscreteScan2D> DiscretizeScans(
        const common::MapLimits& map_limits,
        const std::vector<std::shared_ptr<common::KeyFrame>>& scans,
        const Eigen::Vector2d& initial_xy) {
    std::vector<DiscreteScan2D> discrete_scans;
    discrete_scans.reserve(scans.size());
    for (const auto& scan : scans) {
        discrete_scans.emplace_back();
        discrete_scans.back().reserve(scan->points.points.size());
        for (const auto& point : scan->points.points) {
            const Eigen::Vector2d translated_point =
                    initial_xy + point.head<2>();
            discrete_scans.back().push_back(
                    map_limits.GetCellIndex(translated_point));
        }
    }
    return discrete_scans;
}

void SlidingWindowMaximum::RemoveValue(const float value) {
    // DCHECK for performance, since this is done for every value in the
    // precomputation grid.
    CHECK(!non_ascending_maxima_.empty());
    CHECK_LE(value, non_ascending_maxima_.front());
    if (value == non_ascending_maxima_.front()) {
        non_ascending_maxima_.pop_front();
    }
}

float SlidingWindowMaximum::GetMaximum() const {
    // DCHECK for performance, since this is done for every value in the
    // precomputation grid.
    CHECK(!non_ascending_maxima_.empty());
    return non_ascending_maxima_.front();
}

inline void SlidingWindowMaximum::CheckIsEmpty() const {
    CHECK(non_ascending_maxima_.empty());
}

PrecomputationGrid2D::PrecomputationGrid2D(
        const common::Grid2D& grid,
        const common::CellLimits& limits,
        const int width,
        std::vector<float>* reusable_intermediate_grid)
        : offset_(-width + 1, -width + 1),
          wide_limits_(limits.num_x_cells + width - 1,
                       limits.num_y_cells + width - 1),
          min_score_(common::CorrespondenceCostToProbability(
                  grid.GetMaxCorrespondenceCost())),
          max_score_(common::CorrespondenceCostToProbability(
                  grid.GetMinCorrespondenceCost())),
          cells_(wide_limits_.num_x_cells * wide_limits_.num_y_cells) {
    CHECK_GE(width, 1);
    CHECK_GE(limits.num_x_cells, 1);
    CHECK_GE(limits.num_y_cells, 1);
    const int stride = wide_limits_.num_x_cells;

    // First we compute the maximum probability for each (x0, y) achieved in the
    // span defined by x0 <= x < x0 + width.
    std::vector<float>& intermediate = *reusable_intermediate_grid;
    intermediate.resize(wide_limits_.num_x_cells * limits.num_y_cells);
    for (int y = 0; y != limits.num_y_cells; ++y) {
        SlidingWindowMaximum current_values;
        current_values.AddValue(common::CorrespondenceCostToProbability(
                grid.GetCorrespondenceCost(Eigen::Array2i(0, y))));
        for (int x = -width + 1; x != 0; ++x) {
            intermediate[x + width - 1 + y * stride] =
                    current_values.GetMaximum();
            if (x + width < limits.num_x_cells) {current_values.AddValue(
                    common::CorrespondenceCostToProbability(grid.GetCorrespondenceCost(
                    Eigen::Array2i(x + width, y))));
            }
        }
        for (int x = 0; x < limits.num_x_cells - width; ++x) {
            intermediate[x + width - 1 + y * stride] =
                    current_values.GetMaximum();
            current_values.RemoveValue(common::CorrespondenceCostToProbability(
                    grid.GetCorrespondenceCost(Eigen::Array2i(x, y))));
            current_values.AddValue(common::CorrespondenceCostToProbability(
                    grid.GetCorrespondenceCost(Eigen::Array2i(x + width, y))));
        }
        for (int x = std::max(limits.num_x_cells - width, 0);
             x != limits.num_x_cells; ++x) {
            intermediate[x + width - 1 + y * stride] =
                    current_values.GetMaximum();
            current_values.RemoveValue(common::CorrespondenceCostToProbability(
                    grid.GetCorrespondenceCost(Eigen::Array2i(x, y))));
        }
        current_values.CheckIsEmpty();
    }

    // For each (x, y), we compute the maximum probability in the width x width
    // region starting at each (x, y) and precompute the resulting bound on the
    // score.
    for (int x = 0; x != wide_limits_.num_x_cells; ++x) {
        SlidingWindowMaximum current_values;
        current_values.AddValue(intermediate[x]);
        for (int y = -width + 1; y != 0; ++y) {
            cells_[x + (y + width - 1) * stride] =
                    ComputeCellValue(current_values.GetMaximum());
            if (y + width < limits.num_y_cells) {
                current_values.AddValue(intermediate[x + (y + width) * stride]);
            }
        }
        for (int y = 0; y < limits.num_y_cells - width; ++y) {
            cells_[x + (y + width - 1) * stride] =
                    ComputeCellValue(current_values.GetMaximum());
            current_values.RemoveValue(intermediate[x + y * stride]);
            current_values.AddValue(intermediate[x + (y + width) * stride]);
        }
        for (int y = std::max(limits.num_y_cells - width, 0);
             y != limits.num_y_cells; ++y) {
            cells_[x + (y + width - 1) * stride] =
                    ComputeCellValue(current_values.GetMaximum());
            current_values.RemoveValue(intermediate[x + y * stride]);
        }
        current_values.CheckIsEmpty();
    }
}

uint8 PrecomputationGrid2D::ComputeCellValue(const float probability) const {
    const int cell_value = common::RoundToInt(
            (probability - min_score_) * (255.f / (max_score_ - min_score_)));
    CHECK_GE(cell_value, 0);
    CHECK_LE(cell_value, 255);
    return cell_value;
}

PrecomputationGridStack2D::PrecomputationGridStack2D(
            const common::Grid2D& grid, int branch_and_bound_depth) {
    CHECK_GE(branch_and_bound_depth, 1);
    const int max_width = 1 << (branch_and_bound_depth - 1);

    std::vector<float> reusable_intermediate_grid;
    const common::CellLimits limits = grid.limits().cell_limits();
    reusable_intermediate_grid.reserve((limits.num_x_cells + max_width - 1) *
                                       limits.num_y_cells);

    precomputation_grids_.reserve(branch_and_bound_depth);
    for (int i = 0; i != branch_and_bound_depth; ++i) {
        const int width = 1 << i;
        precomputation_grids_.emplace_back(grid, limits, width,
                                           &reusable_intermediate_grid);
    }
}

FastCorrelativeScanMatcher::FastCorrelativeScanMatcher(
        const common::Grid2D& grid,
        int branch_and_bound_depth) :
        limits_(grid.limits()),
        precomputation_grid_stack_(
                std::make_unique<PrecomputationGridStack2D>(
                        grid, branch_and_bound_depth)) {}

MatchingResult FastCorrelativeScanMatcher::Match(
        const common::KeyFrame& scan, const common::MatchingOption& option) const {

    common::AdaptiveVoxelFilterOptions filter_options;
    common::AdaptiveVoxelFilter voxel_filter(filter_options);
    common::PointCloud points_filted = voxel_filter.Filter(scan.points);

    SearchParameters search_parameters(
            option.linear_search_window_,
            option.angular_search_window_,
            scan,
            limits_.resolution());

    const Eigen::Vector3d initial_translation = scan.state.T_OtoG.getPosition();
    const aslam::Transformation initial_rotation(Eigen::Vector3d::Zero(),
            scan.state.T_OtoG.getEigenQuaternion());
    common::KeyFrame rotated_range_data;
    for (const auto& pt : points_filted.points) {
        rotated_range_data.points.points.push_back(initial_rotation.transform(pt));
    }

    const std::vector<std::shared_ptr<common::KeyFrame>> rotated_scans =
            GenerateRotatedScans(rotated_range_data.points.points,
                                 search_parameters);
    const std::vector<DiscreteScan2D> discrete_scans = DiscretizeScans(
            limits_, rotated_scans, initial_translation.head<2>());
    search_parameters.ShrinkToFit(discrete_scans, limits_.cell_limits());

    const std::vector<Candidate2D> lowest_resolution_candidates =
            ComputeLowestResolutionCandidates(
                    discrete_scans, search_parameters);
    const Candidate2D best_candidate = BranchAndBound(
            discrete_scans, search_parameters, lowest_resolution_candidates,
            precomputation_grid_stack_->max_depth(), option.min_score_);

    aslam::Transformation pose_estimate;
    if (best_candidate.score > option.min_score_) {
        const double best_yaw = best_candidate.orientation;
        Eigen::Quaterniond delta_q = common::EulerToQuat(Eigen::Vector3d(0., 0., best_yaw));
        Eigen::Quaterniond q_estimate = delta_q * initial_rotation.getEigenQuaternion();
        Eigen::Vector3d p_estimate = initial_translation +
                Eigen::Vector3d(best_candidate.x, best_candidate.y, 0);
        pose_estimate.update(q_estimate, p_estimate);
        return MatchingResult(pose_estimate, best_candidate.score,
                              best_candidate.valid_rate);
    }
    return MatchingResult(pose_estimate, -1., 0.);
}

std::vector<Candidate2D>
FastCorrelativeScanMatcher::ComputeLowestResolutionCandidates(
        const std::vector<DiscreteScan2D>& discrete_scans,
        const SearchParameters& search_parameters) const {
    const int linear_step_size = 1 << precomputation_grid_stack_->max_depth();
    std::vector<Candidate2D> lowest_resolution_candidates =
            GenerateLowestResolutionCandidates(
                linear_step_size,
                search_parameters);

    ScoreCandidates(precomputation_grid_stack_->Get(
                    precomputation_grid_stack_->max_depth()),
                    discrete_scans, &lowest_resolution_candidates);
    return lowest_resolution_candidates;
}

void FastCorrelativeScanMatcher::ScoreCandidates(
        const PrecomputationGrid2D& precomputation_grid,
        const std::vector<DiscreteScan2D>& discrete_scans,
        std::vector<Candidate2D>* const candidates) const {
    for (Candidate2D& candidate : *candidates) {
        int sum = 0;
        for (const Eigen::Array2i& xy_index :
                discrete_scans[candidate.scan_index]) {
            const Eigen::Array2i proposed_xy_index(
                    xy_index.x() + candidate.x_index_offset,
                    xy_index.y() + candidate.y_index_offset);
            sum += precomputation_grid.GetValue(proposed_xy_index);
        }
        candidate.score = precomputation_grid.ToScore(
                sum / static_cast<float>(
                        discrete_scans[candidate.scan_index].size()));
    }
    std::sort(candidates->begin(), candidates->end(),
              std::greater<Candidate2D>());
}

Candidate2D FastCorrelativeScanMatcher::BranchAndBound(
        const std::vector<DiscreteScan2D>& discrete_scans,
        const SearchParameters& search_parameters,
        const std::vector<Candidate2D>& candidates,
        const int candidate_depth,
        float min_score) const {
    if (candidate_depth == 0) {
        // Return the best candidate.
        return *candidates.begin();
    }

    Candidate2D best_high_resolution_candidate(0, 0, 0, search_parameters);
    best_high_resolution_candidate.score = min_score;
    for (const Candidate2D& candidate : candidates) {
        if (candidate.score <= min_score) {
            break;
        }
        std::vector<Candidate2D> higher_resolution_candidates;
        const int half_width = 1 << (candidate_depth - 1);
        for (int x_offset : {0, half_width}) {
            if (candidate.x_index_offset + x_offset >
                search_parameters.linear_bounds[candidate.scan_index].max_x) {
                break;
            }
            for (int y_offset : {0, half_width}) {
                if (candidate.y_index_offset + y_offset >
                    search_parameters.linear_bounds[
                            candidate.scan_index].max_y) {
                    break;
                }
                higher_resolution_candidates.emplace_back(
                        candidate.scan_index,
                        candidate.x_index_offset + x_offset,
                        candidate.y_index_offset + y_offset,
                        search_parameters);
            }
        }
        ScoreCandidates(precomputation_grid_stack_->Get(candidate_depth - 1),
                        discrete_scans,
                        &higher_resolution_candidates);
        best_high_resolution_candidate = std::max(
                best_high_resolution_candidate,
                BranchAndBound(discrete_scans,
                                 search_parameters,
                                 higher_resolution_candidates, candidate_depth - 1,
                                 best_high_resolution_candidate.score));
    }
    return best_high_resolution_candidate;
}

void ShowScanMatching(const std::shared_ptr<common::Submap2D>& submap,
                      const aslam::Transformation& T_GtoM,
                      const common::KeyFrame& key_frame) {
    const double kMultiple = 10.;
    common::EigenVector3dVec pc_loop =
            submap->ExtractOccupidPixeToPointCloud();
    common::EigenVector3dVec pc_query;
    pc_query.resize(key_frame.points.points.size());
    for (size_t i = 0u; i < pc_query.size(); ++i) {
        pc_query[i] = T_GtoM.transform(key_frame.state.T_OtoG.transform(
                        key_frame.points.points[i]));
    }
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double max_y = std::numeric_limits<double>::min();
    std::function<void(const common::EigenVector3dVec&,
                      double*, double*, double*, double*)> boundary_helper =
            [&](const common::EigenVector3dVec& pc,
               double* min_x_ptr,
               double* min_y_ptr,
               double* max_x_ptr,
               double* max_y_ptr) {
        double& min_x = *CHECK_NOTNULL(min_x_ptr);
        double& min_y = *CHECK_NOTNULL(min_y_ptr);
        double& max_x = *CHECK_NOTNULL(max_x_ptr);
        double& max_y = *CHECK_NOTNULL(max_y_ptr);
        for (const auto& pt : pc) {
            const double x = pt(0) * kMultiple;
            const double y = pt(1) * kMultiple;
            if (x < min_x) {
                min_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y > max_y) {
                max_y = y;
            }
        }
    };
    boundary_helper(pc_loop, &min_x, &min_y, &max_x, &max_y);
    boundary_helper(pc_query, &min_x, &min_y, &max_x, &max_y);
    const int cols = std::ceil(max_x - min_x);
    const int rows = std::ceil(max_y - min_y);
    cv::Mat img_show(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Scalar color_loop(0, 0, 255);
    const cv::Scalar color_query(255, 0, 0);
    for (const auto& pt : pc_loop) {
        cv::Point pt_show;
        pt_show.x = std::ceil(pt(0) * kMultiple - min_x);
        pt_show.y = std::ceil(pt(1) * kMultiple - min_y);
        cv::circle(img_show, pt_show, 2, color_loop, cv::FILLED);
    }
    for (const auto& pt : pc_query) {
        cv::Point pt_show;
        pt_show.x = std::ceil(pt(0) * kMultiple - min_x);
        pt_show.y = std::ceil(pt(1) * kMultiple - min_y);
        cv::circle(img_show, pt_show, 2, color_query, cv::FILLED);
    }

    cv::imshow("matching_result", img_show);
    cv::waitKey(1);
}

}

