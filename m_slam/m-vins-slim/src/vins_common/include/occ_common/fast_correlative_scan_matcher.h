#ifndef OCCUPANCY_GRID_FAST_CSM_H_
#define OCCUPANCY_GRID_FAST_CSM_H_

#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>

#include "data_common/state_structures.h"
#include "occ_common/submap_2d.h"

namespace common {
typedef std::vector<Eigen::Array2i> DiscreteScan2D;
// Describes the search space.
struct SearchParameters {
    // Linear search window in pixel offsets; bounds are inclusive.
    struct LinearBounds {
        int min_x;
        int max_x;
        int min_y;
        int max_y;
    };

    SearchParameters(double linear_search_window,
                     double angular_search_window,
                     const common::KeyFrame& range_data,
                     double resolution);

    SearchParameters(double linear_x_search_window,
                     double linear_y_search_window,
                     double angular_search_window,
                     double angular_step,
                     double resolution);

    // It's better to identify init() as private.
    void init(double linear_x_search_window,
              double linear_y_search_window,
              double angular_search_window);

    // Tightens the search window as much as possible.
    void ShrinkToFit(const std::vector<DiscreteScan2D> &scans,
                     const common::CellLimits &cell_limits);

    int num_angular_perturbations;
    double angular_perturbation_step_size;
    double resolution;
    int num_scans;
    std::vector<LinearBounds> linear_bounds;  // Per rotated scans.
};

// Translates and discretizes the rotated scans into a vector of integer
// indices.
std::vector<DiscreteScan2D> DiscretizeScans(
        const common::MapLimits& map_limits,
        const std::vector<std::shared_ptr<common::KeyFrame>>& scans,
        const Eigen::Vector2d& initial_xy);

// A collection of values which can be added and later removed, and the maximum
// of the current values in the collection can be retrieved.
// All of it in (amortized) O(1).
class SlidingWindowMaximum {
 public:
    void AddValue(const float value) {
        while (!non_ascending_maxima_.empty() &&
               value > non_ascending_maxima_.back()) {
            non_ascending_maxima_.pop_back();
        }
        non_ascending_maxima_.push_back(value);
    }

    void RemoveValue(const float value);

    float GetMaximum() const;

    inline void CheckIsEmpty() const;

 private:
    // Maximum of the current sliding window at the front. Then the maximum of
    // the remaining window that came after this values first occurrence,
    // and so on.
    std::deque<float> non_ascending_maxima_;
};

struct MatchingOption {
    bool compute_scan_rectional = false;
    // Minimum linear search window in which the best possible scan alignment
    // will be found.
    // Should be integral multiple of grid resolution.
    double linear_search_window_ = 1.0;

    // Minimum angular search window in which the best possible scan alignment
    // will be found.
    double angular_search_window_ = 30 * M_PI / 180;

    // Minimum score to filter matching result.
    float min_score_ = 0.65;

    // Weights applied to each part of the score.
    double translation_delta_cost_weight_ = 1e-1;
    double rotation_delta_cost_weight_ = 1e-1;
};

// Generates a collection of rotated scans.
std::vector<std::shared_ptr<common::KeyFrame>> GenerateRotatedScans(
        const common::EigenVector3dVec& range_data,
        const SearchParameters& search_parameters);

std::vector<std::shared_ptr<common::KeyFrame>> GenerateRotatedScans(
        const common::EigenVector3dVec& range_data,
        const MatchingOption option,
        const double angle_search_resolution);

// A precomputed grid that contains in each cell (x0, y0) the maximum
// probability in the width x width area defined by x0 <= x < x0 + width and
// y0 <= y < y0.
class PrecomputationGrid2D {
 public:
    PrecomputationGrid2D(const common::Grid2D& grid, const common::CellLimits& limits,
            int width, std::vector<float>* reusable_intermediate_grid);

    // Returns a value between 0 and 255 to represent probabilities between
    // min_score and max_score.
    int GetValue(const Eigen::Array2i& xy_index) const {
        const Eigen::Array2i local_xy_index = xy_index - offset_;
        // The static_cast<unsigned> is for performance to check with 2
        // comparisons xy_index.x() < offset_.x() ||
        // xy_index.y() < offset_.y() ||
        // local_xy_index.x() >= wide_limits_.num_x_cells ||
        // local_xy_index.y() >= wide_limits_.num_y_cells
        // instead of using 4 comparisons.
        if (static_cast<unsigned>(local_xy_index.x()) >=
            static_cast<unsigned>(wide_limits_.num_x_cells) ||
            static_cast<unsigned>(local_xy_index.y()) >=
            static_cast<unsigned>(wide_limits_.num_y_cells)) {
            return 0;
        }
        const int stride = wide_limits_.num_x_cells;
        return cells_[local_xy_index.x() + local_xy_index.y() * stride];
    }

    // Maps values from [0, 255] to [min_score, max_score].
    float ToScore(float value) const {
        return min_score_ + value * ((max_score_ - min_score_) / 255.f);
    }

 private:
    uint8 ComputeCellValue(float probability) const;

    // Offset of the precomputation grid in relation to the 'grid'
    // including the additional 'width' - 1 cells.
    const Eigen::Array2i offset_;

    // Size of the precomputation grid.
    const common::CellLimits wide_limits_;

    const float min_score_;
    const float max_score_;

    // Probabilites mapped to 0 to 255.
    std::vector<uint8> cells_;
};

class PrecomputationGridStack2D {
 public:
    PrecomputationGridStack2D(
            const common::Grid2D& grid,
            int branch_and_bound_depth);

    const PrecomputationGrid2D& Get(int index) {
        return precomputation_grids_[index];
    }

    int max_depth() const {
        return static_cast<int>(precomputation_grids_.size()) - 1;
    }

 private:
    std::vector<PrecomputationGrid2D> precomputation_grids_;
};

struct MatchingResult {
    MatchingResult() = default;
    explicit MatchingResult(float score, float valid_rate)
    : score(score),
      valid_rate(valid_rate) {}
    MatchingResult(const aslam::Transformation& pose,
                   float score,
                   float valid_rate)
    : pose_estimate(pose),
      score(score),
      valid_rate(valid_rate) {}
    aslam::Transformation pose_estimate;
    float score;
    float valid_rate;
};

// A possible solution.
struct Candidate2D {
    Candidate2D(const int init_scan_index, const int init_x_index_offset,
                const int init_y_index_offset,
                const SearchParameters& search_parameters)
            : scan_index(init_scan_index),
              x_index_offset(init_x_index_offset),
              y_index_offset(init_y_index_offset),
              x(-y_index_offset * search_parameters.resolution),
              y(-x_index_offset * search_parameters.resolution),
              orientation((scan_index -
              search_parameters.num_angular_perturbations) *
                          search_parameters.angular_perturbation_step_size) {}

    // Index into the rotated scans vector.
    int scan_index = 0;

    // Linear offset from the initial pose.
    int x_index_offset = 0;
    int y_index_offset = 0;

    // Pose of this Candidate2D relative to the initial pose.
    double x = 0.;
    double y = 0.;
    double orientation = 0.;

    // Score, higher is better.
    float score = 0.f;
    float valid_rate = 0.f;

    bool operator<(const Candidate2D& other) const {
        return score < other.score;
    }
    bool operator>(const Candidate2D& other) const {
        return score > other.score;
    }
};

std::vector<Candidate2D> GenerateLowestResolutionCandidates(
        const int linear_step_size,
        const SearchParameters& search_parameters);

void ShowScanMatching(const std::shared_ptr<common::Submap2D>& submap,
                      const aslam::Transformation& T_GtoM,
                      const common::KeyFrame& key_frame);

class FastCorrelativeScanMatcher {
 public:
    FastCorrelativeScanMatcher(const common::Grid2D& grid,
                               int branch_and_bound_depth);

    MatchingResult Match(const common::KeyFrame& scan,
                          const common::MatchingOption& option) const;

 private:
    std::vector<Candidate2D> ComputeLowestResolutionCandidates(
            const std::vector<DiscreteScan2D>& discrete_scans,
            const SearchParameters& search_parameters) const;

    void ScoreCandidates(const PrecomputationGrid2D& precomputation_grid,
                         const std::vector<DiscreteScan2D>& discrete_scans,
                         std::vector<Candidate2D>* const candidates) const;

    Candidate2D BranchAndBound(
            const std::vector<DiscreteScan2D>& discrete_scans,
            const SearchParameters& search_parameters,
            const std::vector<Candidate2D>& candidates,
            int candidate_depth, float min_score) const;

    const common::MapLimits limits_;

    std::unique_ptr<PrecomputationGridStack2D> precomputation_grid_stack_;

};
}

#endif
