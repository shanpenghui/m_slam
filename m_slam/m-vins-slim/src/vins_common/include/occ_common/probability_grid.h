#ifndef OCCUPANCY_GRID_PROBABILITY_GRID_H_
#define OCCUPANCY_GRID_PROBABILITY_GRID_H_

#include <vector>

#include "occ_common/grid_2d.h"

namespace common {

// Represents a 2D grid of probabilities.
class ProbabilityGrid : public Grid2D {
 public:
    explicit ProbabilityGrid(
            int id,
            const MapLimits& limits,
            ValueConversionTables* conversion_tables);

    // Sets the probability of the cell at 'cell_index' to the given
    // 'probability'. Only allowed if the cell was unknown before.
    void SetProbability(const Eigen::Array2i& cell_index,
                       const float probability);

    // Applies the 'odds' specified when calling ComputeLookupTableToApplyOdds()
    // to the probability of the cell at 'cell_index' if the cell has not
    // already been updated. Multiple updates of the same cell will be ignored
    // until FinishUpdate() is called. Returns true if the cell was updated.
    //
    // If this is the first call to ApplyOdds() for the specified cell,
    // its value will be set to probability corresponding to 'odds'.
    bool ApplyLookupTable(const Eigen::Array2i& cell_index,
                          const std::vector<uint16>& table);

    GridType GetGridType() const override;

    // Returns the probability of the cell with 'cell_index'.
    float GetValue(const Eigen::Array2i& cell_index) const override;

    std::unique_ptr<Grid2D> ComputeCroppedGrid(
            cv::Mat* submap_odds_ptr) const override;
 private:
    ValueConversionTables* conversion_tables_;
};

}

#endif
