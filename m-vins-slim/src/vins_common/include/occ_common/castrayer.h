#ifndef OCCUPANCY_GRID_RANGE_DATA_INSERTER_2D_H_
#define OCCUPANCY_GRID_RANGE_DATA_INSERTER_2D_H_

#include <utility>
#include <vector>

#include "data_common/state_structures.h"
#include "occ_common/probability_grid.h"
#include "occ_common/xy_index.h"

namespace common {

class CastRayer {
 public:
    CastRayer();

    CastRayer(const CastRayer&) = delete;
    CastRayer& operator=(const CastRayer&) = delete;

    // Inserts 'range_data' into 'probability_grid'.
    void Insert(const common::PointCloud& range_data,
               const Eigen::Vector3d& origin,
               Grid2D* grid) const;

    void SetHitMissProbability(double hit_probability, double miss_probability);

 private:
    // Probability change for a hit (this will be converted to odds
    // and therefore must be greater than 0.5).
    double hit_probability_ = 0.60;

    // Probability change for a miss (this will be converted to odds
    // and therefore must be less than 0.5).
    double miss_probability_ = 0.49;

    // NOTE: These two members will be initialized with hit_probability_
    // and miss_probability_, so keep in mind that they MUST be declared after
    // those two probability.
    const std::vector<uint16_t> hit_table_;
    const std::vector<uint16_t> miss_table_;
};
}

#endif
