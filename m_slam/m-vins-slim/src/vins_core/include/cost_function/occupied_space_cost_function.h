#ifndef COST_FUNCTION_OCCUPIED_SPACE_COST_FUNCTION_H_
#define COST_FUNCTION_OCCUPIED_SPACE_COST_FUNCTION_H_

#include <ceres/ceres.h>

#include "data_common/state_structures.h"
#include "occ_common/grid_2d.h"

namespace vins_core {

ceres::CostFunction* CreateLocalOccupiedSpace2D(
        const common::PointCloud& point_cloud,
        const common::Grid2D& grid,
        const double scan_sigma);

ceres::CostFunction* CreateGlobalOccupiedSpace2D(
        const common::PointCloud& point_cloud,
        const common::Grid2D& grid,
        const double scan_sigma);
}

#endif
