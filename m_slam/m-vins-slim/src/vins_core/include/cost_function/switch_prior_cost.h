#ifndef COST_FUNCTION_SWITCH_PRIOR_COST_FUNCTION_H_
#define COST_FUNCTION_SWITCH_PRIOR_COST_FUNCTION_H_

#include <ceres/ceres.h>

namespace vins_core {

ceres::CostFunction* CreateSwitchPriorCost(
        const double prior,
        const double sqrt_info);

}

#endif