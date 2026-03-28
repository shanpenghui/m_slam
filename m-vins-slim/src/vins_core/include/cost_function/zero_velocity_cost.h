#ifndef MVINS_COST_FUNCTION_ZERO_VELOCITY_COST_H_
#define MVINS_COST_FUNCTION_ZERO_VELOCITY_COST_H_

#include <ceres/ceres.h>

#include "data_common/constants.h"

namespace vins_core {
class ZeroVelocityCost : public ceres::CostFunction {
public:
    ZeroVelocityCost(const Eigen::Matrix<double, 3, 3>& sqrt_info);

    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;
private:
    const Eigen::Matrix<double, 3, 3> sqrt_info_;
};
}

#endif
