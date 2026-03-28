#ifndef MVINS_COST_FUNCTION_ODOM_PROPAGATION_COST_H_
#define MVINS_COST_FUNCTION_ODOM_PROPAGATION_COST_H_

#include <ceres/ceres.h>

#include "sensor_propagator/odom_propagator.h"

#include "data_common/constants.h"

namespace vins_core {
class OdomPropagationCost : public ceres::CostFunction {
public:
    OdomPropagationCost(const common::SlamConfigPtr& config,
                        const common::OdomDatas& propa_data);

    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;
private:
    const common::SlamConfigPtr config_;
    const common::OdomDatas propa_data_;
};

}

#endif
