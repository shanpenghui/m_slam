#ifndef MVINS_COST_FUNCTION_IMU_PROPAGATION_COST_H_
#define MVINS_COST_FUNCTION_IMU_PROPAGATION_COST_H_

#include <ceres/ceres.h>

#include "sensor_propagator/imu_propagator.h"

#include "data_common/constants.h"

namespace vins_core {
class ImuPropagationCost : public ceres::CostFunction {
public:
    ImuPropagationCost(const common::SlamConfigPtr& config,
                       const common::ImuDatas& propa_data);

    virtual bool Evaluate(double const * const *parameters, double *residuals,
                          double **jacobians) const;
private:
    const common::SlamConfigPtr config_;
    const common::ImuDatas& propa_data_;
};

}

#endif
