#ifndef MVINS_ODOM_PROPAGATOR_H_
#define MVINS_ODOM_PROPAGATOR_H_

#include <deque>
#include <memory>

#include "sensor_propagator/sensor_propagator.h"

namespace vins_core {

class OdomPropagator : public SensorPropagator {
public:
    void Propagate(const common::OdomDatas& prop_data,
                   common::State* state_ptr,
                   Eigen::Matrix<double, 9, 9>* Phi_summed_ptr = nullptr,
                   Eigen::Matrix<double, 9, 9>* Qd_summed_ptr = nullptr);
protected:
    void PredictAndCompute(const common::OdomData& data_minus,
                           const common::OdomData& data_plus,
                           common::State* state_ptr,
                           Eigen::Matrix<double, 9, 9>* F_ptr = nullptr,
                           Eigen::Matrix<double, 9, 9>* Qd_ptr = nullptr);

};

}
#endif
