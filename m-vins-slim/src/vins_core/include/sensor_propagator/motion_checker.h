#ifndef MVINS_MOTION_CHECKER_H_
#define MVINS_MOTION_CHECKER_H_

#include "sensor_propagator/imu_propagator.h"
#include "sensor_propagator/odom_propagator.h"

namespace vins_core {

class MotionChecker {
public:
    MotionChecker(const common::SlamConfigPtr& config,
                  const double min_trans,
                  const double min_rot);
    ~MotionChecker() = default;
    bool CheckIsMotionEnough(const common::OdomDatas& prop_data);
    bool CheckIsMotionEnough(const common::ImuDatas& prop_data);
private:
    bool MotionChecking(const common::State& state);
    
    std::unique_ptr<vins_core::OdomPropagator> odom_propagator_ptr_;
    std::unique_ptr<vins_core::ImuPropagator> imu_propagator_ptr_;

    const double min_trans_;
    const double min_rot_;
};

}

#endif
