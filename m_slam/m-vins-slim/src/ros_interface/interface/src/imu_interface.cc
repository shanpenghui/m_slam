#include "interface/interface.h"

#include "data_common/sensor_structures.h"

namespace mvins {
//! Add odometry data from ROS messages.
void Interface::FillImuMsg(const ImuMsgPtr imu) {

    if (nullptr == imu){
        LOG(WARNING) << "WARNNING: Imu message is nullptr";
        return;
    }
    common::ImuData imu_meas;

#ifdef USE_ROS2
    RosTime t_msg = imu->header.stamp;
    if (t_msg.nanoseconds() < 0l) {
        LOG(ERROR) << "Received a negative timestamp in imu message.";
        return;
    }
    imu_meas.timestamp_ns = t_msg.nanoseconds();
#else
    imu_meas.timestamp_ns = imu->header.stamp.toNSec();
#endif
    imu_meas.acc << imu->linear_acceleration.x,
                    imu->linear_acceleration.y,
                    imu->linear_acceleration.z;

    imu_meas.gyro << imu->angular_velocity.x,
                     imu->angular_velocity.y,
                     imu->angular_velocity.z;

    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->AddImu(imu_meas);
    }
}
}
