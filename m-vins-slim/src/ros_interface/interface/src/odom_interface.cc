#include "interface/interface.h"

namespace mvins {
//! Add odometry data from ROS messages.
void Interface::FillOdomMsg(const OdometryMsgPtr odom) {
    if (nullptr == odom){
        LOG(WARNING) << "WARNNING: Odom message is nullptr";
        return;
    }
    common::OdomData odom_meas;

#ifdef USE_ROS2
    RosTime t_msg = odom->header.stamp;
    if (t_msg.nanoseconds() < 0l) {
        LOG(ERROR) << "Received a negative timestamp in odom message.";
        return;
    }
    odom_meas.timestamp_ns = t_msg.nanoseconds();
#else
    odom_meas.timestamp_ns = odom->header.stamp.toNSec();
#endif

    odom_meas.angular_velocity << odom->twist.twist.angular.x,
                                  odom->twist.twist.angular.y,
                                  odom->twist.twist.angular.z;

    odom_meas.linear_velocity << odom->twist.twist.linear.x,
                                 odom->twist.twist.linear.y,
                                 odom->twist.twist.linear.z;

    odom_meas.p << odom->pose.pose.position.x,
                   odom->pose.pose.position.y,
                   odom->pose.pose.position.z;
    if (fabs(odom->pose.pose.position.x) > 1e4 ||
        fabs(odom->pose.pose.position.y) > 1e4 ||
        odom->pose.pose.position.z > 1e4) {
        LOG(ERROR) << "fuck odom msg skip...";
        return;
    }
    odom_meas.q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
                                     odom->pose.pose.orientation.x,
                                     odom->pose.pose.orientation.y,
                                     odom->pose.pose.orientation.z);

    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->AddOdom(odom_meas);
    }
    if (interface_manager_ptr_ != nullptr) {
        interface_manager_ptr_->AddOdom(odom_meas);
    }
}

void Interface::FillGroundTruthMsg(const OdometryMsgPtr gt) {
    if (nullptr == gt){
        LOG(WARNING) << "WARNNING: Ground truth message is nullptr";
        return;
    }
    common::OdomData gt_meas;

#ifdef USE_ROS2
    RosTime t_msg = gt->header.stamp;
    gt_meas.timestamp_ns = t_msg.nanoseconds();
#else
    gt_meas.timestamp_ns = gt->header.stamp.toNSec();
#endif

    gt_meas.p << gt->pose.pose.position.x,
                 gt->pose.pose.position.y,
                 gt->pose.pose.position.z;

    gt_meas.q = Eigen::Quaterniond(gt->pose.pose.orientation.w,
                                   gt->pose.pose.orientation.x,
                                   gt->pose.pose.orientation.y,
                                   gt->pose.pose.orientation.z);
                                   
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        vins_handler_ptr_->AddGroundTruth(gt_meas);
    }
}
}
