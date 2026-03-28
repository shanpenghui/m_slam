#include "interface/interface.h"

#include "sensor_msgs/point_cloud_conversion.hpp"
namespace mvins {
//! Add scan data from ROS messages.
void Interface::FillScanMsg(const LaserScanMsgPtr scan) {
    if (nullptr == scan){
        LOG(WARNING) << "WARNNING: Laser message is nullptr";
        return;
    }
    common::ScanData scan_meas;
    RosTime t_msg = scan->header.stamp;
    uint64_t timestamp = t_msg.nanoseconds();
    scan_meas.ranges = scan->ranges;
    scan_meas.angle_min = scan->angle_min;
    scan_meas.angle_max = scan->angle_max;
    scan_meas.angle_increment = scan->angle_increment;
    scan_meas.range_min = scan->range_min;
    scan_meas.range_max = scan->range_max;
    scan_meas.time_increments = scan->time_increment;
    
    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        double td_s;
        vins_handler_ptr_->GetNewTdScan(&td_s);
        uint64_t td_ns = common::SecondsToNanoSeconds(std::abs(td_s));
        uint64_t timestamp_td_ns;
        if (td_s > 0) {
            timestamp_td_ns = timestamp + td_ns;
        } else {
            timestamp_td_ns = timestamp - td_ns;
        }
        scan_meas.timestamp_ns = timestamp_td_ns;
        common::SensorDataConstPtr scan_meas_ptr = 
            std::make_shared<common::ScanData>(scan_meas);
        vins_handler_ptr_->AddScan(scan_meas_ptr);
    }
}

//! Add scan pointcloud2 data from ROS messages.
void Interface::FillScanPc2Msg(const PointCloudMsgPtr scan) {
    if (nullptr == scan){
        LOG(WARNING) << "WARNNING: Laser pointcloud message is nullptr";
        return;
    }

    RosTime t_msg = scan->header.stamp;
    uint64_t timestamp = t_msg.nanoseconds();
    sensor_msgs::msg::PointCloud pointcloud;

    sensor_msgs::convertPointCloud2ToPointCloud(*scan, pointcloud);
    common::PointCloud scan_pointcloud;
    CHECK_EQ(pointcloud.points.size(), pointcloud.channels[0].values.size());

    for (size_t i = 0u; i < pointcloud.points.size(); ++i) {
        common::PointXYZ scan_point;
        scan_point(0) = static_cast<double>(pointcloud.points[i].x);
        scan_point(1) = static_cast<double>(pointcloud.points[i].y);
        scan_point(2) = static_cast<double>(pointcloud.points[i].z);

        double range = scan_point.head<2>().norm();
        if (std::isnan(range) || std::isinf(range)) {
            continue;
        }
        if (range > common::kMaxRangeScan || range < common::kMinRange) {
            scan_pointcloud.miss_points.emplace_back(scan_point);
        } else {
            scan_pointcloud.points.emplace_back(scan_point);
        }
    }

    std::unique_lock<std::mutex> lock(vins_handler_ptr_mutex_);
    if (vins_handler_ptr_ != nullptr) {
        double td_s;
        vins_handler_ptr_->GetNewTdScan(&td_s);
        uint64_t td_ns = common::SecondsToNanoSeconds(std::abs(td_s));
        uint64_t timestamp_td_ns;
        if (td_s > 0) {
            timestamp_td_ns = timestamp + td_ns;
        } else {
            timestamp_td_ns = timestamp - td_ns;
        }
        scan_pointcloud.timestamp_ns = timestamp_td_ns;
        common::SensorDataConstPtr scan_meas_ptr = 
            std::make_shared<common::PointCloud>(scan_pointcloud);
        vins_handler_ptr_->AddScan(scan_meas_ptr);
    }
}
}
