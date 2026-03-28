#ifndef DATA_COMMON_SENSOR_STRUCTURES_H_
#define DATA_COMMON_SENSOR_STRUCTURES_H_

#include <Eigen/Dense>
#include <deque>
#include <octomap/OcTree.h>

#include "aslam/common/pose-types.h"
#include "data_common/constants.h"
#include "time_common/time.h"

namespace octomap {
struct PointCloudXYZRGB {
    octomap::Pointcloud xyz;
    octomap::Pointcloud rgb;
};
}

namespace common {
// Sensor data base structure.
struct SensorData {
    SensorData() :
        timestamp_ns(0u) {
    }
    virtual ~SensorData() {};

    SensorData(const uint64_t _timestamp_ns) :
        timestamp_ns(_timestamp_ns) {
    }

    SensorData(const SensorData& other) :
        timestamp_ns(other.timestamp_ns) {
    }

    void operator=(const SensorData& other) {
        timestamp_ns = other.timestamp_ns;
    }

    // Sort function to allow for using of STL containers.
    bool operator<(const SensorData& other) const {
        return timestamp_ns < other.timestamp_ns;
    }

    // Timestamp in nanoseconds.
    uint64_t timestamp_ns;
};

// Image measurement data.
struct ImageData : public SensorData {
    ImageData() : SensorData() {}
    ~ImageData() override {} 

    ImageData(const uint64_t _timestamp_ns) :
        SensorData(_timestamp_ns) {
    }

    ImageData(
        const uint64_t _timestamp_ns,
        const CvMatConstPtrVec& _images) :
        SensorData(_timestamp_ns),
        images(_images) {
        depth = nullptr;
    }

    ImageData(
        const uint64_t _timestamp_ns,
        const CvMatConstPtrVec& _images,
        const CvMatConstPtr& _depth) :
        SensorData(_timestamp_ns),
        images(_images),
        depth(_depth) {

    }

    ImageData(const ImageData& other) :
        SensorData(other.timestamp_ns),
        images(other.images),
        depth(other.depth) {
    }

    void operator=(const ImageData& other) {
        timestamp_ns = other.timestamp_ns;
        images = other.images;
        depth = other.depth;
    }

    void Clear() {
        for (size_t i = 0u; i < images.size(); ++i) {
            images[i].reset();
        }
        images.resize(0u);
        depth.reset();
    }

    CvMatConstPtrVec images;
    CvMatConstPtr depth;
};

// Depth measurement data.
struct DepthData : public SensorData {
    DepthData() : SensorData() {}
    ~DepthData() override {} 

    DepthData(const uint64_t _timestamp_ns) :
        SensorData(_timestamp_ns) {
    }

    DepthData(
        const uint64_t _timestamp_ns,
        const CvMatConstPtr& _depth) :
        SensorData(_timestamp_ns),
        depth(_depth) {
    }

    DepthData(const DepthData& other) :
        SensorData(other.timestamp_ns),
        depth(other.depth) {
    }

    void operator=(const DepthData& other) {
        timestamp_ns = other.timestamp_ns;
        depth = other.depth;
    }

    CvMatConstPtr depth;
};

// IMU measurement data.
struct ImuData : public SensorData {
    ImuData() : SensorData() {}
    ~ImuData() override {} 

    ImuData(const uint64_t _timestamp_ns) :
        SensorData(_timestamp_ns) {
    }

    ImuData(const ImuData& other) :
        SensorData(other.timestamp_ns),
        gyro(other.gyro),
        acc(other.acc) {
    }

    void operator=(const ImuData& other) {
        timestamp_ns = other.timestamp_ns;
        gyro = other.gyro;
        acc = other.acc;
    }

    /// Gyroscope reading, angular velocity (rad/s).
    Eigen::Vector3d gyro;

    /// Accelerometer reading, linear acceleration (m/s^2).
    Eigen::Vector3d acc;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline ImuData Interpolate(const uint64_t timestamp_ns,
                          const double lambda,
                          const ImuData& meas_1,
                          const ImuData& meas_2) {
    common::ImuData data(timestamp_ns);
    data.acc = (1. - lambda) * meas_1.acc + lambda * meas_2.acc;
    data.gyro = (1. - lambda) * meas_1.gyro + lambda * meas_2.gyro;
    return data;
}

typedef std::vector<ImuData> ImuDatas;

// Odom measurement data.
struct OdomData : public SensorData {
    OdomData() : SensorData() {}
    ~OdomData() override {}

    OdomData(const uint64_t _timestamp_ns) :
        SensorData(_timestamp_ns) {
    }

    OdomData(const OdomData& other) :
        SensorData(other.timestamp_ns),
        p(other.p),
        q(other.q),
        linear_velocity(other.linear_velocity),
        angular_velocity(other.angular_velocity) {
    }

    void operator=(const OdomData& other) {
        timestamp_ns = other.timestamp_ns;
        p = other.p;
        q = other.q;
        linear_velocity = other.linear_velocity;
        angular_velocity = other.angular_velocity;
    }

    // Position reading from odom measurements.
    Eigen::Vector3d p;

    // Orientation reading from odom measurements.
    Eigen::Quaterniond q;

    // Linear velocity reading from odom measurements.
    Eigen::Vector3d linear_velocity;

    // Angular velocity reading from odom measurements.
    Eigen::Vector3d angular_velocity;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline OdomData Interpolate(const uint64_t timestamp_ns,
                            const double lambda,
                            const OdomData& meas_1,
                            const OdomData& meas_2) {
    common::OdomData data(timestamp_ns);
    data.p = (1. - lambda) * meas_1.p + lambda * meas_2.p;
    data.q = meas_1.q.slerp(lambda, meas_2.q);
    data.angular_velocity = (1. - lambda) * meas_1.angular_velocity 
                            + lambda * meas_2.angular_velocity;
    data.linear_velocity = (1. - lambda) * meas_1.linear_velocity 
                            + lambda * meas_2.linear_velocity;
    return data;
}

typedef std::vector<OdomData> OdomDatas;

inline bool GetGtStateByTimeNs(const uint64_t query_time_ns,
                              const std::deque<common::OdomData>& gt_status,
                              common::OdomData* gt_state_ptr) {
    common::OdomData& gt_state = *CHECK_NOTNULL(gt_state_ptr);
    
    // Check that we even have groundtruth loaded
    if (gt_status.empty()) {
        LOG(ERROR) << "Groundtruth data loaded is empty, make sure you call load before asking for a state.";
        return false;
    }

    const double query_time_s = common::NanoSecondsToSeconds(query_time_ns);

    double closest_time = std::numeric_limits<double>::max();
    size_t closest_idx = std::numeric_limits<size_t>::max();
    for (size_t i = 0u; i < gt_status.size(); ++i) {
        const double curr_time_s = common::NanoSecondsToSeconds(gt_status[i].timestamp_ns);
        if(std::abs(curr_time_s - query_time_s) < std::abs(closest_time - query_time_s)) {
            closest_time = curr_time_s;
            closest_idx = i;
        }
    }

    // If close to this timestamp, then use it
    if(std::abs(closest_time - query_time_s) < 0.10) {
        gt_state = gt_status[closest_idx];
        return true;
    }

    return false;
}

// Laser scan measurements data.
struct ScanData : public SensorData {
    ScanData() : SensorData() {}
    ~ScanData() override {} 

    ScanData(const uint64_t _timestamp_ns) :
        SensorData(_timestamp_ns) {
    }

    ScanData(const ScanData& other) :
        SensorData(other.timestamp_ns) {
        ranges = other.ranges;
        angle_min = other.angle_min;
        angle_max = other.angle_max;
        angle_increment = other.angle_increment;
        range_min = other.range_min;
        range_max = other.range_max;
        time_increments = other.time_increments;
    }

    void operator=(const ScanData& other) {
        timestamp_ns = other.timestamp_ns;
        ranges = other.ranges;
        angle_min = other.angle_min;
        angle_max = other.angle_max;
        angle_increment = other.angle_increment;
        range_min = other.range_min;
        range_max = other.range_max;
        time_increments = other.time_increments;
    }

    // Range data.
    std::vector<float> ranges;

    // Scan parameters.
    double angle_min;
    double angle_max;
    double angle_increment;
    double range_min;
    double range_max;
    double time_increments;
};

typedef std::shared_ptr<const common::SensorData> SensorDataConstPtr;

typedef Eigen::Vector3d PointXYZ;

struct PointCloud : public SensorData {
 public:
    PointCloud() : SensorData() {}
    ~PointCloud() override {} 
    
    PointCloud(const uint64_t _timestamp_ns) :
        SensorData(_timestamp_ns) {
    }

    PointCloud(const PointCloud& other) :
        SensorData(other.timestamp_ns),
        points(other.points),
        miss_points(other.miss_points) {}

    inline PointCloud& operator += (const PointCloud& pc) {
        points.insert(points.end(), pc.points.begin(), pc.points.end());
        miss_points.insert(miss_points.end(),
                pc.miss_points.begin(), pc.miss_points.end());
        return *this;
    }

    // For nanoflann search.
    inline size_t kdtree_get_point_count() const {
        return points.size();
    }

    inline double kdtree_get_pt(const size_t idx, int dim) const {
        return points[idx](dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }

    void operator=(const PointCloud& other) {
        timestamp_ns = other.timestamp_ns;
        points = other.points;
        miss_points = other.miss_points;
    }

    int64_t timestamp_ns;
    std::vector<PointXYZ, Eigen::aligned_allocator<PointXYZ>> points;
    std::vector<PointXYZ, Eigen::aligned_allocator<PointXYZ>> miss_points;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

PointCloud GenerateScanPoints(const SensorDataConstPtr& scan_data,
                               const Eigen::Vector2d& xy_velocity,
                               const double theta_velocity);

class PointCloudMap {
 public:
    PointCloudMap() = default;

    PointCloudMap(PointCloudMap&& other);

    void operator=(PointCloudMap&& other);

    void RemoveFirstPointCloud();

    inline PointCloudMap& operator += (const PointCloud& pc) {
        points.insert(points.end(), pc.points.begin(), pc.points.end());
        pc_idx.emplace_back(points.size());
        return *this;
    }

    inline PointCloudMap& operator += (const PointCloudMap& pc_map) {
        points.insert(points.end(), pc_map.points.begin(), pc_map.points.end());
        pc_idx.emplace_back(points.size());
        return *this;
    }

    inline size_t kdtree_get_point_count() const {
        return points.size();
    }

    inline double kdtree_get_pt(const size_t idx, int dim) const {
        return points[idx](dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }

    std::deque<PointXYZ, Eigen::aligned_allocator<PointXYZ>> points;
    std::deque<size_t> pc_idx;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct SyncedHybridSensorData {
    SyncedHybridSensorData() = default;

    SyncedHybridSensorData(const uint64_t _timestamp_ns,
                           const ImuDatas& _imu_datas,
                           const OdomDatas& _odom_datas,
                           const ImageData& _img_data,
                           const SensorDataConstPtr& _scan_data)
        : timestamp_ns(_timestamp_ns),
          imu_datas(_imu_datas),
          odom_datas(_odom_datas),
          img_data(_img_data),
          scan_data(_scan_data) {
    }

    SyncedHybridSensorData(const SyncedHybridSensorData& other, const bool copy_image = true) {
        timestamp_ns = other.timestamp_ns;
        imu_datas = other.imu_datas;
        odom_datas = other.odom_datas;
        scan_data = other.scan_data;
        if (copy_image) {
            img_data = other.img_data;
        }
    }

    void operator=(const SyncedHybridSensorData& other) {
        timestamp_ns = other.timestamp_ns;
        imu_datas = other.imu_datas;
        odom_datas = other.odom_datas;
        img_data = other.img_data;
        scan_data = other.scan_data;
    }

    // Timestamp.
    uint64_t timestamp_ns;

    // Imu datas.
    // NOTE: the imu meas datas time occupation is
    // from current keyframe to next keyframe.
    ImuDatas imu_datas;

    // Odom datas.
    // NOTE: the odom meas datas time occupation is
    // from current keyframe to next keyframe.
    OdomDatas odom_datas;

    // Image data.
    ImageData img_data;

    // Scan data.
    SensorDataConstPtr scan_data;
};

typedef std::pair<uint64_t, octomap::PointCloudXYZRGB> PointCloudWithTimeStamp;
// First free key, Second occupied key.
typedef std::pair<octomap::KeySet, octomap::KeySet> OctomapKeySetPair;
typedef std::deque<std::pair<octomap::KeySet, octomap::KeySet>> OctomapKeySetPairs;
}

#endif
