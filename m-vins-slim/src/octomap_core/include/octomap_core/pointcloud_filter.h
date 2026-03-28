#ifndef MVINS_CORE_POINT_CLOUD_FILTER_H_
#define MVINS_CORE_POINT_CLOUD_FILTER_H_

#include <unordered_set>

#include "data_common/sensor_structures.h"

namespace vins_core {
// NOTE(chien):
// This is the implementation class of real-time point cloud noise filtering.
// Compare with statistical outlier rejection,
// we use voxel grid filtering instead of nearest neighbor search for
// reduce the computational complexity.
// We assume that the point cloud is in the camera frame,
// so negative values in Z axis will not be allowed.
class RealTimeOutlierRemoval {
public:

    typedef std::vector<std::vector<std::vector<std::vector<std::size_t>>>> Grid;

    struct GridIndex {
        GridIndex(const int _x, const int _y, const int _z)
            : x(_x), y(_y), z(_z) {}
        bool operator == (const GridIndex& other) const {
            bool result = true;
            result &= x == other.x;
            result &= y == other.y;
            result &= z == other.z;
            return result;
        }
        int x;
        int y;
        int z;
    };

    struct GridIndexHash {
        std::size_t operator () (const GridIndex& value) const {
            return std::hash<size_t>()(static_cast<size_t>(value.x)) ^
                   std::hash<size_t>()(static_cast<size_t>(value.y)) ^
                   std::hash<size_t>()(static_cast<size_t>(value.z));
        }
    };

    explicit RealTimeOutlierRemoval(const float resolution);
    ~RealTimeOutlierRemoval() = default;
    void SetInput(octomap::PointCloudXYZRGB* input);
    octomap::PointCloudXYZRGB Filter(const size_t pt_size_per_metre,
                                      const bool voxel_filtering);
private:
    const float resolution_;
    Grid grid_;
    int grid_size_x_;
    int grid_size_y_;
    int grid_size_z_;
    std::unordered_set<GridIndex, GridIndexHash> grid_indices_set_;
    octomap::PointCloudXYZRGB* input_;
};

}
#endif
