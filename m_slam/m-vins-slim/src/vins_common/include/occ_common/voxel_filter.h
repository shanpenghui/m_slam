#ifndef OCC_COMMON_VOXEL_FILTER_H_
#define OCC_COMMON_VOXEL_FILTER_H_

#include <bitset>

#include "data_common/sensor_structures.h"

namespace common {
struct AdaptiveVoxelFilterOptions {
    AdaptiveVoxelFilterOptions() = default;
    AdaptiveVoxelFilterOptions(const double max_length,
                               const size_t min_num_points,
                               const double max_range)
        : max_length_(max_length),
          min_num_points_(min_num_points),
          max_range_(max_range) {}

    const double max_length_ = 0.2;
    const size_t min_num_points_ = 200u;
    const double max_range_ = common::kMaxRangeScan;
};

// Voxel filter for point clouds. For each voxel, the assembled point cloud
// contains the first point that fell into it from any of the inserted point
// clouds.
class VoxelFilter {
 public:
    explicit VoxelFilter(double resolution) :
        resolution_inverse_(1.0 / resolution) {}

    VoxelFilter(const VoxelFilter&) = delete;
    VoxelFilter& operator=(const VoxelFilter&) = delete;

    // Returns a voxel filtered copy of 'point_cloud'.
    template<class PC>
    PC Filter(const PC& point_cloud);

    common::EigenVector3dVec Filter(
            const common::EigenVector3dVec& point_cloud);
 private:
    using KeyType = std::bitset<3 * 32>;

    static KeyType IndexToKey(const Eigen::Array3i& index);

    Eigen::Array3i GetCellIndex(const Eigen::Vector3d& point) const;

    double resolution_inverse_;
    std::unordered_set<KeyType> voxel_set_;
};

class AdaptiveVoxelFilter {
 public:
    AdaptiveVoxelFilter(const AdaptiveVoxelFilterOptions& options);
    AdaptiveVoxelFilter(const double max_length,
                       const size_t min_num_points,
                       const double max_range);

    AdaptiveVoxelFilter(const AdaptiveVoxelFilter&) = delete;
    AdaptiveVoxelFilter& operator=(const AdaptiveVoxelFilter&) = delete;

    common::PointCloud Filter(const common::PointCloud& point_cloud) const;

 private:
    const AdaptiveVoxelFilterOptions vf_option_;
};

}

#endif
