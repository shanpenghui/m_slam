#include "occ_common/voxel_filter.h"

namespace common {

common::PointCloud FilterByMaxRange(
        const common::PointCloud& range_data,
        const double max_range) {
    common::PointCloud result;
    for (const auto& point : range_data.points) {
        if (point.head<2>().norm() <= max_range) {
            result.points.emplace_back(point);
        }
    }
    return result;
}

common::PointCloud AdaptivelyVoxelFiltered(
    const AdaptiveVoxelFilterOptions& options,
    const common::PointCloud& point_cloud) {
    if (point_cloud.points.size() <= options.min_num_points_) {
        // 'point_cloud' is already sparse enough.
        return point_cloud;
    }
    common::PointCloud result = VoxelFilter(options.max_length_).Filter(point_cloud);
    if (result.points.size() >= options.min_num_points_) {
        // Filtering with 'max_length' resulted in a sufficiently dense point
        // cloud.
        return result;
    }
    // Search for a 'low_length' that is known to result in a sufficiently
    // dense point cloud. We give up and use the full 'point_cloud' if reducing
    // the edge length by a factor of 1e-2 is not enough.
    for (double high_length = options.max_length_;
         high_length > 1e-2f * options.max_length_; high_length /= 2.f) {
        double low_length = high_length / 2.f;
        result = VoxelFilter(low_length).Filter(point_cloud);
        if (result.points.size() >= options.min_num_points_) {
            // Binary search to find the right amount of filtering.
            // 'low_length' gave a sufficiently dense 'result',
            // 'high_length' did not.
            // We stop when the edge length is at most 10% off.
            while ((high_length - low_length) / low_length > 1e-1f) {
                const double mid_length = (low_length + high_length) / 2.f;
                common::PointCloud candidate =
                    VoxelFilter(mid_length).Filter(point_cloud);
                if (candidate.points.size() >= options.min_num_points_) {
                    low_length = mid_length;
                    result = std::move(candidate);
                } else {
                    high_length = mid_length;
                }
            }
            return result;
        }
    }
    return result;
}

template<class PC>
PC VoxelFilter::Filter(const PC& point_cloud) {
    PC results;
    for (const auto& point : point_cloud.points) {
        const auto it_inserted =
            voxel_set_.insert(IndexToKey(GetCellIndex(point)));
        if (it_inserted.second) {
            results.points.emplace_back(point);
        }
    }
    return results;
}

common::EigenVector3dVec
VoxelFilter::Filter(const common::EigenVector3dVec& point_cloud) {
    common::EigenVector3dVec results;
    for (const auto& point : point_cloud) {
        const auto it_inserted =
            voxel_set_.insert(IndexToKey(GetCellIndex(point)));
        if (it_inserted.second) {
            results.emplace_back(point);
        }
    }
    return results;
}

VoxelFilter::KeyType VoxelFilter::IndexToKey(const Eigen::Array3i& index) {
    KeyType k_0(static_cast<uint32_t>(index[0]));
    KeyType k_1(static_cast<uint32_t>(index[1]));
    KeyType k_2(static_cast<uint32_t>(index[2]));
    return (k_0 << 2 * 32) | (k_1 << 1 * 32) | k_2;
}

Eigen::Array3i VoxelFilter::GetCellIndex(const Eigen::Vector3d& point) const {
    const Eigen::Vector3d index = point * resolution_inverse_;
    return Eigen::Array3i(std::lround(index.x()),
                          std::lround(index.y()),
                          std::lround(index.z()));
}

AdaptiveVoxelFilter::AdaptiveVoxelFilter(
        const AdaptiveVoxelFilterOptions& options)
    : vf_option_(options) {}

AdaptiveVoxelFilter::AdaptiveVoxelFilter(
        const double max_length,
        const size_t min_num_points,
        const double max_range)
    : vf_option_(max_length, min_num_points, max_range) {}

common::PointCloud AdaptiveVoxelFilter::Filter(
        const common::PointCloud& point_cloud) const {
    return AdaptivelyVoxelFiltered(
            vf_option_, FilterByMaxRange(point_cloud, vf_option_.max_range_));
}

}
