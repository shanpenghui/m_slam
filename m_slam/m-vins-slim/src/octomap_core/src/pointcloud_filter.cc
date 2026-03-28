#include "octomap_core/pointcloud_filter.h"

#include <glog/logging.h>

#include "data_common/constants.h"

namespace vins_core {

RealTimeOutlierRemoval::RealTimeOutlierRemoval(const float resolution)
    : resolution_(resolution) {
    grid_size_x_ = std::ceil((2.f * common::kMaxWidth) / resolution_);
    grid_size_y_ = grid_size_x_;
    grid_size_z_ = std::ceil(common::kMaxRangeVisual / resolution_);

    grid_.resize(grid_size_x_);
    for (int i = 0; i < grid_size_x_; ++i) {
        grid_[i].resize(grid_size_y_);
        for (int j = 0; j < grid_size_y_; ++j) {
            grid_[i][j].resize(grid_size_z_);
        }
    }
}

void RealTimeOutlierRemoval::SetInput(octomap::PointCloudXYZRGB* input) {
    input_ = input;

    // Clear.
    if (!grid_indices_set_.empty()) {
        for (std::unordered_set<GridIndex, GridIndexHash>::iterator it = grid_indices_set_.begin();
             it != grid_indices_set_.end(); ++it) {
            grid_[(*it).x][(*it).y][(*it).z].clear();
        }
    }
    grid_indices_set_.clear();

    // Set.
    for (size_t i = 0u; i < input_->xyz.size(); ++i) {
        if (std::abs(input_->xyz[i].x()) > common::kMaxWidth ||
           std::abs(input_->xyz[i].y()) > common::kMaxWidth ||
           input_->xyz[i].z() < 0.f || input_->xyz[i].z() > common::kMaxRangeVisual) {
            continue;
        }

        const int idx_x = (input_->xyz[i].x() + common::kMaxWidth) / resolution_;
        const int idx_y = (input_->xyz[i].y() + common::kMaxWidth) / resolution_;
        const int idx_z = input_->xyz[i].z() / resolution_;

        if (idx_x >= grid_size_x_ || idx_y >= grid_size_y_ || idx_z >= grid_size_z_) {
            continue;
        }
        grid_[idx_x][idx_y][idx_z].push_back(i);

        GridIndex grid_index(idx_x, idx_y, idx_z);
        grid_indices_set_.insert(grid_index);
    }
}

octomap::PointCloudXYZRGB RealTimeOutlierRemoval::Filter(const size_t pt_size_per_metre,
                                                          const bool voxel_filtering) {
    const size_t pt_size_per_grid = pt_size_per_metre * resolution_;

    octomap::PointCloudXYZRGB output;
    for (std::unordered_set<GridIndex, GridIndexHash>::iterator it = grid_indices_set_.begin();
         it != grid_indices_set_.end(); ++it) {
        if (grid_[(*it).x][(*it).y][(*it).z].size() >= pt_size_per_grid) {
            for (size_t i = 0u; i < grid_[(*it).x][(*it).y][(*it).z].size(); ++i) {
                const size_t pc_idx = grid_[(*it).x][(*it).y][(*it).z][i];
                output.xyz.push_back(input_->xyz[pc_idx]);
                output.rgb.push_back(input_->rgb[pc_idx]);
                if (voxel_filtering) {
                    break;
                }
            }
        }
    }
    return output;
}

}
