#include "occ_common/submap_2d.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "Eigen/Geometry"
#include "glog/logging.h"

#include "occ_common/castrayer.h"

namespace common {

Submap2D::Submap2D(const aslam::Transformation& origin,
                      std::unique_ptr<common::Grid2D> grid)
    : Submap(origin) {
    grid_ = std::move(grid);
    submap_id_ = grid_->id();
}

void Submap2D::InsertRangeData(
        const common::PointCloud& range_data,
        const Eigen::Vector3d& origin,
        const int keyframe_id,
        const common::CastRayer* castrayer_ptr) {
    CHECK(grid_);
    CHECK(!insertion_finished());
    castrayer_ptr->Insert(range_data, origin, grid_.get());
    scanid_in_submap_.insert(keyframe_id);
    set_num_range_data(num_range_data() + 1);
}

common::EigenVector3dVec Submap2D::ExtractOccupidPixeToPointCloud() {
    common::EigenVector3dVec point_cloud;

    const auto& limits = grid_->limits();
    Eigen::Array2i offset;
    common::CellLimits cell_limits;
    grid_->ComputeCroppedLimits(&offset, &cell_limits);
    for (const Eigen::Array2i& xy_index :
            common::XYIndexRangeIterator(cell_limits)) {
        if (grid_->IsKnown(xy_index + offset)) {
            const int delta = 255 - common::ProbabilityToLogOddsInteger(
                            grid_->GetValue(xy_index + offset));
            const u_int8_t value = delta > 0 ? delta : 0;
            if (value < 50) {
                const Eigen::Vector2d xy =
                        limits.GetCellCenter(xy_index + offset);
                point_cloud.emplace_back(xy(0), xy(1), 0.);
            }
        }
    }

    return point_cloud;
}

void Submap2D::Finish() {
    CHECK(grid_);
    CHECK(!insertion_finished());
    std::lock_guard<std::mutex> lck(finish_mutex_);
    grid_ = grid_->ComputeCroppedGrid(&submap_odds_);
    set_insertion_finished(true);
}

common::KeyFrame CreateVirtualKeyframe(
    const std::shared_ptr<common::Submap2D>& submap,
    const common::KeyFrames& keyframes,
    const std::unordered_map<int, size_t>& keyframe_id_to_idx) {
    common::EigenVector3dVec point_cloud =
            submap->ExtractOccupidPixeToPointCloud();

    const int first_keyframe_id = *(submap->scanid_in_submap().begin());
    const auto& iter = keyframe_id_to_idx.find(first_keyframe_id);
    CHECK(iter != keyframe_id_to_idx.end());
    const common::KeyFrame& first_keyframe = keyframes[iter->second];
    common::KeyFrame virtual_keyframe(first_keyframe_id, first_keyframe.state);
    const aslam::Transformation T_GtoO = first_keyframe.state.T_OtoG.inverse();
    for (const auto& pt : point_cloud) {
        virtual_keyframe.points.points.push_back(T_GtoO.transform(pt));
    }
    virtual_keyframe.points.timestamp_ns = first_keyframe.points.timestamp_ns;

    return virtual_keyframe;
}

}

