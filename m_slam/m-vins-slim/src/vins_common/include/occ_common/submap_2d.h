#ifndef OCCUPANCY_GRID_SUBMAP_2D_H_
#define OCCUPANCY_GRID_SUBMAP_2D_H_

#include "aslam/common/pose-types.h"
#include "data_common/state_structures.h"
#include "occ_common/castrayer.h"
#include "occ_common/submap.h"

namespace common {

class Submap2D : public Submap {
 public:
    Submap2D(const aslam::Transformation& origin,
               std::unique_ptr<common::Grid2D> grid);

    const common::Grid2D* grid() const { return grid_.get(); }

    common::Grid2D* mutable_grid() { return grid_.get(); }

    const cv::Mat& submap_odds() const { return submap_odds_; }

    void InsertRangeData(const common::PointCloud& range_data,
                         const Eigen::Vector3d& origin,
                         const int keyframe_id,
                         const common::CastRayer* castrayer_ptr);
    void Finish();

    common::EigenVector3dVec ExtractOccupidPixeToPointCloud();

 private:
    std::unique_ptr<common::Grid2D> grid_;

    // Store the odds of each cell occupied probability in submap.
    // Computed when submap is finished.
    cv::Mat submap_odds_;

};

common::KeyFrame CreateVirtualKeyframe(
   const std::shared_ptr<common::Submap2D>& submap,
   const common::KeyFrames& keyframes,
   const std::unordered_map<int, size_t>& keyframe_id_to_idx);

}

#endif
