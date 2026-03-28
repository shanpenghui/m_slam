#ifndef OCCUPANCY_GRID_LIVE_SUBMAP_H_
#define OCCUPANCY_GRID_LIVE_SUBMAP_H_

#include "occ_common/submap_2d.h"

namespace common {
class LiveSubmaps {

 public:
    LiveSubmaps(const int range_data_size,
                 const double resolution);

    void LoadMap(const cv::Mat& raw_map,
                  const aslam::Transformation& origin);

    std::shared_ptr<common::Submap2D> InsertRangeData(
            const common::KeyFrame& key_frame,
            const common::PointCloud& p_LinGs,
            const aslam::Transformation& T_SToO,
            const int max_submaps_size);

    std::vector<std::shared_ptr<common::Submap2D>> submaps();

    void InsertSubmap(std::shared_ptr<common::Submap2D> submap);

    void AddEmptySubmap(
            const aslam::Transformation& T_OtoG,
            const size_t max_submaps_size);

    void ClearSubmaps();

    std::unique_ptr<common::Grid2D> CreateGrid(const Eigen::Vector2d& origin);
    std::unique_ptr<common::Grid2D> CreateGrid(const Eigen::Vector2d& origin,
                                                const Eigen::Vector2d& range_size);

 private:
    std::unique_ptr<common::CastRayer> CreateCastRayer();
    void FinishSubmap();

    // Number of range data before adding a new submap. Each submap will get
    // twice the number of range data inserted: First for initialization
    // without being matched against, then while being matched.
    int range_data_size_;

    // Map resolution.
    double resolution_;

    // Counter for grid ID.
    int grid_id_counter_;

    std::vector<std::shared_ptr<common::Submap2D>> submaps_;
    std::unique_ptr<common::CastRayer> castrayer_;
    common::ValueConversionTables conversion_tables_;

};
}

#endif
