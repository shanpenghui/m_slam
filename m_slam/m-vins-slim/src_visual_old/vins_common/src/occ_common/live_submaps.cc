#include "occ_common/live_submaps.h"

namespace common {

LiveSubmaps::LiveSubmaps(const int range_data_size,
                           const double resolution)
    : range_data_size_(range_data_size),
      resolution_(resolution),
      grid_id_counter_(0),
      castrayer_(CreateCastRayer()) {}

std::vector<std::shared_ptr<common::Submap2D>> LiveSubmaps::submaps() {
    return submaps_;
}

std::shared_ptr<common::Submap2D> LiveSubmaps::InsertRangeData(
        const common::KeyFrame& key_frame,
        const common::PointCloud& p_LinGs,
        const aslam::Transformation& T_SToO,
        const int max_submaps_size) {
    // Add range data into general submaps.
    int range_data_unit = -1;
    if (range_data_size_ != -1) {
        range_data_unit = range_data_size_ / max_submaps_size;
    }

    if (submaps_.empty() ||
            submaps_.back()->num_range_data() == range_data_unit) {
        AddEmptySubmap(key_frame.state.T_OtoG, max_submaps_size);
    }
    for (auto& submap : submaps_) {
        submap->InsertRangeData(p_LinGs,
                                (key_frame.state.T_OtoG * T_SToO).getPosition(),
                                key_frame.keyframe_id,
                                castrayer_.get());
    }

    if (submaps_.front()->num_range_data() == range_data_size_) {
        submaps_.front()->Finish();
        return submaps_.front();
    }
    return nullptr;
}

std::unique_ptr<common::CastRayer>
LiveSubmaps::CreateCastRayer() {
    return std::make_unique<common::CastRayer>();
}

std::unique_ptr<common::Grid2D> LiveSubmaps::CreateGrid(
        const Eigen::Vector2d& origin) {
    constexpr int kInitialSubmapSize = 600;

    return std::make_unique<common::ProbabilityGrid>(
                grid_id_counter_++,
                common::MapLimits(resolution_,
                                    origin + 0.5 * kInitialSubmapSize * resolution_ *
                                    Eigen::Vector2d::Ones(),
                                    common::CellLimits(kInitialSubmapSize, kInitialSubmapSize)),
                &conversion_tables_);
}

std::unique_ptr<common::Grid2D> LiveSubmaps::CreateGrid(
        const Eigen::Vector2d& origin,
        const Eigen::Vector2d& range_size) {
    return std::make_unique<common::ProbabilityGrid>(
                grid_id_counter_++,
                common::MapLimits(resolution_,
                                    origin + 0.5 * range_size * resolution_,
                                    common::CellLimits(range_size(1), range_size(0))),
                &conversion_tables_);
}

void LiveSubmaps::InsertSubmap(
        std::shared_ptr<common::Submap2D> submap) {
    submaps_.push_back(submap);
}

void LiveSubmaps::AddEmptySubmap(
        const aslam::Transformation& T_OtoG,
        const size_t max_submaps_size) {
    if (submaps_.size() >= max_submaps_size) {
        CHECK(submaps_.front()->insertion_finished());
        submaps_.erase(submaps_.begin());
    }
    submaps_.push_back(std::make_unique<common::Submap2D>(T_OtoG,
            CreateGrid(T_OtoG.getPosition().head<2>())));
}

void LiveSubmaps::LoadMap(const cv::Mat& raw_map,
                          const aslam::Transformation& origin) {
    const int rows = raw_map.rows;
    const int cols = raw_map.cols;
    const Eigen::Vector2d xy_min(origin.getPosition().head<2>());
    const Eigen::Vector2d range_size(cols, rows);
    const Eigen::Vector2d xy_center = xy_min + 0.5 * range_size * resolution_;
    submaps_.push_back(std::make_unique<common::Submap2D>(
                           origin, CreateGrid(xy_center, range_size)));
    CHECK_EQ(submaps_.size(), 1u);
    std::set<std::pair<int, int>> unknow_cells;
    for (int i = 0; i < raw_map.rows; ++i) {
        for (int j = 0; j < raw_map.cols; ++j) {
            if (raw_map.at<uchar>(i, j) == 128) {
                unknow_cells.insert(std::make_pair(i, j));
            }
        }
    }
#if 0
    cv::Mat map_binary;
    cv::threshold(raw_map, map_binary, 100, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::erode(map_binary, map_binary, kernel);

    cv::Mat map_final;
    cv::distanceTransform(map_binary, map_final, cv::DIST_L2, 5);
    map_final.convertTo(map_final, CV_8UC1);
    cv::equalizeHist(map_final, map_final);
#else
    cv::Mat map_final;
    raw_map.copyTo(map_final);
#endif
    common::ProbabilityGrid* map_grid =
            static_cast<common::ProbabilityGrid*>(submaps_.front()->mutable_grid());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar value;
            if (unknow_cells.count(std::make_pair(i, j)) == 0) {
                value = map_final.at<uchar>(i, j);
            } else {
                value = 128;
            }
            const float probility = (255.f - static_cast<float>(value)) / 255.f;
            map_grid->SetProbability(Eigen::Array2i(i, j), probility);
        }
    }
}

void LiveSubmaps::ClearSubmaps() {
    submaps_.clear();
}
}
