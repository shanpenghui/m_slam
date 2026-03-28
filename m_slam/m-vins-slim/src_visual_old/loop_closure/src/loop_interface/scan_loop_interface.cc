#include "loop_interface/scan_loop_interface.h"

#include <opencv2/core.hpp>

#include "file_common/file_system_tools.h"
#include "occ_common/castrayer.h"
#include "occ_common/pointcloud_kdtree.h"
#include "occ_common/probability_values.h"
#include "parallel_common/parallel_process.h"
#include "time_common/time.h"
#include "time_common/time_table.h"
#include "yaml-cpp/yaml.h"

constexpr int kNumThreads = 40;
constexpr int kLoopDetectionInterval = 100;

constexpr double kTransThreshold = 0.1;
constexpr double kRotThreshold = 3.;

namespace loop_closure {

constexpr bool kShowPoseGraphResult = false;

Eigen::AlignedBox2d GetMapBoundary(
        const std::vector<std::shared_ptr<common::Submap2D>>& submaps) {

    Eigen::AlignedBox2d map_boundary;
    const double resolution =
            submaps.front()->grid()->limits().resolution();
    
    for (size_t i = 0u; i < submaps.size(); ++i) {
        const auto& submap = submaps[i];
        CHECK(submap->insertion_finished());
        const auto grid = submap->grid();
        common::MapLimits limit = grid->limits();
        const Eigen::Vector2d& limit_max = limit.max();
        Eigen::Vector2d limit_min = limit.GetCellCenter(
                Eigen::Array2i(limit.cell_limits().num_x_cells - 1,
                               limit.cell_limits().num_y_cells - 1));
        limit_min -= (Eigen::Vector2d::Ones() * resolution * 0.5);
        map_boundary.extend(limit_min);
        map_boundary.extend(limit_max); 
    }
    return map_boundary;
}

cv::Mat MapOdds2OccProbability(const cv::Mat& map_odds) {
    cv::Mat map_occ_grid(map_odds.rows, map_odds.cols,
                        CV_32FC1, cv::Scalar(-1.));

    for (int i = 0; i < map_occ_grid.rows; ++i) {
        for (int j = 0; j < map_occ_grid.cols; ++j) {
            const float odds = map_odds.at<float>(i, j);
            if (odds != 1.f) {
                map_occ_grid.at<float>(i, j) =
                        common::Clamp(common::ProbabilityFromOdds(odds),
                            common::kMinProbability, common::kMaxProbability);
            }
        }
    }
    return map_occ_grid;
}

cv::Point2d GetRotatePoint(cv::Mat srcImage, cv::Point2d Points, const cv::Point2d rotate_center, const double angle) {
	
	int x1 = 0, y1 = 0;
	int row = srcImage.rows;
    x1 = Points.x;
    y1 = row - Points.y;
    int x2 = rotate_center.x;
    int y2 = row - rotate_center.y;
    int x = (x1 - x2)*cos(angle) - (y1 - y2)*sin(angle) + x2;
    int y = (x1 - x2)*sin(angle) + (y1 - y2)*cos(angle) + y2;
    y = row - y;
	cv::Point2d dstPoints(x, y);	
	return dstPoints;
}

void RotateImage(const double& offset_o,
                 cv::Point2i& rotate_center,
                 cv::Mat& img){
    
    cv::Point2d edge1(0,0),edge2(0,img.rows),edge3(img.cols,0),edge4(img.cols,img.rows);
    edge1 = GetRotatePoint(img, edge1, rotate_center, offset_o);
    edge2 = GetRotatePoint(img, edge2, rotate_center, offset_o);
    edge3 = GetRotatePoint(img, edge3, rotate_center, offset_o);
    edge4 = GetRotatePoint(img, edge4, rotate_center, offset_o);
    double min_x = edge1.x,max_x = edge1.x,min_y= edge1.y,max_y= edge1.y;
    min_x = min_x > edge2.x ? edge2.x : min_x;
    max_x = max_x < edge2.x ? edge2.x : max_x;
    min_y = min_y > edge2.y ? edge2.y : min_y;
    max_y = max_y < edge2.y ? edge2.y : max_y;

    min_x = min_x > edge3.x ? edge3.x : min_x;
    max_x = max_x < edge3.x ? edge3.x : max_x;
    min_y = min_y > edge3.y ? edge3.y : min_y;
    max_y = max_y < edge3.y ? edge3.y : max_y;

    min_x = min_x > edge4.x ? edge4.x : min_x;
    max_x = max_x < edge4.x ? edge4.x : max_x;
    min_y = min_y > edge4.y ? edge4.y : min_y;
    max_y = max_y < edge4.y ? edge4.y : max_y;
    
    min_x = min_x < 0 ? -min_x + 5 : 0;
    min_y = min_y < 0 ? -min_y + 5 : 0;
    max_x = max_x > img.cols ? max_x - img.cols + 5 :0;
    max_y = max_y > img.rows ? max_y - img.rows + 5 :0;
    cv::Mat dst;
    cv::Scalar borderColor = cv::Scalar(1.0);
    cv::copyMakeBorder(img, dst, min_y, max_y, min_x, max_x, cv::BORDER_CONSTANT, borderColor);
    rotate_center.x = rotate_center.x + min_x;
    rotate_center.y = rotate_center.y + min_y;    
    cv::Mat rot_mat = cv::getRotationMatrix2D(rotate_center, common::kRadToDeg * offset_o, 1.0);    
    cv::Size dst_sz(dst.cols, dst.rows);
    cv::warpAffine(dst,img,rot_mat, dst_sz, cv::INTER_NEAREST, cv::BORDER_CONSTANT, borderColor);
}


cv::Mat GetSumedMapOdds(
        const std::vector<std::shared_ptr<common::Submap2D>>& submaps,
        const common::KeyFrames& key_frames,
        std::vector<SubMapOddsInfo>& submap_odds_infos,
        common::MapLimits& map_limits,
        Eigen::AlignedBox2d& map_boundary) {

    const double resolution = submaps.front()->grid()->limits().resolution();

    for (size_t i = 0u; i < submaps.size(); ++i) {
        const auto& submap = submaps[i];
        CHECK(submap->insertion_finished());
        const auto& grid = submap->grid();
        
        aslam::Transformation last_global_pose = submap_odds_infos[i].global_pose;
        aslam::Transformation current_global_pose = submap_odds_infos[i].global_pose;
        if (!key_frames.empty() && i < key_frames.size()) {
            current_global_pose = key_frames[i].state.T_OtoG;
        }
        aslam::Transformation delta_pose = current_global_pose * last_global_pose.inverse();

        if (submap_odds_infos[i].odds.empty() ||
            common::QuatToEuler(delta_pose.getEigenQuaternion())(2) != 0 ||
             delta_pose.getPosition().head<2>().norm() > resolution / 2 ) {
            aslam::Transformation delta_pose_to_local = 
                current_global_pose * submap->local_pose().inverse();
            cv::Mat submap_odds = submap->submap_odds().clone();
            Eigen::Array2i center = grid->limits().GetCellIndex(
                    submap->local_pose().getPosition().head<2>());
            cv::Point2i rotate_center(center.x(), center.y());
            double rotate_theta = common::QuatToEuler(delta_pose_to_local.getEigenQuaternion())(2);
            if (rotate_theta != 0) {
                RotateImage(rotate_theta, rotate_center, submap_odds);
            }

            Eigen::Vector2d submap_max = Eigen::Vector2d(current_global_pose.getPosition().head<2>()) +
                                            Eigen::Vector2d(rotate_center.y * resolution + 
                                                        0.5 * resolution ,
                                                        rotate_center.x * resolution + 
                                                        0.5 * resolution);
            Eigen::Vector2d submap_min = submap_max -  
                                        Eigen::Vector2d(submap_odds.rows * grid->limits().resolution() + 
                                                        0.5 * grid->limits().resolution() ,
                                                        submap_odds.cols * grid->limits().resolution() + 
                                                        0.5 * grid->limits().resolution());
            submap_odds_infos[i] = {submap_odds, current_global_pose, submap_max, submap_min};
        }
        map_boundary.extend(submap_odds_infos[i].max + Eigen::Vector2d::Ones() * resolution);
        map_boundary.extend(submap_odds_infos[i].min - Eigen::Vector2d::Ones() * resolution);
    }

    const Eigen::Vector2d map_range =
                map_boundary.max() - map_boundary.min();
    const int num_x_cell =
            std::ceil(map_range(1) / resolution) + 1;
    const int num_y_cell =
            std::ceil(map_range(0) / resolution) + 1;
    map_limits = common::MapLimits(resolution, map_boundary.max(),
            common::CellLimits(num_x_cell, num_y_cell));
    cv::Size map_size = cv::Size(num_x_cell, num_y_cell);
    cv::Mat map_odds(map_size, CV_32FC1, cv::Scalar(1.));
    // Merge the occupancy probability of different submaps.
    for (size_t i = 0u; i < submap_odds_infos.size(); ++i) {
        if (submap_odds_infos[i].odds.empty()) {
            continue;
        }
        const Eigen::Array2i pos_pixel =
                map_limits.GetCellIndex(submap_odds_infos[i].max);
        cv::Mat map_rect = map_odds(cv::Rect(pos_pixel(0), pos_pixel(1),
                                    submap_odds_infos[i].odds.cols, 
                                    submap_odds_infos[i].odds.rows));
        map_rect = map_rect.mul(submap_odds_infos[i].odds);
    }
    return map_odds;
}

void ScanLoopInterface::AddFinishedSubMap(
        const std::shared_ptr<common::Submap2D>& sub_map_ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    submaps_.emplace_back(sub_map_ptr);
    have_new_finished_map_ = true;
    submap_odds_infos_.emplace_back(SubMapOddsInfo());
    submap_odds_infos_.back().odds = cv::Mat();
    submap_odds_infos_.back().global_pose = sub_map_ptr->local_pose();
    scan_matchers_.push_back(std::make_shared<common::FastCorrelativeScanMatcher>(
        *(sub_map_ptr->grid()),
        branch_and_bound_depth_));
}

void ScanLoopInterface::ClearFinishedSubmaps() {
    submaps_.clear();
    scan_matchers_.clear();
}

void ScanLoopInterface::UpdateSubmapOrigin(
    const aslam::Transformation& origin,
    const size_t submap_idx) {
    CHECK_LT(submap_idx, submaps_.size());
    submaps_[submap_idx]->update_local_pose(origin);
}

aslam::Transformation ScanLoopInterface::SubmapOrigin(const size_t submap_idx) const {
    CHECK_LT(submap_idx, submaps_.size());
    return submaps_[submap_idx]->local_pose();
}

void ScanLoopInterface::CollectMap(
        const std::vector<std::shared_ptr<common::Submap2D>>& submaps,
        const common::KeyFrames& key_frames,
        cv::Mat* map_ptr,
        common::MapLimits* map_limits_ptr,
        Eigen::Vector2d* origin_ptr) {
    cv::Mat& map = *CHECK_NOTNULL(map_ptr);
    common::MapLimits& map_limit = *CHECK_NOTNULL(map_limits_ptr);
    Eigen::Vector2d& origin = *CHECK_NOTNULL(origin_ptr);
    CHECK(!submaps.empty());
    cv::Mat map_32f = CollectLocalSubmaps(submaps, key_frames, &origin);
    map_32f.copyTo(map);
    map_limit = GetMapLimits(submaps);
}

common::MapLimits ScanLoopInterface::GetMapLimits(
        const std::vector<std::shared_ptr<common::Submap2D>>& submaps) {
    CHECK(!submaps.empty());
    Eigen::AlignedBox2d map_boundary = GetMapBoundary(submaps);
    const double resolution =
            submaps.front()->grid()->limits().resolution();

    const Eigen::Vector2d map_range =
            map_boundary.max() - map_boundary.min();
    const int num_x_cell = std::ceil(map_range(1) / resolution);
    const int num_y_cell = std::ceil(map_range(0) / resolution);

    common::MapLimits map_limit(resolution, map_boundary.max(),
                                common::CellLimits(num_x_cell, num_y_cell));

    return map_limit;
}

common::KeyFrames ScanLoopInterface::CreateVirtualKeyframes(
        const common::KeyFrames& range_datas,
        const std::unordered_map<int, size_t>& keyframe_id_to_idx) {
    common::KeyFrames virtual_keyframes;
    for (const auto& submap : submaps_) {
        common::KeyFrame virtual_keyframe = common::CreateVirtualKeyframe(
                submap,
                range_datas,
                keyframe_id_to_idx);
        virtual_keyframes.push_back(virtual_keyframe);
    }
    return virtual_keyframes;
}

cv::Mat ScanLoopInterface::CollectLocalSubmaps(
        const std::vector<std::shared_ptr<common::Submap2D>>& active_submaps,
        const common::KeyFrames& key_frames,
        Eigen::Vector2d* origin_ptr) {

    Eigen::Vector2d& origin = *CHECK_NOTNULL(origin_ptr);
    std::vector<std::shared_ptr<common::Submap2D>> finished_submaps;
    std::vector<std::shared_ptr<common::Submap2D>> unfinished_submaps;
    // The first submap in "active_submaps" may have been finished.

    if (active_submaps.empty()) {
        return cv::Mat();
    }

    unfinished_submaps.emplace_back(active_submaps.front());

    if (!key_frames.empty()) {
        // It means that we have finished submaps in online posegraph.
        finished_submaps.clear();
        finished_submaps.assign(submaps_.begin(), submaps_.end());
    }

    if (finished_submaps.empty() && unfinished_submaps.empty()) {
        return cv::Mat();
    }
    
    const double resolution =
            active_submaps.front()->grid()->limits().resolution();
    Eigen::AlignedBox2d map_boundary_all;

    auto get_finished_map_odds_helper =
            [&](const std::vector<std::shared_ptr<common::Submap2D>>& submaps,
                const common::KeyFrames& key_frames,
                common::MapLimits* map_limit_ptr) {
        if (submaps.empty()) {
            return cv::Mat();
        }
        common::MapLimits& map_limit = *CHECK_NOTNULL(map_limit_ptr);
        cv::Mat map_odds = GetSumedMapOdds(
                submaps,
                key_frames,
                submap_odds_infos_,
                map_limit,
                map_boundary_all);
        return map_odds;
    };

    // Collect finished submap odds.
    if (have_new_finished_map_ && !finished_submaps.empty()) {
        finished_map_odds_ = get_finished_map_odds_helper(
            finished_submaps, key_frames, &finished_map_limit_);
        have_new_finished_map_ = false;
    } else {
        Eigen::Vector2d finished_map_min = finished_map_limit_.max() - 
                                                Eigen::Vector2d(finished_map_limit_.cell_limits().num_y_cells * resolution + 
                                                                    0.5 * resolution ,
                                                                finished_map_limit_.cell_limits().num_x_cells * resolution + 
                                                                    0.5 * resolution);
        map_boundary_all.extend(finished_map_limit_.max() + Eigen::Vector2d::Ones() * resolution);
        map_boundary_all.extend(finished_map_min - Eigen::Vector2d::Ones() * resolution);
    }
    std::vector<std::pair<cv::Mat, Eigen::Vector2d>> 
            unfinished_submap_infos(unfinished_submaps.size());
    if (!unfinished_submaps.empty()) {
        // Collect unfinished submap odds.
        for (size_t i = 0u; i < unfinished_submaps.size(); ++i) {
            const auto& submap = unfinished_submaps[i];
            std::lock_guard<std::mutex> lck(submap->finish_mutex_);
            const auto grid = submap->grid();
            Eigen::Array2i offset;
            common::CellLimits cell_limit;
            grid->ComputeCroppedLimits(&offset, &cell_limit);

            const Eigen::Vector2d max = grid->limits().max() -
                    resolution * Eigen::Vector2d(offset.y(), offset.x());
            const Eigen::Vector2d min = max - Eigen::Vector2d(cell_limit.num_y_cells * resolution + 
                                                                    0.5 * resolution ,
                                                                cell_limit.num_x_cells * resolution + 
                                                                    0.5 * resolution);

            map_boundary_all.extend(max + Eigen::Vector2d::Ones() * resolution);
            map_boundary_all.extend(min - Eigen::Vector2d::Ones() * resolution) ;
            
            cv::Mat submap_odds(cell_limit.num_y_cells /*height*/,
                                cell_limit.num_x_cells /*width*/,
                                CV_32FC1,
                                cv::Scalar(1.));
            for (const Eigen::Array2i& xy_index :
                    common::XYIndexRangeIterator(cell_limit)) {
                if (grid->IsKnown(xy_index + offset)) {
                    const float probability = grid->GetValue(xy_index + offset);
                    submap_odds.at<float>(xy_index(1) /*y*/,
                                          xy_index(0) /*x*/) = common::Odds(probability);
                }
            }
            unfinished_submap_infos[i] = {submap_odds, max};
        }
    }

    map_boundary_all.extend(map_boundary_all.max() + Eigen::Vector2d::Ones() * resolution);
    map_boundary_all.extend(map_boundary_all.min() - Eigen::Vector2d::Ones() * resolution);

    origin = Eigen::Vector2d(map_boundary_all.min().x(),
                             map_boundary_all.min().y());

    const Eigen::Vector2d map_range =
                map_boundary_all.max() - map_boundary_all.min();
    const int num_x_cell =
            std::ceil(map_range(1) / resolution);
    const int num_y_cell =
            std::ceil(map_range(0) / resolution);
    common::MapLimits map_limit(resolution, map_boundary_all.max(),
            common::CellLimits(num_x_cell, num_y_cell));

    cv::Mat map_odds(num_y_cell, num_x_cell,
            CV_32FC1, cv::Scalar(1.));
    if (!finished_submaps.empty()) {
        const Eigen::Array2i pos_pixel =
                map_limit.GetCellIndex(finished_map_limit_.max());
        finished_map_odds_.copyTo(map_odds(
                cv::Rect(pos_pixel(0), pos_pixel(1),
                        finished_map_odds_.cols, finished_map_odds_.rows)));
    }

    if (!unfinished_submaps.empty()) {
        // Merge unfinished submap odds into one.
        for (size_t i = 0u; i < unfinished_submaps.size(); ++i) {
            cv::Mat map_odds_tmp(num_y_cell, num_x_cell,
                    CV_32FC1, cv::Scalar(1.));
            const Eigen::Array2i pos_pixel =
                    map_limit.GetCellIndex(unfinished_submap_infos[i].second);
            const cv::Mat& submap_odds = unfinished_submap_infos[i].first;

            cv::Mat map_rect = map_odds(cv::Rect(pos_pixel(0), pos_pixel(1),
                                                submap_odds.cols, submap_odds.rows));
            map_rect = map_rect.mul(submap_odds);
        }
    }

    const cv::Mat map_probability = MapOdds2OccProbability(map_odds);

    if (kShowPoseGraphResult && !key_frames.empty()) {
        cv::Mat map_8u = cv::Mat(map_probability.rows, map_probability.cols, CV_8UC1);
        for (int i = 0; i < map_probability.rows; ++i) {
            for (int j = 0; j < map_probability.cols; ++j) {
                const float occ = map_probability.at<float>(i, j);
                if (occ < 0.) {
                    map_8u.at<uchar>(i, j) = 128;
                } else {
                    map_8u.at<uchar>(i, j) =
                        255 - common::ProbabilityToLogOddsInteger(occ);
                }
            }
        }

        cv::Mat keyframes_show;
        map_8u.copyTo(keyframes_show);
        cv::cvtColor(keyframes_show, keyframes_show, cv::COLOR_GRAY2BGR);
        Eigen::Array2i prev_cell_index;
        for (size_t i = 0u; i < key_frames.size(); ++i) {
            const Eigen::Vector2d cell_homo(key_frames[i].state.T_OtoG.getPosition()(0),
                                            key_frames[i].state.T_OtoG.getPosition()(1));
            const Eigen::Array2i cell_index = map_limit.GetCellIndex(cell_homo);
            const cv::Scalar color = (i == 0u) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            const int circle_size = (i == 0u) ? 4 : 2;
            cv::circle(keyframes_show, cv::Point2i(cell_index(0), cell_index(1)),
                    circle_size, color, cv::FILLED);
            if (i > 0) {
                cv::line(keyframes_show,
                         cv::Point2i(prev_cell_index(0), prev_cell_index(1)),
                         cv::Point2i(cell_index(0), cell_index(1)),
                         cv::Scalar(255, 0, 0));
            }
            prev_cell_index = cell_index;
        }

        cv::imshow("keyframes", keyframes_show);
        cv::waitKey(1);
    }

    return map_probability;
}

void ScanLoopInterface::DetectScanIntraLoop(
        const common::KeyFrames& keyframes,
        common::LoopResults* loop_results_ptr) {
    common::LoopResults& loop_results = *CHECK_NOTNULL(loop_results_ptr);

    common::MatchingOption option_intra;
    option_intra.linear_search_window_ = 0.5;
    option_intra.angular_search_window_ = common::kDegToRad * 3.0;
    option_intra.min_score_ = 0.6;

    std::function<void(const std::vector<size_t>&)> loop_detector_helper =
            [&](const std::vector<size_t>& range) {
        for (const size_t& job_index : range) {
            const auto& scan = keyframes[job_index];
            for (size_t i = 0u; i < submaps_.size(); ++i) {
                const int first_keyframe_id_in_submap =
                        *(submaps_[i]->scanid_in_submap().begin());
                const bool is_intra = submaps_[i]->HasScan(scan.keyframe_id);
                if (scan.keyframe_id != first_keyframe_id_in_submap &&
                     !scan.submap_id.empty() && is_intra) {
                    common::MatchingResult result =
                            scan_matchers_[i]->Match(scan, option_intra);
                    if (result.score < 0.) {
                        continue;
                    }
                    common::LoopResult loop_result(scan.state.timestamp_ns,
                                                   scan.keyframe_id,
                                                   first_keyframe_id_in_submap,
                                                   result.score,
                                                   loop_closure::LoopSensor::kScan,
                                                   loop_closure::VisualLoopType::kGlobal,
                                                   result.pose_estimate);
                    std::lock_guard<std::mutex> lock(mutex_);
                    loop_results.push_back(loop_result);
                }
            }
        }
    };

    const bool kParallelize = true;
    common::ParallelProcess(keyframes.size(),
                            loop_detector_helper,
                            kParallelize,
                            kNumThreads);
    VLOG(0) << "Detected: " << loop_results.size()
            << " intra loops before verification.";
}

void ScanLoopInterface::DetectScanInterLoop(
        const common::KeyFrames& keyframes,
        const std::unordered_map<int, size_t>& keyframe_id_to_idx,
        const double inter_distance_threshold,
        common::LoopResults* loop_results_ptr) {
    common::LoopResults& loop_results = *CHECK_NOTNULL(loop_results_ptr);

    common::KeyFrames virtual_keyframes = CreateVirtualKeyframes(
                keyframes, keyframe_id_to_idx);

    common::MatchingOption option_inter;
    option_inter.linear_search_window_ = 3.0;
    option_inter.angular_search_window_ = common::kDegToRad * 30.0;
    option_inter.min_score_ = 0.55;

    std::function<void(const std::vector<size_t>&)> loop_detector_helper =
            [&](const std::vector<size_t>& range) {
        for (size_t idx = 0u; idx < range.size(); idx++) {
            const auto &virtual_keyframe = virtual_keyframes[range[idx]];
            for (size_t i = 0u; i < submaps_.size(); ++i) {
                const int first_keyframe_id_insubmap =
                        *(submaps_[i]->scanid_in_submap().begin());
                const bool is_intra = submaps_[i]->HasScan(virtual_keyframe.keyframe_id);  
                double keyframe_submap_distance =  
                                    (virtual_keyframe.state.T_OtoG.getPosition() - 
                                    submaps_[i]->local_pose().getPosition()).norm();   
                if (std::abs(virtual_keyframe.keyframe_id - first_keyframe_id_insubmap) >
                     kLoopDetectionInterval && !is_intra &&
                     keyframe_submap_distance < inter_distance_threshold) {
                    common::MatchingResult result =
                            scan_matchers_[i]->Match(virtual_keyframe, option_inter);
                    if (result.score < 0.) {
                        continue;
                    }
                    common::LoopResult loop_result(virtual_keyframe.state.timestamp_ns,
                                                   virtual_keyframe.keyframe_id,
                                                   first_keyframe_id_insubmap,
                                                   result.score,
                                                   loop_closure::LoopSensor::kScan,
                                                   loop_closure::VisualLoopType::kGlobal,
                                                   result.pose_estimate);
                    std::lock_guard<std::mutex> lock(mutex_);
                    loop_results.push_back(loop_result);
                }
            }
        }
    };

    const bool kParallelize = true;
    common::ParallelProcess(virtual_keyframes.size(),
                            loop_detector_helper,
                            kParallelize,
                            kNumThreads);
    VLOG(0) << "Detected: " << loop_results.size()
            << " intra & inter loops before verification.";
}

void ScanLoopInterface::DetectScanInterLoopOnline(
        const common::KeyFrame& keyframe,
        const common::KeyFrames& keyframes_virtual,
        const double inter_distance_threshold,
        common::LoopResults* loop_results_ptr) {
    common::LoopResults& loop_results = *CHECK_NOTNULL(loop_results_ptr);

    common::MatchingOption option_inter;
    option_inter.linear_search_window_ = 3.0;
    option_inter.angular_search_window_ = common::kDegToRad * 30.0;
    option_inter.min_score_ = 0.5;

    for (size_t i = 0u; i < submaps_.size(); ++i) {
        const int first_keyframe_id_insubmap =
            *(submaps_[i]->scanid_in_submap().begin());
        double keyframe_submap_distance =  
            (keyframe.state.T_OtoG.getPosition() - 
                keyframes_virtual[i].state.T_OtoG.getPosition()).norm();
        if (std::abs(keyframe.keyframe_id - first_keyframe_id_insubmap) >
            kLoopDetectionInterval &&
                keyframe_submap_distance < inter_distance_threshold) {
            std::unique_lock<std::mutex> lock(mutex_);
            CHECK_NOTNULL(scan_matchers_[i]);
            common::MatchingResult result =
                scan_matchers_[i]->Match(keyframe, option_inter);
            if (result.score < 0.) {
                continue;
            }
            common::LoopResult loop_result(keyframe.state.timestamp_ns,
                                           keyframe.keyframe_id,
                                           first_keyframe_id_insubmap,
                                           result.score,
                                           loop_closure::LoopSensor::kScan,
                                           loop_closure::VisualLoopType::kGlobal,
                                           result.pose_estimate);

            loop_results.push_back(loop_result);
        }
    }    
}

void ScanLoopInterface::InitSubmapFastScanMatchers() {

    scan_matchers_.resize(submaps_.size());

    std::function<void(const std::vector<size_t>&)> scan_matcher_initial_helper =
            [&](const std::vector<size_t>& range) {
        for (size_t idx = 0u; idx < range.size(); idx++) {
            const auto &submap = submaps_[range[idx]];
            scan_matchers_[range[idx]] = 
                std::make_shared<common::FastCorrelativeScanMatcher>(
                                *(submap->grid()),
                                branch_and_bound_depth_);
            
        }
    };
    const bool kParallelize = true;

    common::ParallelProcess(submaps_.size(),
                            scan_matcher_initial_helper,
                            kParallelize,
                            kNumThreads);
    VLOG(0) << "Init fast scan matcher finished.";
}

void ScanLoopInterface::VerifyScanLoop(
        const common::KeyFrames& keyframes,
        const std::unordered_map<int, size_t>& keyframe_id_to_idx,
        common::LoopResults* loop_results_ptr) {
    common::LoopResults& loop_results = *CHECK_NOTNULL(loop_results_ptr);

    std::map<std::pair<int, int>, size_t> map_loop_index;
    for (size_t i = 0u; i < loop_results.size(); ++i) {
        const auto& loop = loop_results[i];
        map_loop_index.insert(std::make_pair(std::make_pair(
                loop.keyframe_id_query, loop.keyframe_id_result), i));
    }

    std::vector<bool> is_valid(loop_results.size(), true);
    for (size_t i = 0u; i < loop_results.size(); ++i) {
        const auto& loop = loop_results[i];
        CHECK_NE(loop.loop_sensor, loop_closure::LoopSensor::kInvalid);
        if (loop.loop_sensor == loop_closure::LoopSensor::kVisual) {
                continue;
        }
        const int keyframe_id_query = loop.keyframe_id_query;
        const int keyframe_id_result = loop.keyframe_id_result;
        const auto& iter_query = keyframe_id_to_idx.find(keyframe_id_query);
        CHECK(iter_query != keyframe_id_to_idx.end());
        const auto& iter_result = keyframe_id_to_idx.find(keyframe_id_result);
        CHECK(iter_result != keyframe_id_to_idx.end());
        bool flag_erase = false;
        if (std::abs(keyframe_id_query - keyframe_id_result) <=
                kLoopDetectionInterval) {
            const aslam::Transformation T_diff =
                    keyframes[iter_query->second].state.T_OtoG.inverse() *
                    loop.T_estimate;
            const Eigen::Vector3d& p = T_diff.getPosition();
            const Eigen::Quaterniond& q = T_diff.getEigenQuaternion();
            const double trans_norm = p.norm();
            Eigen::Vector3d euler = common::QuatToEuler(q);
            const double euler_norm = euler.norm();
            if (trans_norm > kTransThreshold ||
                    euler_norm > common::kDegToRad * kRotThreshold) {
                flag_erase = true;
            }
        } else {
            CHECK(!keyframes[iter_query->second].submap_id.empty());
            const int latest_submap_id =
                    *(keyframes[iter_query->second].submap_id.rbegin());
            const int latest_submap_keyframe_id =
                    *(submaps_[latest_submap_id]->scanid_in_submap().begin());
            const auto& iter_loop_index = map_loop_index.find(
                        std::make_pair(keyframe_id_result, latest_submap_keyframe_id));
            if (iter_loop_index == map_loop_index.end()) {
                flag_erase = true;
            } else {
                const auto& iter_latest_submap = keyframe_id_to_idx.find(
                            latest_submap_keyframe_id);
                CHECK(iter_latest_submap != keyframe_id_to_idx.end());
                const aslam::Transformation delta_T_real =
                        keyframes[iter_latest_submap->second].state.T_OtoG.inverse() *
                        keyframes[iter_query->second].state.T_OtoG;
                const size_t target_loop_idx = iter_loop_index->second;
                const aslam::Transformation delta_T_est =
                        (keyframes[iter_latest_submap->second].state.T_OtoG.inverse() *
                        loop_results[target_loop_idx].T_estimate) *
                        (keyframes[iter_result->second].state.T_OtoG.inverse() *
                        loop.T_estimate);
                const aslam::Transformation T_diff = delta_T_real.inverse() * delta_T_est;
                const Eigen::Vector3d& p = T_diff.getPosition();
                const Eigen::Quaterniond& q = T_diff.getEigenQuaternion();
                const double trans_norm = p.norm();
                Eigen::Vector3d euler = common::QuatToEuler(q);
                const double euler_norm = euler.norm();
                if (trans_norm > 2.0 * kTransThreshold ||
                        euler_norm > 2.0 * common::kDegToRad * kRotThreshold) {
                    flag_erase = true;
                }
            }
        }
        if (flag_erase) {
            is_valid[i] = false;
        }
    }

    common::ReduceDeque(is_valid, &loop_results);

    VLOG(0) << loop_results.size() << " loops retained after verification.";
}
}
