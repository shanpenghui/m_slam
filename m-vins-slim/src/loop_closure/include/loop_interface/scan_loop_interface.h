#ifndef LOOP_CLOSURE_SCAN_LOOP_INTERFACE_H_
#define LOOP_CLOSURE_SCAN_LOOP_INTERFACE_H_

#include "cfg_common/slam_config.h"
#include "occ_common/fast_correlative_scan_matcher.h"
#include "occ_common/live_submaps.h"
#include "occ_common/submap_2d.h"

namespace loop_closure {

struct SubMapOddsInfo {
    cv::Mat odds;
    aslam::Transformation global_pose;
    Eigen::Vector2d max;
    Eigen::Vector2d min;
};
class ScanLoopInterface {
public:
    explicit ScanLoopInterface() = default;
    ~ScanLoopInterface() = default;
    void AddFinishedSubMap(
            const std::shared_ptr<common::Submap2D>& sub_map_ptr);
    void ClearFinishedSubmaps();
    void UpdateSubmapOrigin(const aslam::Transformation& origin,
                            const size_t submap_idx);
    aslam::Transformation SubmapOrigin(const size_t submap_idx) const;
    void CollectMap(
            const std::vector<std::shared_ptr<common::Submap2D>>& submaps,
            const common::KeyFrames& key_frames,
            cv::Mat* map_ptr,
            common::MapLimits* map_limits_ptr,
            Eigen::Vector2d* origin_ptr);
    common::KeyFrames CreateVirtualKeyframes(
            const common::KeyFrames& range_datas,
            const std::unordered_map<int, size_t>& keyframe_id_to_idx);
    cv::Mat CollectLocalSubmaps(
            const std::vector<std::shared_ptr<common::Submap2D>>& active_submaps,
            const common::KeyFrames& key_frames,
            Eigen::Vector2d* origin_ptr);
    void DetectScanIntraLoop(const common::KeyFrames& keyframes,
                             common::LoopResults* loop_results_ptr);
    void DetectScanInterLoop(const common::KeyFrames& keyframes,
                             const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                             const double inter_distance_threshold,
                             common::LoopResults* loop_results_ptr);
    void DetectScanInterLoopOnline(const common::KeyFrame& keyframe,
                                   const common::KeyFrames& keyframes_virtual,
                                   const double inter_distance_threshold,
                                   common::LoopResults* loop_results_ptr);
    void InitSubmapFastScanMatchers();
    void VerifyScanLoop(const common::KeyFrames& keyframes,
                        const std::unordered_map<int, size_t>& keyframe_id_to_idx,
                        common::LoopResults* loop_results_ptr);
private:
    common::MapLimits GetMapLimits(const std::vector<std::shared_ptr<common::Submap2D>>& submaps);

    std::vector<std::shared_ptr<common::Submap2D>> submaps_;

    std::vector<std::shared_ptr<common::FastCorrelativeScanMatcher>> scan_matchers_;

    const int branch_and_bound_depth_ = 7;
    
    bool have_new_finished_map_;

    common::MapLimits finished_map_limit_;
    cv::Mat finished_map_odds_;

    std::vector<SubMapOddsInfo> submap_odds_infos_;

    std::mutex mutex_;
};

}
#endif
