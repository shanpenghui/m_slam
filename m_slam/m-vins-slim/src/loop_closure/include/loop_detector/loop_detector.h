#ifndef LOOP_CLOSURE_LOOP_DETECTOR_H_
#define LOOP_CLOSURE_LOOP_DETECTOR_H_

#include "loop_index/index_interface.h"

#include <aslam/common/reader-writer-lock.h>

#include "data_common/visual_structures.h"
#include "loop_detector/scoring.h"
#include "loop_interface/loop_settings.h"
#include "flann_common/nano_flann.hpp"
#include "flann_common/pointcloud.h"
#include "summary_map/summary_map.h"

namespace loop_closure {
struct InlierIndexWithReprojectionError {
  InlierIndexWithReprojectionError() = delete;
  InlierIndexWithReprojectionError(
      const int inlier_index, const double reprojection_error)
      : inlier_index_(inlier_index), reprojection_error_(reprojection_error) {}
  virtual ~InlierIndexWithReprojectionError() = default;
  inline int GetInlierIndex() const {
    return inlier_index_;
  }
  inline void SetInlierIndex(const int inlier_index) {
    inlier_index_ = inlier_index;
  }
  inline double GetReprojectionError() const {
    return reprojection_error_;
  }
  inline void SetReprojectionError(const double reprojection_error) {
    reprojection_error_ = reprojection_error;
  }

 private:
  int inlier_index_;
  double reprojection_error_;
};
typedef std::unordered_map<loop_closure::KeypointIdentifier,
                            InlierIndexWithReprojectionError>
    KeypointToInlierIndexWithReprojectionErrorMap;

void GetBestStructureMatchForEveryKeypoint(
    const std::vector<int>& inliers,
    const std::vector<double>& inlier_distances_to_model,
    const loop_closure::VertexKeyPointToStructureMatchList& structure_matches,
    const common::VisualFrameDataPtrVec& frame_datas,
    KeypointToInlierIndexWithReprojectionErrorMap*
        keypoint_to_best_structure_match);

typedef std::vector<std::vector<std::vector<std::size_t>>> Grid;

struct MapTrackingResult {
    double min_dist;
    double second_min_dist;
    double reprojection_error;
    int query_keypoint_idx;
};

class LoopDetector {
public:
    explicit LoopDetector(const aslam::NCamera::ConstPtr& cameras,
                         std::shared_ptr<LoopSettings> settings,
                         std::shared_ptr<SummaryMap> summary_map);

    ~LoopDetector() = default;

    void SetSummaryMap(const std::shared_ptr<SummaryMap>& summary_map_ptr);

    void Insert(const common::VisualFrameDataPtr& frame_data,
                const Eigen::Vector3d& p_OinM,
                const Eigen::Vector3d& euler_OtoM);

    void BuildKdTree();

    void ProjectDescriptors(const common::DescriptorsMatUint8& raw_des,
                           common::DescriptorsMatF32* projected_des);

    bool FindNFrameInSummaryMapDatabase(
            const common::VisualFrameDataPtrVec& frame_datas,
            const aslam::Transformation& T_GtoM,
            common::LoopResult* result_ptr,
            loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr);

    bool FindFrameToFrameLoop(
            const common::VisualFrameDataPtrVec& frame_datas,
            const loop_closure::VertexKeyPointToStructureMatchList& inlier_structure_matches,
            common::LoopResult* result_ptr);

    void FindGlobally(
            const common::VisualFrameDataPtrVec& frame_datas,
            loop_closure::KeyFrameToFrameKeyPointMatches* frame_matches_list_ptr);

    void FindMapTracking(
            const aslam::Transformation& T_GtoM,
            const common::VisualFrameDataPtrVec& frame_datas,
            common::LoopResult* result_ptr);
private:
    void FindNearestNeighborMatchesForNFrame(
            const common::VisualFrameDataPtrVec& frame_datas,
            const aslam::Transformation& T_GtoM,
            common::LoopResult* result_ptr,
            loop_closure::KeyFrameToFrameKeyPointMatches* frame_matches_list_ptr,
            loop_closure::LandmarkIdListVec* query_vertex_observed_landmark_ids_ptr);

    bool ComputeAbsoluteTransformFromFrameMatches(
            const common::VisualFrameDataPtrVec& frame_datas,
            const loop_closure::LandmarkIdListVec& query_vertex_observed_landmark_ids,
            const loop_closure::KeyFrameToFrameKeyPointMatches& frame_to_matches,
            common::LoopResult* localization_result_ptr,
            loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr);

    bool ConvertFrameMatchesToConstraint(
            const loop_closure::KeyFrameToFrameKeyPointMatchesPair& query_frame_id_and_matches,
            loop_closure::LoopClosureConstraint* constraint_ptr) const;

    bool DoProjectedImagesBelongToSameVertex(
            const common::VisualFrameDataPtrVec& frame_datas);

    bool PnpCheck(
            const Eigen::Matrix2Xd& keypoints,
            const Eigen::Matrix3Xd& p_LinMs,
            const Eigen::VectorXi& cam_indices,
            std::vector<int>* inliers_ptr,
            std::vector<double>* inlier_distances_to_model_ptr,
            common::LoopResult* localization_result_ptr);

    bool HandleLoopClosure(
            const common::VisualFrameDataPtrVec& frame_datas,
            const loop_closure::LandmarkIdListVec& query_vertex_landmark_ids,
            const VertexKeyPointToStructureMatchList& structure_matches,
            int* num_inliers_ptr,
            loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr,
            common::LoopResult* localization_result_ptr,
            std::mutex* map_mutex_ptr);

    void CheckMatchingDistance(
            const Eigen::VectorXf& query_desriptor,
            const common::EigenVectorXfVec& projected_descriptors_candidate,
            int* best_match_idx_ptr,
            double* min_dist_ptr,
            double* second_min_dist_ptr);

    void CheckMatchingDistance(
            const Grid& grids,
            const double grid_width_inv,
            const double grid_height_inv,
            const int grid_cols,
            const int grid_rows,
            const Eigen::VectorXf& query_desriptor,
            const Eigen::Vector2d& query_keypoint,
            const common::EigenVector2dVec& reprojected_keypoint_candidate,
            const common::EigenVectorXfVec& projected_descriptors_candidate,
            int* best_match_idx_ptr,
            double* min_dist_ptr,
            double* second_min_dist_ptr);

    int GetKeyframeIdWithMostOverlappingLandmarks(
            const int keyframe_id_query,
            const loop_closure::VertexKeyPointToStructureMatchList& structure_matches,
            loop_closure::LandmarkIdList* overlap_landmarks_ptr,
            std::vector<int>* overlap_keypoints_ptr);

    int GetNumNeighborsToSearch() const;

    int NumDescriptors() const {
        return index_interface_->GetNumDescriptorsInIndex();
    }

    template <typename IdType>
    void DoCovisibilityFiltering(
        const loop_closure::IdToFrameKeyPointMatches<IdType>& id_to_matches,
        const bool make_matches_unique,
        loop_closure::KeyFrameToFrameKeyPointMatches* frame_matches,
        std::mutex* frame_matches_mutex = nullptr) const;

    template <typename IdType>
    using IdToScoreMap = std::map<IdType, scoring::ScoreType>;

    template <typename IdType>
    void ComputeRelevantIdsForFiltering(
        const loop_closure::IdToFrameKeyPointMatches<IdType>& frame_to_matches,
        IdToScoreMap<IdType>* frame_to_score_map) const;

    template <typename IdType>
    bool SkipMatch(
        const IdToScoreMap<IdType>& frame_to_score_map,
        const loop_closure::FrameKeyPointToStructureMatch& match) const;

    template <typename IdType>
    typename loop_closure::IdToFrameKeyPointMatches<IdType>::const_iterator
    GetIteratorForMatch(
        const loop_closure::IdToFrameKeyPointMatches<IdType>& frame_to_matches,
        const loop_closure::FrameKeyPointToStructureMatch& match) const;

    bool GetMatchForDescriptorIndex(
        int nn_match_descriptor_index, int keypoint_index_query,
        const common::VisualFrameDataPtr& frame_data_query,
        loop_closure::FrameKeyPointToStructureMatch* structure_match_ptr) const;

    // Typedef
    typedef std::unordered_map<loop_closure::KeyFrameId, common::VisualFrameDataPtr>
        Database;
    typedef loop_closure::IdToNumDescriptors<loop_closure::KeyFrameId>
        KeyframeIdToNumDescriptorsMap;
    typedef std::unordered_map<loop_closure::DescriptorIndex, loop_closure::KeyPointId>
        DescriptorIndexToKeypointIdMap;
    typedef loop_closure::IdToFrameKeyPointMatches<loop_closure::VertexId>
        VertexToMatchesMap;
    typedef std::unordered_map<loop_closure::KeyFrameId, Eigen::Vector3d>
        KeyframeIdToPositionMap;

    const aslam::NCamera::ConstPtr cameras_;

    std::shared_ptr<LoopSettings> settings_;

    std::shared_ptr<SummaryMap> summary_map_;

    std::shared_ptr<loop_closure::IndexInterface> index_interface_;

    scoring::computeScoresFunction<loop_closure::KeyFrameId>
        compute_keyframe_scores_;

    std::vector<aslam::Transformation> T_C0ToCi_;

    mutable aslam::ReaderWriterMutex read_write_mutex_;

    Database database_;

    nano_flann::PointCloud<double> vertex_positions_;
    nano_flann::PointCloud<double> vertex_orientations_;

    typedef nano_flann::KDTreeSingleIndexAdaptor<
        nano_flann::L2_Simple_Adaptor<double, nano_flann::PointCloud<double>>,
        nano_flann::PointCloud<double>,
        3> KdTree;
    std::unique_ptr<KdTree> position_kd_tree_;
    std::unique_ptr<KdTree> orientation_kd_tree_;

    FrameIdList frame_id_list_;

    int descriptor_index_provider_;

    KeyframeIdToNumDescriptorsMap keyframe_id_to_num_descriptors_;

    DescriptorIndexToKeypointIdMap descriptor_index_to_keypoint_id_;
};

}
#endif
