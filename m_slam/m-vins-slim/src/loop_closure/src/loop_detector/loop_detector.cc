#include "loop_detector/loop_detector.h"

#include "parallel_common/parallel_process.h"
#include "loop_detector/pnp.h"
#include "loop_index/kd_tree_index_interface.h"
#include "loop_index/inverted_multi_index_interface.h"
#include "math_common/math.h"
#include "time_common/time.h"
#include "time_common/time_table.h"
#include "summary_map/map_loader.h"

namespace loop_closure {
void GetBestStructureMatchForEveryKeypoint(
    const std::vector<int>& inliers,
    const std::vector<double>& inlier_distances_to_model,
    const loop_closure::VertexKeyPointToStructureMatchList& structure_matches,
    const common::VisualFrameDataPtrVec& frame_datas,
    KeypointToInlierIndexWithReprojectionErrorMap*
        keypoint_to_best_structure_match_ptr) {
    KeypointToInlierIndexWithReprojectionErrorMap& keypoint_to_best_structure_match =
            *CHECK_NOTNULL(keypoint_to_best_structure_match_ptr);
    keypoint_to_best_structure_match.clear();
    CHECK_EQ(inliers.size(), inlier_distances_to_model.size());

    loop_closure::VertexId invalid_vertex_id = -1;
    for (size_t inlier_idx = 0u; inlier_idx < inliers.size(); ++inlier_idx) {
        const int inlier_index = inliers[inlier_idx];
        const double reprojection_error = inlier_distances_to_model[inlier_idx];
        const int frame_index = structure_matches[inlier_index].frame_index_query;
        CHECK_LT(frame_index, static_cast<int>(frame_datas.size()));
        const int keypoint_index = structure_matches[inlier_index].keypoint_index_query;
        CHECK_LT(keypoint_index, frame_datas[frame_index]->key_points.cols());
        const loop_closure::KeypointIdentifier keypoint_identifier(
            invalid_vertex_id, frame_index, keypoint_index);

        KeypointToInlierIndexWithReprojectionErrorMap::iterator best_match_iterator =
                keypoint_to_best_structure_match.find(keypoint_identifier);
        if (best_match_iterator == keypoint_to_best_structure_match.end()) {
            InlierIndexWithReprojectionError inlier_index_with_reprojection_error(
                inlier_index, reprojection_error);
            CHECK(keypoint_to_best_structure_match.emplace(
                       keypoint_identifier, inlier_index_with_reprojection_error).second);
        } else if (best_match_iterator->second.GetReprojectionError() >
                   reprojection_error) {
            // Replace by new best structure-match.
            best_match_iterator->second.SetReprojectionError(reprojection_error);
            best_match_iterator->second.SetInlierIndex(inlier_index);
        }
    }
}

LoopDetector::LoopDetector(const aslam::NCamera::ConstPtr& cameras,
                           std::shared_ptr<LoopSettings> settings,
                           std::shared_ptr<SummaryMap> summary_map)
    : cameras_(cameras),
      settings_(settings),
      summary_map_(summary_map),
      descriptor_index_provider_(0) {
    index_interface_.reset(new loop_closure::InvertedMultiIndexInterface(
                               settings_->projected_quantizer_filename,
                               settings_->num_closest_words_for_nn_search));

    const size_t cam_size = cameras_->getNumCameras();
    T_C0ToCi_.resize(cam_size);
    for (size_t cam_idx = 0u; cam_idx < cam_size; ++cam_idx) {
        T_C0ToCi_[cam_idx] =
                cameras_->get_T_BtoC(cam_idx) * cameras_->get_T_BtoC(0).inverse();
    }

    switch (settings_->keyframe_scoring_function_type) {
        case LoopSettings::KeyframeScoringFunctionType::kAccumulation: {
            compute_keyframe_scores_ = &scoring::ComputeAccumulationScore<
                    loop_closure::KeyFrameId>;
            break;
        }
        case LoopSettings::KeyframeScoringFunctionType::kProbabilistic: {
            compute_keyframe_scores_ = &scoring::ComputeProbabilisticScore<
                    loop_closure::KeyFrameId>;
            break;
        }
        default: {
            LOG(FATAL) << "Invalid selection ("
                        << settings_->scoring_function_type_string
                        << ") for scoring function.";
            break;
        }
    }
}

void LoopDetector::SetSummaryMap(const std::shared_ptr<SummaryMap>& summary_map_ptr) {
    summary_map_ = summary_map_ptr;
}

void LoopDetector::Insert(const common::VisualFrameDataPtr& frame_data,
                          const Eigen::Vector3d& p_OinM,
                          const Eigen::Vector3d& euler_OtoM) {
    for (int keypoint_idx = 0; keypoint_idx < frame_data->projected_descriptors.cols(); ++keypoint_idx) {
        // NOTE: we will emplace this important message here.
        CHECK(descriptor_index_to_keypoint_id_.emplace(
               std::piecewise_construct, std::forward_as_tuple(descriptor_index_provider_++),
               std::forward_as_tuple(frame_data->frame_id_and_idx, keypoint_idx)).second);
    }
    index_interface_->AddDescriptors(frame_data->projected_descriptors);

    const loop_closure::KeyFrameId& frame_id = frame_data->frame_id_and_idx;
    CHECK(frame_id.IsValid());
    CHECK(keyframe_id_to_num_descriptors_.emplace(
            frame_id, frame_data->projected_descriptors.cols()).second);
    // Remove the frame data before adding the projected image to the database.
    common::VisualFrameDataPtr frame_data_copy(
            new common::VisualFrameData(*frame_data, false/*copy_image*/));

    CHECK(database_.emplace(frame_data->frame_id_and_idx, frame_data_copy).second)
            << "Duplicate projected image in database.";

    vertex_positions_.points.emplace_back(p_OinM(0),
                                          p_OinM(1),
                                          p_OinM(2));
    vertex_orientations_.points.emplace_back(euler_OtoM(0),
                                             euler_OtoM(1),
                                             euler_OtoM(2));

    frame_id_list_.push_back(frame_data_copy->frame_id_and_idx);
}

void LoopDetector::BuildKdTree() {
    position_kd_tree_.reset(new KdTree(3,
                              vertex_positions_,
                              nano_flann::KDTreeSingleIndexAdaptorParams(10)));
    position_kd_tree_->buildIndex();
    orientation_kd_tree_.reset(new KdTree(3,
                                vertex_orientations_,
                                nano_flann::KDTreeSingleIndexAdaptorParams(10)));
    orientation_kd_tree_->buildIndex();
}

void LoopDetector::ProjectDescriptors(const common::DescriptorsMatUint8& raw_des,
                                     common::DescriptorsMatF32* projected_des) {
    index_interface_->ProjectDescriptors(raw_des, projected_des);
}

bool LoopDetector::FindNFrameInSummaryMapDatabase(
        const common::VisualFrameDataPtrVec& frame_datas,
        const aslam::Transformation& T_GtoM,
        common::LoopResult* result_ptr,
        loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr) {
    common::LoopResult& result = *CHECK_NOTNULL(result_ptr);

    loop_closure::KeyFrameToFrameKeyPointMatches frame_matches_list;
    loop_closure::LandmarkIdListVec query_vertex_observed_landmark_ids;

    FindNearestNeighborMatchesForNFrame(frame_datas, T_GtoM, &result, &frame_matches_list,
                                        &query_vertex_observed_landmark_ids);
    bool success = false;
    if (result.visual_loop_type == loop_closure::VisualLoopType::kMapTracking) {
        constexpr bool kDoPnpCheckInMapTrackingMode = true;
        if (kDoPnpCheckInMapTrackingMode) {
            const Eigen::Matrix2Xd& keypoints = result.keypoints;
            const Eigen::Matrix3Xd& p_LinMs = result.positions;
            const Eigen::VectorXi& cam_indices = result.cam_indices;
            if (keypoints.cols() >= settings_->min_inlier_count) {
                std::vector<int> inliers;
                std::vector<double> inlier_distances_to_model;
                success = PnpCheck(keypoints, p_LinMs, cam_indices,
                                   &inliers, &inlier_distances_to_model, &result);
            } else {
                success = false;
            }
        } else {
            std::vector<std::pair<int, bool>>& pnp_inliers = result.pnp_inliers;
            pnp_inliers.resize(result.keypoints.cols());
            for (int i = 0; i < result.keypoints.cols(); ++i) {
                pnp_inliers[i] = std::make_pair(i, true);
            }
            success = true;
        }
    } else {
        success = ComputeAbsoluteTransformFromFrameMatches(
                    frame_datas, query_vertex_observed_landmark_ids,
                    frame_matches_list, &result, inlier_structure_matches_ptr);
    }

    return success;
}

void LoopDetector::FindNearestNeighborMatchesForNFrame(
        const common::VisualFrameDataPtrVec& frame_datas,
        const aslam::Transformation& T_GtoM,
        common::LoopResult* result_ptr,
        loop_closure::KeyFrameToFrameKeyPointMatches* frame_matches_list_ptr,
        loop_closure::LandmarkIdListVec* query_vertex_observed_landmark_ids_ptr) {
    common::LoopResult& result = *CHECK_NOTNULL(result_ptr);
    loop_closure::KeyFrameToFrameKeyPointMatches& frame_matches_list =
            *CHECK_NOTNULL(frame_matches_list_ptr);
    loop_closure::LandmarkIdListVec& query_vertex_observed_landmark_ids =
            *CHECK_NOTNULL(query_vertex_observed_landmark_ids_ptr);

    query_vertex_observed_landmark_ids.clear();

    const size_t num_frames = frame_datas.size();
    query_vertex_observed_landmark_ids.resize(num_frames);

    loop_closure::FrameIdList frame_ids;
    frame_ids.reserve(num_frames);

    for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
        const common::VisualFrameDataPtr& frame_data = frame_datas[frame_idx];
        if (frame_data->key_points.cols() == 0) {
            VLOG(2) << "No feature for relocalization in frame "
                    << frame_idx;
            continue;
        }
        frame_ids.push_back(frame_data->frame_id_and_idx);

        const Eigen::Matrix6Xd& keypoints = frame_data->key_points;
        query_vertex_observed_landmark_ids[frame_idx].resize(keypoints.cols());
        for (int i = 0; i < keypoints.cols(); ++i) {
            query_vertex_observed_landmark_ids[frame_idx][i] = frame_data->track_ids(i);
        }
    }
    if (result.visual_loop_type == loop_closure::VisualLoopType::kMapTracking) {
        TIME_TIC(LOOP_MAP_TRACKING);
        FindMapTracking(T_GtoM, frame_datas, &result);
        TIME_TOC(LOOP_MAP_TRACKING);
    } else {
        TIME_TIC(LOOP_GLOBAL_SEARCH);
        FindGlobally(frame_datas, &frame_matches_list);
        TIME_TOC(LOOP_GLOBAL_SEARCH);
    }
}

bool LoopDetector::ComputeAbsoluteTransformFromFrameMatches(
    const common::VisualFrameDataPtrVec& frame_datas,
    const loop_closure::LandmarkIdListVec& query_vertex_observed_landmark_ids,
    const loop_closure::KeyFrameToFrameKeyPointMatches& frame_to_matches,
    common::LoopResult* localization_result_ptr,
    loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr) {
    common::LoopResult& localization_result = *CHECK_NOTNULL(localization_result_ptr);

    const size_t num_matches = internal::GetNumberOfMatches(
        frame_to_matches);

    if (num_matches == 0u) {
        return false;
    }
    loop_closure::VertexId invalid_vertex_id = -1;
    loop_closure::LoopClosureConstraint constraint;
    constraint.query_vertex_id = invalid_vertex_id;

    for (const auto& frame_matches_pair : frame_to_matches) {
        loop_closure::LoopClosureConstraint tmp_constraint;

        const bool conversion_success = ConvertFrameMatchesToConstraint(
            frame_matches_pair, &tmp_constraint);

        if (!conversion_success) {
            continue;
        }

        constraint.query_vertex_id = tmp_constraint.query_vertex_id;
        constraint.structure_matches.insert(
            constraint.structure_matches.end(),
            tmp_constraint.structure_matches.begin(),
            tmp_constraint.structure_matches.end());
    }

    int num_inliers = 0;
    std::mutex map_mutex;
    bool success = HandleLoopClosure(frame_datas,
                                       query_vertex_observed_landmark_ids,
                                       constraint.structure_matches,
                                       &num_inliers,
                                       inlier_structure_matches_ptr,
                                       &localization_result,
                                       &map_mutex);

    return success;
}

bool LoopDetector::PnpCheck(
        const Eigen::Matrix2Xd& keypoints,
        const Eigen::Matrix3Xd& p_LinMs,
        const Eigen::VectorXi& cam_indices,
        std::vector<int>* inliers_ptr,
        std::vector<double>* inlier_distances_to_model_ptr,
        common::LoopResult* localization_result_ptr) {
    std::vector<int>& inliers = *CHECK_NOTNULL(inliers_ptr);
    std::vector<double>& inlier_distances_to_model =
            *CHECK_NOTNULL(inlier_distances_to_model_ptr);
    common::LoopResult& localization_result =
            *CHECK_NOTNULL(localization_result_ptr);

    int num_iters = 0;
    aslam::Transformation& T_OtoM_pnp = localization_result.T_estimate;
    RansacP3P(keypoints,
               cam_indices,
               p_LinMs,
               settings_->loop_closure_sigma_pixel,
               settings_->pnp_num_ransac_iters,
               cameras_,
               &T_OtoM_pnp,
               &inliers,
               &inlier_distances_to_model,
               &num_iters);

    // Optimize PnP based on the inliers obtained above.
    aslam::Transformation T_OtoM_final;
    double cost_init, cost_final;
    int num_iterations = 0;
    const double converge_tolerance = 1e-10;
    const int max_num_iterations = 20;
    bool ret = OptimizePnP(keypoints,
                            cam_indices,
                            T_C0ToCi_,
                            p_LinMs,
                            inliers,
                            cameras_,
                            converge_tolerance,
                            max_num_iterations,
                            T_OtoM_pnp,
                            &T_OtoM_final,
                            &cost_init,
                            &cost_final,
                            &num_iterations);

    if (ret) {
        std::vector<aslam::Transformation> T_MToCi(T_C0ToCi_.size());
        for (size_t cam_idx = 0u; cam_idx < T_C0ToCi_.size(); ++cam_idx) {
            T_MToCi[cam_idx] = cameras_->get_T_BtoC(cam_idx) * T_OtoM_final.inverse();
        }

        CheckInliers(keypoints,
                    cam_indices,
                    p_LinMs,
                    cameras_,
                    T_MToCi,
                    settings_->loop_closure_sigma_pixel,
                    &inliers,
                    &inlier_distances_to_model);

        if (static_cast<int>(inliers.size()) < settings_->min_inlier_count) {
            VLOG(2) << "Not enough inliers after PnP: "
                    << static_cast<int>(inliers.size()) << " vs "
                    << settings_->min_inlier_count;
            return false;
        }

        T_OtoM_pnp = T_OtoM_final;

        localization_result.pnp_inliers.resize(inliers.size());
        for (size_t i = 0u; i < inliers.size(); ++i) {
            localization_result.pnp_inliers[i] = std::make_pair(inliers[i], true);
        }
        CHECK_EQ(localization_result.pnp_inliers.size(),
                   inlier_distances_to_model.size());
        VLOG(2) << "StructureMatch points and inliers after PnP: " << p_LinMs.cols()
                << ", " << localization_result.pnp_inliers.size();
        return true;
    } else {
        return false;
    }
}

bool LoopDetector::HandleLoopClosure(
    const common::VisualFrameDataPtrVec& frame_datas,
    const loop_closure::LandmarkIdListVec& query_vertex_landmark_ids,
    const VertexKeyPointToStructureMatchList& structure_matches,
    int* num_inliers_ptr,
    loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr,
    common::LoopResult* localization_result_ptr,
    std::mutex* map_mutex_ptr) {
    int& num_inliers = *CHECK_NOTNULL(num_inliers_ptr);
    common::LoopResult& localization_result =
            *CHECK_NOTNULL(localization_result_ptr);
    std::mutex& map_mutex = *CHECK_NOTNULL(map_mutex_ptr);

    CHECK_EQ(static_cast<unsigned int>(frame_datas.size()),
               query_vertex_landmark_ids.size());

    num_inliers = 0;

    const int total_matches = static_cast<int>(structure_matches.size());
    if (total_matches < settings_->min_inlier_count) {
        return false;
    }

    Eigen::Matrix2Xd& keypoints = localization_result.keypoints;
    Eigen::Matrix3Xd& p_LinMs = localization_result.positions;
    Eigen::VectorXd& depths = localization_result.depths;
    Eigen::VectorXi& cam_indices = localization_result.cam_indices;

    keypoints.resize(Eigen::NoChange, total_matches);
    p_LinMs.resize(Eigen::NoChange, total_matches);
    depths.resize(total_matches);
    cam_indices.resize(total_matches);

    int idx = 0;
    std::set<loop_closure::LandmarkId> picked_landmark_indices;
    std::set<std::pair<int, int>> picked_keypoint_indices;
    for (const loop_closure::VertexKeyPointToStructureMatch& structure_match : structure_matches) {
        std::unique_lock<std::mutex> lock(map_mutex);

        const int frame_index_query = structure_match.frame_index_query;
        const int keypoint_index_query = structure_match.keypoint_index_query;

        const loop_closure::LandmarkId db_landmark_id = structure_match.landmark_id_result;
        // The loop-closure backend should not return invalid landmarks.
        CHECK(db_landmark_id != -1) << "Found invalid landmark in result set.";

        if (picked_landmark_indices.count(db_landmark_id) == 0 &&
            picked_keypoint_indices.count(std::make_pair(frame_index_query, keypoint_index_query)) == 0) {
            CHECK_LE(static_cast<size_t>(frame_index_query), frame_datas.size());

            keypoints.col(idx) = frame_datas[frame_index_query]->
                    key_points.col(keypoint_index_query).head<2>();

            p_LinMs.col(idx) = summary_map_->GetLandmarkPosition(db_landmark_id);

            depths(idx) = frame_datas[frame_index_query]->
                    depths(keypoint_index_query);

            // Set the frame correspondence to the correct frame for multi-camera
            // systems. We do this for the single-camera case as well.
            cam_indices(idx) = static_cast<size_t>(frame_index_query);
            ++idx;

            picked_landmark_indices.insert(db_landmark_id);
            picked_keypoint_indices.insert(std::make_pair(frame_index_query, keypoint_index_query));
        }
    }

    // Bail for cases where there is no hope to reach the min num inliers.
    const int valid_matches = idx;
    if (valid_matches < settings_->min_inlier_count) {
        VLOG(2) << "Bailing out because too few inliers. (#valid matches: "
                << valid_matches << " vs. min_inlier_count: "
                << settings_->min_inlier_count << ")";
        return false;
    }

    keypoints.conservativeResize(Eigen::NoChange, valid_matches);
    p_LinMs.conservativeResize(Eigen::NoChange, valid_matches);
    depths.conservativeResize(valid_matches);
    cam_indices.conservativeResize(valid_matches);

    std::vector<int> inliers;
    std::vector<double> inlier_distances_to_model;

    if (valid_matches >= settings_->min_inlier_count) {
        if (!PnpCheck(keypoints, p_LinMs, cam_indices,
                      &inliers, &inlier_distances_to_model,
                      &localization_result)) {
            return false;
        }
    }

    if (inlier_structure_matches_ptr != nullptr) {
        loop_closure::VertexKeyPointToStructureMatchList& inlier_structure_matches =
            *CHECK_NOTNULL(inlier_structure_matches_ptr);
        inlier_structure_matches.clear();

        KeypointToInlierIndexWithReprojectionErrorMap
            keypoint_to_best_structure_match;
        GetBestStructureMatchForEveryKeypoint(
            inliers, inlier_distances_to_model, structure_matches,
            frame_datas, &keypoint_to_best_structure_match);

        Eigen::Matrix3Xd landmark_positions;
        landmark_positions.resize(Eigen::NoChange,
            keypoint_to_best_structure_match.size());
        inlier_structure_matches.resize(keypoint_to_best_structure_match.size());

        int inlier_sequential_idx = 0;
        for (const KeypointToInlierIndexWithReprojectionErrorMap::value_type&
                 keypoint_identifier_with_inlier_index :
             keypoint_to_best_structure_match) {
          const int inlier_index =
              keypoint_identifier_with_inlier_index.second.GetInlierIndex();
          CHECK_GE(inlier_index, 0);
          CHECK_LT(inlier_index, p_LinMs.cols());
          landmark_positions.col(inlier_sequential_idx) = p_LinMs.col(inlier_index);
          inlier_structure_matches[inlier_sequential_idx] =
              structure_matches[inlier_index];
          ++inlier_sequential_idx;
        }
    }

    return true;
}

bool LoopDetector::ConvertFrameMatchesToConstraint(
    const loop_closure::KeyFrameToFrameKeyPointMatchesPair& query_frame_id_and_matches,
    loop_closure::LoopClosureConstraint* constraint_ptr) const {
    loop_closure::LoopClosureConstraint& constraint = *CHECK_NOTNULL(constraint_ptr);

    const loop_closure::FrameKeyPointToStructureMatchList& matches =
          query_frame_id_and_matches.second;
    if (matches.empty()) {
        return false;
    }

    // Translate frame_ids to vertex id and frame index.
    constraint.structure_matches.clear();
    constraint.structure_matches.reserve(matches.size());
    constraint.query_vertex_id = query_frame_id_and_matches.first.vertex_id;
    for (const loop_closure::FrameKeyPointToStructureMatch& match : matches) {
        CHECK(match.IsValid());
        CHECK_NOTNULL(summary_map_);
        CHECK(summary_map_->VertexIsValid(match.keyframe_id_result.vertex_id));
        //CHECK(summary_map_->LandmarkIsValid(match.landmark_id_result)) << match.landmark_id_result;
        // Feature may not triangulation successfull.
        if (summary_map_->LandmarkIsValid(match.landmark_id_result)) {
            loop_closure::VertexKeyPointToStructureMatch structure_match;
            structure_match.landmark_id_result = match.landmark_id_result;
            structure_match.keypoint_index_query =
                match.keypoint_id_query.keypoint_index;
            structure_match.keyframe_id_result = match.keyframe_id_result;
            structure_match.frame_index_query =
                match.keypoint_id_query.keyframe_id.frame_index;
            constraint.structure_matches.push_back(structure_match);
        }
    }
    return true;
}


int LoopDetector::GetKeyframeIdWithMostOverlappingLandmarks(
        const int keyframe_id_query,
        const loop_closure::VertexKeyPointToStructureMatchList& structure_matches,
        loop_closure::LandmarkIdList* overlap_landmarks_ptr,
        std::vector<int>* overlap_keypoints_ptr) {
    loop_closure::LandmarkIdList& overlap_landmarks =
            *CHECK_NOTNULL(overlap_landmarks_ptr);
    std::vector<int>& overlap_keypoints = *CHECK_NOTNULL(overlap_keypoints_ptr);
    typedef std::unordered_map<loop_closure::VertexId, loop_closure::LandmarkIdList>
        VertexOverlapLandmarksMap;
    typedef std::unordered_map<loop_closure::VertexId, std::vector<int>>
        VertexOverlapKeypointsMap;
    VertexOverlapLandmarksMap vertex_overlap_landmark_map;
    VertexOverlapKeypointsMap vertex_overlap_keypoint_map;
    std::set<std::pair<int, int>> picked_landmark_set;
    std::set<std::pair<int, int>> picked_keypoint_set;
    for (const loop_closure::VertexKeyPointToStructureMatch& structure_match :
         structure_matches) {
        common::ObservationDeq obs_this_landmark =
                summary_map_->GetObservationsByTrackId(structure_match.landmark_id_result);
        for (const common::Observation& obs_tmp : obs_this_landmark) {
            if (obs_tmp.keyframe_id != keyframe_id_query &&
                picked_landmark_set.count(std::make_pair(obs_tmp.keyframe_id,
                                                         structure_match.landmark_id_result)) == 0 &&
                picked_keypoint_set.count(std::make_pair(obs_tmp.keyframe_id,
                                                         structure_match.keypoint_index_query)) == 0) {
                vertex_overlap_landmark_map[obs_tmp.keyframe_id].push_back(
                            structure_match.landmark_id_result);
                vertex_overlap_keypoint_map[obs_tmp.keyframe_id].push_back(
                            structure_match.keypoint_index_query);
                picked_landmark_set.insert(std::make_pair(obs_tmp.keyframe_id,
                                                          structure_match.landmark_id_result));
                picked_keypoint_set.insert(std::make_pair(obs_tmp.keyframe_id,
                                                          structure_match.keypoint_index_query));
            }
        }
    }

    size_t max_overlap_landmarks = 0u;
    loop_closure::VertexId largest_overlap_vertex_id = -1;
    for (const VertexOverlapLandmarksMap::value_type& item : vertex_overlap_landmark_map) {
      if (item.second.size() > max_overlap_landmarks) {
        largest_overlap_vertex_id = item.first;
        max_overlap_landmarks = item.second.size();
      }
    }

    CHECK_NE(largest_overlap_vertex_id, -1);

    overlap_landmarks = vertex_overlap_landmark_map.at(largest_overlap_vertex_id);
    overlap_keypoints = vertex_overlap_keypoint_map.at(largest_overlap_vertex_id);

    CHECK_EQ(overlap_landmarks.size(), overlap_keypoints.size());

    return largest_overlap_vertex_id;
}

bool LoopDetector::DoProjectedImagesBelongToSameVertex(
        const common::VisualFrameDataPtrVec& frame_datas) {
    CHECK(!frame_datas.empty());
    const int query_vertex_id = frame_datas[0]->frame_id_and_idx.vertex_id;
    for (const common::VisualFrameDataPtr& frame_data : frame_datas) {
        if (query_vertex_id != frame_data->frame_id_and_idx.vertex_id) {
          return false;
        }
    }
    return true;
}

bool LoopDetector::FindFrameToFrameLoop(
        const common::VisualFrameDataPtrVec& frame_data,
        const loop_closure::VertexKeyPointToStructureMatchList& structure_matches,
        common::LoopResult* result_ptr) {
    common::LoopResult& result = *CHECK_NOTNULL(result_ptr);
    // NOTE(chien): only support mono case in now.
    const int cam_idx = 0;

    const int vertex_id_query = frame_data[0]->frame_id_and_idx.vertex_id;
    loop_closure::LandmarkIdList overlap_landmark_ids;
    std::vector<int> overlap_keypoint_indices;
    const int vertex_id_result = GetKeyframeIdWithMostOverlappingLandmarks(
                vertex_id_query, structure_matches,
                &overlap_landmark_ids, &overlap_keypoint_indices);
    if (overlap_keypoint_indices.size() < static_cast<size_t>(settings_->min_inlier_count)) {
        return false;
    }
    Eigen::Matrix2Xd keypoints;
    keypoints.resize(Eigen::NoChange, overlap_keypoint_indices.size());
    Eigen::Matrix3Xd p_LinMs;
    p_LinMs.resize(Eigen::NoChange, overlap_landmark_ids.size());
    Eigen::VectorXi cam_indices;
    cam_indices.resize(overlap_keypoint_indices.size());
    for (size_t i = 0u; i < overlap_keypoint_indices.size(); ++i) {
        const int keypoint_idx = overlap_keypoint_indices[i];
        CHECK_LT(keypoint_idx, frame_data[0]->key_points.cols());
        keypoints.col(i) = frame_data[0]->key_points.col(keypoint_idx).head<2>();
        cam_indices(i) = cam_idx;
    }
    for (size_t i = 0u; i < overlap_landmark_ids.size(); ++i) {
        const loop_closure::LandmarkId track_id = overlap_landmark_ids[i];
        const Eigen::Vector3d& p_LinM = summary_map_->GetLandmarkPosition(track_id);
        p_LinMs.col(i) = p_LinM;
    }
    aslam::Transformation T_OtoM_pnp_frame2frame;
    std::vector<int> inliers;
    std::vector<double> inlier_distances_to_model;
    int num_iters = 0;
    RansacP3P(keypoints,
               cam_indices,
               p_LinMs,
               settings_->loop_closure_sigma_pixel,
               settings_->pnp_num_ransac_iters,
               cameras_,
               &T_OtoM_pnp_frame2frame,
               &inliers,
               &inlier_distances_to_model,
               &num_iters);

    std::vector<aslam::Transformation> T_MtoCi;
    T_MtoCi.push_back(cameras_->get_T_BtoC(cam_idx) * T_OtoM_pnp_frame2frame.inverse());

    CheckInliers(keypoints,
                cam_indices,
                p_LinMs,
                cameras_,
                T_MtoCi,
                settings_->loop_closure_sigma_pixel,
                &inliers,
                &inlier_distances_to_model);

    if (static_cast<int>(inliers.size()) < settings_->min_inlier_count) {
        VLOG(2) << "Not enough inliers after PnP: "
                << static_cast<int>(inliers.size()) << " vs "
                << settings_->min_inlier_count;
        return false;
    } else {
        result.keyframe_id_result = vertex_id_result;
        result.T_estimate = T_OtoM_pnp_frame2frame;
        return true;
    }
}

void LoopDetector::FindGlobally(
        const common::VisualFrameDataPtrVec& frame_datas,
        loop_closure::KeyFrameToFrameKeyPointMatches* frame_matches_list_ptr) {
    loop_closure::KeyFrameToFrameKeyPointMatches& frame_matches_list =
            *CHECK_NOTNULL(frame_matches_list_ptr);
    frame_matches_list.clear();
    if (frame_datas.empty()) {
        return;
    }
    CHECK(DoProjectedImagesBelongToSameVertex(frame_datas));

    aslam::ScopedReadLock lock(&read_write_mutex_);
    const int num_neighbors_to_search = GetNumNeighborsToSearch();
    const size_t num_query_frames = frame_datas.size();

    // Vertex to landmark covisibility filtering only makes sense, if more than
    // one camera is associated with the query vertex.
    const bool use_vertex_covis_filter = num_query_frames > 1u;
    const bool parallelize = num_query_frames > 1u;

    loop_closure::KeyFrameToFrameKeyPointMatches temporary_frame_matches;
    std::mutex covis_frame_matches_mutex;
    std::mutex* covis_frame_matches_mutex_ptr =
        parallelize ? &covis_frame_matches_mutex : nullptr;

    std::function<void(const std::vector<size_t>&)> query_helper = [&](
        const std::vector<size_t>& range) {
        for (const size_t job_index : range) {
            // Which frame.
            const common::VisualFrameDataPtr& frame_data_query =
                frame_datas[job_index];
            const size_t num_descriptors_in_query_frame = static_cast<size_t>(
                frame_data_query->projected_descriptors.cols());
            CHECK_EQ(num_descriptors_in_query_frame,
                       static_cast<size_t>(frame_data_query->key_points.cols()));

            Eigen::MatrixXi indices;
            indices.resize(num_neighbors_to_search,
                         num_descriptors_in_query_frame);
            Eigen::MatrixXf distances;
            distances.resize(num_neighbors_to_search,
                           num_descriptors_in_query_frame);

            // If this loop is running in multiple threads and the inverted
            // multi-index is utilized as NN search structure,
            // ensure that the nearest neighbor back-end (libnabo)
            // is not multi-threaded.
            // Otherwise, performance could decrease drastically.
            index_interface_->GetNNearestNeighborsForFeatures(
                frame_data_query->projected_descriptors,
                num_neighbors_to_search,
                &indices,
                &distances);

            loop_closure::KeyFrameToFrameKeyPointMatches
                keyframe_to_matches_map;
            // cols----feature num.
            // rows----nn num for a certain feature.
            for (int keypoint_idx = 0;
                 keypoint_idx < static_cast<int>(indices.cols());
                 ++keypoint_idx) {
                for (int nn_search_idx = 0;
                     nn_search_idx < static_cast<int>(indices.rows());
                     ++nn_search_idx) {
                    const int nn_match_descriptor_idx =
                        indices(nn_search_idx, keypoint_idx);
                    const float nn_match_distance =
                        distances(nn_search_idx, keypoint_idx);

                    // If we have no matches.
                    if (nn_match_descriptor_idx == -1 ||
                        nn_match_distance ==
                        std::numeric_limits<float>::infinity()) {
                        break;  // No more results for this feature.
                    }

                    loop_closure::FrameKeyPointToStructureMatch structure_match;
                    if (!GetMatchForDescriptorIndex(
                        nn_match_descriptor_idx,
                        keypoint_idx,
                        frame_data_query,
                        &structure_match)) {
                        continue;
                    }

                    keyframe_to_matches_map[
                        structure_match.keyframe_id_result].push_back(
                        structure_match);
                }
            }
            // We don't want to enforce unique matches yet in case of
            // additional vertex-landmark covisibility filtering.
            // The reason for this is that
            // removing non-unique matches can split covisibility clusters.
            DoCovisibilityFiltering(keyframe_to_matches_map,
                                    !use_vertex_covis_filter,
                                    &temporary_frame_matches,
                                    covis_frame_matches_mutex_ptr);

            VLOG(5) << "Total Keyframe to matches size: "
                    << keyframe_to_matches_map.size();
            int sum_matches = 0;
            for (const auto& matches : temporary_frame_matches) {
                sum_matches += matches.second.size();
            }
            VLOG(5) << "Total temporary frame matches size: "
                    << sum_matches;
        }
    };

    if (parallelize) {
        const size_t kNumHardwareThreads = settings_->cached_num_threads;
        const size_t num_threads = std::min<size_t>(
            num_query_frames, kNumHardwareThreads);
        common::ParallelProcess(static_cast<int>(num_query_frames),
                                query_helper,
                                parallelize,
                                num_threads);
    } else {
        std::vector<size_t> proj_img_indices(num_query_frames);
        // Fill in indices: {0, 1, 2, ...}.
        std::iota(proj_img_indices.begin(), proj_img_indices.end(), 0u);
        // Avoid spawning a thread by directly calling the function.
        query_helper(proj_img_indices);
    }

    if (use_vertex_covis_filter) {
        // Convert keyframe matches to vertex matches.
        const size_t num_frame_matches = internal::GetNumberOfMatches(
            temporary_frame_matches);
        VertexToMatchesMap vertex_to_matches_map;
        // Conservative reserve to avoid rehashing.
        vertex_to_matches_map.reserve(num_frame_matches);
        for (const auto& id_frame_matches_pair : temporary_frame_matches) {
            for (const auto& match : id_frame_matches_pair.second) {
                vertex_to_matches_map[match.keyframe_id_result.vertex_id].
                    push_back(match);
            }
        }

        DoCovisibilityFiltering(vertex_to_matches_map,
                                use_vertex_covis_filter,
                                &frame_matches_list);
        for (const auto& frame_matches_pair : frame_matches_list) {
            VLOG(4) << "After DoCovisibilityFiltering, "
                       "find frame_matches for camera: "
                    << frame_matches_pair.first.frame_index;
            VLOG(4) << "frame_matches size: "
                    << frame_matches_pair.second.size();
        }
    } else {
        frame_matches_list.swap(temporary_frame_matches);
    }

    CHECK_LE(frame_matches_list.size(), frame_datas.size())
        << "There cannot be more query frames than projected images.";
}

void LoopDetector::CheckMatchingDistance(
        const Eigen::VectorXf& query_desriptor,
        const common::EigenVectorXfVec& projected_descriptors_candidate,
        int* best_match_idx_ptr,
        double* min_dist_ptr,
        double* second_min_dist_ptr) {
    int& best_match_idx = *CHECK_NOTNULL(best_match_idx_ptr);
    double& min_dist = *CHECK_NOTNULL(min_dist_ptr);
    double& second_min_dist = *CHECK_NOTNULL(second_min_dist_ptr);
    for (size_t i = 0u; i < projected_descriptors_candidate.size(); ++i) {
        const double dist = (query_desriptor - projected_descriptors_candidate[i]).norm();
        if (dist < min_dist) {
            second_min_dist = min_dist;
            min_dist = dist;
            best_match_idx = static_cast<int>(i);
        } else if (dist < second_min_dist) {
            second_min_dist = dist;
        }
    }
}

void LoopDetector::CheckMatchingDistance(
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
        double* second_min_dist_ptr) {
    constexpr double kPixelError = 24;
    int& best_match_idx = *CHECK_NOTNULL(best_match_idx_ptr);
    double& min_dist = *CHECK_NOTNULL(min_dist_ptr);
    double& second_min_dist = *CHECK_NOTNULL(second_min_dist_ptr);

    std::vector<size_t> v_idx_local;

    const int min_cell_x =
        std::max(0, static_cast<int>(
            floor((query_keypoint(0) - kPixelError) * grid_width_inv)));
    CHECK_LT(min_cell_x, grid_cols);

    const int max_cell_x =
        std::min(static_cast<int>(grid_cols - 1), static_cast<int>(
            ceil((query_keypoint(0) + kPixelError) * grid_width_inv)));
    CHECK_GE(max_cell_x, 0);

    const int min_cell_y =
        std::max(0, static_cast<int>(
            floor((query_keypoint(1) - kPixelError) * grid_height_inv)));
    CHECK_LT(min_cell_y, grid_rows);

    const int max_cell_y =
        std::min(static_cast<int>(grid_rows - 1), static_cast<int>(
            ceil((query_keypoint(1) + kPixelError) * grid_height_inv)));
    CHECK_GE(max_cell_y, 0);

    for (int i_x = min_cell_x; i_x <= max_cell_x; ++i_x) {
        for (int i_y = min_cell_y; i_y <= max_cell_y; ++i_y) {
            const std::vector<size_t> vCell = grids[i_x][i_y];

            for (size_t cell : vCell) {
                if ((reprojected_keypoint_candidate[cell] - query_keypoint).norm() <
                        kPixelError) {
                    v_idx_local.push_back(cell);
                }
            }
        }
    }

    for (const size_t current_index : v_idx_local) {
        CHECK_LT(current_index, projected_descriptors_candidate.size());
        const double dist =
                (query_desriptor - projected_descriptors_candidate[current_index]).norm();

        if (dist < min_dist) {
            second_min_dist = min_dist;
            min_dist = dist;
            best_match_idx = static_cast<int>(current_index);
        } else if (dist < second_min_dist) {
            second_min_dist = dist;
        }
    }
}

void LoopDetector::FindMapTracking(
        const aslam::Transformation& T_GtoM,
        const common::VisualFrameDataPtrVec& frame_datas,
        common::LoopResult* result_ptr) {
    common::LoopResult& result = *CHECK_NOTNULL(result_ptr);
    double kSquaredL2P = static_cast<double>(settings_->local_search_radius);
    double kSquaredL2O = static_cast<double>(settings_->local_search_angles);

    Eigen::Matrix2Xd& result_keypoints = result.keypoints;
    result_keypoints.resize(2, 0);
    Eigen::Matrix3Xd& result_positions = result.positions;
    result_positions.resize(3, 0);
    Eigen::VectorXd& result_depths = result.depths;
    result_depths.resize(0);
    Eigen::VectorXi& result_cam_indices = result.cam_indices;
    result_cam_indices.resize(0);

    const aslam::Transformation T_OtoM = T_GtoM * result.T_estimate;

    std::vector<std::pair<size_t, double>> position_indices_dists;
    nano_flann::RadiusResultSet<double, size_t> position_result_set(
        kSquaredL2P, position_indices_dists);
    CHECK_NOTNULL(position_kd_tree_);
    position_kd_tree_->findNeighbors(
        position_result_set, T_OtoM.getPosition().data(), nano_flann::SearchParams());

    const Eigen::Vector3d& euler_OtoM = common::kRadToDeg *
        common::RotToEuler(T_OtoM.getRotationMatrix());
    std::vector<std::pair<size_t, double>> orientation_indices_dists;
    nano_flann::RadiusResultSet<double, size_t> orientation_result_set(
        kSquaredL2O, orientation_indices_dists);
    CHECK_NOTNULL(orientation_kd_tree_);
    orientation_kd_tree_->findNeighbors(
        orientation_result_set, euler_OtoM.data(), nano_flann::SearchParams());

    std::function<std::vector<size_t>(
        const std::vector<std::pair<size_t, double>>&,
        const std::vector<std::pair<size_t, double>>&)> intersection_helper = [&](
        const std::vector<std::pair<size_t, double>>& A,
        const std::vector<std::pair<size_t, double>>& B) {
        std::set<size_t> S;
        for (const auto& a : A) {
            S.insert(a.first);
        }
        std::vector<size_t> res;
        for (const auto& b : B) {
            if (S.erase(b.first)) {
                res.push_back(b.first);
            }
        }
        return res;
    };

    const auto result_set = intersection_helper(position_result_set.m_indices_dists,
                                                orientation_result_set.m_indices_dists);

    constexpr int kGridSize = 10;
    for (size_t cam_idx = 0u; cam_idx < frame_datas.size(); ++cam_idx) {
        const aslam::Transformation T_CtoM = T_GtoM * result.T_estimate *
                cameras_->get_T_BtoC(cam_idx).inverse();
        const aslam::Transformation T_MtoC = T_CtoM.inverse();
        const int width = cameras_->getCamera(cam_idx).imageWidth();
        const int height = cameras_->getCamera(cam_idx).imageHeight();
        const int grid_cols = width / kGridSize;
        const int grid_rows = height / kGridSize;
        const double grid_width_inv = static_cast<double>(grid_cols) / static_cast<double>(width);
        const double grid_height_inv = static_cast<double>(grid_rows) / static_cast<double>(height);
        Grid grids;
        grids.resize(grid_cols);
        for (int col = 0; col < grid_cols; ++col) {
            grids[col].resize(grid_rows);
        }

        common::EigenVector2dVec reprojected_keypoint_candidate;
        std::vector<loop_closure::LandmarkId> track_id_candidate;
        common::EigenVectorXfVec projected_descriptors_candidate;

        std::unordered_set<loop_closure::LandmarkId> track_ids_set;
        int visible_keypoint_counter = 0;
        
        for (size_t result_idx = 0u; result_idx < result_set.size(); ++result_idx) {
            const size_t database_idx = result_set[result_idx];
            CHECK_LT(database_idx, frame_id_list_.size());
            const loop_closure::KeyFrameId result_keyframe_id = frame_id_list_[database_idx];
            const Database::const_iterator iter_database = database_.find(result_keyframe_id);
            CHECK(iter_database != database_.end());
            const common::VisualFrameDataPtr& result_frame_data = iter_database->second;
            if (result_frame_data == nullptr) {
                continue;
            }
            const Eigen::VectorXi& track_ids = result_frame_data->track_ids;
            const common::DescriptorsMatF32& projected_descriptors =
                    result_frame_data->projected_descriptors;
            CHECK_EQ(track_ids.rows(), projected_descriptors.cols());
            if (track_ids.rows() == 0) {
                continue;
            }

            for (int idx = 0; idx < track_ids.rows(); ++idx) {
                const loop_closure::LandmarkId track_id = track_ids(idx);
                if (track_ids_set.count(track_id) > 0) {
                    continue;
                }
                const Eigen::Vector3d& p_LinM = summary_map_->GetLandmarkPosition(track_id);
                const Eigen::Vector3d& p_LinC = T_MtoC.transform(p_LinM);
                Eigen::Vector2d rep_keypoint;
                const aslam::ProjectionResult projection_result = cameras_->getCamera(cam_idx).project3(
                            p_LinC, &rep_keypoint);
                if (projection_result == aslam::ProjectionResult::KEYPOINT_VISIBLE) {
                    const int in_grid_x = std::round(rep_keypoint(0) * grid_width_inv);
                    const int in_grid_y = std::round(rep_keypoint(1) * grid_height_inv);
                    CHECK_GE(in_grid_x, 0);
                    CHECK_GE(in_grid_y, 0);
                    if (in_grid_x < 0 || in_grid_y < 0 ||
                            in_grid_x >= grid_cols || in_grid_y >= grid_rows) {
                        continue;
                    }
                    grids[in_grid_x][in_grid_y].push_back(visible_keypoint_counter++);

                    track_ids_set.insert(track_id);

                    reprojected_keypoint_candidate.push_back(rep_keypoint);
                    track_id_candidate.push_back(track_id);
                    projected_descriptors_candidate.push_back(projected_descriptors.col(idx));
                }
            }
        }
        
        int inlier_counter = 0;
        int unique_match_counter = 0;
        const double kDescRatio = 0.5;
        const double kNNRatio = 0.88;
        std::vector<std::vector<MapTrackingResult>> matching_results;
        matching_results.resize(visible_keypoint_counter);

        const common::VisualFrameDataPtr& frame_data_query =
                frame_datas[cam_idx];
        const Eigen::Matrix6Xd& query_key_points =
                frame_data_query->key_points;
        const Eigen::VectorXd& query_depths =
                frame_data_query->depths;
        const common::DescriptorsMatF32& query_projected_descriptors =
                frame_data_query->projected_descriptors;
        CHECK_EQ(query_depths.rows(), query_key_points.cols());
        CHECK_EQ(query_key_points.cols(), query_projected_descriptors.cols());
        for (int idx = 0; idx < query_key_points.cols(); ++idx) {
            int best_match_idx = -1;
            double min_dist = std::numeric_limits<double>::max();
            double second_min_dist = min_dist;

            CheckMatchingDistance(
                        grids, grid_width_inv, grid_height_inv,
                        grid_cols, grid_rows,
                        query_projected_descriptors.col(idx),
                        query_key_points.col(idx).head<2>(),
                        reprojected_keypoint_candidate,
                        projected_descriptors_candidate,
                        &best_match_idx,
                        &min_dist,
                        &second_min_dist);
            if (min_dist < kDescRatio * loop_closure::kDescriptorDim
                /* && min_dist < kNNRatio * second_min_dist*/) {
                MapTrackingResult this_match;
                this_match.query_keypoint_idx = idx;
                this_match.min_dist = min_dist;
                this_match.second_min_dist = second_min_dist;
                const double reprojection_error = (reprojected_keypoint_candidate[best_match_idx] -
                       query_key_points.col(idx).head<2>()).norm();
                this_match.reprojection_error = reprojection_error;
                matching_results[best_match_idx].push_back(this_match);

                ++inlier_counter;
            }
        }

        for (int idx = 0; idx < visible_keypoint_counter; ++idx) {
            if (matching_results[idx].empty()) {
                continue;
            }

            bool success = true;
            while (matching_results[idx].size() > 1u) {
                auto this_match = matching_results[idx].begin();
                auto next_match = this_match;
                ++next_match;
                CHECK(next_match != matching_results[idx].end());

                if (this_match->reprojection_error < next_match->reprojection_error * kNNRatio) {
                    matching_results[idx].erase(next_match);
                } else if (next_match->reprojection_error < this_match->reprojection_error * kNNRatio) {
                    matching_results[idx].erase(this_match);
                } else {
                    success = false;
                    break;
                }
            }
            if (!success) {
                continue;
            }

            CHECK_EQ(matching_results[idx].size(), 1u);

            // TODO(chien): conservativeResize may less effective, replace it.
            const size_t current_matches_size = result_keypoints.cols();
            result_keypoints.conservativeResize(Eigen::NoChange, current_matches_size + 1u);
            const int query_keypoint_index = matching_results[idx].front().query_keypoint_idx;
            result_keypoints.col(current_matches_size) =
                    query_key_points.col(query_keypoint_index).head<2>();
            result_positions.conservativeResize(Eigen::NoChange, current_matches_size + 1u);
            result_positions.col(current_matches_size) =
                    summary_map_->GetLandmarkPosition(track_id_candidate[idx]);
                        result_depths.conservativeResize(current_matches_size + 1u);
            result_depths(current_matches_size) = query_depths(query_keypoint_index);
            result_cam_indices.conservativeResize(current_matches_size + 1u);
            result_cam_indices(current_matches_size) = cam_idx;
            ++unique_match_counter;
        }

        VLOG(2) << "Final map tracking result (total/inlier/unique): "
                << track_ids_set.size() << "/"
                << inlier_counter << "/"
                << unique_match_counter;
    }
}

int LoopDetector::GetNumNeighborsToSearch() const {
    // Determine the best number of neighbors for a given database size.
    int num_neighbors_to_search = settings_->num_nearest_neighbors;
    if (num_neighbors_to_search == -1) {
        const int num_descriptors_in_db =
                index_interface_->GetNumDescriptorsInIndex();

        if (num_descriptors_in_db < 1e3) {
            num_neighbors_to_search = 1;
        } else if (num_descriptors_in_db < 1e4) {
            num_neighbors_to_search = 2;
        } else if (num_descriptors_in_db < 1e5) {
            num_neighbors_to_search = 4;
        } else if (num_descriptors_in_db < 1e6) {
            num_neighbors_to_search = 6;
        } else {
            num_neighbors_to_search = 8;
        }
    }
    CHECK_GT(num_neighbors_to_search, 0);
    return num_neighbors_to_search;
}

bool LoopDetector::GetMatchForDescriptorIndex(
        int nn_match_descriptor_index, int keypoint_index_query,
        const common::VisualFrameDataPtr& frame_data_query,
        loop_closure::FrameKeyPointToStructureMatch* structure_match_ptr) const {
    loop_closure::FrameKeyPointToStructureMatch& structure_match =
            *CHECK_NOTNULL(structure_match_ptr);
    CHECK_GE(nn_match_descriptor_index, 0);

    const DescriptorIndexToKeypointIdMap::const_iterator
        iter_keypoint_id_result = descriptor_index_to_keypoint_id_.find(
        nn_match_descriptor_index);
    CHECK(iter_keypoint_id_result != descriptor_index_to_keypoint_id_.cend());

    const loop_closure::KeyPointId& keypoint_id_result = iter_keypoint_id_result->second;
    CHECK(keypoint_id_result.IsValid());
    // From the database we can find a image from the summary map.
    const Database::const_iterator& iter_frame_data =
        database_.find(keypoint_id_result.keyframe_id);
    CHECK(iter_frame_data != database_.cend());

    const common::VisualFrameDataPtr& frame_data_base =
        iter_frame_data->second;

    // Skip matches to images which are too close in time.
    if (std::abs(static_cast<int64_t>(frame_data_query->timestamp_ns) -
               static_cast<int64_t>(frame_data_base->timestamp_ns)) <
        common::SecondsToNanoSeconds(settings_->min_image_time_seconds)) {
        return false;
    }

    structure_match.keypoint_id_query.keyframe_id =
        frame_data_query->frame_id_and_idx;
    structure_match.keypoint_id_query.keypoint_index =
        static_cast<size_t>(keypoint_index_query);
    structure_match.keyframe_id_result = keypoint_id_result.keyframe_id;

    if (frame_data_base->track_ids.rows() != 0) {
        CHECK_LT(keypoint_id_result.keypoint_index,
                  frame_data_base->track_ids.rows());
        //NOTE(chien): Use track_id instead of landmark id generation.
        structure_match.landmark_id_result =
            frame_data_base->track_ids(keypoint_id_result.keypoint_index);
        CHECK(structure_match.IsValid());
    }
    return true;
}

template <typename IdType>
void LoopDetector::DoCovisibilityFiltering(
    const loop_closure::IdToFrameKeyPointMatches<IdType>& id_to_matches_map,
    const bool make_matches_unique,
    loop_closure::KeyFrameToFrameKeyPointMatches* frame_matches_ptr,
    std::mutex* frame_matches_mutex_ptr) const {
    // WARNING: Do not clear frame matches. It is intended that new matches can
    // be added to already existing matches. The mutex passed to the function
    // can be nullptr, in which case locking is disabled.
    auto& frame_matches = *CHECK_NOTNULL(frame_matches_ptr);

    constexpr int kInvalidComponentId = -1;
    typedef std::unordered_map<
        loop_closure::FrameKeyPointToStructureMatch, int> MatchesToComponents;

    typedef std::unordered_map<int,
        loop_closure::FrameKeyPointToStructureMatchSet> Components;

    typedef std::unordered_map<int,
        loop_closure::FrameKeyPointToStructureMatchList> LandmarkMatches;

    const size_t num_matches_to_filter =
        internal::GetNumberOfMatches(id_to_matches_map);
    if (num_matches_to_filter == 0u) {
        return;
    }

    MatchesToComponents matches_to_components;
    LandmarkMatches landmark_matches;
    // To avoid rehashing, we reserve at least twice the number of elements.
    matches_to_components.reserve(num_matches_to_filter * 2u);

    // Reserving the number of matches is still conservative because the number
    // of matched landmarks is smaller than the number of matches.
    landmark_matches.reserve(num_matches_to_filter);

    // Every keyframe which have a match to the query frame.
    for (const auto& id_matches_pair : id_to_matches_map) {
        // The keyframe has many matches to the query frame here.
        for (const auto& match : id_matches_pair.second) {
            // LandmarkId in the map has a match.
            landmark_matches[match.landmark_id_result].emplace_back(match);
            // All matches.
            matches_to_components.emplace(match, kInvalidComponentId);
        }
    }

    // The id_to_score_map are keyframes with higher scores.
    IdToScoreMap<IdType> id_to_score_map;
    ComputeRelevantIdsForFiltering(id_to_matches_map, &id_to_score_map);

    int count_component_index = 0;
    int max_component_size = 0;
    int max_component_id = kInvalidComponentId;
    Components components;

    // All the match of 2d_3d,we are trying to get the max_component_siz.
    for (const auto& match_to_component : matches_to_components) {
        // If it's valid, skip it.
        if (match_to_component.second != kInvalidComponentId)
            continue;

        // Component_id Start from 0.
        int component_id = count_component_index++;

        // Find the largest set of keyframes connected by landmark covisibility.
        std::queue<loop_closure::FrameKeyPointToStructureMatch>
            exploration_queue;

        // Push it as the first element.
        exploration_queue.push(match_to_component.first);
        while (!exploration_queue.empty()) {
            // Get the first
            const loop_closure::FrameKeyPointToStructureMatch&
                exploration_match = exploration_queue.front();

            // id_to_score_map is calculated above as the keyframes
            // with higher scores.
            // We should check whether this match
            // is in the precalculated frames.
            if (SkipMatch(id_to_score_map, exploration_match)) {
                // if not in, pop and skip this.
                exploration_queue.pop();
                continue;
            }

            const MatchesToComponents::iterator
                exploration_match_and_component =
                matches_to_components.find(exploration_match);

            CHECK(exploration_match_and_component !=
                matches_to_components.end());

            // If not valid,set its validID as the component_id.
            if (exploration_match_and_component->second ==
                kInvalidComponentId) {
                // Not part of a connected component.
                exploration_match_and_component->second = component_id;

                components[component_id].insert(exploration_match);

                // Mark all observations (which are matches) from this ID
                // (keyframe or vertex) as visited.
                const typename loop_closure::IdToFrameKeyPointMatches<
                    IdType>::const_iterator id_and_matches =
                    GetIteratorForMatch(
                        id_to_matches_map,
                        exploration_match);

                CHECK(id_and_matches != id_to_matches_map.cend());

                // All the match of this keyframe.
                const loop_closure::FrameKeyPointToStructureMatchList&
                    id_matches = id_and_matches->second;
                for (const auto& id_match : id_matches) {
                    // Set as current component_id.
                    matches_to_components[id_match] = component_id;

                    components[component_id].insert(id_match);

                    // Put all observations to this landmark on the stack.
                    const loop_closure::FrameKeyPointToStructureMatchList&
                        lm_matches = landmark_matches[id_match.landmark_id_result];
                    for (const auto& lm_match : lm_matches) {
                        // If have not been visited.
                        if (matches_to_components[lm_match] ==
                            kInvalidComponentId) {
                            exploration_queue.push(lm_match);
                        }
                    }
                }

                if (static_cast<int>(components[component_id].size()) >
                    max_component_size) {
                    max_component_size =
                        static_cast<int>(components[component_id].size());
                    max_component_id = component_id;
                }
            }
            exploration_queue.pop();
        }
    }

    // Only store the structure matches if there is a relevant amount of them.
    // Do we have a larger one.
    if (max_component_size > settings_->min_verify_matches_num) {
        const loop_closure::FrameKeyPointToStructureMatchSet&
            matches_max_component = components[max_component_id];
        typedef std::pair<loop_closure::KeyPointId, loop_closure::LandmarkId>
            KeypointLandmarkPair;
        std::unordered_set<KeypointLandmarkPair> used_matches;
        if (make_matches_unique) {
            // Conservative reserve to avoid rehashing.
            used_matches.reserve(2u * matches_max_component.size());
        }
        auto lock = (frame_matches_mutex_ptr == nullptr)
                    ? std::unique_lock<std::mutex>()
                    : std::unique_lock<std::mutex>(*frame_matches_mutex_ptr);

        // All matches in the matches_max_component.
        for (const auto& structure_match : matches_max_component) {
            if (make_matches_unique) {
                // Clang-format off.
                const bool is_match_unique = used_matches.emplace(
                    structure_match.keypoint_id_query,
                    structure_match.landmark_id_result).second;
                // Clang-format on.
                if (!is_match_unique) {
                    // Skip duplicate (keypoint to landmark) structure matches.
                    continue;
                }
            }

            // Use for RANSAC in the future.
            frame_matches[structure_match.keypoint_id_query.keyframe_id].push_back(
                structure_match);
        }
    }
}

template<>
void LoopDetector::ComputeRelevantIdsForFiltering(
    const loop_closure::KeyFrameToFrameKeyPointMatches& frame_to_matches,
    IdToScoreMap<loop_closure::KeyFrameId>* frame_to_score_map) const {
    CHECK_NOTNULL(frame_to_score_map)->clear();
    // Score each keyframe, then take the part which is in the
    // top fraction and allow only matches to landmarks which are associated with
    // these keyframes.
    scoring::ScoreList<loop_closure::KeyFrameId> score_list;
    CHECK(compute_keyframe_scores_);

    compute_keyframe_scores_(frame_to_matches, keyframe_id_to_num_descriptors_,
                              static_cast<size_t>(NumDescriptors()),
                              &score_list);

    // We want to take matches from the best n score keyframes, but make sure
    // that we evaluate at minimum a given number.
    constexpr size_t kNumMinimumScoreIdsToEvaluate = 4u;
    size_t num_score_ids_to_evaluate =
            std::max<size_t>(static_cast<size_t>(score_list.size() * settings_->fraction_best_scores),
                            kNumMinimumScoreIdsToEvaluate);

    // Ensure staying in bounds.
    num_score_ids_to_evaluate = std::min <size_t>(num_score_ids_to_evaluate, score_list.size());

    std::nth_element(
            score_list.begin(),
            score_list.begin() + num_score_ids_to_evaluate,
            score_list.end(),
            [](const scoring::Score<loop_closure::KeyFrameId>& lhs,
                    const scoring::Score<loop_closure::KeyFrameId>& rhs) -> bool {
                return lhs.second > rhs.second;
            });
    frame_to_score_map->insert(score_list.begin(),
                               score_list.begin() + num_score_ids_to_evaluate);
}

//Not implemented
template<>
void LoopDetector::ComputeRelevantIdsForFiltering(
    const loop_closure::VertexToFrameKeyPointMatches& /* vertex_to_matches */,
    IdToScoreMap<loop_closure::VertexId>* /* vertex_to_score_map */) const {
    // We do not have to score vertices to filter unlikely matches because this
    // is done already at keyframe level.
}

template<>
bool LoopDetector::SkipMatch(
    const IdToScoreMap<loop_closure::KeyFrameId>& frame_to_score_map,
    const loop_closure::FrameKeyPointToStructureMatch& match) const {
    const typename IdToScoreMap<loop_closure::KeyFrameId>::const_iterator iter =
            frame_to_score_map.find(match.keyframe_id_result);
    return iter == frame_to_score_map.cend();
}

//Not implemented
template<>
bool LoopDetector::SkipMatch(
    const IdToScoreMap<loop_closure::VertexId>& /* vertex_to_score_map */,
    const loop_closure::FrameKeyPointToStructureMatch& /* match */) const {
    // We do not skip vertices because we want to consider all keyframes that
    // passed the keyframe covisibility filtering step.
    return false;
}

template<>
typename loop_closure::KeyFrameToFrameKeyPointMatches::const_iterator
    LoopDetector::GetIteratorForMatch(
    const loop_closure::KeyFrameToFrameKeyPointMatches& frame_to_matches,
    const loop_closure::FrameKeyPointToStructureMatch& match) const {
    return frame_to_matches.find(match.keyframe_id_result);
}

template<>
typename loop_closure::VertexToFrameKeyPointMatches::const_iterator
    LoopDetector::GetIteratorForMatch(
    const loop_closure::VertexToFrameKeyPointMatches& vertex_to_matches,
    const loop_closure::FrameKeyPointToStructureMatch& match) const {
    return vertex_to_matches.find(match.keyframe_id_result.vertex_id);
}

}
