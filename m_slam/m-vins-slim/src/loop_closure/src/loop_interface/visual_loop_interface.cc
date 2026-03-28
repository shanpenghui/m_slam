#include "loop_interface/visual_loop_interface.h"

#include "math_common/math.h"
#include "summary_map/map_loader.h"

namespace loop_closure {
VisualLoopInterface::VisualLoopInterface(
    const aslam::NCamera::ConstPtr& cameras,
    const common::SlamConfigPtr& config)
    : cameras_(cameras) {
    settings_.reset(new loop_closure::LoopSettings(config));
    if (config->mapping) {
        summary_map_.reset(new loop_closure::SummaryMap);
    }
    loop_detector_.reset(new loop_closure::LoopDetector(cameras_, settings_, summary_map_));
}

void VisualLoopInterface::LoadSummaryMap(
    const std::shared_ptr<loop_closure::SummaryMap>& summary_map_ptr) {
    summary_map_ = summary_map_ptr;

    loop_detector_->SetSummaryMap(summary_map_ptr);

    LoadFrameDataFromMap();
}

void VisualLoopInterface::LoadFrameDataFromMap() {
    const std::unordered_map<int, size_t>& vertex_id_to_idx =
            summary_map_->GetMapVertexId();
    const std::unordered_map<int, size_t>& track_id_to_idx =
            summary_map_->GetMapTrackId();

    CHECK(!vertex_id_to_idx.empty()) << "Summary map empty,"
                                        " reloc mode cant run in ideal condition";
    common::VisualFrameDataPtrVec frame_datas;
    common::EigenVector3dVec p_OinMs;
    common::EigenVector3dVec euler_OtoMs;
    for (const auto& iter : vertex_id_to_idx) {
        const int vertex_id = iter.first;
        const common::ObservationDeq& observations =
                summary_map_->GetObservationsByVertexId(vertex_id);

        if (observations.empty()) {
            continue;
        }

        common::DescriptorsMatF32 projected_descriptors;
        projected_descriptors.resize(loop_closure::kDescriptorDim, observations.size());
        Eigen::VectorXi track_ids(observations.size());
        int cam_idx = observations.front().camera_idx;
        for (size_t i = 0u; i < observations.size(); ++i) {
            // Check all observations from same camera.
            CHECK_EQ(cam_idx, observations[i].camera_idx);
            projected_descriptors.col(i) = observations[i].projected_descriptors;
            CHECK(track_id_to_idx.find(observations[i].track_id) !=
                    track_id_to_idx.end());
            track_ids(i) = observations[i].track_id;
        }
        const loop_closure::VisualFrameIdentifier frame_id_and_idx(vertex_id, cam_idx);
        const uint64_t timestamp_ns = 0u;

        common::VisualFrameDataPtr frame_data(new common::VisualFrameData);
        frame_data->timestamp_ns = timestamp_ns;
        frame_data->frame_id_and_idx = frame_id_and_idx;
        frame_data->projected_descriptors = projected_descriptors;
        frame_data->track_ids = track_ids;

        frame_datas.push_back(frame_data);

        const Eigen::Matrix<double, 7, 1> T_OtoM = summary_map_->GetObserverPose(vertex_id);
        p_OinMs.push_back(Eigen::Map<const Eigen::Vector3d>(T_OtoM.data()));
        euler_OtoMs.push_back(common::kRadToDeg *
            common::QuatToEuler(Eigen::Map<const Eigen::Quaterniond>(T_OtoM.data() + 3)));
    }

    for (size_t idx = 0u; idx < frame_datas.size(); ++idx) {
        loop_detector_->Insert(frame_datas[idx], p_OinMs[idx], euler_OtoMs[idx]);
    }

    loop_detector_->BuildKdTree();
}

common::EigenVector3dVec VisualLoopInterface::GetMapClouds() {
    return summary_map_->GetLandmarkPositionAll();
}

void VisualLoopInterface::InsertFrameData(
        const common::KeyFrames& keyframes,
        const common::VisualFrameDataPtrVec& frame_datas,
        const common::FeaturePointPtrVec& features) {

    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t cam_idx = 0u; cam_idx < frame_datas.size(); ++cam_idx) {
        if (frame_datas[cam_idx]->key_points.cols() > 0) {
            const aslam::Transformation& last_T_OtoG = keyframes.back().state.T_OtoG;
            const aslam::Transformation& last_T_CtoG = last_T_OtoG *
                    cameras_->get_T_BtoC(cam_idx).inverse();
            const Eigen::Vector3d& last_p_CinG = last_T_CtoG.getPosition();
            const Eigen::Vector3d& last_euler_CtoG = common::kRadToDeg *
                common::RotToEuler(last_T_CtoG.getRotationMatrix());
            loop_detector_->Insert(frame_datas[cam_idx], last_p_CinG, last_euler_CtoG);
        }
    }

    const size_t start_idx = keyframes.size() - 1u;
    const size_t end_idx = keyframes.size() - 1u;
    summary_map_->AddNewFrame(cameras_, keyframes, start_idx, end_idx, features);
}

bool VisualLoopInterface::Query(
        const common::VisualFrameDataPtrVec& frame_datas,
        const aslam::Transformation& T_GtoM,
        common::LoopResult* result_ptr,
        loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr) {
    common::LoopResult& result = *CHECK_NOTNULL(result_ptr);

    std::lock_guard<std::mutex> lock(mutex_);

    bool localize_success = LocalizeNFrame(frame_datas, T_GtoM, &result, inlier_structure_matches_ptr);
    VLOG(1) << "Loop closure inlier size: " << result.pnp_inliers.size();

    if (localize_success && inlier_structure_matches_ptr != nullptr) {
        localize_success = loop_detector_->FindFrameToFrameLoop(frame_datas,
                                                                *inlier_structure_matches_ptr,
                                                                &result);
    }

    return localize_success;
}

bool VisualLoopInterface::LocalizeNFrame(
        const common::VisualFrameDataPtrVec& frame_datas,
        const aslam::Transformation& T_GtoM,
        common::LoopResult* result_ptr,
        loop_closure::VertexKeyPointToStructureMatchList* inlier_structure_matches_ptr) const {
    common::LoopResult& result = *CHECK_NOTNULL(result_ptr);
    const bool success = loop_detector_->FindNFrameInSummaryMapDatabase(
        frame_datas, T_GtoM, &result, inlier_structure_matches_ptr);
    return success;
}

void VisualLoopInterface::ProjectDescriptors(
        const common::DescriptorsMatUint8& raw_des,
        common::DescriptorsMatF32* projected_des) {
    loop_detector_->ProjectDescriptors(raw_des, projected_des);
}
}
