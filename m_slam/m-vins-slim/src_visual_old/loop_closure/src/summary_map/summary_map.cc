#include "summary_map/summary_map.h"

#include "math_common/math.h"

namespace loop_closure {

SummaryMap::SummaryMap() {
    T_OtoMs_.resize(Eigen::NoChange, 0);
    p_LinMs_.resize(Eigen::NoChange, 0);
    observations_.resize(0);
}

SummaryMap::~SummaryMap() {
}

void SummaryMap::AddNewFrame(const aslam::NCamera::ConstPtr& cameras,
                             const common::KeyFrames& keyframes,
                             const size_t start_keyframe_idx,
                             const size_t end_keyframe_idx,
                             const common::FeaturePointPtrVec& features) {
    // LUT.
    std::unordered_map<int, size_t> keyframe_id_to_idx;
    for (size_t i = 0u; i < keyframes.size() ;++i) {
        keyframe_id_to_idx[keyframes[i].keyframe_id] = i;
    }

    // Vertex setting.
    for (size_t idx = start_keyframe_idx; idx <= end_keyframe_idx; ++idx) {
        const int current_keyframe_id = keyframes[idx].keyframe_id;
        const aslam::Transformation& T_OtoG = keyframes[idx].state.T_OtoG;
        // Insert vertex.
        const size_t current_vertex_size = T_OtoMs_.cols();
        T_OtoMs_.conservativeResize(Eigen::NoChange, current_vertex_size + 1u);
        // NOTE(chien): Global frame become to map frame.
        T_OtoMs_.col(current_vertex_size) = T_OtoG.asVector();

        vertex_id_to_idx_[current_keyframe_id] = current_vertex_size;
    }

    for (const common::FeaturePointPtr& feature : features) {
        common::ObservationDeq& observations = feature->observations;
        if (observations.empty() || feature->anchor_frame_idx == -1) {
            continue;
        }
        // Landmark setting.
        const int anchor_frame_idx = feature->anchor_frame_idx;
        const int anchor_keyframe_id = observations[anchor_frame_idx].keyframe_id;
        const aslam::Transformation& T_OtoG_anchor =
                keyframes[keyframe_id_to_idx[anchor_keyframe_id]].state.T_OtoG;
        const aslam::Transformation T_CtoG_anchor = T_OtoG_anchor *
                cameras->get_T_BtoC(observations[anchor_frame_idx].camera_idx).inverse();
        const double inv_depth = feature->inv_depth;
        Eigen::Vector3d bearing_3d;
        cameras->getCamera(observations[anchor_frame_idx].camera_idx).backProject3(
            observations[anchor_frame_idx].key_point, &bearing_3d);
        bearing_3d << bearing_3d(0) / bearing_3d(2), bearing_3d(1) / bearing_3d(2), 1.0;
        const Eigen::Vector3d p_LinG = T_CtoG_anchor.transform(bearing_3d / inv_depth);

        if (track_id_to_idx_.find(feature->track_id) != track_id_to_idx_.end()) {
            const int landmark_idx = track_id_to_idx_.at(feature->track_id);
            // Update landmark position.
            // NOTE(chien): Global frame become to map frame.
            p_LinMs_.col(landmark_idx) = p_LinG;
        } else {
            // Insert landmark.
            const size_t current_landmark_size = p_LinMs_.cols();
            p_LinMs_.conservativeResize(Eigen::NoChange, current_landmark_size + 1u);
            // NOTE(chien): Global frame become to map frame.
            p_LinMs_.col(current_landmark_size) = p_LinG;
            track_id_to_idx_[feature->track_id] = current_landmark_size;
        }

        // Observation setting.
        for (int i = static_cast<int>(observations.size()) - 1u; i >= 0; --i) {
            if (observations[i].add_in_map) {
                continue;
            }

            observations[i].add_in_map = true;
            const size_t current_observations_size = observations_.size();
            observations_.push_back(observations[i]);

            const int vertex_id = observations[i].keyframe_id;
            const int track_id = observations[i].track_id;
            CHECK_EQ(feature->track_id, track_id);
            CHECK(track_id_to_idx_.find(track_id) != track_id_to_idx_.end());

            UpdateObservationLUT(vertex_id, track_id, current_observations_size);
        }
    }
}

Eigen::Matrix<double, 7, 1> SummaryMap::GetObserverPose(const int vertex_id) const {
    CHECK(vertex_id_to_idx_.find(vertex_id) != vertex_id_to_idx_.end());
    const int vertex_idx = vertex_id_to_idx_.at(vertex_id);

    return T_OtoMs_.col(vertex_idx);
}

Eigen::Vector3d SummaryMap::GetLandmarkPosition(const int track_id) const {
    CHECK(track_id_to_idx_.find(track_id) != track_id_to_idx_.end());
    const int landmark_idx = track_id_to_idx_.at(track_id);

    return p_LinMs_.col(landmark_idx);
}

common::ObservationDeq SummaryMap::GetObservationsByTrackId(const int track_id) const {
    common::ObservationDeq obs_result;

    const auto itor = track_id_to_observation_indices_.find(track_id);

    if (itor != track_id_to_observation_indices_.end()) {
        const std::vector<size_t>& obs_indices = itor->second;
        for (const size_t obs_idx : obs_indices) {
            obs_result.push_back(observations_[obs_idx]);
        }
    } else {
        LOG(FATAL) << "Track id: " << track_id << " cant find observation in summary map.";
    }

    return obs_result;
}

common::ObservationDeq SummaryMap::GetObservationsByVertexId(const int vertex_id) const {
    common::ObservationDeq obs_result;

    const auto itor = vertex_id_to_observation_indices_.find(vertex_id);

    if (itor != vertex_id_to_observation_indices_.end()) {
        const std::vector<size_t>& obs_indices = itor->second;
        for (const size_t obs_idx : obs_indices) {
            obs_result.push_back(observations_[obs_idx]);
        }
    } else {
        // Do nothing.
        VLOG(5) << "Vertex id: " << vertex_id << " cant find observation in summary map.";
    }

    return obs_result;
}

common::ObservationDeq SummaryMap::GetObservationsAll() const {
    return observations_;
}

common::EigenVector3dVec SummaryMap::GetLandmarkPositionAll() const {
    common::EigenVector3dVec pc(p_LinMs_.cols());
    for (int i = 0; i < p_LinMs_.cols(); ++i) {
        pc[i] = p_LinMs_.col(i);
    }
    return pc;
}

void SummaryMap::SetObservers(const std::vector<int>& vertex_ids,
                              const Eigen::Matrix<double, 7, Eigen::Dynamic>& T_OtoMs) {
   T_OtoMs_ = T_OtoMs;
    vertex_id_to_idx_.clear();
    for (size_t i = 0u; i < vertex_ids.size(); ++i) {
        vertex_id_to_idx_[vertex_ids[i]] = i;
    }
}

void SummaryMap::SetLandmarks(const std::vector<int>& track_ids,
                              const Eigen::Matrix3Xd& p_LinMs) {
    p_LinMs_ = p_LinMs;
    track_id_to_idx_.clear();
    for (size_t i = 0u; i < track_ids.size(); ++i) {
        track_id_to_idx_[track_ids[i]] = i;
    }
}

void SummaryMap::SetObservations(const std::vector<int>& obs_vertex_ids,
                                   const std::vector<int>& obs_cam_ids,
                                   const std::vector<int>& obs_track_ids,
                                   const Eigen::Matrix2Xd& key_points,
                                   const common::DescriptorsMatF32& descriptors) {
    observations_.clear();
    track_id_to_observation_indices_.clear();
    vertex_id_to_observation_indices_.clear();
    for (size_t i = 0u; i < obs_vertex_ids.size(); ++i) {
        common::Observation obs(obs_vertex_ids[i],
                                obs_cam_ids[i],
                                obs_track_ids[i],
                                key_points.col(i),
                                descriptors.col(i));
        const size_t current_observations_size = observations_.size();
        observations_.push_back(obs);

        const int vertex_id = obs_vertex_ids[i];
        CHECK(vertex_id_to_idx_.find(vertex_id) != vertex_id_to_idx_.end()) << vertex_id;
        const int track_id = obs_track_ids[i];
        CHECK(track_id_to_idx_.find(track_id) != track_id_to_idx_.end()) << track_id;

        UpdateObservationLUT(vertex_id, track_id, current_observations_size);
    }
}

bool SummaryMap::LandmarkIsValid(const int track_id) {
    return track_id_to_idx_.find(track_id) != track_id_to_idx_.end();
}

bool SummaryMap::VertexIsValid(const int vertex_id) {
    return vertex_id_to_idx_.find(vertex_id) != vertex_id_to_idx_.end();
}

void SummaryMap::UpdateObservationLUT(const int vertex_id,
                                          const int track_id,
                                          const size_t current_observations_size) {
    if (track_id_to_observation_indices_.find(track_id) == track_id_to_observation_indices_.end()) {
        std::vector<size_t> observation_indices;
        observation_indices.push_back(current_observations_size);
        track_id_to_observation_indices_[track_id] = observation_indices;
    } else {
        std::vector<size_t>& observation_indices = track_id_to_observation_indices_.at(track_id);
        observation_indices.push_back(current_observations_size);
    }

    if (vertex_id_to_observation_indices_.find(vertex_id) == vertex_id_to_observation_indices_.end()) {
        std::vector<size_t> observation_indices;
        observation_indices.push_back(current_observations_size);
        vertex_id_to_observation_indices_[vertex_id] = observation_indices;
    } else {
        std::vector<size_t>& observation_indices = vertex_id_to_observation_indices_.at(vertex_id);
        observation_indices.push_back(current_observations_size);
    }
}

void SummaryMap::SaveMapAsPly(const std::string &file_path) {
    if (p_LinMs_.cols() == 0 || T_OtoMs_.cols() == 0) {
        return;
    }

    std::ofstream outfile;
    outfile.open(file_path);
    if (!outfile.is_open()) {
        LOG(WARNING) << "Ply file can not be open.";
        return;
    }

    outfile << "ply\n";
    outfile << "format ascii 1.0\n";
    constexpr char kAuthor[] = "ATLAS";
    outfile << "comment author: " << kAuthor << "\n";

    const int landmark_size = p_LinMs_.cols();
    const int vertex_size = T_OtoMs_.cols();

    outfile << "element vertex "
            <<  landmark_size + vertex_size
            << "\n";
    outfile << "property float x\n";
    outfile << "property float y\n";
    outfile << "property float z\n";
    outfile << "property uchar red\n";
    outfile << "property uchar green\n";
    outfile << "property uchar blue\n";
    outfile << "end_header\n";

    // Save vertices position.
    for (int i = 0; i < T_OtoMs_.cols(); ++i) {
        outfile << T_OtoMs_(0, i) << " "
               << T_OtoMs_(1, i) << " "
               << T_OtoMs_(2, i) << " "
               << 0 << " "
               << 255 << " "
               << 0 << "\n";
    }

    // Save landmarks position.
    for (int i = 0; i < p_LinMs_.cols(); ++i) {
        outfile << p_LinMs_(0, i) << " "
               << p_LinMs_(1, i) << " "
               << p_LinMs_(2, i) << " "
               << 255 << " "
               << 255 << " "
               << 255 << "\n";
    }
    outfile.close();
}
}
