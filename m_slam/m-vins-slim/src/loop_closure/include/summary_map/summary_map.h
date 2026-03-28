#ifndef LOOP_CLOSURE_SUMMARY_MAP_H_
#define LOOP_CLOSURE_SUMMARY_MAP_H_

#include <Eigen/Core>
#include <aslam/cameras/ncamera.h>

#include "data_common/state_structures.h"
#include "data_common/visual_structures.h"

namespace loop_closure {
constexpr int kDescriptorDim = 10;

class SummaryMap {
public:
    explicit SummaryMap();
    ~SummaryMap();
    void AddNewFrame(const aslam::NCamera::ConstPtr& cameras,
                       const common::KeyFrames& keyframes,
                       const size_t start_keyframe_idx,
                       const size_t end_keyframe_idx,
                       const common::FeaturePointPtrVec& features);
    Eigen::Matrix<double, 7, 1> GetObserverPose(const int vertex_id) const;
    Eigen::Vector3d GetLandmarkPosition(const int track_id) const;
    common::ObservationDeq GetObservationsByTrackId(const int track_id) const;
    common::ObservationDeq GetObservationsByVertexId(const int vertex_id) const;
    common::ObservationDeq GetObservationsAll() const;
    common::EigenVector3dVec GetLandmarkPositionAll() const;
    void SetObservers(const std::vector<int>& vertex_ids,
                      const Eigen::Matrix<double, 7, Eigen::Dynamic>& T_OtoMs);
    void SetLandmarks(const std::vector<int>& track_ids,
                       const Eigen::Matrix3Xd& p_LinMs);
    void SetObservations(const std::vector<int>& obs_vertex_ids,
                         const std::vector<int>& obs_cam_ids,
                         const std::vector<int>& obs_track_ids,
                         const Eigen::Matrix2Xd& key_points,
                         const common::DescriptorsMatF32& descriptors);
    bool LandmarkIsValid(const int track_id);
    bool VertexIsValid(const int vertex_id);
    void SaveMapAsPly(const std::string& file_path);

    std::unordered_map<int, size_t> GetMapVertexId() const {
        return vertex_id_to_idx_;
    }

    std::unordered_map<int, size_t> GetMapTrackId() const {
        return track_id_to_idx_;
    }

    std::unordered_map<int, std::vector<size_t>> GetMapTrackIdObservation() const {
        return track_id_to_observation_indices_;
    }

    std::unordered_map<int, std::vector<size_t>> GetMapVertexIdObservation() const {
        return vertex_id_to_observation_indices_;
    }
private:
    void UpdateObservationLUT(const int vertex_id,
                              const int track_id,
                              const size_t current_observations_size);
    // Summary map dataset.
    Eigen::Matrix<double, 7, Eigen::Dynamic> T_OtoMs_;
    Eigen::Matrix3Xd p_LinMs_;
    common::ObservationDeq observations_;

    // LUT.
    std::unordered_map<int, size_t> vertex_id_to_idx_;
    std::unordered_map<int, size_t> track_id_to_idx_;
    std::unordered_map<int, std::vector<size_t>> track_id_to_observation_indices_;
    std::unordered_map<int, std::vector<size_t>> vertex_id_to_observation_indices_;
};

}
#endif
