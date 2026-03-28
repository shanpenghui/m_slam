#include "data_common/sensor_structures.h"
#include "data_common/state_structures.h"
#include "data_common/visual_structures.h"

#include "math_common/math.h"

namespace common {

PointCloudMap::PointCloudMap(PointCloudMap&& other) :
        points(std::move(other.points)),
        pc_idx(std::move(other.pc_idx)) {}

void PointCloudMap::operator=(PointCloudMap&& other) {
    points = std::move(other.points);
    pc_idx = std::move(other.pc_idx);
}

void PointCloudMap::RemoveFirstPointCloud() {
    const size_t end_idx = pc_idx.front();
    points.erase(points.begin(), points.begin() + end_idx);
    pc_idx.pop_front();
    for (auto& idx : pc_idx) {
        idx -= end_idx;
    }
}

PointCloud GenerateScanPoints(const SensorDataConstPtr& sensor_data,
                              const Eigen::Vector2d& xy_velocity,
                              const double theta_velocity) {
    PointCloud pc;
    ScanData* scan_data = (ScanData*)(sensor_data.get());
    pc.timestamp_ns = scan_data->timestamp_ns;
    double closest_to_zero = std::numeric_limits<double>::max();
    int closest_to_zero_idx = 0u;
    int range_sizes = scan_data->ranges.size();

    for (int idx = 0u; idx < range_sizes; ++idx) {
        const double raw_angle = scan_data->angle_min + scan_data->angle_increment * static_cast<double>(idx);
        if (fabs(raw_angle) <= closest_to_zero) {
            closest_to_zero_idx = idx;
            closest_to_zero = fabs(raw_angle);
        }
    }

    for (int idx = 0; idx < range_sizes; ++idx) {
        const double raw_range = scan_data->ranges[idx];
        if (std::isnan(raw_range) || std::isinf(raw_range)) {
            continue;
        }
        const double raw_angle = scan_data->angle_min + scan_data->angle_increment * static_cast<double>(idx);

        const bool is_scan_direction_clockwise = true;
        int point_index = 0;
        if (is_scan_direction_clockwise) {
            point_index = closest_to_zero_idx - idx;
        } else {
            point_index = idx - closest_to_zero_idx;
        }
        point_index = point_index < 0 ? range_sizes + point_index : point_index;

        const double angle_offset = scan_data->time_increments * point_index * theta_velocity;
        const double offset_x = scan_data->time_increments * point_index * xy_velocity(0);
        const double offset_y = scan_data->time_increments * point_index * xy_velocity(1);
        const double angle = raw_angle + angle_offset;
        const double x = offset_x + raw_range * std::cos(angle);
        const double y = offset_y + raw_range * std::sin(angle);
        const double z = 0.0;
        const double range = sqrt(x * x + y *y );

        if (range > common::kMaxRangeScan || range < common::kMinRange) {
            pc.miss_points.push_back(PointXYZ(x, y, z));
        } else {
            pc.points.push_back(PointXYZ(x, y, z));
        }
    }

    return pc;
}

Eigen::Matrix6Xd ConvertKeypoints(
        const std::vector<cv::KeyPoint>& key_points,
        const double fixed_keypoint_uncertainty_px) {
    Eigen::Matrix6Xd key_points_eigen;
    key_points_eigen.resize(Eigen::NoChange, key_points.size());
    for (size_t i = 0u; i < key_points.size(); ++i) {
        key_points_eigen(X, i) = key_points[i].pt.x;
        key_points_eigen(Y, i) = key_points[i].pt.y;
        key_points_eigen(SIZE, i) = key_points[i].size;
        key_points_eigen(ANGLE, i) = key_points[i].angle;
        key_points_eigen(RESPONSE, i) = key_points[i].response;
        key_points_eigen(UNCERTAINTLY, i) = fixed_keypoint_uncertainty_px;
    }
    return key_points_eigen;
}

Eigen::Matrix6Xd ConvertKeypoints(
        const Eigen::Matrix<float, 259, Eigen::Dynamic> feature_points,
        const double fixed_keypoint_uncertainty_px) {
    Eigen::Matrix6Xd key_points_eigen;
    key_points_eigen.resize(Eigen::NoChange, feature_points.cols());
    for (int i = 0u; i < feature_points.cols(); ++i) {
        key_points_eigen(X, i) = feature_points(1, i);
        key_points_eigen(Y, i) = feature_points(2, i);
        key_points_eigen(SIZE, i) = 8;
        key_points_eigen(ANGLE, i) = -1;
        key_points_eigen(RESPONSE, i) = feature_points(0, i);
        key_points_eigen(UNCERTAINTLY, i) = fixed_keypoint_uncertainty_px;
    }
    return key_points_eigen;
}

DescriptorsMatUint8 ConvertDescriptorsUInt8(
        const cv::Mat& descriptors) {
    Eigen::Map<DescriptorsMatUint8> descriptors_eigen(
                descriptors.data, descriptors.cols, descriptors.rows);
    return descriptors_eigen;
}

DescriptorsMatF32 ConvertDescriptorsF32(
        const cv::Mat& descriptors) {
    Eigen::Map<DescriptorsMatF32> descriptors_eigen(
        (float*)descriptors.data, descriptors.cols, descriptors.rows);
    return descriptors_eigen;
}

DescriptorsMatF32 ConvertDescriptorsF32(
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& feature_points) {
    if (feature_points.rows() == 256) {
        Eigen::Map<DescriptorsMatF32> descriptors_eigen(
            (float*)feature_points.data(), feature_points.rows(), feature_points.cols());
        return descriptors_eigen;
    } else if (feature_points.rows() == 259) {
        DescriptorsMatF32 descriptors_eigen;
        descriptors_eigen.resize(256, feature_points.cols());
        for (int i = 0; i < feature_points.cols(); ++i) {
            for (int j = 0; j < 256; ++j) {
                descriptors_eigen(j, i) = feature_points(j + 3, i);
            }
        }
        return descriptors_eigen;
    }
    return DescriptorsMatF32();
}

VisualFrameData::VisualFrameData() {
    image_ptr = nullptr;
}

VisualFrameData::VisualFrameData(const VisualFrameData& other, const bool copy_image) {
    timestamp_ns = other.timestamp_ns;
    frame_id_and_idx = other.frame_id_and_idx;
    key_points = other.key_points;
    descriptors = other.descriptors;
    projected_descriptors = other.projected_descriptors;
    track_ids = other.track_ids;
    track_lengths = other.track_lengths;
    depths = other.depths;
    map_track_id_to_idx = other.map_track_id_to_idx;
    if (copy_image) {
        image_ptr = other.image_ptr;
    }
}

size_t VisualFrameData::GetDescriptorSizeBytes() const {
    return descriptors.rows() * sizeof(DescriptorsUint8::Scalar);
}

size_t VisualFrameData::GetProjectedDescriptorSizeBytes() const {
    return projected_descriptors.rows() * sizeof(DescriptorsF32::Scalar);
}

void VisualFrameData::SetFrameIdAndIdx(const int frame_id,
                                       const int frame_idx) {
    frame_id_and_idx = loop_closure::VisualFrameIdentifier(frame_id, frame_idx);
}

void VisualFrameData::SetFeatureIds(
        const std::vector<int>& feature_ids_vec) {
    track_ids.resize(Eigen::NoChange, feature_ids_vec.size());
    for (size_t i = 0u; i < feature_ids_vec.size(); ++i) {
        track_ids(i) = feature_ids_vec[i];
    }
}

void VisualFrameData::SetKeyPoints(
        const std::vector<cv::KeyPoint>& key_points_cv,
        const double fixed_keypoint_uncertainty_px) {
    key_points = ConvertKeypoints(key_points_cv,
                                   fixed_keypoint_uncertainty_px);
}

void VisualFrameData::SetKeyPoints(
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& feature_points,
    const double fixed_keypoint_uncertainty_px) {
    key_points = ConvertKeypoints(feature_points,
                                  fixed_keypoint_uncertainty_px);
}

void VisualFrameData::SetDescriptors(
    const cv::Mat& descriptors_cv) {
    if (descriptors_cv.type() == CV_8UC1) {
        descriptors = ConvertDescriptorsUInt8(descriptors_cv);
    } else if (descriptors_cv.type() == CV_32FC1) {
        projected_descriptors = ConvertDescriptorsF32(descriptors_cv);
    }
}

void VisualFrameData::SetDescriptors(
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& feature_points) {
    projected_descriptors = ConvertDescriptorsF32(feature_points);
}

void VisualFrameData::SetLUT() {
    map_track_id_to_idx.clear();
    for (int idx = 0; idx < track_ids.rows(); ++idx) {
        map_track_id_to_idx[track_ids(idx)] = idx;
    }
}

void VisualFrameData::Update(
    const std::vector<cv::KeyPoint>& _key_points,
    const double fixed_keypoint_uncertainty_px) {
    key_points = ConvertKeypoints(_key_points,
                                  fixed_keypoint_uncertainty_px);
}

void VisualFrameData::GenerateTrackIds(int* track_id_provider_ptr) {
    int& track_id_provider = *CHECK_NOTNULL(track_id_provider_ptr);

    if (key_points.cols() == 0) {
        return;
    }
    track_ids.resize(key_points.cols());

    for (int f_idx = 0; f_idx < key_points.cols(); ++f_idx) {
        track_ids(f_idx) = track_id_provider++;
    }
}

void VisualFrameData::UpdateTrackIds(int* track_id_provider_ptr) {
    int& track_id_provider = *CHECK_NOTNULL(track_id_provider_ptr);
    for (int f_idx = 0; f_idx < track_ids.rows(); ++f_idx) {
        if (track_ids(f_idx) == -1) {
            track_ids(f_idx) = track_id_provider++;
        }
    }
}

void VisualFrameData::InitializeTrackLengths(const size_t size) {
    track_lengths.resize(size);
    track_lengths.setConstant(1);
}

void InsertAdditionalCvKeypointsAndDescriptorsToVisualFrame(
        const std::vector<cv::KeyPoint>& new_cv_keypoints,
        const cv::Mat& new_cv_descriptors,
        const double fixed_keypoint_uncertainty_px,
        VisualFrameData* frame) {
    CHECK_NOTNULL(frame);
    CHECK_GT(fixed_keypoint_uncertainty_px, 0.0);
    CHECK_EQ(frame->key_points.cols(), frame->track_ids.rows());
    CHECK_EQ(frame->key_points.cols(), frame->descriptors.cols());
    CHECK_EQ(new_cv_keypoints.size(),
               static_cast<size_t>(new_cv_descriptors.rows));
    CHECK_EQ(new_cv_descriptors.type(), CV_8UC1);
    CHECK(new_cv_descriptors.isContinuous());

    const size_t kInitialSize = frame->key_points.cols();
    const size_t kAdditionalSize = new_cv_keypoints.size();
    const size_t extended_size = kInitialSize + kAdditionalSize;

    Eigen::Matrix6Xd new_keypoint_measurements;
    new_keypoint_measurements.resize(Eigen::NoChange, kAdditionalSize);
    for (size_t i = 0u; i < kAdditionalSize; ++i) {
      const cv::KeyPoint keypoint = new_cv_keypoints[i];
      new_keypoint_measurements(X, i) = static_cast<double>(keypoint.pt.x);
      new_keypoint_measurements(Y, i) = static_cast<double>(keypoint.pt.y);
      new_keypoint_measurements(SIZE, i) = static_cast<double>(keypoint.size);
      new_keypoint_measurements(ANGLE, i) = static_cast<double>(keypoint.angle);
      new_keypoint_measurements(RESPONSE, i) = static_cast<double>(keypoint.response);
      new_keypoint_measurements(UNCERTAINTLY, i) = fixed_keypoint_uncertainty_px;
    }
    frame->key_points.conservativeResize(Eigen::NoChange, extended_size);
    frame->track_ids.conservativeResize(extended_size);
    frame->key_points.block(0, kInitialSize, 6, kAdditionalSize) =
            new_keypoint_measurements;
    frame->track_ids.segment(kInitialSize, kAdditionalSize).setConstant(-1);
    frame->descriptors.conservativeResize(Eigen::NoChange, extended_size);
    frame->descriptors.block(0, kInitialSize,
                            new_cv_descriptors.cols,
                            new_cv_descriptors.rows) =
            Eigen::Map<common::DescriptorsMatUint8>(
                // Switch cols/rows as Eigen is col-major and cv::Mat is row-major.
                new_cv_descriptors.data, new_cv_descriptors.cols,
                new_cv_descriptors.rows);
}

Observation::Observation() {
    keyframe_id = -1;
    camera_idx = -1;
    track_id = -1;
    key_point = Eigen::Vector2d::Zero();
    velocity = Eigen::Vector2d::Zero();
    depth = kInValidDepth;
    used_counter = 0;
    add_in_map = false;
}

Observation::Observation(
    const int _keyframe_id,
    const int _camera_idx,
    const int _track_id,
    const Eigen::Vector2d& _key_point,
    const DescriptorsF32& _projected_descriptors)
    : keyframe_id(_keyframe_id),
      camera_idx(_camera_idx),
      track_id(_track_id),
      key_point(_key_point),
      projected_descriptors(_projected_descriptors) {
    depth = kInValidDepth;
    used_counter = 0;
    add_in_map = false;
    velocity = Eigen::Vector2d::Zero();
}

Observation::Observation(
        const int _keyframe_id,
        const int _camera_idx,
        const int _track_id,
        const double _depth,
        const Eigen::Vector2d& _key_point,
        const DescriptorsF32& _projected_descriptors)
    : keyframe_id(_keyframe_id),
      camera_idx(_camera_idx),
      track_id(_track_id),
      key_point(_key_point),
      depth(_depth),
      projected_descriptors(_projected_descriptors) {
    used_counter = 0;
    add_in_map = false;
    velocity = Eigen::Vector2d::Zero();
}

void Observation::SetVelocity(const Eigen::Vector2d& _velocity) {
    velocity = _velocity;
}

FeaturePoint::FeaturePoint() : FeaturePoint(-1) {}

FeaturePoint::FeaturePoint(const int _track_id)
    : track_id(_track_id),
      anchor_frame_idx(-1),
      using_in_optimization(false) {}

Eigen::Vector3d FeaturePoint::ReAnchor(const aslam::Transformation& old_T_CtoG,
                                      const aslam::Transformation& new_T_CtoG,
                                      const Eigen::Vector3d& old_p_LinC) {
    const Eigen::Vector3d p_LinG = old_T_CtoG.transform(old_p_LinC);
    const aslam::Transformation new_T_GtoC = new_T_CtoG.inverse();
    const Eigen::Vector3d new_p_LinC = new_T_GtoC.transform(p_LinG);
    return new_p_LinC;
}

Eigen::Matrix<double, 18, 1> State::operator-(const State& other) const {
    Eigen::Matrix<double, 18, 1> output;
    const Eigen::Quaterniond delta_q =
            other.T_OtoG.getEigenQuaternion().conjugate() *
            T_OtoG.getEigenQuaternion();
    const double sign_q = delta_q.w() > 0 ? 1.0 : -1.0;
    output.head<3>() = T_OtoG.getPosition() - other.T_OtoG.getPosition();
    output.segment<3>(3) = sign_q * 2.0 * delta_q.vec();
    output.segment<3>(6) = velocity - other.velocity;
    output.segment<3>(9) = bg - other.bg;
    output.segment<3>(12) = ba - other.ba;
    output.segment<2>(15) = bt - other.bt;
    output(17, 0) = br - other.br;

    CHECK(!output.hasNaN());
    return output;
}

void State::operator=(const State& other) {
    timestamp_ns = other.timestamp_ns;
    T_OtoG = other.T_OtoG;
    velocity = other.velocity;
    bg = other.bg;
    ba = other.ba;
    bt = other.bt;
    br = other.br;
    td_camera = other.td_camera;
    td_scan = other.td_scan;
}

void State::Print(const int log_level, const double t,
                  const std::string& suffix) const {
    const Eigen::Vector3d& p = T_OtoG.getPosition();
    const Eigen::Quaterniond& q = T_OtoG.getEigenQuaternion();
    const Eigen::Vector3d euler = common::QuatToEuler(q);
    VLOG(log_level) << "Pose" << suffix << ": " << t << ", "
                    << p(0) << ", "
                    << p(1) << ", "
                    << p(2) << ", "
                    << common::kRadToDeg * euler(0) << ", "
                    << common::kRadToDeg * euler(1) << ", "
                    << common::kRadToDeg * euler(2);
    VLOG(log_level+1) << "Velocity" << suffix << ": " << t << ", "
                    << velocity(0) << ", "
                    << velocity(1) << ", "
                    << velocity(2);
    VLOG(log_level+1) << "Bg" << suffix << ": " << t << ", "
                    << bg(0) << ", "
                    << bg(1) << ", "
                    << bg(2);
    VLOG(log_level+1) << "Ba" << suffix << ": " << t << ", "
                    << ba(0) << ", "
                    << ba(1) << ", "
                    << ba(2);
    VLOG(log_level+1) << "Bo" << suffix << ": " << t << ", "
                    << bt(0) << ", "
                    << bt(1) << ", "
                    << br;
    VLOG(log_level+1) << "TdCamera" << suffix << ": " << t << ", "
                    << td_camera;
    VLOG(log_level+1) << "TdScan" << suffix << ": " << t << ", "
                    << td_scan;
}

void KeyFrame::operator=(const KeyFrame& other) {
    keyframe_id = other.keyframe_id;
    sensor_meas = other.sensor_meas;
    points = other.points;
    submap_id = other.submap_id;
    state = other.state;
    visual_datas = other.visual_datas;
    score = other.score;
    zero_vm = other.zero_vm;
}

}  // namespace common
