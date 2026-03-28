#ifndef DATA_COMMON_STATE_STRUCTURES_H_
#define DATA_COMMON_STATE_STRUCTURES_H_

#include "data_common/sensor_structures.h"
#include "data_common/visual_structures.h"

namespace common {
struct State {
    uint64_t timestamp_ns;

    // Pose state odometry in global.
    aslam::Transformation T_OtoG;

    // Velocity in global frame.
    Eigen::Vector3d velocity;

    // IMU bias.
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;

    // Odom bias.
    Eigen::Vector2d bt;
    double br;

    // Time drift.
    double td_camera;
    double td_scan;

    State() {
      timestamp_ns = 0u;
      T_OtoG = aslam::Transformation();
      velocity << 0.0, 0.0, 0.0;
      td_camera = 0.0;
      td_scan = 0.0;
      bg << 0.0, 0.0, 0.0;
      ba << 0.0, 0.0, 0.0;
      bt << 0.0, 0.0;
      br = 0.0;
      
    }

    // when perform IMU propagation.
    State(const aslam::Transformation& _T_OtoG,
          const Eigen::Vector3d& _velocity,
          const Eigen::Vector3d& _bg,
          const Eigen::Vector3d& _ba)
      : T_OtoG(_T_OtoG),
        velocity(_velocity),
        bg(_bg),
        ba(_ba) {
      timestamp_ns = 0u;
      bt << 0.0, 0.0;
      br = 0.0;
      td_camera = 0.0;
      td_scan = 0.0;
    }

    // when perform odom propagation.
    State(const aslam::Transformation& _T_OtoG,
          const Eigen::Vector2d& _bt,
          const double _br)
      : T_OtoG(_T_OtoG),
        bt(_bt),
        br(_br) {
      timestamp_ns = 0u;
      velocity << 0.0, 0.0, 0.0;
      bg << 0.0, 0.0, 0.0;
      ba << 0.0, 0.0, 0.0;
      td_camera = 0.0;
      td_scan = 0.0;
    }

    State(const State& other)
      : timestamp_ns(other.timestamp_ns),
        T_OtoG(other.T_OtoG),
        velocity(other.velocity),
        bg(other.bg),
        ba(other.ba),
        bt(other.bt),
        br(other.br),
        td_camera(other.td_camera),
        td_scan(other.td_scan) {}

    Eigen::Matrix<double, 18, 1> operator-(const State& other) const;
    void operator=(const State& other);
    void Print(const int log_level, const double t,
               const std::string& suffix = "") const;
};

enum KeyFrameType {
  Scan = 0,
  Visual = 1,
  ScanAndVisual = 2,
  InValid = 3
};

inline KeyFrameType StringToKeyframeType(const std::string& type_str) {
    if (type_str == "scan") {
        return KeyFrameType::Scan;
    } else if (type_str == "visual") {
        return KeyFrameType::Visual;
    } else if (type_str == "scan_visual" || type_str == "visual_scan") {
        return KeyFrameType::ScanAndVisual;
    } else {
        LOG(ERROR) << "Unknow keyframe type.";
        return KeyFrameType::InValid;
    }
}

enum TuningMode {
  Off = 0,
  Online = 1,
  Offline = 2
};

inline TuningMode StringToTuningMode(const std::string& mode_str) {
  if (mode_str == "off") {
      return TuningMode::Off;
  } else if (mode_str == "online") {
      return TuningMode::Online;
  } else if (mode_str == "offline") {
      return TuningMode::Offline;
  } else {
      LOG(ERROR) << "Unknow tuning mode, set to off";
      return TuningMode::Off;
  }
}

inline bool IsTuningOn(const TuningMode& mode) {
  return mode == TuningMode::Online || mode == TuningMode::Offline;
}

class KeyFrame {
public:
    KeyFrame() {
      keyframe_id = -1;
      score = -1.;
    }

    KeyFrame(const int _keyframe_id,
              const State& _state)
      : keyframe_id(_keyframe_id),
        state(_state) {
      score = -1.;
    }

    KeyFrame(const int _keyframe_id,
              const State& _state,
              const VisualFrameDataPtrVec& _visual_datas)
      : keyframe_id(_keyframe_id),
        state(_state),
        visual_datas(_visual_datas) {
      score = -1.;
    }

    KeyFrame(const int _keyframe_id,
              const common::SyncedHybridSensorData& _sensor_meas,
              const common::PointCloud& point_cloud,
              const State& _state)
      : keyframe_id(_keyframe_id),
        sensor_meas(_sensor_meas),
        points(point_cloud),
        state(_state) {
      visual_datas.resize(0u);
      score = -1.;
    }

    KeyFrame(const int _keyframe_id,
              const common::SyncedHybridSensorData& _sensor_meas,
              const State& _state,
              const VisualFrameDataPtrVec& _visual_datas)
      : keyframe_id(_keyframe_id),
        sensor_meas(_sensor_meas),
        state(_state),
        visual_datas(_visual_datas) {
      score = -1.;
    }

    KeyFrame(const KeyFrame& other, const bool copy_image = true)
      : keyframe_id(other.keyframe_id),
        points(other.points),
        submap_id(other.submap_id),
        state(other.state),
        score(other.score),
        zero_vm(other.zero_vm) {
        if (copy_image) {
            sensor_meas = other.sensor_meas;
            visual_datas = other.visual_datas;
        } else {
            sensor_meas = common::SyncedHybridSensorData(other.sensor_meas, copy_image);
            for (const auto& visua_data : other.visual_datas) { 
                visual_datas.push_back(
                    std::make_shared<common::VisualFrameData>
                    (common::VisualFrameData(*visua_data, copy_image)));
            }
        }
    }

    void SetVisualFrameDatas(const VisualFrameDataPtrVec& _visual_datas) {
      visual_datas = _visual_datas;
    }

    void SetZeroVelocityFlag(const bool _zero_vm) {
      zero_vm = _zero_vm;
    }

    void SetScore(const double _score) {
      score = _score;
    }

    KeyFrameType GetType() const {
      if ((points.points.empty() && points.miss_points.empty()) && !visual_datas.empty()) {
        return KeyFrameType::Visual;
      } else if ((!points.points.empty() || !points.miss_points.empty()) && visual_datas.empty()) {
        return KeyFrameType::Scan;
      } else if ((!points.points.empty() || !points.miss_points.empty()) && !visual_datas.empty()) {
        return KeyFrameType::ScanAndVisual;
      } else {
        return KeyFrameType::InValid;
      }
    }

    void operator=(const KeyFrame& other);

    int keyframe_id;
    common::SyncedHybridSensorData sensor_meas;
    PointCloud points;
    std::set<int, std::less<int>> submap_id;
    State state;
    VisualFrameDataPtrVec visual_datas;
    double score;
    bool zero_vm;
};

typedef std::deque<KeyFrame> KeyFrames;

typedef std::pair<aslam::Transformation, octomap::PointCloudXYZRGB> OctoMappingInput;

inline bool CheckPoseSimilar(const aslam::Transformation& A,
                             const aslam::Transformation& B,
                             const double max_position_error,
                             const double max_rotation_error) {
    const Eigen::Vector3d error_t = A.getPosition() - B.getPosition();
    const Eigen::Quaterniond delta_q = A.getEigenQuaternion().conjugate() *
            B.getEigenQuaternion();
    const double sign_q = delta_q.w() > 0 ? 1. : -1.;
    const Eigen::Vector3d error_q = sign_q * 2.0 * delta_q.vec();

    if (error_t.norm() > max_position_error || error_q.norm() > max_rotation_error) {
        return false;
    } else {
        return true;
    }
}

}  // namespace common

namespace loop_closure {
  typedef std::pair<common::KeyFrame, common::LoopResult> LoopCandidate;
  typedef std::shared_ptr<LoopCandidate> LoopCandidatePtr;
  typedef std::vector<LoopCandidatePtr> LoopCandidatePtrOneFrame;  
}
#endif
