#ifndef DATA_COMMON_VISUAL_STRUCTURE_H_
#define DATA_COMMON_VISUAL_STRUCTURE_H_

#include <Eigen/Dense>
#include <aslam/cameras/camera.h>
#include <aslam/common/hash-id.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <deque>

#include "data_common/constants.h"

enum KEYPOINT {
    X = 0,
    Y = 1,
    SIZE = 2,
    ANGLE = 3,
    RESPONSE = 4,
    UNCERTAINTLY = 5
};

namespace Eigen {
typedef Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::ColMajor> Matrix6Xd;
}

namespace loop_closure {
struct VisualFrameIdentifier {
    inline VisualFrameIdentifier()
        : vertex_id(-1),
          frame_index(-1) {}

    inline VisualFrameIdentifier(
            const int _vertex_id, const int _frame_index)
        : vertex_id(_vertex_id),
          frame_index(_frame_index) {}

    inline bool operator==(const VisualFrameIdentifier& rhs) const {
        return vertex_id == rhs.vertex_id && frame_index == rhs.frame_index;
    }

    inline bool operator!=(const VisualFrameIdentifier& rhs) const {
        return vertex_id != rhs.vertex_id || frame_index != rhs.frame_index;
    }

    inline bool operator<(const VisualFrameIdentifier& rhs) const {
        if (vertex_id == rhs.vertex_id) {
            return frame_index < rhs.frame_index;
        } else {
            return vertex_id < rhs.vertex_id;
        }
    }

    inline bool IsValid() const {
        return vertex_id != -1 && frame_index != -1;
    }

    int vertex_id;
    int frame_index;
};
typedef std::vector<VisualFrameIdentifier> VisualFrameIdentifierList;
typedef std::unordered_map<aslam::FrameId, VisualFrameIdentifier>
    FrameIdToFrameIdentifierMap;

struct KeypointIdentifier {
    inline KeypointIdentifier() : keypoint_index(-1) {}

    inline KeypointIdentifier(
        const VisualFrameIdentifier& _frame_id, const int _keypoint_index)
        : keyframe_id(_frame_id),
          keypoint_index(_keypoint_index) {}

    inline KeypointIdentifier(
            const int _vertex_id, const int _frame_index,
            const int _keypoint_index)
        : keyframe_id(_vertex_id, _frame_index),
          keypoint_index(_keypoint_index) {}

    inline bool operator==(const KeypointIdentifier& other) const {
        return keyframe_id == other.keyframe_id &&
            keypoint_index == other.keypoint_index;
    }

    inline bool IsValid() const {
        return keyframe_id.IsValid() && keypoint_index != -1;
    }

    VisualFrameIdentifier keyframe_id;
    int keypoint_index;
};

struct FrameKeyPointToStructureMatch {
    bool IsValid() const {
        return keypoint_id_query.IsValid() && keyframe_id_result.IsValid() &&
                landmark_id_result != -1;
    }

    bool operator==(const FrameKeyPointToStructureMatch& other) const {
        bool result = true;
        result &= keypoint_id_query == other.keypoint_id_query;
        result &= keyframe_id_result == other.keyframe_id_result;
        result &= landmark_id_result == other.landmark_id_result;
        return result;
    }

    KeypointIdentifier keypoint_id_query;
    VisualFrameIdentifier keyframe_id_result;
    int landmark_id_result = -1;
};

struct VertexKeyPointToStructureMatch {
    VertexKeyPointToStructureMatch()
        : keypoint_index_query(-1),
          frame_index_query(-1),
          landmark_id_result(-1) {}

    VertexKeyPointToStructureMatch(
            const int _keypoint_index_query,
            const int _frame_index_query,
            const int _landmark_result)
        : keypoint_index_query(_keypoint_index_query),
          frame_index_query(_frame_index_query),
          landmark_id_result(_landmark_result) {}

    VertexKeyPointToStructureMatch(
            const int _keypoint_index_query,
            const int _frame_index_query,
            const int _landmark_result,
            const VisualFrameIdentifier& _frame_identifier_result)
        : keypoint_index_query(_keypoint_index_query),
          frame_index_query(_frame_index_query),
          landmark_id_result(_landmark_result),
          keyframe_id_result(_frame_identifier_result) {}

    bool operator==(const VertexKeyPointToStructureMatch& other) const {
        bool result = true;
        result &= keypoint_index_query == other.keypoint_index_query;
        result &= frame_index_query == other.frame_index_query;
        result &= landmark_id_result == other.landmark_id_result;
        result &= keyframe_id_result == other.keyframe_id_result;
        return result;
    }

    // The keypoint index in query image.
    int keypoint_index_query;
    // The frame index of query image.
    int frame_index_query;
    // The landmark id in result image.
    int landmark_id_result;
    // The frame id of result image.
    VisualFrameIdentifier keyframe_id_result;
};

typedef int DescriptorIndex;
typedef int LandmarkId;
typedef int VertexId;

typedef std::vector<VertexKeyPointToStructureMatch>
    VertexKeyPointToStructureMatchList;
typedef std::vector<std::pair<loop_closure::VertexId, VertexKeyPointToStructureMatchList>>
    VertexKeyPointToStructureMatchListVec;

typedef std::vector<LandmarkId> LandmarkIdList;
typedef std::vector<LandmarkIdList> LandmarkIdListVec;

typedef loop_closure::VisualFrameIdentifier KeyFrameId;
typedef loop_closure::KeypointIdentifier KeyPointId;

typedef std::vector<KeyPointId> KeypointIdList;

typedef std::vector<KeyFrameId> FrameIdList;
typedef std::vector<FrameKeyPointToStructureMatch>
    FrameKeyPointToStructureMatchList;
typedef std::unordered_set<FrameKeyPointToStructureMatch>
    FrameKeyPointToStructureMatchSet;
template <typename IdType>
using IdToFrameKeyPointMatches =
    std::unordered_map<IdType, FrameKeyPointToStructureMatchList>;
typedef IdToFrameKeyPointMatches<KeyFrameId> KeyFrameToFrameKeyPointMatches;
typedef IdToFrameKeyPointMatches<VertexId> VertexToFrameKeyPointMatches;

typedef std::unordered_map<KeyFrameId, std::vector<size_t>>
    KeyFrameToKeypointReindexMap;

template <typename IdType>
using IdToNumDescriptors = std::unordered_map<IdType, size_t>;

struct LoopClosureConstraint {
    loop_closure::VertexId query_vertex_id;
    VertexKeyPointToStructureMatchList structure_matches;
};
typedef std::vector<LoopClosureConstraint> LoopClosureConstraintList;

enum LoopSensor {
    kVisual = 0,
    kScan = 1,
    kInvalid = 2
};

enum VisualLoopType {
    kGlobal = 0,
    kMapTracking = 1
};

}

namespace std {
template <>
struct hash<loop_closure::VisualFrameIdentifier> {
    std::size_t operator()(
            const loop_closure::VisualFrameIdentifier& identifier) const {
        return std::hash<size_t>()(static_cast<size_t>(identifier.vertex_id)) ^
                std::hash<size_t>()(static_cast<size_t>(identifier.frame_index));
    }
};

template <>
struct hash<loop_closure::KeypointIdentifier> {
    std::size_t operator()(const loop_closure::KeypointIdentifier& identifier) const {
        return std::hash<loop_closure::VisualFrameIdentifier>()(identifier.keyframe_id) ^
                std::hash<size_t>()(static_cast<size_t>(identifier.keypoint_index));
    }
};

template <>
struct hash<loop_closure::FrameKeyPointToStructureMatch> {
    std::size_t operator()(
        const loop_closure::FrameKeyPointToStructureMatch& value) const {
        const std::size_t h0(
        std::hash<loop_closure::KeypointIdentifier>()(value.keypoint_id_query));
        const std::size_t h1(
        std::hash<loop_closure::VisualFrameIdentifier>()(value.keyframe_id_result));
        const std::size_t h2(
        std::hash<loop_closure::LandmarkId>()(value.landmark_id_result));
        return h0 ^ h1 ^ h2;
    }
};

template <>
struct hash<std::pair<loop_closure::KeypointIdentifier, loop_closure::LandmarkId>> {
    std::size_t operator()(const std::pair<loop_closure::KeypointIdentifier,
        loop_closure::LandmarkId>& value) const {
        const std::size_t h1(
            std::hash<loop_closure::KeypointIdentifier>()(value.first));
        const std::size_t h2(std::hash<loop_closure::LandmarkId>()(value.second));
        return h1 ^ h2;
    }
};
}  // namespace std

namespace loop_closure {
typedef KeyFrameToFrameKeyPointMatches::value_type
    KeyFrameToFrameKeyPointMatchesPair;
}

namespace common {
typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> Edge;
typedef std::vector<Edge> EdgeVec;

typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DescriptorsMatUint8;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DescriptorsMatF32;
typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> DescriptorsUint8;
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> DescriptorsF32;

constexpr double kInValidDepth = -1.0;

class Observation {
 public:
    Observation();

    Observation(const int _keyframe_id,
                const int _camera_idx,
                const int _track_id,
                const Eigen::Vector2d& _key_point,
                const DescriptorsF32& _projected_descriptors);

    Observation(const int _keyframe_id,
                const int _camera_idx,
                const int _track_id,
                const double _depth,
                const Eigen::Vector2d& _key_point,
                const DescriptorsF32& _projected_descriptors);

    void SetVelocity(const Eigen::Vector2d& _velocity);

    // Keyframe id of this observation.
    int keyframe_id;

    // Camera index of the observation frame.
    int camera_idx;

    // Track id of landmark.
    int track_id;

    // Keypoint that this feature has been seen from.
    Eigen::Vector2d key_point;

    // Keypoint velocity (pixel/s).
    // NOTE: not used in summary map.
    Eigen::Vector2d velocity;

    // Depth observation if use depth camera.
    // NOTE: not used in summary map.
    double depth;

    // Projected descriptors.
    DescriptorsF32 projected_descriptors;

    // Each observation will be use twice,
    // once is in optimization, once is in marginalization.
    // NOTE: not used in summary map.
    int used_counter;

    bool add_in_map;
};
typedef std::deque<Observation> ObservationDeq;

// Optimizable instances of points.
class FeaturePoint {
 public:
    FeaturePoint();

    FeaturePoint(const int track_id);

    Eigen::Vector3d ReAnchor(const aslam::Transformation& old_T_CtoG,
                             const aslam::Transformation& new_T_CtoG,
                             const Eigen::Vector3d& old_p_LinG);

    // Track id.
    int track_id;

    // Feaute anchor frame idx
    // NOTE(chien): If anchor frame idx is -1,
    // means that this feature is not triangulation success
    // or never been do triangulation.
    int anchor_frame_idx;

    // Inverse depth in local anchor frame.
    double inv_depth;

    // Collection of its observations, and the first observation is anchor frame.
    common::ObservationDeq observations;

    // Whether the feature is using in sliding window optimization.
    bool using_in_optimization;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
typedef std::shared_ptr<FeaturePoint> FeaturePointPtr;
typedef std::vector<FeaturePointPtr> FeaturePointPtrVec;

inline cv::Mat GetCvMat(const DescriptorsMatUint8& input) {
    const int des_size = input.cols();
    const int des_dim = input.rows();
    cv::Mat output(cv::Size(des_size, des_dim), CV_8UC1);
    output.reserve(des_size);
    for (int i = 0; i < des_size; ++i) {
        for (int j = 0; j < des_dim; ++j) {
            output.at<uchar>(i, j) = input(j, i);
        }
    }
    return output;
}

inline std::vector<cv::Mat> GetCvMatVec(const DescriptorsMatUint8& input) {
    cv::Mat input_cv = GetCvMat(input);
    std::vector<cv::Mat> output_vec(input_cv.rows);
    for (int i = 0; i < input_cv.rows; ++i) {
        output_vec[i] = input_cv.row(i).clone();
    }

    return output_vec;
}

inline std::vector<cv::KeyPoint> GetCvKeyPointVec(
        const Eigen::Matrix6Xd& input) {
    const int keypoint_size = input.cols();
    std::vector<cv::KeyPoint> output;
    output.reserve(keypoint_size);
    for (int i = 0; i < keypoint_size; ++i) {
        cv::KeyPoint temp;
        temp.pt.x = input(X, i);
        temp.pt.y = input(Y, i);
        temp.size = input(SIZE, i);
        temp.angle = input(ANGLE, i);
        temp.response = input(RESPONSE, i);
        output.push_back(temp);
    }
    return output;
}

struct LoopResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    uint64_t timestamp_ns;

    int keyframe_id_query;

    // Only use for pose graph.
    int keyframe_id_result;

    double score;

    // If keyframe_id_result not equal -1,
    // this value is compute from frame to frame.
    // In mapping case, it means transformation from I to G.
    // In reloc case, it means transformation from I to M.
    aslam::Transformation T_estimate;

    // first: inlier index, second: is using for optimization.
    std::vector<std::pair<int, bool>> pnp_inliers;

    // Measurements (include outliers).
    Eigen::Matrix2Xd keypoints;
    Eigen::Matrix3Xd positions;
    Eigen::VectorXd depths;
    Eigen::VectorXi cam_indices;

    loop_closure::LoopSensor loop_sensor;

    loop_closure::VisualLoopType visual_loop_type;

    LoopResult()
      : timestamp_ns(0u),
        keyframe_id_query(-1),
        keyframe_id_result(-1),
        score(0.),
        loop_sensor(loop_closure::LoopSensor::kInvalid),
        visual_loop_type(loop_closure::VisualLoopType::kGlobal) {
        pnp_inliers.resize(0u);
    }

    LoopResult(const uint64_t _timestamp,
               const int _keyframe_id_query,
               const loop_closure::LoopSensor& _loop_sensor,
               const loop_closure::VisualLoopType& _visual_loop_type,
               const aslam::Transformation& _T_estimate)
      : timestamp_ns(_timestamp),
        keyframe_id_query(_keyframe_id_query),
        keyframe_id_result(-1),
        score(0.),
        T_estimate(_T_estimate),
        loop_sensor(_loop_sensor),
        visual_loop_type(_visual_loop_type) {
        pnp_inliers.resize(0u);
    }

    LoopResult(const uint64_t _timestamp,
               const int _keyframe_id_query,
               const int _keyframe_id_result,
               const double _score,
               const loop_closure::LoopSensor& _loop_sensor,
               const loop_closure::VisualLoopType& _visual_loop_type,
               const aslam::Transformation& _T_estimate)
      : timestamp_ns(_timestamp),
        keyframe_id_query(_keyframe_id_query),
        keyframe_id_result(_keyframe_id_result),
        score(_score),
        T_estimate(_T_estimate),
        loop_sensor(_loop_sensor),
        visual_loop_type(_visual_loop_type) {
        pnp_inliers.resize(0u);
    }

    LoopResult(const LoopResult& other) {
        timestamp_ns = other.timestamp_ns;
        keyframe_id_query = other.keyframe_id_query;
        keyframe_id_result = other.keyframe_id_result;
        score = other.score;
        T_estimate = other.T_estimate;
        pnp_inliers = other.pnp_inliers;
        keypoints = other.keypoints;
        positions = other.positions;
        depths = other.depths;
        cam_indices = other.cam_indices;
        loop_sensor = other.loop_sensor;
        visual_loop_type = other.visual_loop_type;
    }

    void operator=(const LoopResult& other) {
        timestamp_ns = other.timestamp_ns;
        keyframe_id_query = other.keyframe_id_query;
        keyframe_id_result = other.keyframe_id_result;
        score = other.score;
        T_estimate = other.T_estimate;
        pnp_inliers = other.pnp_inliers;
        keypoints = other.keypoints;
        positions = other.positions;
        depths = other.depths;
        cam_indices = other.cam_indices;
        loop_sensor = other.loop_sensor;
        visual_loop_type = other.visual_loop_type;
    }
};
typedef std::deque<LoopResult> LoopResults;

struct VisualFrameData {
    VisualFrameData();

    // Copy constructor.
    VisualFrameData(const VisualFrameData& other, const bool copy_image = true);

    void Update(
        const std::vector<cv::KeyPoint>& _key_points,
        const double fixed_keypoint_uncertainty_px);

    size_t GetDescriptorSizeBytes() const;

    size_t GetProjectedDescriptorSizeBytes() const;

    void SetFrameIdAndIdx(const int frame_id,
                          const int frame_idx);

    void SetFeatureIds(const std::vector<int>& feature_ids_vec);

    void SetKeyPoints(const std::vector<cv::KeyPoint>& key_points_cv,
                      const double fixed_keypoint_uncertainty_px);

    void SetKeyPoints(const Eigen::Matrix<float, 259, Eigen::Dynamic>& feature_points,
                      const double fixed_keypoint_uncertainty_px);

    void SetDescriptors(const cv::Mat& descriptors_cv);

    void SetDescriptors(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& feature_points);

    void SetLUT();

    void GenerateTrackIds(int* track_id_provider_ptr);

    void UpdateTrackIds(int* track_id_provider_ptr);

    void InitializeTrackLengths(const size_t size);

    // NOTE(chien): The variable marked as "Used in loop detector"
    // mast be fill data when construct from summary map.
    // timestamp can not get from summary map,
    // can set it as zero.
    uint64 timestamp_ns; // Used in loop detector.

    loop_closure::VisualFrameIdentifier frame_id_and_idx; // Used in loop detector.

    // 0: x, 1: y, 2: size, 3: angle, 4: response, 5: uncertaintly_px.
    Eigen::Matrix6Xd key_points;

    DescriptorsMatUint8 descriptors;

    DescriptorsMatF32 projected_descriptors; // Used in loop detector.

    Eigen::VectorXi track_ids; // Used in loop detector.

    Eigen::VectorXi track_lengths;

    Eigen::VectorXd depths;

    CvMatConstPtr image_ptr;

    std::unordered_map<int, int> map_track_id_to_idx;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::shared_ptr<VisualFrameData> VisualFrameDataPtr;
typedef std::vector<VisualFrameData> VisualFrameDataVec;
typedef std::vector<VisualFrameDataPtr> VisualFrameDataPtrVec;

void InsertAdditionalCvKeypointsAndDescriptorsToVisualFrame(
        const std::vector<cv::KeyPoint>& new_cv_keypoints,
        const cv::Mat& new_cv_descriptors,
        const double fixed_keypoint_uncertainty_px,
        VisualFrameData* frame);

/// \brief A struct to encapsulate a match between two lists and associated
///        matching score. There are two lists, A and B and the matches are
///        indices into these lists.
struct MatchWithScore;
typedef Aligned<std::vector, MatchWithScore> MatchesWithScore;
typedef std::pair<size_t, size_t> Match;
typedef Aligned<std::vector, Match> Matches;
typedef Aligned<std::vector, Matches> MatchesList;
typedef Aligned<std::vector, cv::DMatch> OpenCvMatches;

struct MatchWithScore {
    template <typename MatchWithScore, typename Match>
    friend void ConvertMatchesWithScoreToMatches(
        const Aligned<std::vector, MatchWithScore>& matches_with_score_A_B,
        Aligned<std::vector, Match>* matches_A_B);
    template <typename MatchWithScore>
    friend void ConvertMatchesWithScoreToOpenCvMatches(
        const Aligned<std::vector, MatchWithScore>& matches_with_score_A_B,
        OpenCvMatches* matches_A_B);
    template<typename MatchingProblem> friend class MatchingEngineGreedy;
    /// \brief Initialize to an invalid match.
    MatchWithScore() : correspondence {-1, -1}, score(0.0) {}

    /// \brief Initialize with correspondences and a score.
    MatchWithScore(int index_apple, int index_banana, double _score)
        : correspondence {index_apple, index_banana}, score(_score) {}

    void SetIndexApple(int index_apple) {
      correspondence[0] = index_apple;
    }

    void SetIndexBanana(int index_banana) {
        correspondence[1] = index_banana;
    }

    void SetScore(double _score) {
      score = _score;
    }

    /// \brief Get the score given to the match.
    double GetScore() const {
      return score;
    }

    bool operator<(const MatchWithScore &other) const {
      return this->score < other.score;
    }
    bool operator>(const MatchWithScore &other) const {
      return this->score > other.score;
    }

    bool operator==(const MatchWithScore& other) const {
      return (this->correspondence[0] == other.correspondence[0]) &&
             (this->correspondence[1] == other.correspondence[1]) &&
             (this->score == other.score);
    }

   protected:
    /// \brief Get the index into list A.
    int GetIndexApple() const {
      return correspondence[0];
    }

    /// \brief Get the index into list B.
    int GetIndexBanana() const {
      return correspondence[1];
    }
    int correspondence[2];
    double score;
};

// Macro that generates the match types for a given matching problem.
// getAppleIndexAlias and getBananaIndexAlias specify function aliases for retrieving the apple
// and banana index respectively of a match. These aliases should depend on how the apples and
// bananas are associated within the context of the given matching problem.

#define ASLAM_CREATE_MATCH_TYPES_WITH_ALIASES(                                                    \
    MatchType, GetAppleIndexAlias, GetBananaIndexAlias)                                           \
  struct MatchType ## MatchWithScore : public common::MatchWithScore {                             \
  MatchType ## MatchWithScore(int index_apple, int index_banana, double score)                    \
        : common::MatchWithScore(index_apple, index_banana, score) {}                              \
    virtual ~MatchType ## MatchWithScore() = default;                                             \
    int GetAppleIndexAlias() const {                                                              \
      return common::MatchWithScore::GetIndexApple();                                              \
    }                                                                                             \
    int GetBananaIndexAlias() const {                                                             \
      return common::MatchWithScore::GetIndexBanana();                                             \
    }                                                                                             \
  };                                                                                              \
  typedef Aligned<std::vector, MatchType ## MatchWithScore> MatchType ## MatchesWithScore;  \
  typedef Aligned<std::vector, MatchType ## MatchesWithScore>                               \
    MatchType ## MatchesWithScoreList;                                                            \
  struct MatchType ## Match : public common::Match {                                               \
      MatchType ## Match() = default;                                                             \
      MatchType ## Match(size_t first_, size_t second_) : common::Match(first_, second_){}         \
    virtual ~MatchType ## Match() = default;                                                      \
    size_t GetAppleIndexAlias() const { return first; }                                           \
    size_t GetBananaIndexAlias() const { return second; }                                         \
  };                                                                                              \
  typedef Aligned<std::vector, MatchType ## Match> MatchType ## Matches;                    \
  typedef Aligned<std::vector, MatchType ## Matches> MatchType ## MatchesList;              \

#define ASLAM_ADD_MATCH_TYPEDEFS(MatchType)                                                       \
  typedef MatchType ## MatchWithScore MatchWithScore;                                             \
  typedef Aligned<std::vector, MatchType ## MatchWithScore> MatchesWithScore;               \
  typedef Aligned<std::vector, MatchType ## MatchesWithScore> MatchesWithScoreList;         \
  typedef MatchType ## Match Match;                                                               \
  typedef Aligned<std::vector, MatchType ## Match> Matches;                                 \
  typedef Aligned<std::vector, MatchType ## Matches> MatchesList;                           \

ASLAM_CREATE_MATCH_TYPES_WITH_ALIASES(
    FrameToFrame, GetKeypointIndexAppleFrame, GetKeypointIndexBananaFrame)
ASLAM_CREATE_MATCH_TYPES_WITH_ALIASES(
    LandmarksToFrame, GetKeypointIndex, GetLandmarkIndex)

struct Object2D {
  cv::Rect_<float> bbox;
  int label;
  float prob;
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

}  // namespace common

#endif
