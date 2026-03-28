#ifndef MVINS_COMMON_CONSTANTS_H_
#define MVINS_COMMON_CONSTANTS_H_

#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

namespace common {

constexpr double kMinRange = 0.2;
constexpr double kMaxRangeScan = 45.0;
constexpr double kMaxRangeVisual = 10.0;
constexpr double kMaxWidth = 10.f;

constexpr int kLocalPoseSize = 6;
constexpr int kGlobalPoseSize = 7;

constexpr double kDegToRad = M_PI / 180.;
constexpr double kRadToDeg = 180. / M_PI;

constexpr char kMapFileName[] = "map.pgm";
constexpr char kMapYamlFileName[] = "map.yaml";
constexpr char kVisualMapFileName[] = "map.db";
constexpr char kVisualMapPlyFileName[] = "map.ply";
constexpr char kOctoMapFileName[] = "map.ot";
constexpr char kMaskFileName[] = "mask.db";

inline std::string BoolToString(const bool flag) {
    if (flag) {
        return "TRUE";
    } else {
        return "FALSE";
    }
}

inline int RoundToInt(const float x) { return std::lround(x); }

inline int RoundToInt(const double x) { return std::lround(x); }

inline int64 RoundToInt64(const float x) { return std::lround(x); }

inline int64 RoundToInt64(const double x) { return std::lround(x); }

//! Values to compare equal for variables.
constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

typedef std::shared_ptr<cv::Mat> CvMatPtr;
typedef std::shared_ptr<const cv::Mat> CvMatConstPtr;
typedef std::vector<CvMatPtr> CvMatPtrVec;
typedef std::vector<CvMatConstPtr> CvMatConstPtrVec;

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar>
Positify(const Eigen::QuaternionBase<Derived>& q) {
    if (q.w() < 0.0) {
        Eigen::Quaternion<typename Derived::Scalar> q_after;
        q_after.w() = -q.w();
        q_after.x() = -q.x();
        q_after.y() = -q.y();
        q_after.z() = -q.z();
        return q_after;
    } else {
        return q;
    }
}

// Use flags for removing elements in the vector.
template<class Flag, class Type, class Allocator>
inline void ReduceVector(
    const std::vector<Flag>& flags,
    std::vector<Type, Allocator>* vec_ptr) {
        auto& vec = *vec_ptr;
    assert(flags.size() == vec.size());
    size_t actual_index = 0u;
    for (size_t i = 0u; i < flags.size(); ++i) {
        if (flags[i]) {
            if (actual_index != i) {
                vec[actual_index] = vec[i];
            }
            ++actual_index;
        }
    }
    vec.resize(actual_index);
}

// Use flags for removing elements in the vector.
template<class Flag, class Type, class Allocator>
inline void ReduceDeque(
    const std::vector<Flag>& flags,
    std::deque<Type, Allocator>* vec_ptr) {
        auto& vec = *vec_ptr;
    assert(flags.size() == vec.size());
    size_t actual_index = 0u;
    for (size_t i = 0u; i < flags.size(); ++i) {
        if (flags[i]) {
            if (actual_index != i) {
                vec[actual_index] = vec[i];
            }
            ++actual_index;
        }
    }
    vec.resize(actual_index);
}

// Typedef Eigen##name##Vec for std::vector<Eigen::##name>.
#define REGISTER_EIGEN_VEC_TYPE(name) \
typedef std::vector<Eigen::name, Eigen::aligned_allocator<Eigen::name>> \
    Eigen##name##Vec; \

// Typedef Eigen##type for Eigen::#name, then its vector.
#define REGISTER_EIGEN_VEC_TYPE_AND_RENAME(name, ...) \
typedef Eigen::__VA_ARGS__ Eigen##name; \
typedef std::vector<Eigen::__VA_ARGS__, \
    Eigen::aligned_allocator<Eigen::__VA_ARGS__>> Eigen##name##Vec; \

REGISTER_EIGEN_VEC_TYPE(VectorXf)  // EigenVectorXfVec.
REGISTER_EIGEN_VEC_TYPE(VectorXd)  // EigenVectorXdVec.
REGISTER_EIGEN_VEC_TYPE(Vector2i)  // EigenVector2iVec.
REGISTER_EIGEN_VEC_TYPE(Vector2d)  // EigenVector2dVec.
REGISTER_EIGEN_VEC_TYPE(Vector3f)  // EigenVector3fVec.
REGISTER_EIGEN_VEC_TYPE(Vector3d)  // EigenVector3dVec.
REGISTER_EIGEN_VEC_TYPE(Vector4d)  // EigenVector4dVec.
REGISTER_EIGEN_VEC_TYPE(Matrix3d)  // EigenMatrix3dVec.
REGISTER_EIGEN_VEC_TYPE(Matrix4d)  // EigenMatrix4dVec.

REGISTER_EIGEN_VEC_TYPE_AND_RENAME(Vector6d, Matrix<double, 6, 1>)
REGISTER_EIGEN_VEC_TYPE_AND_RENAME(Vector7d, Matrix<double, 7, 1>)
REGISTER_EIGEN_VEC_TYPE_AND_RENAME(Matrix34d, Matrix<double, 3, 4>)

#undef REGISTER_EIGEN_VEC_TYPE

}
#endif // MVINS_COMMON_CONSTANTS_H_
