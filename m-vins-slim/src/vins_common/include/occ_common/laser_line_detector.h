#ifndef OCC_COMMON_LINE_DETECTOR_H_
#define OCC_COMMON_LINE_DETECTOR_H_

#include <deque>
#include <vector>

#include "Eigen/Core"

#include "data_common/sensor_structures.h"

namespace common {
// NOTE: Define line function: By = Ax + C
typedef Eigen::Vector3d Line;
struct LineWithId {
    int id;
    Line data;
    LineWithId() {
        id = -1;
        data = Line::Zero();
    }
    LineWithId(const int _id, const Line& _data) {
        id = _id;
        data = _data;
    }
};

struct LineSegment {
        size_t start;
        size_t end;
    };

typedef std::deque<LineWithId> LinesWithId;
typedef std::vector<std::vector<size_t>> InlierPointIndices;

class LaserLineDetector {
public:
    explicit LaserLineDetector() = default;
    ~LaserLineDetector() = default;

    // Use this algorithm, please make sure that point cloud is insert by scan order.
    void Detect(const common::PointCloud& pc,
                LinesWithId* lines_ptr,
                InlierPointIndices* inlier_indices_ptr);
private:
    bool GetLine(const common::EigenVector3dVec& pc,
                 Line* line_ptr);

    bool FitByLeastSquares(const common::EigenVector3dVec& pc,
                           Line* line_ptr);

    int id_provider_ = 0;
};
}

#endif
