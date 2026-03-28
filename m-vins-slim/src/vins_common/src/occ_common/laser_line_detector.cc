#include "occ_common/laser_line_detector.h"

namespace common {

constexpr double kReprojectionErrorThreshold = 0.06; // meter.
constexpr int kMinPointInlierSize = 10;

bool LaserLineDetector::GetLine(const common::EigenVector3dVec& pc,
                               Line* line_ptr) {
    Line& line = *CHECK_NOTNULL(line_ptr);
    CHECK_EQ(pc.size(), 2u);
    const double x1 = pc[0](0), y1 = pc[0](1);
    const double x2 = pc[1](0), y2 = pc[1](1);
    if (std::abs(x1 - x2) < common::kEpsilon) {
        line = Line(1.0, 0.0, -x1);
    } else if (std::abs(y1 - y2) < common::kEpsilon) {
        line = Line(0.0, 1.0, -y1);
    } else {
        const double A = y2 - y1;
        const double B = x1 - x2;
        const double C = x2 * y1 - x1 * y2;
        line = Line(A, B, C);
    }
    return true;
}

bool LaserLineDetector::FitByLeastSquares(const common::EigenVector3dVec& pc,
                                         Line* line_ptr) {
    Line& line = *CHECK_NOTNULL(line_ptr);
    const size_t pc_size = pc.size();

    double& A = line(0);
    double& B = line(1);
    double& C = line(2);

    if (pc.size() < 2u)
        return false;

    double mx = 0.0, my = 0.0;
    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (size_t i = 0u; i < pc_size; i++) {
        const double x = static_cast<double>(pc[i](0));
        const double y = static_cast<double>(pc[i](1));
        mx += x;
        my += y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    mx /= static_cast<double>(pc_size);
    my /= static_cast<double>(pc_size);

    sxx -= static_cast<double>(pc_size) * mx * mx;
    sxy -= static_cast<double>(pc_size) * mx * my;
    syy -= static_cast<double>(pc_size) * my * my;
    if (std::abs(sxx) < common::kEpsilon) {
        A = 1.0;
        B = 0.0;
    } else {
        const double ev = (sxx + syy + std::sqrt((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy)) / 2.0;
        A = -sxy;
        B = ev - syy;
        const double norm = std::sqrt(A * A + B * B);
        A /= norm;
        B /= norm;
    }
    CHECK(!(std::abs(A) < common::kEpsilon && std::abs(B) < common::kEpsilon));

    C = -(A * mx + B * my);

    return true;
}

void LaserLineDetector::Detect(const common::PointCloud& pc,
                              LinesWithId* lines_ptr,
                              InlierPointIndices* inlier_indices_ptr) {
    LinesWithId& lines = *CHECK_NOTNULL(lines_ptr);
    InlierPointIndices& inlier_indices = *CHECK_NOTNULL(inlier_indices_ptr);

    common::EigenVector3dVec selected_pts;
    selected_pts.resize(2u);
    selected_pts[0] = pc.points[0];
    selected_pts[1] = pc.points.back();

    Line line;
    GetLine(selected_pts, &line);
    
    LinesWithId raw_lines;
    InlierPointIndices raw_inlier_indices;
    LineWithId line_with_id(id_provider_++, line);
    raw_lines.push_back(line_with_id);

    std::deque<LineSegment> line_segments;
    LineSegment line_segment;
    line_segment.start = 0;
    line_segment.end =  pc.points.size() - 1u;
    line_segments.push_back(line_segment);

    size_t lines_num = 1u;
    bool detection_completed = true;
    do {
        detection_completed = true;
        for (size_t i = 0u; i < lines_num; ++i) {
            Line& line = raw_lines.at(i).data;
            LineSegment& line_segment = line_segments.at(i);

            if (line_segment.end == line_segment.start) {
                continue;
            }

            const double A = line(0), B = line(1), C = line(2);
            const double deno = std::sqrt(A * A + B * B);

            double max_dist = 0;
            size_t max_dist_idx = 0;

            for (size_t start = line_segment.start + 1u; start < line_segment.end; ++start) {
                const double dis = std::abs(A * pc.points[start](0) + B * pc.points[start](1) + C) / deno;
                if (dis > max_dist) {
                    max_dist = dis;
                    max_dist_idx = start;
                }
            }
            if (max_dist > kReprojectionErrorThreshold) {
                common::EigenVector3dVec selected_pts;
                Line line_tmp;

                selected_pts.resize(2u);
                selected_pts[0] = pc.points[max_dist_idx];
                selected_pts[1] = pc.points[line_segment.end];
                GetLine(selected_pts, &line_tmp);
                LineWithId line_with_id(id_provider_++, line_tmp);
                raw_lines.push_back(line_with_id);

                LineSegment line_segment_temp;
                line_segment_temp.start = max_dist_idx;
                line_segment_temp.end = line_segment.end;
                line_segments.push_back(line_segment_temp);

                selected_pts[0] = pc.points[line_segment.start];
                selected_pts[1] = pc.points[max_dist_idx];

                GetLine(selected_pts, &line_tmp);
                line = line_tmp;
                line_segment.end = max_dist_idx;

                detection_completed = false;
                i--;

                lines_num = raw_lines.size();
            } else {
                common::EigenVector3dVec inlier_pts;
                std::vector<size_t> inlier_indices_tmp;
                for(size_t i = line_segment.start; i <= line_segment.end; ++i) {
                    inlier_pts.push_back(pc.points[i]);
                    inlier_indices_tmp.push_back(i);
                }
                FitByLeastSquares(inlier_pts, &line);
                raw_inlier_indices.push_back(inlier_indices_tmp);

                line_segment.end = line_segment.start;
            }
        }
    } while (!detection_completed);

    for (size_t i = 0u; i < raw_lines.size(); ++i) {
        double delta_x = std::abs(pc.points[raw_inlier_indices.at(i).back()](0) -
                pc.points[raw_inlier_indices.at(i)[0]](0));
        double delta_y = std::abs(pc.points[raw_inlier_indices.at(i).back()](1) -
                pc.points[raw_inlier_indices.at(i)[0]](1));
        double line_dist = sqrt(delta_x * delta_x + delta_y * delta_y);
        if (raw_inlier_indices.at(i).size() >= kMinPointInlierSize && line_dist >= 0.5) {
            inlier_indices.push_back(raw_inlier_indices.at(i));
            lines.push_back(raw_lines.at(i));
        }
    }
}

}
