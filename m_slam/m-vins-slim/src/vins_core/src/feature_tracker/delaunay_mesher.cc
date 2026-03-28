#include "feature_tracker/delaunay_mesher.h"

namespace vins_core {

bool IsAlmostEqual(const double x, const double y, int ulp) {
    // The machine epsilon has to be scaled to the magnitude
    // of the values used and multiplied by the desired precision
    // in ULPs (units in the last place).
    return std::abs(x - y) <=
                std::numeric_limits<double>::epsilon() *
                std::abs(x + y) * static_cast<double>(ulp)
           // Unless the result is subnormal.
           || std::abs(x - y) < std::numeric_limits<double>::min();
}

inline double Half(const double x) {
    return 0.5 * x;
}

bool IsAlmostEqual(const Vertex& v1, const Vertex& v2, int ulp = 2) {
    return IsAlmostEqual(v1.x, v2.x, ulp) && IsAlmostEqual(v1.y, v2.y, ulp);
}

bool IsAlmostEqual(const Edge& e1, const Edge& e2) {
    return (IsAlmostEqual(*e1.v1, *e2.v1) && IsAlmostEqual(*e1.v2, *e2.v2)) ||
           (IsAlmostEqual(*e1.v1, *e2.v2) && IsAlmostEqual(*e1.v2, *e2.v1));
}

bool Triangle::ContainVertex(const Vertex& v) const {
    // Return p1 == v || p2 == v || p3 == v.
    return IsAlmostEqual(*v1, v) ||
           IsAlmostEqual(*v2, v) ||
           IsAlmostEqual(*v3, v);
}

bool Triangle::CircumCircleContains(const Vertex& v) const {
    const double ab = v1->Norm2();
    const double cd = v2->Norm2();
    const double ef = v3->Norm2();

    const double ax = v1->x;
    const double ay = v1->y;
    const double bx = v2->x;
    const double by = v2->y;
    const double cx = v3->x;
    const double cy = v3->y;

    const double circum_x =
            (ab * (cy - by) + cd * (ay - cy) + ef * (by - ay)) /
            (ax * (cy - by) + bx * (ay - cy) + cx * (by - ay));
    const double circum_y =
            (ab * (cx - bx) + cd * (ax - cx) + ef * (bx - ax)) /
            (ay * (cx - bx) + by * (ax - cx) + cy * (bx - ax));

    const Vertex circum(Half(circum_x), Half(circum_y));
    const double circum_radius = v1->Dist2(circum);
    const double Dist = v.Dist2(circum);
    return Dist <= circum_radius;
}

bool IsAlmostEqual(const Triangle& t1, const Triangle& t2) {
    return (IsAlmostEqual(*t1.v1, *t2.v1) || IsAlmostEqual(*t1.v1, *t2.v2) ||
            IsAlmostEqual(*t1.v1, *t2.v3)) &&
           (IsAlmostEqual(*t1.v2, *t2.v1) || IsAlmostEqual(*t1.v2, *t2.v2) ||
            IsAlmostEqual(*t1.v2, *t2.v3)) &&
           (IsAlmostEqual(*t1.v3, *t2.v1) || IsAlmostEqual(*t1.v3, *t2.v2) ||
            IsAlmostEqual(*t1.v3, *t2.v3));
}

const std::vector<Triangle>& DelaunayMeshGenerator::GenerateMesh(
        const std::vector<Vertex>& _vertices) {
    vertices_.clear();
    edges_.clear();
    triangles_.clear();

    // Store the vertices_ locally.
    vertices_ = _vertices;

    // Determinate the super triangle.
    double min_x = vertices_[0].x;
    double min_y = vertices_[0].y;
    double max_x = min_x;
    double max_y = min_y;

    for (size_t i = 0u; i < vertices_.size(); ++i) {
        if (vertices_[i].x < min_x) {
            min_x = vertices_[i].x;
        }

        if (vertices_[i].y < min_y) {
            min_y = vertices_[i].y;
        }

        if (vertices_[i].x > max_x) {
            max_x = vertices_[i].x;
        }

        if (vertices_[i].y > max_y) {
            max_y = vertices_[i].y;
        }
    }

    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double delta_max = std::max(dx, dy);
    const double mid_x = Half(min_x + max_x);
    const double mid_y = Half(min_y + max_y);

    const Vertex p1(mid_x - 20 * delta_max, mid_y - delta_max);
    const Vertex p2(mid_x, mid_y + 20 * delta_max);
    const Vertex p3(mid_x + 20 * delta_max, mid_y - delta_max);

    // Create a list of triangles_, and add the super-triangle in it.
    triangles_.emplace_back(p1, p2, p3);

    for (auto p = std::begin(vertices_); p != std::end(vertices_); ++p) {
        std::vector<Edge> polygon;

        for (auto& t : triangles_) {
            if (t.CircumCircleContains(*p)) {
                t.is_bad = true;
                polygon.emplace_back(*t.v1, *t.v2);
                polygon.emplace_back(*t.v2, *t.v3);
                polygon.emplace_back(*t.v3, *t.v1);
            }
        }

        triangles_.erase(
            std::remove_if(std::begin(triangles_), std::end(triangles_),
                [](Triangle &t) {return t.is_bad;}), std::end(triangles_));

        for (auto e1 = std::begin(polygon); e1 != std::end(polygon); ++e1) {
            for (auto e2 = e1 + 1; e2 != std::end(polygon); ++e2) {
                if (IsAlmostEqual(*e1, *e2)) {
                    e1->is_bad = true;
                    e2->is_bad = true;
                }
            }
        }

        polygon.erase(std::remove_if(begin(polygon), end(polygon),
                [](Edge &e) {return e.is_bad;}), end(polygon));

        for (const auto& e : polygon) {
            triangles_.emplace_back(*e.v1, *e.v2, *p);
        }

    }

    triangles_.erase(std::remove_if(
            std::begin(triangles_), std::end(triangles_),
            [p1, p2, p3](Triangle &t) {return t.ContainVertex(p1) ||
            t.ContainVertex(p2) ||
            t.ContainVertex(p3); }), std::end(triangles_));

    for (const auto& t : triangles_) {
        edges_.emplace_back(*t.v1, *t.v2);
        edges_.emplace_back(*t.v2, *t.v3);
        edges_.emplace_back(*t.v3, *t.v1);
    }

    return triangles_;
}

const std::vector<Triangle>& DelaunayMeshGenerator::GetTriangles() const {
    return triangles_;
}

const std::vector<Edge>& DelaunayMeshGenerator::GetEdges() const {
    return edges_;
}

const std::vector<Vertex>& DelaunayMeshGenerator::GetVertices() const {
    return vertices_;
}

}  // namespace vins_core
